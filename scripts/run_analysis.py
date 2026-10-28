#!/usr/bin/env python
"""Phase 5: interpretability, uncertainty, calibration and applicability domain.

Requires Phase 4 artefacts (OOF predictions and a clear gate). The script
refuses to run if the leakage gate file does not say PASSED.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from src.config import PATHS
from src.data.loader import load_cached, load_fold_indices
from src.evaluation.applicability import evaluate_ad_by_fold
from src.evaluation.uncertainty import conformal_from_oof
from src.evaluation.validation import learning_curves, parity_and_residuals
from src.interpretability.importance import permutation_importance_cv, shap_available
from src.models.train import load_oof
from src.utils.io import read_json, write_json
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _gate_passed() -> bool:
    path = PATHS["docs"] / "LEAKAGE_AUDIT.md"
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    return "LEAKAGE_GATE: PASSED" in text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Default: model selected by benchmark metrics.")
    parser.add_argument("--skip-learning-curves", action="store_true")
    args = parser.parse_args()

    best = read_json(PATHS["metrics"] / "best_model.json")
    model_name = args.model or best["model"]

    if not _gate_passed():
        raise SystemExit(
            "LEAKAGE_GATE is not PASSED. Phase 5 is blocked. "
            "Run scripts/finalize_gate.py after Phase 4 artefacts exist."
        )

    frame, _ = load_cached()
    folds = load_fold_indices(frame)
    oof = load_oof(model_name)

    logger.info("Residual and parity diagnostics for %s", model_name)
    residual_stats = parity_and_residuals(oof, model_name)

    logger.info("Conformal uncertainty for %s", model_name)
    uncertainty = conformal_from_oof(oof, model_name)

    logger.info("Applicability domain (per official fold + deployment reference)")
    ad = evaluate_ad_by_fold(frame, folds)

    importance = permutation_importance_cv(frame, folds, model_name)

    curves = None
    if not args.skip_learning_curves:
        curves = learning_curves(frame, folds, [model_name])

    summary = {
        "model": model_name,
        "shap_available": shap_available(),
        "shap_status": (
            "not run in this build; permutation importance is the primary, stable method"
            if not shap_available()
            else "available but not required for the gate"
        ),
        "residual_stats": residual_stats,
        "conformal_coverage": uncertainty["coverage"],
        "ad_overall": ad["overall"],
        "top_features": (importance or {}).get("top_10", []),
        "learning_curves": (curves or {}).get("models", {}),
    }
    write_json(PATHS["interpretability"] / "phase5_summary.json", summary)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
