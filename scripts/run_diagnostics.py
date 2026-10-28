#!/usr/bin/env python
"""Phase 4 diagnostics: parity, residuals and learning curves.

These artefacts are inputs to the leakage gate. They are produced here rather
than in ``scripts/run_analysis.py`` because Phase 5 is itself blocked until the
gate passes; requiring Phase-5 outputs to finalise a Phase-4 gate would be
circular. Nothing here fits on held-out data: parity/residuals use the stored
out-of-fold predictions, and learning curves train only on one official
training fold.
"""

from __future__ import annotations

import argparse
import json

from src.config import PATHS
from src.data.loader import load_cached, load_fold_indices
from src.evaluation.validation import learning_curves, parity_and_residuals
from src.models.train import load_oof
from src.utils.io import read_json
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Default: best model from Phase 3.")
    parser.add_argument("--skip-learning-curves", action="store_true")
    args = parser.parse_args()

    best = read_json(PATHS["metrics"] / "best_model.json")
    model_name = args.model or best["model"]

    oof = load_oof(model_name)
    logger.info("Parity and residual diagnostics for %s (n=%d OOF rows)", model_name, len(oof))
    stats = parity_and_residuals(oof, model_name)

    curves = None
    if not args.skip_learning_curves:
        frame, _ = load_cached()
        folds = load_fold_indices(frame)
        logger.info("Learning curves for %s (official fold 0)", model_name)
        curves = learning_curves(frame, folds, [model_name], fold_id=0)

    print(
        json.dumps(
            {
                "model": model_name,
                "residual_stats": stats,
                "learning_curve_points": len((curves or {}).get("models", {}).get(model_name, [])),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
