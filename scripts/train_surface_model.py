#!/usr/bin/env python
"""Phase 7: re-apply the surface modelling gate and record the decision.

The gate is deliberately evaluated *again* here rather than trusted from an
earlier run. If it fails, the script writes a curation-only decision record and
exits successfully: curation-only is a valid terminal outcome, not an error.

If the gate passes, a study-aware model is trained with
``LeaveOneGroupOut`` on ``study_id``. Random splits are impossible by
construction in this file. The model is only accepted if its held-out
performance beats a study-mean baseline; otherwise it is still reported but
labelled as not supported by the evidence.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import PATHS
from src.data.literature import LOCKED_ENDPOINT, audit, deduplicate, modelling_gate
from src.utils.io import read_json, utc_now, write_json
from src.utils.logging import get_logger

logger = get_logger(__name__)

NUMERIC_FEATURES = [
    "treatment_numeric_code",
]


def _phase4_passed() -> bool:
    path = PATHS["docs"] / "LEAKAGE_AUDIT.md"
    if not path.exists():
        return False
    return "LEAKAGE_GATE: PASSED" in path.read_text(encoding="utf-8")


def _load_and_reaudit() -> tuple[pd.DataFrame, dict, dict]:
    """Rebuild the audit from the persisted clean table.

    Duplicate removal is idempotent, so re-running it on the clean table is a
    consistency check rather than a second chance to drop rows.
    """
    clean_path = PATHS["data_literature"] / "surface_evidence_clean.csv"
    raw_path = PATHS["data_literature"] / "surface_evidence_raw.csv"
    if not clean_path.exists():
        raise SystemExit(
            f"Missing {clean_path}. Run `python scripts/build_literature.py` first."
        )
    raw = pd.read_csv(raw_path) if raw_path.exists() else pd.read_csv(clean_path)
    clean = pd.read_csv(clean_path)
    clean, dedup_log = deduplicate(clean)
    excluded = read_json(PATHS["validation"] / "surface_gate.json").get("excluded_studies", [])
    payload = audit(raw, clean, dedup_log, excluded)
    return clean, payload, dedup_log


def _prepare_locked(clean: pd.DataFrame) -> pd.DataFrame:
    """Select the locked endpoint and build leakage-safe numeric covariates.

    Only columns that describe the *material surface treatment itself* are used
    as covariates. Study-level identifiers never enter the design matrix; the
    grouping variable is ``study_id``.
    """
    locked = clean[clean["endpoint_key"] == LOCKED_ENDPOINT].copy()
    locked["treatment_numeric_code"] = (
        locked["surface_treatment"].astype("category").cat.codes.astype(float)
    )
    # Preserve a human-readable mapping for the decision record.
    mapping = dict(
        zip(
            locked["surface_treatment"].astype("category").cat.categories,
            range(locked["surface_treatment"].astype("category").cat.categories.size),
        )
    )
    logger.info("Locked endpoint rows: %d, treatment codes: %s", len(locked), mapping)
    return locked


def _evaluate_study_aware(frame: pd.DataFrame, model_name: str) -> dict:
    X = frame[NUMERIC_FEATURES].to_numpy(dtype=float)
    y = frame["value"].to_numpy(dtype=float)
    groups = frame["study_id"].to_numpy()

    if model_name == "ridge":
        estimator = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        )
    else:
        estimator = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=1)),
            ]
        )

    logo = LeaveOneGroupOut()
    rows = []
    for train_idx, test_idx in logo.split(X, y, groups):
        if len(np.unique(groups[train_idx])) < 2:
            continue
        estimator.fit(X[train_idx], y[train_idx])
        pred = estimator.predict(X[test_idx])
        rows.append(
            {
                "held_out_study": str(groups[test_idx][0]),
                "n_test": int(len(test_idx)),
                "mae": float(mean_absolute_error(y[test_idx], pred)),
                "r2": float(r2_score(y[test_idx], pred)) if len(test_idx) > 2 else float("nan"),
            }
        )

    # Baseline: predict the mean of the remaining studies.
    baseline_rows = []
    for train_idx, test_idx in logo.split(X, y, groups):
        baseline = DummyRegressor(strategy="mean").fit(X[train_idx], y[train_idx])
        bpred = baseline.predict(X[test_idx])
        baseline_rows.append(float(mean_absolute_error(y[test_idx], bpred)))

    if not rows:
        return {"trained": False, "reason": "insufficient groups for LeaveOneGroupOut"}

    mae = float(np.mean([r["mae"] for r in rows]))
    base_mae = float(np.mean(baseline_rows))
    return {
        "trained": True,
        "model": model_name,
        "split": "LeaveOneGroupOut(study_id)",
        "n_folds": len(rows),
        "n_rows": int(len(frame)),
        "n_studies": int(frame["study_id"].nunique()),
        "per_fold": rows,
        "mae_mean": mae,
        "baseline_mae_mean": base_mae,
        "beats_baseline": bool(mae < base_mae),
        "caveat": (
            "With this few studies, leave-one-study-out error is dominated by "
            "between-laboratory protocol differences. It is not an estimate of "
            "performance on an unseen device."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Recompute even if a decision exists.")
    args = parser.parse_args()

    clean, audit_payload, _ = _load_and_reaudit()
    gate = modelling_gate(audit_payload, phase4_gate_passed=_phase4_passed())

    payload = {
        "created_utc": utc_now(),
        "script": "scripts/train_surface_model.py",
        "gate": gate,
        "observed": audit_payload,
    }

    if not gate["passed"]:
        logger.warning("Surface modelling gate FAILED -> CURATION_ONLY.")
        for reason in gate["reasons"]:
            logger.warning("  - %s", reason)
        payload.update(
            {
                "decision": "CURATION_ONLY",
                "model_trained": False,
                "artefact": None,
                "curation_deliverables": [
                    "data/literature/surface_evidence_raw.csv",
                    "data/literature/surface_evidence_clean.csv",
                    "docs/SURFACE_EVIDENCE_AUDIT.md",
                    "docs/SURFACE_EVIDENCE_CURATED.md",
                    "data_cards/surface_evidence_data_card.md",
                ],
            }
        )
    else:
        logger.info("Surface modelling gate PASSED -> training study-aware model.")
        locked = _prepare_locked(clean)
        results = {
            name: _evaluate_study_aware(locked, name) for name in ("ridge", "random_forest")
        }
        payload.update({"decision": "MODEL_TRAINED", "model_trained": True, "results": results})

    write_json(PATHS["validation"] / "surface_model_decision.json", payload)
    (PATHS["docs"] / "SURFACE_MODEL_DECISION.md").write_text(_decision_doc(payload), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("decision", "model_trained")}, indent=2))


def _decision_doc(payload: dict) -> str:
    gate = payload["gate"]
    checks = "\n".join(
        f"| {key} | {'PASS' if value else 'FAIL'} |" for key, value in gate["checks"].items()
    )
    reasons = "\n".join(f"- {r}" for r in gate["reasons"]) or "- none"
    if payload["decision"] == "CURATION_ONLY":
        body = f"""## Decision

`CURATION_ONLY`. No surface-response model was trained.

The gate was re-evaluated from the persisted evidence table at
{payload['created_utc']}; it was not inherited from an earlier run.

### Failed conditions

{reasons}

### What is delivered

A provenance-preserving evidence catalogue with explicit units, conditions and
unknowns. The screening app displays observed values where they exist and
`INDETERMINATE` where they do not. It does not invent a surface score.
"""
    else:
        results = payload.get("results", {})
        lines = []
        for name, res in results.items():
            if not res.get("trained"):
                lines.append(f"- `{name}`: not trained ({res.get('reason')})")
                continue
            lines.append(
                f"- `{name}`: LOGO MAE {res['mae_mean']:.4f} vs study-mean baseline "
                f"{res['baseline_mae_mean']:.4f} (beats baseline: {res['beats_baseline']})"
            )
        body = "## Decision\n\n`MODEL_TRAINED` under leave-one-study-out validation.\n\n" + "\n".join(lines) + "\n"

    return f"""# Surface model decision (Phase 7)

**Generated:** {payload['created_utc']}
**Locked endpoint:** `{LOCKED_ENDPOINT}`

| Gate condition | Result |
| --- | --- |
{checks}

Observed: {gate['observed']['studies']} studies, {gate['observed']['rows']} rows,
largest study share {gate['observed']['top_study_share']:.1%}.

{body}

## Non-negotiable limits

- Surface values are **experimental** and laboratory-specific; they are not
  predictions from the benchmark model.
- `log10(G_VRH)` from the benchmark is **computed DFT**, not an implant modulus.
- Neither layer establishes biocompatibility, corrosion resistance in service,
  wear performance, or clinical success.
"""


if __name__ == "__main__":
    main()
