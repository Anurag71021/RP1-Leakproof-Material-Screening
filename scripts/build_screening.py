#!/usr/bin/env python
"""Phase 8: fit the deployment model once, score candidates, and screen them.

Important separation of concerns
--------------------------------
* Reported benchmark metrics come from the **official-fold OOF** predictions
  written in Phase 3. They are never recomputed from the deployment model.
* The deployment model is fitted on all labelled data only because it must
  score *new, unlabelled* candidates. This is standard practice and is labelled
  as such in every artefact.
* Screening output is criterion-level. There is no composite score, and
  candidates with unknown critical criteria are `INDETERMINATE`, never ranked.
"""

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd

from src.config import DISCLAIMER, PATHS, TARGET_COLUMN
from src.data.loader import load_cached
from src.evaluation.applicability import ApplicabilityDomain
from src.models.pipelines import build_pipeline, feature_names
from src.screening.engine import (
    add_modulus_predictions,
    build_criterion_matrix,
    candidate_frame,
    default_candidates,
    rank_candidates,
    sensitivity_analysis,
    summarise_candidates,
)
from src.utils.io import read_json, utc_now, write_json
from src.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_CONSTRAINTS = {
    "modulus_min_log10": 1.40,
    "modulus_max_log10": 2.00,
    "ra_max_um": 2.00,
    "ca_max_deg": 90.0,
}


def _gate_passed() -> bool:
    path = PATHS["docs"] / "LEAKAGE_AUDIT.md"
    if not path.exists():
        return False
    return "LEAKAGE_GATE: PASSED" in path.read_text(encoding="utf-8")


def _load_or_fit_ad(frame: pd.DataFrame) -> ApplicabilityDomain:
    path = PATHS["artifacts"] / "ad_reference.joblib"
    if path.exists():
        logger.info("Loading deployment applicability-domain reference: %s", path)
        return ApplicabilityDomain.load(path)
    logger.warning("AD reference missing; fitting a deployment reference now (all data).")
    ad = ApplicabilityDomain("magpie", k=5).fit(frame)
    ad.save(path)
    return ad


def _load_halfwidth(model_name: str, level: float = 0.9) -> float:
    path = PATHS["uncertainty"] / f"conformal_{model_name}.json"
    if not path.exists():
        raise SystemExit(
            f"Missing {path}. Run `python scripts/run_analysis.py` after the leakage gate passes."
        )
    payload = read_json(path)
    key = str(level)
    if key not in payload["deployment_halfwidths"]:
        raise SystemExit(f"No conformal half-width for level {level} in {path}.")
    return float(payload["deployment_halfwidths"][key])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Default: best model from Phase 3 metrics.")
    parser.add_argument("--ra-max-um", type=float, default=DEFAULT_CONSTRAINTS["ra_max_um"])
    parser.add_argument("--ca-max-deg", type=float, default=DEFAULT_CONSTRAINTS["ca_max_deg"])
    parser.add_argument(
        "--modulus-min", type=float, default=DEFAULT_CONSTRAINTS["modulus_min_log10"]
    )
    parser.add_argument(
        "--modulus-max", type=float, default=DEFAULT_CONSTRAINTS["modulus_max_log10"]
    )
    args = parser.parse_args()

    if not _gate_passed():
        raise SystemExit(
            "LEAKAGE_GATE is not PASSED. Screening is blocked. "
            "Run scripts/finalize_gate.py first."
        )

    metrics = read_json(PATHS["metrics"] / "benchmark_metrics.json")
    best = read_json(PATHS["metrics"] / "best_model.json")
    model_name = args.model or best["model"]
    if model_name not in metrics["models"]:
        raise SystemExit(f"Model {model_name!r} has no benchmark metrics.")

    frame, provenance = load_cached()
    if TARGET_COLUMN not in frame.columns:
        raise SystemExit("Benchmark frame is missing the target column.")

    # ------------------------------------------------------------------ model
    logger.info("Fitting deployment pipeline (%s) on all %d labelled rows.", model_name, len(frame))
    pipeline = build_pipeline(model_name, "magpie", random_state=42)
    pipeline.fit(frame[["structure"]], frame[TARGET_COLUMN].to_numpy(dtype=float))
    names = feature_names(pipeline)
    if not names:
        raise SystemExit("Fitted pipeline produced no feature names.")

    model_path = PATHS["artifacts"] / "model_log_gvrh.joblib"
    joblib.dump(
        {
            "pipeline": pipeline,
            "model_name": model_name,
            "target": TARGET_COLUMN,
            "target_units": "log10(GPa)",
            "n_train": int(len(frame)),
            "feature_names": names,
            "trained_utc": utc_now(),
            "caveat": (
                "Deployment fit on all labelled data. Reported benchmark metrics come "
                "from official-fold OOF predictions, not from this object."
            ),
        },
        model_path,
    )
    logger.info("Saved deployment model to %s (%d features).", model_path, len(names))

    # ------------------------------------------------------------------ AD + UQ
    ad = _load_or_fit_ad(frame)
    halfwidth = _load_halfwidth(model_name, 0.9)

    # -------------------------------------------------------------- candidates
    candidates = default_candidates()
    cand_frame = candidate_frame(candidates)
    ad_scores = ad.score(cand_frame)
    predictions = add_modulus_predictions(cand_frame, pipeline, halfwidth, ad_scores)

    clean_path = PATHS["data_literature"] / "surface_evidence_clean.csv"
    if not clean_path.exists():
        raise SystemExit(
            f"Missing {clean_path}. Run `python scripts/build_literature.py` first."
        )
    clean = pd.read_csv(clean_path)

    constraints = {
        "modulus_min_log10": args.modulus_min,
        "modulus_max_log10": args.modulus_max,
        "ra_max_um": args.ra_max_um,
        "ca_max_deg": args.ca_max_deg,
    }
    matrix = build_criterion_matrix(predictions, clean, candidates, constraints)
    summary = summarise_candidates(matrix)
    ranking = rank_candidates(
        matrix,
        summary,
        order=("predicted_log_gvrh", "surface_roughness_ra_um"),
        ascending=(True, True),
    )
    sensitivity = sensitivity_analysis(matrix, summary, constraints=constraints)

    # ------------------------------------------------------------------- write
    PATHS["screening"].mkdir(parents=True, exist_ok=True)
    candidates_out = cand_frame.drop(columns=["structure"]).copy()
    predictions_out = predictions.drop(columns=["structure"]).reset_index(drop=True)
    predictions_out.to_csv(PATHS["screening"] / "candidate_predictions.csv", index=False)
    candidates_out.to_csv(PATHS["screening"] / "candidates.csv", index=False)
    matrix.to_csv(PATHS["screening"] / "criterion_matrix.csv", index=False)
    summary.to_csv(PATHS["screening"] / "candidate_summary.csv", index=False)
    if len(ranking):
        ranking.to_csv(PATHS["screening"] / "ranking.csv", index=False)
    else:
        pd.DataFrame(
            columns=[
                "rank",
                "candidate_id",
                "candidate_label",
                "status",
                "predicted_log_gvrh",
                "surface_roughness_ra_um",
            ]
        ).to_csv(PATHS["screening"] / "ranking.csv", index=False)
    write_json(PATHS["screening"] / "sensitivity.json", sensitivity)

    payload = {
        "created_utc": utc_now(),
        "model": model_name,
        "model_artefact": "artifacts/model_log_gvrh.joblib",
        "model_fit": "all labelled rows (deployment only)",
        "benchmark_metrics_source": "results/metrics/benchmark_metrics.json (official OOF)",
        "benchmark_metrics": metrics["models"][model_name]["summary"],
        "target": TARGET_COLUMN,
        "target_units": "log10(GPa)",
        "conformal_level": 0.9,
        "conformal_halfwidth": halfwidth,
        "ad_reference": "artifacts/ad_reference.joblib",
        "constraints": constraints,
        "n_candidates": int(len(candidates)),
        "status_counts": summary["status"].value_counts().to_dict(),
        "ranking_order": ["predicted_log_gvrh", "surface_roughness_ra_um"],
        "sensitivity_top_candidates": sensitivity["top_candidates"],
        "sensitivity_top_flips": sensitivity["top_flips"],
        "provenance": {
            "dataset": provenance.get("dataset_id"),
            "official_hash": provenance.get("official_hash"),
            "cache_sha256": provenance.get("cache_sha256"),
        },
        "disclaimer": DISCLAIMER,
        "warnings": [
            "Predicted log10(G_VRH) is computed DFT shear modulus, not an experimental implant modulus.",
            "Observed Ra and contact angle values are laboratory-specific and not directly comparable across protocols.",
            "No composite suitability score is produced. INDETERMINATE candidates are not ranked.",
            "Applicability-domain status describes descriptor-space support, not probability of success.",
        ],
    }
    write_json(PATHS["screening"] / "screening_summary.json", payload)
    (PATHS["screening"] / "README.md").write_text(_readme(payload, summary, ranking), encoding="utf-8")

    print(json.dumps(payload, indent=2, default=str))


def _readme(payload: dict, summary: pd.DataFrame, ranking: pd.DataFrame) -> str:
    status_lines = "\n".join(
        f"| {status} | {count} |" for status, count in payload["status_counts"].items()
    )
    ranked_lines = (
        "\n".join(
            f"| {row.rank} | {row.candidate_label} | {row.predicted_log_gvrh:.3f} | "
            f"{row.surface_roughness_ra_um:.3f} |"
            for row in ranking.itertuples()
        )
        if len(ranking)
        else "| - | No candidate met all critical constraints with complete evidence | - | - |"
    )
    indeterminate = summary[summary["status"] == "INDETERMINATE"]
    indeterminate_lines = "\n".join(
        f"| {row.candidate_label} | {row.indeterminate_reasons} |"
        for row in indeterminate.itertuples()
    ) or "| none | - |"

    return f"""# Screening output - reading guide

**Generated:** {payload['created_utc']}
**Model:** `{payload['model']}` (deployment fit on all labelled rows)
**Target:** `{payload['target']}` in `{payload['target_units']}` - computed DFT shear modulus

> {DISCLAIMER}

## Constraint thresholds used

| Constraint | Value |
| --- | --- |
| Modulus (log10 GPa) | {payload['constraints']['modulus_min_log10']} - {payload['constraints']['modulus_max_log10']} |
| Ra max (um) | {payload['constraints']['ra_max_um']} |
| Contact angle max (deg, non-critical) | {payload['constraints']['ca_max_deg']} |

These thresholds are **user research constraints**, not clinical acceptance
limits.

## Candidate status counts

| Status | Candidates |
| --- | --- |
{status_lines}

## Ranked candidates

Lexicographic order: `{ ' -> '.join(payload['ranking_order']) }`. Ties are ties;
no weighted aggregate is computed. Ascending modulus reflects a stress-shielding
research hypothesis (lower stiffness is closer to bone), **not** a clinical
requirement; ascending Ra reflects the user's surface constraint.

| Rank | Candidate | Predicted log10(GPa) | Ra (um) |
| --- | --- | --- | --- |
{ranked_lines}

## Indeterminate candidates

| Candidate | Why |
| --- | --- |
{indeterminate_lines}

## Files

| File | Contents |
| --- | --- |
| `candidate_predictions.csv` | Prediction, 90% conformal interval, AD status, KNN distance |
| `candidates.csv` | Candidate identities and alloy family |
| `criterion_matrix.csv` | Long-format criterion-level evidence with source and evidence level |
| `candidate_summary.csv` | Status and transparent counts (not a score) |
| `ranking.csv` | Lexicographic ranking of fully supported candidates only |
| `sensitivity.json` | Whether the top candidate changes when priority order changes |

## Mandatory caveats

1. Modulus is predicted from **composition only**, using a model trained on a
   broad inorganic materials dataset. It is not measured on the candidate.
2. Ra and contact angle values come from specific laboratory protocols. Values
   from different studies are not interchangeable.
3. A candidate marked `RANKED` means "all critical criteria are known and meet
   the user's constraints". It does not mean "suitable for implantation".
4. `INDETERMINATE` means the evidence is missing, not that the candidate is bad.
5. Sensitivity: top candidates under alternative priorities were
   `{', '.join(payload['sensitivity_top_candidates']) or 'none'}`;
   flips observed: `{payload['sensitivity_top_flips']}`.
"""


if __name__ == "__main__":
    main()
