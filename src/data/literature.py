"""Curation-first literature evidence layer.

The layer is deliberately conservative:

* every row carries a DOI, source location and extraction note;
* units are harmonised only when the conversion is unambiguous;
* "not reported" is NaN, never zero;
* duplicates are removed on a documented composite key;
* a modelling gate decides whether a surface model is defensible at all.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = [
    "study_id",
    "doi",
    "title",
    "year",
    "source_url",
    "access_status",
    "extraction_location",
    "extraction_note",
    "alloy_family",
    "material_grade_or_formula",
    "substrate_form",
    "surface_treatment",
    "endpoint_key",
    "endpoint_label",
    "value_original",
    "unit_original",
    "conditions_key",
    "evidence_level",
]

ALLOWED = {
    "access_status": {"open_access", "abstract_only"},
    "alloy_family": {"Ti_alloy", "CoCr_alloy", "stainless_steel", "Mg_alloy", "other_metal"},
    "substrate_form": {"bulk", "coating", "porous", "additive"},
    "evidence_level": {
        "observed_literature",
        "literature_extracted",
        "benchmark_predicted",
        "unknown",
    },
    "dispersion_type": {"SD", "SEM", "CI95", "range"},
    "reference_electrode": {"SCE", "Ag/AgCl", "SHE"},
}

ENDPOINT_UNITS = {
    "surface_roughness_ra_um": {"um", "µm", "nm"},
    "surface_roughness_rz_um": {"um", "µm", "nm"},
    "water_contact_angle_deg": {"deg", "degrees", "°"},
    "corrosion_ecorr_v": {"V", "mV"},
    "corrosion_icorr_ua_cm2": {"uA/cm2", "µA/cm²", "µA/cm2"},
    "oxide_layer_thickness_nm": {"nm", "um", "µm"},
    "hardness_hv": {"HV"},
    "elastic_modulus_gpa": {"GPa"},
}

DEDUP_KEY = [
    "study_id",
    "material_grade_or_formula",
    "surface_treatment",
    "coating_chemistry",
    "endpoint_key",
    "conditions_key",
]

LOCKED_ENDPOINT = "surface_roughness_ra_um"


def _require_columns(frame: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required evidence columns: {missing}")


def validate_evidence(frame: pd.DataFrame) -> dict[str, Any]:
    """Validate the curated table and return a machine-readable report."""
    _require_columns(frame)
    if frame.empty:
        raise ValueError("Evidence table is empty.")

    report: dict[str, Any] = {"n_rows": int(len(frame)), "errors": []}

    for column, allowed in ALLOWED.items():
        if column not in frame.columns:
            continue
        bad = sorted(set(frame[column].dropna().astype(str)) - allowed)
        if bad:
            report["errors"].append(f"{column}: disallowed values {bad}")

    for idx, row in frame.iterrows():
        endpoint = str(row["endpoint_key"])
        if endpoint not in ENDPOINT_UNITS:
            report["errors"].append(f"row {idx}: unknown endpoint {endpoint}")
            continue
        unit = str(row["unit_original"]).strip()
        if unit not in ENDPOINT_UNITS[endpoint]:
            report["errors"].append(f"row {idx}: unit {unit!r} invalid for {endpoint}")
        try:
            value = float(row["value_original"])
        except (TypeError, ValueError):
            report["errors"].append(f"row {idx}: non-numeric value_original")
            continue
        if not np.isfinite(value):
            report["errors"].append(f"row {idx}: non-finite value_original")
        if endpoint == "surface_roughness_ra_um" and value < 0:
            report["errors"].append(f"row {idx}: negative Ra")

        doi = str(row["doi"]).lower()
        study_id = str(row["study_id"]).lower()
        if not doi.startswith("10."):
            report["errors"].append(f"row {idx}: DOI does not start with 10.")
        if not study_id.startswith(doi):
            report["errors"].append(f"row {idx}: study_id {study_id!r} not derived from DOI {doi!r}")

        if "n_replicates" in frame.columns and pd.notna(row.get("n_replicates")):
            try:
                if int(row["n_replicates"]) < 1:
                    report["errors"].append(f"row {idx}: n_replicates < 1")
            except (TypeError, ValueError):
                report["errors"].append(f"row {idx}: n_replicates not an integer")

        if "sd_or_sem" in frame.columns and pd.notna(row.get("sd_or_sem")):
            try:
                if float(row["sd_or_sem"]) < 0:
                    report["errors"].append(f"row {idx}: negative dispersion")
            except (TypeError, ValueError):
                report["errors"].append(f"row {idx}: non-numeric dispersion")

    report["passed"] = not report["errors"]
    if not report["passed"]:
        raise ValueError("Evidence validation failed:\n" + "\n".join(report["errors"][:20]))
    return report


def harmonise_units(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert units only where the mapping is unambiguous.

    Original values and units remain in the ``*_original`` columns.
    """
    out = frame.copy()
    values, units = [], []
    for _, row in out.iterrows():
        endpoint = str(row["endpoint_key"])
        unit = str(row["unit_original"]).strip()
        value = float(row["value_original"])
        if endpoint in {"surface_roughness_ra_um", "surface_roughness_rz_um"}:
            if unit == "nm":
                value, unit = value / 1000.0, "um"
            else:
                unit = "um"
        elif endpoint == "water_contact_angle_deg":
            unit = "degrees"
        elif endpoint == "corrosion_ecorr_v":
            if unit == "mV":
                value, unit = value / 1000.0, "V"
        elif endpoint == "oxide_layer_thickness_nm":
            if unit in {"um", "µm"}:
                value, unit = value * 1000.0, "nm"
        elif endpoint == "corrosion_icorr_ua_cm2":
            unit = "uA/cm2"
        values.append(value)
        units.append(unit)
    out["value"] = values
    out["unit"] = units
    return out


def deduplicate(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Remove duplicates on the documented key and log what was dropped."""
    for column in DEDUP_KEY:
        if column not in frame.columns:
            frame[column] = np.nan
    key = frame[DEDUP_KEY].astype(str).agg(" | ".join, axis=1)
    duplicated = key.duplicated(keep="first")
    dropped = frame.loc[duplicated].copy()
    dropped["duplicate_key"] = key[duplicated]
    clean = frame.loc[~duplicated].copy()
    log = {
        "key": DEDUP_KEY,
        "n_input": int(len(frame)),
        "n_duplicates_removed": int(duplicated.sum()),
        "n_output": int(len(clean)),
        "removed_rows": dropped[
            ["study_id", "material_grade_or_formula", "surface_treatment", "endpoint_key", "conditions_key", "duplicate_key"]
        ].to_dict(orient="records")
        if len(dropped)
        else [],
    }
    return clean.reset_index(drop=True), log


def audit(
    raw: pd.DataFrame,
    clean: pd.DataFrame,
    dedup_log: dict[str, Any],
    excluded_studies: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Compute the evidence audit, including the modelling gate inputs."""
    n = len(clean)
    per_study = clean.groupby("study_id").size().sort_values(ascending=False)
    per_endpoint = clean.groupby("endpoint_key").size().sort_values(ascending=False)

    locked = clean[clean["endpoint_key"] == LOCKED_ENDPOINT]
    locked_studies = locked["study_id"].nunique()
    locked_top_share = (
        float(locked.groupby("study_id").size().max() / len(locked)) if len(locked) else float("nan")
    )

    missing = {
        column: int(clean[column].isna().sum())
        for column in [
            "coating_chemistry",
            "treatment_parameters",
            "sd_or_sem",
            "dispersion_type",
            "n_replicates",
            "measurement_method",
            "reference_electrode",
            "conditions_detail",
        ]
        if column in clean.columns
    }

    return {
        "n_raw": int(len(raw)),
        "n_clean": int(n),
        "n_duplicates_removed": int(dedup_log["n_duplicates_removed"]),
        "n_distinct_studies": int(clean["study_id"].nunique()),
        "rows_per_study": {
            "min": int(per_study.min()) if len(per_study) else 0,
            "median": float(per_study.median()) if len(per_study) else 0.0,
            "max": int(per_study.max()) if len(per_study) else 0,
            "by_study": {k: int(v) for k, v in per_study.items()},
        },
        "top_study_share": float(per_study.max() / n) if n else float("nan"),
        "endpoint_counts": {k: int(v) for k, v in per_endpoint.items()},
        "units_by_endpoint": {
            endpoint: sorted(clean.loc[clean["endpoint_key"] == endpoint, "unit"].dropna().unique().tolist())
            for endpoint in sorted(clean["endpoint_key"].unique())
        },
        "missing_counts": missing,
        "locked_endpoint": LOCKED_ENDPOINT,
        "locked_endpoint_rows": int(len(locked)),
        "locked_endpoint_studies": int(locked_studies),
        "locked_endpoint_top_study_share": locked_top_share,
        "excluded_studies": excluded_studies or [],
        "dedup_log": dedup_log,
    }


def modelling_gate(
    audit_payload: dict[str, Any],
    phase4_gate_passed: bool,
    min_studies: int = 3,
    min_rows: int = 25,
    max_single_study_share: float = 0.40,
) -> dict[str, Any]:
    """Apply the Phase 7 gate. All conditions must hold to train."""
    checks = {
        "a_min_studies": audit_payload["locked_endpoint_studies"] >= min_studies,
        "b_min_rows": audit_payload["locked_endpoint_rows"] >= min_rows,
        "c_no_dominant_study": audit_payload["locked_endpoint_top_study_share"] <= max_single_study_share,
        "d_study_aware_split_feasible": audit_payload["locked_endpoint_studies"] >= 3,
        "e_phase4_gate_passed": bool(phase4_gate_passed),
    }
    passed = all(checks.values())
    reasons = []
    if not checks["a_min_studies"]:
        reasons.append(
            f"only {audit_payload['locked_endpoint_studies']} distinct studies for "
            f"{LOCKED_ENDPOINT} (< {min_studies})"
        )
    if not checks["b_min_rows"]:
        reasons.append(
            f"only {audit_payload['locked_endpoint_rows']} deduplicated rows for "
            f"{LOCKED_ENDPOINT} (< {min_rows})"
        )
    if not checks["c_no_dominant_study"]:
        reasons.append(
            f"largest study contributes {audit_payload['locked_endpoint_top_study_share']:.0%} "
            f"of rows (> {max_single_study_share:.0%})"
        )
    if not checks["d_study_aware_split_feasible"]:
        reasons.append("fewer than 3 groups; LOGO/GroupKFold would not be meaningful")
    if not checks["e_phase4_gate_passed"]:
        reasons.append("Phase 4 leakage gate is not PASSED")

    return {
        "locked_endpoint": LOCKED_ENDPOINT,
        "thresholds": {
            "min_studies": min_studies,
            "min_rows": min_rows,
            "max_single_study_share": max_single_study_share,
        },
        "observed": {
            "studies": audit_payload["locked_endpoint_studies"],
            "rows": audit_payload["locked_endpoint_rows"],
            "top_study_share": audit_payload["locked_endpoint_top_study_share"],
        },
        "checks": checks,
        "passed": passed,
        "decision": "TRAIN_SURFACE_MODEL" if passed else "CURATION_ONLY",
        "reasons": reasons,
    }
