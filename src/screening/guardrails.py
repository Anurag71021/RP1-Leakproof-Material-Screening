"""Guardrails that make dishonest screening fail loudly.

The project forbids a single aggregate "suitability" percentage. These helpers
are intentionally asserts, not documentation: if a future edit tries to collapse
criterion-level evidence into one number, the call raises.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

FORBIDDEN_COLUMN_TOKENS = (
    "suitability",
    "composite",
    "overall_score",
    "rank_score",
    "fit_score",
    "implant_score",
    "clinical_score",
)

REQUIRED_MATRIX_COLUMNS = (
    "candidate_id",
    "candidate_label",
    "criterion",
    "value",
    "units",
    "source",
    "evidence_level",
    "applicability",
    "constraint_met",
)


class CompositeScoreError(AssertionError):
    """Raised when code attempts to produce a single suitability number."""


def assert_no_composite_columns(columns: Iterable[str]) -> None:
    lowered = {str(c).lower() for c in columns}
    hits = sorted({token for token in FORBIDDEN_COLUMN_TOKENS if any(token in c for c in lowered)})
    if hits:
        raise CompositeScoreError(
            f"Composite-score columns are forbidden: {hits}. Report criterion-level evidence instead."
        )


def assert_criterion_matrix(frame: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_MATRIX_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Criterion matrix is missing required columns: {missing}")
    assert_no_composite_columns(frame.columns)
    if frame.empty:
        raise ValueError("Criterion matrix is empty.")


def forbid_composite_score(*args, **kwargs) -> None:
    """Deliberate tripwire. Calling this is a design error."""
    raise CompositeScoreError(
        "RP-1 does not produce a composite suitability score. Use criterion-level "
        "evidence, explicit constraints and INDETERMINATE status."
    )


def assert_evidence_levels(frame: pd.DataFrame) -> None:
    allowed = {"observed_literature", "literature_extracted", "benchmark_predicted", "unknown"}
    bad = sorted(set(frame["evidence_level"].dropna()) - allowed)
    if bad:
        raise ValueError(f"Unknown evidence levels: {bad}")
    if (frame["evidence_level"] == "unknown").any():
        # Unknown rows are allowed only when the value is missing.
        unknown_with_value = frame[(frame["evidence_level"] == "unknown") & frame["value"].notna()]
        if len(unknown_with_value):
            raise ValueError("Rows marked 'unknown' must not carry a value.")
