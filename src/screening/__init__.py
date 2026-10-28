"""Criterion-level screening engine and composite-score guardrails."""

from src.screening.engine import (
    build_criterion_matrix,
    default_candidates,
    rank_candidates,
    sensitivity_analysis,
    summarise_candidates,
)
from src.screening.guardrails import CompositeScoreError, assert_no_composite_columns

__all__ = [
    "build_criterion_matrix",
    "default_candidates",
    "rank_candidates",
    "sensitivity_analysis",
    "summarise_candidates",
    "CompositeScoreError",
    "assert_no_composite_columns",
]
