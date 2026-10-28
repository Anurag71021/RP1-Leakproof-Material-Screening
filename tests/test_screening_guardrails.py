"""Screening honesty: no composite score, honest INDETERMINATE handling, provenance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import PATHS
from src.screening.engine import (
    build_criterion_matrix,
    default_candidates,
    rank_candidates,
    sensitivity_analysis,
    summarise_candidates,
)
from src.screening.guardrails import (
    CompositeScoreError,
    assert_criterion_matrix,
    assert_evidence_levels,
    assert_no_composite_columns,
    forbid_composite_score,
)

MATRIX_COLUMNS = [
    "candidate_id",
    "candidate_label",
    "criterion",
    "value",
    "units",
    "source",
    "evidence_level",
    "applicability",
    "constraint_met",
    "critical",
]


def _row(cid, criterion, value, status, critical=True, evidence="benchmark_predicted"):
    return {
        "candidate_id": cid,
        "candidate_label": cid.upper(),
        "criterion": criterion,
        "value": value,
        "units": "log10(GPa)" if criterion == "predicted_log_gvrh" else "um",
        "source": "test",
        "evidence_level": evidence,
        "applicability": "IN_DOMAIN",
        "constraint_met": status,
        "critical": critical,
    }


def _matrix(rows):
    return pd.DataFrame(rows, columns=MATRIX_COLUMNS)


@pytest.mark.parametrize(
    "column",
    ["suitability", "composite_score", "overall_score", "implant_score", "clinical_score"],
)
def test_composite_columns_are_rejected(column):
    with pytest.raises(CompositeScoreError):
        assert_no_composite_columns(["candidate_id", column])


def test_composite_score_function_is_a_tripwire():
    with pytest.raises(CompositeScoreError):
        forbid_composite_score()


def test_missing_required_matrix_columns_raise():
    frame = _matrix([]).drop(columns=["source"])
    with pytest.raises(ValueError):
        assert_criterion_matrix(frame)


def test_unknown_critical_criterion_makes_candidate_indeterminate():
    matrix = _matrix(
        [
            _row("A", "predicted_log_gvrh", 1.5, "YES"),
            _row("A", "surface_roughness_ra_um", np.nan, "NA", evidence="unknown"),
            _row("B", "predicted_log_gvrh", 1.6, "YES"),
            _row("B", "surface_roughness_ra_um", 0.4, "YES", evidence="observed_literature"),
        ]
    )
    summary = summarise_candidates(matrix).set_index("candidate_id")
    assert summary.loc["A", "status"] == "INDETERMINATE"
    assert "surface_roughness_ra_um unknown" in summary.loc["A", "indeterminate_reasons"]
    assert summary.loc["B", "status"] == "RANKED"


def test_unknown_noncritical_criterion_does_not_block_ranking():
    matrix = _matrix(
        [
            _row("A", "predicted_log_gvrh", 1.5, "YES"),
            _row("A", "surface_roughness_ra_um", 0.5, "YES", evidence="observed_literature"),
            _row("A", "water_contact_angle_deg", np.nan, "NA", critical=False, evidence="unknown"),
        ]
    )
    summary = summarise_candidates(matrix)
    assert summary.loc[0, "status"] == "RANKED"
    assert summary.loc[0, "criteria_unknown"] == 0


def test_failed_critical_constraint_is_not_ranked():
    matrix = _matrix(
        [
            _row("A", "predicted_log_gvrh", 2.5, "NO"),
            _row("A", "surface_roughness_ra_um", 0.5, "YES", evidence="observed_literature"),
        ]
    )
    summary = summarise_candidates(matrix)
    assert summary.loc[0, "status"] == "CONSTRAINT_FAILED"
    ranked = rank_candidates(matrix, summary, order=("predicted_log_gvrh",), ascending=(True,))
    assert ranked.empty


def test_noncritical_failure_is_visible_but_not_gating():
    matrix = _matrix(
        [
            _row("A", "predicted_log_gvrh", 1.5, "YES"),
            _row("A", "surface_roughness_ra_um", 0.5, "YES", evidence="observed_literature"),
            _row("A", "water_contact_angle_deg", 120.0, "NO", critical=False, evidence="observed_literature"),
        ]
    )
    summary = summarise_candidates(matrix)
    assert summary.loc[0, "status"] == "RANKED"
    assert summary.loc[0, "criteria_failed"] == 0  # critical failures only


def test_ranking_is_lexicographic_and_keeps_ties_transparent():
    matrix = _matrix(
        [
            _row("A", "predicted_log_gvrh", 1.5, "YES"),
            _row("A", "surface_roughness_ra_um", 0.3, "YES", evidence="observed_literature"),
            _row("B", "predicted_log_gvrh", 1.5, "YES"),
            _row("B", "surface_roughness_ra_um", 0.2, "YES", evidence="observed_literature"),
            _row("C", "predicted_log_gvrh", 1.4, "YES"),
            _row("C", "surface_roughness_ra_um", 1.9, "YES", evidence="observed_literature"),
        ]
    )
    summary = summarise_candidates(matrix)
    ranked = rank_candidates(matrix, summary)
    assert ranked["candidate_id"].tolist() == ["C", "B", "A"]
    assert "score" not in " ".join(ranked.columns).lower()
    assert ranked["rank"].tolist() == [1, 2, 3]


def test_sensitivity_detects_priority_flip():
    # Genuine trade-off: A is better on modulus, B is better on roughness.
    # If one candidate dominated on both criteria there would be no flip, and
    # asserting one would be testing a fiction.
    matrix = _matrix(
        [
            _row("A", "predicted_log_gvrh", 1.4, "YES"),
            _row("A", "surface_roughness_ra_um", 1.5, "YES", evidence="observed_literature"),
            _row("B", "predicted_log_gvrh", 1.9, "YES"),
            _row("B", "surface_roughness_ra_um", 0.2, "YES", evidence="observed_literature"),
        ]
    )
    summary = summarise_candidates(matrix)
    result = sensitivity_analysis(matrix, summary)
    assert result["top_flips"] is True
    assert set(result["top_candidates"]) == {"A", "B"}
    assert result["orders_tested"] >= 2


def test_unknown_evidence_level_must_not_carry_a_value():
    matrix = _matrix([_row("A", "surface_roughness_ra_um", 0.4, "NA", evidence="unknown")])
    with pytest.raises(ValueError):
        assert_evidence_levels(matrix)


def test_default_candidates_have_unique_ids_and_valid_compositions():
    candidates = default_candidates()
    ids = [c.candidate_id for c in candidates]
    assert len(ids) == len(set(ids))
    for candidate in candidates:
        composition = candidate.composition()
        assert len(composition.elements) >= 1
        assert composition.num_atoms > 0


def test_default_candidate_evidence_links_exist_in_curated_table(clean_evidence):
    """Evidence references must not silently drift away from the curated data."""
    linked = 0
    for candidate in default_candidates():
        for ref in candidate.evidence:
            hit = clean_evidence[
                (clean_evidence["study_id"] == ref["study_id"])
                & (clean_evidence["material_grade_or_formula"] == ref["material"])
                & (clean_evidence["surface_treatment"] == ref["treatment"])
            ]
            assert not hit.empty, f"Evidence link not found for {candidate.candidate_id}: {ref}"
            linked += 1
    assert linked >= 10


def test_persisted_matrix_is_traceable_and_not_composite(screening_matrix):
    assert_criterion_matrix(screening_matrix)
    assert screening_matrix["candidate_id"].nunique() >= 10
    observed = screening_matrix[screening_matrix["evidence_level"] == "observed_literature"]
    assert len(observed) > 0
    assert observed["source"].str.startswith("doi:10.").all()
    unknowns = screening_matrix[screening_matrix["evidence_level"] == "unknown"]
    assert unknowns["value"].isna().all()


def test_persisted_summary_matches_matrix(screening_matrix):
    summary = pd.read_csv(PATHS["screening"] / "candidate_summary.csv")
    assert set(summary["candidate_id"]) == set(screening_matrix["candidate_id"])
    assert summary["status"].isin(["RANKED", "CONSTRAINT_FAILED", "INDETERMINATE"]).all()
    indeterminate = summary[summary["status"] == "INDETERMINATE"]
    for row in indeterminate.itertuples():
        assert row.criteria_unknown >= 1
        assert row.indeterminate_reasons
    for column in summary.columns:
        assert "suitability" not in column.lower()
        assert "score" not in column.lower()


def test_ranking_file_contains_only_supported_candidates():
    path = PATHS["screening"] / "ranking.csv"
    ranking = pd.read_csv(path)
    summary = pd.read_csv(PATHS["screening"] / "candidate_summary.csv")
    allowed = set(summary.loc[summary["status"] == "RANKED", "candidate_id"])
    assert set(ranking["candidate_id"]).issubset(allowed)
    assert set(ranking.columns) == {
        "rank",
        "candidate_id",
        "candidate_label",
        "status",
        "predicted_log_gvrh",
        "surface_roughness_ra_um",
    }


def test_screening_artefacts_carry_disclaimer_and_caveats():
    import json

    payload = json.loads((PATHS["screening"] / "screening_summary.json").read_text())
    assert "not for clinical use" in payload["disclaimer"].lower()
    assert payload["target_units"] == "log10(GPa)"
    assert len(payload["warnings"]) >= 3
    assert payload["status_counts"]
