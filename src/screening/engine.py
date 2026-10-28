"""Transparent candidate screening.

Four source pools are kept separate and labelled on every row:

1. ``observed_literature`` - measured surface values from curated studies;
2. ``benchmark_predicted`` - computed log10(G_VRH) from the ML model;
3. ``literature_experimental`` - reserved for other experimental values;
4. ``user_constraint`` - thresholds supplied by the user.

There is no composite score. Candidates are either ``RANKED`` (all critical
criteria known and constraints met), ``CONSTRAINT_FAILED``, or
``INDETERMINATE`` (a critical criterion is unknown). Ranking is lexicographic by
a user-selected criterion order, never by an aggregate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element

from src.screening.guardrails import assert_criterion_matrix, assert_evidence_levels

IN_DOMAIN = "IN_DOMAIN"
NEAR_BOUNDARY = "NEAR_BOUNDARY"
OUT_OF_DOMAIN = "OUT_OF_DOMAIN"


@dataclass
class Candidate:
    candidate_id: str
    label: str
    alloy_family: str
    wt_percent: dict[str, float]
    evidence: list[dict[str, str]] = field(default_factory=list)
    note: str = ""

    def composition(self) -> Composition:
        moles = {el: w / Element(el).atomic_mass for el, w in self.wt_percent.items()}
        total = sum(moles.values())
        return Composition({el: 100.0 * m / total for el, m in moles.items()})


def default_candidates() -> list[Candidate]:
    """Material + surface-state candidates with explicit evidence links."""
    ti64 = {"Ti": 90.0, "Al": 6.0, "V": 4.0}
    return [
        Candidate(
            "ti64_eli_tumbled",
            "Ti-6Al-4V ELI - vibration tumbled",
            "Ti_alloy",
            ti64,
            [{"study_id": "10.1038/s41598-020-76448-w", "material": "Ti-6Al-4V ELI", "treatment": "vibration tumbling (control)"}],
        ),
        Candidate(
            "ti64_eli_mao",
            "Ti-6Al-4V ELI - MAO (alkaline silicate)",
            "Ti_alloy",
            ti64,
            [{"study_id": "10.1038/s41598-020-76448-w", "material": "Ti-6Al-4V ELI", "treatment": "micro-arc oxidation (MAO, alkaline silicate)"}],
        ),
        Candidate(
            "ti64_eli_mao_blasted",
            "Ti-6Al-4V ELI - MAO + blasting",
            "Ti_alloy",
            ti64,
            [{"study_id": "10.1038/s41598-020-76448-w", "material": "Ti-6Al-4V ELI", "treatment": "MAO + blasting (outer layer removed)"}],
        ),
        Candidate(
            "ti64_am_uncoated",
            "Ti-6Al-4V AM - chemically polished (uncoated)",
            "Ti_alloy",
            ti64,
            [{"study_id": "10.3390/ijms27104335", "material": "Ti-6Al-4V (additively manufactured porous implant)", "treatment": "uncoated (chemically polished)"}],
        ),
        Candidate(
            "ti64_am_nanoporous",
            "Ti-6Al-4V AM - anodized nanoporous",
            "Ti_alloy",
            ti64,
            [{"study_id": "10.3390/ijms27104335", "material": "Ti-6Al-4V (additively manufactured porous implant)", "treatment": "anodized nanoporous"}],
        ),
        Candidate(
            "ti64_am_nanotubular",
            "Ti-6Al-4V AM - anodized nanotubular",
            "Ti_alloy",
            ti64,
            [{"study_id": "10.3390/ijms27104335", "material": "Ti-6Al-4V (additively manufactured porous implant)", "treatment": "anodized nanotubular"}],
        ),
        Candidate(
            "ti64_laser",
            "Ti-6Al-4V - laser textured (P29)",
            "Ti_alloy",
            ti64,
            [{"study_id": "10.1016/j.heliyon.2024.e37808", "material": "Ti-6Al-4V", "treatment": "laser texturing (P29, 29 um pitch)"}],
            note="Source reports Sa, not Ra; surface roughness criterion is therefore unknown.",
        ),
        Candidate(
            "ti64_laser_cyclohexane",
            "Ti-6Al-4V - laser textured + cyclohexane 70 C",
            "Ti_alloy",
            ti64,
            [{"study_id": "10.1016/j.heliyon.2024.e37808", "material": "Ti-6Al-4V", "treatment": "laser texturing + cyclohexane 70 C 2 h"}],
            note="Source reports Sa, not Ra; surface roughness criterion is therefore unknown.",
        ),
        Candidate(
            "cocr_polished_2",
            "CoCr (Co-28Cr-6Mo) - polishing level 2",
            "CoCr_alloy",
            {"Co": 66.0, "Cr": 28.0, "Mo": 6.0},
            [{"study_id": "10.3390/ma14133702", "material": "CoCr (Co-28Cr-6Mo)", "treatment": "polishing_level_2"}],
        ),
        Candidate(
            "ss316l_polished_2",
            "316L stainless steel - polishing level 2",
            "stainless_steel",
            {"Fe": 65.0, "Cr": 17.0, "Ni": 12.0, "Mo": 2.5, "Mn": 2.0, "Si": 1.0},
            [{"study_id": "10.3390/ma14133702", "material": "316L stainless steel (SUS 316)", "treatment": "polishing_level_2"}],
        ),
        Candidate(
            "ti15mo_polished_2",
            "Ti-15Mo-5Zr-3Al - polishing level 2",
            "Ti_alloy",
            {"Ti": 77.0, "Mo": 15.0, "Zr": 5.0, "Al": 3.0},
            [{"study_id": "10.3390/ma14133702", "material": "Ti-15Mo-5Zr-3Al", "treatment": "polishing_level_2"}],
        ),
        Candidate(
            "ti13nb13zr_bare",
            "Ti-13Nb-13Zr - no surface evidence",
            "Ti_alloy",
            {"Ti": 74.0, "Nb": 13.0, "Zr": 13.0},
            [],
            note="No in-scope surface measurement curated; cannot be ranked.",
        ),
        Candidate(
            "cpti_grade4",
            "cp-Ti grade 4 - no surface evidence",
            "Ti_alloy",
            {"Ti": 100.0},
            [],
            note="No in-scope surface measurement curated; cannot be ranked.",
        ),
        Candidate(
            "mg_az31",
            "Mg AZ31 - no surface evidence",
            "Mg_alloy",
            {"Mg": 96.0, "Al": 3.0, "Zn": 1.0},
            [],
            note="Biodegradable Mg alloy; no in-scope surface measurement curated.",
        ),
    ]


def candidate_frame(candidates: list[Candidate]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": [c.candidate_id for c in candidates],
            "candidate_label": [c.label for c in candidates],
            "alloy_family": [c.alloy_family for c in candidates],
            "structure": [c.composition() for c in candidates],
        }
    ).set_index("candidate_id", drop=False)


def add_modulus_predictions(
    frame: pd.DataFrame,
    pipeline,
    interval_halfwidth: float,
    ad_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Attach benchmark prediction, conformal interval and AD label."""
    preds = pipeline.predict(frame[["structure"]])
    out = frame.copy()
    out["predicted_log_gvrh"] = np.asarray(preds, dtype=float)
    out["prediction_lower"] = out["predicted_log_gvrh"] - interval_halfwidth
    out["prediction_upper"] = out["predicted_log_gvrh"] + interval_halfwidth
    out = out.join(ad_scores[["ad_status", "knn_distance"]], how="left")
    return out


def _evidence_rows(clean: pd.DataFrame, refs: list[dict[str, str]]) -> pd.DataFrame:
    if not refs:
        return clean.iloc[0:0]
    parts = []
    for ref in refs:
        hit = clean[
            (clean["study_id"] == ref["study_id"])
            & (clean["material_grade_or_formula"] == ref["material"])
            & (clean["surface_treatment"] == ref["treatment"])
        ]
        parts.append(hit)
    return pd.concat(parts, ignore_index=True) if parts else clean.iloc[0:0]


def _classify(value: float, low: float, high: float) -> str:
    if pd.isna(value):
        return "NA"
    return "YES" if low <= value <= high else "NO"


def build_criterion_matrix(
    predictions: pd.DataFrame,
    clean: pd.DataFrame,
    candidates: list[Candidate],
    constraints: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Long-format criterion matrix, one row per candidate/criterion."""
    constraints = constraints or {}
    modulus_min = float(constraints.get("modulus_min_log10", 1.40))
    modulus_max = float(constraints.get("modulus_max_log10", 2.00))
    ra_max = float(constraints.get("ra_max_um", 2.00))
    ca_max = float(constraints.get("ca_max_deg", 90.0))

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        pred = predictions.loc[candidate.candidate_id]
        interval_note = (
            f"90% conformal interval [{pred['prediction_lower']:.3f}, {pred['prediction_upper']:.3f}] "
            "log10(GPa); computed DFT shear modulus, not experimental and not a clinical stiffness target."
        )
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "candidate_label": candidate.label,
                "criterion": "predicted_log_gvrh",
                "value": float(pred["predicted_log_gvrh"]),
                "units": "log10(GPa)",
                "source": "artifacts/model_log_gvrh.joblib; OOF-calibrated conformal interval",
                "evidence_level": "benchmark_predicted",
                "applicability": str(pred["ad_status"]),
                "constraint_met": _classify(float(pred["predicted_log_gvrh"]), modulus_min, modulus_max),
                "notes": interval_note,
                "critical": True,
            }
        )

        evidence = _evidence_rows(clean, candidate.evidence)
        for endpoint, low, high, critical in [
            ("surface_roughness_ra_um", 0.0, ra_max, True),
            ("water_contact_angle_deg", 0.0, ca_max, False),
        ]:
            hit = evidence[evidence["endpoint_key"] == endpoint]
            if hit.empty:
                rows.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "candidate_label": candidate.label,
                        "criterion": endpoint,
                        "value": np.nan,
                        "units": "um" if endpoint.startswith("surface") else "degrees",
                        "source": "no in-scope observation",
                        "evidence_level": "unknown",
                        "applicability": "NOT_APPLICABLE",
                        "constraint_met": "NA",
                        "notes": candidate.note or "Not reported for this material/treatment in the curated evidence.",
                        "critical": critical,
                    }
                )
                continue
            for _, ev in hit.iterrows():
                rows.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "candidate_label": candidate.label,
                        "criterion": endpoint,
                        "value": float(ev["value"]),
                        "units": str(ev["unit"]),
                        "source": f"doi:{ev['doi']} ({ev.get('pmcid', 'n/a')}), {ev['extraction_location']}",
                        "evidence_level": "observed_literature",
                        "applicability": "NOT_APPLICABLE",
                        "constraint_met": _classify(float(ev["value"]), low, high),
                        "notes": (
                            f"{ev['endpoint_label']}; conditions={ev['conditions_key']}; "
                            f"method={ev.get('measurement_method', 'not reported')}. "
                            "Observed value, not a prediction."
                        ),
                        "critical": critical,
                    }
                )

    matrix = pd.DataFrame(rows)
    matrix["critical"] = matrix["critical"].astype(bool)
    assert_criterion_matrix(matrix)
    assert_evidence_levels(matrix)
    return matrix


def summarise_candidates(matrix: pd.DataFrame) -> pd.DataFrame:
    """One row per candidate with transparent counts, not a percentage score."""
    records = []
    for candidate_id, group in matrix.groupby("candidate_id", sort=False):
        critical = group[group["critical"]]
        unknown = critical[critical["constraint_met"] == "NA"]
        failed = critical[critical["constraint_met"] == "NO"]
        met = group[group["constraint_met"] == "YES"]

        if len(unknown):
            status = "INDETERMINATE"
        elif len(failed):
            status = "CONSTRAINT_FAILED"
        else:
            status = "RANKED"

        records.append(
            {
                "candidate_id": candidate_id,
                "candidate_label": group["candidate_label"].iloc[0],
                "status": status,
                "criteria_met": int(len(met)),
                "criteria_failed": int(len(failed)),
                "criteria_unknown": int(len(unknown)),
                "indeterminate_reasons": "; ".join(
                    f"{row.criterion} unknown" for row in unknown.itertuples()
                ),
                "evidence_completeness": f"{len(group) - len(group[group['evidence_level'] == 'unknown'])} of {len(group)} criteria reported",
            }
        )
    summary = pd.DataFrame(records)
    assert_no_summary_score(summary)
    return summary


def assert_no_summary_score(summary: pd.DataFrame) -> None:
    from src.screening.guardrails import assert_no_composite_columns

    assert_no_composite_columns(summary.columns)
    for column in ("criteria_met", "criteria_failed", "criteria_unknown"):
        if column not in summary.columns:
            raise ValueError(f"Missing transparency count: {column}")


def rank_candidates(
    matrix: pd.DataFrame,
    summary: pd.DataFrame,
    order: Iterable[str] = ("predicted_log_gvrh", "surface_roughness_ra_um"),
    ascending: Iterable[bool] = (True, True),
) -> pd.DataFrame:
    """Rank only RANKED candidates by a lexicographic criterion order.

    The result is a view with a ``rank`` column. No column is a composite
    score; ties remain ties.
    """
    ranked_ids = summary.loc[summary["status"] == "RANKED", "candidate_id"].tolist()
    if not ranked_ids:
        return summary.iloc[0:0].assign(rank=pd.Series(dtype=int))

    wide = (
        matrix[matrix["candidate_id"].isin(ranked_ids)]
        .pivot_table(index="candidate_id", columns="criterion", values="value", aggfunc="first")
        .reindex(ranked_ids)
    )
    order = list(order)
    ascending = list(ascending)
    for criterion in order:
        if criterion not in wide.columns:
            raise ValueError(f"Unknown ranking criterion: {criterion}")
    ordered = wide.sort_values(by=order, ascending=ascending, kind="mergesort").reset_index()
    ordered["rank"] = np.arange(1, len(ordered) + 1)
    ordered = ordered.merge(
        summary[["candidate_id", "candidate_label", "status"]], on="candidate_id", how="left"
    )
    return ordered[["rank", "candidate_id", "candidate_label", "status", *order]]


def sensitivity_analysis(
    matrix: pd.DataFrame,
    summary: pd.DataFrame,
    orders: list[tuple[list[str], list[bool]]] | None = None,
    constraints: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Check whether the top candidate flips under priority/threshold changes."""
    orders = orders or [
        (["predicted_log_gvrh", "surface_roughness_ra_um"], [True, True]),
        (["surface_roughness_ra_um", "predicted_log_gvrh"], [True, True]),
    ]
    results = []
    for order, ascending in orders:
        ranked = rank_candidates(matrix, summary, order=order, ascending=ascending)
        results.append(
            {
                "order": order,
                "ascending": ascending,
                "n_ranked": int(len(ranked)),
                "top_candidate": ranked.iloc[0]["candidate_id"] if len(ranked) else None,
                "top_label": ranked.iloc[0]["candidate_label"] if len(ranked) else None,
                "ranking": ranked.to_dict(orient="records"),
            }
        )

    tops = {r["top_candidate"] for r in results if r["top_candidate"]}
    return {
        "orders_tested": len(results),
        "top_candidates": sorted(tops),
        "top_flips": len(tops) > 1,
        "results": results,
        "interpretation": (
            "Rank flips under a different criterion priority show that no single "
            "'best' material is supported by the evidence. Flips are a feature of "
            "honest screening, not a bug."
        ),
    }
