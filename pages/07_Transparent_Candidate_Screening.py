"""Page 7: criterion-level candidate screening with explicit constraints."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import DISCLAIMER, PATHS
from src.ui import (
    artefact_caption,
    csv_download,
    disclaimer,
    load_csv,
    load_json,
    missing,
    page_header,
    show_dataframe,
)

st.set_page_config(page_title="07 Screening | RP-1", layout="wide")


def main() -> None:
    page_header(
        "07 · Transparent candidate screening",
        "Criterion-level evidence, explicit constraints and honest INDETERMINATE status",
    )
    disclaimer()

    summary_payload = load_json(PATHS["screening"] / "screening_summary.json")
    matrix = load_csv(PATHS["screening"] / "criterion_matrix.csv")
    summary = load_csv(PATHS["screening"] / "candidate_summary.csv")
    ranking = load_csv(PATHS["screening"] / "ranking.csv")
    sensitivity = load_json(PATHS["screening"] / "sensitivity.json", {})
    predictions = load_csv(PATHS["screening"] / "candidate_predictions.csv")
    if not summary_payload or matrix is None or summary is None:
        missing(PATHS["screening"] / "criterion_matrix.csv", "python scripts/build_screening.py")
        return

    st.error(
        "No composite suitability score exists in this project. Candidates are never "
        "ranked by a weighted average, and missing evidence is never treated as a low score. "
        "A `RANKED` label means *all critical criteria are known and meet the user's "
        "constraints*, not that the material is suitable for implantation."
    )

    st.subheader("Constraints used")
    constraints = summary_payload["constraints"]
    st.markdown(
        f"""
| Constraint | Value | Role |
| --- | --- | --- |
| Predicted log10(G_VRH) | {constraints['modulus_min_log10']} – {constraints['modulus_max_log10']} log10(GPa) | critical |
| Ra | ≤ {constraints['ra_max_um']} µm | critical |
| Water contact angle | ≤ {constraints['ca_max_deg']}° | non-critical (informs, does not gate) |

These are **research demonstration thresholds supplied by the user**, not clinical
acceptance limits. Change them only in `scripts/build_screening.py` and regenerate
the artefacts.
"""
    )
    st.caption(
        f"Prediction model `{summary_payload['model']}`; 90% conformal half-width "
        f"±{summary_payload['conformal_halfwidth']:.3f} log10(GPa). Target is computed DFT "
        "shear modulus, not an experimental implant modulus."
    )

    counts = summary["status"].value_counts()
    col1, col2, col3 = st.columns(3)
    col1.metric("RANKED", int(counts.get("RANKED", 0)))
    col2.metric("CONSTRAINT_FAILED", int(counts.get("CONSTRAINT_FAILED", 0)))
    col3.metric("INDETERMINATE", int(counts.get("INDETERMINATE", 0)))
    st.caption(
        "`INDETERMINATE` means a critical criterion has no comparable evidence. "
        "It is a statement about missing evidence, not about material quality."
    )

    st.subheader("Candidate status")
    show_dataframe(
        summary[
            [
                "candidate_label",
                "status",
                "criteria_met",
                "criteria_failed",
                "criteria_unknown",
                "indeterminate_reasons",
                "evidence_completeness",
            ]
        ]
    )
    csv_download(summary, "candidate_summary.csv", "Download candidate summary CSV")

    st.subheader("Indeterminate candidates")
    indeterminate = summary[summary["status"] == "INDETERMINATE"]
    if len(indeterminate):
        for row in indeterminate.itertuples():
            st.markdown(f"- **{row.candidate_label}** — {row.indeterminate_reasons}")
    else:
        st.info("No candidate is indeterminate under the current constraints.")

    st.subheader("Criterion-level evidence")
    candidates = st.multiselect(
        "Filter candidates",
        options=list(matrix["candidate_id"].unique()),
        default=list(matrix["candidate_id"].unique()),
    )
    criteria = st.multiselect(
        "Filter criteria",
        options=list(matrix["criterion"].unique()),
        default=list(matrix["criterion"].unique()),
    )
    filtered = matrix[matrix["candidate_id"].isin(candidates) & matrix["criterion"].isin(criteria)]
    st.dataframe(
        filtered[
            [
                "candidate_label",
                "criterion",
                "value",
                "units",
                "source",
                "evidence_level",
                "applicability",
                "constraint_met",
                "critical",
                "notes",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Every row carries its source: a DOI + table/section for measured values, or the "
        "deployment model artefact for predictions. `evidence_level` distinguishes "
        "`observed_literature` from `benchmark_predicted` and `unknown`."
    )
    csv_download(filtered, "criterion_matrix_filtered.csv", "Download filtered criterion matrix CSV")

    st.subheader("Ranking (fully supported candidates only)")
    if ranking is None or ranking.empty:
        st.info(
            "No candidate has complete evidence for every critical criterion under the "
            "current constraints, so no ranking is produced."
        )
    else:
        st.dataframe(ranking, use_container_width=True, hide_index=True)
        st.caption(
            "Lexicographic order: `predicted_log_gvrh` then `surface_roughness_ra_um`, "
            "both ascending. Ties remain ties. No weighted aggregate is computed."
        )

    st.subheader("Priority sensitivity")
    if sensitivity:
        st.markdown(
            f"- Priority orders tested: {sensitivity['orders_tested']}\n"
            f"- Top candidates across orders: {', '.join(sensitivity['top_candidates']) or 'none'}\n"
            f"- Top selection flips: **{sensitivity['top_flips']}**\n\n"
            f"{sensitivity['interpretation']}"
        )
        for result in sensitivity["results"]:
            with st.expander(f"Order: {' → '.join(result['order'])} (ascending={result['ascending']})"):
                st.dataframe(pd.DataFrame(result["ranking"]), use_container_width=True, hide_index=True)

    st.subheader("Predictions, intervals and applicability domain")
    if predictions is not None:
        st.dataframe(
            predictions[
                [
                    "candidate_label",
                    "alloy_family",
                    "predicted_log_gvrh",
                    "prediction_lower",
                    "prediction_upper",
                    "ad_status",
                    "knn_distance",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Predictions are composition-only. Surface state is represented separately in "
            "the criterion matrix and is never fed into the modulus model."
        )

    st.subheader("Mandatory caveats")
    for warning in summary_payload.get("warnings", []):
        st.markdown(f"- {warning}")
    st.caption(summary_payload.get("disclaimer", DISCLAIMER))
    artefact_caption(
        PATHS["screening"] / "criterion_matrix.csv",
        PATHS["screening"] / "screening_summary.json",
        PATHS["docs"] / "SURFACE_EVIDENCE_AUDIT.md",
    )
    st.caption(DISCLAIMER)


main()
