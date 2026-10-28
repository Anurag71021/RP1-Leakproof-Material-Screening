"""Page 6: curated surface evidence, provenance and the curation-only decision."""

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
    read_text,
    show_dataframe,
)

st.set_page_config(page_title="06 Surface Evidence | RP-1", layout="wide")


def main() -> None:
    page_header(
        "06 · Surface evidence layer",
        "Measured values from open-access studies, with provenance and explicit gaps",
    )
    disclaimer()

    gate_payload = load_json(PATHS["validation"] / "surface_gate.json")
    clean = load_csv(PATHS["data_literature"] / "surface_evidence_clean.csv")
    if not gate_payload or clean is None:
        missing(PATHS["data_literature"] / "surface_evidence_clean.csv", "python scripts/build_literature.py")
        return
    gate = gate_payload["gate"]

    if gate["decision"] == "CURATION_ONLY":
        st.warning(
            "**CURATION-ONLY.** The Phase 7 modelling gate failed, so no surface-response "
            "model was trained. This is the honest outcome, not a hidden failure."
        )
    else:
        st.success(f"Modelling gate decision: `{gate['decision']}`")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Clean rows", f"{gate_payload['n_clean']}")
    col2.metric("Distinct studies", f"{gate_payload['n_distinct_studies']}")
    col3.metric("Locked-endpoint rows", f"{gate_payload['locked_endpoint_rows']}")
    col4.metric("Top study share", f"{gate_payload['locked_endpoint_top_study_share']:.0%}")

    st.subheader("Why curation-only")
    st.markdown(
        "\n".join(f"- {reason}" for reason in gate["reasons"])
        or "- All gate conditions were met; a model would be trained by `scripts/train_surface_model.py`."
    )
    st.markdown(
        f"""
Gate thresholds: at least {gate['thresholds']['min_studies']} studies,
{gate['thresholds']['min_rows']} deduplicated rows for one locked endpoint, and no
study contributing more than {gate['thresholds']['max_single_study_share']:.0%} of rows.

Observed: {gate['observed']['studies']} studies, {gate['observed']['rows']} rows,
largest share {gate['observed']['top_study_share']:.1%}.
"""
    )

    st.subheader("Endpoints and units")
    endpoints = pd.DataFrame(
        [
            {
                "endpoint_key": endpoint,
                "rows": count,
                "harmonised units": ", ".join(gate_payload["units_by_endpoint"].get(endpoint, [])),
            }
            for endpoint, count in gate_payload["endpoint_counts"].items()
        ]
    )
    st.dataframe(endpoints, use_container_width=True, hide_index=True)
    st.info(
        "Ra and Rz/Sa are different quantities and are never merged. Water contact angle "
        "is not a biological outcome. Saline and water contact angles are not pooled. "
        "Potentials are converted only when the reference electrode is stated."
    )

    st.subheader("Rows per study")
    per_study = pd.DataFrame(
        [
            {"study_id": study, "rows": count}
            for study, count in gate_payload["rows_per_study"]["by_study"].items()
        ]
    )
    st.dataframe(per_study, use_container_width=True, hide_index=True)

    st.subheader("Measured roughness (Ra)")
    ra = clean[clean["endpoint_key"] == "surface_roughness_ra_um"]
    if len(ra):
        st.scatter_chart(
            ra,
            x="study_id",
            y="value",
            color="material_grade_or_formula",
            height=360,
        )
        st.caption(
            "Each point is a reported measurement under specific conditions. Values from "
            "different studies are not interchangeable and are not pooled into a mean here."
        )
    else:
        st.info("No Ra rows curated.")

    st.subheader("Measured water contact angle")
    ca = clean[clean["endpoint_key"] == "water_contact_angle_deg"]
    if len(ca):
        st.scatter_chart(ca, x="study_id", y="value", color="material_grade_or_formula", height=320)
        st.caption(
            "Contact angle is a physical surface descriptor. It is not evidence of "
            "biocompatibility, bacterial adhesion resistance, or clinical success."
        )

    st.subheader("Evidence table (long format)")
    display_columns = [
        "study_id",
        "material_grade_or_formula",
        "surface_treatment",
        "endpoint_key",
        "value",
        "unit",
        "conditions_key",
        "extraction_location",
        "measurement_method",
        "evidence_level",
    ]
    available = [c for c in display_columns if c in clean.columns]
    show_dataframe(clean[available], empty="No curated rows.")
    csv_download(clean, "surface_evidence_clean.csv", "Download curated evidence CSV")

    st.subheader("Missingness — not reported, never imputed")
    missing_counts = pd.DataFrame(
        [
            {"Field": field, "Rows missing": count}
            for field, count in gate_payload["missing_counts"].items()
        ]
    )
    st.dataframe(missing_counts, use_container_width=True, hide_index=True)

    st.subheader("Deduplication")
    dedup = gate_payload["dedup_log"]
    st.markdown(
        f"Key: `({' | '.join(dedup['key'])})`. "
        f"{dedup['n_duplicates_removed']} duplicate row(s) removed from {dedup['n_input']} input rows."
    )

    st.subheader("Exclusions")
    excluded = pd.DataFrame(gate_payload["excluded_studies"])
    if len(excluded):
        st.dataframe(excluded, use_container_width=True, hide_index=True)
    st.caption(
        "Polystyrene control rows from PMC7666130 are excluded because polystyrene is "
        "not a metal implant material. Figure-only values were not digitised."
    )

    st.subheader("Audit and data card")
    audit = read_text(PATHS["docs"] / "SURFACE_EVIDENCE_AUDIT.md")
    if audit:
        with st.expander("Open the full surface evidence audit"):
            st.markdown(audit)
    card = read_text(PATHS["data_cards"] / "surface_evidence_data_card.md")
    if card:
        with st.expander("Open the surface evidence data card"):
            st.markdown(card)

    artefact_caption(
        PATHS["data_literature"] / "surface_evidence_clean.csv",
        PATHS["validation"] / "surface_gate.json",
        PATHS["docs"] / "SURFACE_EVIDENCE_AUDIT.md",
    )
    st.caption(DISCLAIMER)


main()
