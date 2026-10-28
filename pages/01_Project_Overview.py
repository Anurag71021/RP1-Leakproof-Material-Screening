"""Page 1: project overview, research questions and hypothesis status."""

from __future__ import annotations

import streamlit as st

from src.config import DISCLAIMER, PATHS
from src.ui import (
    artefact_caption,
    disclaimer,
    gate_badge,
    load_json,
    page_header,
    read_text,
)

st.set_page_config(page_title="01 Overview | RP-1", layout="wide")


def _hypothesis_rows(metrics: dict, yrand: dict, surface_gate: dict) -> list[dict]:
    rows = []
    if metrics:
        dummy = metrics["models"]["dummy"]["summary"]["mae"]["mean"]
        ridge = metrics["models"]["ridge"]["summary"]["mae"]["mean"]
        rf = metrics["models"]["random_forest"]["summary"]["mae"]["mean"]
        rows.append(
            {
                "Hypothesis": "H1: Ridge beats the mean baseline",
                "Evidence": f"Ridge MAE {ridge:.4f} vs Dummy {dummy:.4f} (log10 GPa)",
                "Verdict": "supported" if ridge < dummy else "not supported",
            }
        )
        gap = ridge - rf
        rows.append(
            {
                "Hypothesis": "H2: Random Forest does not beat Ridge by a large margin",
                "Evidence": f"RF MAE {rf:.4f} vs Ridge {ridge:.4f}; absolute gap {gap:.4f}",
                "Verdict": (
                    "not supported (RF clearly better)"
                    if gap > 0.03
                    else "supported"
                ),
            }
        )
    if yrand and metrics:
        model = load_json(PATHS["metrics"] / "best_model.json", {}).get("model", "random_forest")
        agg = yrand["models"][model]["aggregate"]
        rows.append(
            {
                "Hypothesis": "H3: Y-randomized models collapse (R² ≤ 0.10)",
                "Evidence": f"{model}: randomized R² max {agg['r2_max']:.4f}, MAE {agg['mae_mean']:.4f}",
                "Verdict": "supported" if agg["r2_max"] <= 0.10 else "not supported",
            }
        )
    if surface_gate:
        gate = surface_gate["gate"]
        rows.append(
            {
                "Hypothesis": "H4: Surface evidence is insufficient for a model",
                "Evidence": f"{gate['observed']['studies']} studies, {gate['observed']['rows']} rows, "
                f"top study share {gate['observed']['top_study_share']:.0%} → {gate['decision']}",
                "Verdict": "supported" if gate["decision"] == "CURATION_ONLY" else "not supported",
            }
        )
    return rows


def main() -> None:
    page_header(
        "01 · Project overview",
        "Leak-proof benchmark, uncertainty, curated surface evidence and transparent screening",
    )
    disclaimer()
    gate_badge()

    st.subheader("Research questions")
    st.markdown(
        """
| ID | Question | Where answered |
| --- | --- | --- |
| RQ1 | How accurately can composition-only descriptors predict `log10(G_VRH)` under official Matbench folds, vs a mean baseline? | Page 03 |
| RQ2 | Does the pipeline collapse under Y-randomization, and is all preprocessing fitted inside the training fold? | Page 05 |
| RQ3 | Are conformal intervals calibrated, and how much of the data is inside the applicability domain? | Pages 04-05 |
| RQ4 | Is public surface evidence for metallic orthopaedic implants sufficient for a study-aware model? | Pages 06-07 |
"""
    )

    st.subheader("Hypothesis status (computed from persisted artefacts)")
    metrics = load_json(PATHS["metrics"] / "benchmark_metrics.json", {})
    yrand = load_json(PATHS["validation"] / "y_randomization.json", {})
    surface_gate = load_json(PATHS["validation"] / "surface_gate.json", {})
    rows = _hypothesis_rows(metrics, yrand, surface_gate)
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("Run the pipeline to populate hypothesis status. No values are hardcoded.")
    artefact_caption(
        PATHS["metrics"] / "benchmark_metrics.json",
        PATHS["validation"] / "y_randomization.json",
        PATHS["validation"] / "surface_gate.json",
    )

    st.subheader("What is computed vs what is measured")
    left, right = st.columns(2)
    with left:
        st.markdown(
            """
**Computed (benchmark layer)**

- `log10(G_VRH)` from DFT elastic tensors in Materials Project-derived data.
- 132 Magpie composition descriptors.
- Random Forest / Ridge / Dummy predictions with OOF metrics.
- Conformal intervals and KNN applicability domain.

This layer answers: *how well can composition predict a computed elastic
descriptor?* It does not measure any real specimen.
"""
        )
    with right:
        st.markdown(
            """
**Measured (evidence layer)**

- Ra, water contact angle and Ecorr read from open-access papers.
- Each value has a DOI, PMCID, table/section, conditions and method.
- Values are laboratory-specific and are never pooled across incompatible
  protocols.
- `INDETERMINATE` is shown wherever no comparable measurement exists.

This layer answers: *what has actually been reported for this material and
surface state?*
"""
        )

    st.subheader("Read the protocol before interpreting results")
    protocol = read_text(PATHS["docs"] / "RESEARCH_PROTOCOL.md")
    if protocol:
        with st.expander("Open the locked research protocol"):
            st.markdown(protocol)
    else:
        st.warning("`docs/RESEARCH_PROTOCOL.md` is missing.")
    st.caption(DISCLAIMER)


main()
