"""Page 8: limitations, non-use and disclaimers."""

from __future__ import annotations

import streamlit as st

from src.config import DISCLAIMER, PATHS
from src.ui import (
    artefact_caption,
    disclaimer,
    gate_badge,
    read_text,
)

st.set_page_config(page_title="08 Limitations | RP-1", layout="wide")


def main() -> None:
    page_header(
        "08 · Limitations & disclaimers",
        "Read this before quoting any number from this demonstrator",
    )
    disclaimer()
    gate_badge()

    st.subheader("Three distinctions that must survive every summary")
    st.markdown(
        """
| Distinction | What this project does | What it must never imply |
| --- | --- | --- |
| **Computed vs experimental** | Predicts `log10(G_VRH)`, a DFT-derived shear modulus | That a candidate has a measured modulus or a validated design allowable |
| **Association vs causation** | Reports permutation importance for one fitted model | That a descriptor causes stiffness, biocompatibility or clinical success |
| **Screening vs clinical validation** | Ranks research hypotheses under user constraints | That any candidate is safe, effective, approved, or recommended for patients |
"""
    )

    st.subheader("Hard non-use statement")
    st.error(
        "This software is a research demonstrator. It is not a medical device, not a "
        "clinical decision-support tool, and not a substitute for materials testing, "
        "biocompatibility assessment, regulatory review or clinical judgement. Do not use "
        "it to select an implant for a patient."
    )

    st.subheader("Known limitations")
    text = read_text(PATHS["docs"] / "LIMITATIONS.md")
    if text:
        st.markdown(text)
    else:
        st.warning("`docs/LIMITATIONS.md` is missing. Run the documentation phase.")
        st.markdown(
            """
1. The target is computed DFT shear modulus, not an experimental implant property.
2. Composition-only descriptors ignore processing, microstructure, texture, defects,
   porosity and surface state.
3. Official folds measure interpolation within a broad inorganic dataset, not
   extrapolation to new alloy families.
4. Surface evidence is a small purposive sample with laboratory-specific protocols.
5. No surface model was trained; the gate failed on study count, row count and
   single-study dominance.
6. `INDETERMINATE` reflects missing comparable evidence, not a material's quality.
"""
        )

    st.subheader("Interpretability limits")
    interp = read_text(PATHS["docs"] / "INTERPRETABILITY_LIMITS.md")
    if interp:
        with st.expander("Open `docs/INTERPRETABILITY_LIMITS.md`"):
            st.markdown(interp)

    st.subheader("Research gaps")
    gaps = read_text(PATHS["docs"] / "RESEARCH_GAPS.md")
    if gaps:
        with st.expander("Open `docs/RESEARCH_GAPS.md`"):
            st.markdown(gaps)

    st.subheader("Project log and reproducibility")
    st.markdown(
        """
Exact commands are listed in `docs/IMPLEMENTATION_PLAN.md` and `README.md`.
Every displayed value is read from a file under `results/`, `reports/`, `docs/` or
`data_cards/`. The test suite checks that the OOF files reproduce the reported
metrics and that the leakage gate is signed.
"""
    )
    artefact_caption(
        PATHS["docs"] / "LIMITATIONS.md",
        PATHS["docs"] / "LEAKAGE_AUDIT.md",
        PATHS["docs"] / "PROJECT_LOG.md",
    )
    st.caption(DISCLAIMER)


main()
