"""RP-1 research demonstrator - entry point.

Run with ``streamlit run app.py``. Importing this module has no side effects
beyond defining functions, so ``python -c "import app"`` is a valid smoke test.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import DISCLAIMER, PATHS
from src.ui import (
    artefact_caption,
    disclaimer,
    gate_status,
    load_json,
    page_header,
    read_text,
)


def _phase_table() -> pd.DataFrame:
    """Derive phase completion from the artefacts that prove each phase."""
    gate_passed, gate_state = gate_status()
    best = load_json(PATHS["metrics"] / "best_model.json", {})
    best_name = best.get("model")
    uq_proof = (
        (PATHS["uncertainty"] / f"conformal_{best_name}.json").exists()
        if best_name
        else False
    )
    checks = {
        "0. Repository audit": (PATHS["docs"] / "REPOSITORY_AUDIT.md").exists(),
        "1. Research protocol": (PATHS["docs"] / "RESEARCH_PROTOCOL.md").exists(),
        "2. Data ingestion + folds": (PATHS["splits"] / "split_manifest.json").exists(),
        "3. Benchmark models": (PATHS["metrics"] / "benchmark_metrics.json").exists(),
        "4. Leakage gate": gate_passed,
        "5. Interpretability + uncertainty": uq_proof,
        "6. Surface evidence curation": (PATHS["data_literature"] / "surface_evidence_clean.csv").exists(),
        "7. Surface model decision": (PATHS["validation"] / "surface_model_decision.json").exists(),
        "8. Screening engine": (PATHS["screening"] / "criterion_matrix.csv").exists(),
        "9. Streamlit app": True,
    }
    return pd.DataFrame(
        {
            "Phase": list(checks),
            "Status": ["complete" if ok else "pending" for ok in checks.values()],
        }
    )


def main() -> None:
    st.set_page_config(
        page_title="RP-1 | Material & Surface Screening",
        page_icon="🔬",
        layout="wide",
    )
    page_header(
        "RP-1: leak-proof material and surface screening",
        "Research demonstrator for metallic orthopaedic implant materials - decision support only",
    )
    disclaimer()

    passed, state = gate_status()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Leakage gate", state)
    metrics = load_json(PATHS["metrics"] / "benchmark_metrics.json", {})
    if metrics:
        best = load_json(PATHS["metrics"] / "best_model.json", {})
        name = best.get("model", "n/a")
        summary = metrics.get("models", {}).get(name, {}).get("summary", {})
        col2.metric("Selected model", name)
        col3.metric("OOF MAE (log10 GPa)", f"{summary.get('mae', {}).get('mean', float('nan')):.4f}")
        col4.metric("OOF R²", f"{summary.get('r2', {}).get('mean', float('nan')):.4f}")
    else:
        col2.metric("Selected model", "pending")
        col3.metric("OOF MAE", "pending")
        col4.metric("OOF R²", "pending")

    st.divider()
    left, right = st.columns([3, 2])

    with left:
        st.subheader("What this project does")
        st.markdown(
            """
1. Predicts a **computed** DFT elastic descriptor, `log10(G_VRH)`, from chemical
   composition under the official Matbench 5-fold protocol.
2. Quantifies uncertainty and descriptor-space support for every prediction.
3. Curates published **measured** surface evidence for metallic orthopaedic
   implant materials with DOI-level provenance.
4. Screens material + surface-state candidates against explicit user constraints
   and reports criterion-level evidence, including `INDETERMINATE` when evidence
   is missing.

It does **not** predict implant success, biocompatibility, infection risk,
fatigue life, or any clinical outcome.
"""
        )
        st.subheader("Read this first")
        st.info(
            "`log10(G_VRH)` is a computed shear modulus from DFT elastic tensors. "
            "It is not an experimental modulus, not a design allowable, and not a "
            "measure of clinical performance."
        )

    with right:
        st.subheader("Phase status (artefact-derived)")
        st.dataframe(_phase_table(), use_container_width=True, hide_index=True)
        artefact_caption(PATHS["docs"] / "IMPLEMENTATION_PLAN.md", PATHS["docs"] / "LEAKAGE_AUDIT.md")
        if passed:
            st.success("Gate evidence: `docs/LEAKAGE_AUDIT.md`")
        else:
            st.warning(
                "Gate is not passed. Downstream pages render stored artefacts but must "
                "not be interpreted as validated results."
            )

    st.divider()
    st.subheader("Navigation")
    st.markdown(
        """
Use the sidebar pages:

| Page | Content |
| --- | --- |
| 02 Dataset provenance & quality | Official folds, checksums, duplicate audit |
| 03 Benchmark results | Per-fold and mean ± std metrics vs Dummy |
| 04 Prediction, AD & uncertainty | Interactive composition prediction with support status |
| 05 Interpretability & leakage | Permutation importance, Y-randomization, residuals |
| 06 Surface evidence | Curated measured evidence, missingness, curation-only decision |
| 07 Candidate screening | Criterion matrix, constraints, INDETERMINATE list |
| 08 Limitations & disclaimers | Scope, non-use, and known gaps |
"""
    )
    st.caption(DISCLAIMER)


if __name__ == "__main__":
    main()
