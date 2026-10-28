"""Page 4: single-composition prediction with support and uncertainty."""

from __future__ import annotations

import joblib
import pandas as pd
import streamlit as st

from src.config import DISCLAIMER, PATHS
from src.ui import (
    artefact_caption,
    disclaimer,
    load_json,
    missing,
    page_header,
)

st.set_page_config(page_title="04 Prediction | RP-1", layout="wide")

EXAMPLES = {
    "Ti-6Al-4V": "Ti-6Al-4V",
    "cp-Ti (grade 4)": "Ti",
    "316L stainless steel": "Fe-17Cr-12Ni-2.5Mo-2Mn-1Si",
    "Co-28Cr-6Mo": "Co-28Cr-6Mo",
    "Ti-15Mo-5Zr-3Al": "Ti-15Mo-5Zr-3Al",
    "Mg AZ31": "Mg-3Al-1Zn",
}


@st.cache_resource(show_spinner=False)
def _load_bundle(path: str):
    return joblib.load(path)


@st.cache_resource(show_spinner=False)
def _load_ad(path: str):
    from src.evaluation.applicability import ApplicabilityDomain

    return ApplicabilityDomain.load(path)


def main() -> None:
    page_header(
        "04 · Prediction, applicability domain & uncertainty",
        "Interactive composition-only prediction — research use only",
    )
    disclaimer()

    bundle_path = PATHS["artifacts"] / "model_log_gvrh.joblib"
    ad_path = PATHS["artifacts"] / "ad_reference.joblib"
    if not bundle_path.exists():
        missing(bundle_path, "python scripts/build_screening.py")
        return
    if not ad_path.exists():
        missing(ad_path, "python scripts/run_analysis.py")
        return

    bundle = _load_bundle(str(bundle_path))
    ad = _load_ad(str(ad_path))
    metrics = load_json(PATHS["metrics"] / "benchmark_metrics.json", {})
    best = load_json(PATHS["metrics"] / "best_model.json", {})
    uq = load_json(PATHS["uncertainty"] / f"conformal_{best.get('model')}.json", {})

    st.markdown(
        f"""
**Model:** `{bundle['model_name']}` fitted on {bundle['n_train']:,} labelled rows
({len(bundle['feature_names'])} composition descriptors).
**Target:** `{bundle['target']}` in `{bundle['target_units']}` — computed DFT shear modulus.
**Fitted (UTC):** `{bundle['trained_utc']}`.

Reported benchmark performance comes from official-fold OOF predictions, not from
this deployment object: MAE
{metrics.get('models', {}).get(bundle['model_name'], {}).get('summary', {}).get('mae', {}).get('mean', float('nan')):.4f},
R² {metrics.get('models', {}).get(bundle['model_name'], {}).get('summary', {}).get('r2', {}).get('mean', float('nan')):.4f}.
"""
    )
    st.warning(bundle["caveat"])

    st.subheader("Enter a composition")
    preset = st.selectbox("Example compositions", list(EXAMPLES), index=0)
    formula = st.text_input(
        "Chemical formula or alloy notation (element-weight fractions)",
        value=EXAMPLES[preset],
        help="Examples: Ti-6Al-4V, Fe-17Cr-12Ni-2.5Mo, Ti. Parsed with pymatgen Composition.",
    )
    predict = st.button("Predict", type="primary")

    if predict:
        try:
            from pymatgen.core import Composition

            composition = Composition(formula)
        except Exception as exc:  # user input can be invalid
            st.error(f"Could not parse composition: {exc}")
            return
        if not composition.elements:
            st.error("Composition has no elements.")
            return

        frame = pd.DataFrame({"structure": [composition]})
        prediction = float(bundle["pipeline"].predict(frame)[0])
        ad_score = ad.score(frame)
        status = str(ad_score["ad_status"].iloc[0])
        distance = float(ad_score["knn_distance"].iloc[0])

        st.subheader("Result")
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted log10(G_VRH)", f"{prediction:.3f}")
        col2.metric("Equivalent G_VRH", f"{10 ** prediction:,.1f} GPa")
        col3.metric("Applicability domain", status)

        st.markdown(
            f"""
- Reduced formula: `{composition.reduced_formula}`
- Elements: {', '.join(str(el) for el in composition.elements)}
- KNN distance: **{distance:.4f}** (training thresholds q95 = {float(ad_score['ad_q95'].iloc[0]):.4f},
  q99.5 = {float(ad_score['ad_q995'].iloc[0]):.4f})
"""
        )

        if uq:
            half = float(uq["deployment_halfwidths"].get("0.9", float("nan")))
            st.markdown(
                f"**90% split-conformal interval:** "
                f"[{prediction - half:.3f}, {prediction + half:.3f}] log10(GPa) "
                f"(half-width {half:.3f}). "
                f"Empirical coverage on held-out folds: "
                f"{next((r['empirical_coverage'] for r in uq['coverage'] if abs(r['nominal_coverage'] - 0.9) < 1e-9), float('nan')):.3f}."
            )
            st.caption(
                "The interval is calibrated on OOF residuals, not a Bayesian posterior. "
                "It describes average error magnitude, not a guarantee for this composition."
            )

        if status == "OUT_OF_DOMAIN":
            st.error(
                "This composition is outside the descriptor-space support of the training "
                "data. The prediction is shown for completeness but must not be used for "
                "screening or comparison. There is no evidence that the model generalises here."
            )
        elif status == "NEAR_BOUNDARY":
            st.warning(
                "This composition is near the applicability-domain boundary. Treat the "
                "prediction as weak support only, and prefer candidates with observed evidence."
            )
        else:
            st.info(
                "The composition is inside the descriptor-space applicability domain. "
                "This is support in feature space only; it is not evidence of implant suitability."
            )

        st.markdown(
            """
**What this number is not**

- Not a measured modulus of a real specimen.
- Not a design allowable, fatigue limit, or hardness.
- Not evidence of biocompatibility, corrosion resistance, osseointegration or
  clinical performance.
- Composition-only: it cannot represent surface treatment, porosity, texture,
  residual stress or coating state.
"""
        )

    st.divider()
    st.subheader("Calibration and support across the benchmark")
    if uq:
        coverage = pd.DataFrame(uq["coverage"])
        coverage["nominal_coverage"] = coverage["nominal_coverage"].map(lambda v: f"{v:.0%}")
        coverage["empirical_coverage"] = coverage["empirical_coverage"].map(lambda v: f"{v:.3f}")
        coverage["mean_interval_halfwidth"] = coverage["mean_interval_halfwidth"].map(lambda v: f"{v:.4f}")
        st.dataframe(
            coverage[["nominal_coverage", "empirical_coverage", "mean_interval_halfwidth"]],
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Coverage is evaluated leave-one-fold-out: each fold's quantile is calibrated "
            "on the other four folds. Deployment half-widths use all OOF residuals and are "
            "slightly optimistic."
        )
    else:
        missing(PATHS["uncertainty"] / f"conformal_{best.get('model')}.json", "python scripts/run_analysis.py")

    ad_payload = load_json(PATHS["validation"] / "applicability_domain.json", {})
    if ad_payload:
        overall = ad_payload["overall"]
        st.markdown(
            "**Applicability-domain counts (OOF, one fold-local reference per fold):** "
            + ", ".join(f"{k}: {v:,}" for k, v in overall.items())
        )
        st.caption(ad_payload["interpretation"])
        st.caption(f"Method: {ad_payload['method']}")

    artefact_caption(
        PATHS["artifacts"] / "model_log_gvrh.joblib",
        PATHS["artifacts"] / "ad_reference.joblib",
        PATHS["uncertainty"] / f"conformal_{best.get('model')}.json",
    )
    st.caption(DISCLAIMER)


main()
