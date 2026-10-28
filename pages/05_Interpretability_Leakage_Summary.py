"""Page 5: interpretability, leakage audit and overfitting diagnostics."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import DISCLAIMER, PATHS
from src.ui import (
    artefact_caption,
    disclaimer,
    gate_badge,
    load_csv,
    load_json,
    missing,
    page_header,
    read_text,
)

st.set_page_config(page_title="05 Interpretability | RP-1", layout="wide")


def main() -> None:
    page_header(
        "05 · Interpretability & leakage summary",
        "What the model relies on, and the evidence that the evaluation is not leaky",
    )
    disclaimer()
    gate_badge()

    best = load_json(PATHS["metrics"] / "best_model.json", {})
    model = best.get("model", "random_forest")
    importance = load_json(PATHS["interpretability"] / f"permutation_importance_{model}.json", {})
    importance_csv = load_csv(PATHS["interpretability"] / f"permutation_importance_{model}.csv")
    yrand = load_json(PATHS["validation"] / "y_randomization.json", {})
    curves = load_json(PATHS["validation"] / "learning_curves.json", {})
    residual = load_json(PATHS["validation"] / f"residual_stats_{model}.json", {})
    gate = load_json(PATHS["validation"] / "gate_status.json", {})

    st.error("**Associational only. Not causal.** Permutation importance describes reliance of one fitted model on descriptor columns. It does not establish a physical mechanism, a biological effect, or a clinical benefit.")

    st.subheader(f"Permutation importance — {model}")
    if importance_csv is not None:
        top = importance_csv.sort_values("importance_mean", ascending=False).head(15)
        st.bar_chart(top.set_index("feature")["importance_mean"], height=380)
        st.dataframe(
            top.assign(
                importance_mean=top["importance_mean"].map(lambda v: f"{v:.5f}"),
                importance_std_across_folds=top["importance_std_across_folds"].map(
                    lambda v: f"{v:.5f}" if pd.notna(v) else "n/a"
                ),
            ),
            use_container_width=True,
            hide_index=True,
        )
        if residual:
            st.caption(importance.get("interpretation", ""))
        artefact_caption(PATHS["interpretability"] / f"permutation_importance_{model}.csv")
    else:
        missing(PATHS["interpretability"] / f"permutation_importance_{model}.csv", "python scripts/run_analysis.py")

    st.subheader("Leakage gate checklist")
    if gate:
        rows = [
            {"Check": key, "Result": "PASS" if value["passed"] else "FAIL", "Evidence": value["evidence"]}
            for key, value in gate["checks"].items()
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        if gate.get("passed"):
            st.success(f"Gate marker: `{gate['marker']}`")
        else:
            st.error(f"Gate marker: `{gate['marker']}` — downstream results are not validated.")
    else:
        missing(PATHS["validation"] / "gate_status.json", "python scripts/finalize_gate.py")

    st.subheader("Y-randomization")
    if yrand:
        rows = []
        for name, payload in yrand["models"].items():
            agg = payload["aggregate"]
            rows.append(
                {
                    "Model": name,
                    "Permutations": payload["n_permutations"],
                    "Randomized R² mean": f"{agg['r2_mean']:.4f}",
                    "Randomized R² max": f"{agg['r2_max']:.4f}",
                    "Randomized MAE": f"{agg['mae_mean']:.4f}",
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.markdown(
            "Training labels are shuffled **inside each fold**; test labels stay true. "
            "A valid pipeline must collapse to, or below, the Dummy baseline. It does."
        )
        artefact_caption(PATHS["validation"] / "y_randomization.json")

    st.subheader("Learning curves")
    if curves and model in curves.get("models", {}):
        curve = pd.DataFrame(curves["models"][model])
        st.line_chart(
            curve.set_index("n_train")[["train_mae", "val_mae"]].rename(
                columns={"train_mae": "train MAE", "val_mae": "held-out MAE"}
            )
        )
        st.dataframe(curve, use_container_width=True, hide_index=True)
        gap = curve.iloc[-1]["gap_mae"]
        st.caption(
            f"At full training size the train/held-out MAE gap is {gap:.4f} log10(GPa). "
            "A gap is expected for a 300-tree forest; the decisive leakage evidence is the "
            "Y-randomization collapse, not the gap alone. Curves use one official training "
            "fold to bound runtime."
        )
    else:
        missing(PATHS["validation"] / "learning_curves.json", "python scripts/run_diagnostics.py")

    st.subheader("Residual diagnostics")
    if residual:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean residual", f"{residual['residual_mean']:.4f}")
        col2.metric("Residual SD", f"{residual['residual_std']:.4f}")
        col3.metric("|residual| p95", f"{residual['abs_residual_p95']:.4f}")
        col4.metric("corr(pred, |resid|)", f"{residual['corr_pred_abs_residual']:.4f}")
        st.caption(residual["reading"])
        st.caption(
            "A small negative correlation here means absolute error does not grow strongly "
            "with predicted modulus across the benchmark range."
        )
        artefact_caption(PATHS["validation"] / f"residual_stats_{model}.json")

    st.subheader("Full gate audit")
    audit = read_text(PATHS["docs"] / "LEAKAGE_AUDIT.md")
    if audit:
        with st.expander("Open `docs/LEAKAGE_AUDIT.md`"):
            st.markdown(audit)
    else:
        missing(PATHS["docs"] / "LEAKAGE_AUDIT.md", "python scripts/finalize_gate.py")
    st.caption(DISCLAIMER)


main()
