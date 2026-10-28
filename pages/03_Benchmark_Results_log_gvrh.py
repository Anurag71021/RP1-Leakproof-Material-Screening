"""Page 3: benchmark results under the official Matbench folds."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import DISCLAIMER, PATHS
from src.ui import (
    artefact_caption,
    csv_download,
    disclaimer,
    gate_badge,
    load_csv,
    load_json,
    missing,
    page_header,
)

st.set_page_config(page_title="03 Benchmark | RP-1", layout="wide")


def main() -> None:
    page_header(
        "03 · Benchmark results — `matbench_log_gvrh`",
        "Official 5-fold out-of-fold performance, composition-only descriptors",
    )
    disclaimer()
    gate_badge()

    metrics = load_json(PATHS["metrics"] / "benchmark_metrics.json")
    best = load_json(PATHS["metrics"] / "best_model.json")
    if not metrics or not best:
        missing(PATHS["metrics"] / "benchmark_metrics.json", "python scripts/run_benchmark.py")
        return

    st.info(
        "Target: `log10(G_VRH)` in log10(GPa) — a **computed DFT shear modulus**. "
        "Scores describe interpolation within a broad inorganic materials dataset, "
        "not performance on implant alloys or devices."
    )

    st.subheader("Model comparison (mean ± sample SD across 5 official folds)")
    rows = []
    for name, payload in metrics["models"].items():
        s = payload["summary"]
        rows.append(
            {
                "Model": name,
                "Features": payload["n_features"],
                "MAE": f"{s['mae']['mean']:.4f} ± {s['mae']['std']:.4f}",
                "RMSE": f"{s['rmse']['mean']:.4f} ± {s['rmse']['std']:.4f}",
                "R²": f"{s['r2']['mean']:.4f} ± {s['r2']['std']:.4f}",
                "OOF file": payload["oof_file"],
            }
        )
    table = pd.DataFrame(rows).sort_values("MAE")
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.success(
        f"Selected model: `{best['model']}` — selected by **lowest mean OOF MAE before "
        "any screening or literature analysis**."
    )
    st.caption(f"Selection rule: {best['selected_by']}")

    st.subheader(f"Per-fold metrics — {best['model']}")
    per_fold = pd.DataFrame(metrics["models"][best["model"]]["per_fold"])
    display = per_fold[["fold_id", "n_train", "n_test", "mae", "rmse", "r2"]].copy()
    for column in ("mae", "rmse", "r2"):
        display[column] = display[column].map(lambda v: f"{v:.4f}")
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.caption(
        "Each held-out fold is predicted only by the model trained on the other "
        "four folds. No row is predicted by a model that saw it."
    )

    oof = load_csv(PATHS["predictions"] / f"oof_{best['model']}_matbench_log_gvrh.csv")
    if oof is not None:
        st.subheader("Out-of-fold residuals")
        residuals = oof.assign(residual=oof["y_true"] - oof["y_pred"])
        st.bar_chart(
            residuals["residual"].rename("residual").reset_index(drop=True),
            height=220,
        )
        st.caption(
            f"Mean residual {residuals['residual'].mean():.4f}; "
            f"SD {residuals['residual'].std():.4f}; "
            f"95th percentile |residual| {residuals['residual'].abs().quantile(0.95):.4f} log10(GPa)."
        )
        csv_download(oof, f"oof_{best['model']}.csv", "Download OOF predictions CSV")

    st.subheader("Diagnostic plots")
    plot_names = [
        (f"parity_plot_{best['model']}.png", "Parity: predicted vs true"),
        (f"residuals_{best['model']}.png", "Residuals vs prediction"),
        (f"learning_curves_{best['model']}.png", "Learning curve (official fold 0)"),
        (f"permutation_importance_{best['model']}.png", "Permutation importance"),
    ]
    columns = st.columns(2)
    for index, (filename, caption) in enumerate(plot_names):
        path = PATHS["plots"] / filename
        with columns[index % 2]:
            if path.exists():
                st.image(str(path), caption=caption, use_container_width=True)
            else:
                st.warning(f"Missing plot: `{filename}`")

    st.subheader("What these numbers do and do not mean")
    st.markdown(
        f"""
- MAE is in **log10(GPa)**. An MAE of {metrics['models'][best['model']]['summary']['mae']['mean']:.4f}
  is a factor of about {10 ** metrics['models'][best['model']]['summary']['mae']['mean']:.2f} in GPa on a typical error.
- The Dummy model is the honest floor: MAE
  {metrics['models']['dummy']['summary']['mae']['mean']:.4f}, R²
  {metrics['models']['dummy']['summary']['r2']['mean']:.4f}.
- R² is variance explained on held-out folds, not a probability of correct use.
- Official folds are random with respect to composition: this is interpolation,
  not extrapolation to a new alloy family.
"""
    )
    artefact_caption(
        PATHS["metrics"] / "benchmark_metrics.json",
        PATHS["splits"] / "split_manifest.json",
        PATHS["predictions"] / f"oof_{best['model']}_matbench_log_gvrh.csv",
    )
    st.caption(DISCLAIMER)


main()
