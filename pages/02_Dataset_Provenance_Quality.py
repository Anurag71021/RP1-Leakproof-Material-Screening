"""Page 2: dataset provenance, quality checks and locked official folds."""

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
    yes_no,
)

st.set_page_config(page_title="02 Dataset | RP-1", layout="wide")


def main() -> None:
    page_header(
        "02 · Dataset provenance & quality",
        "Official Matbench v0.1 target, checksums, duplicate audit and immutable folds",
    )
    disclaimer()

    report = load_json(PATHS["reports"] / "data_validation.json")
    manifest = load_json(PATHS["splits"] / "split_manifest.json")
    if not report or not manifest:
        missing(PATHS["reports"] / "data_validation.json", "python scripts/build_dataset.py")
        return

    provenance = report["provenance"]
    validation = report["validation"]
    duplicates = report["duplicates"]

    st.subheader("Identity")
    st.markdown(
        f"""
| Field | Value |
| --- | --- |
| Dataset | `{provenance['dataset_id']}` |
| Benchmark | `{provenance['benchmark_name']}` |
| Rows | {provenance['n_samples']:,} |
| Target | `log10(G_VRH)` = log10 Voigt–Reuss–Hill shear modulus |
| Units | log10(GPa) |
| Matbench version | `{provenance['matbench_version']}` |
| Retrieved (UTC) | `{provenance['retrieved_utc']}` |
| Source URL | {provenance['source_url']} |
| Official metadata hash | `{provenance['official_hash']}` |
| Cached file SHA-256 | `{provenance['cache_sha256'][:32]}…` |
| Row-index SHA-256 | `{provenance['index_sha256'][:32]}…` |
"""
    )
    st.info(
        "The target is a **computed** DFT property. It is not an experimental "
        "implant modulus, not a fatigue limit, and not a biocompatibility measure."
    )

    st.subheader("Validation checks")
    checks = {
        "Required columns present": validation["required_columns_present"],
        "Index unique": validation["index_unique"],
        "Target numeric": validation["target_numeric"],
        "Target finite": validation["target_finite"],
        "Structures non-empty": validation["structures_nonempty"],
        "No identifier columns": validation["leakage_safe_columns"],
        "Manifest checks": manifest["checks_passed"],
    }
    st.dataframe(
        pd.DataFrame({"Check": list(checks), "Result": [yes_no(v) for v in checks.values()]}),
        use_container_width=True,
        hide_index=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Target min", f"{validation['target_min']:.4f}")
    col2.metric("Target max", f"{validation['target_max']:.4f}")
    col3.metric("Target mean", f"{validation['target_mean']:.4f}")
    col4.metric("Target std", f"{validation['target_std']:.4f}")
    st.caption("Units: log10(GPa). Values are read from `reports/data_validation.json`.")

    st.subheader("Target distribution (from OOF truth, all rows)")
    best = load_json(PATHS["metrics"] / "best_model.json", {})
    best_name = best.get("model", "random_forest")
    oof = load_csv(PATHS["predictions"] / f"oof_{best_name}_matbench_log_gvrh.csv")
    if oof is not None:
        st.bar_chart(
            pd.cut(oof["y_true"], bins=40).value_counts().sort_index().rename("count")
        )
        st.caption(
            "All 10,987 truth values are shown by reusing the OOF file. "
            "The distribution of a broad inorganic materials set is **not** an "
            "implant-alloy distribution."
        )
    else:
        missing(PATHS["predictions"] / f"oof_{best_name}_matbench_log_gvrh.csv", "python scripts/run_benchmark.py")

    st.subheader("Locked official folds")
    folds = pd.DataFrame(manifest["folds"])[
        ["fold_id", "n_train", "n_test", "val_equals_test"]
    ]
    st.dataframe(folds, use_container_width=True, hide_index=True)
    st.markdown(
        f"""
- Split type: `{manifest['split_type']}` (source: `{manifest['source']}`).
- The held-out official fold is also the validation fold for OOF prediction;
  no extra split is carved from training data.
- All manifest checks passed: **{yes_no(manifest['checks_passed'])}**.
- Index checksum: `{manifest['index']['sha256'][:32]}…`
"""
    )

    st.subheader("Duplicate audit")
    st.markdown(
        f"""
| Metric | Value |
| --- | --- |
| Duplicate groups | {duplicates['n_duplicate_groups']} |
| Redundant rows | {duplicates['n_duplicate_rows_beyond_first']} |
| Cross-fold duplicate groups | {duplicates['n_cross_fold_duplicate_groups']} |
| Method | {duplicates['method']} |

{duplicates['action']}
"""
    )

    st.subheader("Data card")
    card = read_text(PATHS["data_cards"] / "matbench_log_gvrh_data_card.md")
    if card:
        with st.expander("Open the full data card"):
            st.markdown(card)
    else:
        missing(PATHS["data_cards"] / "matbench_log_gvrh_data_card.md", "python scripts/build_dataset.py")

    st.subheader("Raw fold manifest")
    csv_download(folds, "official_folds_summary.csv", "Download fold summary CSV")
    artefact_caption(
        PATHS["reports"] / "data_validation.json",
        PATHS["splits"] / "split_manifest.json",
        PATHS["data_cards"] / "matbench_log_gvrh_data_card.md",
    )
    st.caption(DISCLAIMER)


main()
