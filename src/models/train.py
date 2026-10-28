"""Leak-proof cross-validation over the official Matbench folds."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import PATHS, TARGET_COLUMN
from src.evaluation.metrics import regression_metrics, summarize_folds
from src.models.pipelines import build_pipeline, feature_names
from src.utils.io import utc_now, write_json
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _slice(frame: pd.DataFrame, idx: list[int]) -> pd.DataFrame:
    return frame.iloc[idx].copy()


def run_official_cv(
    frame: pd.DataFrame,
    folds: dict[int, dict[str, list[int]]],
    model_names: list[str],
    featurizer_name: str = "magpie",
    random_state: int = 42,
    target_column: str = TARGET_COLUMN,
    persist: bool = True,
) -> dict[str, Any]:
    """Fit each model once per official fold and save OOF predictions.

    Returns a payload with per-fold and summary metrics. All predictions are
    out-of-fold: a row is predicted only by the model that never saw its fold
    during training.
    """
    results: dict[str, Any] = {
        "dataset": "matbench_log_gvrh",
        "target": target_column,
        "target_units": "log10(GPa)",
        "split_type": "official_folds",
        "featurizer": featurizer_name,
        "created_utc": utc_now(),
        "random_state": random_state,
        "n_samples": int(len(frame)),
        "models": {},
    }

    for model_name in model_names:
        logger.info("=== Model %s | official 5-fold CV ===", model_name)
        per_fold: list[dict[str, Any]] = []
        oof_rows: list[pd.DataFrame] = []
        feature_count: int | None = None

        for fold_id in sorted(folds):
            train_idx = folds[fold_id]["train_idx"]
            test_idx = folds[fold_id]["test_idx"]
            train = _slice(frame, train_idx)
            test = _slice(frame, test_idx)

            pipeline = build_pipeline(model_name, featurizer_name, random_state=random_state)
            logger.info(
                "fold %d: featurizing %d train / %d test compositions (per-fold, no global matrix)",
                fold_id,
                len(train),
                len(test),
            )
            pipeline.fit(train[["structure"]], train[target_column].to_numpy(dtype=float))
            pred = pipeline.predict(test[["structure"]])
            metrics = regression_metrics(test[target_column].to_numpy(dtype=float), pred)
            metrics.update({"fold_id": fold_id, "n_train": len(train), "n_test": len(test)})
            per_fold.append(metrics)
            if feature_count is None:
                feature_count = len(feature_names(pipeline))
            logger.info(
                "fold %d metrics: MAE=%.4f RMSE=%.4f R2=%.4f",
                fold_id,
                metrics["mae"],
                metrics["rmse"],
                metrics["r2"],
            )

            oof_rows.append(
                pd.DataFrame(
                    {
                        "material_id": test.index.astype(str),
                        "row_index": test_idx,
                        "fold_id": fold_id,
                        "y_true": test[target_column].to_numpy(dtype=float),
                        "y_pred": np.asarray(pred, dtype=float),
                    }
                )
            )

        oof = pd.concat(oof_rows, ignore_index=True).sort_values("row_index").reset_index(drop=True)
        if len(oof) != len(frame) or oof["row_index"].nunique() != len(frame):
            raise AssertionError("OOF predictions do not cover every sample exactly once.")

        oof_path = PATHS["predictions"] / f"oof_{model_name}_matbench_log_gvrh.csv"
        oof.to_csv(oof_path, index=False)

        summary = summarize_folds(per_fold)
        results["models"][model_name] = {
            "per_fold": per_fold,
            "summary": summary,
            "n_features": feature_count,
            "oof_file": str(oof_path.relative_to(PATHS["root"])),
        }
        logger.info(
            "%s summary: MAE=%.4f±%.4f RMSE=%.4f±%.4f R2=%.4f±%.4f",
            model_name,
            summary["mae"]["mean"],
            summary["mae"]["std"],
            summary["rmse"]["mean"],
            summary["rmse"]["std"],
            summary["r2"]["mean"],
            summary["r2"]["std"],
        )

    if persist:
        _persist(results)
    return results


def _persist(results: dict[str, Any]) -> None:
    metrics_path = PATHS["metrics"] / "benchmark_metrics.json"
    write_json(metrics_path, results)

    rows = []
    for model_name, payload in results["models"].items():
        summary = payload["summary"]
        rows.append(
            {
                "Model": model_name,
                "Features": payload["n_features"],
                "MAE (mean)": round(summary["mae"]["mean"], 4),
                "MAE (std)": round(summary["mae"]["std"], 4),
                "RMSE (mean)": round(summary["rmse"]["mean"], 4),
                "RMSE (std)": round(summary["rmse"]["std"], 4),
                "R2 (mean)": round(summary["r2"]["mean"], 4),
                "R2 (std)": round(summary["r2"]["std"], 4),
            }
        )
    table = pd.DataFrame(rows)
    best = min(results["models"], key=lambda m: results["models"][m]["summary"]["mae"]["mean"])
    table = table.sort_values("MAE (mean)")
    markdown = (
        "# Benchmark comparison - matbench_log_gvrh\n\n"
        "Target: `log10(G_VRH)` in log10(GPa), computed DFT shear modulus.\n"
        "Official Matbench 5-fold splits; composition-only descriptors.\n\n"
        + table.to_markdown(index=False)
        + f"\n\n**Selected model (lowest mean OOF MAE):** `{best}`\n"
    )
    (PATHS["tables"] / "model_comparison.md").write_text(markdown, encoding="utf-8")
    write_json(
        PATHS["metrics"] / "best_model.json",
        {
            "selected_by": "lowest mean OOF MAE across official folds",
            "model": best,
            "summary": results["models"][best]["summary"],
            "oof_file": results["models"][best]["oof_file"],
            "created_utc": utc_now(),
        },
    )
    logger.info("Wrote %s and model comparison table.", metrics_path)


def load_oof(model_name: str) -> pd.DataFrame:
    path = PATHS["predictions"] / f"oof_{model_name}_matbench_log_gvrh.csv"
    if not path.exists():
        raise FileNotFoundError(f"OOF file missing: {path}. Run scripts/run_benchmark.py first.")
    return pd.read_csv(path)
