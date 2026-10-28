"""Phase 4 validation: Y-randomization, learning curves, parity and residuals.

These are *gate* artefacts. They are generated from the same official folds,
the same featurizer and the same estimator configuration as the real model.
Y-randomization deliberately destroys the composition-target relationship; a
model that still predicts well is evidence of leakage or a broken split.
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import PATHS, TARGET_COLUMN
from src.evaluation.metrics import regression_metrics, summarize_folds
from src.models.pipelines import build_pipeline
from src.utils.io import utc_now, write_json
from src.utils.logging import get_logger

logger = get_logger(__name__)


def y_randomization(
    frame: pd.DataFrame,
    folds: dict[int, dict[str, list[int]]],
    model_names: list[str],
    n_permutations: dict[str, int] | None = None,
    featurizer_name: str = "magpie",
    random_state: int = 42,
) -> dict[str, Any]:
    """Shuffle training labels within each fold and re-run the identical pipeline.

    The test labels are never shuffled. A valid pipeline must collapse to
    Dummy-level performance (R² near zero or negative).
    """
    n_permutations = n_permutations or {"ridge": 5, "random_forest": 2}
    payload: dict[str, Any] = {
        "description": (
            "Y-randomization with official folds, identical featurization and "
            "estimator settings. Only training labels are permuted; test labels "
            "remain true. Expect R2 approximately 0 or negative."
        ),
        "created_utc": utc_now(),
        "featurizer": featurizer_name,
        "random_state": random_state,
        "models": {},
    }

    for model_name in model_names:
        n_perm = int(n_permutations.get(model_name, 2))
        logger.info("Y-randomization: %s, %d permutations", model_name, n_perm)
        per_permutation: list[dict[str, Any]] = []
        for perm_id in range(n_perm):
            fold_metrics = []
            for fold_id in sorted(folds):
                train_idx = folds[fold_id]["train_idx"]
                test_idx = folds[fold_id]["test_idx"]
                train = frame.iloc[train_idx]
                test = frame.iloc[test_idx]

                y_train = train[TARGET_COLUMN].to_numpy(dtype=float)
                rng = np.random.RandomState(random_state + 7919 * perm_id + fold_id)
                shuffled = rng.permutation(y_train)

                pipeline = build_pipeline(model_name, featurizer_name, random_state=random_state)
                pipeline.fit(train[["structure"]], shuffled)
                pred = pipeline.predict(test[["structure"]])
                metrics = regression_metrics(test[TARGET_COLUMN].to_numpy(dtype=float), pred)
                metrics["fold_id"] = fold_id
                fold_metrics.append(metrics)

            summary = summarize_folds(fold_metrics)
            per_permutation.append(
                {"permutation_id": perm_id, "per_fold": fold_metrics, "summary": summary}
            )
            logger.info(
                "  perm %d: R2=%.4f MAE=%.4f",
                perm_id,
                summary["r2"]["mean"],
                summary["mae"]["mean"],
            )

        r2_means = [p["summary"]["r2"]["mean"] for p in per_permutation]
        payload["models"][model_name] = {
            "n_permutations": n_perm,
            "per_permutation": per_permutation,
            "aggregate": {
                "r2_mean": float(np.mean(r2_means)),
                "r2_max": float(np.max(r2_means)),
                "mae_mean": float(np.mean([p["summary"]["mae"]["mean"] for p in per_permutation])),
                "rmse_mean": float(np.mean([p["summary"]["rmse"]["mean"] for p in per_permutation])),
            },
        }

    write_json(PATHS["validation"] / "y_randomization.json", payload)
    return payload


def learning_curves(
    frame: pd.DataFrame,
    folds: dict[int, dict[str, list[int]]],
    model_names: list[str],
    sizes: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0),
    fold_id: int = 0,
    featurizer_name: str = "magpie",
    random_state: int = 42,
) -> dict[str, Any]:
    """Train on increasing subsets of one official training fold.

    One representative fold is used because each additional fold multiplies
    the (already expensive) Random Forest fits. This is stated in the report.
    """
    fold = folds[fold_id]
    train_idx = np.array(fold["train_idx"])
    test = frame.iloc[fold["test_idx"]]
    y_test = test[TARGET_COLUMN].to_numpy(dtype=float)

    rng = np.random.RandomState(random_state)
    shuffled = rng.permutation(train_idx)

    payload: dict[str, Any] = {
        "created_utc": utc_now(),
        "fold_id": fold_id,
        "sizes": list(sizes),
        "note": "Learning curves use one official training fold, not all five.",
        "models": {},
    }

    for model_name in model_names:
        records = []
        for fraction in sizes:
            n = max(50, int(len(train_idx) * fraction))
            subset = frame.iloc[shuffled[:n]]
            pipeline = build_pipeline(model_name, featurizer_name, random_state=random_state)
            pipeline.fit(subset[["structure"]], subset[TARGET_COLUMN].to_numpy(dtype=float))
            train_pred = pipeline.predict(subset[["structure"]])
            test_pred = pipeline.predict(test[["structure"]])
            train_metrics = regression_metrics(subset[TARGET_COLUMN].to_numpy(dtype=float), train_pred)
            test_metrics = regression_metrics(y_test, test_pred)
            records.append(
                {
                    "fraction": fraction,
                    "n_train": n,
                    "train_mae": train_metrics["mae"],
                    "train_r2": train_metrics["r2"],
                    "val_mae": test_metrics["mae"],
                    "val_r2": test_metrics["r2"],
                    "gap_mae": test_metrics["mae"] - train_metrics["mae"],
                }
            )
            logger.info(
                "learning curve %s n=%d: train MAE=%.4f val MAE=%.4f",
                model_name,
                n,
                train_metrics["mae"],
                test_metrics["mae"],
            )
        payload["models"][model_name] = records

        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.plot([r["n_train"] for r in records], [r["train_mae"] for r in records], "o-", label="train MAE")
        ax.plot([r["n_train"] for r in records], [r["val_mae"] for r in records], "s--", label="held-out MAE")
        ax.set_xlabel("Training samples")
        ax.set_ylabel("MAE (log10 GPa)")
        ax.set_title(f"Learning curve - {model_name} (official fold {fold_id})")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(PATHS["plots"] / f"learning_curves_{model_name}.png", dpi=150)
        plt.close(fig)

    write_json(PATHS["validation"] / "learning_curves.json", payload)
    return payload


def parity_and_residuals(oof: pd.DataFrame, model_name: str) -> dict[str, Any]:
    """Create parity and residual plots and return residual diagnostics."""
    y_true = oof["y_true"].to_numpy(dtype=float)
    y_pred = oof["y_pred"].to_numpy(dtype=float)
    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.scatter(y_true, y_pred, s=6, alpha=0.35, edgecolors="none")
    lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="parity")
    ax.set_xlabel("True log10(G_VRH)  [log10 GPa]")
    ax.set_ylabel("Predicted log10(G_VRH)  [log10 GPa]")
    ax.set_title(f"Parity - {model_name} (5-fold OOF, n={len(oof)})")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PATHS["plots"] / f"parity_plot_{model_name}.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.scatter(y_pred, residuals, s=6, alpha=0.35, edgecolors="none")
    ax.axhline(0.0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted log10(G_VRH)")
    ax.set_ylabel("Residual (true - predicted)")
    ax.set_title(f"Residuals - {model_name}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PATHS["plots"] / f"residuals_{model_name}.png", dpi=150)
    plt.close(fig)

    # Correlation between |residual| and prediction: a crude heteroscedasticity probe.
    if np.std(y_pred) > 0 and np.std(np.abs(residuals)) > 0:
        hetero_r = float(np.corrcoef(y_pred, np.abs(residuals))[0, 1])
    else:
        hetero_r = float("nan")

    stats = {
        "model": model_name,
        "n": int(len(oof)),
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals, ddof=1)),
        "residual_median": float(np.median(residuals)),
        "abs_residual_p95": float(np.quantile(np.abs(residuals), 0.95)),
        "corr_pred_abs_residual": hetero_r,
        "parity_plot": f"results/plots/parity_plot_{model_name}.png",
        "residual_plot": f"results/plots/residuals_{model_name}.png",
        "reading": (
            "A strong positive corr(pred, |residual|) indicates error growth at "
            "high predicted modulus (heteroscedasticity), not necessarily leakage."
        ),
    }
    write_json(PATHS["validation"] / f"residual_stats_{model_name}.json", stats)
    return stats
