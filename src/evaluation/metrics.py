"""Regression metrics with explicit edge-case handling."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Compute MAE, RMSE and R² with explicit finite checks.

    R² is defined as NaN when the truth has zero variance, rather than
    silently returning an arbitrary value.
    """
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError(f"Shape mismatch: {true.shape} vs {pred.shape}")
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("Non-finite values in y_true/y_pred; refusing to report metrics.")

    mae = float(mean_absolute_error(true, pred))
    rmse = float(np.sqrt(mean_squared_error(true, pred)))
    if np.allclose(true, true[0]):
        r2 = float("nan")
    else:
        r2 = float(r2_score(true, pred))
    return {"mae": mae, "rmse": rmse, "r2": r2, "n": int(len(true))}


def summarize_folds(per_fold: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Mean and sample standard deviation across folds for each metric."""
    summary: dict[str, dict[str, float]] = {}
    for metric in ("mae", "rmse", "r2"):
        values = np.array([f[metric] for f in per_fold], dtype=float)
        summary[metric] = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0,
        }
    return summary
