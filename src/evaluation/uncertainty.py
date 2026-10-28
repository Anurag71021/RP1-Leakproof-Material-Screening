"""Split-conformal uncertainty from out-of-fold residuals.

We avoid claiming probabilistic clinical risk. Intervals are calibrated with
leave-one-fold-out residual quantiles and reported with empirical coverage so
the reader can see how honest they are.
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import PATHS
from src.utils.io import utc_now, write_json
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _quantile(values: np.ndarray, level: float) -> float:
    # Finite-sample conformal correction.
    n = len(values)
    q = min(1.0, np.ceil((n + 1) * level) / n)
    return float(np.quantile(values, q))


def conformal_from_oof(
    oof: pd.DataFrame,
    model_name: str,
    levels: tuple[float, ...] = (0.80, 0.90, 0.95),
) -> dict[str, Any]:
    """Leave-one-fold-out conformal calibration and coverage evaluation."""
    y_true = oof["y_true"].to_numpy(dtype=float)
    y_pred = oof["y_pred"].to_numpy(dtype=float)
    folds = oof["fold_id"].to_numpy()
    abs_resid = np.abs(y_true - y_pred)

    coverage_records: list[dict[str, Any]] = []
    for level in levels:
        widths = []
        covered = []
        for fold_id in np.unique(folds):
            cal = abs_resid[folds != fold_id]
            test = (folds == fold_id)
            q = _quantile(cal, level)
            widths.append(q)
            covered.append(np.mean(abs_resid[test] <= q))
        coverage_records.append(
            {
                "nominal_coverage": float(level),
                "mean_interval_halfwidth": float(np.mean(widths)),
                "empirical_coverage": float(np.mean(covered)),
                "coverage_by_fold": [float(c) for c in covered],
                "abs_coverage_gap": float(abs(np.mean(covered) - level)),
            }
        )

    # Deployment intervals: calibrate on all OOF residuals.
    deployment = {
        str(level): _quantile(abs_resid, level) for level in levels
    }
    intervals = oof.copy()
    for level in levels:
        half = deployment[str(level)]
        intervals[f"lower_{int(level * 100)}"] = intervals["y_pred"] - half
        intervals[f"upper_{int(level * 100)}"] = intervals["y_pred"] + half
    intervals.to_csv(PATHS["uncertainty"] / f"conformal_intervals_{model_name}.csv", index=False)

    payload = {
        "method": "split conformal, absolute OOF residual nonconformity",
        "model": model_name,
        "created_utc": utc_now(),
        "n": int(len(oof)),
        "levels": list(levels),
        "calibration_scheme": (
            "Coverage: for each held-out fold, quantiles are calibrated on the "
            "other four folds' OOF residuals. Deployment half-widths use all OOF "
            "residuals and are therefore slightly optimistic on the training data."
        ),
        "deployment_halfwidths": deployment,
        "coverage": coverage_records,
        "absolute_residual_median": float(np.median(abs_resid)),
        "absolute_residual_p95": float(np.quantile(abs_resid, 0.95)),
    }
    write_json(PATHS["uncertainty"] / f"conformal_{model_name}.json", payload)
    _plot(payload)
    return payload


def _plot(payload: dict[str, Any]) -> None:
    records = payload["coverage"]
    nominal = [r["nominal_coverage"] for r in records]
    empirical = [r["empirical_coverage"] for r in records]

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.plot([0.75, 1.0], [0.75, 1.0], "k--", linewidth=1, label="perfect calibration")
    ax.plot(nominal, empirical, "o-", label=payload["model"])
    for x, y in zip(nominal, empirical):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(6, -12), fontsize=8)
    ax.set_xlim(0.75, 1.0)
    ax.set_ylim(0.75, 1.0)
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage (held-out folds)")
    ax.set_title("Conformal interval calibration")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PATHS["plots"] / "uncertainty_calibration.png", dpi=150)
    plt.close(fig)


def interval_for_prediction(pred: float, level: float, model_name: str) -> tuple[float, float]:
    """Look up a stored deployment half-width for the app."""
    from src.utils.io import read_json

    path = PATHS["uncertainty"] / f"conformal_{model_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Uncertainty artefact missing: {path}")
    payload = read_json(path)
    half = payload["deployment_halfwidths"][str(level)]
    return float(pred - half), float(pred + half)
