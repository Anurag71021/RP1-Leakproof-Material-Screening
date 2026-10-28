"""Permutation importance on held-out folds.

Importance is computed for the *fitted estimator* on each held-out fold's
transformed features. The featurizer and preprocessing are those fitted inside
that fold, so no held-out information influences the transform.
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.config import PATHS, TARGET_COLUMN
from src.models.pipelines import build_pipeline
from src.utils.io import utc_now, write_json
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _transform(pipeline, frame: pd.DataFrame) -> np.ndarray:
    X = pipeline.named_steps["features"].transform(frame[["structure"]])
    X = pipeline.named_steps["impute"].transform(X)
    X = pipeline.named_steps["scale"].transform(X)
    return np.asarray(X, dtype=float)


def permutation_importance_cv(
    frame: pd.DataFrame,
    folds: dict[int, dict[str, list[int]]],
    model_name: str,
    n_repeats: int = 5,
    max_samples: int = 1500,
    featurizer_name: str = "magpie",
    random_state: int = 42,
) -> dict[str, Any] | None:
    """Aggregate permutation importance across the official folds."""
    if model_name == "dummy":
        logger.info("Skipping permutation importance for DummyRegressor.")
        return None

    per_fold: list[pd.DataFrame] = []
    rng = np.random.RandomState(random_state)

    for fold_id in sorted(folds):
        train = frame.iloc[folds[fold_id]["train_idx"]]
        test = frame.iloc[folds[fold_id]["test_idx"]]
        if len(test) > max_samples:
            pick = rng.choice(len(test), size=max_samples, replace=False)
            test = test.iloc[pick]

        pipeline = build_pipeline(model_name, featurizer_name, random_state=random_state)
        pipeline.fit(train[["structure"]], train[TARGET_COLUMN].to_numpy(dtype=float))
        X = _transform(pipeline, test)
        y = test[TARGET_COLUMN].to_numpy(dtype=float)
        names = list(pipeline.named_steps["features"].get_feature_names_out())

        result = permutation_importance(
            pipeline.named_steps["model"],
            X,
            y,
            scoring="neg_mean_absolute_error",
            n_repeats=n_repeats,
            random_state=random_state + fold_id,
            n_jobs=1,
        )
        per_fold.append(
            pd.DataFrame(
                {
                    "feature": names,
                    "importance_mean": result.importances_mean,
                    "importance_std": result.importances_std,
                    "fold_id": fold_id,
                }
            )
        )
        logger.info("Permutation importance done for fold %d (%d samples).", fold_id, len(test))

    combined = pd.concat(per_fold, ignore_index=True)
    aggregate = (
        combined.groupby("feature", as_index=False)
        .agg(
            importance_mean=("importance_mean", "mean"),
            importance_std_across_folds=("importance_mean", "std"),
            n_folds=("fold_id", "nunique"),
        )
        .sort_values("importance_mean", ascending=False)
    )
    aggregate.to_csv(PATHS["interpretability"] / f"permutation_importance_{model_name}.csv", index=False)

    top = aggregate.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std_across_folds"], alpha=0.85)
    ax.set_xlabel("Mean decrease in MAE when permuted (log10 GPa)")
    ax.set_title(f"Permutation importance - {model_name} (top 20)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PATHS["plots"] / f"permutation_importance_{model_name}.png", dpi=150)
    plt.close(fig)

    payload = {
        "model": model_name,
        "created_utc": utc_now(),
        "n_repeats": n_repeats,
        "max_samples_per_fold": max_samples,
        "interpretation": (
            "Associational only. Permutation importance describes reliance of "
            "this fitted model on descriptor columns; it is not causal and does "
            "not imply a physical mechanism or a clinical effect."
        ),
        "top_10": aggregate.head(10).to_dict(orient="records"),
        "n_features": int(len(aggregate)),
        "csv": f"results/interpretability/permutation_importance_{model_name}.csv",
        "plot": f"results/plots/permutation_importance_{model_name}.png",
    }
    write_json(PATHS["interpretability"] / f"permutation_importance_{model_name}.json", payload)
    return payload


def shap_available() -> bool:
    """SHAP is optional; report the actual availability instead of assuming."""
    try:
        import shap  # noqa: F401

        return True
    except Exception:
        return False
