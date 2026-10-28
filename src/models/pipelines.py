"""Leak-proof scikit-learn pipeline construction.

Every model is a ``Pipeline`` whose first step is a stateless composition
featurizer and whose remaining steps (imputation, scaling, estimator) are
fitted *inside* the training fold only. No transformer ever sees the held-out
fold and no pre-featurized global matrix exists on disk.
"""

from __future__ import annotations

from typing import Any

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.build import build_featurizer

IDENTIFIER_FEATURES = frozenset(
    {"id", "material_id", "mp-id", "mp_id", "icsd_id", "structure_id", "cif", "formula"}
)

MODEL_NAMES = ("dummy", "ridge", "random_forest")


def make_estimator(name: str, random_state: int = 42) -> Any:
    """Return an unfitted estimator by registry name."""
    if name == "dummy":
        return DummyRegressor(strategy="mean")
    if name == "dummy_median":
        return DummyRegressor(strategy="median")
    if name == "ridge":
        return Ridge(alpha=1.0)
    if name == "linear":
        from sklearn.linear_model import LinearRegression

        return LinearRegression()
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=300,
            max_features=0.33,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=random_state,
        )
    if name == "random_forest_fast":  # used only for y-randomization / diagnostics
        return RandomForestRegressor(
            n_estimators=100,
            max_features=0.33,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=random_state,
        )
    raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")


def build_pipeline(
    model_name: str,
    featurizer_name: str = "magpie",
    random_state: int = 42,
) -> Pipeline:
    """Build ``features -> impute -> scale -> estimator``.

    The featurizer is stateless. ``SimpleImputer`` and ``StandardScaler`` are
    fitted per fold by ``Pipeline.fit``. Calling this function creates a fresh
    object graph so no fitted state can leak between folds.
    """
    featurizer = build_featurizer(featurizer_name)
    return Pipeline(
        steps=[
            ("features", featurizer),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", make_estimator(model_name, random_state=random_state)),
        ]
    )


def feature_names(pipeline: Pipeline) -> list[str]:
    """Return the featurizer's output names for a fitted pipeline."""
    featurizer = pipeline.named_steps["features"]
    return list(featurizer.get_feature_names_out())


def assert_no_identifier_features(names: list[str]) -> None:
    """Fail loudly if an identifier slipped into the feature matrix."""
    lowered = {n.strip().lower() for n in names}
    bad = sorted(IDENTIFIER_FEATURES & lowered)
    if bad:
        raise AssertionError(f"Identifier columns found in feature set: {bad}")
