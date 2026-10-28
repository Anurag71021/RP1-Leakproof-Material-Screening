"""Composition-only featurizers.

Why composition and not structure?
----------------------------------
The screening engine must score hypothetical alloy candidates and literature
grades for which no relaxed crystal structure exists. Using structure-derived
features for the benchmark and composition features for screening would create
an inconsistency between the validated model and its deployment path. We
therefore restrict the benchmark model to composition descriptors.

Leakage properties
------------------
Both featurizers are **stateless**: ``fit`` learns nothing from the training
labels or feature distribution. All imputation/scaling is performed downstream
inside the per-fold scikit-learn pipeline. A test asserts that features for a
given composition are identical regardless of which fold was used to fit.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils.logging import get_logger

logger = get_logger(__name__)


def _as_composition(item: Any):
    from pymatgen.core import Composition, Structure

    if isinstance(item, Structure):
        return item.composition
    if isinstance(item, Composition):
        return item
    if isinstance(item, str):
        return Composition(item)
    if hasattr(item, "composition"):
        return item.composition
    raise TypeError(f"Cannot interpret {type(item)!r} as a composition.")


class _BaseCompositionFeaturizer(BaseEstimator, TransformerMixin):
    """Shared plumbing for stateless composition featurizers."""

    _column = "structure"

    def fit(self, X: Any, y: Any = None) -> "_BaseCompositionFeaturizer":
        self.is_fitted_ = True
        return self

    def _extract(self, X: Any) -> list[Any]:
        if isinstance(X, pd.DataFrame):
            if self._column in X.columns:
                values = X[self._column].tolist()
            elif "composition" in X.columns:
                values = X["composition"].tolist()
            elif X.shape[1] == 1:
                values = X.iloc[:, 0].tolist()
            else:
                raise ValueError(
                    f"Expected a '{self._column}' column or a single column; got {list(X.columns)}"
                )
        elif isinstance(X, pd.Series):
            values = X.tolist()
        elif isinstance(X, (list, tuple, np.ndarray)):
            values = list(X)
        else:
            raise TypeError(f"Unsupported input type {type(X)!r}")
        return [_as_composition(v) for v in values]

    def get_feature_names_out(self, input_features: Iterable[str] | None = None) -> np.ndarray:
        if not hasattr(self, "feature_names_out_"):
            raise RuntimeError("Featurizer has not been fitted/transformed yet.")
        return np.asarray(self.feature_names_out_, dtype=object)


class MagpieCompositionFeaturizer(_BaseCompositionFeaturizer):
    """Matminer Magpie composition descriptors (132 features)."""

    def __init__(self, impute_nan: bool = True):
        self.impute_nan = impute_nan

    def _make_featurizer(self):
        from matminer.featurizers.composition import ElementProperty

        featurizer = ElementProperty.from_preset("magpie", impute_nan=self.impute_nan)
        featurizer.set_n_jobs(1)  # process spawning is unsafe in notebooks / Streamlit
        return featurizer

    def transform(self, X: Any) -> pd.DataFrame:
        compositions = self._extract(X)
        featurizer = self._make_featurizer()
        rows = featurizer.featurize_many(compositions, ignore_errors=True, pbar=False)
        labels = list(featurizer.feature_labels())
        array = np.asarray(rows, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        array[~np.isfinite(array)] = np.nan
        self.feature_names_out_ = labels
        return pd.DataFrame(array, columns=labels, index=getattr(X, "index", None))


class SimpleCompositionFeaturizer(_BaseCompositionFeaturizer):
    """Dependency-light fallback using pymatgen elemental properties.

    Used only if matminer is unavailable. It is intentionally simple and is
    reported as a secondary configuration; results are not mixed with Magpie
    results.
    """

    PROPERTIES = (
        "X",
        "atomic_mass",
        "atomic_radius",
        "average_ionic_radius",
        "melting_point",
        "density_of_solid",
        "thermal_conductivity",
        "bulk_modulus",
        "shear_modulus",
        "poisson_ratio",
        "atomic_number",
    )

    def __init__(self) -> None:
        pass

    @staticmethod
    def _prop(element: Any, name: str) -> float:
        value = getattr(element, name, None)
        if value is None:
            return float("nan")
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    def _one(self, composition) -> tuple[list[float], list[str]]:
        fractions = np.array([composition.get_atomic_fraction(el) for el in composition.elements], dtype=float)
        fractions = fractions / fractions.sum()
        values: list[float] = []
        names: list[str] = []
        for prop in self.PROPERTIES:
            raw = np.array([self._prop(el, prop) for el in composition.elements], dtype=float)
            if np.all(np.isnan(raw)):
                stats = [np.nan] * 7
            else:
                mean = float(np.nansum(raw * fractions))
                finite = raw[np.isfinite(raw)]
                std = float(np.sqrt(np.nansum(fractions * (raw - mean) ** 2)))
                avg_dev = float(np.nansum(fractions * np.abs(raw - mean)))
                stats = [
                    mean,
                    float(np.nanmin(finite)),
                    float(np.nanmax(finite)),
                    float(np.nanmax(finite) - np.nanmin(finite)),
                    std,
                    avg_dev,
                    float(fractions[int(np.nanargmax(np.where(np.isnan(raw), -np.inf, raw)))]),
                ]
            for stat, value in zip(("mean", "min", "max", "range", "std", "avg_dev", "max_frac"), stats):
                names.append(f"simple_{prop}_{stat}")
                values.append(value)
        entropy = float(-np.sum(fractions * np.log(fractions + 1e-12)))
        values.extend([float(len(composition.elements)), entropy, float(fractions.max())])
        names.extend(["simple_n_elements", "simple_fraction_entropy", "simple_max_fraction"])
        return values, names

    def transform(self, X: Any) -> pd.DataFrame:
        compositions = self._extract(X)
        rows, names = [], None
        for composition in compositions:
            values, names = self._one(composition)
            rows.append(values)
        assert names is not None
        array = np.asarray(rows, dtype=float)
        array[~np.isfinite(array)] = np.nan
        self.feature_names_out_ = names
        return pd.DataFrame(array, columns=names, index=getattr(X, "index", None))


def build_featurizer(name: str = "magpie") -> _BaseCompositionFeaturizer:
    """Build a featurizer, falling back to the dependency-light version."""
    if name == "simple_composition":
        return SimpleCompositionFeaturizer()
    try:
        featurizer = MagpieCompositionFeaturizer()
        featurizer.fit(pd.DataFrame(columns=["structure"]))
        # Probe one composition so import/feature-library problems surface now.
        featurizer.transform(pd.DataFrame({"structure": ["Ti6Al4V"]}))
        return featurizer
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning("Magpie featurization unavailable (%s); using simple fallback.", exc)
        return SimpleCompositionFeaturizer()
