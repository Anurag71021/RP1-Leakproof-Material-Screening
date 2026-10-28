"""Feature-building facade.

The public contract is intentionally narrow: build a *fresh* featurizer for a
fold (or for deployment) and let scikit-learn call it inside a Pipeline.
"""

from __future__ import annotations

from src.features.composition import (
    MagpieCompositionFeaturizer,
    SimpleCompositionFeaturizer,
    build_featurizer,
)

__all__ = [
    "MagpieCompositionFeaturizer",
    "SimpleCompositionFeaturizer",
    "build_featurizer",
    "DEFAULT_FEATURIZER",
]

DEFAULT_FEATURIZER = "magpie"
