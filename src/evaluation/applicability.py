"""Applicability domain (AD) via k-nearest-neighbour distance in descriptor space.

AD is a *data-support* statement, not a probability of success. A candidate
outside the domain is not "bad"; it is simply not supported by the training
distribution and must not receive a confident prediction.

Per-fold evaluation fits the imputer, scaler and neighbour index on the
official training fold only. The deployment reference is fitted on all
available data and stored for the screening engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.config import PATHS
from src.features.build import build_featurizer
from src.utils.io import utc_now, write_json
from src.utils.logging import get_logger

logger = get_logger(__name__)

IN_DOMAIN = "IN_DOMAIN"
NEAR_BOUNDARY = "NEAR_BOUNDARY"
OUT_OF_DOMAIN = "OUT_OF_DOMAIN"


@dataclass
class _Core:
    imputer: SimpleImputer
    scaler: StandardScaler
    neighbors: NearestNeighbors
    q95: float
    q995: float
    k: int = 5


def _fit_core(X_train: np.ndarray, k: int = 5) -> _Core:
    imputer = SimpleImputer(strategy="median").fit(X_train)
    scaled = imputer.transform(X_train)
    scaler = StandardScaler().fit(scaled)
    train_scaled = scaler.transform(scaled)
    k_eff = min(k, max(1, len(train_scaled) - 1))
    neighbors = NearestNeighbors(n_neighbors=k_eff + 1).fit(train_scaled)
    distances, _ = neighbors.kneighbors(train_scaled)
    train_dist = distances[:, 1:].mean(axis=1)
    return _Core(
        imputer=imputer,
        scaler=scaler,
        neighbors=neighbors,
        q95=float(np.quantile(train_dist, 0.95)),
        q995=float(np.quantile(train_dist, 0.995)),
        k=k_eff,
    )


def _distance(core: _Core, X: np.ndarray) -> np.ndarray:
    scaled = core.scaler.transform(core.imputer.transform(X))
    distances, _ = core.neighbors.kneighbors(scaled, n_neighbors=core.k)
    return distances.mean(axis=1)


def _label(distances: np.ndarray, q95: float, q995: float) -> np.ndarray:
    labels = np.full(len(distances), OUT_OF_DOMAIN, dtype=object)
    labels[distances <= q995] = NEAR_BOUNDARY
    labels[distances <= q95] = IN_DOMAIN
    return labels


class ApplicabilityDomain:
    """Fitted AD reference that can score new compositions."""

    def __init__(self, featurizer_name: str = "magpie", k: int = 5):
        self.featurizer_name = featurizer_name
        self.k = k
        self.featurizer = build_featurizer(featurizer_name)
        self.core: _Core | None = None
        self.feature_names: list[str] = []
        self.fitted_utc: str | None = None

    def fit(self, frame: pd.DataFrame) -> "ApplicabilityDomain":
        X = self.featurizer.fit_transform(frame[["structure"]]).to_numpy(dtype=float)
        X[~np.isfinite(X)] = np.nan
        self.feature_names = list(self.featurizer.get_feature_names_out())
        self.core = _fit_core(X, self.k)
        self.fitted_utc = utc_now()
        return self

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.core is None:
            raise RuntimeError("ApplicabilityDomain is not fitted.")
        X = self.featurizer.transform(frame[["structure"]]).to_numpy(dtype=float)
        X[~np.isfinite(X)] = np.nan
        distances = _distance(self.core, X)
        labels = _label(distances, self.core.q95, self.core.q995)
        return pd.DataFrame(
            {
                "knn_distance": distances,
                "ad_status": labels,
                "ad_q95": self.core.q95,
                "ad_q995": self.core.q995,
            },
            index=frame.index,
        )

    def save(self, path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path) -> "ApplicabilityDomain":
        return joblib.load(path)


def evaluate_ad_by_fold(
    frame: pd.DataFrame,
    folds: dict[int, dict[str, list[int]]],
    featurizer_name: str = "magpie",
    k: int = 5,
) -> dict[str, Any]:
    """Score every sample using an AD fitted only on its official training fold."""
    featurizer = build_featurizer(featurizer_name)
    X = featurizer.fit_transform(frame[["structure"]]).to_numpy(dtype=float)
    X[~np.isfinite(X)] = np.nan

    rows = []
    fold_summary = {}
    for fold_id in sorted(folds):
        train_idx = np.array(folds[fold_id]["train_idx"])
        test_idx = np.array(folds[fold_id]["test_idx"])
        core = _fit_core(X[train_idx], k=k)
        distances = _distance(core, X[test_idx])
        labels = _label(distances, core.q95, core.q995)
        rows.append(
            pd.DataFrame(
                {
                    "material_id": frame.index[test_idx].astype(str),
                    "row_index": test_idx,
                    "fold_id": fold_id,
                    "knn_distance": distances,
                    "ad_status": labels,
                    "ad_q95": core.q95,
                    "ad_q995": core.q995,
                }
            )
        )
        fold_summary[str(fold_id)] = {
            "n": int(len(test_idx)),
            "in_domain": int(np.sum(labels == IN_DOMAIN)),
            "near_boundary": int(np.sum(labels == NEAR_BOUNDARY)),
            "out_of_domain": int(np.sum(labels == OUT_OF_DOMAIN)),
            "q95": core.q95,
            "q995": core.q995,
        }
        logger.info("AD fold %d: %s", fold_id, fold_summary[str(fold_id)])

    result = pd.concat(rows, ignore_index=True)
    result.to_csv(PATHS["validation"] / "applicability_domain_oof.csv", index=False)

    # Deployment reference on all data (documented as such; not used for scoring metrics).
    deployment = ApplicabilityDomain(featurizer_name, k).fit(frame)
    deployment.save(PATHS["artifacts"] / "ad_reference.joblib")
    deployment_scores = deployment.score(frame)

    payload = {
        "created_utc": utc_now(),
        "method": (
            "mean Euclidean distance to k=5 nearest training neighbours in "
            "standardized Magpie composition space; thresholds are the 95th and "
            "99.5th percentiles of training-fold self-distances"
        ),
        "per_fold": fold_summary,
        "overall": {
            status: int((result["ad_status"] == status).sum())
            for status in (IN_DOMAIN, NEAR_BOUNDARY, OUT_OF_DOMAIN)
        },
        "deployment_thresholds": {
            "q95": deployment.core.q95 if deployment.core else None,
            "q995": deployment.core.q995 if deployment.core else None,
        },
        "deployment_counts": {
            status: int((deployment_scores["ad_status"] == status).sum())
            for status in (IN_DOMAIN, NEAR_BOUNDARY, OUT_OF_DOMAIN)
        },
        "artefact": "artifacts/ad_reference.joblib",
        "interpretation": (
            "AD is support in descriptor space. It is not a probability of "
            "implant success, biocompatibility or clinical performance."
        ),
    }
    write_json(PATHS["validation"] / "applicability_domain.json", payload)
    return payload
