"""Shared fixtures.

The suite reads the *real* cached dataset and persisted artefacts. If an
artefact is missing, tests fail with an actionable message rather than being
silently skipped: this project treats missing evidence as a defect.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.config import PATHS, TARGET_COLUMN
from src.data.loader import load_cached, load_fold_indices

ROOT = Path(__file__).resolve().parents[1]


def _require(path: Path) -> Path:
    if not path.exists():
        pytest.fail(f"Required artefact missing: {path}. Run the documented pipeline first.")
    return path


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def dataset() -> tuple[pd.DataFrame, dict]:
    if not (PATHS["data_raw"] / "matbench_log_gvrh.pkl.gz").exists():
        pytest.fail("Raw cache missing. Run `python scripts/build_dataset.py` first.")
    return load_cached()


@pytest.fixture(scope="session")
def benchmark_frame(dataset) -> pd.DataFrame:
    return dataset[0]


@pytest.fixture(scope="session")
def provenance(dataset) -> dict:
    return dataset[1]


@pytest.fixture(scope="session")
def folds(benchmark_frame) -> dict:
    return load_fold_indices(benchmark_frame)


@pytest.fixture(scope="session")
def target_column() -> str:
    return TARGET_COLUMN


@pytest.fixture(scope="session")
def metrics() -> dict:
    return json.loads(_require(PATHS["metrics"] / "benchmark_metrics.json").read_text())


@pytest.fixture(scope="session")
def split_manifest() -> dict:
    return json.loads(_require(PATHS["splits"] / "split_manifest.json").read_text())


@pytest.fixture(scope="session")
def data_validation() -> dict:
    return json.loads(_require(PATHS["reports"] / "data_validation.json").read_text())


@pytest.fixture(scope="session")
def best_model() -> dict:
    return json.loads(_require(PATHS["metrics"] / "best_model.json").read_text())


@pytest.fixture(scope="session")
def y_randomization() -> dict:
    return json.loads(_require(PATHS["validation"] / "y_randomization.json").read_text())


@pytest.fixture(scope="session")
def surface_gate() -> dict:
    return json.loads(_require(PATHS["validation"] / "surface_gate.json").read_text())


@pytest.fixture(scope="session")
def clean_evidence() -> pd.DataFrame:
    return pd.read_csv(_require(PATHS["data_literature"] / "surface_evidence_clean.csv"))


@pytest.fixture(scope="session")
def screening_matrix() -> pd.DataFrame:
    return pd.read_csv(_require(PATHS["screening"] / "criterion_matrix.csv"))
