"""Official Matbench loading, caching and provenance capture.

Design rules enforced here
--------------------------
* The official ``matbench.bench`` API provides both the dataset and the
  official fold definitions. No custom split is ever created for the
  benchmark.
* Raw data is cached once under ``data/raw/`` and treated as immutable. Every
  read returns a copy.
* The provenance record stores the official hash, retrieval time, source URL,
  fold sizes and citation so a reader can audit what was actually used.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import PATHS, TARGET_COLUMN
from src.utils.io import read_json, sha256_text, write_json
from src.utils.logging import get_logger

logger = get_logger(__name__)

DATASET_ID = "matbench_log_gvrh"
CACHE_PATH = PATHS["data_raw"] / "matbench_log_gvrh.pkl.gz"
FOLDS_PATH = PATHS["data_raw"] / "matbench_log_gvrh_folds.json.gz"
PROVENANCE_PATH = PATHS["data_raw"] / "matbench_log_gvrh_provenance.json"


def _official_task():  # pragma: no cover - network dependent
    from matbench.bench import MatbenchBenchmark

    benchmark = MatbenchBenchmark(autoload=False, subset=[DATASET_ID])
    task = getattr(benchmark, DATASET_ID)
    task.load()
    return benchmark, task


def download_and_cache(force: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Download the official dataset once, cache it, and record provenance."""
    if CACHE_PATH.exists() and PROVENANCE_PATH.exists() and not force:
        logger.info("Using cached raw dataset at %s", CACHE_PATH)
        return load_cached()

    import matbench

    benchmark, task = _official_task()
    frame = task.df.copy()
    metadata = dict(task.metadata)
    benchmark_name = getattr(benchmark, "benchmark_name", "matbench_v0.1")

    if TARGET_COLUMN not in frame.columns:
        raise RuntimeError(f"Official target column {TARGET_COLUMN!r} missing; columns={list(frame.columns)}")

    frame = frame[[ "structure", TARGET_COLUMN ]].copy()
    frame.index = frame.index.astype(str)
    if not frame.index.is_unique:
        raise RuntimeError("Official dataset index is not unique; refusing to continue.")

    PATHS["data_raw"].mkdir(parents=True, exist_ok=True)
    frame.to_pickle(CACHE_PATH)

    fold_records: dict[str, dict[str, list[int]]] = {}
    positions = {material_id: pos for pos, material_id in enumerate(frame.index)}
    for fold_number in task.folds:
        key = task.folds_map[fold_number]
        split = task.validation[key]
        train_idx = [positions[i] for i in split.train]
        test_idx = [positions[i] for i in split.test]
        fold_records[str(fold_number)] = {"train_idx": train_idx, "test_idx": test_idx}

    with gzip.open(FOLDS_PATH, "wt", encoding="utf-8") as handle:
        json.dump(fold_records, handle)

    provenance = {
        "dataset_id": DATASET_ID,
        "task_name": DATASET_ID,
        "benchmark_name": benchmark_name,
        "matbench_version": getattr(matbench, "__version__", "unknown"),
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_url": metadata.get("url"),
        "official_hash": metadata.get("hash"),
        "license": (
            "Matbench v0.1 dataset (Dunn et al. 2020, npj Comput. Mater., "
            "doi:10.1038/s41524-020-00406-3). Underlying elastic data from "
            "de Jong et al. 2015 (Sci. Data, doi:10.1038/sdata.2015.9, CC BY 4.0). "
            "Benchmark code is MIT licensed. Confirm redistribution terms before publishing."
        ),
        "n_samples": int(len(frame)),
        "columns": list(frame.columns),
        "target": TARGET_COLUMN,
        "target_units": "log10(GPa)",
        "target_task_type": metadata.get("task_type"),
        "fold_definitions": "official matbench 5-fold (task.validation[fold].train/test)",
        "fold_sizes": {
            key: {"n_train": len(val["train_idx"]), "n_test": len(val["test_idx"])}
            for key, val in fold_records.items()
        },
        "n_folds": len(fold_records),
        "index_sha256": sha256_text("\n".join(frame.index.tolist())),
        "cache_sha256": hashlib.sha256(CACHE_PATH.read_bytes()).hexdigest(),
        "citation": metadata.get("bibtex_refs", []),
        "retrieval_environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "note": "Downloaded through the official matbench.bench API.",
        },
    }
    write_json(PROVENANCE_PATH, provenance)
    logger.info("Cached %s samples; official folds: %s", len(frame), list(fold_records))
    return frame.copy(), provenance


def load_cached() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the immutable raw cache without touching the network."""
    if not CACHE_PATH.exists():
        raise FileNotFoundError(
            f"Raw cache missing at {CACHE_PATH}. Run `python scripts/run_benchmark.py --download-only`."
        )
    frame = pd.read_pickle(CACHE_PATH)
    provenance = read_json(PROVENANCE_PATH) if PROVENANCE_PATH.exists() else {}
    expected = provenance.get("index_sha256")
    if expected and sha256_text("\n".join(frame.index.astype(str).tolist())) != expected:
        raise RuntimeError("Cached dataset index checksum does not match provenance. Refusing to continue.")
    return frame.copy(), provenance


def load_fold_indices(frame: pd.DataFrame) -> dict[int, dict[str, list[int]]]:
    """Return official folds as positional indices.

    ``test_idx`` is the official held-out fold. It is reused as ``val_idx`` for
    out-of-fold prediction generation, which is the standard Matbench protocol:
    no separate validation split is carved out of the official training data.
    """
    if not FOLDS_PATH.exists():
        raise FileNotFoundError(f"Fold file missing: {FOLDS_PATH}")
    with gzip.open(FOLDS_PATH, "rt", encoding="utf-8") as handle:
        raw = json.load(handle)
    folds = {int(k): {"train_idx": list(v["train_idx"]), "test_idx": list(v["test_idx"])} for k, v in raw.items()}
    n = len(frame)
    for fold_id, fold in folds.items():
        for name in ("train_idx", "test_idx"):
            idx = fold[name]
            if min(idx) < 0 or max(idx) >= n:
                raise ValueError(f"Fold {fold_id} {name} contains out-of-range positions.")
        if set(fold["train_idx"]) & set(fold["test_idx"]):
            raise ValueError(f"Fold {fold_id} has overlapping train/test indices.")
    return folds


def get_matbench() -> tuple[pd.DataFrame, dict[str, Any], dict[int, dict[str, list[int]]]]:
    """Convenience accessor used by scripts and tests."""
    frame, provenance = download_and_cache(force=False)
    folds = load_fold_indices(frame)
    return frame, provenance, folds
