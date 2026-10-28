"""Schema, integrity and split validation.

These checks are the first line of defence for the leakage gate. They are
deliberately strict: unknown columns, non-finite targets or overlapping folds
raise instead of being silently tolerated.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from src.config import TARGET_COLUMN
from src.utils.io import sha256_text, utc_now, write_json

IDENTIFIER_COLUMNS = {
    "id",
    "material_id",
    "mp-id",
    "mp_id",
    "icsd_id",
    "structure_id",
    "cif",
    "formula_id",
}


def validate_matbench_frame(frame: pd.DataFrame) -> dict[str, Any]:
    """Validate required schema and return a machine-readable report."""
    checks: dict[str, Any] = {}
    checks["required_columns_present"] = bool({"structure", TARGET_COLUMN}.issubset(frame.columns))
    checks["index_unique"] = bool(frame.index.is_unique)
    checks["index_non_null"] = bool(frame.index.notna().all())
    checks["n_rows"] = int(len(frame))
    checks["n_columns"] = int(frame.shape[1])

    target = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    checks["target_numeric"] = bool(target.notna().all())
    checks["target_finite"] = bool(np.isfinite(target.to_numpy(dtype=float)).all())
    checks["target_min"] = float(target.min())
    checks["target_max"] = float(target.max())
    checks["target_mean"] = float(target.mean())
    checks["target_std"] = float(target.std())

    sites = frame["structure"].map(lambda s: len(getattr(s, "sites", [])))
    checks["structures_nonempty"] = bool((sites > 0).all())
    checks["n_elements_min"] = int(frame["structure"].map(lambda s: len(s.composition.elements)).min())
    checks["n_elements_max"] = int(frame["structure"].map(lambda s: len(s.composition.elements)).max())

    checks["identifier_columns_in_frame"] = sorted(IDENTIFIER_COLUMNS & set(frame.columns))
    checks["leakage_safe_columns"] = not checks["identifier_columns_in_frame"]

    failed = [
        key
        for key in (
            "required_columns_present",
            "index_unique",
            "index_non_null",
            "target_numeric",
            "target_finite",
            "structures_nonempty",
            "leakage_safe_columns",
        )
        if not checks[key]
    ]
    checks["failed_checks"] = failed
    checks["passed"] = not failed
    return checks


def structure_key(structure: Any) -> str:
    """Coarse, deterministic duplicate key for a pymatgen Structure."""
    symbol, number = structure.get_space_group_info()
    return f"{structure.composition.reduced_formula}|{symbol}{number}|{round(float(structure.volume), 2)}"


def duplicate_structure_audit(frame: pd.DataFrame, folds: dict[int, dict[str, list[int]]]) -> dict[str, Any]:
    """Find duplicate structure keys and whether they cross official folds.

    A cross-fold duplicate is a benchmark-level leakage risk. It is reported
    rather than "fixed", because official folds must not be altered.
    """
    keys = frame["structure"].map(structure_key)
    groups: dict[str, list[int]] = defaultdict(list)
    for position, key in enumerate(keys):
        groups[key].append(position)

    row_to_fold = {}
    for fold_id, fold in folds.items():
        for idx in fold["test_idx"]:
            row_to_fold[idx] = fold_id

    duplicate_groups = {k: v for k, v in groups.items() if len(v) > 1}
    cross_fold = []
    for key, positions in duplicate_groups.items():
        fold_ids = {row_to_fold[p] for p in positions}
        if len(fold_ids) > 1:
            cross_fold.append({"key": key, "positions": positions, "fold_ids": sorted(fold_ids)})

    n_duplicate_rows = int(sum(len(v) - 1 for v in duplicate_groups.values()))
    return {
        "n_rows": int(len(frame)),
        "n_unique_keys": int(len(groups)),
        "n_duplicate_groups": int(len(duplicate_groups)),
        "n_duplicate_rows_beyond_first": n_duplicate_rows,
        "n_cross_fold_duplicate_groups": int(len(cross_fold)),
        "cross_fold_examples": cross_fold[:10],
        "method": "reduced formula + space group + volume rounded to 0.01 A^3",
        "action": (
            "Reported, not altered: Matbench official folds are immutable. "
            "Cross-fold duplicates, if any, can only inflate scores and must be "
            "stated as a benchmark limitation."
        ),
    }


def build_split_manifest(
    frame: pd.DataFrame,
    folds: dict[int, dict[str, list[int]]],
    out_path,
    dataset: str = "matbench_log_gvrh",
    random_state: int = 42,
) -> dict[str, Any]:
    """Create the auditable official-fold manifest."""
    n = len(frame)
    fold_payload = []
    for fold_id in sorted(folds):
        train = folds[fold_id]["train_idx"]
        test = folds[fold_id]["test_idx"]
        fold_payload.append(
            {
                "fold_id": int(fold_id),
                "train_idx": train,
                "val_idx": test,
                "test_idx": test,
                "n_train": len(train),
                "n_val": len(test),
                "n_test": len(test),
                "val_equals_test": True,
                "val_equals_test_reason": (
                    "Matbench supplies one held-out fold per official split. "
                    "Out-of-fold predictions are generated on it; no extra "
                    "validation split is carved from training data."
                ),
            }
        )

    covered = sorted({i for f in folds.values() for i in f["test_idx"]})
    checks = {
        "all_official_folds_present": sorted(folds) == [0, 1, 2, 3, 4],
        "test_folds_partition_dataset": covered == list(range(n)),
        "no_train_test_overlap": all(
            not (set(f["train_idx"]) & set(f["test_idx"])) for f in folds.values()
        ),
        "train_plus_test_equals_n": all(
            len(f["train_idx"]) + len(f["test_idx"]) == n for f in folds.values()
        ),
    }

    payload = {
        "dataset": dataset,
        "split_type": "official_folds",
        "source": "matbench.bench task.validation[train/test]",
        "n_samples": int(n),
        "n_folds": len(folds),
        "group_key": None,
        "random_state": random_state,
        "target": TARGET_COLUMN,
        "created_utc": utc_now(),
        "index": {
            "type": "matbench_material_id",
            "sha256": sha256_text("\n".join(frame.index.astype(str).tolist())),
        },
        "folds": fold_payload,
        "checks": checks,
        "checks_passed": all(checks.values()),
    }
    write_json(out_path, payload)
    return payload
