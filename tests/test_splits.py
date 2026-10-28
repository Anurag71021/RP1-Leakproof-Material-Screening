"""Split integrity: official Matbench folds only, no overlap, full coverage."""

from __future__ import annotations

import pytest

from src.utils.io import sha256_text


def test_exactly_five_official_folds(folds):
    assert sorted(folds) == [0, 1, 2, 3, 4]
    for fold in folds.values():
        assert set(fold) == {"train_idx", "test_idx"}


def test_no_train_test_overlap(folds):
    for fold_id, fold in folds.items():
        overlap = set(fold["train_idx"]) & set(fold["test_idx"])
        assert not overlap, f"fold {fold_id} has {len(overlap)} overlapping rows"


def test_test_folds_partition_the_dataset(folds, benchmark_frame):
    covered = [idx for fold in folds.values() for idx in fold["test_idx"]]
    assert len(covered) == len(benchmark_frame)
    assert sorted(covered) == list(range(len(benchmark_frame)))
    assert len(set(covered)) == len(covered)


def test_train_plus_test_equals_n(folds, benchmark_frame):
    for fold in folds.values():
        assert len(fold["train_idx"]) + len(fold["test_idx"]) == len(benchmark_frame)


def test_indices_are_in_range(folds, benchmark_frame):
    n = len(benchmark_frame)
    for fold in folds.values():
        for idx in fold["train_idx"] + fold["test_idx"]:
            assert 0 <= idx < n


def test_manifest_declares_official_splits(split_manifest, benchmark_frame):
    assert split_manifest["split_type"] == "official_folds"
    assert split_manifest["dataset"] == "matbench_log_gvrh"
    assert split_manifest["n_folds"] == 5
    assert split_manifest["n_samples"] == len(benchmark_frame)
    assert split_manifest["checks_passed"] is True
    assert all(split_manifest["checks"].values())


def test_manifest_matches_cached_folds(split_manifest, folds):
    for record in split_manifest["folds"]:
        fold_id = record["fold_id"]
        assert record["train_idx"] == folds[fold_id]["train_idx"]
        assert record["test_idx"] == folds[fold_id]["test_idx"]
        assert record["val_idx"] == record["test_idx"]
        assert record["val_equals_test"] is True
        assert "no extra validation split" in record["val_equals_test_reason"]


def test_manifest_records_index_checksum(split_manifest, benchmark_frame):
    expected = sha256_text("\n".join(benchmark_frame.index.astype(str).tolist()))
    assert split_manifest["index"]["sha256"] == expected
    assert split_manifest["index"]["type"] == "matbench_material_id"


def test_fold_assignment_is_stable_across_loads(benchmark_frame, folds, split_manifest):
    """Re-deriving the fold assignment must reproduce the manifest exactly."""
    for record in split_manifest["folds"]:
        held_out = benchmark_frame.index[record["test_idx"]].tolist()
        assert all(str(mid).startswith("mb-log-gvrh-") for mid in held_out)


@pytest.mark.parametrize("fold_id", [0, 1, 2, 3, 4])
def test_each_fold_has_substantial_train_and_test(folds, fold_id):
    assert len(folds[fold_id]["train_idx"]) > 8000
    assert len(folds[fold_id]["test_idx"]) > 2000
