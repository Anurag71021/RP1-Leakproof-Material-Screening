"""Dataset schema, dtype and integrity validation against persisted evidence."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PATHS, TARGET_COLUMN
from src.utils.io import read_json, sha256_file, sha256_text


def test_required_columns_and_no_extra_labels(benchmark_frame):
    assert {"structure", TARGET_COLUMN}.issubset(benchmark_frame.columns)
    assert len(benchmark_frame.columns) == 2, "Unexpected extra columns could carry leakage."


def test_index_is_unique_and_well_formed(benchmark_frame):
    assert benchmark_frame.index.is_unique
    assert benchmark_frame.index.notna().all()
    assert benchmark_frame.index.astype(str).str.startswith("mb-log-gvrh-").all()


def test_target_is_numeric_finite_and_plausible(benchmark_frame):
    target = pd.to_numeric(benchmark_frame[TARGET_COLUMN], errors="coerce")
    assert target.notna().all()
    assert np.isfinite(target.to_numpy(dtype=float)).all()
    # log10 shear modulus in GPa: plausible physical range for inorganic crystals.
    assert target.min() > -1.5
    assert target.max() < 4.0
    assert target.std() > 0


def test_target_matches_persisted_validation_report(benchmark_frame, data_validation):
    observed = data_validation["validation"]
    target = benchmark_frame[TARGET_COLUMN]
    assert observed["n_rows"] == len(benchmark_frame)
    assert observed["target_min"] == float(target.min())
    assert observed["target_max"] == float(target.max())
    assert observed["target_mean"] == float(target.mean())
    assert observed["passed"] is True
    assert observed["failed_checks"] == []


def test_structures_are_valid_and_nonempty(benchmark_frame):
    for structure in benchmark_frame["structure"].head(200):
        assert len(structure.sites) > 0
        assert len(structure.composition.elements) >= 1
        assert structure.volume > 0


def test_no_identifier_columns_are_present(benchmark_frame, data_validation):
    assert data_validation["validation"]["identifier_columns_in_frame"] == []
    forbidden = {"material_id", "mp_id", "mp-id", "icsd_id", "cif", "id"}
    assert not (forbidden & set(benchmark_frame.columns))


def test_duplicate_audit_is_clean_and_reported(data_validation):
    duplicates = data_validation["duplicates"]
    assert duplicates["n_rows"] == 10987
    assert duplicates["n_duplicate_groups"] == 0
    assert duplicates["n_cross_fold_duplicate_groups"] == 0
    assert "not altered" in duplicates["action"].lower()


def test_provenance_checksums_and_licence(provenance):
    cache = PATHS["data_raw"] / "matbench_log_gvrh.pkl.gz"
    assert cache.exists()
    assert provenance["cache_sha256"] == sha256_file(cache)
    assert provenance["dataset_id"] == "matbench_log_gvrh"
    assert provenance["n_folds"] == 5
    assert provenance["source_url"].startswith("https://")
    assert provenance["official_hash"]
    assert "matbench" in provenance["license"].lower()
    assert set(provenance["fold_sizes"]) == {"0", "1", "2", "3", "4"}


def test_index_checksum_matches_provenance(benchmark_frame, provenance):
    expected = sha256_text("\n".join(benchmark_frame.index.astype(str).tolist()))
    assert provenance["index_sha256"] == expected


def test_data_card_states_computed_not_clinical():
    text = (PATHS["data_cards"] / "matbench_log_gvrh_data_card.md").read_text(encoding="utf-8")
    lowered = text.lower()
    assert "computed" in lowered
    assert "not an experimental" in lowered or "not a clinical" in lowered
    assert "log10(g_vrh)" in lowered
    assert "out-of-scope" in lowered


def test_no_missing_target_or_structure(benchmark_frame):
    assert not benchmark_frame["structure"].isna().any()
    assert not benchmark_frame[TARGET_COLUMN].isna().any()
    assert len(benchmark_frame) == 10987
