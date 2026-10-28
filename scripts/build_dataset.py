#!/usr/bin/env python
"""Phase 2: cache the official dataset, validate it, lock folds, write data card."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from src.config import PATHS, TARGET_COLUMN
from src.data.loader import download_and_cache, load_fold_indices
from src.data.validate import build_split_manifest, duplicate_structure_audit, validate_matbench_frame
from src.utils.io import utc_now, write_json
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _data_card(frame: pd.DataFrame, provenance: dict, validation: dict, duplicates: dict, manifest: dict) -> str:
    target = frame[TARGET_COLUMN]
    elements = frame["structure"].map(lambda s: len(s.composition.elements))
    fold_sizes = provenance.get("fold_sizes", {})
    fold_lines = "\n".join(
        f"| {k} | {v['n_train']} | {v['n_test']} |" for k, v in sorted(fold_sizes.items())
    )
    return f"""# Data card - matbench_log_gvrh

**Status:** generated from the actual cached artefact on {utc_now()}.
**Purpose:** document what this dataset is, what it is not, and how it was obtained.

## 1. Identity and provenance

| Field | Value |
| --- | --- |
| Dataset ID | `{provenance.get('dataset_id')}` |
| Benchmark | `{provenance.get('benchmark_name')}` |
| Matbench package version | `{provenance.get('matbench_version')}` |
| Retrieved (UTC) | `{provenance.get('retrieved_utc')}` |
| Source URL | {provenance.get('source_url')} |
| Official hash (metadata) | `{provenance.get('official_hash')}` |
| Cached file SHA-256 | `{provenance.get('cache_sha256')}` |
| Rows | {provenance.get('n_samples')} |
| Target | `{TARGET_COLUMN}` = log10(Voigt-Reuss-Hill shear modulus in GPa) |
| Target units | log10(GPa) |
| Rows checksum | `{provenance.get('index_sha256')}` |

**License note.** {provenance.get('license')}

Primary references:

- Dunn, A., Wang, Q., Ganose, A., Dopp, D., Jain, A. *Benchmarking materials
  property prediction methods: the Matbench test set and Automatminer reference
  algorithm.* npj Computational Materials **6**, 138 (2020).
  doi:10.1038/s41524-020-00406-3
- de Jong, M. et al. *Charting the complete elastic properties of inorganic
  crystalline compounds.* Scientific Data **2**, 150009 (2015).
  doi:10.1038/sdata.2015.9

## 2. What the target actually is

`log10(G_VRH)` is a **computed** quantity derived from density-functional
theory elastic tensors in the Materials Project. It is **not** an experimental
measurement of an implant, not a fatigue limit, not a biocompatibility score,
and not a clinical outcome. A high predicted shear modulus means "the DFT
elastic model predicts a stiff crystal"; it says nothing about osseointegration,
corrosion, wear, or patient safety.

## 3. Composition of the dataset

| Property | Value |
| --- | --- |
| Rows | {len(frame)} |
| Target min | {validation['target_min']:.4f} |
| Target max | {validation['target_max']:.4f} |
| Target mean | {validation['target_mean']:.4f} |
| Target std | {validation['target_std']:.4f} |
| Elements per material (min/max) | {validation['n_elements_min']} / {validation['n_elements_max']} |
| Duplicate structure keys | {duplicates['n_duplicate_groups']} groups ({duplicates['n_duplicate_rows_beyond_first']} redundant rows) |
| Cross-fold duplicate groups | {duplicates['n_cross_fold_duplicate_groups']} |

Element-count distribution:

{_element_table(elements)}

## 4. Official splits (locked, never regenerated)

| Fold | Train rows | Held-out rows |
| --- | --- | --- |
{fold_lines}

Manifest: `results/splits/split_manifest.json`. All checks passed:
`{manifest['checks_passed']}`. The held-out fold is used for out-of-fold
predictions; no separate validation split is carved out of training data.

## 5. Validation checks

```json
{json.dumps(validation, indent=2)}
```

## 6. Duplicate audit

Method: {duplicates['method']}.

{duplicates['action']}

## 7. Intended use

- Benchmarking composition-based regression pipelines under official folds.
- Demonstrating leakage-aware model development and uncertainty reporting.
- Screening support for *ranking hypotheses*, never for clinical decisions.

## 8. Out-of-scope use

- Clinical or patient-specific advice.
- Predicting implant success, biocompatibility, corrosion, wear or fatigue.
- Treating the benchmark distribution as representative of orthopedic alloys
  (it is a broad inorganic materials set dominated by simple crystalline
  compounds).
- Extrapolating to surface-treated, additively manufactured, porous or
  coated devices without separate evidence.

## 9. Known limitations

1. The target is computed, not experimental.
2. The dataset is not an implant-alloy dataset; applicability-domain labels
   must accompany every prediction.
3. Duplicate structures, if any cross folds, can inflate benchmark scores.
4. Composition-only descriptors ignore processing, microstructure, texture,
   defects and surface state, all of which dominate real implant behaviour.
"""


def _element_table(elements: pd.Series) -> str:
    counts = elements.value_counts().sort_index()
    lines = ["| Elements | Materials |", "| --- | --- |"]
    lines += [f"| {int(k)} | {int(v)} |" for k, v in counts.items()]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-download", action="store_true", help="Re-download despite cache.")
    args = parser.parse_args()

    PATHS["data_cards"].mkdir(parents=True, exist_ok=True)
    frame, provenance = download_and_cache(force=args.force_download)
    folds = load_fold_indices(frame)

    logger.info("Validating schema and target.")
    validation = validate_matbench_frame(frame)

    logger.info("Auditing duplicate structures (coarse key + official fold assignment).")
    duplicates = duplicate_structure_audit(frame, folds)

    logger.info("Writing locked official-fold manifest.")
    manifest = build_split_manifest(frame, folds, PATHS["splits"] / "split_manifest.json")

    report = {
        "created_utc": utc_now(),
        "provenance": provenance,
        "validation": validation,
        "duplicates": duplicates,
        "manifest_checks": manifest["checks"],
        "manifest_checks_passed": manifest["checks_passed"],
    }
    write_json(PATHS["reports"] / "data_validation.json", report)

    card = _data_card(frame, provenance, validation, duplicates, manifest)
    (PATHS["data_cards"] / "matbench_log_gvrh_data_card.md").write_text(card, encoding="utf-8")

    print(json.dumps(
        {
            "rows": len(frame),
            "validation_passed": validation["passed"],
            "manifest_checks_passed": manifest["checks_passed"],
            "folds": {k: {"train": len(v["train_idx"]), "test": len(v["test_idx"])} for k, v in folds.items()},
            "duplicate_groups": duplicates["n_duplicate_groups"],
            "cross_fold_duplicate_groups": duplicates["n_cross_fold_duplicate_groups"],
            "data_card": "data_cards/matbench_log_gvrh_data_card.md",
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
