# Data card - matbench_log_gvrh

**Status:** generated from the actual cached artefact on 2026-10-28T17:35:54+00:00.
**Purpose:** document what this dataset is, what it is not, and how it was obtained.

## 1. Identity and provenance

| Field | Value |
| --- | --- |
| Dataset ID | `matbench_log_gvrh` |
| Benchmark | `matbench_v0.1` |
| Matbench package version | `0.6` |
| Retrieved (UTC) | `2026-10-28T17:34:56+00:00` |
| Source URL | https://ml.materialsproject.org/projects/matbench_log_gvrh.json.gz |
| Official hash (metadata) | `098af941f4c663270f1fe21abf20ffad6fb85ecbfcba5786ceac03983ac29da7` |
| Cached file SHA-256 | `ee6fcf8b0edf884a0db4d64b2c117ddc1b1da58035d16d379d2e2bacca172474` |
| Rows | 10987 |
| Target | `log10(G_VRH)` = log10(Voigt-Reuss-Hill shear modulus in GPa) |
| Target units | log10(GPa) |
| Rows checksum | `5a2b602efe695b21794ec98edc9581661973bf6a3d5b7bab0cb294662b355b1d` |

**License note.** Matbench v0.1 dataset (Dunn et al. 2020, npj Comput. Mater., doi:10.1038/s41524-020-00406-3). Underlying elastic data from de Jong et al. 2015 (Sci. Data, doi:10.1038/sdata.2015.9, CC BY 4.0). Benchmark code is MIT licensed. Confirm redistribution terms before publishing.

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
| Rows | 10987 |
| Target min | 0.0000 |
| Target max | 2.7185 |
| Target mean | 1.5536 |
| Target std | 0.3716 |
| Elements per material (min/max) | 1 / 5 |
| Duplicate structure keys | 0 groups (0 redundant rows) |
| Cross-fold duplicate groups | 0 |

Element-count distribution:

| Elements | Materials |
| --- | --- |
| 1 | 195 |
| 2 | 3742 |
| 3 | 6556 |
| 4 | 471 |
| 5 | 23 |

## 4. Official splits (locked, never regenerated)

| Fold | Train rows | Held-out rows |
| --- | --- | --- |
| 0 | 8789 | 2198 |
| 1 | 8789 | 2198 |
| 2 | 8790 | 2197 |
| 3 | 8790 | 2197 |
| 4 | 8790 | 2197 |

Manifest: `results/splits/split_manifest.json`. All checks passed:
`True`. The held-out fold is used for out-of-fold
predictions; no separate validation split is carved out of training data.

## 5. Validation checks

```json
{
  "required_columns_present": true,
  "index_unique": true,
  "index_non_null": true,
  "n_rows": 10987,
  "n_columns": 2,
  "target_numeric": true,
  "target_finite": true,
  "target_min": 0.0,
  "target_max": 2.718501688867274,
  "target_mean": 1.553638510911706,
  "target_std": 0.37161473108170545,
  "structures_nonempty": true,
  "n_elements_min": 1,
  "n_elements_max": 5,
  "identifier_columns_in_frame": [],
  "leakage_safe_columns": true,
  "failed_checks": [],
  "passed": true
}
```

## 6. Duplicate audit

Method: reduced formula + space group + volume rounded to 0.01 A^3.

Reported, not altered: Matbench official folds are immutable. Cross-fold duplicates, if any, can only inflate scores and must be stated as a benchmark limitation.

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
