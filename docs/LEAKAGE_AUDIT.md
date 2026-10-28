# Leakage and overfitting audit

**Generated:** 2026-10-28T18:00:08+00:00 by `scripts/finalize_gate.py`.
**Selected model:** `random_forest` (lowest mean OOF MAE).
**Gate status:** `LEAKAGE_GATE: PASSED`

> This gate is evidence-based. A check is YES only if the named artefact
> was read and the stated condition held at generation time.

## 1. Critical checklist

| # | Check | Result | Evidence |
| --- | --- | --- | --- |
| 1 | `official_folds_used` | YES | results/splits/split_manifest.json (split_type=official_folds, 5 folds) |
| 2 | `split_integrity` | YES | manifest checks: {'all_official_folds_present': True, 'test_folds_partition_dataset': True, 'no_train_test_overlap': True, 'train_plus_test_equals_n': True} |
| 3 | `identifiers_excluded` | YES | runtime feature names contain no identifiers; hits=[]; frame has no identifier columns |
| 4 | `pipeline_only_preprocessing` | YES | Pipeline(features -> impute -> scale -> estimator); train.py fits on fold-local train slice only; no global fit_transform |
| 5 | `per_fold_featurization` | YES | same compositions featurized after fitting on disjoint subsets produce identical values (stateless featurizer, 132 Magpie descriptors) |
| 6 | `oof_coverage` | YES | OOF files cover every sample exactly once: {'dummy': 'ok', 'ridge': 'ok', 'random_forest': 'ok'} |
| 7 | `y_randomization_collapse` | YES | best=random_forest; randomized R2 max=-0.1116 (<=0.10 required); randomized MAE=0.3093 vs Dummy MAE=0.2931 |
| 8 | `model_beats_dummy` | YES | best=random_forest; MAE 0.1024 vs Dummy 0.2931; R2 0.8325 vs Dummy -0.0002 |
| 9 | `learning_curves_present` | YES | learning_curves.json; model=random_forest |
| 10 | `residuals_parity_present` | YES | results/plots/parity_plot_random_forest.png; results/plots/residuals_random_forest.png |
| 11 | `duplicates_audited` | YES | 0 duplicate groups, 0 cross-fold |
| 12 | `leakage_tests_present` | YES | tests/test_leakage_sanity.py exists (executed in Phase 10) |

## 2. Benchmark results (official 5-fold OOF)

| Model | MAE | RMSE | R2 |
| --- | --- | --- | --- |
| dummy | 0.2931 ± 0.0035 | 0.3716 ± 0.0042 | -0.0002 ± 0.0002 |
| ridge | 0.1568 ± 0.0011 | 0.2144 ± 0.0028 | 0.6671 ± 0.0069 |
| random_forest | 0.1024 ± 0.0016 | 0.1520 ± 0.0046 | 0.8325 ± 0.0100 |

## 3. Y-randomization

| Model | Permutations | Randomized R2 mean | Randomized R2 max | Randomized MAE |
| --- | --- | --- | --- | --- |
| ridge | 5 | -0.0145 | -0.0068 | 0.2952 |
| random_forest | 2 | -0.1155 | -0.1116 | 0.3093 |

A valid model must collapse to Dummy-level performance when labels are
shuffled. The real model must remain clearly better than Dummy.

## 4. Overfitting / stability diagnostics

- Fold-wise R2 for `random_forest`: 0.8453, 0.8256, 0.8375, 0.8345, 0.8197
- Fold R2 standard deviation: 0.0100
- Learning curve at full training size: full-size train MAE=0.0443, held-out MAE=0.1019, gap=0.0575
- Residual diagnostics: `residual_stats_random_forest.json`

A train/held-out gap is expected for a 300-tree Random Forest. It is
reported, not hidden. The decisive leakage evidence is the Y-randomization
collapse, not the size of the training gap alone.

## 5. Duplicate and split audit

- Duplicate groups: 0
- Cross-fold duplicate groups: 0
- Duplicate method: reduced formula + space group + volume rounded to 0.01 A^3
- Split checks: {"all_official_folds_present": true, "test_folds_partition_dataset": true, "no_train_test_overlap": true, "train_plus_test_equals_n": true}

## 6. Limitations that remain true even after a pass

1. Passing this gate proves the *implementation* is leak-aware under the
   checks performed. It does not prove the benchmark target is clinically
   meaningful.
2. `log10(G_VRH)` is a computed DFT property. It is not an implant modulus,
   a fatigue limit, or a biocompatibility measure.
3. Cross-fold duplicate structures, if present, would inflate scores. The
   audit reports zero for the coarse key used; near-duplicate polymorphs
   cannot be fully excluded by this key.
4. Official folds are random with respect to composition, so the benchmark
   measures interpolation, not extrapolation to new alloy families.

## 7. Gate rule

Phases 5-9 may proceed **only** if every critical check above is YES and
Y-randomization is convincingly worse than the real model. The final line
of this file is the machine-readable gate marker.

LEAKAGE_GATE: PASSED
