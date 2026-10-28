# Research protocol - RP-1

**Version:** 1.0 (locked before modelling)
**Date:** 2026-10-28
**Primary benchmark:** `matbench_log_gvrh` (Matbench v0.1)
**Scope of surface evidence:** metallic orthopedic implant materials only

---

## 1. Purpose

Build and honestly evaluate a **decision-support screening demonstrator** that:

1. predicts a computed elastic descriptor (log10 shear modulus) from composition
   under the official Matbench protocol, with leakage controls and uncertainty;
2. curates published surface-response evidence for metallic orthopedic implants
   with full provenance;
3. presents criterion-level evidence and explicit unknowns instead of a single
   "implant suitability" score.

## 2. Mandatory distinctions

| Distinction | Rule |
| --- | --- |
| **Computed vs experimental** | `log10(G_VRH)` is a DFT-derived computed property. It is never described as an experimental modulus, an in-vivo outcome, or a clinical finding. |
| **Association vs causation** | Feature importance and model coefficients are associational descriptions of a fitted model. They do not establish physical or biological causation. |
| **Screening vs clinical validation** | Outputs support hypothesis ranking for research. They are not device qualification, not a clinical opinion, and not patient-specific advice. |
| **Missing vs zero** | "Not reported" is `NaN`. A measured zero remains `0`. No imputation in the literature layer. |
| **Evidence level** | Every value is tagged `observed_literature`, `literature_extracted`, `benchmark_predicted`, or `unknown`. Predicted values are never presented as observed. |

## 3. Research questions

**RQ1 (benchmark).** How accurately can composition-only descriptors predict
`log10(G_VRH)` under the official Matbench 5-fold protocol, compared with a
mean baseline?

**RQ2 (leakage).** Does the pipeline collapse to baseline performance when
training labels are randomized, and are all preprocessing steps fitted inside
the training fold?

**RQ3 (uncertainty and support).** Are split-conformal intervals calibrated on
held-out folds, and what fraction of benchmark samples fall inside, near, or
outside the descriptor-space applicability domain?

**RQ4 (evidence).** For metallic orthopedic implant surfaces, is the publicly
reported evidence sufficient — in study count, rows per study and endpoint
comparability — to train a study-aware surface-response model? If not, produce
a defensible curation-only evidence layer.

### Hypotheses

- **H1:** A Ridge model on Magpie composition descriptors will beat the mean
  baseline (lower MAE) on the official folds.
- **H2:** Random Forest will not beat Ridge by a large margin, because the
  composition-to-shear-modulus relationship is smooth and the dataset is
  moderate-sized.
- **H3:** Y-randomized models will have R² ≤ 0.10 and MAE no better than the
  Dummy baseline.
- **H4:** Public surface evidence for metallic orthopedic implants will be too
  heterogeneous for a defensible multi-study model, so Phase 7 will end in
  curation-only mode.

Hypotheses are stated before results and reported as supported / not supported
/ inconclusive in `docs/LIMITATIONS.md` and the final report.

## 4. Target and units

| Item | Value |
| --- | --- |
| Column | `log10(G_VRH)` |
| Meaning | Base-10 logarithm of the Voigt–Reuss–Hill average shear modulus |
| Units | log10(GPa) |
| Source | Matbench v0.1, derived from Materials Project DFT elastic tensors |
| Not | Experimental modulus, hardness, yield strength, fatigue limit, biocompatibility |

## 5. Datasets

### 5.1 Benchmark

- Official dataset and official folds via `matbench.bench`.
- Cached raw file: `data/raw/matbench_log_gvrh.pkl.gz` (immutable; gitignored).
- Provenance: `data/raw/matbench_log_gvrh_provenance.json` and data card.
- No custom split, no random re-shuffle, no stratification.

### 5.2 Surface evidence

- Inclusion: metallic orthopedic implant materials (Ti alloys, Co–Cr alloys,
  316L stainless steel, biodegradable Mg alloys used as orthopaedic implants).
- Exclusion: dental-only studies, soft-tissue scaffolds, drug-eluting systems,
  purely in-silico studies without measured surface values, and studies whose
  numerical values cannot be read from an accessible source.
- Only publicly accessible sources are used. Each row records DOI, URL, and the
  exact table/figure/section used.
- One locked endpoint for any modelling attempt: `surface_roughness_ra_um`.
  Ra is never merged with Rz/Rq, and different corrosion metrics or cell lines
  are never merged.

## 6. Features and leakage policy

1. **Composition-only** descriptors (Magpie, 132 features). Structure is not
   used, so the deployment path for a hypothetical alloy is identical to the
   benchmark path.
2. The featurizer is **stateless**. Imputation and scaling are fitted inside
   each training fold.
3. Identifiers (`material_id`, index, `formula` as an ID) are never features.
4. No EDA-derived feature selection is performed before the gate. No target
   encoding, no global statistics.
5. For literature data (if modelled), `GroupKFold`/`LeaveOneGroupOut` by
   `study_id` only. Random splits are forbidden.

## 7. Splits

- Benchmark: 5 official Matbench folds. Each sample is held out exactly once.
- The official held-out fold is used as the validation fold for OOF
  predictions. No extra validation split is carved from training data.
- Manifest: `results/splits/split_manifest.json` with indices, sizes and checks.

## 8. Models and baselines

Mandatory: `DummyRegressor(mean)`, `Ridge(alpha=1.0)`, `RandomForestRegressor`
(300 trees, `max_features=0.33`). All wrapped in the same pipeline:
`composition featurizer → median imputer → standard scaler → estimator`.

No hyperparameter tuning on held-out folds. `alpha=1.0` and RF settings are
fixed a priori. If tuning is added later it must be nested inside each training
fold and reported.

## 9. Metrics

- MAE (primary, interpretable in log10 GPa)
- RMSE (penalises large errors)
- R² (variance explained; may be negative)
- Reported per fold and as mean ± sample standard deviation.
- Comparison against Dummy is mandatory.

## 10. Validation and gate

| Check | Pass condition |
| --- | --- |
| Official folds | Manifest says `official_folds`, checks all true |
| No identifier features | Runtime feature-name assertion + test |
| Per-fold pipeline | Code structure and stateless-featurizer test |
| OOF coverage | Every sample predicted once |
| Y-randomization | Mean R² ≤ 0.10 and MAE not better than Dummy |
| Learning curves | Present; gap described |
| Residuals / parity | Present; patterns described |
| Duplicate audit | Performed and reported |
| Leakage tests | Pass |

If any critical check fails, `LEAKAGE_GATE: FAILED` and Phases 5–9 are blocked.

## 11. Uncertainty and applicability domain

- Split conformal intervals from OOF absolute residuals, with leave-one-fold-out
  calibration for coverage checks.
- AD: mean distance to 5 nearest training neighbours in standardized descriptor
  space; thresholds at the 95th and 99.5th percentiles of training distances.
- Labels: `IN_DOMAIN`, `NEAR_BOUNDARY`, `OUT_OF_DOMAIN`.
- No statement of "probability of implant success".

## 12. Surface modelling gate (Phase 7)

Train only if **all** hold:

- (a) ≥ 3 distinct studies (target ≥ 4);
- (b) ≥ 25 deduplicated, compatible rows for one locked endpoint;
- (c) no single study contributes > 40% of rows;
- (d) a study-aware split is feasible (≥ 3 groups);
- (e) Phase 4 gate is PASSED.

Otherwise: curation-only, documented explicitly. Curation-only is a successful
completion, not a failure to be hidden.

## 13. Screening rules

- Four separate source pools: observed literature, benchmark predictions, other
  experimental literature, user constraints.
- Criterion-level matrix only. Columns include value, units, source, evidence
  level, applicability, and constraint status.
- Any critical criterion that is unknown with no comparable evidence makes the
  candidate `INDETERMINATE`. Indeterminate candidates are never silently ranked.
- No composite percentage score. Ordering is lexicographic by user-selected
  criterion priority; sensitivity to priority order is reported.
- Every row traces to a DOI/study_id, an OOF file, or a provenance record.

## 14. Acceptance criteria

1. All Phase 4 gate checks pass with linked evidence.
2. All five required test suites pass.
3. `python -c "import app"` prints `APP_IMPORT_OK`.
4. Every displayed number is read from a persisted artefact.
5. Documentation states computed vs experimental, association vs causation, and
   screening vs clinical validation.

## 15. Deviations

Any deviation from this protocol is recorded in `docs/PROJECT_LOG.md` with the
reason and its effect. The protocol is not silently edited after seeing results.
