## RP-1 FINAL REPORT – LEAK-PROOF, EVIDENCE-AWARE IMPLEMENTATION

### 1) Repository Audit

- **Root:** `/Users/anurag/Desktop/Projects_RP`
- **Git:** yes — repository initialised at the end of the build with a single commit. Tracked: source, scripts, tests, docs, configs, curated CSV evidence, results and reports. Excluded by `.gitignore`: `.venv/`, raw dataset cache, joblib model binaries, third-party full-text XML (`data/literature/sources/`), and the unrelated `.mattpocock-skills/` repository.
- **Python/.venv:** Python 3.10.18, used from `.venv` for every command.
- **Existing Matbench loader/tests found:** no. No `src/`, `tests/`, `app.py` or loader existed.
- **Preserved:** `RP1_Leakproof_Implementation_Plan.md` (unmodified), `opencode.jsonc`.
- **Modified:** none of the pre-existing files.
- **Created:** `src/` (config, UI, data, features, models, evaluation, interpretability, screening, utils), `scripts/` (10 pipeline scripts + 2 probes), `pages/` (8 pages), `app.py`, `tests/` (5 required suites + `conftest.py`), `configs/` (2 YAML), `docs/` (12 documents), `data_cards/` (2), `results/` (splits, metrics, predictions, validation, uncertainty, interpretability, screening, plots, tables), `reports/`, `README.md`, `pyproject.toml`, `requirements.txt`, `.gitignore`.

### 2) Environment

- **OS/Arch:** macOS 26.6.1 (build 25G76), arm64 (Apple Silicon).
- **Dependencies installed:** numpy 1.26.4, pandas 2.2.3, scikit-learn 1.7.2, scipy 1.15.3, matminer 0.9.3, pymatgen 2025.10.7, matbench 0.6, streamlit 1.65.0, pytest 9.2.0, joblib 1.6.0.
- **Network to Matbench:** OK. Dataset and official folds were loaded through the official `matbench.bench` API.
- **Approval requested for large installs/downloads:** no. The user instructed the build; installs were local into `.venv` and the dataset is cached locally. `matbench==0.6` was installed with `--no-deps` because its pinned `matminer==0.7.4` has no Python 3.10 arm64 build. This is a packaging deviation only; the data and folds remain official and are recorded in `docs/PROJECT_LOG.md`.

### 3) Benchmark (matbench_log_gvrh)

- **Official folds used:** yes — exactly 5 official Matbench folds; held-out fold = validation fold.
- **split_manifest.json:** `results/splits/split_manifest.json` — exists, `split_type: official_folds`, `n_folds: 5`, `n_samples: 10987`, all checks passed.
- **Featurization:** composition-only Magpie descriptors (132 features). The featurizer is stateless; it is fitted inside each pipeline fold, never on the full dataset.
- **Pipelines used:** yes — `features → impute → scale → model` for Ridge; `features → impute → model` for Random Forest and Dummy.
- **Models:** Dummy (mean), Ridge (α = 1.0), Random Forest (300 trees, `max_features=0.33`).
- **OOF saved:**
  - `results/predictions/oof_dummy_matbench_log_gvrh.csv`
  - `results/predictions/oof_ridge_matbench_log_gvrh.csv`
  - `results/predictions/oof_random_forest_matbench_log_gvrh.csv`
- **Metrics (mean ± sample SD across 5 official folds):**

| Model | MAE (log10 GPa) | RMSE | R² |
| --- | --- | --- | --- |
| Dummy | 0.2931 ± 0.0035 | 0.3716 ± 0.0042 | -0.0002 ± 0.0002 |
| Ridge | 0.1568 ± 0.0011 | 0.2144 ± 0.0028 | 0.6671 ± 0.0069 |
| **Random Forest** | **0.1024 ± 0.0016** | **0.1520 ± 0.0046** | **0.8325 ± 0.0100** |

- **Dummy comparison:** beaten, yes. Random Forest MAE 0.1024 vs Dummy 0.2931; R² 0.8325 vs -0.0002. Selected model by lowest mean OOF MAE before any screening or literature analysis: `random_forest`.

### 4) Leakage & Overfitting Gate

- **LEAKAGE_AUDIT.md:** present and filled — `docs/LEAKAGE_AUDIT.md`; signed `LEAKAGE_GATE: PASSED` at the end.
- **Y-randomization run:** done — `results/validation/y_randomization.json`, log `reports/y_randomization_run.log`. Ridge (5 permutations) R² mean -0.0145, max -0.0068, MAE 0.2952. Random Forest (2 permutations) R² mean -0.1155, max -0.1116, MAE 0.3093 — both collapse to or below Dummy performance (Dummy MAE 0.2931).
- **Learning curves/residuals/parity:** present — `results/validation/learning_curves.json`, `results/validation/residual_stats_random_forest.json`, `results/plots/parity_plot_random_forest.png`, `results/plots/residuals_random_forest.png`, `results/plots/learning_curves_random_forest.png`. Residual mean 0.0006, SD 0.1521, |residual| p95 0.3185.
- **Leakage tests:** passed — `tests/test_leakage_sanity.py` and the full suite (75 passed).
- **LEAKAGE_GATE:** **PASSED** — 12/12 checks true in `results/validation/gate_status.json`.
- **Blocked phases until passed:** stated, yes. `scripts/run_analysis.py` and `scripts/build_screening.py` refuse to run without the gate marker; Phases 5–9 were not run before it passed.

### 5) Interpretability, Uncertainty, AD

- **Permutation importance:** done — 5 official folds, 5 repeats, up to 1,500 held-out rows per fold; `results/interpretability/permutation_importance_random_forest.csv` / `.json`.
- **SHAP:** skipped — `shap` is not installed. No SHAP values are reported or implied; stated in `docs/INTERPRETABILITY_LIMITS.md` and the app.
- **Uncertainty method:** split conformal prediction on out-of-fold absolute residuals.
- **Calibration done:** yes — empirical coverage 0.8001 / 0.9001 / 0.9502 at nominal 0.80 / 0.90 / 0.95; deployment half-widths 0.1570 / 0.2355 / 0.3186 log10(GPa).
- **AD implemented:** yes — k-nearest-neighbour distance in standardized Magpie space with 95th / 99.5th percentile thresholds. OOF: 10,433 `IN_DOMAIN`, 511 `NEAR_BOUNDARY`, 43 `OUT_OF_DOMAIN`. Deployment reference: 10,870 / 106 / 11.
- **INTERPRETABILITY_LIMITS.md present:** yes.

### 6) Surface Evidence Layer

- **Source:** publicly accessible, open-access papers only, retrieved by PMCID from Europe PMC full text; DOI, PMCID and extraction location recorded per row.
- **study_id assigned:** yes — one per DOI/article.
- **Raw/Deduplicated:** 40 raw rows → 40 clean rows (0 duplicates removed on the documented key `(study_id, material_grade_or_formula, surface_treatment, endpoint_key, conditions_key)`).
- **Distinct studies:** 5 included; 2 dental-only studies excluded with recorded reasons.
- **Rows per study (all endpoints):** min 3, median 6, max 20; largest study 50.0% of all rows.
- **Locked endpoint `surface_roughness_ra_um`:** 23 rows, 2 studies, largest study 87%.
- **Unit harmonization:** conservative. Ecorr converted mV → V only when the reference electrode is stated; Sa/Rz/Rq were **not** merged into Ra; water and saline contact angles were not pooled.
- **SURFACE_EVIDENCE_AUDIT.md + data card:** present — `docs/SURFACE_EVIDENCE_AUDIT.md`, `docs/SURFACE_EVIDENCE_CURATED.md`, `data_cards/surface_evidence_data_card.md`.
- **Modelling:** **CURATION-ONLY**. No surface-response model was trained and no model artefact was written.
- **If trained:** not applicable.
- **If curation-only, reason stated (exact text):**
  - `only 2 distinct studies for surface_roughness_ra_um (< 3)`
  - `only 23 deduplicated rows for surface_roughness_ra_um (< 25)`
  - `largest study contributes 87% of rows (> 40%)`
  - `fewer than 3 groups; LOGO/GroupKFold would not be meaningful`

  The gate was re-evaluated from the persisted table inside `scripts/train_surface_model.py`; it was not inherited from an earlier run.

### 7) Screening & App

- **Screening engine:** criterion matrix plus traceability and `INDETERMINATE` — `results/screening/criterion_matrix.csv` (long format; each row carries source, evidence level, units, conditions and critical flag), `candidate_summary.csv`, `ranking.csv`, `sensitivity.json`. 14 candidates: 5 `RANKED`, 1 `CONSTRAINT_FAILED`, 8 `INDETERMINATE`.
- **Guardrails present:** `tests/test_screening_guardrails.py` — composite-score columns rejected, missing critical criterion forces `INDETERMINATE`, lexicographic ranking, evidence-link traceability, non-critical failures visible but non-gating.
- **Streamlit app import:** `python -c "import app"` → **OK** (`APP_IMPORT_OK`).
- **Pages present:**
  - `01_Project_Overview.py`
  - `02_Dataset_Provenance_Quality.py`
  - `03_Benchmark_Results_log_gvrh.py`
  - `04_Prediction_AD_Uncertainty.py`
  - `05_Interpretability_Leakage_Summary.py`
  - `06_Surface_Evidence_Layer.py`
  - `07_Transparent_Candidate_Screening.py`
  - `08_Limitations_Disclaimers.py`
- **Persistent disclaimer:** yes — in `app.py`, on every page, and in exported artefacts.
- **AD/uncertainty shown on prediction:** yes — Page 04 shows the 90% conformal interval and `IN_DOMAIN` / `NEAR_BOUNDARY` / `OUT_OF_DOMAIN`, with explicit warnings for out-of-domain input.

### 8) Tests & Reproducibility

- **pytest -v command:** `.venv/bin/python -m pytest -v 2>&1 | tee reports/pytest_run_$(date -u +%Y%m%d_%H%M%S).log`
- **Results:** **75 passed, 0 failed, 0 skipped** (4.89 s).
- **Log file:** `reports/pytest_run_20261028_181218.log`.
- **README matches actual commands:** yes — `README.md` lists the same script order and install workaround.
- **PROJECT_LOG.md updated with real actions:** yes — `docs/PROJECT_LOG.md` records commands, timings, results and deviations (matbench `--no-deps`, early log timezone bug, AI-assisted extraction, no SHAP, one-fold learning curves, Git initialised at the end).
- **configs/ present:** yes — `configs/benchmark.yaml`, `configs/featurization.yaml`.

### 9) Remaining Blockers & Next Recommended Task

- **Blockers:** None for the implemented scope. Two explicit caveats are not blockers but must travel with the results: (1) literature extraction is AI-assisted and needs independent human verification against the PDFs; (2) Phase 7 is intentionally `CURATION_ONLY`.
- **Recommended next task:** independently verify all 40 curated literature rows against the source PDFs/tables, then record the verification outcome in `docs/SURFACE_EVIDENCE_AUDIT.md`. This is the single action that most improves the evidence layer without changing scope.
- **Scientific caution note:** `log10(G_VRH)` is a **computed DFT** shear modulus, not an experimental implant modulus. Screening is a research hypothesis-ranking exercise under user-supplied constraints, **not** clinical validation, biocompatibility assessment, or an implant recommendation. `RANKED` means "all critical criteria are known and meet the stated constraints"; it does not mean safe, effective or clinically preferable.
