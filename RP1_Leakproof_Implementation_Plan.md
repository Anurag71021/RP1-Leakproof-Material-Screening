# RP-1: ML-Based, Evidence-Aware Material and Surface Screening for Orthopedic Implant Applications

**Version:** 1.0 (Final Executable Plan)  
**Date:** October 28, 2026  
**Target:** B.Tech Final Year Research Project  
**Primary Benchmark:** `matbench_log_gvrh` (log-transformed Voigt–Reuss–Hill shear modulus)  
**Scope:** Leak-proof reproduction, provenance-preserving surface evidence curation, transparent screening (decision-support only)

---

## Table of Contents

- [1. Core Principles](#1-core-principles)
- [2. Tech Stack](#2-tech-stack)
- [3. Project Structure](#3-project-structure)
- [4. Phase 0 - Repository Audit (Blocker)](#4-phase-0---repository-audit-blocker)
- [5. Phase 1 - Research Specification (Blocker)](#5-phase-1---research-specification-blocker)
- [6. Phase 2 - Data Ingestion, Provenance & Caching (Blocker)](#6-phase-2---data-ingestion-provenance--caching-blocker)
- [7. Phase 3 - Leak-Proof Feature Engineering & Baseline Models](#7-phase-3---leak-proof-feature-engineering--baseline-models)
- [8. Phase 4 - Rigorous Validation & Leakage & Overfitting Gate (HARD BLOCKER)](#8-phase-4---rigorous-validation--leakage--overfitting-gate-hard-blocker)
- [9. Phase 5 - Model Interpretability, Uncertainty, Applicability Domain & Calibration](#9-phase-5---model-interpretability-uncertainty-applicability-domain--calibration)
- [10. Phase 6 - Biomedical Evidence Layer (Curation-First, Study-Aware)](#10-phase-6---biomedical-evidence-layer-curation-first-study-aware)
- [11. Phase 7 - Surface-Response Models: Only If Strictly Justified (GATED)](#11-phase-7---surface-response-models-only-if-strictly-justified-gated)
- [12. Phase 8 - Transparent Candidate Screening Engine](#12-phase-8---transparent-candidate-screening-engine)
- [13. Phase 9 - Streamlit Application (Integrated & Honest)](#13-phase-9---streamlit-application-integrated--honest)
- [14. Phase 10 - Reproducibility, Software Quality, Tests & Final Documentation](#14-phase-10---reproducibility-software-quality-tests--final-documentation)
- [15. Evaluation Metrics](#15-evaluation-metrics)
- [16. Testing Strategy](#16-testing-strategy)
- [17. VS Code Execution Workflow](#17-vs-code-execution-workflow)
- [18. Start Sequence (Execute in Exact Order)](#18-start-sequence-execute-in-exact-order)
- [19. Mandatory Final Report Template](#19-mandatory-final-report-template)
- [20. Final Instructions](#20-final-instructions)

---

## 1. Core Principles

| Principle | Rationale |
| --- | --- |
| **Inspect > Assume** | Audit the repository first. Do not rewrite working code blindly. |
| **Preserve > Replace** | If an existing Matbench experimental band-gap loader or its tests are present, **preserve and extend** them. Do not swap it out for a different benchmark. |
| **Prove > Claim** | Every result must be backed by actual command outputs, saved files, logs and metrics. |
| **Computed vs Experimental** | Treat `log_gvrh` as a *computed* property from Matbench. Never equate it to an *in-vivo* implant performance or clinical outcome. |
| **Evidence > Inference** | Distinguish between *observed*, *literature-extracted*, *benchmark-predicted* and *unknown*. Do not infer biological compatibility from a single parameter. |
| **Conservative > Overconfident** | If data is insufficient or validation is weak, choose **curation-only** over manufacturing a weak model. |
| **Leakage-Aware by Design** | Prevention > post-hoc justification. **Enforcement > awareness.** |
| **Screening vs Clinical Validation** | This is decision-support only. No patient-specific advice, no implant recommendation implied. |

---

## 2. Tech Stack

### Core ML/Data Science
| Package | Purpose | Notes |
| --- | --- | --- |
| `Python 3.10+` | Base language | Prefer `3.10.x` for compatibility with matminer/matbench on Apple Silicon |
| `pandas` | Data manipulation | Core dataframes for all datasets |
| `numpy` | Numerical operations | Arrays, statistics |
| `scikit-learn` | ML pipelines, baselines, CV, metrics | **Pipeline-first** discipline enforced |
| `scipy` | Statistical utilities | Distributions, tests |

### Materials Informatics
| Package | Purpose | Notes |
| --- | --- | --- |
| `matbench` | Official benchmark datasets & folds | **Must use official fold definitions** |
| `matminer` | Featurization (composition/structure) | Compute **per-fold** only, never on full dataset pre-split |
| `pymatgen` | Structure/composition handling | Dependency of matminer, handle carefully on Apple Silicon |

### Explainability & Uncertainty
| Package | Purpose | Notes |
| --- | --- | --- |
| `shap` | Feature attribution | Optional - skip if installation unstable; use permutation importance as fallback |
| `mapie` | Conformal prediction for uncertainty | Optional - use bootstrap ensembles as fallback if not available |

### Visualization & UI
| Package | Purpose | Notes |
| --- | --- | --- |
| `matplotlib` | Static plots | Parity, residuals, learning curves |
| `seaborn` | Statistical visualization | Enhanced plots |
| `streamlit` | Web app demonstrator | Honest, data-driven UI with disclaimers |

### Development & Testing
| Package | Purpose | Notes |
| --- | --- | --- |
| `pytest` | Unit/integration tests | Enforced leakage sanity tests |
| `black`/`ruff` (optional) | Linting/formatting | Keep consistent with existing style |
| `jupyter` (optional) | Exploratory analysis | Only for post-gate inspection, never for preprocessing decisions |

### Environment Management
- **Virtual Environment:** `venv` (`.venv/`) - activate with `source .venv/bin/activate` (macOS/zsh)
- **Package Management:** `pip` with `requirements.txt`
- **Git Hygiene:** `.gitignore` must exclude `.venv/`, `data/raw/*`, `artifacts/`, `*.pkl`, `*.joblib` (large models), `.env`

---

## 3. Project Structure

```
RP-1/
├── app.py                          # Main Streamlit entry point with persistent disclaimer
├── configs/                        # YAML configs (benchmark, featurization)
│   ├── benchmark.yaml
│   └── featurization.yaml
├── data/
│   ├── raw/                        # Untouched cached raw data (gitignored)
│   ├── processed/                  # Processed artifacts (if any)
│   └── literature/                 # Curated surface evidence
│       ├── surface_evidence_raw.csv
│       └── surface_evidence_clean.csv
├── data_cards/                     # Data cards for transparency
│   ├── matbench_log_gvrh_data_card.md
│   └── surface_evidence_data_card.md
├── docs/                           # All documentation (audit, protocol, reports)
│   ├── REPOSITORY_AUDIT.md
│   ├── IMPLEMENTATION_PLAN.md
│   ├── RESEARCH_PROTOCOL.md
│   ├── DATA_DICTIONARY.md
│   ├── LEAKAGE_AUDIT.md           # HARD GATE FILE
│   ├── SURFACE_EVIDENCE_AUDIT.md
│   ├── INTERPRETABILITY_LIMITS.md
│   ├── LITERATURE_REVIEW.md
│   ├── RESEARCH_GAPS.md
│   ├── LIMITATIONS.md
│   └── PROJECT_LOG.md
├── pages/                          # Streamlit multipage app
│   ├── 01_Project_Overview.py
│   ├── 02_Dataset_Provenance_Quality.py
│   ├── 03_Benchmark_Results_log_gvrh.py
│   ├── 04_Prediction_AD_Uncertainty.py
│   ├── 05_Interpretability_Leakage_Summary.py
│   ├── 06_Surface_Evidence_Layer.py
│   ├── 07_Transparent_Candidate_Screening.py
│   └── 08_Limitations_Disclaimers.py
├── reports/                        # Test logs, auto-generated reports
├── results/                        # All results (metrics, predictions, plots)
│   ├── splits/
│   │   ├── split_manifest.json
│   │   └── surface_split_manifest.json
│   ├── predictions/
│   ├── metrics/
│   ├── plots/
│   ├── interpretability/
│   ├── uncertainty/
│   ├── validation/
│   │   └── y_randomization.json
│   └── screening/
├── scripts/                        # Utility scripts
│   └── y_randomize.py
├── src/                            # Source code (package)
│   ├── __init__.py
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   ├── interpretability/
│   ├── screening/
│   └── utils/
└── tests/                          # Test suite
    ├── test_leakage_sanity.py
    ├── test_splits.py
    ├── test_data_validation.py
    ├── test_app_imports.py
    └── test_screening_guardrails.py
```

---

## 4. Phase 0 - Repository Audit (Blocker)

*Duration: Discovery. No heavy downloads or model training.*

| Step | Task | Verification Criteria |
| --- | --- | --- |
| **0.1** | **Environment Scan** | Run `pwd`, `ls -la`, `git status --short --branch`, `python3 --version`, `which python3`, `pip --version`. Detect OS, shell, `.venv`/conda. |
| **0.2** | **Dependency Mapping** | Inspect `requirements.txt`, `pyproject.toml`, `environment.yml`, `Pipfile`. Run `pip list` and log. |
| **0.3** | **Codebase Mapping** | Trace `src/`, `app/`, `pages/`, `tests/`, `data/`, `configs/`. Identify existing Matbench loader, featurization, tests. |
| **0.4** | **Resource & Network Check** | Check free disk space/RAM. Safe network probe to Matbench. Log results. |
| **0.5** | **Approval Gate** | Before large downloads, new external services, destructive edits, or expensive computation -> **state need and ask for approval**. |
| **0.6** | **Generate Audit** | Create `docs/REPOSITORY_AUDIT.md` and `docs/IMPLEMENTATION_PLAN.md` with: repo state, keep/modify/create, existing modules, risks, Apple Silicon notes, blockers. |

**Exit Criteria:** Both files exist with verifiable evidence.

---

## 5. Phase 1 - Research Specification (Blocker)

| Step | Task | Verification Criteria |
| --- | --- | --- |
| **1.1** | **Define Research Questions** | Write answerable questions tied to `log_gvrh`, limitations, and surface-evidence layer. |
| **1.2** | **Lock Primary Target** | **Primary:** `matbench_log_gvrh`. **Optional:** `log_kvrh` only if time remains after leak-proof core is complete. |
| **1.3** | **Distinction Clause** | Explicitly state computed (benchmark) vs experimental (surface). Forbid equating modulus to "implant suitability". |
| **1.4** | **Scope Lock** | **Only metallic orthopedic implants** for surface-evidence. Exclude dental, soft-tissue scaffolds, drug-eluting systems. |
| **1.5** | **Write Protocol & Data Dictionary** | Create `docs/RESEARCH_PROTOCOL.md` (RQs, hypotheses, targets/units, datasets, inclusion/exclusion, provenance, splits, metrics, baselines, leakage policy, limitations, acceptance) and `docs/DATA_DICTIONARY.md`. |

**Exit Criteria:** Both files mention computed vs experimental, association vs causation, screening vs clinical validation.

---

## 6. Phase 2 - Data Ingestion, Provenance & Caching (Blocker)

| Step | Task | Verification Criteria |
| --- | --- | --- |
| **2.1** | **Official Matbench** | Use official `matbench` API and official fold definitions. Record `dataset_id`, `task_name`, `matbench_version`, source URL, retrieval timestamp (UTC), license, checksums if available. |
| **2.2** | **Raw Data Caching** | Cache raw files under `data/raw/` untouched. Never modify cached raw data. |
| **2.3** | **Schema Validation** | Validate columns, dtypes, target presence, units, sanity. Create `data_cards/matbench_log_gvrh_data_card.md`. |
| **2.4** | **Literature Intake Plan** | Create `docs/LITERATURE_REVIEW.md` with extraction template (DOI, alloy, treatment, roughness, wettability, corrosion, biological assay, study_id, etc.). |
| **2.5** | **Hygiene** | Update `.gitignore` for `.venv/`, `__pycache__/`, `.pytest_cache/`, `.env`, `data/raw/*`, `artifacts/`, large binaries. Add `.env.example` only. |

**Exit Criteria:** Provenance logged, raw cached untouched, data cards created, template ready.

---

## 7. Phase 3 - Leak-Proof Feature Engineering & Baseline Models

| Step | Task | Verification Criteria |
| --- | --- | --- |
| **3.1** | **Split Discipline** | Load official folds. Create `results/splits/split_manifest.json` with `dataset`, `split_type: "official_folds"`, `fold_id`, `train_idx`, `val/test_idx`, `group_key: null`, `random_state`, `timestamp`. |
| **3.2** | **Identity Removal** | Drop non-predictive identifiers (`id`, `material_id`, `mp-id`, `icsd_id`, `structure_id`, `cif`, bare `formula` as ID). Only descriptors allowed. |
| **3.3** | **Per-Fold Featurization** | If using `matminer`, compute inside each fold. No global pre-featurized CSV reused blindly. Prefer composition-based; pure-sklearn fallback ready. |
| **3.4** | **Sklearn Pipelines** | Wrap `imputer -> scaler -> (optional safe transformer) -> estimator` in `Pipeline`. Use `ColumnTransformer` only if necessary and fitted per-fold. **No fitting on full X/y.** |
| **3.5** | **Baselines First** | Implement `DummyRegressor` (mean/median), `Ridge`/`LinearRegression`, `RandomForestRegressor`. Optional boosting only if justified. |
| **3.6** | **OOF Predictions** | Save OOF to `results/predictions/oof_<model>_<task>.csv` with `fold_id`, `index/original_id`, `y_true`, `y_pred`. |
| **3.7** | **Metrics** | Compute MAE, RMSE, R² per fold + mean ± std. Save to `results/metrics/benchmark_metrics.json` and `results/tables/model_comparison.md`. |

**Exit Criteria:** Pipelines used, official folds locked, OOF saved, baselines evaluated.

---

## 8. Phase 4 - Rigorous Validation & Leakage & Overfitting Gate (HARD BLOCKER)

**NOTHING from Phases 5-9 may proceed until this gate is PASSED with evidence.**

| Step | Task | Verification Criteria |
| --- | --- | --- |
| **4.1** | **Create Leakage Audit** | Generate `docs/LEAKAGE_AUDIT.md` with checklist: official folds, pipeline-only preprocessing, IDs dropped, per-fold featurization, no pre-split EDA influence, OOF generated, group keys (if lit), duplicates audited. |
| **4.2** | **Split Manifest Verification** | Validate indices don't overlap between train/val/test for same evaluation unit. |
| **4.3** | **Y-Randomization (Mandatory)** | Run `scripts/y_randomize.py` with exact same pipeline/folds/features. Shuffle `y` appropriately. Expect R²≈0 or negative, MAE/RMSE near Dummy. Save to `results/validation/y_randomization.json`. **If close to real model -> FAIL.** |
| **4.4** | **Learning Curves** | Generate `results/plots/learning_curves_<model>.png`. Compare train vs validation across training sizes. Flag large gaps. |
| **4.5** | **Residuals & Parity** | Create `results/plots/parity_plot_<model>.png` and `results/plots/residuals_<model>.png`. Check heteroscedasticity, bias, clusters. |
| **4.6** | **Stability Check** | Review fold-wise scores. High variance must be reported with explanation. |
| **4.7** | **Leakage Sanity Tests** | Ensure `tests/test_leakage_sanity.py` checks: no identifier columns in features, pipelines exist, manifest present, OOF files exist. |
| **4.8** | **Fill & Sign-Off** | Complete `docs/LEAKAGE_AUDIT.md` with YES/NO + evidence paths. Mark `LEAKAGE_GATE: PASSED` only if all critical items YES and Y-randomization convincingly worse. **If ANY critical NO -> BLOCK.** |

**Gate Rule:** Explicit `LEAKAGE_GATE: PASSED` with evidence required before Phases 5-9.

---

## 9. Phase 5 - Model Interpretability, Uncertainty, Applicability Domain & Calibration

| Step | Task | Verification Criteria |
| --- | --- | --- |
| **5.1** | **Permutation Importance** | Use `sklearn.inspection.permutation_importance` on held-out/OoF context. Save CSV + plot. State associational, not causal. |
| **5.2** | **SHAP (Conditional)** | If `shap` installed and compatible, generate summary/beeswarm. Skip if unstable; prioritize permutation importance. |
| **5.3** | **Uncertainty Estimation** | Bootstrap ensembles of same pipeline or conformal prediction (MAPIE) if feasible. Save intervals to `results/uncertainty/`. |
| **5.4** | **Calibration Check** | Compute empirical coverage vs nominal (80/90/95%). Plot `results/plots/uncertainty_calibration.png`. Don't call variance "probability of implant success". |
| **5.5** | **Applicability Domain (AD)** | Lightweight AD (distance-to-training or leverage-based). Define `IN_DOMAIN`, `NEAR_BOUNDARY`, `OUT_OF_DOMAIN`. |
| **5.6** | **Document Limits** | Create `docs/INTERPRETABILITY_LIMITS.md` covering correlation vs causation, extrapolation, SHAP scope, calibration gaps, failure modes. |

**Exit Criteria:** Interpretability, uncertainty, AD implemented with non-clinical wording.

---

## 10. Phase 6 - Biomedical Evidence Layer (Curation-First, Study-Aware)

| Step | Task | Verification Criteria |
| --- | --- | --- |
| **6.1** | **Narrow Search** | Strictly **metallic orthopedic implants**. Prioritize publicly accessible papers with extractable tables/figures. |
| **6.2** | **Enforce Schema** | Create `data/literature/surface_evidence_raw.csv` per `DATA_DICTIONARY.md` with all mandatory keys including `study_id`, `doi`, etc. |
| **6.3** | **Assign study_id** | Stable `study_id` for every row (DOI-based). If multiple series under same DOI, split with suffix and justify. |
| **6.4** | **Deduplication Audit** | Deduplicate by `(study_id, material_grade_or_formula, surface_treatment, coating_chemistry, endpoint_key, conditions_key)`. Log in `docs/SURFACE_EVIDENCE_AUDIT.md`. |
| **6.5** | **Unit Harmonisation (Conservative)** | Only unambiguous conversions with cited rationale. Keep originals in `*_original`. Don't merge Ra/Rz, different corrosion metrics, different cell lines/endpoints. |
| **6.6** | **Missing vs Zero** | "Not reported" = `NaN`. Don't impute. Distinguish measured zero from missing. |
| **6.7** | **Evidence Audit** | Generate `docs/SURFACE_EVIDENCE_AUDIT.md` with raw/dedup counts, distinct studies, rows_per_study (min/med/max), % top, endpoints, units, gaps, exclusions. |
| **6.8** | **Data Card** | Create `data_cards/surface_evidence_data_card.md` noting public access, extraction method, limitations, not a clinical dataset. |

**Exit Criteria:** Clean, groupable, deduplicated table with full provenance and honest audit.

---

## 11. Phase 7 - Surface-Response Models: Only If Strictly Justified (GATED)

**Default: CURATION-ONLY. Train only if ALL conditions met.**

| Step | Task | Verification Criteria |
| --- | --- | --- |
| **7.1** | **Apply Modelling Gate** | Check: (a) `n_distinct_studies >= 3` (ideally >=4), (b) `n_deduplicated_compatible >= 25` for single locked endpoint, (c) no single study > 40% of rows, (d) feasible study-aware split exists, (e) Phase 4 gate `PASSED`. |
| **7.2** | **Lock Single Endpoint** | Exactly one endpoint with well-defined measurement/protocol. No multi-endpoint bio-score. |
| **7.3** | **Study-Aware Splits Only** | Use `GroupKFold` by `study_id`. If few groups, use `LeaveOneGroupOut` (LOGO) with per-group reporting. **Never** random split/CV. Save to `results/splits/surface_split_manifest.json`. |
| **7.4** | **Leak-Proof Pipeline** | Same pipeline discipline. Features derivable from reported values only. Don't inject study name/DOI/title as numeric features. |
| **7.5** | **Baselines + Y-Randomization** | Compare to `DummyRegressor`. Run Y-randomization with groups preserved. Require clearly worse than real labels. |
| **7.6** | **Tiny-Data Safeguards** | Report per-fold/LOGO, n per group, uncertainty (bootstrap if possible), learning curves. Expect high variance. |
| **7.7** | **Decision Rule** | **If gate fails:** No `model.pkl`. Create `docs/SURFACE_EVIDENCE_CURATED.md`, update data card, document curation-only rationale. **This is successful completion.** |
| **7.8** | **If Trained** | Save OOF-by-group, metrics by group, leakage audit entry, explicit "not generalizable beyond tested alloys/protocols". |

**Exit Criteria:** Either leak-audited group-validated micro-model with strict limits OR complete defensible curation-only module.

---

## 12. Phase 8 - Transparent Candidate Screening Engine

| Step | Task | Verification Criteria |
| --- | --- | --- |
| **8.1** | **Separate Data Sources** | Keep four pools separate: (a) observed literature, (b) benchmark predictions for `log_gvrh`, (c) other literature experimental, (d) user constraints. Never merge blindly. |
| **8.2** | **Criterion-Level Matrix** | Per candidate: `Criterion`, `Value`, `Units`, `Source`, `Evidence Level` (observed/literature/predicted/unknown), `Applicability`, `Constraint Met?` (YES/NO/NA). |
| **8.3** | **Enforce Hard Rules** | **No** composite % score. If any critical criterion unknown with no comparable evidence -> mark candidate as **`INDETERMINATE`**. Exclude from Ranked or separate tab. |
| **8.4** | **Modulus Discipline** | Never convert predicted `log_gvrh` to clinical recommendation. Treat as mechanical stiffness indicator from computed benchmark. State explicitly. |
| **8.5** | **Multi-Criteria with Transparency** | Weights only user-explorable lens. Show raw matrix first, weighted view second. Allow toggling, show trade-offs. |
| **8.6** | **Sensitivity Analysis** | Perturb weights slightly, show if top candidate flips. Log in `results/screening/sensitivity_report.md`. |
| **8.7** | **Evidence Tracing** | Every row traceable to DOI, `study_id`, benchmark file (`oof_*.csv`), or provenance entry. |
| **8.8** | **Guardrails in Code** | Defensive checks in `src/screening/`. If function tries to collapse to single numeric "fit" value -> block with comment + assertion. Add test `test_screening_guardrails.py`. |

**Exit Criteria:** Traceable, rule-based engine with evidence, missing data, trade-offs, indeterminate cases, zero fabricated rankings.

---

## 13. Phase 9 - Streamlit Application (Integrated & Honest)

| Step | Task | Verification Criteria |
| --- | --- | --- |
| **9.1** | **Global Disclaimer** | Persistent banner in `app.py`: *"RESEARCH DEMONSTRATOR – Not for clinical use. No patient-specific advice. No implant recommendation implied."* Visible on load. |
| **9.2** | **Page Structure** | Create/update pages 01-08 as specified. |
| **9.3** | **Provenance-Driven UI** | Pull from `results/`, `reports/`, `docs/`, `data_cards/`. Read `split_manifest.json`, `benchmark_metrics.json`, `LEAKAGE_AUDIT.md`, `SURFACE_EVIDENCE_AUDIT.md`. No hardcoded metrics. |
| **9.4** | **Benchmark Tab** | Show official folds, per-fold + mean±std MAE/RMSE/R², Dummy comparison, OOF availability, link to `LEAKAGE_AUDIT.md`, display `LEAKAGE_GATE` status. |
| **9.5** | **Prediction Tab** | Show target `log_gvrh (computed)`, units `log(GPa)` per Matbench, model version, folds, AD status, uncertainty interval if available, limitations. Reject unsupported with reason. |
| **9.6** | **Interpretability & Leakage Tab** | Show permutation importance, SHAP (if available), Y-randomization, learning curves, residuals/parity, checklist from `LEAKAGE_AUDIT.md`. Text: *"Associational only. Not causal."* |
| **9.7** | **Surface Evidence Tab** | Show counts, studies, distribution, dedup log, harmonization notes, missing-vs-zero, `CURATION-ONLY` badge if no model. No auto biological compatibility score. |
| **9.8** | **Screening Tab** | Criterion matrix, evidence trail, constraints, `INDETERMINATE` list, weight-sensitivity. Block UI implying "best for implant"/"clinically superior". |
| **9.9** | **Import-Safe Launch Test** | App must import cleanly: `python -c "import app; print('APP_IMPORT_OK')"` from repo root. |

**Exit Criteria:** Honest, data-driven Streamlit app reflecting true state with disclaimers.

---

## 14. Phase 10 - Reproducibility, Software Quality, Tests & Final Documentation

| Step | Task | Verification Criteria |
| --- | --- | --- |
| **10.1** | **Modular Structure** | `src/` package with `__init__.py`. Logic split by modules. Avoid dumping in `app.py`. |
| **10.2** | **Configs** | Add `configs/benchmark.yaml`, `configs/featurization.yaml` with relative paths. |
| **10.3** | **Determinism** | Set seeds where feasible. Note non-determinism in `docs/LIMITATIONS.md`. |
| **10.4** | **Type Hints & Docstrings** | Add to core functions in `src/` (loader, pipeline builder, validation, screening). |
| **10.5** | **Structured Logging** | Add `src/utils/logging.py` if missing. Log split loading, per-fold featurization, gate checks, deduplication. |
| **10.6** | **Meaningful Tests** | Ensure all 5 test files exist with substantive checks including screening guardrails. |
| **10.7** | **Run Full Test Suite** | `pytest -v 2>&1 | tee reports/pytest_run_$(date +%Y%m%d_%H%M%S).log`. Fix failures from your changes; document pre-existing with proof. |
| **10.8** | **Final Documentation Sync** | Update `README.md` to match reality. Complete `LITERATURE_REVIEW.md`, `RESEARCH_GAPS.md`, `LIMITATIONS.md`. Update `PROJECT_LOG.md` with only actions performed (timestamps/commands/files). |
| **10.9** | **Results Hygiene** | Ensure `results/` has all subdirs. Don't commit large `.pkl/.joblib` unless necessary and justified. Prefer metrics/predictions/plots. |
| **10.10** | **Thesis Skeleton Check** | Minimal LaTeX structure if exists; don't rewrite chapters. |

**Exit Criteria:** Tests logged, docs match code, reproducible commands work, no false claims.

---

## 15. Evaluation Metrics

### For Benchmark (matbench_log_gvrh)
| Metric | Formula/Notes | Purpose |
| --- | --- | --- |
| **MAE** (Mean Absolute Error) | `mean(\|y_true - y_pred\|)` | Primary interpretability metric, same units as target (log GPa) |
| **RMSE** (Root Mean Squared Error) | `sqrt(mean((y_true - y_pred)^2))` | Penalizes larger errors |
| **R²** (Coefficient of Determination) | `1 - SS_res/SS_tot` | Explained variance; can be negative for bad models |

All reported as **mean ± standard deviation** across official folds. Compare against `DummyRegressor` baseline.

### For Surface Evidence (if model trained)
| Metric | Notes | Purpose |
| --- | --- | --- |
| **MAE/RMSE/R²** | Computed per group and overall (LOGO/GroupKFold) | Performance with group-awareness |
| **Per-group metrics** | Mandatory for LOGO | Reveal variability across studies |
| **Baseline comparison** | Must beat or be explained vs DummyRegressor | Sanity check |

### Validation Checks
| Check | Expected | Action if Failed |
| --- | --- | --- |
| **Y-Randomization** | R² ≈ 0 or negative, error near dummy | **GATE FAIL - block progression** |
| **Learning Curves** | Converging train/val or reasonable gap | Investigate if large growing gap (overfit) |
| **Residual Analysis** | Random scatter around zero, homoscedastic | Investigate patterns (misspecification/leakage) |
| **Calibration** | Empirical coverage ≈ nominal coverage | Report honestly if miscalibrated |

---

## 16. Testing Strategy

### Required Test Files

| Test File | Purpose | Key Assertions |
| --- | --- | --- |
| `tests/test_leakage_sanity.py` | **Critical leakage checks** | No identifier columns in feature set, pipelines exist, `split_manifest.json` present, OOF files exist for evaluated models, no fitting on full data patterns |
| `tests/test_splits.py` | **Split integrity** | Official folds load correctly, train/val/test indices don't overlap, split manifest has required keys |
| `tests/test_data_validation.py` | **Data schema validation** | Required columns present, dtypes correct, no unexpected NaN patterns, target exists |
| `tests/test_app_imports.py` | **App import safety** | `import app` succeeds, no missing deps crash on import |
| `tests/test_screening_guardrails.py` | **Screening honesty** | No function creates single aggregate "suitability %" score, INDETERMINATE logic exists, evidence tracing present, guardrails enforced |

### Running Tests

```bash
# Activate environment first
source .venv/bin/activate

# Run all tests with verbose output and logging
pytest -v 2>&1 | tee reports/pytest_run_$(date +%Y%m%d_%H%M%S).log

# Run specific test file
pytest tests/test_leakage_sanity.py -v

# Run with coverage (optional)
pytest --cov=src -v
```

---

## 17. VS Code Execution Workflow

| Step | Action | Notes |
| --- | --- | --- |
| **A)** | **Open Project** | `File > Open Folder` → select repo root. Verify Explorer shows `src/`, `docs/`, `tests/`, `app.py` |
| **B)** | **Inspect First** | Terminal: `pwd && ls -la`, `git status --short --branch`, `python3 --version && which python3`. Read `README.md` fully. |
| **C)** | **Create/Activate .venv** | If missing: `python3 -m venv .venv`. Activate: `source .venv/bin/activate`. Verify `(.venv)` in prompt. |
| **D)** | **Install Dependencies** | `pip install --upgrade pip setuptools wheel`. Install from `requirements.txt` if exists, else incrementally. Watch Apple Silicon warnings. |
| **E)** | **Read Existing Code** | Open `src/data/loader.py`, Matbench-related tests, `app.py` in tabs. Understand before editing. Use `Cmd+P`/`Ctrl+P` to navigate. |
| **F)** | **Work File-By-File** | Small, testable chunks. Test after meaningful edits. Use Problems panel for syntax issues. |
| **G)** | **Run Checks Often** | `python -c "import src, src.data, src.models"` for imports. `pytest -v` after touching modules. |
| **H)** | **Keep Logs** | Every command goes to `docs/PROJECT_LOG.md` with timestamp, files touched, outcome (OK/FAIL + reason). |

---

## 18. Start Sequence (Execute in Exact Order)

**BEGIN NOW with zero assumptions. Use VS Code integrated terminal.**

| Order | Phase | Actions |
| --- | --- | --- |
| **1.** | **PHASE 0 START** | Run inspection commands → create `docs/REPOSITORY_AUDIT.md` + `docs/IMPLEMENTATION_PLAN.md`. List keep/modify/create. |
| **2.** | **PHASE 1** | Create `docs/RESEARCH_PROTOCOL.md`, `docs/DATA_DICTIONARY.md`. Lock `log_gvrh` as primary, metallic implants only, computed vs experimental. |
| **3.** | **PHASE 2** | Set up provenance, raw cache, data cards, literature template. Audit existing Matbench loader/tests. |
| **4.** | **PHASE 3** | Lock official folds → `results/splits/split_manifest.json`. Build leak-proof pipelines. Implement baselines. Save OOF. |
| **5.** | **PHASE 4 – GATE (MANDATORY)** | Create `LEAKAGE_AUDIT.md`, run Y-randomization, learning curves, residuals/parity, add leakage tests. Set `LEAKAGE_GATE: PASSED` only with evidence. **DO NOT SKIP.** |
| **6.** | **PHASE 5** | Add permutation importance, AD, uncertainty + calibration. Write `INTERPRETABILITY_LIMITS.md`. |
| **7.** | **PHASE 6** | Curate literature into groupable form. Deduplicate, assign `study_id`, build `SURFACE_EVIDENCE_AUDIT.md`, data card. |
| **8.** | **PHASE 7** | Run gate. If insufficient → **curation-only** (document clearly). If justified → group-aware training + groups-preserved Y-randomization. |
| **9.** | **PHASE 8** | Build transparent screening engine with criterion matrix, traceability, `INDETERMINATE`, sensitivity. Add guardrails. |
| **10.** | **PHASE 9** | Update Streamlit app with all pages + persistent disclaimer + gate/status display. Test import. |
| **11.** | **PHASE 10** | Tests, logs, README sync, results hygiene, final verification. Generate final report. |

---

## 19. Mandatory Final Report Template

At the very end, output this report with **actual values only**. No guesses, no "should be", no fabricated numbers.

```text
## RP-1 FINAL REPORT – LEAK-PROOF, EVIDENCE-AWARE IMPLEMENTATION

### 1) Repository Audit
- Root: [actual path]
- Git: [repo? yes/no + short status]
- Python/.venv: [version + used?]
- Existing Matbench loader/tests found: [yes/no + paths]
- Preserved: [list]
- Modified: [list with justification]
- Created: [list]

### 2) Environment
- OS/Arch: [e.g. macOS Apple Silicon]
- Dependencies installed: [key ones from pip show]
- Network to Matbench: [OK/UNREACHABLE]
- Approval requested for large installs/downloads: [yes/no + when]

### 3) Benchmark (matbench_log_gvrh)
- Official folds used: [yes/no]
- split_manifest.json: [path + exists?]
- Featurization: [composition/structure + per-fold?]
- Pipelines used: [yes/no]
- Models: [Dummy, Ridge/RF/others]
- OOF saved: [paths]
- Metrics (mean ± std): MAE | RMSE | R² per model
- Dummy comparison: [beaten? y/n + brief]

### 4) Leakage & Overfitting Gate
- LEAKAGE_AUDIT.md: [present? filled?]
- Y-randomization run: [done? file? result summary]
- Learning curves/residuals/parity: [present?]
- Leakage tests: [passed/failed]
- LEAKAGE_GATE: [PASSED / FAILED]
- Blocked phases until passed: [stated? yes]

### 5) Interpretability, Uncertainty, AD
- Permutation importance: [done]
- SHAP: [done/skipped + reason]
- Uncertainty method: [bootstrap/conformal/none]
- Calibration done: [yes/no]
- AD implemented: [yes/no]
- Docs: INTERPRETABILITY_LIMITS.md present? [yes/no]

### 6) Surface Evidence Layer
- Source: [publicly accessible papers only]
- study_id assigned: [yes/no]
- Raw/Deduplicated: [n_raw -> n_dedup]
- Distinct studies: [n]
- Rows per study (min/med/max) + % top: [...]
- Unit harmonization: [conservative/none]
- SURFACE_EVIDENCE_AUDIT.md + data card: [present]
- Modelling: [TRAINED / CURATION-ONLY]
- If trained: group splits + groups-preserved Y-randomization [done]
- If curation-only: reason stated [exact text]

### 7) Screening & App
- Screening engine: [criterion matrix + traceability + INDETERMINATE]
- Guardrails present: [test_screening_guardrails.py or manual check]
- Streamlit app import: [python -c "import app" -> OK/FAIL]
- Pages present: [list]
- Persistent disclaimer: [yes/no]
- AD/uncertainty shown on prediction: [yes/no]

### 8) Tests & Reproducibility
- pytest -v command: [actual command]
- Results: Passed | Failed | Skipped
- Log file: [reports/pytest_run_*.log]
- README matches actual commands: [yes/no]
- PROJECT_LOG.md updated with real actions: [yes/no]
- configs/ present: [yes/no]

### 9) Remaining Blockers & Next Recommended Task
- Blockers: [concrete list or "None"]
- Recommended next task: [single, actionable step tied to current state]
- Scientific caution note: [state computed vs experimental + screening vs clinical validation]
```

---

## 20. Final Instructions

- **Use VS Code only for inspection/editing.** Do everything in the repo you have open.
- **Inspect before modifying.** Read files fully before changing them.
- **Enforce the gate.** Do **not** proceed past PHASE 4 without `LEAKAGE_GATE: PASSED`.
- **Conservative with literature.** Curation-only is a **correct and complete** outcome.
- **No false numbers.** If something wasn't run, say so.
- **Preserve working code.** Extend, don't erase.
- **Stay within scope.** Metallic orthopedic implants only for the surface layer.
- **Be brutally honest.** Distinguish *preliminary*, *insufficient*, *failed* explicitly.
- **Pipeline-first always.** No fitting on full data. Per-fold featurization. Official folds only.
- **Study-aware for literature.** Never random split literature data.
- **Evidence over score.** Criterion-level with traceability, no aggregate suitability %.

**START NOW: Run PHASE 0 in the VS Code integrated terminal and create `docs/REPOSITORY_AUDIT.md` before making any code changes.**