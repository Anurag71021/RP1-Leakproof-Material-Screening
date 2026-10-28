# Project log

**Project:** RP-1 leak-proof, evidence-aware material and surface screening
**Started:** 2026-10-28
**Environment:** macOS 26.6.1 arm64, Python 3.10.18 in `.venv`
**Rule:** only actions actually performed are recorded. Times are UTC.

---

## 2026-10-28

### 17:2x — Phase 0 repository audit

Commands: `pwd`, `ls -la`, `python3 --version`, `sw_vers`, `uname -m`,
`df -h .`, `git status --short --branch`, network checks to
`matbench.materialsproject.org` and PyPI.

Findings:

- Repository contained only the specification and `opencode.jsonc`; no application
  code, no tests, no Git repository.
- System Python 3.9.6; Homebrew Python 3.10 available.
- Network reachable (HTTP 200); ~28 GiB free disk.

Outputs: `docs/REPOSITORY_AUDIT.md`, `reports/environment.txt`.

### 17:2x — dependency resolution (blocker and fix)

First attempt: `pip install -r requirements.txt` → `ResolutionImpossible`.

Cause: `matbench==0.6` pins `matminer==0.7.4`, which has no Python 3.10 / arm64
build. Resolution used:

```bash
python3.10 -m venv .venv
.venv/bin/pip install numpy pandas scikit-learn scipy pyyaml matplotlib seaborn streamlit pytest
.venv/bin/pip install matminer pymatgen
.venv/bin/pip install --no-deps matbench==0.6
.venv/bin/pip install -e . --no-deps
```

Verified `from matbench.bench import MatbenchBenchmark` and a full
`matbench_log_gvrh` load. Installed: matbench 0.6, matminer 0.9.3,
pymatgen 2025.10.7, numpy 1.26.4, pandas 2.2.3, scikit-learn 1.7.2.

Second environment issue: matminer's default multiprocessing featurizer fails when
the entry point is stdin or Streamlit. All featurization is forced to
`set_n_jobs(1)` in `src/features/composition.py`.

### 17:2x — Phase 1 protocol

Created `docs/RESEARCH_PROTOCOL.md` and `docs/DATA_DICTIONARY.md`; locked the
target as `log10(G_VRH)` in log10(GPa), composition-only features, official folds
only, and the Phase 7 gate thresholds (≥ 3 studies, ≥ 25 rows, no study > 40%).

### 17:34 — Phase 2 ingestion and validation

Command: `.venv/bin/python scripts/build_dataset.py`

Result: 10,987 rows cached; 5 official folds (train 8,789/8,790; test
2,198/2,197); target min 0.0000, max 2.7185, mean 1.5536, std 0.3716; 0 duplicate
groups; 0 cross-fold duplicate groups; all manifest checks passed.

Outputs: `data/raw/*` (gitignored), `results/splits/split_manifest.json`,
`reports/data_validation.json`, `data_cards/matbench_log_gvrh_data_card.md`.

Note: an earlier draft of `docs/DATA_DICTIONARY.md` contained provisional target
statistics. It was corrected to the persisted values above after checking the
generated report.

### 17:38–17:39 — Phase 3 benchmark

Command: `.venv/bin/python scripts/run_benchmark.py`

Results (mean ± SD across 5 official folds):

| Model | MAE | RMSE | R² |
| --- | --- | --- | --- |
| Dummy | 0.2931 ± 0.0035 | 0.3716 ± 0.0042 | -0.0002 ± 0.0002 |
| Ridge | 0.1568 ± 0.0011 | 0.2144 ± 0.0028 | 0.6671 ± 0.0069 |
| Random Forest | 0.1024 ± 0.0016 | 0.1520 ± 0.0046 | 0.8325 ± 0.0100 |

Selected model by lowest mean OOF MAE: `random_forest`.

Outputs: `results/predictions/oof_*.csv`, `results/metrics/benchmark_metrics.json`,
`results/metrics/best_model.json`, `results/tables/model_comparison.md`.

### 17:42–17:53 — Phase 4 Y-randomization

Command: `.venv/bin/python scripts/y_randomize.py` (Ridge 5 permutations, Random
Forest 2 permutations)

Results: Ridge randomized R² mean -0.0145 (max -0.0068), MAE 0.2952; Random
Forest randomized R² mean -0.1155 (max -0.1116), MAE 0.3093. Both collapse to or
below Dummy-level performance.

Output: `results/validation/y_randomization.json`, `reports/y_randomization_run.log`.

### 17:53–17:54 — Phase 4 diagnostics

Command: `.venv/bin/python scripts/run_diagnostics.py`

Learning curve (official fold 0, Random Forest): held-out MAE 0.1434 → 0.1288 →
0.1148 → 0.1019 as training size grows from 878 to 8,789; train MAE 0.0443 at full
size. Residual mean 0.0006, SD 0.1521, |residual| p95 0.3185,
corr(pred, |residual|) = -0.1718.

Deliberate ordering decision: parity/residuals and learning curves are gate inputs,
so they were produced by a Phase-4 script rather than by `scripts/run_analysis.py`.
Requiring Phase-5 outputs to pass a Phase-4 gate would be circular.

Outputs: `results/validation/learning_curves.json`,
`results/validation/residual_stats_random_forest.json`,
`results/plots/parity_plot_random_forest.png`,
`results/plots/residuals_random_forest.png`,
`results/plots/learning_curves_random_forest.png`.

### 17:5x — Phase 4 gate, first attempt

Command: `.venv/bin/python scripts/finalize_gate.py`

Result: **FAILED** on one check only: `leakage_tests_present` — the five required
test files had not yet been written. All 11 other checks passed. The gate was not
marked passed and Phases 5–9 were not run.

### 18:0x — required tests written

Created:

- `tests/conftest.py`
- `tests/test_leakage_sanity.py`
- `tests/test_splits.py`
- `tests/test_data_validation.py`
- `tests/test_app_imports.py`
- `tests/test_screening_guardrails.py`

### 18:0x — Phase 4 gate, second attempt

Command: `.venv/bin/python scripts/finalize_gate.py`

Result: **PASSED**; all 12 checks true. `docs/LEAKAGE_AUDIT.md` ends with
`LEAKAGE_GATE: PASSED`; `results/validation/gate_status.json` records the same.

### 18:00–18:0x — Phase 5 analysis

Command: `.venv/bin/python scripts/run_analysis.py`

Produced: permutation importance (5 folds, 5 repeats, 1,500 held-out rows/fold),
split-conformal intervals, per-fold applicability domain and a deployment AD
reference, plus parity/residual refreshes.

Outputs: `results/interpretability/`, `results/uncertainty/`,
`results/validation/applicability_domain*.json`, `artifacts/ad_reference.joblib`,
`results/plots/uncertainty_calibration.png`.

### 18:00–18:0x — Phase 6 evidence curation

Commands: retrieved open-access full text by PMCID from the Europe PMC
`fullTextXML` endpoint; read tables programmatically; then
`.venv/bin/python scripts/build_literature.py` (run twice: once before the gate
passed, once after, so the recorded gate reasons reflect the final Phase-4 state).

Result: 40 raw rows, 40 clean rows (0 duplicates on the documented key),
5 studies, endpoints Ra 23 / water contact angle 9 / Ecorr 8. Locked endpoint
`surface_roughness_ra_um`: 23 rows, 2 studies, largest study share 87%.

Note: the Europe PMC *search* endpoint returned HTTP 503 during the session, so
discovery used NCBI E-utilities and full text was fetched by PMCID. Two
dental-only studies were excluded with reasons. Two environment/API issues were
handled: NCBI requested rate limiting, and one full-text fetch initially returned
an HTML 429 page before succeeding via Europe PMC.

Outputs: `data/literature/surface_evidence_{raw,clean}.csv`,
`docs/SURFACE_EVIDENCE_AUDIT.md`, `docs/SURFACE_EVIDENCE_CURATED.md`,
`data_cards/surface_evidence_data_card.md`, `results/validation/surface_gate.json`.

### 18:0x — Phase 7 gate decision

Command: `.venv/bin/python scripts/train_surface_model.py`

Result: `CURATION_ONLY`. The gate was re-evaluated from the persisted table; it
failed on study count (2 < 3), row count (23 < 25), single-study dominance
(87% > 40%) and feasibility of a study-aware split (< 3 groups). No surface model
was trained and no model artefact was written.

Output: `results/validation/surface_model_decision.json`,
`docs/SURFACE_MODEL_DECISION.md`.

### 18:0x — Phase 8 screening

Command: `.venv/bin/python scripts/build_screening.py`

Fitted the deployment Random Forest on all labelled rows (deployment only; reported
metrics remain the official-fold OOF values). Scored 14 candidate material +
surface-state variants against user constraints, joined curated evidence, and wrote
a criterion matrix, summary, lexicographic ranking and priority sensitivity.

Outputs: `artifacts/model_log_gvrh.joblib` (gitignored),
`results/screening/*` (candidates, predictions, matrix, summary, ranking,
sensitivity, summary JSON, README).

### 18:0x — Phase 9 app

Created `app.py`, `src/ui.py`, and eight pages under `pages/`. Every page reads
persisted artefacts through `src/ui.py`; no metric is hardcoded. Each page carries
the research disclaimer and the app exposes gate status.

### 18:0x — Phase 10 documentation and tests

Created `README.md`, `docs/LIMITATIONS.md`, `docs/INTERPRETABILITY_LIMITS.md`,
`docs/LITERATURE_REVIEW.md`, `docs/RESEARCH_GAPS.md`, this log, and the final report.

### 18:11–18:12 — final verification

Command: `.venv/bin/python -m pytest -v 2>&1 | tee reports/pytest_run_$(date -u +%Y%m%d_%H%M%S).log`

First run: 74 passed, 1 failed. The failure was in the synthetic
`test_sensitivity_detects_priority_flip` fixture, which gave candidate A the better
value on *both* ranked criteria, so no priority order could flip the top result. The
engine was correct; the fixture was changed to a genuine trade-off (A better on
modulus, B better on roughness). The persisted screening sensitivity was unaffected
and still reports a real flip between `ti15mo_polished_2` and `ss316l_polished_2`.

Second run: **75 passed, 0 failed, 0 skipped** in 4.89 s. Log:
`reports/pytest_run_20261028_181218.log`.

Command: `.venv/bin/python -c "import app; print('APP_IMPORT_OK')"` → `APP_IMPORT_OK`.

Final report written to `reports/FINAL_REPORT.md` using the mandatory template and
only values read from persisted artefacts.

### 2026-10-29 — Colab bootstrap

Added `colab_setup.py` (a Colab bootstrap, deliberately not named `setup.py` so it
cannot interfere with setuptools/pyproject packaging). It creates a Python 3.10
`uv` environment, installs the pinned stack and `matbench==0.6 --no-deps`,
regenerates the gitignored dataset/model artefacts, runs the tests, and can launch
Streamlit through a Cloudflare quick tunnel. Added `docs/COLAB.md` and a Colab
section in `README.md`. Verified `--help` and `--dry-run --run --app`; the test
suite still reports 75 passed.

---

## Open deviations and known issues

1. **Matbench pin bypass.** `matbench==0.6` was installed with `--no-deps` because
   its pinned matminer has no Python 3.10 arm64 build. Dataset and folds still come
   from the official API; this is a packaging deviation, not a data deviation.
2. **Logging timezone fixed late.** Early log files append `Z` to local time
   (IST). The formatter was corrected to use `time.gmtime` before the later runs;
   file artefacts use `datetime.now(timezone.utc)` and are correct.
3. **AI-assisted extraction.** Every literature row is AI-extracted and requires
   human verification against the PDF before submission.
4. **No SHAP.** `shap` is not installed; permutation importance is the only
   interpretability method reported. This is stated in the app and docs.
5. **Learning curves use one fold.** Runtime bounding; documented in the artefact.
6. **Git repository initialised at the end.** No history exists before this build;
   this is stated in the audit.
