# RP-1 — Leak-proof, evidence-aware material and surface screening

A research demonstrator for metallic orthopaedic implant materials. It combines:

1. a **leak-proof benchmark** of composition-only `log10(G_VRH)` prediction under the
   official Matbench 5-fold protocol;
2. **uncertainty and applicability-domain** reporting for every prediction;
3. a **provenance-preserving, curation-only surface evidence layer** built from
   open-access literature;
4. a **criterion-level screening engine** with explicit constraints and honest
   `INDETERMINATE` status — no composite suitability score;
5. an **8-page Streamlit app** that reads only persisted artefacts.

> **RESEARCH DEMONSTRATOR — Not for clinical use. No patient-specific advice.
> No implant recommendation implied.**

---

## 1. What the numbers mean

| Layer | Quantity | Nature |
| --- | --- | --- |
| Benchmark target | `log10(G_VRH)` (log10 GPa) | **Computed** DFT shear modulus from Matbench v0.1 |
| Model output | predicted `log10(G_VRH)` | Statistical prediction from **composition only** |
| Evidence layer | Ra, water contact angle, Ecorr | **Measured** values read from papers, laboratory-specific |
| Screening status | `RANKED` / `CONSTRAINT_FAILED` / `INDETERMINATE` | Evidence completeness under user constraints |

None of these is an implant success rate, a biocompatibility score, a fatigue
limit, a corrosion warranty, or a clinical recommendation.

## 2. Headline results (actual, from `results/metrics/benchmark_metrics.json`)

Official Matbench 5-fold out-of-fold metrics, mean ± sample SD:

| Model | MAE (log10 GPa) | RMSE | R² |
| --- | --- | --- | --- |
| Dummy (mean) | 0.2931 ± 0.0035 | 0.3716 ± 0.0042 | -0.0002 ± 0.0002 |
| Ridge (α = 1) | 0.1568 ± 0.0011 | 0.2144 ± 0.0028 | 0.6671 ± 0.0069 |
| **Random Forest (300 trees)** | **0.1024 ± 0.0016** | **0.1520 ± 0.0046** | **0.8325 ± 0.0100** |

- Selected model: `random_forest`, chosen by lowest mean OOF MAE **before** any
  screening or literature analysis.
- `LEAKAGE_GATE: PASSED` (12/12 checks). Y-randomization collapses: Random Forest
  randomized R² max **-0.1116**, MAE 0.3093 versus Dummy 0.2931.
- Split-conformal coverage on held-out folds: 0.8001 / 0.9001 / 0.9502 at nominal
  0.80 / 0.90 / 0.95.
- Applicability domain (OOF): 10,433 in-domain, 511 near-boundary, 43 out-of-domain.

Surface evidence: 40 rows, 5 studies, 3 endpoints. Locked endpoint Ra has **23 rows
from 2 studies with 87% from one study**, so Phase 7 is **`CURATION_ONLY`**. No
surface model was trained.

## 3. Repository structure

```
.
├── app.py                      # Streamlit entry point (persistent disclaimer)
├── pages/                      # 8 pages, artefact-driven
├── src/
│   ├── config.py               # paths, target constants, disclaimer
│   ├── ui.py                   # shared app loaders/badges
│   ├── data/                   # loader, validation, literature curation
│   ├── features/               # stateless Magpie composition featurizer
│   ├── models/                 # pipelines, official-fold CV
│   ├── evaluation/             # metrics, validation, uncertainty, AD
│   ├── interpretability/       # permutation importance
│   ├── screening/              # engine + composite-score guardrails
│   └── utils/                  # IO and structured logging
├── scripts/                    # reproducible pipeline entry points
├── configs/                    # benchmark + featurization YAML
├── tests/                      # 5 required suites + shared fixtures
├── docs/                       # protocol, audit, gate, evidence, limitations
├── data_cards/                 # benchmark and evidence data cards
├── results/                    # metrics, predictions, plots, screening
├── reports/                    # run logs, environment, final report
├── data/                       # raw cache (gitignored), literature CSV/XML
└── artifacts/                  # deployment model, AD reference (gitignored)
```

## 4. Setup

Python 3.10 is required (matminer/pymatgen wheels on Apple Silicon).

`matbench==0.6` pins `matminer==0.7.4`, which has no Python 3.10 arm64 build. The
working installation therefore installs matbench without dependencies:

```bash
python3.10 -m venv .venv
source .venv/bin/activate

pip install "numpy>=1.23,<2.0" "pandas>=1.5,<2.3" "scikit-learn>=1.2,<2.0" \
            "scipy>=1.9,<2.0" pyyaml matplotlib seaborn streamlit pytest joblib
pip install matminer pymatgen
pip install --no-deps matbench==0.6
pip install -e . --no-deps
```

The dataset and folds still come from the official `matbench.bench` API. This is a
packaging workaround, recorded in `docs/PROJECT_LOG.md`.

## 5. Reproduce the pipeline

Run from the repository root, in this order:

```bash
python scripts/build_dataset.py          # cache, validate, lock official folds
python scripts/run_benchmark.py          # Dummy / Ridge / Random Forest, 5 folds
python scripts/y_randomize.py            # mandatory label-shuffle control
python scripts/run_diagnostics.py        # parity, residuals, learning curves
python scripts/finalize_gate.py          # writes docs/LEAKAGE_AUDIT.md + gate JSON
python scripts/run_analysis.py           # importance, conformal, AD (gate-guarded)
python scripts/build_literature.py       # curation + Phase 7 gate
python scripts/train_surface_model.py    # re-applies gate; records CURATION_ONLY
python scripts/build_screening.py        # deployment model + criterion matrix
pytest -v 2>&1 | tee reports/pytest_run_$(date -u +%Y%m%d_%H%M%S).log
python -c "import app; print('APP_IMPORT_OK')"
streamlit run app.py
```

`finalize_gate.py` refuses to pass if any artefact is missing. `run_analysis.py`
and `build_screening.py` refuse to run unless `docs/LEAKAGE_AUDIT.md` contains
`LEAKAGE_GATE: PASSED`.

## 6. App pages

| Page | Content |
| --- | --- |
| 01 Project overview | RQs, hypothesis status, phase table, scope |
| 02 Dataset provenance & quality | hashes, folds, validation, duplicates, data card |
| 03 Benchmark results | per-fold and mean metrics, parity/residual/learning plots |
| 04 Prediction, AD & uncertainty | interactive composition prediction, interval, support status |
| 05 Interpretability & leakage | permutation importance, gate checklist, Y-randomization |
| 06 Surface evidence | counts, endpoints, missingness, exclusions, curation-only badge |
| 07 Candidate screening | criterion matrix, constraints, INDETERMINATE list, sensitivity |
| 08 Limitations & disclaimers | hard non-use statement and known gaps |

## 7. Tests

Five required suites:

- `tests/test_leakage_sanity.py` — identifiers excluded, stateless featurizer,
  fold-local fitting, OOF reproduced metrics, Y-randomization collapse, signed gate.
- `tests/test_splits.py` — official folds partition the data, no overlap, manifest
  matches cached indices.
- `tests/test_data_validation.py` — schema, target, provenance checksums, data card.
- `tests/test_app_imports.py` — `APP_IMPORT_OK`, eight pages parse, no forbidden UI claims.
- `tests/test_screening_guardrails.py` — composite scores rejected, INDETERMINATE
  logic, lexicographic ranking, evidence traceability.

## 8. Screening rules (short version)

- Critical criteria: predicted modulus and Ra. Contact angle informs but does not gate.
- A missing critical criterion → `INDETERMINATE`; never ranked, never given a low score.
- Ranking is lexicographic by user priority; no weighted aggregate exists anywhere.
- Priority sensitivity is reported; a flip means no single material is supported.
- Thresholds are research constraints, not clinical acceptance limits.

## 9. Data and licence notes

- Matbench v0.1: Dunn et al. 2020, doi:10.1038/s41524-020-00406-3; elastic data from
  de Jong et al. 2015, doi:10.1038/sdata.2015.9 (CC BY 4.0). Check redistribution
  terms before publishing derived data.
- Curated literature rows retain DOI, PMCID and licence context per source. Some
  sources are CC BY-NC-ND; check before redistributing extracted tables.
- The raw dataset cache and model binaries are gitignored; provenance and checksums
  are recorded instead.

## 10. Documentation

Start with:

1. `docs/RESEARCH_PROTOCOL.md` — locked scope and rules.
2. `docs/LEAKAGE_AUDIT.md` — signed gate with evidence paths.
3. `docs/LIMITATIONS.md` — what must not be claimed.
4. `docs/LITERATURE_REVIEW.md` — search, extraction, exclusions and gaps.
5. `docs/SURFACE_EVIDENCE_CURATED.md` — why curation-only is the correct outcome.
6. `docs/PROJECT_LOG.md` — commands, timings and deviations.
7. `reports/FINAL_REPORT.md` — final self-audit against the specification.

## 11. Non-use

Do not use this repository for clinical decisions, patient-specific advice, device
qualification, regulatory submission, or as evidence that a material is
biocompatible or safe. It is a research and teaching artefact.
