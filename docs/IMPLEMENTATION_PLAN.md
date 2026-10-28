# Implementation plan - RP-1

**Status:** living document. Each phase is marked with the artefact that proves
it, not with an intention.

| Phase | Scope | Status | Evidence |
| --- | --- | --- | --- |
| 0 | Repository audit | complete | `docs/REPOSITORY_AUDIT.md`, `reports/environment.txt` |
| 1 | Research specification | complete | `docs/RESEARCH_PROTOCOL.md`, `docs/DATA_DICTIONARY.md` |
| 2 | Ingestion, provenance, caching | complete | `data/raw/*` (gitignored), `data_cards/matbench_log_gvrh_data_card.md`, `reports/data_validation.json` |
| 3 | Leak-proof features + baselines | complete | `results/metrics/benchmark_metrics.json`, `results/predictions/oof_*.csv`, `results/splits/split_manifest.json` |
| 4 | Validation / leakage gate | complete | `docs/LEAKAGE_AUDIT.md`, `results/validation/y_randomization.json`, plots, `tests/test_leakage_sanity.py` |
| 5 | Interpretability / uncertainty / AD | complete | `results/interpretability/`, `results/uncertainty/`, `results/validation/applicability_domain.json` |
| 6 | Biomedical evidence layer | complete | `data/literature/surface_evidence_*.csv`, `docs/SURFACE_EVIDENCE_AUDIT.md`, `data_cards/surface_evidence_data_card.md` |
| 7 | Surface-response model (gated) | gate decision recorded | `docs/SURFACE_EVIDENCE_CURATED.md` or model artefacts |
| 8 | Screening engine | complete | `results/screening/`, `src/screening/`, `tests/test_screening_guardrails.py` |
| 9 | Streamlit app | complete | `app.py`, `pages/01..08`, import check in `reports/` |
| 10 | Tests, docs, reproducibility | complete | `reports/pytest_run_*.log`, `README.md`, `docs/PROJECT_LOG.md` |

## File ownership by phase

- **Phase 2**: `src/data/loader.py`, `src/data/validate.py`, `scripts/build_dataset.py`
- **Phase 3**: `src/features/`, `src/models/`, `scripts/run_benchmark.py`
- **Phase 4**: `src/evaluation/validation.py`, `scripts/y_randomize.py`, `scripts/finalize_gate.py`
- **Phase 5**: `src/evaluation/{uncertainty,applicability}.py`, `src/interpretability/`, `scripts/run_analysis.py`
- **Phase 6**: `src/data/literature.py`, `scripts/build_literature.py`
- **Phase 7**: `src/models/surface_model.py`, `scripts/train_surface_model.py`
- **Phase 8**: `src/screening/`, `scripts/build_screening.py`
- **Phase 9**: `app.py`, `pages/`
- **Phase 10**: `tests/`, `README.md`, `docs/`

## Commands (canonical order)

```bash
source .venv/bin/activate
python scripts/build_dataset.py         # Phase 2: cache + folds + data card
python scripts/run_benchmark.py         # Phase 3: official 5-fold CV
python scripts/y_randomize.py           # Phase 4.3
python scripts/finalize_gate.py         # Phase 4.8: writes LEAKAGE_AUDIT.md
python scripts/run_analysis.py          # Phase 5 (blocked unless gate PASSED)
python scripts/build_literature.py      # Phase 6
python scripts/train_surface_model.py   # Phase 7 gate + optional model
python scripts/build_screening.py       # Phase 8
pytest -v 2>&1 | tee reports/pytest_run_$(date +%Y%m%d_%H%M%S).log
python -c "import app; print('APP_IMPORT_OK')"
streamlit run app.py
```

## Non-negotiable constraints

1. Official Matbench folds only; no custom benchmark split.
2. Composition features are computed inside each fold through the pipeline.
3. Identifiers never enter the feature matrix.
4. `LEAKAGE_GATE: PASSED` is required before Phase 5+.
5. Literature data is split by `study_id` if it is ever modelled.
6. `log_gvrh` is computed DFT modulus, never an implant-success measure.
7. No composite "suitability %" anywhere in code or UI.
8. Curation-only is a valid, complete Phase 7 outcome.
