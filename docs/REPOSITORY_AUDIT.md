# Repository audit - Phase 0

**Date:** 2026-10-28 (UTC timestamps in `reports/environment.txt`)
**Scope:** discovery only. No source code changes were made before this audit.
**Evidence:** `reports/environment.txt`, `reports/data_validation.json`, this file.

---

## 1. What was found

The working directory `/Users/anurag/Desktop/Projects_RP` contained **no application
code** at the start of the project:

```
RP1_Leakproof_Implementation_Plan.md   (the specification, preserved untouched)
opencode.jsonc                         (editor/MCP configuration, preserved untouched)
```

There was no `src/`, `tests/`, `app.py`, `requirements.txt`, dataset loader,
Matbench test, or Git repository. Consequently:

- There is **no existing Matbench experimental band-gap loader** to preserve.
  The plan's "preserve > replace" rule is therefore vacuous for this checkout,
  and no working code was erased.
- There are **no pre-existing tests** to keep passing. All tests in this
  repository were written for this implementation.
- The plan itself is treated as read-only input and is not modified.

## 2. Environment

| Item | Observed value |
| --- | --- |
| OS / architecture | macOS 26.6.1, arm64 (Apple Silicon) |
| System Python | 3.9.6 (`/usr/bin/python3`) |
| Project interpreter | Python 3.10.18 (`/opt/homebrew/bin/python3.10`) inside `.venv/` |
| Package manager | pip 26.2.2 |
| Free disk at start | ~28 GiB |
| Git repository | **No** at Phase 0 (`fatal: not a git repository`) |
| Network | Matbench host and PyPI reachable (HTTP 200) |

The plan recommends Python 3.10.x for matminer/pymatgen compatibility on Apple
Silicon. That recommendation was followed by creating `.venv` with Homebrew
Python 3.10 rather than using the 3.9 system interpreter.

## 3. Dependency findings and one real blocker

`matbench==0.6` declares a hard pin `matminer==0.7.4`. That matminer version has
no Python 3.10 / arm64 distribution, so a plain
`pip install matbench matminer` fails with `ResolutionImpossible`.

Resolution used and verified:

1. Install current `matminer` (0.9.3) and `pymatgen` (2025.10.7).
2. Install `matbench==0.6` with `--no-deps`.
3. Verify `from matbench.bench import MatbenchBenchmark` and a full official
   dataset load succeed.

This is recorded in `docs/PROJECT_LOG.md`. The pin is a packaging constraint,
not a scientific one; the official dataset and fold definitions come from the
official API unchanged.

A second environment issue: matminer's default multiprocessing featurizer
spawns processes that fail when the entry point is stdin / Streamlit. All
featurization is therefore forced to `n_jobs=1` and run in-process.

## 4. Keep / modify / create

**Keep (unmodified):**

- `RP1_Leakproof_Implementation_Plan.md`
- `opencode.jsonc`

**Create:**

- `src/` package (`data`, `features`, `models`, `evaluation`, `interpretability`,
  `screening`, `utils`)
- `configs/benchmark.yaml`, `configs/featurization.yaml`
- `scripts/` reproducibility entry points
- `tests/` (5 required suites + shared fixtures)
- `pages/` and `app.py` (8-page Streamlit demonstrator)
- `docs/`, `data_cards/`, `results/`, `reports/`
- `requirements.txt`, `pyproject.toml`, `.gitignore`, `.env.example`, `README.md`

**Modify:** nothing pre-existing.

## 5. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Matbench/matminer version conflict on Apple Silicon | `matbench --no-deps` + current matminer, verified import and load |
| Accidental leakage through global featurization | Stateless featurizer, per-fold `Pipeline.fit`, explicit leakage tests |
| `log_gvrh` misread as implant performance | Wording rules in protocol, data card, UI and screening constraints |
| Literature data insufficiency | Modelling gate with curation-only as a valid terminal state |
| Streamlit/matminer import cost | App reads artefacts; imports guarded; explicit import test |
| Random Forest runtime | 300 trees, `n_jobs=-1`, 5 official folds; runtime logged |
| No Git history | Repository initialised and committed after artefacts exist |

## 6. Phase 0 exit criteria

- [x] Environment scanned (`reports/environment.txt`)
- [x] Dependencies resolved and import-verified
- [x] Codebase mapped (empty application, specification only)
- [x] Network and resources checked
- [x] No destructive edits; nothing to preserve was overwritten
- [x] Audit written (`docs/REPOSITORY_AUDIT.md`)
- [x] Implementation plan written (`docs/IMPLEMENTATION_PLAN.md`)

**Blockers:** none at Phase 0. The matbench/matminer pin was resolved before any
modelling, with evidence in `docs/PROJECT_LOG.md`.
