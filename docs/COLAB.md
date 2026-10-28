# Run RP-1 in Google Colab

`colab_setup.py` is a one-command bootstrap for Colab. It creates a Python 3.10
environment, installs the pinned scientific stack, regenerates the gitignored
dataset/model artefacts, runs the tests, and can launch the Streamlit app through
a temporary Cloudflare tunnel.

It is deliberately **not** called `setup.py`: that name belongs to setuptools and
would interfere with this repository's `pyproject.toml` packaging.

---

## Two-step quick start

**Step 1 — clone and enter the repo**

```python
!git clone https://github.com/Anurag71021/RP1-Leakproof-Material-Screening.git
%cd RP1-Leakproof-Material-Screening
```

**Step 2 — set up, verify and launch the app**

```python
!python colab_setup.py --run --app
```

The script prints a `https://....trycloudflare.com` URL. Open it and keep the
Colab runtime alive.

---

## One-step version

```python
!git clone https://github.com/Anurag71021/RP1-Leakproof-Material-Screening.git \
  && cd RP1-Leakproof-Material-Screening \
  && python colab_setup.py --run --app
```

---

## What `--run` does

1. Installs `uv` if needed and downloads Python 3.10.
2. Creates `.venv` (or reuses it only if it is Python 3.10).
3. Installs the exact verified versions of NumPy, pandas, scikit-learn, SciPy,
   matminer, pymatgen and Streamlit.
4. Installs `matbench==0.6` with `--no-deps` to bypass its obsolete matminer pin.
5. Installs this repository in editable mode.
6. Runs `scripts/build_dataset.py` — downloads the official Matbench data and
   locks the five official folds.
7. Runs `scripts/build_screening.py` — fits the deployment model and writes
   `artifacts/model_log_gvrh.joblib`, `artifacts/ad_reference.joblib` and the
   screening tables.
8. Runs `pytest -q` and `import app`.

Expected final lines:

```text
75 passed
APP_IMPORT_OK
```

---

## Common options

| Command | Purpose |
| --- | --- |
| `python colab_setup.py` | Environment only; no data build, no app |
| `python colab_setup.py --run` | Environment + dataset + artefacts + tests |
| `python colab_setup.py --run --app` | Same, then launch Streamlit and print a tunnel URL |
| `python colab_setup.py --run --skip-tests` | Faster; skip pytest |
| `python colab_setup.py --run --force` | Rebuild the deployment artefacts |
| `python colab_setup.py --recreate-venv --run` | Delete and rebuild `.venv` |
| `python colab_setup.py --full` | Rebuild the whole scientific pipeline (20–40 min) |
| `python colab_setup.py --full --app` | Full rebuild, then launch the app |
| `python colab_setup.py --dry-run --run --app` | Print commands without executing |

`--app` uses port `8501` by default; change it with `--port 8502`.

---

## Runtime expectations

| Stage | Approximate time |
| --- | --- |
| Dependency install | 2–4 min |
| `build_dataset.py` | 1–3 min |
| `build_screening.py` | 5–15 min |
| Test suite | under 10 s |
| Full pipeline | 20–40 min |

Use a **CPU** runtime. No GPU is required.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `ResolutionImpossible` | Do not run `pip install -r requirements.txt`. Use `colab_setup.py`. |
| Existing `.venv` has the wrong Python | The script recreates it automatically; or use `--recreate-venv`. |
| Tests report missing artefacts | Run `python colab_setup.py --run --force`. |
| No tunnel URL | Check `reports/colab_tunnel.log`; quick tunnels are occasionally rate-limited. |
| Streamlit not healthy | Check `reports/colab_streamlit.log`; raise `--startup-timeout`. |
| Colab disconnected | Reconnect and rerun `--run --app`; `.venv` and downloaded data may survive. |

---

## Notes

- The repository is public, so cloning needs no token.
- The raw dataset cache (`data/raw/`), model binaries (`artifacts/`) and
  third-party full-text XML (`data/literature/sources/`) are gitignored and are
  regenerated or re-downloaded by the script.
- All committed results remain valid; `--run` only recreates the missing inputs.
- The app is a research demonstrator, not clinical decision support.
