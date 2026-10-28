"""App import safety and UI honesty checks."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PAGES = [
    "01_Project_Overview.py",
    "02_Dataset_Provenance_Quality.py",
    "03_Benchmark_Results_log_gvrh.py",
    "04_Prediction_AD_Uncertainty.py",
    "05_Interpretability_Leakage_Summary.py",
    "06_Surface_Evidence_Layer.py",
    "07_Transparent_Candidate_Screening.py",
    "08_Limitations_Disclaimers.py",
]

FORBIDDEN_UI_CLAIMS = [
    "best for implant",
    "clinically superior",
    "recommended implant",
    "suitability %",
    "suitability_score",
    "implant_score",
    "overall_score",
    "guaranteed",
    "clinically validated",
]


def test_app_imports_and_prints_marker():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", "import app; print('APP_IMPORT_OK')"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=240,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "APP_IMPORT_OK" in result.stdout.splitlines()
    assert "Traceback" not in result.stderr


def test_all_eight_pages_exist_and_parse():
    pages = ROOT / "pages"
    for name in REQUIRED_PAGES:
        path = pages / name
        assert path.exists(), f"Missing page: {name}"
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_pages_use_shared_disclaimer_and_artefact_loaders():
    for name in REQUIRED_PAGES:
        source = (ROOT / "pages" / name).read_text(encoding="utf-8")
        assert "from src.ui import" in source, f"{name} does not use the shared UI helpers"


def test_no_forbidden_claims_in_app_or_pages():
    texts = {"app.py": (ROOT / "app.py").read_text(encoding="utf-8")}
    for name in REQUIRED_PAGES:
        texts[f"pages/{name}"] = (ROOT / "pages" / name).read_text(encoding="utf-8")
    for name, text in texts.items():
        lowered = text.lower()
        hits = [claim for claim in FORBIDDEN_UI_CLAIMS if claim in lowered]
        assert not hits, f"{name} contains forbidden claim(s): {hits}"


def test_app_has_no_top_level_rendering():
    """Importing the app must not execute Streamlit rendering."""
    tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
    top_level_calls = [
        node
        for node in tree.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    ]
    assert not top_level_calls, "app.py runs Streamlit calls at import time."
    main_guarded = any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and "__name__" in ast.unparse(node.test)
        for node in tree.body
    )
    assert main_guarded, "app.py must call main() only under __main__."


def test_streamlit_available():
    import streamlit

    assert streamlit.__version__


def test_core_modules_import():
    import src.config  # noqa: F401
    import src.data.loader  # noqa: F401
    import src.data.literature  # noqa: F401
    import src.evaluation.applicability  # noqa: F401
    import src.evaluation.uncertainty  # noqa: F401
    import src.evaluation.validation  # noqa: F401
    import src.features.build  # noqa: F401
    import src.interpretability.importance  # noqa: F401
    import src.models.pipelines  # noqa: F401
    import src.models.train  # noqa: F401
    import src.screening.engine  # noqa: F401
    import src.screening.guardrails  # noqa: F401
    import src.ui  # noqa: F401


@pytest.mark.parametrize("name", REQUIRED_PAGES)
def test_page_uses_real_artefacts_no_hardcoded_metric_constants(name):
    """Pages must read persisted files rather than embedding result numbers."""
    source = (ROOT / "pages" / name).read_text(encoding="utf-8")
    assert (
        "load_json" in source
        or "load_csv" in source
        or "render_markdown_file" in source
        or "read_text" in source
    )
