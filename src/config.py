"""Central path and configuration handling for RP-1.

All modules resolve paths relative to the repository root so that commands can
be run from any working directory (including from Streamlit pages).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]

PATHS = {
    "root": ROOT,
    "configs": ROOT / "configs",
    "data_raw": ROOT / "data" / "raw",
    "data_processed": ROOT / "data" / "processed",
    "data_literature": ROOT / "data" / "literature",
    "data_cards": ROOT / "data_cards",
    "docs": ROOT / "docs",
    "reports": ROOT / "reports",
    "results": ROOT / "results",
    "splits": ROOT / "results" / "splits",
    "predictions": ROOT / "results" / "predictions",
    "metrics": ROOT / "results" / "metrics",
    "plots": ROOT / "results" / "plots",
    "interpretability": ROOT / "results" / "interpretability",
    "uncertainty": ROOT / "results" / "uncertainty",
    "validation": ROOT / "results" / "validation",
    "screening": ROOT / "results" / "screening",
    "tables": ROOT / "results" / "tables",
    "artifacts": ROOT / "artifacts",
}

DISCLAIMER = (
    "RESEARCH DEMONSTRATOR - Not for clinical use. No patient-specific advice. "
    "No implant recommendation implied."
)

TARGET_COLUMN = "log10(G_VRH)"
TARGET_NAME = "log_gvrh"
TARGET_UNITS = "log10(GPa)"


def ensure_dirs() -> None:
    """Create every output directory used by the pipeline."""
    for key, path in PATHS.items():
        if key in {"root", "configs", "docs"}:
            continue
        path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=None)
def load_config(name: str) -> dict[str, Any]:
    """Load a YAML config by stem (e.g. ``benchmark``)."""
    path = PATHS["configs"] / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def rel(path: Path | str) -> str:
    """Return a repository-relative path string for logs and reports."""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)
