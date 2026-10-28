"""Shared UI helpers for the Streamlit demonstrator.

Every number shown in the app is read from a persisted artefact through these
helpers. Nothing is recomputed for display except lightweight derived views
(counts, tables, charts) of already-persisted data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.config import DISCLAIMER, PATHS, rel


def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def disclaimer() -> None:
    st.error(DISCLAIMER)


def gate_status() -> tuple[bool, str]:
    path = PATHS["docs"] / "LEAKAGE_AUDIT.md"
    if not path.exists():
        return False, "LEAKAGE_AUDIT.md missing"
    text = path.read_text(encoding="utf-8")
    if "LEAKAGE_GATE: PASSED" in text:
        return True, "PASSED"
    if "LEAKAGE_GATE: FAILED" in text:
        return False, "FAILED"
    return False, "UNDECLARED"


def gate_badge() -> None:
    passed, state = gate_status()
    if passed:
        st.success(f"Leakage gate: {state}")
    else:
        st.error(f"Leakage gate: {state} - phases 5-9 must not be presented as validated.")


def load_json(path: Path, default: Any = None) -> Any:
    if not Path(path).exists():
        return default
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_csv(path: Path) -> pd.DataFrame | None:
    if not Path(path).exists():
        return None
    return pd.read_csv(path)


def read_text(path: Path) -> str | None:
    if not Path(path).exists():
        return None
    return Path(path).read_text(encoding="utf-8")


def missing(path: Path, command: str) -> None:
    st.warning(
        f"Artefact not found: `{rel(path)}`. Run `{command}` to generate it. "
        "No placeholder numbers are displayed."
    )


def artefact_caption(*paths: Path) -> None:
    st.caption("Sources: " + ", ".join(f"`{rel(p)}`" for p in paths))


def yes_no(value: Any) -> str:
    return "YES" if bool(value) else "NO"


def show_dataframe(frame: pd.DataFrame | None, *, empty: str = "No rows.", **kwargs) -> None:
    if frame is None or frame.empty:
        st.info(empty)
        return
    st.dataframe(frame, use_container_width=True, **kwargs)


def render_markdown_file(path: Path, command: str) -> None:
    text = read_text(path)
    if text is None:
        missing(path, command)
        return
    st.markdown(text)


def csv_download(frame: pd.DataFrame | None, filename: str, label: str) -> None:
    if frame is None or frame.empty:
        return
    st.download_button(
        label,
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )
