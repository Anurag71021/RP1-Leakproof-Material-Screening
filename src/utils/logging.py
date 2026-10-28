"""Structured, file-backed logging.

Every gate-relevant action (split loading, per-fold featurization, dedup,
modelling decisions) is logged to ``reports/rp1.log`` with a UTC timestamp so
the project log can be reconstructed from machine evidence.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from src.config import PATHS

_FORMAT = "%(asctime)sZ %(levelname)-8s %(name)s | %(message)s"


def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(_FORMAT, datefmt="%Y-%m-%dT%H:%M:%S")
    formatter.converter = time.gmtime  # the trailing Z must mean UTC

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    target = log_file or (PATHS["reports"] / "rp1.log")
    target.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(target, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger
