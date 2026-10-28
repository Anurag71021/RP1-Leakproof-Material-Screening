#!/usr/bin/env python
"""Phase 3: train baselines with official folds and save OOF predictions."""

from __future__ import annotations

import argparse
import json

from src.config import PATHS
from src.data.loader import load_fold_indices, load_cached
from src.models.pipelines import MODEL_NAMES
from src.models.train import run_official_cv
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(MODEL_NAMES))
    parser.add_argument("--featurizer", default="magpie", choices=["magpie", "simple_composition"])
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()

    frame, provenance = load_cached()
    folds = load_fold_indices(frame)
    if args.download_only:
        print(json.dumps({"rows": len(frame), "folds": sorted(folds)}, indent=2))
        return

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in MODEL_NAMES]
    if unknown:
        raise SystemExit(f"Unknown models: {unknown}. Available: {MODEL_NAMES}")

    results = run_official_cv(
        frame,
        folds,
        models,
        featurizer_name=args.featurizer,
        random_state=args.random_state,
    )
    print(json.dumps({m: v["summary"] for m, v in results["models"].items()}, indent=2))
    print(f"Metrics: {PATHS['metrics'] / 'benchmark_metrics.json'}")


if __name__ == "__main__":
    main()
