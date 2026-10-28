#!/usr/bin/env python
"""Phase 4: mandatory Y-randomization on official folds."""

from __future__ import annotations

import argparse
import json

from src.data.loader import load_cached, load_fold_indices
from src.evaluation.validation import y_randomization
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="ridge,random_forest")
    parser.add_argument("--featurizer", default="magpie")
    parser.add_argument("--ridge-perms", type=int, default=5)
    parser.add_argument("--rf-perms", type=int, default=2)
    args = parser.parse_args()

    frame, _ = load_cached()
    folds = load_fold_indices(frame)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    payload = y_randomization(
        frame,
        folds,
        models,
        n_permutations={"ridge": args.ridge_perms, "random_forest": args.rf_perms},
        featurizer_name=args.featurizer,
    )
    print(json.dumps({m: v["aggregate"] for m, v in payload["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
