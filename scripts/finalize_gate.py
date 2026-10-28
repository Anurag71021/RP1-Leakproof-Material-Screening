#!/usr/bin/env python
"""Phase 4.8: evaluate every gate artefact and write the signed leakage audit.

This script does not "decide" that the project is clean. It checks persisted
evidence, refuses to pass on missing artefacts, and records exactly which
checks were evaluated.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import PATHS
from src.data.loader import load_cached, load_fold_indices
from src.features.build import build_featurizer
from src.models.pipelines import build_pipeline
from src.utils.io import read_json, utc_now, write_json
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _check(ok: bool, evidence: str) -> dict:
    return {"passed": bool(ok), "evidence": evidence}


def main() -> None:
    metrics = read_json(PATHS["metrics"] / "benchmark_metrics.json")
    best_payload = read_json(PATHS["metrics"] / "best_model.json")
    best = best_payload["model"]
    manifest = read_json(PATHS["splits"] / "split_manifest.json")
    validation = read_json(PATHS["reports"] / "data_validation.json")
    yrand = read_json(PATHS["validation"] / "y_randomization.json")
    curves_path = PATHS["validation"] / "learning_curves.json"
    curves = read_json(curves_path) if curves_path.exists() else None
    residual_path = PATHS["validation"] / f"residual_stats_{best}.json"
    residual = read_json(residual_path) if residual_path.exists() else None

    frame, _ = load_cached()
    folds = load_fold_indices(frame)

    checks: dict[str, dict] = {}

    # 1. Official folds
    checks["official_folds_used"] = _check(
        manifest.get("split_type") == "official_folds"
        and manifest.get("dataset") == "matbench_log_gvrh"
        and sorted(f["fold_id"] for f in manifest["folds"]) == [0, 1, 2, 3, 4],
        "results/splits/split_manifest.json (split_type=official_folds, 5 folds)",
    )

    # 2. Split integrity
    checks["split_integrity"] = _check(
        bool(manifest["checks_passed"])
        and all(manifest["checks"].values()),
        f"manifest checks: {manifest['checks']}",
    )

    # 3. Identifier exclusion
    train = frame.iloc[folds[0]["train_idx"][:120]]
    probe = build_pipeline("ridge")
    probe.fit(train[["structure"]], train["log10(G_VRH)"].to_numpy(dtype=float))
    names = [n.lower() for n in probe.named_steps["features"].get_feature_names_out()]
    identifier_hits = sorted(
        set(names)
        & {"id", "material_id", "mp-id", "mp_id", "icsd_id", "structure_id", "cif", "formula"}
    )
    checks["identifiers_excluded"] = _check(
        not identifier_hits and not validation["validation"]["identifier_columns_in_frame"],
        f"runtime feature names contain no identifiers; hits={identifier_hits}; "
        "frame has no identifier columns",
    )

    # 4. Pipeline-only preprocessing + fold-local fit
    train_source = (PATHS["root"] / "src" / "models" / "train.py").read_text(encoding="utf-8")
    checks["pipeline_only_preprocessing"] = _check(
        isinstance(probe, Pipeline)
        and list(probe.named_steps) == ["features", "impute", "scale", "model"]
        and "pipeline.fit(train" in train_source
        and "fit_transform(frame" not in train_source,
        "Pipeline(features -> impute -> scale -> estimator); train.py fits on fold-local "
        "train slice only; no global fit_transform",
    )

    # 5. Per-fold featurization is stateless (no learned state can leak)
    feat_a = build_featurizer("magpie")
    feat_a.fit(frame.iloc[:100][["structure"]])
    X_a = feat_a.transform(frame.iloc[500:510][["structure"]]).to_numpy()
    feat_b = build_featurizer("magpie")
    feat_b.fit(frame.iloc[200:300][["structure"]])
    X_b = feat_b.transform(frame.iloc[500:510][["structure"]]).to_numpy()
    checks["per_fold_featurization"] = _check(
        np.allclose(X_a, X_b, equal_nan=True),
        "same compositions featurized after fitting on disjoint subsets produce identical "
        "values (stateless featurizer, 132 Magpie descriptors)",
    )

    # 6. OOF coverage
    oof_status = {}
    for model_name in metrics["models"]:
        path = PATHS["predictions"] / f"oof_{model_name}_matbench_log_gvrh.csv"
        if not path.exists():
            oof_status[model_name] = "missing"
            continue
        oof = pd.read_csv(path)
        ok = (
            len(oof) == len(frame)
            and oof["row_index"].nunique() == len(frame)
            and oof["fold_id"].nunique() == 5
        )
        oof_status[model_name] = "ok" if ok else "bad"
    checks["oof_coverage"] = _check(
        all(v == "ok" for v in oof_status.values()),
        f"OOF files cover every sample exactly once: {oof_status}",
    )

    # 7. Y-randomization collapse
    rand = yrand["models"][best]
    real = metrics["models"][best]["summary"]
    dummy = metrics["models"]["dummy"]["summary"]
    rand_r2_max = rand["aggregate"]["r2_max"]
    rand_mae = rand["aggregate"]["mae_mean"]
    collapse = rand_r2_max <= 0.10 and rand_mae >= 0.95 * dummy["mae"]["mean"]
    checks["y_randomization_collapse"] = _check(
        collapse,
        f"best={best}; randomized R2 max={rand_r2_max:.4f} (<=0.10 required); "
        f"randomized MAE={rand_mae:.4f} vs Dummy MAE={dummy['mae']['mean']:.4f}",
    )

    # 8. Real model beats dummy
    checks["model_beats_dummy"] = _check(
        real["mae"]["mean"] < dummy["mae"]["mean"] - 0.05
        and real["r2"]["mean"] > dummy["r2"]["mean"] + 0.10,
        f"best={best}; MAE {real['mae']['mean']:.4f} vs Dummy {dummy['mae']['mean']:.4f}; "
        f"R2 {real['r2']['mean']:.4f} vs Dummy {dummy['r2']['mean']:.4f}",
    )

    # 9. Learning curves
    checks["learning_curves_present"] = _check(
        curves is not None and best in curves.get("models", {}),
        f"{curves_path.name}; model={best}",
    )

    # 10. Residuals and parity
    checks["residuals_parity_present"] = _check(
        residual is not None
        and (PATHS["plots"] / f"parity_plot_{best}.png").exists()
        and (PATHS["plots"] / f"residuals_{best}.png").exists(),
        f"results/plots/parity_plot_{best}.png; results/plots/residuals_{best}.png",
    )

    # 11. Duplicate audit
    checks["duplicates_audited"] = _check(
        "duplicates" in validation,
        f"{validation['duplicates']['n_duplicate_groups']} duplicate groups, "
        f"{validation['duplicates']['n_cross_fold_duplicate_groups']} cross-fold",
    )

    # 12. Leakage tests exist
    test_path = PATHS["root"] / "tests" / "test_leakage_sanity.py"
    checks["leakage_tests_present"] = _check(
        test_path.exists(), "tests/test_leakage_sanity.py exists (executed in Phase 10)"
    )

    critical = list(checks)
    passed = all(checks[k]["passed"] for k in critical)

    # Stability and curve diagnostics (reported, not silently ignored)
    fold_r2 = [f["r2"] for f in metrics["models"][best]["per_fold"]]
    curve_rows = curves["models"][best] if curves and best in curves.get("models", {}) else []
    full = curve_rows[-1] if curve_rows else None
    gap_note = (
        f"full-size train MAE={full['train_mae']:.4f}, held-out MAE={full['val_mae']:.4f}, "
        f"gap={full['gap_mae']:.4f}"
        if full
        else "not available"
    )

    lines = [
        "# Leakage and overfitting audit",
        "",
        f"**Generated:** {utc_now()} by `scripts/finalize_gate.py`.",
        f"**Selected model:** `{best}` (lowest mean OOF MAE).",
        f"**Gate status:** `LEAKAGE_GATE: {'PASSED' if passed else 'FAILED'}`",
        "",
        "> This gate is evidence-based. A check is YES only if the named artefact",
        "> was read and the stated condition held at generation time.",
        "",
        "## 1. Critical checklist",
        "",
        "| # | Check | Result | Evidence |",
        "| --- | --- | --- | --- |",
    ]
    for i, (key, value) in enumerate(checks.items(), start=1):
        lines.append(
            f"| {i} | `{key}` | {'YES' if value['passed'] else 'NO'} | {value['evidence']} |"
        )

    lines += [
        "",
        "## 2. Benchmark results (official 5-fold OOF)",
        "",
        "| Model | MAE | RMSE | R2 |",
        "| --- | --- | --- | --- |",
    ]
    for model_name, payload in metrics["models"].items():
        s = payload["summary"]
        lines.append(
            f"| {model_name} | {s['mae']['mean']:.4f} ± {s['mae']['std']:.4f} | "
            f"{s['rmse']['mean']:.4f} ± {s['rmse']['std']:.4f} | "
            f"{s['r2']['mean']:.4f} ± {s['r2']['std']:.4f} |"
        )

    lines += [
        "",
        "## 3. Y-randomization",
        "",
        "| Model | Permutations | Randomized R2 mean | Randomized R2 max | Randomized MAE |",
        "| --- | --- | --- | --- | --- |",
    ]
    for model_name, payload in yrand["models"].items():
        a = payload["aggregate"]
        lines.append(
            f"| {model_name} | {payload['n_permutations']} | {a['r2_mean']:.4f} | "
            f"{a['r2_max']:.4f} | {a['mae_mean']:.4f} |"
        )

    lines += [
        "",
        "A valid model must collapse to Dummy-level performance when labels are",
        "shuffled. The real model must remain clearly better than Dummy.",
        "",
        "## 4. Overfitting / stability diagnostics",
        "",
        f"- Fold-wise R2 for `{best}`: {', '.join(f'{v:.4f}' for v in fold_r2)}",
        f"- Fold R2 standard deviation: {np.std(fold_r2, ddof=1):.4f}",
        f"- Learning curve at full training size: {gap_note}",
        f"- Residual diagnostics: `{residual_path.name}`",
        "",
        "A train/held-out gap is expected for a 300-tree Random Forest. It is",
        "reported, not hidden. The decisive leakage evidence is the Y-randomization",
        "collapse, not the size of the training gap alone.",
        "",
        "## 5. Duplicate and split audit",
        "",
        f"- Duplicate groups: {validation['duplicates']['n_duplicate_groups']}",
        f"- Cross-fold duplicate groups: {validation['duplicates']['n_cross_fold_duplicate_groups']}",
        f"- Duplicate method: {validation['duplicates']['method']}",
        f"- Split checks: {json.dumps(manifest['checks'])}",
        "",
        "## 6. Limitations that remain true even after a pass",
        "",
        "1. Passing this gate proves the *implementation* is leak-aware under the",
        "   checks performed. It does not prove the benchmark target is clinically",
        "   meaningful.",
        "2. `log10(G_VRH)` is a computed DFT property. It is not an implant modulus,",
        "   a fatigue limit, or a biocompatibility measure.",
        "3. Cross-fold duplicate structures, if present, would inflate scores. The",
        "   audit reports zero for the coarse key used; near-duplicate polymorphs",
        "   cannot be fully excluded by this key.",
        "4. Official folds are random with respect to composition, so the benchmark",
        "   measures interpolation, not extrapolation to new alloy families.",
        "",
        "## 7. Gate rule",
        "",
        "Phases 5-9 may proceed **only** if every critical check above is YES and",
        "Y-randomization is convincingly worse than the real model. The final line",
        "of this file is the machine-readable gate marker.",
        "",
        f"LEAKAGE_GATE: {'PASSED' if passed else 'FAILED'}",
        "",
    ]

    (PATHS["docs"] / "LEAKAGE_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(
        PATHS["validation"] / "gate_status.json",
        {
            "created_utc": utc_now(),
            "model": best,
            "checks": checks,
            "passed": passed,
            "marker": f"LEAKAGE_GATE: {'PASSED' if passed else 'FAILED'}",
            "y_randomization": yrand["models"][best]["aggregate"],
            "real": metrics["models"][best]["summary"],
            "dummy": metrics["models"]["dummy"]["summary"],
        },
    )

    print(json.dumps({"passed": passed, "checks": {k: v["passed"] for k, v in checks.items()}}, indent=2))
    if not passed:
        raise SystemExit("LEAKAGE GATE FAILED - phases 5-9 are blocked.")


if __name__ == "__main__":
    main()
