"""Critical leakage checks.

These tests encode the project's hard rules:

* identifiers never become features;
* every preprocessing step is fitted inside a fold;
* the featurizer is stateless, so "per-fold featurization" is structural rather
  than a label-dependent transform;
* OOF predictions cover every row exactly once and reproduce the reported
  metrics;
* Y-randomization collapses to Dummy-level performance;
* the signed gate marker exists and every recorded check passed.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.config import PATHS, TARGET_COLUMN
from src.evaluation.metrics import regression_metrics
from src.features.build import build_featurizer
from src.models.pipelines import build_pipeline, feature_names
from src.models.train import load_oof

IDENTIFIERS = {"id", "material_id", "mp-id", "mp_id", "icsd_id", "structure_id", "cif", "formula"}


def test_pipeline_structure_and_no_identifiers(benchmark_frame, folds):
    train = benchmark_frame.iloc[folds[0]["train_idx"][:150]]
    pipeline = build_pipeline("ridge")
    assert list(pipeline.named_steps) == ["features", "impute", "scale", "model"]
    pipeline.fit(train[["structure"]], train[TARGET_COLUMN].to_numpy(dtype=float))
    names = [n.lower() for n in feature_names(pipeline)]
    assert names, "Pipeline produced no feature names."
    assert not (IDENTIFIERS & set(names))
    assert len(names) == 132


def test_featurizer_is_stateless_across_subsets(benchmark_frame):
    probe = benchmark_frame.iloc[500:520][["structure"]]
    feat_a = build_featurizer("magpie").fit(benchmark_frame.iloc[:100][["structure"]])
    feat_b = build_featurizer("magpie").fit(benchmark_frame.iloc[5000:5200][["structure"]])
    np.testing.assert_allclose(
        feat_a.transform(probe).to_numpy(),
        feat_b.transform(probe).to_numpy(),
        equal_nan=True,
        err_msg="Featurizer values depend on the fitting subset; it is not stateless.",
    )


def test_no_global_fit_in_training_source():
    source = (PATHS["root"] / "src" / "models" / "train.py").read_text(encoding="utf-8")
    assert "pipeline.fit(train" in source
    assert "pipeline.fit(frame" not in source
    assert "fit_transform(frame" not in source
    assert "pipeline.predict(test" in source


@pytest.mark.parametrize("model_name", ["dummy", "ridge", "random_forest"])
def test_oof_covers_every_row_once(model_name, benchmark_frame, metrics):
    oof = load_oof(model_name)
    assert len(oof) == len(benchmark_frame)
    assert oof["row_index"].nunique() == len(benchmark_frame)
    assert sorted(oof["fold_id"].unique()) == [0, 1, 2, 3, 4]
    assert np.isfinite(oof["y_pred"]).all()
    assert len(metrics["models"][model_name]["per_fold"]) == 5


@pytest.mark.parametrize("model_name", ["dummy", "ridge", "random_forest"])
def test_oof_reproduces_reported_fold_metrics(model_name, metrics):
    oof = load_oof(model_name)
    for record in metrics["models"][model_name]["per_fold"]:
        fold_rows = oof[oof["fold_id"] == record["fold_id"]]
        recomputed = regression_metrics(fold_rows["y_true"], fold_rows["y_pred"])
        assert recomputed["mae"] == pytest.approx(record["mae"], abs=1e-9)
        assert recomputed["rmse"] == pytest.approx(record["rmse"], abs=1e-9)
        assert recomputed["r2"] == pytest.approx(record["r2"], abs=1e-9)


def test_y_randomization_collapses(best_model, metrics, y_randomization):
    model = best_model["model"]
    randomized = y_randomization["models"][model]["aggregate"]
    dummy_mae = metrics["models"]["dummy"]["summary"]["mae"]["mean"]
    real_mae = metrics["models"][model]["summary"]["mae"]["mean"]

    assert randomized["r2_max"] <= 0.10
    assert randomized["r2_mean"] <= 0.05
    assert randomized["mae_mean"] >= 0.95 * dummy_mae
    # And the real model must be clearly better than both.
    assert real_mae < dummy_mae - 0.05


def test_gate_file_is_signed_and_all_checks_pass():
    audit = (PATHS["docs"] / "LEAKAGE_AUDIT.md").read_text(encoding="utf-8")
    assert "LEAKAGE_GATE: PASSED" in audit
    status = json.loads((PATHS["validation"] / "gate_status.json").read_text())
    assert status["passed"] is True
    assert all(status["checks"][key]["passed"] for key in status["checks"])
    assert "LEAKAGE_GATE: PASSED" in status["marker"]


def test_gate_audit_does_not_claim_causality_or_clinical_meaning():
    audit = (PATHS["docs"] / "LEAKAGE_AUDIT.md").read_text(encoding="utf-8").lower()
    assert "not an implant modulus" in audit
    assert "computed" in audit
    assert "does not prove" in audit


def test_learning_curves_and_residuals_exist_for_selected_model(best_model):
    model = best_model["model"]
    curves = json.loads((PATHS["validation"] / "learning_curves.json").read_text())
    assert model in curves["models"]
    assert len(curves["models"][model]) >= 3
    assert (PATHS["plots"] / f"parity_plot_{model}.png").exists()
    assert (PATHS["plots"] / f"residuals_{model}.png").exists()
    residual = json.loads((PATHS["validation"] / f"residual_stats_{model}.json").read_text())
    assert residual["n"] == 10987


def test_permutation_importance_has_no_identifiers(best_model):
    display = best_model["model"]
    path = PATHS["interpretability"] / f"permutation_importance_{display}.csv"
    importance = pd.read_csv(path)
    assert len(importance) == 132
    assert not (IDENTIFIERS & {f.lower() for f in importance["feature"]})
    assert importance["importance_mean"].notna().all()


def test_identifiers_are_not_in_saved_deployment_features():
    import joblib

    path = PATHS["artifacts"] / "model_log_gvrh.joblib"
    bundle = joblib.load(path)
    names = {n.lower() for n in bundle["feature_names"]}
    assert not (IDENTIFIERS & names)
    assert "deployment" in bundle["caveat"].lower()
