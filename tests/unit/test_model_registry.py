"""
tests/unit/test_model_registry.py
Unit tests for model registry serialization, deserialization, and inference uncertainty.
"""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.models.registry import (
    evaluate_candidate_readiness,
    load_model,
    save_model,
    train_and_save_placement_model,
)


def test_save_and_load_model(tmp_path):
    clf = LogisticRegression()
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    clf.fit(X, y)

    meta = {"test_metric": 0.95}
    path = save_model(clf, "test_dummy_model", metadata=meta, artifacts_dir=tmp_path)
    assert path.exists()

    loaded_model, loaded_meta = load_model("test_dummy_model", artifacts_dir=tmp_path)
    assert loaded_model is not None
    assert loaded_meta.get("test_metric") == 0.95
    assert loaded_model.predict([[2.5]])[0] in [0, 1]


def test_train_and_save_placement_model():
    path = train_and_save_placement_model()
    assert path.exists()

    model, meta = load_model("placement_model")
    assert model is not None
    assert meta.get("epv_constrained") is True
    assert "degree_p" in meta.get("features", [])


def test_evaluate_candidate_readiness_ci_bounds():
    res = evaluate_candidate_readiness(
        ssc_p=70.0,
        hsc_p=72.0,
        degree_p=68.0,
        etest_p=80.0,
        mba_p=65.0,
        workex=True,
        specialisation="Mkt&Fin",
    )
    assert 0.0 <= res["placement_probability"] <= 1.0
    assert len(res["confidence_interval_95"]) == 2
    ci_low, ci_high = res["confidence_interval_95"]
    assert 0.0 <= ci_low <= res["placement_probability"] <= ci_high <= 1.0
    assert res["readiness_tier"] in [
        "High Employability",
        "Moderate Employability",
        "Needs Targeted Career Development",
    ]
    assert len(res["expected_salary_inr_range"]) == 2
    assert len(res["top_readiness_factors"]) >= 1
