"""
tests/unit/test_calibration.py
Unit tests for probability calibration diagnostics, ECE, MCE, and reliability curves.
"""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.models.base import (
    ClassificationMetrics,
    calibrate_classifier,
    compute_calibration_curve,
    compute_classification_metrics,
    compute_ece,
    compute_mce,
)


def test_compute_ece_perfect_calibration():
    # If predicted probability matches empirical label rate perfectly
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_prob = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.95, 0.95, 0.95, 0.95, 0.95])
    ece = compute_ece(y_true, y_prob, n_bins=10)
    # ECE should be very low (< 0.1)
    assert ece < 0.10


def test_compute_ece_uncalibrated_overconfident():
    # Model always predicts 0.99 confidence but labels are 50/50
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_prob = np.array([0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99])
    ece = compute_ece(y_true, y_prob, n_bins=5)
    # Predicted is 0.99, empirical is 0.50 -> ECE should be ~0.49
    assert np.isclose(ece, 0.49, atol=0.02)


def test_compute_mce_bound():
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_prob = np.array([0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99])
    mce = compute_mce(y_true, y_prob, n_bins=5)
    assert mce >= compute_ece(y_true, y_prob, n_bins=5)
    assert np.isclose(mce, 0.49, atol=0.02)


def test_compute_calibration_curve_structure():
    y_true = np.random.RandomState(42).binomial(1, 0.3, size=200)
    y_prob = np.random.RandomState(42).uniform(0, 1, size=200)
    curve = compute_calibration_curve(y_true, y_prob, n_bins=10)

    assert "prob_true" in curve
    assert "prob_pred" in curve
    assert "bin_counts" in curve
    assert "bin_edges" in curve
    assert "ece" in curve
    assert "mce" in curve
    assert "brier_score" in curve
    assert len(curve["prob_true"]) == 10
    assert len(curve["prob_pred"]) == 10
    assert sum(curve["bin_counts"]) == 200


def test_calibrate_classifier_wrapper():
    X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 5.0], [5.0, 6.0], [6.0, 7.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    base_clf = LogisticRegression()
    calibrated = calibrate_classifier(base_clf, method="sigmoid", cv=2)
    calibrated.fit(X, y)
    probs = calibrated.predict_proba(X)
    assert probs.shape == (6, 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_classification_metrics_contains_calibration_fields():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_prob = np.array([0.2, 0.8, 0.3, 0.9, 0.7, 0.1])
    metrics = compute_classification_metrics(y_true, y_prob, model_name="TestLR")

    assert hasattr(metrics, "brier_score")
    assert hasattr(metrics, "ece")
    assert hasattr(metrics, "mce")
    assert hasattr(metrics, "reliability_curve")
    assert metrics.ece >= 0.0
    assert metrics.mce >= metrics.ece - 1e-6
    assert "MCE:" in metrics.summary()
