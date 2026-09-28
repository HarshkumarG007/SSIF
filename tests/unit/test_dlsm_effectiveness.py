"""
test_dlsm_effectiveness.py — Unit tests for DLSM effectiveness ablation experiment.
Student Success Intelligence Framework (SSIF)
"""
import pytest
from src.dlsm.effectiveness_test import (
    DLSMFeatureAblationResult,
    run_dlsm_effectiveness_ablation,
)


def test_dlsm_effectiveness_ablation():
    # Run ablation with 2 splits for quick unit test validation
    res = run_dlsm_effectiveness_ablation(n_splits=2, random_state=42)
    assert isinstance(res, DLSMFeatureAblationResult)
    assert res.a0_metrics.auroc > 0.70
    assert res.a1_metrics.auroc > 0.70
    # Difference must be negligible (absolute delta < 0.01)
    assert abs(res.delta_auroc) < 0.01
    assert "NO INCREMENTAL PREDICTIVE VALUE" in res.scientific_verdict
    assert len(res.a0_metrics.fold_aurocs) == 2
    assert len(res.a1_metrics.fold_aurocs) == 2
