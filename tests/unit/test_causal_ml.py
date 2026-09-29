"""
test_causal_ml.py — Unit Tests for Double Machine Learning & Causal Inference Engine
Student Success Intelligence Framework (SSIF)
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from src.causal.double_ml import (
    CATEEstimator,
    CausalEstimate,
    DoubleMLAIPW,
    DoubleMLPLR,
    compute_e_value,
)


@pytest.fixture
def synthetic_causal_dgp():
    """
    Generates synthetic data with known Ground Truth Average Treatment Effect (ATE = 2.5).
    Confounders: X_0, X_1, X_2.
    Treatment: D ~ Bernoulli(sigmoid(0.8*X_0 - 0.5*X_1))
    Outcome: Y = 2.5 * D + 1.2 * X_0 + 0.9 * X_1**2 + Normal(0, 1)
    """
    np.random.seed(42)
    n = 1500
    X = np.random.randn(n, 3)
    # Treatment propensity
    logit = 0.8 * X[:, 0] - 0.5 * X[:, 1]
    prob = 1.0 / (1.0 + np.exp(-logit))
    d = np.random.binomial(1, prob)
    # Outcome with true tau = 2.5
    y = 2.5 * d + 1.2 * X[:, 0] + 0.5 * (X[:, 1] ** 2) + np.random.randn(n) * 0.5

    df_X = pd.DataFrame(X, columns=["X0", "X1", "X2"])
    return df_X, pd.Series(d, name="D"), pd.Series(y, name="Y")


def test_e_value_properties():
    """E-value must be >= 1.0 and increase monotonically with effect magnitude."""
    e1 = compute_e_value(0.1, 0.05)
    e2 = compute_e_value(0.5, 0.05)
    e3 = compute_e_value(1.2, 0.05)

    assert e1 >= 1.0
    assert e2 > e1
    assert e3 > e2


def test_double_ml_plr_recovers_ground_truth(synthetic_causal_dgp):
    """DoubleML-PLR must estimate true treatment effect (tau=2.5) with high precision."""
    X, d, y = synthetic_causal_dgp
    plr = DoubleMLPLR(n_splits=3, random_state=42)
    est = plr.fit(X, d, y)

    assert isinstance(est, CausalEstimate)
    assert est.method == "DoubleML-PLR"
    assert est.p_value < 0.001
    assert np.isclose(est.ate, 2.5, atol=0.20), f"Estimated ATE {est.ate} not within 0.20 of true 2.5"
    assert est.ci_upper > est.ci_lower


def test_double_ml_aipw_doubly_robust(synthetic_causal_dgp):
    """DoubleML-AIPW must estimate true treatment effect (tau=2.5) within 95% CI."""
    X, d, y = synthetic_causal_dgp
    aipw = DoubleMLAIPW(n_splits=3, random_state=42)
    est = aipw.fit(X, d, y)

    assert isinstance(est, CausalEstimate)
    assert est.method == "DoubleML-AIPW"
    assert est.p_value < 0.001
    assert est.ci_lower <= 2.5 <= est.ci_upper
    assert np.isclose(est.ate, 2.5, atol=0.25)


def test_cate_heterogeneous_treatment_effect():
    """
    Tests heterogeneous treatment effect recovery:
    Y = (1.0 + 2.0 * V) * D + X + noise
    Interaction coefficient on V must be positive and close to +2.0.
    """
    np.random.seed(42)
    n = 2000
    X_cov = np.random.randn(n, 2)
    V = np.random.choice([0, 1], size=n)
    d = np.random.binomial(1, 0.5, size=n)
    # Effect is 1.0 for V=0, and 3.0 for V=1
    y = (1.0 + 2.0 * V) * d + 0.8 * X_cov[:, 0] + np.random.randn(n) * 0.5

    df_X = pd.DataFrame(X_cov, columns=["X0", "X1"])
    df_X["V"] = V

    cate = CATEEstimator(dml_plr=DoubleMLPLR(n_splits=3, random_state=42))
    res = cate.fit_cate(df_X, pd.Series(d), pd.Series(y), modifiers=["V"])

    assert "V" in res.interaction_terms
    assert res.interaction_terms["V"] > 1.2  # Expected near +2.0
    assert res.interaction_p_values["V"] < 0.01
    assert "V=0" in res.subgroup_effects
    assert "V=1" in res.subgroup_effects
    assert res.subgroup_effects["V=1"].ate > res.subgroup_effects["V=0"].ate
