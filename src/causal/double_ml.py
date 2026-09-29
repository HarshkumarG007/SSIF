"""
double_ml.py — Double / Debiased Machine Learning & Causal Inference Engine
Student Success Intelligence Framework (SSIF)

Implements:
  1. Neyman-Orthogonal Double Machine Learning for Partially Linear Models (PLR)
     (Chernozhukov et al., 2018) with K-Fold & GroupKFold cross-fitting (RULE-004).
  2. Augmented Inverse Probability Weighting (AIPW) Doubly-Robust Estimator for binary treatments.
  3. Conditional Average Treatment Effect (CATE) decomposition over socio-demographic modifiers
     (e.g., First_Generation, Financial_Stress, Household_Size).
  4. Sensitivity Analysis to Unobserved Confounders (VanderWeele & Ding E-Value).

Governed by:
  - RULE-004: GroupKFold on student grouping to prevent cross-semester contamination.
  - RULE-007: Out-of-fold cross-fitting for all nuisance estimators.
  - RULE-009: Strict temporal precedence of confounders relative to treatment and outcome.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold, GroupKFold

from src.logger import get_module_logger

logger = get_module_logger("causal.double_ml")


@dataclass
class CausalEstimate:
    """Represents a rigorous causal estimate with confidence intervals and sensitivity."""
    ate: float
    se: float
    ci_lower: float
    ci_upper: float
    p_value: float
    z_stat: float
    e_value: float
    method: str
    n_samples: int
    nuisance_r2_y: float = 0.0
    nuisance_r2_d: float = 0.0

    def summary(self) -> str:
        sig = "***" if self.p_value < 0.001 else "**" if self.p_value < 0.01 else "*" if self.p_value < 0.05 else "ns"
        return (
            f"[{self.method}] ATE = {self.ate:+.4f} (SE: {self.se:.4f}, 95% CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}], "
            f"p={self.p_value:.4e} {sig}, E-Value: {self.e_value:.2f}, N={self.n_samples})"
        )


@dataclass
class HeterogeneousCausalEstimate:
    """Conditional Average Treatment Effects (CATE) across subgroup modifiers."""
    base_ate: float
    subgroup_effects: dict[str, CausalEstimate]
    interaction_terms: dict[str, float]
    interaction_p_values: dict[str, float]


def compute_e_value(effect: float, se: float, is_risk_ratio: bool = False) -> float:
    """
    Computes the VanderWeele & Ding (2017) E-value for unmeasured confounding.
    The E-value is the minimum strength of association that an unmeasured confounder
    would need to have with both treatment and outcome to explain away the observed effect.
    """
    if is_risk_ratio:
        rr = max(effect, 1e-6)
    else:
        # Approximate risk ratio from standardized linear difference
        rr = float(np.exp(effect))

    if rr < 1.0:
        rr = 1.0 / rr

    if rr <= 1.0:
        return 1.0

    e_val = rr + np.sqrt(rr * (rr - 1.0))
    return float(e_val)


class DoubleMLPLR:
    """
    Double Machine Learning Partially Linear Regression (DML-PLR).
    Model:
      Y = theta * D + g(X) + U,   E[U | D, X] = 0
      D = m(X) + V,               E[V | X] = 0
    Neyman Orthogonal score:
      psi(W; theta, eta) = (Y - g(X) - theta * (D - m(X))) * (D - m(X))
    """

    def __init__(
        self,
        model_y: BaseEstimator | None = None,
        model_d: BaseEstimator | None = None,
        n_splits: int = 5,
        random_state: int = 42,
    ):
        self.model_y = model_y or HistGradientBoostingRegressor(max_iter=100, max_depth=5, random_state=random_state)
        self.model_d = model_d or HistGradientBoostingRegressor(max_iter=100, max_depth=5, random_state=random_state)
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        d: pd.Series | np.ndarray,
        y: pd.Series | np.ndarray,
        groups: pd.Series | np.ndarray | None = None,
    ) -> CausalEstimate:
        """
        Fits DML-PLR via cross-fitting to eliminate regularizer/overfitting bias.
        """
        X_arr = np.asarray(X, dtype=float)
        d_arr = np.asarray(d, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        n = len(y_arr)

        if groups is not None:
            cv = GroupKFold(n_splits=self.n_splits)
            splits = list(cv.split(X_arr, y_arr, groups=groups))
        else:
            cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            splits = list(cv.split(X_arr, y_arr))

        y_hat = np.zeros(n)
        d_hat = np.zeros(n)

        logger.debug("[DoubleML-PLR] Starting %d-fold cross-fitting for N=%d...", self.n_splits, n)
        for train_idx, val_idx in splits:
            X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
            y_tr = y_arr[train_idx]
            d_tr = d_arr[train_idx]

            # Fit nuisance model y ~ X
            my = clone(self.model_y)
            my.fit(X_tr, y_tr)
            y_hat[val_idx] = my.predict(X_val)

            # Fit nuisance model d ~ X
            md = clone(self.model_d)
            md.fit(X_tr, d_tr)
            d_hat[val_idx] = md.predict(X_val)

        # Neyman orthogonalized residuals
        y_res = y_arr - y_hat
        d_res = d_arr - d_hat

        # Closed form orthogonal estimator: theta = sum(d_res * y_res) / sum(d_res^2)
        denom = np.mean(d_res ** 2)
        if denom < 1e-12:
            raise ValueError("Treatment variation conditional on confounders is near zero (insufficient overlap/positivity).")

        num = np.mean(d_res * y_res)
        theta = float(num / denom)

        # Asymptotic sandwich variance (Chernozhukov et al. 2018)
        u_hat = y_res - theta * d_res
        j_hat = denom
        sigma2 = np.mean((u_hat * d_res) ** 2) / (j_hat ** 2)
        se = float(np.sqrt(sigma2 / n))

        # Inferential statistics
        z_stat = float(theta / se) if se > 0 else 0.0
        p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(z_stat))))
        ci_low = float(theta - 1.96 * se)
        ci_high = float(theta + 1.96 * se)

        # Nuisance R2 quality
        ss_tot_y = np.sum((y_arr - np.mean(y_arr)) ** 2)
        r2_y = float(1.0 - np.sum(y_res ** 2) / ss_tot_y) if ss_tot_y > 0 else 0.0

        ss_tot_d = np.sum((d_arr - np.mean(d_arr)) ** 2)
        r2_d = float(1.0 - np.sum(d_res ** 2) / ss_tot_d) if ss_tot_d > 0 else 0.0

        e_val = compute_e_value(theta, se)

        return CausalEstimate(
            ate=theta,
            se=se,
            ci_lower=ci_low,
            ci_upper=ci_high,
            p_value=p_val,
            z_stat=z_stat,
            e_value=e_val,
            method="DoubleML-PLR",
            n_samples=n,
            nuisance_r2_y=r2_y,
            nuisance_r2_d=r2_d,
        )


class DoubleMLAIPW:
    """
    Augmented Inverse Probability Weighting (Doubly Robust) Estimator for Binary Treatment.
    Unbiased if EITHER the propensity model OR the outcome conditional expectation is correct.
    """

    def __init__(
        self,
        model_propensity: BaseEstimator | None = None,
        model_outcome: BaseEstimator | None = None,
        n_splits: int = 5,
        trim_eps: float = 0.02,
        random_state: int = 42,
    ):
        self.model_propensity = model_propensity or HistGradientBoostingClassifier(max_iter=100, max_depth=5, random_state=random_state)
        self.model_outcome = model_outcome or HistGradientBoostingRegressor(max_iter=100, max_depth=5, random_state=random_state)
        self.n_splits = n_splits
        self.trim_eps = trim_eps
        self.random_state = random_state

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        d: pd.Series | np.ndarray,
        y: pd.Series | np.ndarray,
        groups: pd.Series | np.ndarray | None = None,
    ) -> CausalEstimate:
        X_arr = np.asarray(X, dtype=float)
        d_arr = np.asarray(d, dtype=int)
        y_arr = np.asarray(y, dtype=float)
        n = len(y_arr)

        if groups is not None:
            cv = GroupKFold(n_splits=self.n_splits)
            splits = list(cv.split(X_arr, y_arr, groups=groups))
        else:
            cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            splits = list(cv.split(X_arr, y_arr))

        mu0_hat = np.zeros(n)
        mu1_hat = np.zeros(n)
        e_hat = np.zeros(n)

        for train_idx, val_idx in splits:
            X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
            d_tr, d_val = d_arr[train_idx], d_arr[val_idx]
            y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

            # Fit propensity model e(X) = P(D=1|X)
            mp = clone(self.model_propensity)
            mp.fit(X_tr, d_tr)
            e_hat[val_idx] = mp.predict_proba(X_val)[:, 1]

            # Fit outcome model mu0(X) = E[Y|D=0, X]
            m0 = clone(self.model_outcome)
            mask_tr0 = (d_tr == 0)
            m0.fit(X_tr[mask_tr0], y_tr[mask_tr0])
            mu0_hat[val_idx] = m0.predict(X_val)

            # Fit outcome model mu1(X) = E[Y|D=1, X]
            m1 = clone(self.model_outcome)
            mask_tr1 = (d_tr == 1)
            m1.fit(X_tr[mask_tr1], y_tr[mask_tr1])
            mu1_hat[val_idx] = m1.predict(X_val)

        # Positivity trimming to prevent unbounded weights
        e_hat = np.clip(e_hat, self.trim_eps, 1.0 - self.trim_eps)

        # Doubly robust pseudo-outcomes (AIPW scores)
        # Gamma_i = mu1(X_i) - mu0(X_i) + D_i*(Y_i - mu1(X_i))/e(X_i) - (1-D_i)*(Y_i - mu0(X_i))/(1-e(X_i))
        dr_scores = (
            (mu1_hat - mu0_hat)
            + (d_arr * (y_arr - mu1_hat) / e_hat)
            - ((1 - d_arr) * (y_arr - mu0_hat) / (1.0 - e_hat))
        )

        tau = float(np.mean(dr_scores))
        se = float(np.std(dr_scores, ddof=1) / np.sqrt(n))

        z_stat = float(tau / se) if se > 0 else 0.0
        p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(z_stat))))
        ci_low = float(tau - 1.96 * se)
        ci_high = float(tau + 1.96 * se)
        e_val = compute_e_value(tau, se)

        return CausalEstimate(
            ate=tau,
            se=se,
            ci_lower=ci_low,
            ci_upper=ci_high,
            p_value=p_val,
            z_stat=z_stat,
            e_value=e_val,
            method="DoubleML-AIPW",
            n_samples=n,
        )


class CATEEstimator:
    """
    Estimates Conditional Average Treatment Effects (CATE): tau(V) = E[Y(1) - Y(0) | V]
    by regressing orthogonalized residual Y on treatment residual D interacted with effect modifiers V.
    """

    def __init__(self, dml_plr: DoubleMLPLR | None = None):
        self.dml_plr = dml_plr or DoubleMLPLR()

    def fit_cate(
        self,
        X: pd.DataFrame,
        d: pd.Series,
        y: pd.Series,
        modifiers: Sequence[str],
        groups: pd.Series | None = None,
    ) -> HeterogeneousCausalEstimate:
        """
        Decomposes treatment effect into baseline + moderator interaction coefficients.
        """
        # 1. Run DML cross-fitting to get orthogonal residuals
        X_arr = np.asarray(X, dtype=float)
        d_arr = np.asarray(d, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        n = len(y_arr)

        if groups is not None:
            cv = GroupKFold(n_splits=self.dml_plr.n_splits)
            splits = list(cv.split(X_arr, y_arr, groups=groups))
        else:
            cv = KFold(n_splits=self.dml_plr.n_splits, shuffle=True, random_state=self.dml_plr.random_state)
            splits = list(cv.split(X_arr, y_arr))

        y_hat = np.zeros(n)
        d_hat = np.zeros(n)

        for train_idx, val_idx in splits:
            my = clone(self.dml_plr.model_y)
            my.fit(X_arr[train_idx], y_arr[train_idx])
            y_hat[val_idx] = my.predict(X_arr[val_idx])

            md = clone(self.dml_plr.model_d)
            md.fit(X_arr[train_idx], d_arr[train_idx])
            d_hat[val_idx] = md.predict(X_arr[val_idx])

        y_res = y_arr - y_hat
        d_res = d_arr - d_hat

        # Baseline ATE
        base_ate = float(np.mean(d_res * y_res) / np.mean(d_res ** 2))

        # 2. Build interaction design matrix: [d_res, d_res * V_1, ..., d_res * V_k]
        V_df = X[list(modifiers)].copy()
        interaction_cols = {}
        for mod in modifiers:
            interaction_cols[f"d_x_{mod}"] = d_res * V_df[mod].values

        inter_df = pd.DataFrame(interaction_cols)
        inter_df.insert(0, "d_res", d_res)

        # Weighted least squares regression of y_res on [d_res, d_res * V]
        W = inter_df.values
        # Solve (W^T W)^{-1} W^T y_res
        beta, residuals, rank, s = np.linalg.lstsq(W, y_res, rcond=None)

        # Standard errors via robust HC0
        e_fit = y_res - W @ beta
        bread = np.linalg.pinv(W.T @ W)
        meat = W.T @ np.diag(e_fit ** 2) @ W
        cov = bread @ meat @ bread
        se_beta = np.sqrt(np.diag(cov))

        inter_terms = {}
        inter_pvals = {}
        for i, mod in enumerate(modifiers):
            coeff = float(beta[i + 1])
            se_c = float(se_beta[i + 1])
            z_c = coeff / se_c if se_c > 0 else 0.0
            p_c = float(2.0 * (1.0 - stats.norm.cdf(abs(z_c))))
            inter_terms[mod] = coeff
            inter_pvals[mod] = p_c

        # 3. Subgroup breakdown
        subgroup_effects = {}
        for mod in modifiers:
            # Check unique values or binary splits
            vals = np.unique(V_df[mod])
            if len(vals) == 2:
                # Binary indicator
                for v in vals:
                    sub_mask = (V_df[mod] == v)
                    if sub_mask.sum() > 30:
                        denom_sub = np.mean(d_res[sub_mask] ** 2)
                        ate_sub = float(np.mean(d_res[sub_mask] * y_res[sub_mask]) / denom_sub) if denom_sub > 0 else 0.0
                        se_sub = float(np.std(d_res[sub_mask] * y_res[sub_mask]) / (denom_sub * np.sqrt(sub_mask.sum()))) if denom_sub > 0 else 0.0
                        z_sub = ate_sub / se_sub if se_sub > 0 else 0.0
                        p_sub = float(2.0 * (1.0 - stats.norm.cdf(abs(z_sub))))
                        subgroup_effects[f"{mod}={v}"] = CausalEstimate(
                            ate=ate_sub,
                            se=se_sub,
                            ci_lower=ate_sub - 1.96 * se_sub,
                            ci_upper=ate_sub + 1.96 * se_sub,
                            p_value=p_sub,
                            z_stat=z_sub,
                            e_value=compute_e_value(ate_sub, se_sub),
                            method="CATE-Subgroup",
                            n_samples=int(sub_mask.sum()),
                        )

        return HeterogeneousCausalEstimate(
            base_ate=base_ate,
            subgroup_effects=subgroup_effects,
            interaction_terms=inter_terms,
            interaction_p_values=inter_pvals,
        )
