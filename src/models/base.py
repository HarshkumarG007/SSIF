"""
base.py — Model Evaluation Infrastructure & Metrics
Student Success Intelligence Framework (SSIF)

Standardizes model training, GroupKFold cross-validation,
probability calibration evaluation (Brier score, ECE), and metrics reporting.

RULE-004: All retention models evaluated via GroupKFold (groups=Student_ID).
RULE-007: Preprocessing fitted on training data only.
RULE-016: Always report AUROC and PR-AUC for imbalanced targets.
RULE-020: Report uncertainty across folds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

from src.logger import get_module_logger

logger = get_module_logger("models.base")


@dataclass
class ClassificationMetrics:
    model_name: str
    feature_set: str
    n_samples: int
    n_positive: int
    prevalence: float
    auroc: float
    pr_auc: float
    brier_score: float
    ece: float
    f1: float
    precision: float
    recall: float
    fold_aurocs: list[float] = field(default_factory=list)
    fold_pr_aucs: list[float] = field(default_factory=list)
    auroc_std: float = 0.0
    mce: float = 0.0
    reliability_curve: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"[{self.model_name} | {self.feature_set}]\n"
            f"  AUROC:    {self.auroc:.4f} ± {self.auroc_std:.4f}  (folds: {[round(x, 3) for x in self.fold_aurocs]})\n"
            f"  PR-AUC:   {self.pr_auc:.4f} (baseline prevalence: {self.prevalence:.3f})\n"
            f"  Brier:    {self.brier_score:.4f} | ECE: {self.ece:.4f} | MCE: {self.mce:.4f}\n"
            f"  F1:       {self.f1:.4f} (Prec: {self.precision:.4f}, Rec: {self.recall:.4f})"
        )


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    Measures difference between predicted confidence and empirical accuracy.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper if i < n_bins - 1 else y_prob <= bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            ece += (bin_size / n) * np.abs(bin_acc - bin_conf)

    return float(ece)


def compute_mce(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Maximum Calibration Error (MCE).
    Measures the maximum absolute deviation between predicted confidence and empirical accuracy across bins.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    max_err = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper if i < n_bins - 1 else y_prob <= bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            bin_acc = float(np.mean(y_true[in_bin]))
            bin_conf = float(np.mean(y_prob[in_bin]))
            err = abs(bin_acc - bin_conf)
            if err > max_err:
                max_err = err

    return float(max_err)


def compute_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """
    Compute reliability diagram coordinates and calibration diagnostics.

    Returns:
        Dictionary containing empirical positive proportions ('prob_true'),
        mean predicted probabilities ('prob_pred'), bin sample counts ('bin_counts'),
        bin boundaries ('bin_edges'), ECE, MCE, and Brier score.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    prob_true = []
    prob_pred = []
    bin_counts = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper if i < n_bins - 1 else y_prob <= bin_upper)
        cnt = int(np.sum(in_bin))
        bin_counts.append(cnt)
        if cnt > 0:
            prob_true.append(float(np.mean(y_true[in_bin])))
            prob_pred.append(float(np.mean(y_prob[in_bin])))
        else:
            mid = float((bin_lower + bin_upper) / 2.0)
            prob_true.append(0.0)
            prob_pred.append(mid)

    return {
        "prob_true": prob_true,
        "prob_pred": prob_pred,
        "bin_counts": bin_counts,
        "bin_edges": bin_boundaries.tolist(),
        "ece": compute_ece(y_true, y_prob, n_bins=n_bins),
        "mce": compute_mce(y_true, y_prob, n_bins=n_bins),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }


def calibrate_classifier(
    estimator: BaseEstimator,
    method: str = "sigmoid",
    cv: int | str = 3,
) -> CalibratedClassifierCV:
    """
    Wrap an estimator in a CalibratedClassifierCV to perform Platt scaling (sigmoid)
    or Isotonic Regression (isotonic).
    """
    return CalibratedClassifierCV(estimator=estimator, method=method, cv=cv)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    feature_set: str = "Default",
    threshold: float = 0.5,
    fold_aurocs: list[float] | None = None,
    fold_pr_aucs: list[float] | None = None,
) -> ClassificationMetrics:
    """Compute comprehensive classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    n_pos = int(np.sum(y_true))
    prev = float(n_pos / len(y_true)) if len(y_true) > 0 else 0.0

    auroc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    pr_auc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else prev
    brier = float(brier_score_loss(y_true, y_prob))
    ece = compute_ece(y_true, y_prob)
    mce = compute_mce(y_true, y_prob)
    rel_curve = compute_calibration_curve(y_true, y_prob)

    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))

    aurocs = fold_aurocs or [auroc]
    pr_aucs = fold_pr_aucs or [pr_auc]
    auroc_std = float(np.std(aurocs)) if len(aurocs) > 1 else 0.0

    return ClassificationMetrics(
        model_name=model_name,
        feature_set=feature_set,
        n_samples=len(y_true),
        n_positive=n_pos,
        prevalence=prev,
        auroc=auroc,
        pr_auc=pr_auc,
        brier_score=brier,
        ece=ece,
        f1=f1,
        precision=prec,
        recall=rec,
        fold_aurocs=aurocs,
        fold_pr_aucs=pr_aucs,
        auroc_std=auroc_std,
        mce=mce,
        reliability_curve=rel_curve,
    )


class MajorityClassBaseline(BaseEstimator):
    """Tier 0 Baseline: Constant predictor matching class prevalence."""

    def __init__(self, constant_value: float | None = None):
        self.constant_value = constant_value
        self.prevalence_ = 0.0

    def fit(self, X, y):
        self.prevalence_ = float(np.mean(y)) if self.constant_value is None else self.constant_value
        return self

    def predict_proba(self, X):
        probs = np.full(len(X), self.prevalence_)
        return np.column_stack([1.0 - probs, probs])

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


def evaluate_grouped_model(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    model_name: str = "Model",
    feature_set: str = "Default",
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[ClassificationMetrics, np.ndarray]:
    """
    Perform 5-fold GroupKFold cross-validation ensuring zero student-level leakage.

    Args:
        model: Scikit-learn estimator or pipeline.
        X: Feature matrix.
        y: Binary target (0/1).
        groups: Group identifier (Student_ID).
        model_name: Display name.
        feature_set: Feature configuration label.
        n_splits: Number of folds (default 5).

    Returns:
        (metrics, oof_probabilities)
    """
    logger.info(
        "[Evaluation] Running %d-fold GroupKFold for '%s' (%s features, N=%d, Groups=%d)...",
        n_splits, model_name, feature_set, len(X), groups.nunique()
    )
    gkf = GroupKFold(n_splits=n_splits)
    oof_probs = np.zeros(len(y), dtype=float)
    fold_aurocs = []
    fold_pr_aucs = []

    X_mat = X.values if isinstance(X, pd.DataFrame) else X
    y_arr = y.values if isinstance(y, pd.Series) else y
    groups_arr = groups.values if isinstance(groups, pd.Series) else groups

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_mat, y_arr, groups=groups_arr)):
        X_train, y_train = X_mat[train_idx], y_arr[train_idx]
        X_test, y_test = X_mat[test_idx], y_arr[test_idx]

        clf = clone(model)
        clf.fit(X_train, y_train)

        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X_test)[:, 1]
        elif hasattr(clf, "decision_function"):
            raw_scores = clf.decision_function(X_test)
            probs = 1.0 / (1.0 + np.exp(-raw_scores))
        else:
            probs = clf.predict(X_test).astype(float)

        oof_probs[test_idx] = probs

        fold_auc = float(roc_auc_score(y_test, probs))
        fold_pr = float(average_precision_score(y_test, probs))
        fold_aurocs.append(fold_auc)
        fold_pr_aucs.append(fold_pr)

        logger.debug(
            "Fold %d/%d: AUROC=%.4f, PR-AUC=%.4f",
            fold_idx + 1, n_splits, fold_auc, fold_pr
        )

    metrics = compute_classification_metrics(
        y_true=y_arr,
        y_prob=oof_probs,
        model_name=model_name,
        feature_set=feature_set,
        fold_aurocs=fold_aurocs,
        fold_pr_aucs=fold_pr_aucs,
    )

    logger.info(
        "[Evaluation Complete] %s | %s: AUROC=%.4f (±%.4f), PR-AUC=%.4f, Brier=%.4f, ECE=%.4f, MCE=%.4f",
        model_name, feature_set, metrics.auroc, metrics.auroc_std, metrics.pr_auc, metrics.brier_score, metrics.ece, metrics.mce
    )
    return metrics, oof_probs

