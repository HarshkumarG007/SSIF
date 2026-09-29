"""
features.py — Placement Feature Engineering Pipeline
Student Success Intelligence Framework (SSIF)

Engineers academic progression, composite academic indices, and
encodes educational specializations for candidate employability modeling.

RULE-007: Preprocessing fitted on training data only.
RULE-009: No future information in features (salary forbidden as feature for status).
RULE-025: N=215 sample size awareness.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from src.logger import get_module_logger
from src.validation.leakage_detector import check_placement_leakage

logger = get_module_logger("placement.features")

NUMERIC_COLUMNS = [
    "ssc_p",
    "hsc_p",
    "degree_p",
    "etest_p",
    "mba_p",
]

CATEGORICAL_COLUMNS = [
    "gender",
    "ssc_b",
    "hsc_b",
    "hsc_s",
    "degree_t",
    "workex",
    "specialisation",
]

ENGINEERED_FEATURE_NAMES = [
    "academic_progression",
    "degree_deviation",
    "composite_academic_score",
    "etest_academic_ratio",
]


def engineer_placement_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute domain-specific composite academic features for employment placement.

    Args:
        df: Raw placement DataFrame.

    Returns:
        DataFrame with added engineered columns.
    """
    res = df.copy()

    # Academic trajectory across schooling and university stages
    res["academic_progression"] = res["hsc_p"] - res["ssc_p"]
    res["degree_deviation"] = res["degree_p"] - res["hsc_p"]

    # Composite academic score across all 4 stages
    res["composite_academic_score"] = (
        0.25 * res["ssc_p"]
        + 0.25 * res["hsc_p"]
        + 0.25 * res["degree_p"]
        + 0.25 * res["mba_p"]
    )

    # Ratio of technical employability test score to degree GPA
    res["etest_academic_ratio"] = res["etest_p"] / np.maximum(res["degree_p"], 1.0)

    logger.debug("[Placement] Added engineered features: %s", ENGINEERED_FEATURE_NAMES)
    return res


CONSTRAINED_DOF_FEATURES = [
    "degree_p",
    "etest_p",
    "ssc_p",
    "workex_Yes",
    "specialisation_Mkt&HR",
]  # <= 6 DoF feature constraint ensuring EPV >= 10 on Dataset B (SEC-10)


def anonymize_placement_quasi_identifiers(
    df: pd.DataFrame,
    bin_width: float = 5.0,
) -> pd.DataFrame:
    """
    Coarsens continuous quasi-identifiers in Dataset B (academic percentages:
    ssc_p, hsc_p, degree_p, etest_p, mba_p) into discretized 5% intervals (e.g. [60, 65))
    to mitigate linkage re-identification risks and uphold privacy guarantees (SEC-02).
    """
    res = df.copy()
    for col in NUMERIC_COLUMNS:
        if col in res.columns:
            binned = (np.floor(res[col] / bin_width) * bin_width).astype(int)
            res[f"{col}_binned"] = binned.apply(lambda v: f"[{v}, {int(v + bin_width)})")
    return res


def prepare_placement_classification_data(
    df: pd.DataFrame,
    include_engineered: bool = True,
    constrained_dof: bool = False,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Prepare feature matrix X and binary target y for placement status classification.

    Target: y = 1 if status == 'Placed', else 0.
    If constrained_dof is True, restricts features to <= 6 degrees of freedom to respect
    the EPV >= 10 guideline for the 67-event minority class (SEC-10).
    """
    df_feats = engineer_placement_features(df) if include_engineered else df.copy()

    feature_cols = NUMERIC_COLUMNS.copy()
    if include_engineered:
        feature_cols += ENGINEERED_FEATURE_NAMES

    # Encode categoricals with dummy variables
    cat_df = pd.get_dummies(df_feats[CATEGORICAL_COLUMNS], drop_first=True, dtype=float)
    X = pd.concat([df_feats[feature_cols], cat_df], axis=1)

    if constrained_dof:
        selected_cols = [c for c in CONSTRAINED_DOF_FEATURES if c in X.columns]
        X = X[selected_cols]

    # Strictly verify zero target leakage
    leak_report = check_placement_leakage(X.columns.tolist(), "status")
    if leak_report.has_critical_leakage:
        raise ValueError(f"Target leakage in placement features: {leak_report.summary()}")

    y = (df_feats["status"] == "Placed").astype(int)

    logger.info(
        "[Placement Classification] Prepared X=%s, y=%d placed (%.1f%%)",
        X.shape, y.sum(), y.mean() * 100
    )
    return X, y, X.columns.tolist()


def prepare_salary_regression_data(
    df: pd.DataFrame,
    include_engineered: bool = True,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Prepare feature matrix X and regression target y for placed candidates (N=148).
    Unplaced candidates (N=67) are filtered out to prevent structural target leakage.
    """
    placed_mask = df["status"] == "Placed"
    df_placed = df[placed_mask].copy()

    df_feats = engineer_placement_features(df_placed) if include_engineered else df_placed.copy()

    feature_cols = NUMERIC_COLUMNS.copy()
    if include_engineered:
        feature_cols += ENGINEERED_FEATURE_NAMES

    cat_df = pd.get_dummies(df_feats[CATEGORICAL_COLUMNS], drop_first=True, dtype=float)
    X = pd.concat([df_feats[feature_cols], cat_df], axis=1)

    # Strictly verify zero target leakage (status cannot be used to predict salary)
    leak_report = check_placement_leakage(X.columns.tolist(), "salary")
    if leak_report.has_critical_leakage:
        raise ValueError(f"Target leakage in salary features: {leak_report.summary()}")

    y = df_placed["salary"].astype(float)

    logger.info(
        "[Salary Regression] Prepared X=%s, y=N=%d placed candidates (Median salary: INR %d)",
        X.shape, len(y), int(y.median())
    )
    return X, y, X.columns.tolist()
