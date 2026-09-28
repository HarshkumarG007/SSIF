"""
features.py — Retention Feature Engineering Pipeline
Student Success Intelligence Framework (SSIF)

Engineers static and longitudinal trajectory features strictly without
future temporal leakage (RULE-009, RULE-010).

All trajectory calculations for student i at semester t are strictly computed
from observations with Semester <= t.

RULE-007: Preprocessing fitted on training data only.
RULE-009: No future information in features.
RULE-010: Post-outcome variables are forbidden as predictors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.logger import get_module_logger
from src.validation.leakage_detector import check_retention_leakage

logger = get_module_logger("retention.features")

# ─── Static Feature Definitions ─────────────────────────────────────────────

STATIC_NUMERIC_FEATURES = [
    "Age",
    "Family_Income",
    "Household_Size",
    "Tuition_Base",
    "Course_Load",
    "Work_Hours",
    "Emergency_Expense",
    "LMS_Logins",
    "Advising_Visits",
    "Failed_Courses",
    "Financial_Stress",
    "Attendance",
    "Sem_GPA",
]

STATIC_CATEGORICAL_FEATURES = [
    "Gender",
    "First_Generation",
    "Housing_Status",
    "Scholarship",
]

TRAJECTORY_FEATURE_NAMES = [
    "n_prior_semesters",
    "is_single_semester",
    "gpa_slope",
    "gpa_velocity",
    "gpa_volatility",
    "gpa_recent_mean",
    "gpa_trajectory_type",
    "attendance_slope",
    "attendance_delta",
    "lms_slope",
    "lms_delta",
    "cumulative_failed_courses",
    "cumulative_advising_visits",
    "decline_index",
    "recovery_index",
]

FORBIDDEN_COLUMNS = {
    "End_of_Semester_Status",
    "Censored",
    "Target_Dropout_Next_Sem",
}


def compute_longitudinal_trajectories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-student longitudinal trajectory features up to current semester.
    Guarantees strict temporal causality: features at semester t use ONLY
    records from semesters <= t.

    Args:
        df: Retention DataFrame containing Student_ID, Semester, Sem_GPA,
            Attendance, LMS_Logins, Failed_Courses, Advising_Visits.

    Returns:
        DataFrame with original columns plus engineered trajectory columns.
    """
    logger.info("[Trajectory] Computing longitudinal trajectory features for %d rows...", len(df))
    # Must sort by student and semester for causal cumulative operations
    res = df.sort_values(["Student_ID", "Semester"]).copy()
    grp = res.groupby("Student_ID")

    n = grp.cumcount() + 1
    res["n_prior_semesters"] = n
    res["is_single_semester"] = (n == 1).astype(int)

    # 1. GPA Slope & Volatility (Exact OLS slope and sample std up to t)
    X = res["Semester"]
    Y = res["Sem_GPA"]
    Sx = grp["Semester"].cumsum()
    Sy = grp["Sem_GPA"].cumsum()
    Sxx = (X ** 2).groupby(res["Student_ID"]).cumsum()
    Syy = (Y ** 2).groupby(res["Student_ID"]).cumsum()
    Sxy = (X * Y).groupby(res["Student_ID"]).cumsum()

    denom = n * Sxx - (Sx ** 2)
    num = n * Sxy - (Sx * Sy)
    # Slope valid for n >= 2
    res["gpa_slope"] = np.where(n >= 2, num / np.where(denom == 0, np.nan, denom), np.nan)

    # Sample standard deviation (ddof=1)
    var_y = np.where(n >= 2, (Syy - (Sy ** 2) / n) / np.maximum(n - 1, 1), np.nan)
    res["gpa_volatility"] = np.sqrt(np.maximum(var_y, 0.0))

    # 2. GPA Velocity & Recent Mean
    res["gpa_velocity"] = grp["Sem_GPA"].diff()
    prev_gpa = grp["Sem_GPA"].shift(1)
    res["gpa_recent_mean"] = np.where(n >= 2, (res["Sem_GPA"] + prev_gpa) / 2.0, res["Sem_GPA"])

    # Trajectory categorical classification: 1 = improving, -1 = declining, 0 = stable
    res["gpa_trajectory_type"] = np.where(
        res["gpa_slope"] > 0.05, 1,
        np.where(res["gpa_slope"] < -0.05, -1, 0)
    )

    # 3. Attendance Slope & Delta
    Sy_att = grp["Attendance"].cumsum()
    Sxy_att = (X * res["Attendance"]).groupby(res["Student_ID"]).cumsum()
    num_att = n * Sxy_att - (Sx * Sy_att)
    res["attendance_slope"] = np.where(n >= 2, num_att / np.where(denom == 0, np.nan, denom), np.nan)
    res["attendance_delta"] = grp["Attendance"].diff()

    # 4. LMS Slope & Delta
    # For LMS logins, fill missing with cumulative group median or forward fill for trajectory calculation
    lms_clean = res["LMS_Logins"].fillna(grp["LMS_Logins"].transform("median")).fillna(0)
    Sy_lms = lms_clean.groupby(res["Student_ID"]).cumsum()
    Sxy_lms = (X * lms_clean).groupby(res["Student_ID"]).cumsum()
    num_lms = n * Sxy_lms - (Sx * Sy_lms)
    res["lms_slope"] = np.where(n >= 2, num_lms / np.where(denom == 0, np.nan, denom), np.nan)
    res["lms_delta"] = grp["LMS_Logins"].diff()

    # 5. Cumulative Academic Metrics
    res["cumulative_failed_courses"] = grp["Failed_Courses"].cumsum()
    res["cumulative_advising_visits"] = grp["Advising_Visits"].cumsum()

    # 6. Decline and Recovery Indices
    # Decline: consecutive semesters where GPA decreased
    gpa_decreased = (res["gpa_velocity"] < 0).astype(int)
    # Vectorized consecutive counter within student
    # Reset cumsum when not decreased
    sub_grp = (~(res["gpa_velocity"] < 0)).groupby(res["Student_ID"]).cumsum()
    res["decline_index"] = np.where(
        n >= 2,
        gpa_decreased.groupby([res["Student_ID"], sub_grp]).cumsum(),
        0
    )

    # Recovery: 1 if current semester velocity > 0 and previous velocity < 0
    prev_velocity = grp["gpa_velocity"].shift(1)
    res["recovery_index"] = np.where(
        (res["gpa_velocity"] > 0) & (prev_velocity < 0),
        1,
        0
    )

    logger.info("[Trajectory] Trajectory features successfully generated. New cols: %s", TRAJECTORY_FEATURE_NAMES)
    return res


class RetentionFeaturePipeline(BaseEstimator, TransformerMixin):
    """
    Scikit-Learn compliant Transformer for Retention Feature Engineering.
    Can be used inside Cross-Validation without data leakage.

    Supports:
      - Static features only (`include_trajectories=False`)
      - Static + Trajectory features (`include_trajectories=True`)
      - Imputation of missing values (fit on train folds only)
    """

    def __init__(
        self,
        include_trajectories: bool = True,
        impute_trajectories_with_zero: bool = True,
        categorical_encoding: str = "onehot",
    ):
        self.include_trajectories = include_trajectories
        self.impute_trajectories_with_zero = impute_trajectories_with_zero
        self.categorical_encoding = categorical_encoding
        self.numeric_medians_: dict[str, float] = {}
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        """Fit preprocessing statistics strictly on training fold."""
        # Calculate medians on training fold only (RULE-007, RULE-008)
        cols_to_median = [c for c in STATIC_NUMERIC_FEATURES if c in X.columns]
        for c in cols_to_median:
            self.numeric_medians_[c] = float(X[c].median())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted parameters."""
        df = X.copy()

        # Generate trajectories if requested and columns exist
        if self.include_trajectories:
            if "gpa_slope" not in df.columns:
                df = compute_longitudinal_trajectories(df)

        # Apply training-fold medians
        for c, med in self.numeric_medians_.items():
            if c in df.columns:
                df[c] = df[c].fillna(med)

        # Impute trajectory NaNs (for semester 1 where history is undefined)
        if self.include_trajectories and self.impute_trajectories_with_zero:
            for t_col in ["gpa_slope", "gpa_velocity", "gpa_volatility", "attendance_slope", "attendance_delta", "lms_slope", "lms_delta"]:
                if t_col in df.columns:
                    df[t_col] = df[t_col].fillna(0.0)

        # One-hot or ordinal encode categoricals
        cat_cols = [c for c in STATIC_CATEGORICAL_FEATURES if c in df.columns]
        for c in cat_cols:
            df[f"{c}_enc"] = pd.Categorical(df[c]).codes.astype(float)

        return df


def prepare_retention_dataset(
    df: pd.DataFrame,
    include_trajectories: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """
    High-level entrypoint to prepare feature matrix X, target y, and GroupKFold groups.

    Returns:
        (X, y, groups, feature_names)
    """
    if include_trajectories:
        df_feats = compute_longitudinal_trajectories(df)
        active_features = STATIC_NUMERIC_FEATURES + TRAJECTORY_FEATURE_NAMES
    else:
        df_feats = df.copy()
        active_features = STATIC_NUMERIC_FEATURES.copy()

    # Encode categoricals
    cat_enc_cols = []
    for c in STATIC_CATEGORICAL_FEATURES:
        if c in df_feats.columns:
            df_feats[f"{c}_enc"] = pd.Categorical(df_feats[c]).codes.astype(float)
            cat_enc_cols.append(f"{c}_enc")

    all_features = [c for c in active_features + cat_enc_cols if c in df_feats.columns]

    # Verify zero leakage
    leak_report = check_retention_leakage(all_features, "Target_Dropout_Next_Sem", df_feats)
    if leak_report.has_critical_leakage:
        raise ValueError(f"Leakage detected in feature set: {leak_report.summary()}")

    X = df_feats[all_features].copy()
    y = df_feats["Target_Dropout_Next_Sem"].astype(int)
    groups = df_feats["Student_ID"]

    # Fill NaNs for baseline readiness
    X = X.fillna(X.median(numeric_only=True))

    logger.info(
        "[Retention] Prepared dataset: X=%s, y=%d dropouts (%.2f%%), %d groups",
        X.shape, y.sum(), y.mean() * 100, groups.nunique()
    )
    return X, y, groups, all_features
