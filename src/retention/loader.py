"""
retention/loader.py — Retention Dataset Loader & Feature Preparation
Student Success Intelligence Framework (SSIF)

Loads, validates and prepares the academic persistence dataset.
Returns train/validation splits that respect the longitudinal structure
via GroupKFold (groups=Student_ID).

RULE-014: Always use GroupKFold for retention dataset.
RULE-007: Preprocessing fitted on training data only.
RULE-009: No future information in features.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.logger import get_module_logger
from src.validation.leakage_detector import check_retention_leakage
from src.validation.schema_validator import validate_retention

logger = get_module_logger("retention.loader")

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "academic_survival_longitudinal.csv"

# ─── Column groupings ────────────────────────────────────────────────────────

STATIC_NUMERIC_FEATURES = [
    "Age", "Family_Income", "Household_Size", "Tuition_Base",
    "Course_Load", "Work_Hours", "Emergency_Expense",
    "LMS_Logins", "Advising_Visits", "Failed_Courses",
    "Financial_Stress", "Attendance", "Sem_GPA",
]

STATIC_CATEGORICAL_FEATURES = [
    "Gender", "First_Generation", "Housing_Status", "Scholarship",
]

TARGET_BINARY = "Target_Dropout_Next_Sem"
TARGET_MULTICLASS = "End_of_Semester_Status"
FORBIDDEN_AS_FEATURES = {"End_of_Semester_Status", "Censored", "Target_Dropout_Next_Sem"}
ID_COL = "Student_ID"
TIME_COL = "Semester"


def load_retention(path: Path | str | None = None) -> pd.DataFrame:
    """
    Load and validate the retention CSV.

    Returns:
        Raw DataFrame, no imputation performed.
    """
    from src.data_loader import load_retention as _load
    return _load(path)


def get_feature_columns() -> tuple[list[str], list[str]]:
    """
    Return (numeric_feature_cols, categorical_feature_cols) for static baseline model.
    Excludes all forbidden/target/meta columns.
    """
    return STATIC_NUMERIC_FEATURES.copy(), STATIC_CATEGORICAL_FEATURES.copy()


def build_baseline_feature_matrix(
    df: pd.DataFrame,
    include_semester: bool = True,
    target: str = TARGET_BINARY,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Build X, y, groups for retention baseline modelling.

    Args:
        df: Raw retention DataFrame.
        include_semester: Whether to include Semester as a feature.
        target: Target column name.

    Returns:
        (X, y, groups) where groups = Student_ID for GroupKFold.

    Raises:
        ValueError: If leakage is detected in the feature set.
    """
    numeric_cols = STATIC_NUMERIC_FEATURES.copy()
    if include_semester:
        numeric_cols.append("Semester")

    # Encode categoricals simply for baseline
    cat_encoded = []
    df = df.copy()
    for col in STATIC_CATEGORICAL_FEATURES:
        if col in df.columns:
            df[f"{col}_enc"] = pd.Categorical(df[col]).codes.astype(float)
            cat_encoded.append(f"{col}_enc")

    feature_cols = [c for c in numeric_cols + cat_encoded if c in df.columns]

    # Leakage check
    report = check_retention_leakage(feature_cols, target, df)
    if report.has_critical_leakage:
        raise ValueError(f"Leakage detected in feature set: {report.summary()}")

    X = df[feature_cols].copy()
    y = df[target].astype(int)
    groups = df[ID_COL]

    logger.info(
        "[Retention] Feature matrix built: X=%s, y=%s (dropout rate=%.2f%%), groups=%d students",
        X.shape, y.shape, y.mean() * 100, groups.nunique()
    )
    return X, y, groups


def group_kfold_splits(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = 5,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Generate GroupKFold splits ensuring no student appears in both train and test.

    Args:
        X, y, groups: Feature matrix, target, group IDs.
        n_splits: Number of folds (default 5).

    Yields:
        (train_idx, test_idx) index arrays.
    """
    gkf = GroupKFold(n_splits=n_splits)
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        train_groups = groups.iloc[train_idx].nunique()
        test_groups = groups.iloc[test_idx].nunique()
        logger.info(
            "Fold %d/%d — train: %d rows (%d students), test: %d rows (%d students)",
            fold_idx + 1, n_splits,
            len(train_idx), train_groups,
            len(test_idx), test_groups,
        )
        yield train_idx, test_idx
