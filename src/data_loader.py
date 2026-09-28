"""
loader.py — Data Loaders for All SSIF Datasets
Student Success Intelligence Framework (SSIF)

Provides clean, validated DataFrames for all four datasets.
All loaders enforce schema validation (RULE-001).
No preprocessing is performed here — raw data only.

RULE-001: Inspect schemas before implementing models.
RULE-012: Log every data-cleaning decision.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.logger import get_module_logger
from src.validation.schema_validator import (
    validate_dlsm_a,
    validate_dlsm_b,
    validate_placement,
    validate_retention,
)

logger = get_module_logger("data.loader")

# ─── Default paths (can be overridden by config) ─────────────────────────────
_SSIF_ROOT = Path(__file__).resolve().parent.parent
_SSIF_A_DEFAULT = _SSIF_ROOT / "academic_survival_longitudinal.csv"
_SSIF_B_DEFAULT = _SSIF_ROOT / "Placement_Data_Full_Class.csv"
_DLSM_A_DEFAULT = Path(r"C:\Users\Lenovo\Downloads\DLSM\data\raw\dataset_a\bedtime_screentime_sleep_debt.csv")
_DLSM_B_DEFAULT = Path(r"C:\Users\Lenovo\Downloads\DLSM\data\raw\dataset_b\AI_SocialMedia_Student_Dataset.csv")


def load_retention(path: Path | str | None = None) -> pd.DataFrame:
    """
    Load and validate the academic persistence (retention) dataset.

    Returns:
        DataFrame with 79,239 rows, 22 columns.
        Index: default integer.
        Key columns: Student_ID (str), Semester (int), Sem_GPA (float),
            Target_Dropout_Next_Sem (int 0/1), End_of_Semester_Status (str),
            Censored (int 0/1).

    Notes:
        - Family_Income has 4.55% missing (expected).
        - LMS_Logins has 1.09% missing (expected).
        - These are NOT imputed here — imputation is pipeline responsibility.
    """
    p = Path(path) if path else _SSIF_A_DEFAULT
    logger.info("[Retention] Loading from: %s", p)

    df = pd.read_csv(p, dtype={"Student_ID": str})

    # Cast numeric columns
    numeric_cols = [
        "Age", "First_Generation", "Household_Size", "Scholarship", "Tuition_Base",
        "Semester", "Course_Load", "Work_Hours", "Emergency_Expense", "Sem_GPA",
        "Attendance", "LMS_Logins", "Advising_Visits", "Failed_Courses",
        "Financial_Stress", "Target_Dropout_Next_Sem", "Censored",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Family_Income may be parsed as float already
    if "Family_Income" in df.columns:
        df["Family_Income"] = pd.to_numeric(df["Family_Income"], errors="coerce")

    # Canonicalize Gender categories to prevent demographic fragmentation
    # Unifies: {'F', 'female', 'Female'} -> 'Female'; {'M', 'male', 'Male'} -> 'Male'
    if "Gender" in df.columns:
        gender_map = {
            "female": "Female",
            "f": "Female",
            "male": "Male",
            "m": "Male",
            "other": "Other",
            "prefer not to say": "Prefer not to say",
        }
        df["Gender"] = (
            df["Gender"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(lambda x: gender_map.get(x, x.title()))
        )

    validate_retention(df)

    logger.info(
        "[Retention] Loaded: %d rows, %d students, %d cols",
        len(df),
        df["Student_ID"].nunique(),
        len(df.columns),
    )
    return df


def load_placement(path: Path | str | None = None) -> pd.DataFrame:
    """
    Load and validate the placement (employability) dataset.

    Returns:
        DataFrame with 215 rows, 15 columns.
        Key columns: status (str 'Placed'/'Not Placed'),
            salary (float, NaN for Not Placed).

    Notes:
        - salary is structurally missing for all 67 'Not Placed' rows.
        - This is NOT an error — model salary separately (N=148).
    """
    p = Path(path) if path else _SSIF_B_DEFAULT
    logger.info("[Placement] Loading from: %s", p)

    df = pd.read_csv(p, dtype={"ssc_b": str, "hsc_b": str, "hsc_s": str, "degree_t": str})

    numeric_cols = ["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "salary" in df.columns:
        df["salary"] = pd.to_numeric(df["salary"], errors="coerce")

    validate_placement(df)
    n_placed = (df["status"] == "Placed").sum()
    logger.info(
        "[Placement] Loaded: %d rows (%d Placed, %d Not Placed), salary defined for %d",
        len(df),
        n_placed,
        len(df) - n_placed,
        df["salary"].notna().sum(),
    )
    return df


def load_dlsm_a(path: Path | str | None = None) -> pd.DataFrame:
    """
    Load and validate DLSM Dataset A (Bedtime Screen Time & Sleep Debt).

    Returns:
        DataFrame with 8,500 rows, 18 columns.
        Key columns: next_day_fatigue_score (regression target),
            sleep_debt_category (classification target — NO sleep leakage!).

    Notes:
        - This is the DLSM's own data, NOT to be merged with SSIF data.
        - Used for: DLSM effectiveness experiments, representation comparison.
    """
    p = Path(path) if path else _DLSM_A_DEFAULT
    logger.info("[DLSM-A] Loading from: %s", p)

    df = pd.read_csv(p, dtype={"user_id": str})

    numeric_cols = [
        "age", "bedtime_phone_minutes", "screen_brightness_pct",
        "blue_light_filter_active", "caffeine_post_5pm_mg", "physical_activity_min",
        "sleep_latency_min", "total_sleep_hours", "deep_sleep_pct", "rem_sleep_pct",
        "morning_alarm_snoozes", "next_day_fatigue_score",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    validate_dlsm_a(df)
    logger.info("[DLSM-A] Loaded: %d rows, %d cols", len(df), len(df.columns))
    return df


def load_dlsm_b(path: Path | str | None = None) -> pd.DataFrame:
    """
    Load and validate DLSM Dataset B (AI & Social Media Student Health).

    Returns:
        DataFrame with 16,000 rows, 10 columns.
        Key columns: Mental_Health_Score (primary target),
            Physical_Health_Score (secondary target).

    Notes:
        - Population: students (College/High School/University).
        - NO academic grades column — DLSM README confirmed this empirically.
        - Education_Level distribution: College=4890, HS=6083, University=5027.
        - This is the DLSM's own data — NOT to be row-merged with SSIF data.
        - Representation-level comparison with SSIF-A IS permitted.
    """
    p = Path(path) if path else _DLSM_B_DEFAULT
    logger.info("[DLSM-B] Loading from: %s", p)

    df = pd.read_csv(p, dtype={"Student_ID": str})

    numeric_cols = [
        "Age", "Daily_Social_Media_Hours", "Daily_AI_Tool_Usage_Hours",
        "Sleep_Hours", "Physical_Activity_Hours",
        "Mental_Health_Score", "Physical_Health_Score",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    validate_dlsm_b(df)
    edu_dist = df["Education_Level"].value_counts().to_dict()
    logger.info(
        "[DLSM-B] Loaded: %d rows, %d cols. Education dist: %s",
        len(df), len(df.columns), edu_dist,
    )
    return df
