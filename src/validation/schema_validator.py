"""
schema_validator.py — Dataset Schema Validation
Student Success Intelligence Framework (SSIF)

Validates that loaded DataFrames match the empirically verified schemas.
Halts pipeline if required columns are missing or dtypes are wrong.

RULE-001: Inspect schemas before implementing models.
RULE-026: Schema validation must run before any model training.
"""
from __future__ import annotations

import pandas as pd
from src.logger import get_module_logger

logger = get_module_logger("validation.schema")

# ─── Empirically verified schemas (from Python audit 2026-09-29) ─────────────

RETENTION_REQUIRED_COLS = {
    "Student_ID": object,
    "Age": "numeric",
    "Gender": object,
    "First_Generation": "numeric",
    "Family_Income": "numeric",   # 4.55% missing — expected
    "Household_Size": "numeric",
    "Housing_Status": object,
    "Scholarship": "numeric",
    "Tuition_Base": "numeric",
    "Semester": "numeric",
    "Course_Load": "numeric",
    "Work_Hours": "numeric",
    "Emergency_Expense": "numeric",
    "Sem_GPA": "numeric",
    "Attendance": "numeric",
    "LMS_Logins": "numeric",      # 1.09% missing — expected
    "Advising_Visits": "numeric",
    "Failed_Courses": "numeric",
    "Financial_Stress": "numeric",
    "Target_Dropout_Next_Sem": "numeric",
    "End_of_Semester_Status": object,
    "Censored": "numeric",
}

RETENTION_EXPECTED_ROWS = 79_239
RETENTION_EXPECTED_STUDENTS = 20_000

PLACEMENT_REQUIRED_COLS = {
    "sl_no": "numeric",
    "gender": object,
    "ssc_p": "numeric",
    "ssc_b": object,
    "hsc_p": "numeric",
    "hsc_b": object,
    "hsc_s": object,
    "degree_p": "numeric",
    "degree_t": object,
    "workex": object,
    "etest_p": "numeric",
    "specialisation": object,
    "mba_p": "numeric",
    "status": object,
    "salary": "numeric",  # 67 structurally missing (Not Placed)
}

PLACEMENT_EXPECTED_ROWS = 215

DLSM_A_REQUIRED_COLS = {
    "user_id": object,
    "age": "numeric",
    "gender": object,
    "occupation_type": object,
    "chronotype": object,
    "bedtime_phone_minutes": "numeric",
    "primary_bedtime_app": object,
    "screen_brightness_pct": "numeric",
    "blue_light_filter_active": "numeric",
    "caffeine_post_5pm_mg": "numeric",
    "physical_activity_min": "numeric",
    "sleep_latency_min": "numeric",
    "total_sleep_hours": "numeric",
    "deep_sleep_pct": "numeric",
    "rem_sleep_pct": "numeric",
    "morning_alarm_snoozes": "numeric",
    "next_day_fatigue_score": "numeric",
    "sleep_debt_category": object,
}

DLSM_B_REQUIRED_COLS = {
    "Student_ID": object,
    "Age": "numeric",
    "Gender": object,
    "Education_Level": object,
    "Daily_Social_Media_Hours": "numeric",
    "Daily_AI_Tool_Usage_Hours": "numeric",
    "Sleep_Hours": "numeric",
    "Physical_Activity_Hours": "numeric",
    "Mental_Health_Score": "numeric",
    "Physical_Health_Score": "numeric",
}


# ─── Validator ───────────────────────────────────────────────────────────────

class SchemaValidationError(Exception):
    """Raised when a dataset fails schema validation. Halts pipeline."""


def _validate_schema(
    df: pd.DataFrame,
    required_cols: dict[str, str],
    dataset_name: str,
    expected_rows: int | None = None,
) -> None:
    """
    Validate that a DataFrame matches required column names and approximate dtypes.

    Args:
        df: The loaded DataFrame to validate.
        required_cols: Dict mapping col_name → 'numeric' or object type.
        dataset_name: Human-readable name for logging.
        expected_rows: If provided, warns if row count deviates by >5%.

    Raises:
        SchemaValidationError: On missing columns or dtype mismatch.
    """
    errors: list[str] = []

    # Check all required columns present
    missing = set(required_cols) - set(df.columns)
    if missing:
        errors.append(f"Missing columns: {sorted(missing)}")

    # Check dtypes for present columns
    for col, expected_type in required_cols.items():
        if col not in df.columns:
            continue
        if expected_type == "numeric":
            if not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(
                    f"Column '{col}' expected numeric, got {df[col].dtype}"
                )

    if errors:
        msg = f"[{dataset_name}] Schema validation FAILED:\n" + "\n".join(
            f"  - {e}" for e in errors
        )
        logger.error(msg)
        raise SchemaValidationError(msg)

    logger.info(
        "[%s] Schema OK — %d rows, %d cols",
        dataset_name,
        len(df),
        len(df.columns),
    )

    if expected_rows is not None:
        deviation = abs(len(df) - expected_rows) / expected_rows
        if deviation > 0.05:
            logger.warning(
                "[%s] Row count deviation: expected ~%d, got %d (%.1f%%)",
                dataset_name,
                expected_rows,
                len(df),
                deviation * 100,
            )


def validate_retention(df: pd.DataFrame) -> None:
    """Validate academic_survival_longitudinal.csv schema."""
    _validate_schema(df, RETENTION_REQUIRED_COLS, "Retention", RETENTION_EXPECTED_ROWS)
    # Additional: check Student_ID uniqueness per semester
    dupes = df.duplicated(subset=["Student_ID", "Semester"]).sum()
    if dupes > 0:
        logger.warning("[Retention] %d duplicate (Student_ID, Semester) pairs found", dupes)


def validate_placement(df: pd.DataFrame) -> None:
    """Validate Placement_Data_Full_Class.csv schema."""
    _validate_schema(df, PLACEMENT_REQUIRED_COLS, "Placement", PLACEMENT_EXPECTED_ROWS)


def validate_dlsm_a(df: pd.DataFrame) -> None:
    """Validate DLSM Dataset A (bedtime_screentime_sleep_debt.csv) schema."""
    _validate_schema(df, DLSM_A_REQUIRED_COLS, "DLSM-A", 8_500)


def validate_dlsm_b(df: pd.DataFrame) -> None:
    """Validate DLSM Dataset B (AI_SocialMedia_Student_Dataset.csv) schema."""
    _validate_schema(df, DLSM_B_REQUIRED_COLS, "DLSM-B", 16_000)
