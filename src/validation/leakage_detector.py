"""
leakage_detector.py — Data Leakage Detection
Student Success Intelligence Framework (SSIF)

Detects post-outcome variables, target-feature contamination,
and temporal ordering violations before any model training.

RULE-009: No future information in features.
RULE-010: Post-outcome variables are forbidden as predictors.
RULE-027: Never hide negative results — leakage flags are always reported.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import pandas as pd
from src.logger import get_module_logger

logger = get_module_logger("validation.leakage")


@dataclass
class LeakageReport:
    """Result of a leakage detection scan."""
    dataset: str
    forbidden_features_found: list[str] = field(default_factory=list)
    target_in_features: bool = False
    temporal_violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_critical_leakage(self) -> bool:
        return bool(self.forbidden_features_found) or self.target_in_features

    @property
    def critical_leakage(self) -> bool:
        """Alias for has_critical_leakage for CI/CD compatibility."""
        return self.has_critical_leakage

    def summary(self) -> str:
        lines = [f"=== Leakage Report: {self.dataset} ==="]
        if self.has_critical_leakage:
            lines.append("STATUS: ❌ CRITICAL LEAKAGE DETECTED — DO NOT TRAIN")
        else:
            lines.append("STATUS: ✅ No critical leakage detected")
        if self.forbidden_features_found:
            lines.append(f"Forbidden features in feature set: {self.forbidden_features_found}")
        if self.target_in_features:
            lines.append("Target variable detected in feature set!")
        if self.temporal_violations:
            lines.append(f"Temporal violations: {self.temporal_violations}")
        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        return "\n".join(lines)


# ─── Retention leakage rules ─────────────────────────────────────────────────

# Variables that are FORBIDDEN as features when predicting Target_Dropout_Next_Sem
RETENTION_FORBIDDEN_FEATURES = {
    "End_of_Semester_Status",  # realized outcome — post-outcome variable
    "Censored",                 # survival meta-variable
    "Target_Dropout_Next_Sem",  # the target itself
}

# Features that should never appear together with the binary target
RETENTION_TARGET_BINARY = "Target_Dropout_Next_Sem"
RETENTION_TARGET_MULTICLASS = "End_of_Semester_Status"


def check_retention_leakage(
    feature_cols: Sequence[str] | pd.DataFrame,
    target_col: str = "Target_Dropout_Next_Sem",
    df: pd.DataFrame | None = None,
) -> LeakageReport:
    """
    Check for leakage in a proposed retention feature set or DataFrame.

    Args:
        feature_cols: List of column names being used as features, or a DataFrame.
        target_col: The target column name (default: 'Target_Dropout_Next_Sem').
        df: Optional DataFrame for deeper temporal checks.

    Returns:
        LeakageReport with findings.
    """
    if isinstance(feature_cols, pd.DataFrame):
        df = feature_cols
        feature_cols = [c for c in df.columns if c not in [target_col, "Student_ID", "End_of_Semester_Status", "Censored"]]

    report = LeakageReport(dataset="Retention")
    feature_set = set(feature_cols)


    # 1. Check forbidden features
    forbidden_present = feature_set & RETENTION_FORBIDDEN_FEATURES
    if forbidden_present:
        report.forbidden_features_found = sorted(forbidden_present)
        logger.error(
            "[Retention] CRITICAL: Forbidden features in feature set: %s",
            forbidden_present,
        )

    # 2. Check target not in features
    if target_col in feature_set:
        report.target_in_features = True
        logger.error("[Retention] CRITICAL: Target '%s' is in feature set!", target_col)

    # 3. Cross-target contamination
    if (
        RETENTION_TARGET_BINARY in feature_set
        and target_col == RETENTION_TARGET_MULTICLASS
    ):
        report.warnings.append(
            "Binary target 'Target_Dropout_Next_Sem' appears as feature while predicting 'End_of_Semester_Status' — potential contamination"
        )

    if (
        RETENTION_TARGET_MULTICLASS in feature_set
        and target_col == RETENTION_TARGET_BINARY
    ):
        report.forbidden_features_found.append(RETENTION_TARGET_MULTICLASS)
        logger.error(
            "[Retention] CRITICAL: 'End_of_Semester_Status' (realized outcome) used as feature when predicting '%s'",
            target_col,
        )

    # 4. Temporal check: verify Semester column is present for ordering
    if df is not None and "Semester" not in df.columns:
        report.temporal_violations.append(
            "Semester column missing — cannot verify temporal ordering for trajectory features"
        )

    logger.info("[Retention] Leakage scan complete. Critical: %s", report.has_critical_leakage)
    return report


# ─── Placement leakage rules ─────────────────────────────────────────────────

PLACEMENT_TARGET_CLASSIFICATION = "status"
PLACEMENT_TARGET_REGRESSION = "salary"


def check_placement_leakage(
    feature_cols: Sequence[str] | pd.DataFrame,
    target_col: str = "status",
) -> LeakageReport:
    """
    Check for leakage in a proposed placement feature set or DataFrame.

    Key risk: salary is structurally missing for Not Placed — using salary
    as a feature when predicting placement would be leakage.

    Args:
        feature_cols: Columns being used as features, or a DataFrame.
        target_col: Target column name (default: 'status').

    Returns:
        LeakageReport with findings.
    """
    if isinstance(feature_cols, pd.DataFrame):
        feature_cols = [c for c in feature_cols.columns if c not in ["status", "salary", "sl_no"]]

    report = LeakageReport(dataset="Placement")
    feature_set = set(feature_cols)


    # salary as feature when predicting placement = leakage
    if (
        target_col == PLACEMENT_TARGET_CLASSIFICATION
        and "salary" in feature_set
    ):
        report.forbidden_features_found.append("salary")
        logger.error(
            "[Placement] CRITICAL: 'salary' used as feature when predicting 'status' — "
            "salary is only defined for Placed students (post-outcome leakage)"
        )

    if target_col in feature_set:
        report.target_in_features = True
        logger.error("[Placement] CRITICAL: Target '%s' in feature set!", target_col)

    # status as feature when predicting salary = leakage
    if (
        target_col == PLACEMENT_TARGET_REGRESSION
        and "status" in feature_set
    ):
        report.forbidden_features_found.append("status")
        logger.error(
            "[Placement] CRITICAL: 'status' used as feature when predicting salary — "
            "status determines whether salary is defined (post-outcome leakage)"
        )

    logger.info("[Placement] Leakage scan complete. Critical: %s", report.has_critical_leakage)
    return report
