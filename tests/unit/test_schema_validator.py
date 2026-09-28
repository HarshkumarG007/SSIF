"""
tests/unit/test_schema_validator.py
Student Success Intelligence Framework (SSIF)

Unit tests for schema validation of all four datasets.
Tests are designed to run against actual files when available,
and use minimal mock DataFrames for CI/CD compatibility.

RULE-046: Tests must run after any meaningful change.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.validation.schema_validator import (
    SchemaValidationError,
    validate_placement,
    validate_retention,
)


# ─── Retention Tests ──────────────────────────────────────────────────────────

class TestRetentionSchema:
    """Tests for validate_retention()."""

    def _make_valid_retention_row(self) -> dict:
        return {
            "Student_ID": "STU_00001",
            "Age": 19,
            "Gender": "Female",
            "First_Generation": 0,
            "Family_Income": 50000.0,
            "Household_Size": 3,
            "Housing_Status": "Off-Campus",
            "Scholarship": 0,
            "Tuition_Base": 15000,
            "Semester": 1,
            "Course_Load": 15,
            "Work_Hours": 10.0,
            "Emergency_Expense": 0.0,
            "Sem_GPA": 3.2,
            "Attendance": 85.0,
            "LMS_Logins": 45.0,
            "Advising_Visits": 1,
            "Failed_Courses": 0,
            "Financial_Stress": 2.5,
            "Target_Dropout_Next_Sem": 0,
            "End_of_Semester_Status": "Enrolled",
            "Censored": 0,
        }

    def test_valid_schema_passes(self):
        """Valid DataFrame matching expected schema should not raise."""
        df = pd.DataFrame([self._make_valid_retention_row()])
        # Should not raise
        validate_retention(df)

    def test_missing_column_raises(self):
        """DataFrame missing required column should raise SchemaValidationError."""
        row = self._make_valid_retention_row()
        del row["Sem_GPA"]
        df = pd.DataFrame([row])
        with pytest.raises(SchemaValidationError):
            validate_retention(df)

    def test_missing_student_id_raises(self):
        """Missing Student_ID column must be caught."""
        row = self._make_valid_retention_row()
        del row["Student_ID"]
        df = pd.DataFrame([row])
        with pytest.raises(SchemaValidationError):
            validate_retention(df)

    def test_non_numeric_gpa_raises(self):
        """Non-numeric Sem_GPA (object dtype) must raise SchemaValidationError."""
        row = self._make_valid_retention_row()
        row["Sem_GPA"] = "not_a_number"
        df = pd.DataFrame([row])
        # When ALL values in a column are non-numeric strings, pandas infers
        # object dtype — our validator correctly raises SchemaValidationError.
        with pytest.raises(SchemaValidationError):
            validate_retention(df)

    def test_duplicate_student_semester_warns(self, caplog):
        """Duplicate (Student_ID, Semester) pairs should log a warning."""
        row = self._make_valid_retention_row()
        df = pd.DataFrame([row, row])  # exact duplicate
        import logging
        with caplog.at_level(logging.WARNING, logger="ssif.validation.schema"):
            validate_retention(df)
        assert any("duplicate" in r.message.lower() for r in caplog.records)


# ─── Placement Tests ──────────────────────────────────────────────────────────

class TestPlacementSchema:
    """Tests for validate_placement()."""

    def _make_valid_placement_row(self, placed: bool = True) -> dict:
        return {
            "sl_no": 1,
            "gender": "M",
            "ssc_p": 67.0,
            "ssc_b": "Others",
            "hsc_p": 91.0,
            "hsc_b": "Others",
            "hsc_s": "Commerce",
            "degree_p": 58.0,
            "degree_t": "Sci&Tech",
            "workex": "No",
            "etest_p": 55.0,
            "specialisation": "Mkt&HR",
            "mba_p": 58.8,
            "status": "Placed" if placed else "Not Placed",
            "salary": 270000.0 if placed else None,
        }

    def test_valid_schema_passes(self):
        """Valid DataFrame should pass validation."""
        df = pd.DataFrame([self._make_valid_placement_row()])
        validate_placement(df)

    def test_missing_status_column_raises(self):
        """Missing 'status' column should raise."""
        row = self._make_valid_placement_row()
        del row["status"]
        df = pd.DataFrame([row])
        with pytest.raises(SchemaValidationError):
            validate_placement(df)

    def test_salary_structurally_missing_is_ok(self):
        """salary=NaN for Not Placed rows is expected — should not raise."""
        rows = [
            self._make_valid_placement_row(placed=True),
            self._make_valid_placement_row(placed=False),
        ]
        df = pd.DataFrame(rows)
        validate_placement(df)  # Should not raise


# ─── Leakage Tests ────────────────────────────────────────────────────────────

class TestLeakageDetector:
    """Tests for leakage detection."""

    def test_end_of_semester_status_flagged_as_forbidden(self):
        """End_of_Semester_Status must be flagged when used as feature."""
        from src.validation.leakage_detector import check_retention_leakage

        feature_cols = ["Age", "Sem_GPA", "Attendance", "End_of_Semester_Status"]
        report = check_retention_leakage(feature_cols, "Target_Dropout_Next_Sem")
        assert report.has_critical_leakage
        assert "End_of_Semester_Status" in report.forbidden_features_found

    def test_censored_flagged_as_forbidden(self):
        """Censored must be flagged when used as feature."""
        from src.validation.leakage_detector import check_retention_leakage

        feature_cols = ["Age", "Sem_GPA", "Censored"]
        report = check_retention_leakage(feature_cols, "Target_Dropout_Next_Sem")
        assert report.has_critical_leakage
        assert "Censored" in report.forbidden_features_found

    def test_clean_feature_set_passes(self):
        """Valid feature set without leakage should have no critical findings."""
        from src.validation.leakage_detector import check_retention_leakage

        feature_cols = ["Age", "Gender", "Sem_GPA", "Attendance", "Financial_Stress"]
        report = check_retention_leakage(feature_cols, "Target_Dropout_Next_Sem")
        assert not report.has_critical_leakage

    def test_salary_as_feature_for_placement_flagged(self):
        """salary used as feature when predicting placement status = leakage."""
        from src.validation.leakage_detector import check_placement_leakage

        feature_cols = ["degree_p", "mba_p", "salary"]
        report = check_placement_leakage(feature_cols, "status")
        assert report.has_critical_leakage
        assert "salary" in report.forbidden_features_found

    def test_clean_placement_features_pass(self):
        """Valid placement features without leakage."""
        from src.validation.leakage_detector import check_placement_leakage

        feature_cols = ["degree_p", "mba_p", "etest_p", "workex"]
        report = check_placement_leakage(feature_cols, "status")
        assert not report.has_critical_leakage


# ─── DLSM Compatibility Gate Tests ───────────────────────────────────────────

class TestDLSMCompatibilityGate:
    """Tests for DLSM compatibility assessment."""

    def _make_minimal_retention_df(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "Student_ID": "STU_00001",
            "Age": 19,
            "Gender": "Female",
            "Semester": 1,
            "Sem_GPA": 3.2,
        }])

    def _make_minimal_placement_df(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "sl_no": 1,
            "gender": "M",
            "degree_p": 67.0,
            "status": "Placed",
        }])

    def test_ssif_a_vs_dlsm_b_verdict_no_go(self):
        """SSIF-A vs DLSM-B must return NO-GO verdict."""
        from src.dlsm.compatibility_gate import (
            IntegrationVerdict,
            run_ssif_a_vs_dlsm_b,
        )
        df = self._make_minimal_retention_df()
        report = run_ssif_a_vs_dlsm_b(df)
        assert report.verdict == IntegrationVerdict.NO_GO

    def test_ssif_a_vs_dlsm_b_row_merge_not_permitted(self):
        """Row merge must always be forbidden."""
        from src.dlsm.compatibility_gate import run_ssif_a_vs_dlsm_b
        df = self._make_minimal_retention_df()
        report = run_ssif_a_vs_dlsm_b(df)
        assert report.row_merge_permitted is False

    def test_ssif_a_vs_dlsm_b_representation_bridge_permitted(self):
        """Representation-level bridge is permitted for student populations."""
        from src.dlsm.compatibility_gate import run_ssif_a_vs_dlsm_b
        df = self._make_minimal_retention_df()
        report = run_ssif_a_vs_dlsm_b(df)
        assert report.representation_bridge_permitted is True

    def test_ssif_b_vs_dlsm_a_verdict_no_go(self):
        """SSIF-B vs DLSM-A must return NO-GO verdict."""
        from src.dlsm.compatibility_gate import (
            IntegrationVerdict,
            run_ssif_b_vs_dlsm_a,
        )
        df = self._make_minimal_placement_df()
        report = run_ssif_b_vs_dlsm_a(df)
        assert report.verdict == IntegrationVerdict.NO_GO

    def test_compatibility_score_below_threshold(self):
        """Compatibility score for SSIF-A must be below GO threshold (0.80)."""
        from src.dlsm.compatibility_gate import run_ssif_a_vs_dlsm_b
        df = self._make_minimal_retention_df()
        report = run_ssif_a_vs_dlsm_b(df)
        assert report.compatibility_score < 0.40

    def test_markdown_report_generated(self):
        """Markdown report generation should not raise and must include verdict."""
        from src.dlsm.compatibility_gate import (
            generate_compatibility_markdown,
            run_ssif_a_vs_dlsm_b,
            run_ssif_b_vs_dlsm_a,
        )
        df_a = self._make_minimal_retention_df()
        df_b = self._make_minimal_placement_df()
        reports = {
            "ssif_a_vs_dlsm_b": run_ssif_a_vs_dlsm_b(df_a),
            "ssif_b_vs_dlsm_a": run_ssif_b_vs_dlsm_a(df_b),
        }
        md = generate_compatibility_markdown(reports)
        assert "NO-GO" in md
        assert "Row Merge Permitted" in md
