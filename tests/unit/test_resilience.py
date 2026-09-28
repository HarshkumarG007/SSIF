"""
test_resilience.py — Unit tests for academic resilience analysis.
Student Success Intelligence Framework (SSIF)
"""
import numpy as np
import pandas as pd
import pytest

from src.retention.resilience import (
    identify_resilience_cohorts,
    run_resilience_analysis,
    ResilienceAnalysisResult,
)


@pytest.fixture
def mock_trajectory_data():
    """Create synthetic longitudinal trajectory data for 10 students across 4 semesters."""
    np.random.seed(42)
    rows = []
    for stu_idx in range(10):
        stu_id = f"STU_{stu_idx:03d}"
        for sem in range(1, 5):
            # Student 0, 1: Recovery pattern (dip then increase)
            if stu_idx in [0, 1]:
                gpa = 3.5 if sem == 1 else (2.8 if sem == 2 else 3.4)
                velocity = -0.7 if sem == 2 else (0.6 if sem >= 3 else 0.0)
                recovery = 1 if sem == 3 else 0
                status = "Enrolled"
            elif stu_idx in [2, 3]:
                # Continuing decline
                gpa = 3.5 - 0.4 * sem
                velocity = -0.4
                recovery = 0
                status = "Dropped_Out" if sem == 4 else "Enrolled"
            else:
                # Stable
                gpa = 3.2
                velocity = 0.0
                recovery = 0
                status = "Enrolled"

            rows.append({
                "Student_ID": stu_id,
                "Semester": sem,
                "Sem_GPA": gpa,
                "gpa_velocity": velocity,
                "decline_index": 1 if velocity < 0 else 0,
                "recovery_index": recovery,
                "Attendance": 85.0,
                "Advising_Visits": 1 if stu_idx in [0, 1] else 0,
                "Financial_Stress": 2 if stu_idx in [0, 1] else 4,
                "Work_Hours": 10,
                "LMS_Logins": 40,
                "Scholarship": 1 if stu_idx in [0, 1] else 0,
                "First_Generation": 0,
                "Gender": "Female",
                "Housing_Status": "On-Campus",
                "End_of_Semester_Status": status,
            })
    return pd.DataFrame(rows)


def test_identify_resilience_cohorts(mock_trajectory_data):
    cohorts_df = identify_resilience_cohorts(mock_trajectory_data)
    assert len(cohorts_df) == 10
    assert "resilience_cohort" in cohorts_df.columns
    
    # Students 0 and 1 should be in Resilient Recovery
    rec_stus = cohorts_df[cohorts_df["resilience_cohort"] == "Resilient Recovery"]["Student_ID"].tolist()
    assert "STU_000" in rec_stus
    assert "STU_001" in rec_stus
    
    # Students 2 and 3 should be in Continuing Decline
    dec_stus = cohorts_df[cohorts_df["resilience_cohort"] == "Continuing Decline"]["Student_ID"].tolist()
    assert "STU_002" in dec_stus
    assert "STU_003" in dec_stus


def test_resilience_odds_ratios_structure():
    res = run_resilience_analysis()
    assert isinstance(res, ResilienceAnalysisResult)
    assert res.n_recovery_students > 0
    assert res.n_continuing_decline_students > 0
    assert res.dropout_rate_recovery < res.dropout_rate_continuing_decline
    
    # Check odds ratios
    assert "Feature" in res.odds_ratios.columns
    assert "Odds_Ratio" in res.odds_ratios.columns
    assert "CI_Lower_95" in res.odds_ratios.columns
    assert "CI_Upper_95" in res.odds_ratios.columns
    assert "mean_advising" in res.odds_ratios["Feature"].values
    assert "mean_financial_stress" in res.odds_ratios["Feature"].values
