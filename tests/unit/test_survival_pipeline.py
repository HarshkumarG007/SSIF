"""
tests/unit/test_survival_pipeline.py
Unit tests for survival analysis pipeline (TASK-061).
"""
import numpy as np
import pandas as pd
import pytest

from src.retention.survival import (
    prepare_survival_dataset,
    run_cox_proportional_hazards,
    run_kaplan_meier_analysis,
)


@pytest.fixture
def mock_retention_data():
    """Mock retention panel for testing survival calculations."""
    rows = []
    # Student 1: 3 semesters, dropped out
    for s in [1, 2, 3]:
        rows.append({
            "Student_ID": "S1", "Semester": s, "Age": 20, "Gender": "Female",
            "First_Generation": 1, "Scholarship": 0, "Family_Income": 40000.0,
            "Financial_Stress": 4, "Attendance": 80.0, "Sem_GPA": 2.5,
            "End_of_Semester_Status": "Dropped_Out" if s == 3 else "Enrolled",
            "Censored": 0,
        })
    # Student 2: 4 semesters, censored
    for s in [1, 2, 3, 4]:
        rows.append({
            "Student_ID": "S2", "Semester": s, "Age": 21, "Gender": "Male",
            "First_Generation": 0, "Scholarship": 1, "Family_Income": 80000.0,
            "Financial_Stress": 1, "Attendance": 95.0, "Sem_GPA": 3.8,
            "End_of_Semester_Status": "Enrolled",
            "Censored": 1,
        })
    return pd.DataFrame(rows)


class TestSurvivalPipeline:
    def test_survival_dataset_preparation(self, mock_retention_data):
        student_surv, cph_data = prepare_survival_dataset(mock_retention_data)
        assert len(student_surv) == 2
        # S1 dropped out at semester 3
        s1 = student_surv[student_surv["Student_ID"] == "S1"].iloc[0]
        assert s1["duration"] == 3.0
        assert s1["event"] == 1

        # S2 censored at semester 4
        s2 = student_surv[student_surv["Student_ID"] == "S2"].iloc[0]
        assert s2["duration"] == 4.0
        assert s2["event"] == 0

    def test_kaplan_meier_properties(self, mock_retention_data):
        student_surv, cph_data = prepare_survival_dataset(mock_retention_data)
        km_table, logrank_results = run_kaplan_meier_analysis(student_surv, mock_retention_data)
        # Check survival probabilities decrease or stay flat
        probs = list(km_table.values())
        for i in range(len(probs) - 1):
            assert probs[i] >= probs[i + 1]
