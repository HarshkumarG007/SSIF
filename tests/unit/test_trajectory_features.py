"""
tests/unit/test_trajectory_features.py
Unit tests for longitudinal trajectory feature engineering (TASK-043).
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.retention.features import (
    RetentionFeaturePipeline,
    compute_longitudinal_trajectories,
    prepare_retention_dataset,
)


@pytest.fixture
def mock_longitudinal_student():
    """Create a mock 4-semester trajectory for a single student."""
    return pd.DataFrame({
        "Student_ID": ["STU_001"] * 4,
        "Semester": [1, 2, 3, 4],
        "Age": [19, 19, 20, 20],
        "Gender": ["Female"] * 4,
        "First_Generation": [0] * 4,
        "Housing_Status": ["On-Campus"] * 4,
        "Scholarship": [1] * 4,
        "Tuition_Base": [12000.0] * 4,
        "Family_Income": [65000.0] * 4,
        "Household_Size": [3] * 4,
        "Course_Load": [15, 15, 16, 14],
        "Work_Hours": [10, 10, 15, 20],
        "Emergency_Expense": [0, 0, 500, 0],
        "LMS_Logins": [45, 40, 30, 20],
        "Advising_Visits": [1, 0, 2, 3],
        "Failed_Courses": [0, 0, 1, 0],
        "Financial_Stress": [2, 2, 4, 5],
        "Attendance": [95.0, 90.0, 80.0, 75.0],
        "Sem_GPA": [3.6, 3.2, 2.8, 2.4],  # Constant linear decline of -0.4 per semester
        "Target_Dropout_Next_Sem": [0, 0, 0, 1],
        "End_of_Semester_Status": ["Enrolled", "Enrolled", "Enrolled", "Dropout"],
        "Censored": [0, 0, 0, 0],
    })


class TestTrajectoryFeatures:
    def test_single_semester_produces_nan_slope(self):
        """Single-semester observations must produce NaN for slope and velocity."""
        df = pd.DataFrame({
            "Student_ID": ["STU_ONLY_ONE"],
            "Semester": [1],
            "Sem_GPA": [3.5],
            "Attendance": [90.0],
            "LMS_Logins": [50],
            "Failed_Courses": [0],
            "Advising_Visits": [0],
        })
        res = compute_longitudinal_trajectories(df)
        assert res["is_single_semester"].iloc[0] == 1
        assert pd.isna(res["gpa_slope"].iloc[0])
        assert pd.isna(res["gpa_velocity"].iloc[0])
        assert pd.isna(res["gpa_volatility"].iloc[0])

    def test_exact_ols_slope_calculation(self, mock_longitudinal_student):
        """GPA slope must match exact OLS linear regression."""
        res = compute_longitudinal_trajectories(mock_longitudinal_student)

        # Sem 1: NaN
        assert pd.isna(res["gpa_slope"].iloc[0])

        # Sem 2: delta = (3.2 - 3.6)/(2 - 1) = -0.4
        assert np.isclose(res["gpa_slope"].iloc[1], -0.4)

        # Sem 4: linregress over semesters [1,2,3,4] and GPAs [3.6, 3.2, 2.8, 2.4]
        scipy_res = stats.linregress([1, 2, 3, 4], [3.6, 3.2, 2.8, 2.4])
        assert np.isclose(res["gpa_slope"].iloc[3], scipy_res.slope)
        assert np.isclose(res["gpa_slope"].iloc[3], -0.4)

    def test_temporal_causality_zero_future_leakage(self, mock_longitudinal_student):
        """
        CRITICAL TEST (RULE-009):
        Modifying semester 4 GPA must NOT alter semester 2 or semester 3 trajectory features.
        """
        df1 = mock_longitudinal_student.copy()
        df2 = mock_longitudinal_student.copy()
        # Change semester 4 GPA dramatically in df2
        df2.loc[df2["Semester"] == 4, "Sem_GPA"] = 4.0

        res1 = compute_longitudinal_trajectories(df1)
        res2 = compute_longitudinal_trajectories(df2)

        # Semester 1, 2, and 3 must have identical features in both!
        for sem_idx in [0, 1, 2]:
            assert np.isclose(
                res1["gpa_recent_mean"].iloc[sem_idx],
                res2["gpa_recent_mean"].iloc[sem_idx]
            )
            if not pd.isna(res1["gpa_slope"].iloc[sem_idx]):
                assert np.isclose(
                    res1["gpa_slope"].iloc[sem_idx],
                    res2["gpa_slope"].iloc[sem_idx]
                )

        # Semester 4 slope should differ
        assert not np.isclose(res1["gpa_slope"].iloc[3], res2["gpa_slope"].iloc[3])

    def test_decline_and_recovery_indices(self):
        """Test decline counter and recovery detection."""
        df = pd.DataFrame({
            "Student_ID": ["S1"] * 5,
            "Semester": [1, 2, 3, 4, 5],
            "Sem_GPA": [3.5, 3.2, 2.9, 3.3, 3.0],  # Drop, Drop, Recovery, Drop
            "Attendance": [90] * 5,
            "LMS_Logins": [50] * 5,
            "Failed_Courses": [0] * 5,
            "Advising_Visits": [0] * 5,
        })
        res = compute_longitudinal_trajectories(df)
        # Semester 4 increased from 2.9 to 3.3 after decline -> recovery_index should be 1
        assert res["recovery_index"].iloc[3] == 1
        # Semester 5 dropped from 3.3 to 3.0 -> recovery_index should be 0
        assert res["recovery_index"].iloc[4] == 0

    def test_pipeline_fit_transform_no_leakage(self, mock_longitudinal_student):
        """Test Scikit-Learn transformer interface."""
        pipe = RetentionFeaturePipeline(include_trajectories=True)
        pipe.fit(mock_longitudinal_student)
        transformed = pipe.transform(mock_longitudinal_student)

        assert "gpa_slope" in transformed.columns
        assert "gpa_velocity" in transformed.columns
        assert not transformed["gpa_slope"].isna().any()  # Imputed with 0 for sem 1
