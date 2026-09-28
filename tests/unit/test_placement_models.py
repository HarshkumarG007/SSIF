"""
tests/unit/test_placement_models.py
Unit tests for placement feature engineering and modeling (TASK-083).
"""
import numpy as np
import pandas as pd
import pytest

from src.placement.features import (
    engineer_placement_features,
    prepare_placement_classification_data,
    prepare_salary_regression_data,
)


@pytest.fixture
def mock_placement_df():
    return pd.DataFrame({
        "sl_no": [1, 2, 3, 4],
        "gender": ["M", "F", "M", "F"],
        "ssc_p": [67.0, 79.33, 65.0, 56.0],
        "ssc_b": ["Others", "Central", "Central", "Central"],
        "hsc_p": [91.0, 78.33, 68.0, 52.0],
        "hsc_b": ["Others", "Others", "Central", "Central"],
        "hsc_s": ["Commerce", "Science", "Arts", "Science"],
        "degree_p": [58.0, 77.48, 64.0, 52.0],
        "degree_t": ["Sci&Tech", "Sci&Tech", "Comm&Mgmt", "Sci&Tech"],
        "workex": ["No", "Yes", "No", "No"],
        "etest_p": [55.0, 86.5, 75.0, 66.0],
        "specialisation": ["Mkt&HR", "Mkt&Fin", "Mkt&Fin", "Mkt&HR"],
        "mba_p": [58.8, 66.28, 57.8, 59.43],
        "status": ["Placed", "Placed", "Placed", "Not Placed"],
        "salary": [270000.0, 200000.0, 250000.0, np.nan],
    })


class TestPlacementPipeline:
    def test_feature_engineering_composite_scores(self, mock_placement_df):
        res = engineer_placement_features(mock_placement_df)
        assert "academic_progression" in res.columns
        assert "degree_deviation" in res.columns
        assert "composite_academic_score" in res.columns

        # Check formula for row 0: hsc_p(91) - ssc_p(67) = 24.0
        assert np.isclose(res["academic_progression"].iloc[0], 24.0)

    def test_zero_salary_leakage_in_classification(self, mock_placement_df):
        X, y, feats = prepare_placement_classification_data(mock_placement_df)
        assert "salary" not in X.columns
        assert "status" not in X.columns
        assert len(y) == 4
        assert y.iloc[3] == 0  # Not Placed

    def test_salary_regression_placed_only(self, mock_placement_df):
        X, y, feats = prepare_salary_regression_data(mock_placement_df)
        # Row 4 is unplaced -> must be excluded
        assert len(y) == 3
        assert not y.isna().any()
        assert "status" not in X.columns
