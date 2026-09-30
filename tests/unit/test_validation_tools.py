"""
tests/unit/test_validation_tools.py
Tests for data profiler and missingness analyzer modules.
"""
import numpy as np
import pandas as pd
import pytest

from src.validation.data_profiler import profile_dataset, profile_placement, profile_retention
from src.validation.missingness_analyzer import analyze_missingness


class TestDataProfiler:
    def test_basic_profiling(self):
        df = pd.DataFrame({
            "id": ["1", "2", "3", "4"],
            "val": [10.0, 20.0, 30.0, np.nan],
            "cat": ["A", "B", "A", "B"]
        })
        profile = profile_dataset(df, "TestDataset", id_col="id")
        assert profile.n_rows == 4
        assert profile.n_cols == 3
        assert profile.total_missing_cells == 1
        assert profile.n_unique_ids == 4

        df_summary = profile.to_dataframe()
        assert len(df_summary) == 3
        assert "val" in df_summary["Column"].values

    def test_constant_column_warning(self):
        df = pd.DataFrame({
            "const": [1, 1, 1, 1],
            "var": [1, 2, 3, 4]
        })
        profile = profile_dataset(df, "ConstTest")
        warnings = [w for w in profile.quality_warnings if "constant" in w.lower()]
        assert len(warnings) >= 1

    def test_retention_specific_checks(self):
        df = pd.DataFrame({
            "Student_ID": ["S1", "S2"],
            "Semester": [1, 1],
            "Target_Dropout_Next_Sem": [0, 0]  # 0% dropout rate -> should warn on imbalance
        })
        profile = profile_retention(df)
        assert any("class imbalance" in w.lower() for w in profile.quality_warnings)
        assert any("only 1 semester" in w.lower() for w in profile.quality_warnings)


class TestMissingnessAnalyzer:
    def test_no_missingness(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        report = analyze_missingness(df, "CleanData")
        assert report.total_missing_cells == 0
        assert len(report.column_summaries) == 0

    def test_placement_structural_salary(self):
        df = pd.DataFrame({
            "status": ["Placed", "Placed", "Not Placed", "Not Placed"],
            "salary": [250000.0, 300000.0, np.nan, np.nan]
        })
        report = analyze_missingness(df, "SSIF-B Placement")
        assert len(report.column_summaries) == 1
        col_sum = report.column_summaries[0]
        assert col_sum.column == "salary"
        assert "Structural" in col_sum.mechanism_diagnosis
        assert "100%" in col_sum.evidence

    def test_add_missingness_indicators(self):
        from src.validation.missingness_analyzer import add_missingness_indicators
        df = pd.DataFrame({"income": [1000.0, np.nan, 3000.0], "age": [20, 21, 22]})
        res, ind_cols = add_missingness_indicators(df)
        assert "income_is_missing" in res.columns
        assert "income_is_missing" in ind_cols
        assert res["income_is_missing"].tolist() == [0.0, 1.0, 0.0]

    def test_missingness_indicator_transformer_pipeline(self):
        from src.validation.missingness_analyzer import MissingnessIndicatorTransformer
        train_df = pd.DataFrame({"val": [1.0, np.nan, 3.0], "static": [10, 20, 30]})
        test_df = pd.DataFrame({"val": [np.nan, 2.0], "static": [40, 50]})
        transformer = MissingnessIndicatorTransformer()
        transformer.fit(train_df)
        assert "val" in transformer.indicator_cols_

        transformed_test = transformer.transform(test_df)
        assert "val_is_missing" in transformed_test.columns
        assert transformed_test["val_is_missing"].tolist() == [1.0, 0.0]

