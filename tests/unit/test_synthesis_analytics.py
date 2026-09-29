"""
test_synthesis_analytics.py — Unit Tests for Cross-Dataset Synthesis Analytics
Student Success Intelligence Framework (SSIF)
"""
from src.cross_dataset.synthesis_analytics import (
    compute_cross_dataset_synthesis_metrics,
    compute_placement_deep_insights,
    compute_retention_deep_insights,
)


def test_retention_deep_insights():
    res = compute_retention_deep_insights()
    assert "<1.5" in res.gpa_brackets
    assert res.gpa_brackets["<1.5"] > res.gpa_brackets["3.5-4.0"]
    assert res.course_load_hazard[18] > res.course_load_hazard[15]
    assert res.scholarship_q1_reduction > 5.0  # At least 5% drop in Q1


def test_placement_deep_insights():
    res = compute_placement_deep_insights()
    assert res.workex_odds_ratio > 3.0
    assert res.workex_p_value < 0.01
    assert res.rescue_rate_low_gpa_workex > res.rescue_rate_low_gpa_no_workex
    assert res.rescue_rate_low_gpa_workex > 60.0
    assert res.mkt_fin_placement_rate > res.mkt_hr_placement_rate
    assert res.board_prestige_p_value > 0.05  # Board prestige is not significant


def test_cross_dataset_synthesis_metrics():
    res = compute_cross_dataset_synthesis_metrics()
    assert res.work_hours_retention_or > 1.0  # In-college work increases dropout odds
    assert res.work_hours_retention_p_val < 0.01
    assert res.workex_placement_or > 3.0  # Post-degree workex supercharges placement
    assert res.workex_placement_p_val < 0.01
    assert res.placement_hiring_threshold_p == 65.0
