"""
test_recourse.py — Unit tests for algorithmic counterfactual recourse engine.
Student Success Intelligence Framework (SSIF)
"""
import pytest
from src.explainability.recourse import (
    StudentProfile,
    compute_calibrated_dropout_prob,
    find_counterfactual_recourse,
    RecourseRecommendation,
)


def test_low_risk_student_needs_no_action():
    low_risk = StudentProfile(
        gpa=3.6,
        gpa_slope=0.10,
        financial_stress=1,
        work_hours=5.0,
        attendance=95.0,
        first_gen=False,
        scholarship=True,
    )
    prob = compute_calibrated_dropout_prob(low_risk)
    assert prob < 0.15
    rec = find_counterfactual_recourse(low_risk, target_risk=0.15)
    assert rec.effort_score == 0.0
    assert rec.is_feasible is True
    assert len(rec.action_plan) == 1
    assert "already within target" in rec.action_plan[0]


def test_elevated_risk_student_generates_feasible_recourse():
    elevated_risk = StudentProfile(
        gpa=2.8,
        gpa_slope=-0.10,
        financial_stress=3,
        work_hours=20.0,
        attendance=80.0,
        first_gen=False,
        scholarship=False,
    )
    orig_prob = compute_calibrated_dropout_prob(elevated_risk)
    assert orig_prob > 0.40  # Elevated risk (42.7%)
    
    rec = find_counterfactual_recourse(elevated_risk, target_risk=0.20)
    assert isinstance(rec, RecourseRecommendation)
    assert rec.is_feasible is True
    assert rec.counterfactual_risk <= 0.20
    assert rec.risk_reduction_pct > 20.0
    assert len(rec.action_plan) >= 2
    
    plan_text = " ".join(rec.action_plan).lower()
    assert any(term in plan_text for term in ["scholarship", "attendance", "stress", "work"])


def test_acute_crisis_student_triggers_emergency_review():
    crisis_student = StudentProfile(
        gpa=2.0,
        gpa_slope=-0.40,
        financial_stress=5,
        work_hours=35.0,
        attendance=65.0,
        first_gen=True,
        scholarship=False,
    )
    orig_prob = compute_calibrated_dropout_prob(crisis_student)
    assert orig_prob > 0.90
    
    rec = find_counterfactual_recourse(crisis_student, target_risk=0.15)
    assert rec.is_feasible is False
    assert any("emergency" in act.lower() for act in rec.action_plan)


def test_recourse_carries_human_in_the_loop_disclaimer():
    prof = StudentProfile(
        gpa=2.8,
        gpa_slope=-0.10,
        financial_stress=3,
        work_hours=20.0,
        attendance=80.0,
        first_gen=False,
        scholarship=False,
    )
    rec = find_counterfactual_recourse(prof, target_risk=0.20)
    assert hasattr(rec, "disclaimer")
    assert "EU AI Act" in rec.disclaimer
    assert "human" in rec.disclaimer.lower()


def test_recourse_supports_custom_predictor_callable():
    prof = StudentProfile(
        gpa=2.8,
        gpa_slope=-0.10,
        financial_stress=3,
        work_hours=20.0,
        attendance=80.0,
        first_gen=False,
        scholarship=False,
    )
    # Define custom non-proxy predictor (e.g., simulating LightGBM/GBM endpoint)
    def mock_fitted_predictor(p: StudentProfile) -> float:
        # Mock risk where attendance + scholarship is highly rewarded
        risk = 0.50
        if p.scholarship:
            risk -= 0.35
        if p.attendance > 90.0:
            risk -= 0.10
        return max(0.01, risk)

    rec = find_counterfactual_recourse(prof, target_risk=0.15, predictor=mock_fitted_predictor)
    assert rec.is_feasible is True
    assert rec.counterfactual_risk <= 0.15
    assert any("scholarship" in act.lower() for act in rec.action_plan)

