"""
schemas.py — Pydantic v2 Data Models for SSIF Enterprise REST API
Student Success Intelligence Framework (SSIF)
"""
from __future__ import annotations

from typing import Any, List, Optional
from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "2.0.0"
    engine: str = "Student Success Intelligence Framework (SSIF)"
    regulatory_frameworks: list[str] = [
        "EU AI Act Annex III (High-Risk AI Systems / Art. 14)",
        "FERPA (20 U.S.C. § 1232g)",
        "India DPDP Act (2023)",
    ]


class StudentProfileInput(BaseModel):
    """Real-time student state ingested from Student Information System (SIS) or LMS."""
    student_id: Optional[str] = Field(None, description="Pseudonymized student ID")
    gpa: float = Field(..., ge=0.0, le=4.0, description="Current cumulative GPA", examples=[2.75])
    gpa_slope: float = Field(..., ge=-2.0, le=2.0, description="Semester-over-semester GPA trajectory slope", examples=[-0.15])
    financial_stress: int = Field(..., ge=1, le=5, description="Self-reported financial stress (1=Low, 5=Severe)", examples=[3])
    work_hours: float = Field(..., ge=0.0, le=80.0, description="Weekly external employment hours", examples=[25.0])
    attendance: float = Field(..., ge=0.0, le=100.0, description="Course attendance rate percentage", examples=[78.5])
    first_gen: bool = Field(..., description="First-generation university student flag", examples=[True])
    scholarship: bool = Field(..., description="Institutional scholarship recipient flag", examples=[False])
    semester: int = Field(3, ge=1, le=8, description="Current enrolled semester index", examples=[3])


class RiskPredictionResponse(BaseModel):
    """Calibrated academic departure risk score with confidence intervals."""
    student_id: Optional[str]
    calibrated_dropout_prob: float
    risk_level: str  # "Low", "Moderate", "Elevated", "Critical"
    confidence_interval_95: list[float]
    primary_risk_drivers: list[str]
    brier_score_calibration: float = 0.1768
    regulatory_notice: str = (
        "Advisory decision-support estimate. Must not be used for automated punitive or exclusionary "
        "determinations without human academic counseling oversight (EU AI Act Art. 14 / FERPA compliant)."
    )


class RecourseRequest(BaseModel):
    """Request payload to generate minimal-effort counterfactual action plans."""
    student: StudentProfileInput
    target_risk: float = Field(0.15, ge=0.01, le=0.50, description="Target persistence risk threshold", examples=[0.15])


class RecourseResponse(BaseModel):
    """Optimal minimal-cost policy levers to return student to persistence zone."""
    original_risk: float
    target_risk: float
    counterfactual_risk: float
    risk_reduction_pct: float
    is_feasible: bool
    effort_score: float
    action_plan: list[str]
    disclaimer: str


class PlacementEvaluationInput(BaseModel):
    """Candidate profile for employability readiness evaluation."""
    ssc_p: float = Field(..., ge=0.0, le=100.0, description="10th grade exam percentage", examples=[67.0])
    hsc_p: float = Field(..., ge=0.0, le=100.0, description="12th grade exam percentage", examples=[70.0])
    degree_p: float = Field(..., ge=0.0, le=100.0, description="Undergraduate degree percentage", examples=[65.0])
    etest_p: float = Field(..., ge=0.0, le=100.0, description="Employability test percentage", examples=[75.0])
    mba_p: float = Field(..., ge=0.0, le=100.0, description="MBA percentage", examples=[62.0])
    workex: bool = Field(..., description="Prior work experience flag", examples=[True])
    specialisation: str = Field("Mkt&Fin", description="Specialisation track (Mkt&HR or Mkt&Fin)", examples=["Mkt&Fin"])


class PlacementEvaluationResponse(BaseModel):
    placement_probability: float
    readiness_tier: str  # "High Employability", "Moderate Employability", "Needs Development"
    expected_salary_inr_range: list[int]
    top_readiness_factors: list[str]
    sample_size_limitation: str = "Evaluated against benchmark cohort N=215 with constrained EPV regularization."


class CausalInquiryRequest(BaseModel):
    intervention: str = Field("Scholarship", description="Intervention name (e.g. Scholarship, Work_Hours_Relief)")
    outcome: str = Field("Target_Dropout_Next_Sem", description="Target outcome (Sem_GPA or Target_Dropout_Next_Sem)")


class CausalInquiryResponse(BaseModel):
    intervention: str
    outcome: str
    causal_ate: float
    ci_95: list[float]
    p_value: float
    e_value: float
    interpretation: str
