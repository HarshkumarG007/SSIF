import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_endpoint():
    """Verify system diagnostics and regulatory framework listing."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "2.0.0" in data["version"]
    assert any("EU AI Act" in f for f in data["regulatory_frameworks"])
    assert "X-Request-ID" in resp.headers
    assert "X-Process-Time-Ms" in resp.headers


def test_retention_risk_prediction():
    """Verify real-time SIS/LMS risk scoring endpoint."""
    payload = {
        "student_id": "STU_TEST_01",
        "gpa": 2.2,
        "gpa_slope": -0.25,
        "financial_stress": 4,
        "work_hours": 30.0,
        "attendance": 70.0,
        "first_gen": True,
        "scholarship": False,
        "semester": 3,
    }
    resp = client.post("/v1/retention/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["student_id"] == "STU_TEST_01"
    assert data["calibrated_dropout_prob"] > 0.50
    assert "Risk" in data["risk_level"]
    assert len(data["confidence_interval_95"]) == 2
    assert len(data["primary_risk_drivers"]) >= 2
    assert "EU AI Act" in data["regulatory_notice"]


def test_retention_validation_error():
    """Verify 422 Unprocessable Entity when GPA exceeds bounds (4.0)."""
    invalid_payload = {
        "gpa": 4.5,  # Out of bounds [0.0, 4.0]
        "gpa_slope": 0.0,
        "financial_stress": 2,
        "work_hours": 10.0,
        "attendance": 90.0,
        "first_gen": False,
        "scholarship": True,
    }
    resp = client.post("/v1/retention/predict", json=invalid_payload)
    assert resp.status_code == 422


def test_recourse_solver_endpoint():
    """Verify counterfactual recourse optimization endpoint."""
    payload = {
        "student": {
            "gpa": 2.8,
            "gpa_slope": -0.10,
            "financial_stress": 3,
            "work_hours": 20.0,
            "attendance": 80.0,
            "first_gen": False,
            "scholarship": False,
            "semester": 3,
        },
        "target_risk": 0.20,
    }
    resp = client.post("/v1/recourse/solve", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_feasible"] is True
    assert data["counterfactual_risk"] <= 0.20
    assert len(data["action_plan"]) >= 1
    assert "EU AI Act" in data["disclaimer"]


def test_placement_evaluation_endpoint():
    """Verify employability readiness evaluation endpoint."""
    payload = {
        "ssc_p": 72.0,
        "hsc_p": 75.0,
        "degree_p": 68.0,
        "etest_p": 80.0,
        "mba_p": 65.0,
        "workex": True,
        "specialisation": "Mkt&Fin",
    }
    resp = client.post("/v1/placement/evaluate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert 0.0 <= data["placement_probability"] <= 1.0
    assert "confidence_interval_95" in data
    assert len(data["confidence_interval_95"]) == 2
    assert 0.0 <= data["confidence_interval_95"][0] <= data["confidence_interval_95"][1] <= 1.0
    assert "Employability" in data["readiness_tier"]
    assert len(data["expected_salary_inr_range"]) == 2
    assert any("work experience" in factor.lower() for factor in data["top_readiness_factors"])


def test_causal_inquiry_endpoint():
    """Verify Double ML causal inquiry endpoint."""
    payload = {
        "intervention": "Scholarship",
        "outcome": "Target_Dropout_Next_Sem",
    }
    resp = client.post("/v1/causal/estimate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["causal_ate"] < 0  # Reduces dropout
    assert data["p_value"] < 0.001
    assert data["e_value"] > 1.0
