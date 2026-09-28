"""
test_feasibility_auditor.py — Unit & Integration Tests for Tabular Feasibility Auditor
Student Success Intelligence Framework (SSIF)

Integrates dataset_feasibility_audit.py into pytest to ensure data leakage,
provenance, signal, structure, and fairness checks act as continuous automated gates.
"""
from __future__ import annotations

import pandas as pd
import pytest

from dataset_feasibility_audit import run_feasibility_audit
from src.data_loader import load_placement, load_retention


def test_gender_canonicalization_unifies_categories():
    """Verify that load_retention() fixes the 8-variant Gender fragmentation bug."""
    df_ret = load_retention()
    unique_genders = set(df_ret["Gender"].unique())
    expected = {"Female", "Male", "Other", "Prefer not to say"}
    assert unique_genders == expected, f"Unexpected fragmented gender categories: {unique_genders}"
    # Verify no lowercase or single-letter variants exist
    for bad in ["f", "F", "female", "m", "M", "male"]:
        assert bad not in unique_genders


def test_placement_audit_catches_salary_leakage():
    """Verifies that the feasibility auditor catches salary as critical post-outcome leakage."""
    df_plc = load_placement()
    report = run_feasibility_audit(
        df=df_plc,
        target="status",
        id_cols=["sl_no"],
        drop_cols=[],  # Do NOT drop salary
        verbose=False,
    )
    assert "leakage" in report["gates"]
    assert report["gates"]["leakage"]["status"] == "FAIL"
    assert "salary" in report["gates"]["leakage"]["detail"].lower()


def test_placement_audit_clean_passes_leakage():
    """Verifies that dropping salary allows placement classification to pass the leakage gate."""
    df_plc = load_placement()
    report = run_feasibility_audit(
        df=df_plc,
        target="status",
        id_cols=["sl_no"],
        drop_cols=["salary"],
        verbose=False,
    )
    assert report["gates"]["leakage"]["status"] == "PASS"
    assert report["gates"]["signal"]["status"] == "PASS"


def test_retention_audit_catches_end_of_semester_status_leakage():
    """Verifies that including End_of_Semester_Status triggers FAIL on the leakage gate."""
    df_ret = load_retention().sample(n=1000, random_state=42)
    report = run_feasibility_audit(
        df=df_ret,
        target="Target_Dropout_Next_Sem",
        id_cols=["Student_ID"],
        drop_cols=["Censored"],  # Leaving End_of_Semester_Status in features
        max_rows=1000,
        verbose=False,
    )
    assert report["gates"]["leakage"]["status"] == "FAIL"
    assert "end_of_semester_status" in report["gates"]["leakage"]["detail"].lower()


def test_retention_audit_clean_passes_gates():
    """Verifies that clean retention panel passes leakage, signal, and group structure gates."""
    df_ret = load_retention().sample(n=2000, random_state=42)
    report = run_feasibility_audit(
        df=df_ret,
        target="Target_Dropout_Next_Sem",
        id_cols=["Student_ID"],
        drop_cols=["End_of_Semester_Status", "Censored", "Semester"],
        max_rows=2000,
        verbose=False,
    )
    assert report["gates"]["leakage"]["status"] == "PASS"
    assert report["gates"]["signal"]["status"] == "PASS"
