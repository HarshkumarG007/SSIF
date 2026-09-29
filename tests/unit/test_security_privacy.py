"""
test_security_privacy.py — Automated Security, Privacy, and Regulatory Regression Tests
Student Success Intelligence Framework (SSIF)

Verifies remediations for RED-team security and compliance findings (SEC-01 through SEC-11):
  - SEC-01: FERPA Small-cell metric suppression (n < 5)
  - SEC-02: Quasi-identifier anonymization & discretization on Dataset B
  - SEC-03: Supply chain cryptographic SHA-256 verification of compiler binaries
  - SEC-04: Streamlit CORS protection enabled
  - SEC-05: Recourse custom predictor support & EU AI Act Art. 14 advisory
  - SEC-06: DevSecOps least-privilege permissions & action commit pinning
  - SEC-07: Pinned requirements lockfile integrity
  - SEC-08 & SEC-11: Legal NOTICE, Apache-2.0 copyright & dataset attribution
  - SEC-09: LMS Logins causal forward-fill temporal leakage elimination
  - SEC-10: Dataset B EPV <= 6 DoF feature constraint
"""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.explainability.recourse import (
    StudentProfile,
    find_counterfactual_recourse,
    RecourseRecommendation,
)
from src.placement.features import (
    anonymize_placement_quasi_identifiers,
    prepare_placement_classification_data,
)
from src.retention.features import compute_longitudinal_trajectories


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_streamlit_cors_hardened():
    """SEC-04: Streamlit configuration must have CORS enabled."""
    config_path = REPO_ROOT / ".streamlit" / "config.toml"
    assert config_path.exists(), "Streamlit config file missing!"
    content = config_path.read_text(encoding="utf-8")
    assert "enableCORS = true" in content, "enableCORS must be set to true in config.toml"


def test_exp003_small_cell_suppression():
    """SEC-01: EXP-003 reports must suppress continuous metrics for n < 5 cells (FERPA)."""
    csv_path = REPO_ROOT / "reports" / "experiments" / "EXP-003" / "qualified_excluded_profiles.csv"
    assert csv_path.exists(), "EXP-003 qualified_excluded_profiles.csv missing!"
    df = pd.read_csv(csv_path)
    
    # Identify small-cell cohorts (0 < n < 5)
    small_cells = df[(df["n_qualified_excluded"] > 0) & (df["n_qualified_excluded"] < 5)]
    assert len(small_cells) > 0, "Expected small-cell demographic cohorts in test data"
    
    for _, row in small_cells.iterrows():
        # Metric columns must be NaN / suppressed
        assert pd.isna(row["mean_gpa_slope"]), f"Cell n={row['n_qualified_excluded']} leaked mean_gpa_slope"
        assert pd.isna(row["mean_max_gpa"]), f"Cell n={row['n_qualified_excluded']} leaked mean_max_gpa"


def test_tectonic_sha256_verification_enforced():
    """SEC-03: compile_paper.py must cryptographically verify SHA-256 before extraction."""
    script_path = REPO_ROOT / "scripts" / "compile_paper.py"
    assert script_path.exists(), "compile_paper.py missing!"
    content = script_path.read_text(encoding="utf-8")
    assert "TECTONIC_ZIP_SHA256 =" in content
    assert "f61ce51f0b0ade1015b7de7ef368541c5424e9756ecbd0d7af97d6d48030845f" in content
    assert "hashlib.sha256" in content
    assert "Cryptographic hash mismatch" in content


def test_github_actions_supply_chain_hardened():
    """SEC-06: CI workflow must enforce least privilege and pin action commit SHAs."""
    ci_path = REPO_ROOT / ".github" / "workflows" / "ci.yml"
    assert ci_path.exists(), "ci.yml missing!"
    content = ci_path.read_text(encoding="utf-8")
    assert "permissions:" in content
    assert "contents: read" in content
    assert "actions/checkout@b4ffde65f46336ab88eb53be808477a3936bae11" in content
    assert "actions/setup-python@39cd14951b08e74b54015e9e001cdefcf80e669f" in content


def test_requirements_lock_exists_and_pinned():
    """SEC-07: Pinned requirements.lock must exist with exact version specifications."""
    lock_path = REPO_ROOT / "requirements.lock"
    assert lock_path.exists(), "requirements.lock missing!"
    lines = [
        line.strip()
        for line in lock_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert len(lines) >= 20, "requirements.lock should contain all core dependencies"
    for line in lines:
        assert "==" in line, f"Dependency line '{line}' is not pinned to an exact version (==)"


def test_recourse_predictor_and_disclaimer():
    """SEC-05: Recourse must support custom predictors and include EU AI Act / FERPA advisory."""
    prof = StudentProfile(
        gpa=2.5,
        gpa_slope=-0.2,
        financial_stress=4,
        work_hours=25.0,
        attendance=75.0,
        first_gen=True,
        scholarship=False,
    )
    rec = find_counterfactual_recourse(prof, target_risk=0.20)
    assert hasattr(rec, "disclaimer")
    assert "EU AI Act" in rec.disclaimer
    assert "Art. 14" in rec.disclaimer

    # Test custom predictor
    custom_called = [False]
    def custom_predictor(p: StudentProfile) -> float:
        custom_called[0] = True
        return 0.10 if p.scholarship else 0.50

    rec_custom = find_counterfactual_recourse(prof, target_risk=0.15, predictor=custom_predictor)
    assert custom_called[0], "Custom predictor was not invoked by recourse solver"
    assert rec_custom.is_feasible is True
    assert rec_custom.counterfactual_risk <= 0.15


def test_placement_quasi_identifier_anonymization():
    """SEC-02: Quasi-identifiers in Dataset B must support interval binning."""
    df_sample = pd.DataFrame({
        "ssc_p": [62.4, 78.9],
        "hsc_p": [65.0, 81.2],
        "degree_p": [59.9, 74.5],
        "etest_p": [70.0, 85.0],
        "mba_p": [64.1, 68.3],
    })
    anon = anonymize_placement_quasi_identifiers(df_sample, bin_width=5.0)
    assert "ssc_p_binned" in anon.columns
    assert anon["ssc_p_binned"].iloc[0] == "[60, 65)"
    assert anon["ssc_p_binned"].iloc[1] == "[75, 80)"


def test_placement_epv_constrained_dof():
    """SEC-10: Dataset B constrained DoF feature set must not exceed 6 variables."""
    df_sample = pd.DataFrame({
        "gender": ["M", "F"],
        "ssc_p": [60.0, 70.0],
        "ssc_b": ["Others", "Central"],
        "hsc_p": [60.0, 70.0],
        "hsc_b": ["Others", "Central"],
        "hsc_s": ["Commerce", "Science"],
        "degree_p": [60.0, 70.0],
        "degree_t": ["Sci&Tech", "Comm&Mgmt"],
        "workex": ["No", "Yes"],
        "etest_p": [60.0, 70.0],
        "specialisation": ["Mkt&HR", "Mkt&Fin"],
        "mba_p": [60.0, 70.0],
        "status": ["Placed", "Not Placed"],
    })
    X, y, feats = prepare_placement_classification_data(df_sample, constrained_dof=True)
    assert X.shape[1] <= 6, f"Constrained feature count {X.shape[1]} exceeds EPV limit of 6 DoF"


def test_lms_logins_strictly_causal_no_future_leakage():
    """SEC-09: Imputation of LMS_Logins must be forward-causal (no lookahead leakage)."""
    # Create two histories:
    # Student A: Sem 1 LMS missing, Sem 2 LMS = 15
    # Future Sem 3 is 20 in df1, but 200 in df2
    df1 = pd.DataFrame({
        "Student_ID": [101, 101, 101],
        "Semester": [1, 2, 3],
        "Sem_GPA": [3.0, 3.2, 3.4],
        "LMS_Logins": [np.nan, 15.0, 20.0],
        "Attendance": [80.0, 85.0, 90.0],
        "Failed_Courses": [0, 0, 0],
        "Advising_Visits": [0, 0, 0],
    })
    df2 = pd.DataFrame({
        "Student_ID": [101, 101, 101],
        "Semester": [1, 2, 3],
        "Sem_GPA": [3.0, 3.2, 3.4],
        "LMS_Logins": [np.nan, 15.0, 200.0],  # Future semester 3 is drastically different
        "Attendance": [80.0, 85.0, 90.0],
        "Failed_Courses": [0, 0, 0],
        "Advising_Visits": [0, 0, 0],
    })
    res1 = compute_longitudinal_trajectories(df1)
    res2 = compute_longitudinal_trajectories(df2)

    # Historical trajectory features at Sem 1 and Sem 2 must be identical
    assert np.isclose(res1.loc[res1["Semester"] == 2, "lms_slope"].iloc[0],
                      res2.loc[res2["Semester"] == 2, "lms_slope"].iloc[0])
    # Future semester 3 slope will differ
    assert not np.isclose(res1.loc[res1["Semester"] == 3, "lms_slope"].iloc[0],
                          res2.loc[res2["Semester"] == 3, "lms_slope"].iloc[0])



def test_legal_notice_and_license_attribution():
    """SEC-08 & SEC-11: Legal NOTICE file and LICENSE must exist and properly attribute authors."""
    license_path = REPO_ROOT / "LICENSE"
    notice_path = REPO_ROOT / "NOTICE"

    assert license_path.exists(), "LICENSE missing!"
    assert notice_path.exists(), "NOTICE missing!"

    lic_content = license_path.read_text(encoding="utf-8")
    assert "Copyright 2026 Harshkumar G. and SSIF Contributors" in lic_content

    not_content = notice_path.read_text(encoding="utf-8")
    assert "Razan Ihab Abdellatif" in not_content
    assert "Amey Thakur" in not_content
    assert "Samar Talwar" in not_content
    assert "Sri Syra" in not_content
    assert "EU AI Act" in not_content
    assert "FERPA" in not_content
    assert "DPDP Act" in not_content
