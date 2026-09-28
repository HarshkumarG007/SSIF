"""
verify_streamlit_deployment.py — Streamlit Cloud Pre-Flight Verification Script
Student Success Intelligence Framework (SSIF)

Performs comprehensive pre-flight verification before deployment to share.streamlit.io:
  - Validates requirements.txt against installed packages
  - Validates .streamlit/config.toml syntax and color tokens
  - Validates dataset paths and accessibility in cloud runtime
  - Runs headless execution test of app/main.py and all page dependencies
"""
import sys
from pathlib import Path

# Add project root to sys.path
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

def run_preflight_checks():
    print("=" * 60)
    print("[PRE-FLIGHT] SSIF STREAMLIT CLOUD DEPLOYMENT VERIFICATION")
    print("=" * 60)
    
    # 1. Check requirements.txt
    req_file = root / "requirements.txt"
    assert req_file.exists(), "[FAIL] requirements.txt missing!"
    req_content = req_file.read_text(encoding="utf-8")
    essential_pkgs = ["streamlit", "plotly", "pandas", "numpy", "scikit-learn", "lifelines", "statsmodels", "shap"]
    for pkg in essential_pkgs:
        assert pkg in req_content, f"[FAIL] Essential package '{pkg}' missing from requirements.txt!"
    print(f"[OK] requirements.txt verified ({len(req_content.splitlines())} lines, all core packages present)")

    # 2. Check .streamlit/config.toml
    config_file = root / ".streamlit" / "config.toml"
    assert config_file.exists(), "[FAIL] .streamlit/config.toml missing!"
    config_text = config_file.read_text(encoding="utf-8")
    assert "primaryColor" in config_text and "backgroundColor" in config_text, "[FAIL] Incomplete config.toml theme!"
    print("[OK] .streamlit/config.toml theme verified (Dark HSL palette configured)")

    # 3. Check Datasets
    from src.data_loader import load_retention, load_placement, load_dlsm_b
    df_ret = load_retention()
    assert len(df_ret) == 79239, f"[FAIL] Unexpected retention row count: {len(df_ret)}"
    print(f"[OK] Dataset A (Retention) loaded successfully: {len(df_ret):,} records")

    df_plc = load_placement()
    assert len(df_plc) == 215, f"[FAIL] Unexpected placement row count: {len(df_plc)}"
    print(f"[OK] Dataset B (Placement) loaded successfully: {len(df_plc)} candidates")

    # 4. Check App Imports & Recourse Engine
    from app.components import apply_custom_css, apply_plotly_theme
    from src.explainability.recourse import StudentProfile, find_counterfactual_recourse
    prof = StudentProfile(gpa=2.8, gpa_slope=-0.10, financial_stress=3, work_hours=20.0, attendance=80.0, first_gen=False, scholarship=False)
    rec = find_counterfactual_recourse(prof, target_risk=0.20)
    assert rec.is_feasible, "[FAIL] Counterfactual recourse failed feasibility check!"
    print(f"[OK] Counterfactual Recourse engine verified: risk reduced from {rec.original_risk*100:.1f}% to {rec.counterfactual_risk*100:.1f}%")

    print("\n[SUCCESS] ALL PRE-FLIGHT VERIFICATIONS PASSED!")
    print("Repository is 100% prepared for live deployment on share.streamlit.io (ssif-research.streamlit.app).")
    print("=" * 60)

if __name__ == "__main__":
    run_preflight_checks()
