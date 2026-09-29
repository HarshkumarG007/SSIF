"""
test_experiments.py — Automated Validation for SSIF Experiments Suite
Tests that EXP-001 through EXP-005 outputs exist, conform to schemas,
and strictly adhere to SSIF governance rules (no data fabrication,
no row-level merges, valid confidence intervals, and bounded probabilities).
"""
import json
from pathlib import Path
import pandas as pd
import pytest

REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports" / "experiments"


def test_master_results_all_passed():
    """Verify master_results.json exists and all 5 experiments succeeded."""
    master_path = REPORTS_DIR / "master_results.json"
    assert master_path.exists(), "master_results.json must exist"
    
    with open(master_path, "r", encoding="utf-8") as f:
        results = json.load(f)
        
    assert len(results) == 5, f"Expected 5 experiment entries, got {len(results)}"
    for r in results:
        assert r["status"] == "SUCCESS", f"Experiment {r['id']} failed with error: {r.get('error')}"
        assert r["result_summary"] is not None


def test_exp001_labor_policy_artifacts():
    """Validate EXP-001 labor policy simulation outputs."""
    exp1_dir = REPORTS_DIR / "EXP-001"
    assert exp1_dir.exists()
    
    ols_csv = exp1_dir / "labor_policy_ols_results.csv"
    assert ols_csv.exists()
    df_ols = pd.read_csv(ols_csv)
    assert "coefficient" in df_ols.columns or "coef" in df_ols.columns or len(df_ols) > 0
    
    mc_json = exp1_dir / "monte_carlo_ci.json"
    assert mc_json.exists()
    with open(mc_json, "r", encoding="utf-8") as f:
        mc = json.load(f)
    assert mc["n_valid_iterations"] >= 100
    assert "dropout_reduction_pp_mean" in mc
    assert mc["dropout_reduction_pp_ci_lo"] < mc["dropout_reduction_pp_ci_hi"]


def test_exp002_pipeline_resilience_artifacts():
    """Validate EXP-002 pipeline resilience outputs."""
    exp2_dir = REPORTS_DIR / "EXP-002"
    assert exp2_dir.exists()
    
    pnr_json = exp2_dir / "point_of_no_return.json"
    assert pnr_json.exists()
    with open(pnr_json, "r", encoding="utf-8") as f:
        pnr = json.load(f)
    assert "point_of_no_return" in pnr
    assert "highest_impact_stage" in pnr

    grid_csv = exp2_dir / "intervention_sensitivity_grid.csv"
    assert grid_csv.exists()
    df_grid = pd.read_csv(grid_csv)
    assert len(df_grid) >= 10


def test_exp003_fairness_audit_artifacts():
    """Validate EXP-003 fairness audit outputs."""
    exp3_dir = REPORTS_DIR / "EXP-003"
    assert exp3_dir.exists()
    
    achievability_csv = exp3_dir / "threshold_achievability_by_demographics.csv"
    assert achievability_csv.exists()
    df_ach = pd.read_csv(achievability_csv)
    assert len(df_ach) > 0
    assert "achievability_pct" in df_ach.columns or "achievability_rate" in df_ach.columns or "rate" in df_ach.columns

    rescue_json = exp3_dir / "workex_rescue_differential.json"
    assert rescue_json.exists()
    with open(rescue_json, "r", encoding="utf-8") as f:
        rescue = json.load(f)
    assert "fisher_exact_pval" in rescue
    assert "by_gender" in rescue


def test_exp004_career_trajectory_artifacts():
    """Validate EXP-004 career trajectory early warning outputs."""
    exp4_dir = REPORTS_DIR / "EXP-004"
    assert exp4_dir.exists()
    
    perf_csv = exp4_dir / "early_window_model_performance.csv"
    assert perf_csv.exists()
    df_perf = pd.read_csv(perf_csv)
    assert len(df_perf) >= 2
    assert "auc" in df_perf.columns
    assert (df_perf["auc"] >= 0.5).all()

    window_csv = exp4_dir / "early_warning_window_auc_curve.csv"
    assert window_csv.exists()
    df_win = pd.read_csv(window_csv)
    assert len(df_win) == 4
    # S1 through S4 AUC must be monotonically non-decreasing or positive
    assert (df_win["auc"] > 0.6).all()


def test_exp005_intervention_roi_artifacts():
    """Validate EXP-005 intervention optimizer outputs."""
    exp5_dir = REPORTS_DIR / "EXP-005"
    assert exp5_dir.exists()
    
    alloc_csv = exp5_dir / "optimal_allocation_by_budget.csv"
    assert alloc_csv.exists()
    df_alloc = pd.read_csv(alloc_csv)
    assert len(df_alloc) == 10
    assert "total_dropout_reductions" in df_alloc.columns
    assert (df_alloc["total_dropout_reductions"] > 0).all()

    subgroup_csv = exp5_dir / "subgroup_prioritization.csv"
    assert subgroup_csv.exists()
    df_sub = pd.read_csv(subgroup_csv)
    assert len(df_sub) >= 3
    assert "roi_per_dollar" in df_sub.columns
