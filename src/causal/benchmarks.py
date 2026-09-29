"""
benchmarks.py — Empirical Causal Policy Benchmarks on SSIF Longitudinal Cohorts
Student Success Intelligence Framework (SSIF)

Executes Double Machine Learning evaluations for:
  1. Institutional Scholarship Treatment Effect on GPA and Dropout Mitigation.
  2. Excessive Work Hour Relief Treatment Effect on Academic Recovery.
  3. Heterogeneous Treatment Effects (CATE) comparing First-Generation vs Continuing-Gen students.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.causal.double_ml import (
    CATEEstimator,
    CausalEstimate,
    DoubleMLAIPW,
    DoubleMLPLR,
    HeterogeneousCausalEstimate,
)
from src.data_loader import load_retention
from src.logger import get_module_logger

logger = get_module_logger("causal.benchmarks")


def prepare_causal_dataset(df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Prepares confounder matrix X, treatment D (Scholarship), work hours D_work,
    and outcomes Y_dropout, Y_gpa with GroupKFold clusters (Student_ID).
    """
    if df is None:
        df = load_retention()

    # Drop missing rows or impute causally
    clean_df = df.copy()
    clean_df["Family_Income"] = clean_df["Family_Income"].fillna(clean_df["Family_Income"].median())
    clean_df["LMS_Logins"] = clean_df["LMS_Logins"].fillna(0.0)

    confounder_cols = [
        "Age",
        "Household_Size",
        "Family_Income",
        "Course_Load",
        "Attendance",
        "LMS_Logins",
        "Advising_Visits",
        "Failed_Courses",
        "Financial_Stress",
        "First_Generation",
    ]

    # One-hot encode Housing_Status if present
    if "Housing_Status" in clean_df.columns:
        dummies = pd.get_dummies(clean_df["Housing_Status"], prefix="Housing", drop_first=True, dtype=float)
        X = pd.concat([clean_df[confounder_cols], dummies], axis=1)
    else:
        X = clean_df[confounder_cols].copy()

    d_scholarship = clean_df["Scholarship"].astype(int)
    # Work reduction treatment: 1 if Work_Hours <= 15, 0 if Work_Hours > 20
    d_work_relief = (clean_df["Work_Hours"] <= 15).astype(int)

    y_dropout = clean_df["Target_Dropout_Next_Sem"].astype(float)
    y_gpa = clean_df["Sem_GPA"].astype(float)
    groups = clean_df["Student_ID"]

    return X, d_scholarship, d_work_relief, y_dropout, y_gpa, groups


def run_scholarship_causal_benchmark(sample_size: int = 15000, random_state: int = 42) -> dict[str, Any]:
    """
    Runs DoubleML-PLR and DoubleML-AIPW for Scholarship -> Dropout & GPA.
    """
    logger.info("=== Running Double Machine Learning Causal Benchmark ===")
    X, d_schol, d_work, y_drop, y_gpa, groups = prepare_causal_dataset()

    # Sample if requested for rapid iteration while preserving full groups
    if sample_size and sample_size < len(X):
        unique_students = groups.drop_duplicates().sample(n=sample_size // 4, random_state=random_state)
        sample_mask = groups.isin(unique_students)
        X = X[sample_mask].reset_index(drop=True)
        d_schol = d_schol[sample_mask].reset_index(drop=True)
        y_drop = y_drop[sample_mask].reset_index(drop=True)
        y_gpa = y_gpa[sample_mask].reset_index(drop=True)
        groups = groups[sample_mask].reset_index(drop=True)

    # 1. DoubleML-PLR for GPA Lift
    plr = DoubleMLPLR(n_splits=5, random_state=random_state)
    gpa_estimate = plr.fit(X, d_schol, y_gpa, groups=groups)
    logger.info("[Causal DML] Scholarship -> GPA: %s", gpa_estimate.summary())

    # 2. DoubleML-AIPW for Dropout Mitigation
    aipw = DoubleMLAIPW(n_splits=5, random_state=random_state)
    dropout_estimate = aipw.fit(X, d_schol, y_drop, groups=groups)
    logger.info("[Causal DML] Scholarship -> Dropout Mitigation: %s", dropout_estimate.summary())

    # 3. Heterogeneous Treatment Effects (CATE)
    cate_engine = CATEEstimator(dml_plr=plr)
    cate_results = cate_engine.fit_cate(
        X, d_schol, y_drop,
        modifiers=["First_Generation", "Financial_Stress"],
        groups=groups,
    )

    # 4. Save reports to disk
    out_dir = Path(__file__).resolve().parents[2] / "reports" / "causal"
    out_dir.mkdir(parents=True, exist_ok=True)

    report_json = {
        "n_samples": len(X),
        "scholarship_gpa_ate": {
            "ate": gpa_estimate.ate,
            "se": gpa_estimate.se,
            "ci_95": [gpa_estimate.ci_lower, gpa_estimate.ci_upper],
            "p_value": gpa_estimate.p_value,
            "e_value": gpa_estimate.e_value,
        },
        "scholarship_dropout_ate": {
            "ate": dropout_estimate.ate,
            "se": dropout_estimate.se,
            "ci_95": [dropout_estimate.ci_lower, dropout_estimate.ci_upper],
            "p_value": dropout_estimate.p_value,
            "e_value": dropout_estimate.e_value,
        },
        "cate_interaction_terms": cate_results.interaction_terms,
        "cate_interaction_p_values": cate_results.interaction_p_values,
    }

    import json
    (out_dir / "double_ml_policy_report.json").write_text(json.dumps(report_json, indent=2), encoding="utf-8")

    md_report = f"""# Double Machine Learning Causal Policy Report
**Student Success Intelligence Framework (SSIF)**  
**Methodology:** Neyman-Orthogonal Double ML (Chernozhukov et al., 2018) + AIPW Doubly-Robust Estimator  
**Cross-Fitting:** 5-Fold GroupKFold (grouped by Student_ID to prevent temporal contamination)  
**Sample Analyzed:** N={len(X):,} student-semesters  

---

## 1. Average Treatment Effects (ATE)

| Policy Intervention | Target Outcome | Estimator | Causal ATE | 95% Confidence Interval | p-value | E-Value (Confounder Robustness) |
|---|---|---|---|---|---|---|
| Institutional Scholarship | Sem_GPA Lift | DoubleML-PLR | **{gpa_estimate.ate:+.4f}** | [{gpa_estimate.ci_lower:+.4f}, {gpa_estimate.ci_upper:+.4f}] | {gpa_estimate.p_value:.4e} | {gpa_estimate.e_value:.2f} |
| Institutional Scholarship | Dropout Mitigation | DoubleML-AIPW | **{dropout_estimate.ate:+.4f}** | [{dropout_estimate.ci_lower:+.4f}, {dropout_estimate.ci_upper:+.4f}] | {dropout_estimate.p_value:.4e} | {dropout_estimate.e_value:.2f} |

---

## 2. Heterogeneous Treatment Effects (CATE)

| Effect Modifier | Interaction Coefficient | p-value | Interpretation |
|---|---|---|---|
| **First_Generation** | `{cate_results.interaction_terms.get('First_Generation', 0.0):+.4f}` | {cate_results.interaction_p_values.get('First_Generation', 1.0):.4e} | Differential impact of scholarship on first-generation students. |
| **Financial_Stress** | `{cate_results.interaction_terms.get('Financial_Stress', 0.0):+.4f}` | {cate_results.interaction_p_values.get('Financial_Stress', 1.0):.4e} | Differential impact of scholarship per unit of baseline financial distress. |

---

*Generated by `src.causal.benchmarks` • SSIF Causal Machine Learning Engine*
"""
    (out_dir / "double_ml_policy_report.md").write_text(md_report, encoding="utf-8")
    logger.info("[Causal DML] Saved causal reports to %s", out_dir)

    return {
        "scholarship_gpa_ate": gpa_estimate,
        "scholarship_dropout_ate": dropout_estimate,
        "cate_results": cate_results,
        "n_samples": len(X),
    }


if __name__ == "__main__":
    results = run_scholarship_causal_benchmark(sample_size=12000)
    print("\n" + "=" * 60)
    print("DOUBLE MACHINE LEARNING CAUSAL BENCHMARK RESULTS")
    print("=" * 60)
    print(results["scholarship_gpa_ate"].summary())
    print(results["scholarship_dropout_ate"].summary())
    print("\nHeterogeneous Treatment Effects (CATE Interaction Terms):")
    for mod, coeff in results["cate_results"].interaction_terms.items():
        pval = results["cate_results"].interaction_p_values[mod]
        print(f"  • {mod}: {coeff:+.4f} (p={pval:.4e})")

