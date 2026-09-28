"""
generate_audit_reports.py — Generates Phase 2 Data Audit Markdown Reports
Student Success Intelligence Framework (SSIF)

Produces:
  - reports/retention/audit_report.md
  - reports/placement/audit_report.md
  - reports/dlsm/compatibility_report.md
"""
from pathlib import Path
import pandas as pd

from src.data_loader import load_retention, load_placement
from src.validation.data_profiler import profile_retention, profile_placement
from src.validation.missingness_analyzer import analyze_missingness
from src.validation.leakage_detector import check_retention_leakage, check_placement_leakage
from src.dlsm.compatibility_gate import run_all_gates, generate_compatibility_markdown
from src.logger import get_module_logger

logger = get_module_logger("validation.generate_reports")


def generate_all_audit_reports():
    reports_dir = Path("reports")
    ret_dir = reports_dir / "retention"
    plc_dir = reports_dir / "placement"
    dlsm_dir = reports_dir / "dlsm"

    ret_dir.mkdir(parents=True, exist_ok=True)
    plc_dir.mkdir(parents=True, exist_ok=True)
    dlsm_dir.mkdir(parents=True, exist_ok=True)

    # 1. Retention Report
    logger.info("Generating Retention Audit Report...")
    df_r = load_retention()
    pr_r = profile_retention(df_r)
    mis_r = analyze_missingness(df_r, "SSIF-A: Retention Panel")
    clean_features_r = [c for c in df_r.columns if c not in {"End_of_Semester_Status", "Censored", "Target_Dropout_Next_Sem", "Student_ID"}]
    leak_r = check_retention_leakage(clean_features_r, "Target_Dropout_Next_Sem", df_r)

    dropout_rate = df_r["Target_Dropout_Next_Sem"].mean() * 100
    n_dropouts = int(df_r["Target_Dropout_Next_Sem"].sum())

    lines_r = [
        "# Academic Persistence (Retention) Dataset Audit Report",
        "**Dataset:** `academic_survival_longitudinal.csv` (SSIF-A)  ",
        "**Generated:** 2026-09-29  ",
        "**Evaluation:** Pre-modeling Data Quality & Governance Audit (RULE-001, RULE-009, RULE-014)  ",
        "",
        "## 1. Executive Summary",
        "```text",
        pr_r.summary_text(),
        "```",
        "",
        "## 2. Target Variable & Panel Dynamics",
        "- **Primary Target:** `Target_Dropout_Next_Sem` (0 = Persisted, 1 = Dropped out)",
        f"- **Overall Dropout Rate:** {dropout_rate:.2f}% ({n_dropouts:,} events across {len(df_r):,} observations)",
        f"- **Cohort Size:** {df_r['Student_ID'].nunique():,} unique students observed across semesters 1 to 8",
        "- **Realized Status (`End_of_Semester_Status`):**",
    ]
    for k, v in df_r["End_of_Semester_Status"].value_counts().items():
        lines_r.append(f"  - `{k}`: {v:,} ({v/len(df_r)*100:.2f}%)")

    lines_r += [
        "",
        "## 3. Missingness Mechanism Analysis",
        "```text",
        mis_r.summary(),
        "```",
        "",
        "## 4. Column Statistical Profiles",
        pr_r.to_dataframe().to_markdown(index=False),
        "",
        "## 5. Data Leakage & Feature Integrity Check",
        f"- **Leakage Audit Status:** {'PASSED' if not leak_r.has_critical_leakage else 'FAILED'}",
        f"- **Target Contamination Risk:** {len(leak_r.forbidden_features_found)} forbidden features detected",
        f"- **Clean Baseline Predictors ({len(clean_features_r)} features):** `{clean_features_r}`",
        "",
        "## 6. Actionable Preprocessing Recommendations",
        "1. **GroupKFold Strategy:** Group by `Student_ID` (k=5) to prevent multi-semester student data leakage (RULE-014).",
        "2. **Imputation:** Impute `Family_Income` and `LMS_Logins` inside the CV pipeline (fit on train fold only) using MedianImputer (RULE-007).",
        "3. **Class Imbalance:** Apply `class_weight='balanced'` or calibrated decision thresholds to accommodate the 8.73% dropout prevalence.",
        "4. **Trajectory Handling:** For 3,404 single-semester students, flag missing slope features with an indicator or use static fallbacks.",
    ]

    ret_report_path = ret_dir / "audit_report.md"
    ret_report_path.write_text("\n".join(lines_r), encoding="utf-8")
    logger.info("Saved %s", ret_report_path)

    # 2. Placement Report
    logger.info("Generating Placement Audit Report...")
    df_p = load_placement()
    pr_p = profile_placement(df_p)
    mis_p = analyze_missingness(df_p, "SSIF-B: Placement Cohort")
    clean_features_p = [c for c in df_p.columns if c not in {"sl_no", "status", "salary"}]
    leak_p = check_placement_leakage(clean_features_p, "status")

    lines_p = [
        "# Academic Placement (Employability) Dataset Audit Report",
        "**Dataset:** `Placement_Data_Full_Class.csv` (SSIF-B)  ",
        "**Generated:** 2026-09-29  ",
        "**Evaluation:** Pre-modeling Data Quality & Governance Audit (RULE-001, RULE-006, RULE-009)  ",
        "",
        "## 1. Executive Summary",
        "```text",
        pr_p.summary_text(),
        "```",
        "",
        "## 2. Target Distributions",
        "- **Classification Target (`status`):**",
    ]
    for k, v in df_p["status"].value_counts().items():
        lines_p.append(f"  - `{k}`: {v:,} ({v/len(df_p)*100:.2f}%)")

    placed_count = int(df_p["salary"].dropna().count())
    median_sal = float(df_p["salary"].median())
    mean_sal = float(df_p["salary"].mean())
    missing_sal = int(df_p["salary"].isna().sum())

    lines_p += [
        "- **Regression Target (`salary`):**",
        f"  - Observed for {placed_count} placed students",
        f"  - Median salary: INR {median_sal:,.0f} (Mean: INR {mean_sal:,.0f})",
        f"  - Structurally unobserved for {missing_sal} unplaced students",
        "",
        "## 3. Missingness Mechanism Analysis",
        "```text",
        mis_p.summary(),
        "```",
        "",
        "## 4. Column Statistical Profiles",
        pr_p.to_dataframe().to_markdown(index=False),
        "",
        "## 5. Data Leakage & Feature Integrity Check",
        f"- **Leakage Audit Status:** {'PASSED' if not leak_p.has_critical_leakage else 'FAILED'}",
        "- **Salary as Feature for Placement:** Strictly forbidden (salary only exists after placement).",
        f"- **Clean Baseline Predictors ({len(clean_features_p)} features):** `{clean_features_p}`",
        "",
        "## 6. Actionable Preprocessing Recommendations",
        "1. **Sample Size Awareness:** N=215 is an exploratory sample. Use repeated StratifiedKFold (k=5, 10 repeats) and regularization to prevent overfitting.",
        "2. **Dual Model Architecture:** Model 1 = Binary Classifier (`Placed` vs `Not Placed`); Model 2 = Salary Regressor trained exclusively on placed candidates (`N=148`).",
        "3. **Categorical Encoding:** One-hot encode MBA specialization, degree stream, and work experience.",
    ]

    plc_report_path = plc_dir / "audit_report.md"
    plc_report_path.write_text("\n".join(lines_p), encoding="utf-8")
    logger.info("Saved %s", plc_report_path)

    # 3. DLSM Compatibility Report
    logger.info("Generating DLSM Compatibility Report...")
    gates = run_all_gates(df_r, df_p)
    dlsm_md = generate_compatibility_markdown(gates)
    dlsm_report_path = dlsm_dir / "compatibility_report.md"
    dlsm_report_path.write_text(dlsm_md, encoding="utf-8")
    logger.info("Saved %s", dlsm_report_path)

    print("All Phase 2 reports successfully generated!")


if __name__ == "__main__":
    generate_all_audit_reports()
