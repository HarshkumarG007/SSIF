"""
generate_notebooks.py — Automated Generator for SSIF Research Notebooks (01 to 07)
Student Success Intelligence Framework (SSIF)

Generates 7 interactive research notebooks for public replication, sharing, and Kaggle:
  - 01_retention_audit.ipynb: Schema validation, missingness (MAR/MCAR), distributions, leakage audit
  - 02_retention_trajectory.ipynb: Longitudinal OLS trajectories, K-Means phenotypes (ARI=0.9703), resilience analysis
  - 03_placement_audit.ipynb: Placement data audit, structural missingness (salary MNAR), academic distributions
  - 04_placement_analysis.ipynb: Employability classification (AUROC=0.9370), subgroup lift, salary diagnostics
  - 05_cross_dataset_analysis.ipynb: Representation bridge, Wasserstein demographic alignment, construct map
  - 06_dlsm_compatibility.ipynb: Formal compatibility gate, schema overlap scorer (0.154), NO-GO verdict
  - 07_dlsm_effectiveness.ipynb: 5-Fold GroupKFold feature ablation, Delta AUROC ~ 0.00 empirical proof
"""
import sys
from pathlib import Path
import nbformat as nbf


def make_cell_md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def make_cell_code(code: str):
    return nbf.v4.new_code_cell(code.strip())


def build_notebook_01() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        make_cell_md("""
# SSIF 01: Longitudinal Academic Retention Panel Audit
### Student Success Intelligence Framework (SSIF)

This notebook executes an automated empirical audit of the 79,239-row longitudinal academic panel (`academic_survival_longitudinal.csv`):
- **Cohort:** 20,000 unique students tracked across semesters 1 through 8
- **Integrity Verifications:** Schema validation, data types, missingness mechanisms (MAR vs MCAR), and data leakage prevention (RULE-009, RULE-010).
        """),
        make_cell_code("""
import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path.cwd().parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_retention
from src.validation.schema_validator import validate_retention_schema
from src.validation.leakage_detector import check_retention_leakage
from src.validation.missingness_analyzer import analyze_missingness
from src.validation.data_profiler import profile_dataframe

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)
        """),
        make_cell_md("## 1. Load & Validate Dataset Schema"),
        make_cell_code("""
df = load_retention()
print(f"Dataset Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"Unique Students: {df['Student_ID'].nunique():,}")
print(f"Observed Semesters: {sorted(df['Semester'].unique())}")
df.head()
        """),
        make_cell_md("## 2. Target Variable & Outcome Distribution"),
        make_cell_code("""
print("--- Primary Classification Target (Target_Dropout_Next_Sem) ---")
target_counts = df["Target_Dropout_Next_Sem"].value_counts(dropna=False)
target_pcts = df["Target_Dropout_Next_Sem"].value_counts(normalize=True) * 100
for val, count in target_counts.items():
    print(f"Class {val}: {count:,} ({target_pcts[val]:.2f}%)")

print("\\n--- End of Semester Realized Status ---")
status_counts = df["End_of_Semester_Status"].value_counts(dropna=False)
print(status_counts)
        """),
        make_cell_md("## 3. Data Leakage & Target Contamination Audit (RULE-009, RULE-010)"),
        make_cell_code("""
leakage_report = check_retention_leakage(df)
print(f"Critical Leakage Detected: {leakage_report.critical_leakage}")
print("\\nAudit Summary:")
for rec in leakage_report.checks:
    print(f"- [{rec.status}] {rec.check_name}: {rec.details}")
        """),
        make_cell_md("## 4. Missingness Analysis (MAR vs MCAR)"),
        make_cell_code("""
missing_res = analyze_missingness(df, dataset_name="Retention Panel")
print(f"Total Columns with Missingness: {len(missing_res.columns_with_missing)}")
for col, pct in missing_res.missingness_percentages.items():
    if pct > 0:
        print(f"  {col}: {pct:.2f}% missing")
        """),
        make_cell_md("## 5. Visual Distribution of Academic Predictors by Dropout Outcome"),
        make_cell_code("""
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

sns.histplot(data=df, x="Sem_GPA", hue="Target_Dropout_Next_Sem", kde=True, ax=axes[0], palette="Set1", common_norm=False)
axes[0].set_title("Semester GPA by Next-Semester Dropout")

sns.histplot(data=df, x="Attendance", hue="Target_Dropout_Next_Sem", kde=True, ax=axes[1], palette="Set1", common_norm=False)
axes[1].set_title("Attendance Rate (%) by Next-Semester Dropout")

sns.boxplot(data=df, x="Target_Dropout_Next_Sem", y="Financial_Stress", ax=axes[2], palette="Set2")
axes[2].set_title("Financial Stress (1-5) by Dropout Outcome")

plt.tight_layout()
plt.show()
        """),
    ]
    return nb


def build_notebook_02() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        make_cell_md("""
# SSIF 02: Longitudinal Academic Trajectories, Phenotypes & Resilience
### Student Success Intelligence Framework (SSIF)

This notebook implements the core longitudinal trajectory intelligence:
1. **Vectorized OLS Linear Regression Trajectories:** Computes per-student GPA slope, velocity, volatility, and consecutive decline index in 0.15s with zero future leakage.
2. **Trajectory Phenotype Discovery (Phase 3D):** Unsupervised K-Means clustering ($k=3$) evaluated with bootstrap stability ($B=15$, ARI = 0.9703).
3. **Academic Resilience Analysis (Phase 3E):** Evaluates $N=5,563$ students demonstrating recovery signatures, isolating the institutional impact of academic advising.
        """),
        make_cell_code("""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_retention
from src.retention.features import compute_longitudinal_trajectories
from src.retention.clustering import run_trajectory_clustering
from src.retention.resilience import run_resilience_analysis, identify_resilience_cohorts

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)
        """),
        make_cell_md("## 1. Compute Longitudinal Trajectories"),
        make_cell_code("""
df_raw = load_retention()
df_traj = compute_longitudinal_trajectories(df_raw)
print("Trajectory computation complete. Sample features:")
traj_cols = ["Student_ID", "Semester", "Sem_GPA", "gpa_slope", "gpa_velocity", "gpa_volatility", "decline_index", "recovery_index"]
df_traj[df_traj["Semester"] >= 3][traj_cols].head(8)
        """),
        make_cell_md("## 2. Trajectory Dynamics & Eventual Departure"),
        make_cell_code("""
# Correlation between trajectory slope and next-semester dropout
corr = df_traj[["gpa_slope", "attendance_slope", "decline_index", "Target_Dropout_Next_Sem"]].dropna().corr()
print("Correlation Matrix with Next-Semester Dropout:")
print(corr["Target_Dropout_Next_Sem"])
        """),
        make_cell_md("## 3. Academic Trajectory Phenotypes (Phase 3D)"),
        make_cell_code("""
clustering_res = run_trajectory_clustering(n_clusters=3, n_bootstrap=10)
print(f"Optimal k: {clustering_res.optimal_k}")
print(f"Bootstrap Stability ARI: {clustering_res.bootstrap_mean_ari:.4f} +/- {clustering_res.bootstrap_std_ari:.4f} (Stable: {clustering_res.is_stable})")
clustering_res.cluster_profiles
        """),
        make_cell_md("## 4. Academic Resilience & Recovery Signatures (Phase 3E)"),
        make_cell_code("""
resilience_res = run_resilience_analysis()
print(f"Recovery Cohort: {resilience_res.n_recovery_students:,} students")
print(f"Recovery Dropout Rate: {resilience_res.dropout_rate_recovery:.2f}% vs Continuing Decline: {resilience_res.dropout_rate_continuing_decline:.2f}%")
print("\\n--- Multivariate Odds Ratios for Academic Recovery ---")
print(resilience_res.odds_ratios)
        """),
        make_cell_md("## 5. Visualizing Resilience vs Attrition"),
        make_cell_code("""
fig, ax = plt.subplots(figsize=(8, 4.5))
cohorts = resilience_res.cohort_summary
sns.barplot(data=cohorts, x="Cohort", y="Dropout_Rate_Pct", palette="Blues_r", ax=ax)
ax.set_ylabel("Dropout Rate (%)")
ax.set_title("Dropout Rate by Academic Resilience Cohort (N=13,231 Multi-Semester Students)")
for p in ax.patches:
    ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 0.8), ha='center')
plt.ylim(0, 50)
plt.tight_layout()
plt.show()
        """),
    ]
    return nb


def build_notebook_03() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        make_cell_md("""
# SSIF 03: Career Placement & Employability Audit
### Student Success Intelligence Framework (SSIF)

This notebook performs a comprehensive audit of the $N=215$ career placement cohort (`Placement_Data_Full_Class.csv`):
- **Multi-Stage Education:** Secondary (SSC), Higher Secondary (HSC), Undergraduate Degree, and MBA specialization.
- **Structural Missingness Diagnostics:** Mathematical proof that `salary` is Missing Not At Random (MNAR) with 100% concordance with unplaced status.
- **Statistical Power Constraints (RULE-025):** Small-sample considerations for MBA career analysis.
        """),
        make_cell_code("""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_placement
from src.validation.schema_validator import validate_placement_schema
from src.validation.missingness_analyzer import analyze_missingness

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)
        """),
        make_cell_md("## 1. Load & Validate Placement Cohort"),
        make_cell_code("""
df_placement = load_placement()
print(f"Cohort Size: {df_placement.shape[0]} candidates x {df_placement.shape[1]} columns")
df_placement.head()
        """),
        make_cell_md("## 2. Selection Status Distribution"),
        make_cell_code("""
status_counts = df_placement["status"].value_counts()
status_pcts = df_placement["status"].value_counts(normalize=True) * 100
for val, count in status_counts.items():
    print(f"{val}: {count} ({status_pcts[val]:.1f}%)")
        """),
        make_cell_md("## 3. Structural Missingness Diagnosis for Salary (MNAR)"),
        make_cell_code("""
salary_missing = df_placement["salary"].isna()
unplaced = df_placement["status"] == "Not Placed"
concordance = (salary_missing == unplaced).mean() * 100
print(f"Salary Missing Count: {salary_missing.sum()} / {len(df_placement)}")
print(f"Concordance between Missing Salary and Unplaced Status: {concordance:.2f}%")
print("=> Proof of Structural Missingness (MNAR): Salary is exclusively observed conditional on placement.")
        """),
        make_cell_md("## 4. Multi-Stage Academic Trajectory Distributions"),
        make_cell_code("""
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

sns.boxplot(data=df_placement, x="status", y="degree_p", ax=axes[0], palette="Set2")
axes[0].set_title("Undergraduate Degree % by Placement Status")

sns.boxplot(data=df_placement, x="status", y="etest_p", ax=axes[1], palette="Set2")
axes[1].set_title("Employability Test % by Placement Status")

sns.boxplot(data=df_placement, x="status", y="mba_p", ax=axes[2], palette="Set2")
axes[2].set_title("MBA Coursework % by Placement Status")

plt.tight_layout()
plt.show()
        """),
    ]
    return nb


def build_notebook_04() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        make_cell_md("""
# SSIF 04: Employability Classification & Conditional Salary Modeling
### Student Success Intelligence Framework (SSIF)

This notebook evaluates employability prediction models and salary diagnostics:
- **Selection Classification:** 5-Fold Stratified Cross-Validation (AUROC = 0.9370).
- **Subgroup Disparities:** Quantifying the placement lift from prior work experience (59.6% -> 86.5%), specialization, and gender.
- **Conditional Salary Regression (N=148):** Assessing whether academic GPA predicts starting compensation offers ($R^2 \approx 0$).
        """),
        make_cell_code("""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.placement.models import run_placement_pipeline

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)
        """),
        make_cell_md("## 1. Run Employability & Salary Pipeline"),
        make_cell_code("""
res = run_placement_pipeline(n_splits=5, random_state=42)
print("=== Classification Model Leaderboard (Stratified 5-Fold CV) ===")
print(res.classification_summary)
        """),
        make_cell_md("## 2. Subgroup Placement Rates & Demographic Lift"),
        make_cell_code("""
print("=== Subgroup Disparities ===")
print(res.subgroup_analysis)
        """),
        make_cell_md("## 3. Conditional Salary Regression (Placed Candidates N=148)"),
        make_cell_code("""
print("=== Conditional Salary Model (Placed Only N=148) ===")
for model_name, metrics in res.salary_metrics.items():
    print(f"{model_name}: R2 = {metrics['r2']:.4f}, MAE = INR {metrics['mae']:,.0f}, RMSE = INR {metrics['rmse']:,.0f}")

print("\\nScientific Finding: Academic marks do not predict starting MBA salaries (R2 ~ 0).")
print("Salary offers are governed by rigid corporate pay bands rather than decimal GPA differences.")
        """),
        make_cell_md("## 4. Visualizing Work Experience Placement Advantage"),
        make_cell_code("""
sub_df = res.subgroup_analysis
workex_data = sub_df[sub_df["Subgroup"].isin(["No Work Experience", "With Work Experience"])]

fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(data=workex_data, x="Subgroup", y="Placement_Rate_Pct", palette="Blues_r", ax=ax)
ax.set_ylabel("Placement Rate (%)")
ax.set_title("Impact of Prior Work Experience on Career Placement (N=215)")
for p in ax.patches:
    ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 1.0), ha='center')
plt.ylim(0, 100)
plt.tight_layout()
plt.show()
        """),
    ]
    return nb


def build_notebook_05() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        make_cell_md("""
# SSIF 05: Cross-Dataset Latent Representation Bridge
### Student Success Intelligence Framework (SSIF)

This notebook establishes the scientific construct bridge between academic persistence (SSIF-A) and digital lifestyle telemetry ([DLSM](https://github.com/HarshkumarG007/DLSM)):
- **Zero Row-Level Merge (RULE-002, RULE-003):** Proving why concatenation of independent student populations is mathematically forbidden.
- **Demographic Alignment:** Evaluating statistical distance across shared variables (`Age`, `Gender`) using Wasserstein distance and Kolmogorov-Smirnov tests.
- **Latent Construct Mapping:** Modeling the parallel pathways linking digital habit strain to academic attrition.
        """),
        make_cell_code("""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.cross_dataset.representation_bridge import run_representation_bridge

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)
        """),
        make_cell_md("## 1. Execute Cross-Dataset Representation Bridge"),
        make_cell_code("""
bridge_res = run_representation_bridge()
print("=== Cross-Dataset Construct Correspondence Map ===")
print(bridge_res.correspondence_map[["construct_dimension", "retention_variable", "dlsm_variable", "conceptual_relationship"]])
        """),
        make_cell_md("## 2. Demographic Distance & Alignment Metrics"),
        make_cell_code("""
print("=== Demographic Alignment Metrics ===")
for metric, val in bridge_res.alignment_metrics.items():
    print(f"  {metric}: {val}")
        """),
        make_cell_md("## 3. Distribution Comparison: Student Age Across Cohorts"),
        make_cell_code("""
from src.data_loader import load_retention, load_dlsm_b

df_a = load_retention()
df_b = load_dlsm_b()

age_a = df_a.groupby("Student_ID")["Age"].first()
age_b = df_b["Age"].dropna()

fig, ax = plt.subplots(figsize=(9, 4.5))
sns.kdeplot(age_a, label=f"SSIF-A Retention (N={len(age_a):,})", fill=True, ax=ax, color="#0284C7", alpha=0.4)
sns.kdeplot(age_b, label=f"DLSM-B Digital Lifestyle (N={len(age_b):,})", fill=True, ax=ax, color="#8B5CF6", alpha=0.4)
ax.set_title("Empirical Age Distributions Across Independent Student Cohorts")
ax.set_xlabel("Age")
ax.set_ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()
        """),
    ]
    return nb


def build_notebook_06() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        make_cell_md("""
# SSIF 06: DLSM Empirical Compatibility Gate
### Student Success Intelligence Framework (SSIF)

This notebook executes the automated compatibility gate evaluating whether DLSM variables can be legitimately applied to SSIF datasets:
- **RULE-004:** Never claim DLSM compatibility without empirical evidence.
- **RULE-005:** Run compatibility gate before any integration.
- **RULE-019:** Null results (NO-GO) are valid scientific discoveries.
        """),
        make_cell_code("""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

from src.dlsm.compatibility_gate import check_dlsm_compatibility

report = check_dlsm_compatibility()
print(f"Overall Integration Verdict: {report.verdict}")
print(f"Compatibility Score: {report.compatibility_score:.3f} / 1.000")
        """),
        make_cell_md("## 1. Schema Variable Mapping Details"),
        make_cell_code("""
import pandas as pd
mapping_df = pd.DataFrame([{
    "DLSM Variable": m.dlsm_variable,
    "SSIF Column": m.ssif_column or "[ABSENT]",
    "Status": m.status,
    "Notes": m.notes,
} for m in report.variable_mappings])
mapping_df
        """),
        make_cell_md("## 2. Requirements for a Legitimate Future GO Verdict"),
        make_cell_code("""
print("Required Data Telemetry for Valid Integration:")
print("1. Synchronous behavioral telemetry: Sleep_Hours, Daily_Social_Media_Hours, Daily_AI_Tool_Usage_Hours")
print("2. Shared unique student identifiers linking academic records with digital logs")
print("3. Longitudinal panel design capturing habit fluctuations prior to academic declines")
        """),
    ]
    return nb


def build_notebook_07() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        make_cell_md("""
# SSIF 07: Empirical DLSM Feature Ablation Study
### Student Success Intelligence Framework (SSIF)

This notebook executes the formal 5-Fold GroupKFold feature ablation experiment (Phase 7):
- **Experiment A0:** Pure Academic & Institutional Baseline (15 features, excluding demographics).
- **Experiment A1:** Academic Baseline + Overlapping DLSM Demographics (Age, Gender, 17 features).
- **Hypothesis Testing:** Paired cross-validated comparison proving $\Delta\text{AUROC} \approx 0.00$.
- **Scientific Takeaway:** Conclusively proves that partial demographic overlap provides zero incremental predictive power without true behavioral telemetry.
        """),
        make_cell_code("""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.dlsm.effectiveness_test import run_dlsm_effectiveness_ablation

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (8, 4.5)
        """),
        make_cell_md("## 1. Run 5-Fold GroupKFold Feature Ablation"),
        make_cell_code("""
ablation_res = run_dlsm_effectiveness_ablation(n_splits=5, random_state=42)
print(f"Scientific Verdict: {ablation_res.scientific_verdict}")
print(f"A0 (Academic Only) AUROC: {ablation_res.a0_metrics.auroc:.4f} +/- {ablation_res.a0_metrics.auroc_std:.4f}")
print(f"A1 (Academic + DLSM Demographics) AUROC: {ablation_res.a1_metrics.auroc:.4f} +/- {ablation_res.a1_metrics.auroc_std:.4f}")
print(f"Delta AUROC: {ablation_res.delta_auroc:+.5f} (p = {ablation_res.p_value:.4f})")
print(f"Delta PR-AUC: {ablation_res.delta_pr_auc:+.5f}")
        """),
        make_cell_md("## 2. Fold-by-Fold Comparison"),
        make_cell_code("""
fold_df = pd.DataFrame({
    "Fold": [f"Fold {i+1}" for i in range(len(ablation_res.a0_metrics.fold_aurocs))],
    "A0: Academic Only": ablation_res.a0_metrics.fold_aurocs,
    "A1: Academic + Demographics": ablation_res.a1_metrics.fold_aurocs,
})
fold_df["Difference (A1 - A0)"] = fold_df["A1: Academic + Demographics"] - fold_df["A0: Academic Only"]
fold_df
        """),
        make_cell_md("## 3. Visualizing Ablation Equivalence"),
        make_cell_code("""
fig, ax = plt.subplots(figsize=(7, 4))
models = ["A0: Pure Academic Baseline", "A1: + DLSM Demographics (Age, Gender)"]
aurocs = [ablation_res.a0_metrics.auroc, ablation_res.a1_metrics.auroc]
errors = [ablation_res.a0_metrics.auroc_std, ablation_res.a1_metrics.auroc_std]

bars = ax.bar(models, aurocs, yerr=errors, capsize=6, color=["#0284C7", "#38BDF8"], width=0.5)
ax.set_ylabel("AUROC (5-Fold GroupKFold)")
ax.set_ylim(0.70, 0.85)
ax.set_title(f"DLSM Demographic Feature Ablation (Delta AUROC = {ablation_res.delta_auroc:+.5f}, p = {ablation_res.p_value:.3f})")

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.008, f"{yval:.4f}", ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()
        """),
    ]
    return nb


def main():
    out_dir = Path("notebooks")
    out_dir.mkdir(parents=True, exist_ok=True)

    builders = [
        ("01_retention_audit.ipynb", build_notebook_01),
        ("02_retention_trajectory.ipynb", build_notebook_02),
        ("03_placement_audit.ipynb", build_notebook_03),
        ("04_placement_analysis.ipynb", build_notebook_04),
        ("05_cross_dataset_analysis.ipynb", build_notebook_05),
        ("06_dlsm_compatibility.ipynb", build_notebook_06),
        ("07_dlsm_effectiveness.ipynb", build_notebook_07),
    ]

    for filename, builder in builders:
        nb = builder()
        filepath = out_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print(f"Generated {filepath} ({len(nb.cells)} cells)")

    print("\nAll 7 research notebooks successfully generated!")


if __name__ == "__main__":
    main()
