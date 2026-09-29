# SSIF Experiments Directory
## Student Success Intelligence Framework — Research Experiments

> **Discipline Team:** Educational Policy · Labor Economics · Data Science · Institutional Advising  
> **Governance:** All experiments comply with RULE-002 (no fabricated IDs), RULE-003 (no row merge), RULE-009 (zero leakage)

---

## Experiment Registry

| ID | Name | Domain | Key Question | Script |
|----|------|--------|-------------|--------|
| EXP-001 | Labor-Policy Intervention Simulation | Educational Policy + Labor Econ | What is the net ROI of converting students from unstructured survival labor to institutional work-study? | `exp_001_labor_policy_simulation.py` |
| EXP-002 | Pipeline Resilience Stress Test | Data Science + Institutional Research | How sensitive are placement rates to retention intervention failures at each college lifecycle stage? | `exp_002_pipeline_resilience_stress_test.py` |
| EXP-003 | Socio-Economic Fairness Audit | Educational Equity + Policy | Does the 65% degree GPA hiring threshold disproportionately exclude low-income, first-generation, or female students? | `exp_003_fairness_audit.py` |
| EXP-004 | Career Trajectory Forecasting | Predictive Analytics | Can early (Semester 1–2) academic signals predict eventual placement tier and salary band 4+ years later? | `exp_004_career_trajectory_forecast.py` |
| EXP-005 | Intervention ROI Optimizer | Operations Research + Policy | Given limited institutional resources, which intervention portfolio (advising, scholarships, work-study) maximizes expected retained & placed students per dollar? | `exp_005_intervention_roi_optimizer.py` |

---

## Running Experiments

```bash
# Run a single experiment
python experiments/exp_001_labor_policy_simulation.py

# Run all experiments (orchestrated)
python experiments/run_all_experiments.py

# Run with verbose output
python experiments/run_all_experiments.py --verbose
```

All experiment outputs (CSV metrics, JSON summaries, PNG figures) are written to:
- `reports/experiments/EXP-00X/`

---

## Empirical Benchmark Results (Master Run: 5/5 SUCCESS)

Executed and verified in **60.9s** via `python experiments/run_all_experiments.py`:

| ID | Experiment | Primary Result & Significance | Key Metric | Output Directory |
|---|---|---|---|---|
| **EXP-001** | Labor-Policy Intervention Simulation | Converting survival labor to work-study yields **+0.077 GPA lift** and prevents **2.97 pp dropout risk** (N=200 MC bootstrap iterations) | 95% CI: [2.64, 3.29] pp | `reports/experiments/EXP-001/` |
| **EXP-002** | Pipeline Resilience Stress Test | Stage 1 Early intervention is the **highest systemic multiplier** (+81.6 graduates per 1,000 students vs +62.6 for late stage) | 18 perturbation scenarios | `reports/experiments/EXP-002/` |
| **EXP-003** | Socio-Economic Fairness Audit | Degree GPA 65% gate disproportionately excludes Q1 low-income students; identified **N=992 Qualified-But-Excluded** resilient candidates | Fisher's Exact OR=1.03 ($p=1.00$) | `reports/experiments/EXP-003/` |
| **EXP-004** | Career Trajectory Forecasting | S1–S2 signals alone predict 4-year success with **AUC = 0.7469** (XGBoost); AUC expands monotonically to **0.8387** by S4 | GroupKFold (k=5) | `reports/experiments/EXP-004/` |
| **EXP-005** | Intervention ROI Optimizer | **Advising Boost delivers the highest entry ROI** (0.0533 reductions/dollar) up to $100K budgets; larger budgets optimally blend with micro-scholarships | HiGHS Simplex LP | `reports/experiments/EXP-005/` |

---

## Output Artifacts Catalog

```
reports/experiments/
├── master_results.json           # Consolidated machine-readable benchmark metrics
├── MASTER_EXPERIMENT_SUMMARY.md  # Comprehensive executive research summary
├── EXP-001/
│   ├── labor_policy_ols_results.csv
│   ├── monte_carlo_ci.json
│   ├── policy_roi_summary.json
│   ├── counterfactual_gpa_shift.csv
│   └── EXP001_summary.md
├── EXP-002/
│   ├── lifecycle_attrition_baseline.csv
│   ├── intervention_sensitivity_grid.csv
│   ├── point_of_no_return.json
│   ├── compounding_failure_matrix.csv
│   └── EXP002_summary.md
├── EXP-003/
│   ├── threshold_achievability_by_demographics.csv
│   ├── placement_fairness_metrics.csv
│   ├── workex_rescue_differential.json
│   ├── qualified_excluded_profiles.csv
│   └── EXP003_summary.md
├── EXP-004/
│   ├── early_window_model_performance.csv
│   ├── early_warning_window_auc_curve.csv
│   ├── career_readiness_score_distribution.csv
│   ├── crs_by_demographics.csv
│   ├── shap_top10_early_features.csv
│   └── EXP004_summary.md
└── EXP-005/
    ├── optimal_allocation_by_budget.csv
    ├── subgroup_prioritization.csv
    ├── sensitivity_analysis.csv
    ├── intervention_parameters.json
    └── EXP005_summary.md
```

---

## Interactive What-If Policy Lab

All experimental results are integrated into the live Streamlit Observatory (`app/main.py`):
- **Pareto Efficiency Frontier:** Interactive Plotly curve displaying expected dropout reductions and marginal ROI across budget tiers ($10K–$500K).
- **Intervention Sensitivity Heatmap:** 2D matrix displaying student retention yields across stages and efficacy tiers.
- **Demographic Equity Explorer:** Interactive horizontal bar charts comparing threshold achievability across income and demographic brackets.
- **AUC Stabilization Diagnostic:** Visualizing information gain across multi-semester observation windows.

---

## Scientific Governance

- **EXP-001**: Uses OLS + Monte Carlo to avoid confounding work_hours with placement outcome
- **EXP-002**: Uses synthetic scenario perturbation — does NOT retrain on held-out data
- **EXP-003**: Uses group-stratified fairness metrics (Equalized Odds, Demographic Parity)
- **EXP-004**: Strictly enforces temporal causality — semester 1–2 features only as predictors
- **EXP-005**: Uses linear programming (LP) with budget constraints — no speculative parameters

---

## 🙏 Primary Datasets, Attribution & Thanksgiving

These multi-disciplinary experiments are powered by datasets open-sourced by the global academic research community:

1. **Student Retention & Academic Performance Panel (79,239 records):** Curated by **Razan Ihab Abdellatif** on Kaggle.  
   🔗 [Student Retention and Academic Performance Data](https://www.kaggle.com/datasets/razanihababdellatif/student-retention-and-academic-performance-data)
2. **MBA Campus Placement & Employability Dataset (215 candidates):** Curated by **Amey Thakur** ([@ameythakur20](https://www.kaggle.com/ameythakur20)) on Kaggle.  
   🔗 [Campus Recruitment (Placement Data Full Class)](https://www.kaggle.com/datasets/ameythakur20/placement-data)
3. **Sleep Debt & Screen Time Telemetry (8,500 records):** Curated by **Samar Talwar** on Kaggle.  
   🔗 [Sleep Debt and Screen Time / Late Night Phone Habits](https://www.kaggle.com/datasets/samartalwar/sleep-debt-and-screen-time-late-night-phone-habits)
4. **AI Tool Usage, Social Media & Student Health (16,000 records):** Curated by **Sri Syra** ([@srisyra02](https://www.kaggle.com/srisyra02)) on Kaggle.  
   🔗 [AI and Social Media Impact: Student Health & Grades](https://www.kaggle.com/datasets/srisyra02/ai-and-social-media-impact-student-health-and-grades)
5. **Sister Framework:** [Digital Lifestyle Spillover Modeling (DLSM)](https://github.com/HarshkumarG007/DLSM)

> 📢 **Ethical Data Citation:** Please visit the original Kaggle repositories above, give the curators a well-deserved star/upvote, and download raw CSV datasets directly from their profiles.

---

*Generated by SSIF Research Team · Apache 2.0*


