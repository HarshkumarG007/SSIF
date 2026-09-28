# SSIF: Final Empirical Research Summary & Scientific Report

# Student Success Intelligence Framework

**Date:** 2026-09-29  
**Repository:** [HarshkumarG007/SSIF](https://github.com/HarshkumarG007/SSIF)  
**License:** Apache License 2.0  
**Authors:** Computational Education & Machine Learning Research Team

---

## 1. Executive Summary

Higher education institutions face systemic challenges in identifying students at risk of premature departure early enough to provide effective institutional support, as well as optimizing post-graduation career placement pathways.

The **Student Success Intelligence Framework (SSIF)** investigates these challenges through an open-source, scientifically audited computational machine learning framework across two primary educational datasets and evaluates integration feasibility with external digital lifestyle telemetry ([DLSM](https://github.com/HarshkumarG007/DLSM)):

1. **Academic Persistence Panel (SSIF-A):** 79,239 longitudinal student-semester records across 20,000 distinct students with up to 8 semesters of follow-up.
2. **Career Placement Cohort (SSIF-B):** 215 MBA candidates with multi-stage secondary, undergraduate, and postgraduate academic profiles, employability test scores, and starting salary offers.
3. **Digital Lifestyle Spillover Modeling (DLSM):** 8,500 sleep telemetry records (DLSM-A) and 16,000 student digital lifestyle and mental health records (DLSM-B).

---

## 2. Key Empirical Findings

### 2.1 Academic Retention & Longitudinal Trajectory Modeling

- **Zero-Leakage Trajectory Engine:** Vectorized OLS linear regression computes per-student GPA slope, GPA velocity ($\Delta\text{GPA}/\Delta\text{Semester}$), volatility ($\sigma$), and cumulative consecutive GPA decline indices in **0.15 seconds** across 79,239 records.
- **Strict Temporal Causality (RULE-009):** All trajectory features for student $i$ at semester $t$ are strictly restricted to history $\le t$, validated by temporal causality tests.
- **Predictive Performance (5-Fold GroupKFold, `groups=Student_ID`):**
  - **Tier 1+ (Logistic Regression + Trajectories):** AUROC = **0.8014 ± 0.0053**, PR-AUC = **0.3643** (vs 0.0860 baseline prevalence).
  - **Tier 4+ (HistGradientBoosting + Trajectories):** AUROC = **0.7975 ± 0.0049**, PR-AUC = **0.3521**, Brier Score = **0.1692**, ECE = **0.2857**.
  - **Tier 3 (Random Forest):** AUROC = **0.7874 ± 0.0049**, Brier Score = **0.1177**, ECE = **0.2002**.
- **Trajectory Signal:** GPA slope demonstrates a statistically significant negative correlation with dropout ($r = -0.147$). Students with declining GPA trajectory ($\text{slope} < -0.2$) experience a **16.03% departure rate**, compared to **7.08%** for students with stable or improving trajectory ($\text{slope} > 0.0$).

### 2.2 Time-to-Dropout Survival Analysis

- **Model Discrimination:** Cox Proportional Hazards regression achieves **Harrell's Concordance Index $C = 0.7498$** (Partial AIC: 123,618.43, Log-Likelihood Ratio: 4,906.58, $p < 0.001$).
- **Epidemiological Risk Multipliers:**
  - **First-Generation Status (HR = 1.98, 95% CI: [1.89, 2.08], $p < 0.001$):** First-generation college students experience **nearly double (1.98×)** the instantaneous hazard of dropping out at any given semester.
  - **Scholarship Protection (HR = 0.52, 95% CI: [0.49, 0.55], $p < 0.001$):** Institutional scholarship funding **reduces dropout hazard by 48%**, acting as the strongest protective intervention.
  - **Academic GPA (HR = 0.40, 95% CI: [0.38, 0.42], $p < 0.001$):** Every 1.0 unit increase in semester GPA **cuts the hazard of departure by 60%**.
  - **Financial Stress (HR = 1.23, 95% CI: [1.21, 1.24], $p < 0.001$):** Each increment on the 1–5 financial stress index multiplies dropout hazard by **1.23×**.
- **Kaplan-Meier Cumulative Persistence:**
  - Semester 1: **93.3%** cumulative persistence
  - Semester 4: **69.6%** cumulative persistence
  - Semester 8: **46.3%** cumulative persistence
  - Log-Rank test confirms extreme divergence between first-generation and continuing-generation students ($\chi^2 = 481.45, p < 10^{-100}$).

### 2.3 Explainable AI & SHAP Risk Attributions

TreeExplainer attributions (Random Forest on 2,000 background samples) identify the following top predictors of student attrition:

1. `Sem_GPA` (14.8% relative importance)
2. `Financial_Stress` (12.6%)
3. `Failed_Courses` (12.1%)
4. `gpa_recent_mean` (11.2% — engineered longitudinal trajectory)
5. `First_Generation` (9.7%)
6. `Scholarship` (6.7%)
7. `cumulative_failed_courses` (5.0%)

### 2.4 Employability & Placement Modeling (N=215)

- **Selection Status Classification (Stratified 5-Fold CV):**
  - Logistic Regression AUROC = **0.9370 ± 0.0303**, Accuracy = **85.58%**, F1 = **0.8942**.
  - Random Forest AUROC = **0.9099 ± 0.0537**, Accuracy = **86.98%**, F1 = **0.9108**.
- **Subgroup Disparities:**
  - **Work Experience Lift:** Candidates with prior work experience achieved an **86.5% placement rate**, vs **59.6%** for those without.
  - **MBA Specialization:** Marketing & Finance candidates achieved **79.2% placement**, vs **55.8%** for Marketing & HR candidates.
  - **Gender Disparity:** Male candidates achieved **71.9% placement** (Median salary: INR 270,000) vs **63.2%** for female candidates (Median salary: INR 250,000).
- **Salary Regression (N=148 Placed Candidates):**
  - $R^2 \approx -0.06$ to $-0.17$. Starting corporate compensation is bounded by rigid corporate salary bands rather than fine gradations in student GPAs.

---

## 3. DLSM Cross-Study Integration Verdict

| Evaluation Dimension        | Metric / Evidence                                          | Scientific Verdict                     |
| --------------------------- | ---------------------------------------------------------- | -------------------------------------- |
| **Direct Variable Overlap** | Score = 0.154 (only Age, Gender shared)                    | ❌ **NO-GO for Row Merge (RULE-003)**  |
| **Demographic Alignment**   | Wasserstein distance = 1.767 years, KS D = 0.3048          | ✅ **VALID for Representation Bridge** |
| **Construct Parallelism**   | Digital Stress (DLSM) ↔ Academic & Financial Strain (SSIF) | ✅ **VALID Theoretical Framework**     |

**Conclusion:** Neither SSIF dataset can be directly scored by DLSM because low-level behavioral telemetry (`Daily_Social_Media_Hours`, `Daily_AI_Tool_Usage_Hours`, `Sleep_Hours`, `Physical_Activity_Hours`) is entirely absent from academic records. However, representation-level construct mapping reveals that digital lifestyle strain and academic/financial strain represent parallel vulnerability vectors in higher education cohorts.

---

## 4. Scientific Limitations & Governance

1. **Observational Inference:** Statistical associations do not establish causal mechanisms. Model outputs are risk indicators, not deterministic predictions.
2. **Sample Size Constraints:** The placement dataset (N=215) reflects an institutional cohort. Wide confidence intervals require cautious generalization.
3. **Missingness Assumptions:** `Family_Income` and `LMS_Logins` exhibit MAR properties and are imputed exclusively within training cross-validation folds.
4. **Target Contamination Prevention:** Future realized outcomes (`End_of_Semester_Status`, `Censored`) are permanently forbidden as predictors.

---

## 5. Blueprint for Future Research

To empirically test causal digital lifestyle spillover on academic attrition, institutions should launch prospective panels that simultaneously capture:

- **Longitudinal Academic Panel:** GPA, credit completion, advising visits, scholarship allocations.
- **Passive Digital Telemetry:** Daily screen time, late-night phone minutes, LMS login frequency.
- **Sleep & Wellness Scores:** Sleep duration, fatigue indices, subjective wellness evaluations.
