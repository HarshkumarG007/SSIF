# PRD.md — Product Requirements Document
# Student Success Intelligence Framework (SSIF)

**Version:** 1.0-DRAFT  
**Date:** 2026-09-29  
**Status:** Active — governing document

---

## 0. Document Hierarchy

```
Scientific Validity  ← SUPREME
       ↓
PRD.md              ← YOU ARE HERE
       ↓
System Architecture.md → Rules.md → task.md → design.md → memory.md
```

---

## 1. Product Overview

**Student Success Intelligence Framework (SSIF)** is a research-grade computational platform investigating relationships between:
1. **Academic Persistence / Retention** — longitudinal trajectories leading to dropout or graduation
2. **Academic Employability / Placement** — factors predicting placement and compensation
3. **Digital Lifestyle / Sleep / Wellbeing** — behavioral load as modelled by DLSM (independent system)

Core principle: **Let the data decide whether the connection exists. Never manufacture a relationship the data does not support.**

---

## 2. Empirically Verified Dataset Summary

### Dataset A — Academic Persistence (Retention)

| Property | Empirical Value |
|---|---|
| File | `academic_survival_longitudinal.csv` |
| Total rows | **79,239** |
| Unique students | **20,000** |
| Columns | 22 |
| Temporal structure | Semester panel (Sem 1–8), **genuine longitudinal** |
| Semester 1 count | 20,000 students |
| Semester 8 count | 2,172 students (real attrition) |
| Dropout events | **6,917** rows (8.73% of rows) |
| Graduated events | 1,968 rows |
| Censored (right) | **11,115** rows — enables survival analysis |
| Missing — Family_Income | 3,604 rows (4.55%) |
| Missing — LMS_Logins | 867 rows (1.09%) |
| All other missingness | 0 |

**Confirmed columns:**
`Student_ID, Age, Gender, First_Generation, Family_Income, Household_Size, Housing_Status, Scholarship, Tuition_Base, Semester, Course_Load, Work_Hours, Emergency_Expense, Sem_GPA, Attendance, LMS_Logins, Advising_Visits, Failed_Courses, Financial_Stress, Target_Dropout_Next_Sem, End_of_Semester_Status, Censored`

**Target candidates:**
- `Target_Dropout_Next_Sem` — binary prospective label (0/1) — **PRIMARY CLASSIFICATION TARGET**
- `End_of_Semester_Status` — realized multi-class status (Enrolled / Dropped_Out / Graduated)
- `Censored` — survival meta-variable (0=observed event, 1=censored)

**Critical leakage rules:**
- `End_of_Semester_Status` must NEVER be used as a feature when `Target_Dropout_Next_Sem` is the target
- `Censored` must NEVER be used as a predictor

---

### Dataset B — Academic Placement (Employability)

| Property | Empirical Value |
|---|---|
| File | `Placement_Data_Full_Class.csv` |
| Total rows | **215** |
| Columns | 15 |
| Temporal structure | **Cross-sectional** — single observation per student |
| Placed | **148** (68.8%) |
| Not Placed | **67** (31.2%) |
| Salary missing | **67** (all "Not Placed" — structurally missing, not random) |
| All other missingness | 0 |

**Confirmed columns:**
`sl_no, gender, ssc_p (Secondary %), ssc_b (Secondary Board), hsc_p (Higher Secondary %), hsc_b (HS Board), hsc_s (HS Stream), degree_p (Degree %), degree_t (Degree Type), workex (Work Experience Yes/No), etest_p (Employability Test %), specialisation (Mkt&HR / Mkt&Fin), mba_p (MBA %), status (Placed/Not Placed), salary`

**Critical size constraint:** N=215 is very small. Limits model complexity. No deep learning. No >5-cluster analysis. Salary regression: N=148 only.

---

### DLSM Compatibility Matrix (Empirical)

| DLSM Variable | Retention Dataset | Placement Dataset | Verdict |
|---|---|---|---|
| Age | ✅ Direct | ⚠️ Absent (academic stage proxy only) | Partial |
| Gender | ✅ Direct | ✅ Direct | Yes |
| Education level | ✅ Semester proxy | ✅ Degree type proxy | Partial |
| Social media hours | ❌ ABSENT | ❌ ABSENT | NO |
| AI usage hours | ❌ ABSENT | ❌ ABSENT | NO |
| Sleep hours | ❌ ABSENT | ❌ ABSENT | NO |
| Physical activity | ❌ ABSENT | ❌ ABSENT | NO |
| Bedtime phone mins | ❌ ABSENT | ❌ ABSENT | NO |
| Screen brightness | ❌ ABSENT | ❌ ABSENT | NO |
| Mental health score | ❌ ABSENT | ❌ ABSENT | NO |
| Financial stress | ✅ Direct (proxy) | ❌ ABSENT | Retention only |

**DLSM Integration Verdict:**
- **Retention → DLSM Direct: NO-GO** (core behavioral variables absent)
- **Placement → DLSM Direct: NO-GO** (core behavioral variables absent)
- **Architecture:** DLSM remains an independent evidence layer. Cross-dataset synthesis uses representation-level comparison only — not row merging.

---

## 3. Research Opportunity

The genuine opportunity is NOT to merge datasets. It is to investigate:

**Research Question:** Do academic trajectory features from the retention panel share a comparable latent structure with digital-lifestyle patterns that DLSM identifies — and do the same academic characteristics that predict persistence also predict placement?

This is a **cross-dataset latent structure** investigation, not a merge.

```
DLSM (independent cohorts)
    ↓ [no bridge — behavioral vars absent]
    
Academic Retention (20,000 longitudinal students)
    ↓ [representation-level comparison only — different populations]
    
Academic Placement (215 cross-sectional students)
```

---

## 4. Users

### Primary Users
| Persona | Primary Need |
|---|---|
| Educational Data Mining Researcher | Reproducible experiments, statistical rigour |
| University Policy Analyst | Early-warning signals, intervention evidence |
| Data Scientist (learning) | Annotated pipelines, explainable models |
| DLSM Extension Researcher | Compatibility gate, integration adapter |

### Excluded Use Cases (Non-Goals)
- ❌ Automated high-stakes individual decisions
- ❌ Predictive surveillance without consent
- ❌ Causal claims from observational data
- ❌ Row-level merge of retention and placement

---

## 5. Goals

### Research Goals
| ID | Goal | Success Condition |
|---|---|---|
| RG-01 | Trajectory features improve dropout prediction | ΔAUROC > 0.02, statistically significant |
| RG-02 | Stable trajectory clusters identified | Bootstrap ARI > 0.80 |
| RG-03 | Survival analysis of persistence | Calibrated KM + Cox models |
| RG-04 | Placement + salary modelled separately | Two models, two evaluations |
| RG-05 | Nonlinear placement effects documented | GAM or SHAP dependence plots |
| RG-06 | DLSM compatibility matrix completed | Every DLSM feature classified |
| RG-07 | Cross-dataset representation compared | Latent-space analysis without row merge |
| RG-08 | Null results preserved and reported | No hiding of negative findings |

### Product Goals
| ID | Goal | Success Condition |
|---|---|---|
| PG-01 | 9-page Streamlit dashboard | All pages functional |
| PG-02 | MLflow experiment tracking | 100% of runs logged |
| PG-03 | Automated data audit | Runs before any model |
| PG-04 | SHAP for all tree models | Summary + local plots |
| PG-05 | Limitations page | Page 9 exists and is accurate |

---

## 6. Minimum Viable Research Product (MVRP)

1. Both datasets fully audited
2. Retention baseline model (GroupKFold by student)
3. Retention trajectory features tested
4. Placement baseline + salary model
5. Cross-dataset representation comparison (no merge)
6. DLSM compatibility matrix with explicit verdict
7. SHAP for both main models
8. Scientific limitations documented
9. Streamlit dashboard — all 9 pages

---

## 7. Functional Requirements

| FR-ID | Requirement | Priority |
|---|---|---|
| FR-01 | Automated dataset schema audit | P0 |
| FR-02 | Leakage detection before model training | P0 |
| FR-03 | Retention baseline with GroupKFold | P0 |
| FR-04 | Trajectory feature engineering (slope, velocity, volatility) | P1 |
| FR-05 | Survival analysis (KM + Cox) | P1 |
| FR-06 | Academic resilience cluster analysis | P2 |
| FR-07 | Placement classification (logistic → RF → XGBoost) | P0 |
| FR-08 | Conditional salary regression (N=148) | P1 |
| FR-09 | Employability phenotype clustering (N=215 constraint) | P2 |
| FR-10 | Nonlinear interaction analysis for placement | P2 |
| FR-11 | DLSM compatibility gate | P1 |
| FR-12 | Cross-dataset representation comparison | P1 |
| FR-13 | SHAP global + local for tree models | P1 |
| FR-14 | MLflow experiment tracking | P1 |
| FR-15 | Streamlit 9-page dashboard | P1 |
| FR-16 | Calibration curves + Brier score | P1 |
| FR-17 | Subgroup analysis (gender, first_generation) | P2 |
| FR-18 | Confidence intervals for major metrics | P1 |
| FR-19 | Scientific limitations page | P0 |

---

## 8. Non-Functional Requirements

| NFR-ID | Category | Requirement |
|---|---|---|
| NFR-01 | Reproducibility | All experiments seeded (random_state=42), logged |
| NFR-02 | Performance | Streamlit pages < 3s with @st.cache_data |
| NFR-03 | Security | No raw student IDs in dashboard; no untrusted pickle |
| NFR-04 | Maintainability | Each module independently importable |
| NFR-05 | Explainability | Every classifier has SHAP summary plot |
| NFR-06 | Accessibility | Colorblind-safe palettes; information not colour-only |
| NFR-07 | Scientific honesty | Null results retained and reported |
| NFR-08 | Ethics | No automated risk scoring exposed without IRB |

---

## 9. Pre-declared Scientific Limitations

These must appear in dashboard Page 9 and in every research report:

1. **No shared students.** Retention and placement are different populations; no causal pathway between them can be established from these data.
2. **DLSM incompatibility.** Core DLSM variables (sleep, digital hours, screen time) are entirely absent from both SSIF datasets; direct DLSM application is invalid.
3. **Placement dataset size (N=215).** Severely limits generalizability. Models trained on this dataset should not be applied to other populations without revalidation.
4. **Observational design.** All associations are observational; no causal claims are warranted.
5. **Missing data.** Family_Income (4.55%) and LMS_Logins (1.09%) missing in retention dataset; sensitivity analysis required.
6. **Right-censoring.** 11,115 survival observations are right-censored; survival models must account for this explicitly.
7. **Synthetic data risk.** Kaggle datasets may be partially synthetic; population-level patterns may not reflect real institutions.

---

*Document owner: Lead Researcher*  
*Update triggers: After each phase completion; after any schema change discovery*
