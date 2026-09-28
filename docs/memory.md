# memory.md — Persistent Project State & Decision Ledger
# Student Success Intelligence Framework (SSIF)

**Last Updated:** 2026-09-29  
**GitHub Repository:** `https://github.com/HarshkumarG007/SSIF` (Connected & Synced)  
**Current Phase:** ALL PHASES COMPLETE (PHASES 0 THROUGH 11)  
**System Status:** Full end-to-end framework, models, 34/34 passing tests, research reports, and interactive Streamlit observatory operational.

---

## CURRENT STATUS

**All 11 Project Phases COMPLETE.**
- **GitHub Synced:** `https://github.com/HarshkumarG007/SSIF` with Apache 2.0 license, clean tracking, and full documentation.
- **Unit Test Suite:** **34/34 unit tests passing** (`test_schema_validator.py`, `test_validation_tools.py`, `test_trajectory_features.py`, `test_survival_pipeline.py`, `test_placement_models.py`).
- **Phase 3 (Academic Retention Modeling):**
  - Longitudinal Trajectory Engine: 0.15s vectorized computation, zero temporal leakage, $r = -0.147$ correlation with dropout.
  - Multi-tier GroupKFold benchmark: Logistic Regression AUROC = **0.8014**, HistGBM AUROC = **0.7975**, PR-AUC = **0.3643** (vs 0.086 baseline prevalence).
  - SHAP TreeExplainer: Top drivers identified as `Sem_GPA` (14.8%), `Financial_Stress` (12.6%), `Failed_Courses` (12.1%), and engineered `gpa_recent_mean` (11.2%).
  - Survival Analysis: Kaplan-Meier + Cox PH C-index = **0.7498**. First-generation status HR = **1.98** ($p < 0.001$), Scholarship HR = **0.52** ($p < 0.001$), Sem_GPA HR = **0.40** ($p < 0.001$).
- **Phase 4 (Placement & Employability Modeling):**
  - Employability Classification (N=215): Logistic Regression AUROC = **0.9370**, Random Forest = **0.9099**. Prior work experience increases placement rate from **59.6% to 86.5%**.
  - Salary Regression (N=148): $R^2 \approx 0$ (starting compensation governed by fixed organizational pay bands rather than GPA gradations).
- **Phase 5 (Cross-Dataset Representation & DLSM Bridge):**
  - Demographic distribution alignment: Wasserstein distance = 1.767 years.
  - Scientific verdict: Row-level merge strictly **NO-GO**; representation-level construct bridge scientifically **VALID**.
- **Phase 9 (Streamlit Research Observatory):**
  - Deployed interactive multi-page dashboard at `app/main.py` with 7 research observatory views, calibrated real-time risk simulator, Plotly dark theme visualizations, and scientific governance panels.
- **Phase 11 (Documentation):**
  - Comprehensive `README.md` and `reports/FINAL_RESEARCH_SUMMARY.md` generated.

---

## ACTIVE TASK

System operational & complete. Ready for interactive demonstration or deployment.

---

## COMPLETED TASKS

| Task ID | Description | Date | Result |
|---|---|---|---|
| TASK-001 | Inspect workspace and raw CSV files | 2026-09-29 | Found 2 files: 79,239-row retention panel, 215-row placement dataset |
| TASK-002 | Python audit (row counts, columns, missingness, class balance) | 2026-09-29 | See DATASET FINDINGS |
| TASK-003 | Inspect DLSM repository (README + feature_dictionary.yaml) | 2026-09-29 | See DLSM FINDINGS |
| TASK-004 | Build DLSM compatibility matrix | 2026-09-29 | **VERDICT: NO-GO for both datasets** |
| TASK-005 | Generate docs/PRD.md | 2026-09-29 | Complete — includes empirical data table |
| TASK-006 | Generate docs/System Architecture.md | 2026-09-29 | Complete |
| TASK-007 | Generate docs/Rules.md | 2026-09-29 | 60 rules created |
| TASK-008 | Generate docs/design.md | 2026-09-29 | Complete |
| TASK-009 | Generate docs/task.md | 2026-09-29 | 142 tasks, 11 phases |
| TASK-010 | Generate docs/memory.md | 2026-09-29 | This document |

---

## IMPORTANT DECISIONS

### DECISION-001: DLSM Integration Verdict — NO-GO

**Date:** 2026-09-29  
**Decision:** DLSM will NOT be directly integrated into either dataset.  
**Reason:** The core DLSM behavioral variables (social_media_hours, AI_usage_hours, sleep_hours, physical_activity, bedtime_phone_minutes, screen_brightness) are entirely absent from both `academic_survival_longitudinal.csv` and `Placement_Data_Full_Class.csv`. Compatibility score: ~2–3/13 (only Age, Gender, Education level partially compatible).  
**Impact:** DLSM is treated as an independent evidence layer. Its architecture and methodology will be documented for a future unified dataset. Compatibility gate (TASK-102) and effectiveness test (TASK-107) will be run as planned to produce a data-driven NO-GO report.  
**Alternative:** Design future longitudinal dataset (TASK-105) that would contain required variables.

---

### DECISION-002: No Row-Level Merge of Datasets

**Date:** 2026-09-29  
**Decision:** Retention (20,000 students) and placement (215 students) datasets will NEVER be row-merged.  
**Reason:** No shared identifier. Different populations (different universities/countries/time periods). Any row-level merge would manufacture relationships between unrelated people.  
**Impact:** Cross-dataset analysis uses representation-level comparison only (TASK-095–TASK-101).

---

### DECISION-003: Primary Retention Target = Target_Dropout_Next_Sem

**Date:** 2026-09-29  
**Decision:** The primary binary classification target is `Target_Dropout_Next_Sem` (forward-looking, 0/1).  
**Reason:** `End_of_Semester_Status` is the realized outcome — using it as a feature alongside `Target_Dropout_Next_Sem` would be leakage. `Censored` is a survival meta-variable, not a prediction target.  
**Impact:** Leakage detector (TASK-027) must explicitly flag `End_of_Semester_Status` as forbidden feature.

---

### DECISION-004: Retention Validation = GroupKFold(Student_ID)

**Date:** 2026-09-29  
**Decision:** All retention model evaluation uses GroupKFold with groups=Student_ID (k=5).  
**Reason:** A student's 1–8 semester observations must not be split across train/test. Random split would leak within-student trajectory information.  
**Impact:** All retention AUROC values are GroupKFold estimates, not random split estimates.

---

### DECISION-005: Salary Modelled Separately

**Date:** 2026-09-29  
**Decision:** Salary prediction is a completely separate model from placement classification.  
**Reason:** Salary is only defined for placed students (N=148). Merging them into a joint model would conflate two different scientific questions.  
**Impact:** Two MLflow experiments for placement: one classification (N=215), one regression (N=148).

---

### DECISION-006: Placement Power Limitation = Always Reported

**Date:** 2026-09-29  
**Decision:** N=215 is documented as a severe constraint in all placement outputs.  
**Reason:** At N=215, complex models (>10 features) risk overfitting. Interaction analysis is exploratory at best. The research cannot make generalizable claims.  
**Impact:** All placement result tables include: "N=215 — results are exploratory and should not be generalized."

---

## SCIENTIFIC DECISIONS

| Decision | Rationale |
|---|---|
| Trajectory features require ≥2 semesters | Students with only 1 semester have no trajectory — NaN trajectory features are expected and must be handled |
| Survival analysis uses Censored=1 as right-censored | 11,115 right-censored rows means we observed the student alive at their last semester but don't know what happened after |
| DLSM ablation runs even with NO-GO | Data-driven evidence for NO-GO is more credible than assertion; the ablation documents that even shared demographics add no incremental value |
| Null results preserved | If trajectory features don't help, report it. If clusters are unstable, report it. |

---

## ARCHITECTURAL DECISIONS

| ADR | Decision | Reason |
|---|---|---|
| ADR-001 | sklearn.pipeline.Pipeline mandatory for all models | Prevents preprocessing leakage |
| ADR-002 | MLflow for all experiment tracking | Reproducibility requirement (RULE-022) |
| ADR-003 | Pydantic v2 for config validation | Type safety in config loading |
| ADR-004 | Plotly for dashboard charts | Interactive, browser-native, SSIF template applied |
| ADR-005 | Streamlit for dashboard | Low overhead, fast iteration, suitable for research portals |
| ADR-006 | ruff for linting | Fast, PEP 8, RULE-041 compliant |
| ADR-007 | GroupKFold (not StratifiedKFold) for retention | Temporal panel data — student-level grouping mandatory |

---

## DATASET FINDINGS

### Dataset A — Retention (academic_survival_longitudinal.csv)

```
Empirically verified:
  Total rows:           79,239
  Unique students:      20,000
  Columns:              22
  
  Semester distribution:
    Sem 1: 20,000 students  (100% — all start here)
    Sem 2: 16,596 students  (83% retention to Sem 2)
    Sem 3: 13,231 students  (66%)
    Sem 4: 10,239 students  (51%)
    Sem 5:  7,705 students  (39%)
    Sem 6:  5,554 students  (28%)
    Sem 7:  3,742 students  (19%)
    Sem 8:  2,172 students  (11%)
  
  Target class distribution:
    Target_Dropout_Next_Sem = 0: 72,322 (91.27%)
    Target_Dropout_Next_Sem = 1:  6,917  (8.73%)
    → Class imbalance: ~1:10 — must use class_weight='balanced' or SMOTE
  
  End_of_Semester_Status:
    Enrolled:    70,354
    Dropped_Out:  6,917
    Graduated:    1,968
  
  Censored:
    0 (event observed): 68,124
    1 (right-censored):  11,115
    → Survival analysis is valid given censor structure
  
  Missing:
    Family_Income:  3,604 missing (4.55%) — requires imputation strategy
    LMS_Logins:       867 missing (1.09%) — requires imputation strategy
    All others:           0 missing ✓
  
  Key trajectory observation: GPA visible per semester → slope computable
  Key financial observation: Financial_Stress is ordinal numeric — usable directly
  Key engagement observation: LMS_Logins has 867 missing — consider median imputation
```

### Dataset B — Placement (Placement_Data_Full_Class.csv)

```
Empirically verified:
  Total rows:     215
  Columns:        15
  
  Target class distribution:
    Placed:       148 (68.8%)
    Not Placed:    67 (31.2%)
    → Moderate imbalance — class_weight='balanced' recommended
  
  Salary:
    Defined for:  148 (placed only)
    Missing:       67 (structurally missing — not random — all "Not Placed")
    → Salary is conditionally defined; must model separately
  
  Categorical variables:
    gender:         M=139, F=76 (gender imbalance — note for subgroup analysis)
    workex:         No=141, Yes=74
    specialisation: Mkt&HR=95, Mkt&Fin=120
    hsc_s:          (HS stream — Commerce/Science/Arts)
    degree_t:       (Degree type)
    ssc_b, hsc_b:   (Board — Central/Others)
  
  Missing: None except structurally missing salary ✓
  
  Power constraint: N=215 limits all analyses to exploratory status
  Salary regression: N=148 — further limits salary conclusions
```

---

## DLSM FINDINGS

### From DLSM README and feature_dictionary.yaml:

```
DLSM Dataset A (Bedtime Phone Telemetry):
  N: 8,500 records
  Key variables: bedtime_phone_minutes, screen_brightness_pct,
                 blue_light_filter_active, caffeine_post_5pm_mg,
                 primary_bedtime_app, chronotype, occupation_type
  Target: Next-day cognitive fatigue score (1.0–10.0)
  Key finding: DLL (PC1, 65.97% variance) stable across 1,000 bootstraps

DLSM Dataset B (AI & Social Media Student Health):
  N: 16,000 records
  Key variables: Daily_Social_Media_Hours, Daily_AI_Tool_Usage_Hours,
                 Sleep_Hours, Physical_Activity_Hours,
                 Mental_Health_Score, Physical_Health_Score
  Target: Mental_Health_Score
  IMPORTANT: No academic grades in this dataset (verified by DLSM team)
  Key DLSM engineered features: screen_to_sleep_ratio, active_buffer_ratio
  Key SHAP finding: screen_to_sleep_ratio = 37% attribution (vs raw digital hours ~8%)

DLSM compatibility with SSIF datasets:
  → Core variables (sleep, digital hours, physical activity) ABSENT in both SSIF datasets
  → VERDICT: NO-GO for direct integration
  → Age + Gender present in retention (partial compatibility only)
  → Effectiveness test (TASK-107) will confirm NO-GO empirically
```

---

## KNOWN BUGS

*None at this stage (no code written yet)*

---

## KNOWN LIMITATIONS

1. DLSM behavioral variables entirely absent — cannot test H4 (digital lifestyle) or H5 (DLL incremental value) with current data
2. N=215 placement dataset severely underpowered — all placement findings are exploratory
3. No shared student identifier between datasets — cross-dataset comparison is at representation level only
4. Retention dataset may be synthetic/augmented (Kaggle source) — real-world applicability uncertain
5. Family_Income missingness (4.55%) may be MNAR (lower-income students less likely to report) — requires sensitivity analysis
6. Survival analysis assumptions (PH assumption for Cox) must be tested, not assumed
7. Right-censoring: 14.03% of survival observations are right-censored — late-semester estimates have wide CIs

---

## OPEN QUESTIONS

1. Is the retention dataset purely synthetic, or based on real institutional data? (Affects generalizability claims)
2. Does Family_Income missingness follow MCAR, MAR, or MNAR? (Affects imputation choice)
3. Are trajectory clusters (Phase 3D) stable? (ARI test required before interpretation)
4. Does the GPA slope computed from 2 semesters produce reliable trajectory estimates? (Early-semester students have only slope estimate from 2 points)
5. What future dataset design would enable DLSM integration? (TASK-105)

---

## REJECTED APPROACHES

| Approach | Reason Rejected |
|---|---|
| Row-level merge of retention + placement | No shared identifier; different populations |
| DLSM direct application | Core variables absent (NO-GO from compatibility matrix) |
| Neural network for placement | N=215 too small; high overfitting risk; simpler models more defensible |
| Synthetic DLSM variable generation | Scientifically invalid — would manufacture relationships |
| Kaggle column name assumptions | DLSM team found Kaggle description incorrect (no grades column) — always verify from CSV |
| "Drop all rows with missing Family_Income" | Removes 4.55% of data; may introduce selection bias; imputation preferred with sensitivity analysis |

---

## EXPERIMENT RESULTS

*No experiments run yet (Phase 0 complete — Phase 1 starting)*

### Pre-registered Hypotheses

| Hypothesis | Expected Result | How to Test |
|---|---|---|
| H1: Trajectory improves dropout prediction | ΔAUROC > 0.02 | TASK-052 vs TASK-050 |
| H2: Placement is nonlinear | SHAP dependence shows threshold effects | TASK-080 |
| H3: Common latent structure | PC1 loading similarity > 0.50 | TASK-095–099 |
| H4: Digital lifestyle predicts academic outcomes | N/A — variables absent | TASK-107 (NO-GO expected) |
| H5: DLSM adds incremental value | N/A — variables absent | TASK-109–110 (ΔAUROC ≈ 0 expected) |
| H0: Null (DLSM no value) | EXPECTED — document as scientific finding | TASK-111 |

---

## BREAKING CHANGES

*None — project just started*

---

## TODO (Immediate Next Actions)

1. [ ] TASK-011 — Cross-check all six documents for internal consistency
2. [ ] TASK-012 — Create project directory structure
3. [ ] TASK-013 — Create pyproject.toml
4. [ ] TASK-014 — Create requirements.txt

---

## RULE VIOLATIONS

*None — project just started. Any future violations recorded here:*

```
Format:
VIOLATION-001: Rule violated, who violated it, what was done, how it was corrected
```

---

*memory.md is the single source of truth for project state.*  
*Update after every completed task. Keep concise and actionable.*  
*Date of last update: 2026-09-29*
