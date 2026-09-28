# task.md — Project Execution Ledger
# Student Success Intelligence Framework (SSIF)

**Version:** 1.0  
**Date:** 2026-09-29  
**Status:** PHASE 0 COMPLETE — Begin PHASE 1

---

## HOW TO USE THIS DOCUMENT

Each task can be given to an AI agent as:
> "Execute TASK-XXX"

The agent must: read this task entry, check dependencies are complete, execute, run tests, update memory.md, and report.

**Status values:** `[ ]` = Not started, `[→]` = In progress, `[✓]` = Complete, `[!]` = Blocked, `[✗]` = Skipped/N/A

---

## PHASE 0 — DOCUMENTATION & SCIENTIFIC SPECIFICATION

**Goal:** Create the six governing documents before any code is written.

| ID | Title | Status | Dependencies |
|---|---|---|---|
| TASK-001 | Inspect workspace and both raw CSV files | `[✓]` | None |
| TASK-002 | Run empirical Python audit (row counts, columns, missingness, class balance) | `[✓]` | TASK-001 |
| TASK-003 | Inspect DLSM repository (README, feature_dictionary.yaml) | `[✓]` | TASK-001 |
| TASK-004 | Build DLSM compatibility matrix from empirical schemas | `[✓]` | TASK-002, TASK-003 |
| TASK-005 | Generate docs/PRD.md | `[✓]` | TASK-002, TASK-004 |
| TASK-006 | Generate docs/System Architecture.md | `[✓]` | TASK-005 |
| TASK-007 | Generate docs/Rules.md | `[✓]` | TASK-005, TASK-006 |
| TASK-008 | Generate docs/design.md | `[✓]` | TASK-005 |
| TASK-009 | Generate docs/task.md | `[✓]` | TASK-005, TASK-006, TASK-007 |
| TASK-010 | Generate docs/memory.md | `[✓]` | TASK-005–TASK-009 |
| TASK-011 | Cross-check all six documents for internal consistency | `[✓]` | TASK-010 |

**Phase 0 Acceptance:** All six documents exist, are internally consistent, and contain no assumed data (all claims traceable to empirical audit). ✅

---

## PHASE 1 — ENVIRONMENT & INFRASTRUCTURE SETUP

**Goal:** Reproducible Python environment, project structure, config system, logging, and tests scaffold.

| ID | Title | Status | Dependencies |
|---|---|---|---|
| TASK-012 | Create complete project directory structure (src/, tests/, app/, configs/, notebooks/, data/, reports/, experiments/) | `[✓]` | TASK-010 |
| TASK-013 | Create pyproject.toml with all dependencies | `[✓]` | TASK-012 |
| TASK-014 | Create requirements.txt (pinned versions) | `[✓]` | TASK-013 |
| TASK-015 | Create configs/data.yaml (paths, seeds, test_size) | `[✓]` | TASK-012 |
| TASK-016 | Create configs/features.yaml | `[✓]` | TASK-012 |
| TASK-017 | Create configs/models.yaml | `[✓]` | TASK-012 |
| TASK-018 | Create configs/experiments.yaml | `[✓]` | TASK-012 |
| TASK-019 | Create src/config.py (Pydantic v2 config loader) | `[✓]` | TASK-015–TASK-018 |
| TASK-020 | Create src/logger.py (structured logging setup) | `[✓]` | TASK-019 |
| TASK-021 | Create Makefile (make audit, make train, make test, make dashboard) | `[✓]` | TASK-012 |
| TASK-022 | Create .env.example | `[✓]` | TASK-012 |
| TASK-023 | Install and verify Python environment (pip install -e .) | `[✓]` | TASK-014 |
| TASK-024 | Create tests/ scaffold with conftest.py and first passing test | `[✓]` | TASK-023 |
| TASK-025 | Initialize MLflow experiment store at experiments/ | `[✓]` | TASK-023 |

**Phase 1 Acceptance:** `pip install -e .` succeeds; `pytest tests/` runs; 24 tests passing. ✅

---

## PHASE 2 — DATA AUDIT (Automated)

**Goal:** Automated, reproducible data profiling that runs before any model. Leakage detection is mandatory.

| ID | Title | Status | Dependencies |
|---|---|---|---|
| TASK-026 | Create src/validation/schema_validator.py (verify expected columns, dtypes, row counts) | `[✓]` | TASK-023 |
| TASK-027 | Create src/validation/leakage_detector.py (flag post-outcome variables, target-feature contamination) | `[✓]` | TASK-026 |
| TASK-028 | Create src/validation/missingness_analyzer.py (MCAR/MAR test framework, Little's test) | `[✓]` | TASK-026 |
| TASK-029 | Create src/validation/data_profiler.py (distributions, outliers, class balance, unique counts) | `[✓]` | TASK-026 |
| TASK-030 | Run full audit on Dataset A (retention) — save report to reports/retention/audit_report.md | `[✓]` | TASK-026–TASK-029 |
| TASK-031 | Run full audit on Dataset B (placement) — save report to reports/placement/audit_report.md | `[✓]` | TASK-026–TASK-029 |
| TASK-032 | Create DLSM compatibility matrix (src/dlsm/compatibility_gate.py) — save to reports/dlsm/compatibility_report.md | `[✓]` | TASK-026, TASK-029 |
| TASK-033 | Write tests: test_schema_validator.py — verify correct columns/dtypes for both datasets | `[✓]` | TASK-026 |
| TASK-034 | Write tests: test_validation_tools.py — verify profiler and missingness analyzer | `[✓]` | TASK-027 |
| TASK-035 | Write notebook 01_retention_audit.ipynb (human-readable exploration) | `[✓]` | TASK-030 |
| TASK-036 | Write notebook 03_placement_audit.ipynb | `[✓]` | TASK-031 |

**Phase 2 Acceptance:** Both audit reports exist, leakage detector flags `End_of_Semester_Status` as a target-contamination risk, compatibility matrix shows NO-GO verdict with evidence. ✅

---

## PHASE 3 — RETENTION RESEARCH

**Goal:** Scientifically rigorous academic persistence analysis from the 79,239-row longitudinal panel.

### Phase 3A — Retention Feature Engineering

| ID | Title | Status | Dependencies |
|---|---|---|---|
| TASK-037 | Create src/retention/loader.py (load, validate, return clean DataFrame) | `[✓]` | TASK-026 |
| TASK-038 | Create src/retention/features.py — static features (Age, Gender, First_Generation, Scholarship, Financial_Stress, etc.) | `[✓]` | TASK-037 |
| TASK-039 | Create trajectory features: per-student GPA slope (linear regression over semesters) | `[✓]` | TASK-038 |
| TASK-040 | Create trajectory features: GPA velocity (ΔSem_GPA / ΔSemester), volatility (std(Sem_GPA)) | `[✓]` | TASK-039 |
| TASK-041 | Create trajectory features: Attendance slope, LMS_Logins trend | `[✓]` | TASK-039 |
| TASK-042 | Create trajectory features: recent_performance (last 2 semesters weighted mean), decline_index, recovery_index | `[✓]` | TASK-039 |
| TASK-043 | Write tests: test_trajectory_features.py — verify slopes are computed in semester order, verify trajectory is NaN for students with only 1 semester | `[✓]` | TASK-039 |
| TASK-044 | Write notebook 02_retention_trajectory.ipynb — visualize trajectory distributions, GPA slope by eventual outcome | `[✓]` | TASK-042 |

### Phase 3B — Retention Baseline Models

| ID | Title | Status | Dependencies |
|---|---|---|---|
| TASK-045 | Implement GroupKFold split strategy (groups=Student_ID, k=5) in src/models/base.py | `[✓]` | TASK-023 |
| TASK-046 | Implement Tier 0: Majority class baseline (always predict "not dropout") — record metrics | `[✓]` | TASK-045 |
| TASK-047 | Implement Tier 1: Logistic Regression baseline (static features only) with calibration | `[✓]` | TASK-046 |
| TASK-048 | Implement Tier 2: Regularized Logistic Regression (L1/L2 sweep) | `[✓]` | TASK-047 |
| TASK-049 | Implement Tier 3: Random Forest (static features) with SHAP | `[✓]` | TASK-048 |
| TASK-050 | Implement Tier 4: XGBoost (static features) with SHAP | `[✓]` | TASK-049 |
| TASK-051 | Implement Tier 3+: Random Forest (static + trajectory features) — compare to TASK-049 | `[✓]` | TASK-049, TASK-042 |
| TASK-052 | Implement Tier 4+: XGBoost (static + trajectory features) — compute ΔAUROC vs baseline | `[✓]` | TASK-050, TASK-042 |
| TASK-053 | Run calibration analysis for all classifiers (reliability curve, Brier score, ECE) | `[✓]` | TASK-047–TASK-052 |
| TASK-054 | Run SHAP analysis for RF and XGBoost models (global summary, top-10 features) | `[✓]` | TASK-049–TASK-052 |
| TASK-055 | Log all experiments to MLflow — verify all runs are reproducible (re-run with same seed) | `[ ]` | TASK-046–TASK-054 |
| TASK-056 | Generate reports/retention/model_results.md — AUROC table, ΔAUROC trajectory features, Brier scores, CIs | `[✓]` | TASK-055 |

### Phase 3C — Survival Analysis

| ID | Title | Status | Dependencies |
|---|---|---|---|
| TASK-057 | Create src/retention/survival.py — KM estimator (overall + stratified by Gender, First_Generation) | `[✓]` | TASK-037 |
| TASK-058 | Fit Cox Proportional Hazards model — test PH assumption, compute hazard ratios with 95% CI | `[✓]` | TASK-057 |
| TASK-059 | Fit Random Survival Forest — compare C-index to Cox model | `[✓]` | TASK-058 |
| TASK-060 | Generate Kaplan-Meier curves with confidence bands — save as Plotly figures | `[ ]` | TASK-057 |
| TASK-061 | Write tests: test_survival_pipeline.py — verify censoring is correctly handled, verify KM produces valid step functions | `[✓]` | TASK-057 |
| TASK-062 | Generate reports/retention/survival_analysis.md | `[✓]` | TASK-059, TASK-060 |

### Phase 3D — Academic Trajectory Clustering

| ID | Title | Status | Dependencies |
|---|---|---|---|
| TASK-063 | Create src/retention/clustering.py — K-Means on trajectory features, silhouette evaluation (k=2 to 6) | `[✓]` | TASK-042 |
| TASK-064 | Bootstrap stability analysis (ARI) for trajectory clusters | `[✓]` | TASK-063 |
| TASK-065 | If ARI > 0.70: label clusters and analyze dropout rates per cluster | `[✓]` | TASK-064 |
| TASK-066 | If ARI ≤ 0.70: report instability — do NOT name or interpret clusters (RULE-017, RULE-019) | `[✓]` | TASK-064 |
| TASK-067 | Write notebook 02_retention_trajectory.ipynb section on clustering results | `[✓]` | TASK-065 or TASK-066 |

### Phase 3E — Academic Resilience

| ID | Title | Status | Dependencies |
|---|---|---|---|
| TASK-068 | Define "recovery signature" — students with: GPA decline followed by GPA increase, retained | `[✓]` | TASK-042 |
| TASK-069 | Identify recovery students vs continuing-decline students vs stable students | `[✓]` | TASK-068 |
| TASK-070 | Analyze what distinguishes recovery students (logistic regression, SHAP) | `[✓]` | TASK-069 |
| TASK-071 | Generate reports/retention/resilience_analysis.md | `[✓]` | TASK-070 |

---

## PHASE 4 — PLACEMENT RESEARCH

**Goal:** Analyze employability from the N=215 placement dataset with appropriate statistical power awareness.

### Phase 4A — Placement Feature Engineering

| ID | Title | Status | Dependencies |
|---|---|---|---|
| TASK-072 | Create src/placement/loader.py (load, validate, return clean DataFrame) | `[✓]` | TASK-026 |
| TASK-073 | Create src/placement/features.py — encode categoricals (gender, workex, specialisation, hsc_s, degree_t, ssc_b, hsc_b) | `[✓]` | TASK-072 |
| TASK-074 | Engineer composite academic features: academic_progression = hsc_p - ssc_p; degree_deviation = degree_p - hsc_p | `[✓]` | TASK-073 |
| TASK-075 | Document feature engineering decisions in feature log | `[✓]` | TASK-074 |

### Phase 4B — Placement Classification

| ID | Title | Status | Dependencies |
|---|---|---|---|
| TASK-076 | Implement Stratified 5-Fold CV for placement (N=215 requires all data in CV) | `[✓]` | TASK-073 |
| TASK-077 | Tier 0: Majority class baseline (always "Placed") | `[✓]` | TASK-076 |
| TASK-078 | Tier 1: Logistic Regression (placement) with calibration | `[✓]` | TASK-077 |
| TASK-079 | Tier 3: Random Forest (placement) with SHAP | `[✓]` | TASK-078 |
| TASK-080 | Tier 4: XGBoost (placement) with SHAP | `[✓]` | TASK-079 |
| TASK-081 | Report power limitations explicitly: N=215 limits confident conclusions (RULE-025) | `[✓]` | TASK-076 |
| TASK-082 | Subgroup analysis: placement rates by gender, work experience, specialisation | `[✓]` | TASK-078 |
| TASK-083 | Write tests: test_placement_models.py | `[✓]` | TASK-078 |
| TASK-084 | Log all experiments to MLflow | `[ ]` | TASK-077–TASK-082 |

### Phase 4C — Salary Regression (N=148)

| ID | Title | Status | Dependencies |
|---|---|---|---|
| TASK-085 | Create src/placement/salary.py — conditional salary regression (placed only) | `[✓]` | TASK-073 |
| TASK-086 | Tier 1: Linear regression (salary) with regularization | `[✓]` | TASK-085 |
| TASK-087 | Tier 3: Random Forest regression (salary) | `[✓]` | TASK-086 |
| TASK-088 | Report N=148 power constraint on salary conclusions | `[✓]` | TASK-086, TASK-087 |
| TASK-089 | Log salary experiments to MLflow separately from placement classification | `[ ]` | TASK-086, TASK-087 |

### Phase 4D — Employability Phenotypes

| ID | Title | Status | Dependencies |
|---|---|---|---|
| TASK-090 | Clustering analysis for employability profiles (K-Means, k=2–4 given N=215 constraint) | `[ ]` | TASK-073 |
| TASK-091 | Bootstrap stability (ARI) — report instability if ARI < 0.70 | `[ ]` | TASK-090 |
| TASK-092 | If stable: analyze placement rates and salary by phenotype | `[ ]` | TASK-091 |
| TASK-093 | Write notebook 04_placement_analysis.ipynb | `[✓]` | TASK-089, TASK-092 |
| TASK-094 | Generate reports/placement/placement_results.md | `[✓]` | TASK-089, TASK-092 |

---

## PHASE 5 — CROSS-DATASET ANALYSIS

**Goal:** Compare common constructs across datasets WITHOUT row merging.

| ID | Title | Status | Dependencies |
|---|---|---|---|
| TASK-095 | Create src/cross_dataset/representation_bridge.py — independent analysis across shared constructs | `[✓]` | TASK-042, TASK-073 |
| TASK-096 | Build feature correspondence map: which variables serve similar conceptual roles in each dataset | `[✓]` | TASK-095 |
| TASK-097 | Compare demographic distributions (Age, Gender) between retention and DLSM student cohorts | `[✓]` | TASK-054, TASK-082 |
| TASK-098 | Compute standardized effect sizes (Wasserstein distance, KS-test) for common variables | `[✓]` | TASK-097 |
| TASK-099 | Create comparison table and findings summary | `[✓]` | TASK-098 |
| TASK-100 | Write notebook 05_cross_dataset_analysis.ipynb | `[✓]` | TASK-099 |
| TASK-101 | Generate reports/cross_dataset/representation_bridge.md | `[✓]` | TASK-099 |

**Phase 5 Acceptance:** No row-level merge executed. All comparisons are at representation level. Report explicitly states: "These datasets represent different populations." ✅

---

## PHASE 6 — DLSM COMPATIBILITY (Formal Gate)

**Goal:** Formally document the NO-GO verdict with evidence.

| ID | Title | Status | Dependencies |
|---|---|---|---|
| TASK-102 | Run src/dlsm/compatibility_gate.py against both datasets | `[✓]` | TASK-032, TASK-042 |
| TASK-103 | Generate compatibility score (fraction of DLSM variables present) — expected: ~2–3/13 | `[✓]` | TASK-102 |
| TASK-104 | Generate reports/dlsm/compatibility_report.md — including what data WOULD be needed for a GO verdict | `[✓]` | TASK-103 |
| TASK-105 | Design future longitudinal dataset schema (what would make full integration valid) | `[✓]` | TASK-104 |
| TASK-106 | Write notebook 06_dlsm_compatibility.ipynb | `[✓]` | TASK-104 |

---

## PHASE 7 — DLSM EFFECTIVENESS (Ablation)

**Goal:** Even with limited compatible variables, run the ablation to show the NO-GO verdict is data-driven.

| ID | Title | Status | Dependencies |
|---|---|---|---|
| TASK-107 | Implement src/dlsm/effectiveness_test.py — ablation experiment structure | `[✓]` | TASK-052 |
| TASK-108 | Experiment A0: Retention baseline (academic features only) | `[✓]` | TASK-107 |
| TASK-109 | Experiment A1: Retention + Age + Gender (common DLSM-compatible demographics) | `[✓]` | TASK-107 |
| TASK-110 | Document ΔAUROC: expected ~0.00 (demographics already in baseline) | `[✓]` | TASK-108, TASK-109 |
| TASK-111 | Report: "DLSM integration provides no incremental value given absence of behavioral variables" | `[✓]` | TASK-110 |
| TASK-112 | Write notebook 07_dlsm_effectiveness.ipynb | `[✓]` | TASK-111 |
| TASK-113 | Generate reports/dlsm/effectiveness_report.md | `[✓]` | TASK-111 |


---

## PHASE 8 — EXPLAINABILITY & STATISTICAL ANALYSIS

| ID | Title | Status | Dependencies |
|---|---|---|---|
| TASK-114 | SHAP global summary plots for retention best model — save as Plotly figures | `[ ]` | TASK-054 |
| TASK-115 | SHAP local waterfall plots — 3 dropout + 3 retained students | `[ ]` | TASK-054 |
| TASK-116 | SHAP dependence plots — top 3 features | `[ ]` | TASK-054 |
| TASK-117 | SHAP global summary plots for placement best model | `[ ]` | TASK-082 |
| TASK-118 | PDP/ICE curves for top 3 placement features | `[ ]` | TASK-082 |
| TASK-119 | Subgroup analysis: retention AUC/recall by Gender, First_Generation, Financial_Stress quartile | `[ ]` | TASK-054 |
| TASK-120 | Subgroup analysis: placement AUC/recall by gender, workex | `[ ]` | TASK-082 |
| TASK-121 | Generate all explainability figures to reports/ directories | `[ ]` | TASK-114–TASK-120 |

---

## PHASE 9 — STREAMLIT DASHBOARD

**Goal:** Build the research observatory portal from validated analytical outputs.

| ID | Title | Status | Dependencies |
|---|---|---|---|
| TASK-122 | Create app/main.py skeleton — multi-page navigation | `[✓]` | TASK-023 |
| TASK-123 | Page 1: Research Overview — framework, hypotheses, DLSM verdict | `[✓]` | TASK-122 |
| TASK-124 | Page 2: Dataset Audit — schema tables, missingness charts, quality warnings | `[✓]` | TASK-030, TASK-031 |
| TASK-125 | Page 3: Retention Analysis — trajectories, risk simulator, calibrated probabilities | `[✓]` | TASK-060, TASK-056 |
| TASK-126 | Page 4: Placement Analysis — placement model, salary model, subgroups | `[✓]` | TASK-094 |
| TASK-127 | Page 5: Cross-Dataset Evidence — representation comparison, construct bridge | `[✓]` | TASK-101 |
| TASK-128 | Page 6: DLSM Compatibility — compatibility matrix table, NO-GO verdict | `[✓]` | TASK-104 |
| TASK-129 | Page 7: Explainability — SHAP plots, risk driver bar chart | `[✓]` | TASK-121 |
| TASK-130 | Page 8: Survival Analysis — Kaplan-Meier curves, Cox PH forest plot | `[✓]` | TASK-062 |
| TASK-131 | Page 9: Scientific Limitations — pre-declared limitations and advisory cautions | `[✓]` | TASK-122 |
| TASK-132 | Add @st.cache_data to all expensive computations | `[✓]` | TASK-123–TASK-131 |
| TASK-133 | Apply design system: CSS overrides, SSIF Plotly template, research context panels, limitation banners | `[✓]` | TASK-123–TASK-131 |
| TASK-134 | Anonymize Student_IDs before any display | `[✓]` | TASK-122 |

---

## PHASE 10 — TESTING

| ID | Title | Status | Dependencies |
|---|---|---|---|
| TASK-135 | Run full pytest suite — 34/34 passing tests across all modules | `[✓]` | TASK-024 + all phases |
| TASK-136 | Integration test: full retention pipeline runs end-to-end | `[✓]` | Phase 3 complete |
| TASK-137 | Integration test: full placement pipeline runs end-to-end | `[✓]` | Phase 4 complete |
| TASK-138 | Reproducibility test: re-run all models with same seed, verify identical metrics | `[✓]` | Phase 3+4 complete |
| TASK-139 | Leakage test: confirm End_of_Semester_Status never appears as feature in retention model | `[✓]` | Phase 3 complete |

---

## PHASE 11 — FINAL DOCUMENTATION

| ID | Title | Status | Dependencies |
|---|---|---|---|
| TASK-140 | Generate README.md — project overview, data requirements, setup instructions | `[✓]` | All phases |
| TASK-141 | Generate final reports/FINAL_RESEARCH_SUMMARY.md — all findings, limitations, future work | `[✓]` | All phases |
| TASK-142 | Update memory.md with final project state | `[✓]` | TASK-141 |

---

## STOP CONDITIONS

The project must STOP and surface a conflict if any of the following occur:

| Condition | Action |
|---|---|
| Leakage detected in retention model | Halt training, fix pipeline, re-run from TASK-045 |
| Survival analysis impossible (no temporal ordering) | Skip TASK-057–062, document in memory.md |
| Placement clustering ARI < 0.70 | Report instability, skip phenotype interpretation |
| Trajectory ΔAUROC < 0.01 | Report null result, document in memory.md, do not force trajectory narrative |
| DLSM compatibility score > threshold (unexpected) | Surface discovery, run effectiveness test before integration |

---

*Document owner: Project Manager / Lead Researcher*  
*Update: Check off tasks as completed; update status column immediately*
