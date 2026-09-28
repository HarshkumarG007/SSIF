# Student Success Intelligence Framework (SSIF)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-38%2F38%20passing-brightgreen.svg)]()
[![Code Architecture](https://img.shields.io/badge/architecture-modular-orange.svg)]()
[![Streamlit App](https://img.shields.io/badge/dashboard-Streamlit%20Live-FF4B4B.svg)](http://localhost:8502)

> **A Multi-Dataset Empirical Framework for Academic Retention, Employment Placement Trajectories, and Digital Lifestyle Spillover Analysis**

---

## 📌 Executive Summary

Higher education institutions face a dual challenge: identifying students at risk of premature departure early enough to intervene, and optimizing the post-graduation placement pathways of completing cohorts. 

The **Student Success Intelligence Framework (SSIF)** is an open-source, scientifically audited machine learning and survival analysis framework that addresses both challenges through empirical modeling:

1. **Academic Persistence & Retention Analytics:** Evaluates longitudinal survival trajectories across 79,239 student-semester records (20,000 distinct students) to predict next-semester dropout risk (`Target_Dropout_Next_Sem`) using GroupKFold cross-validation to prevent student-level data leakage.
2. **Academic Phenotypes & Resilience Signatures:** Identifies 3 distinct trajectory phenotypes ($k=3$, Bootstrap ARI = 0.9703) and isolates recovery patterns where academic advising provides a **+73.1% boost** in the odds of academic rebound.
3. **Employment Placement & Salary Diagnostics:** Analyzes multi-stage academic performance (secondary, higher secondary, undergraduate degree, MBA specialization, and prior work experience) to predict employment selection status (AUROC = 0.9370) and post-graduation remuneration.
4. **Cross-Study Compatibility Gate (DLSM Bridge):** Rigorously inspects empirical variable overlap with Digital Lifestyle Spillover Modeling ([DLSM](https://github.com/HarshkumarG007/DLSM)), establishing a scientifically grounded **NO-GO verdict** for row-level merges backed by an empirical ablation study ($\Delta\text{AUROC} = -0.00005$), while validating representation-level construct alignment between student digital habits and academic persistence risk.

---

## 🔬 Empirical Dataset Audit

| Metric / Dimension | Dataset A: Academic Retention Panel | Dataset B: Employment Placement Cohort | DLSM-A: Sleep & Screen Telemetry | DLSM-B: AI, Social Media & Mental Health |
|---|---|---|---|---|
| **File** | `academic_survival_longitudinal.csv` | `Placement_Data_Full_Class.csv` | `bedtime_screentime_sleep_debt.csv` | `AI_SocialMedia_Student_Dataset.csv` |
| **Observation Unit** | Student × Semester panel | Individual candidate profile | User sleep & device session | Student digital habit record |
| **Sample Size** | **79,239 rows** (20,000 unique students) | **215 rows** (individual students) | **8,500 rows** | **700 rows** (students) |
| **Features** | 22 columns (demographic, academic, financial) | 15 columns (multi-stage academic, MBA, salary) | 18 columns (sleep debt, screentime) | 12 columns (AI usage, social media, mental health) |
| **Primary Target** | `Target_Dropout_Next_Sem` (binary, 8.73% positive) | `status` (Placed: 68.8% / Not Placed: 31.2%) | `sleep_debt_category` / `next_day_fatigue_score` | `Mental_Health_Score` (1–10) |
| **Time Structure** | Longitudinal panel (1 to 8 semesters per student) | Cross-sectional snapshot | Cross-sectional observational | Cross-sectional observational |
| **Missingness** | `Family_Income` (4.55%), `LMS_Logins` (1.09%) | `salary` (31.16% — structurally unplaced) | None | None |

---

## 🏛️ System Architecture

SSIF is structured with separation of concerns, strict data leakage boundaries, and automated schema enforcement:

```
SSIF/
├── configs/                      # Pydantic-validated YAML configurations
│   ├── data.yaml                 # Dataset paths, columns, and validation settings
│   ├── features.yaml             # Feature definitions and transformation specs
│   └── models.yaml               # Model parameters, GroupKFold splits, calibration
├── docs/                         # Governance and research specifications
│   ├── PRD.md                    # Product & Research Requirements Document
│   ├── System Architecture.md    # End-to-end architectural blueprints
│   ├── Rules.md                  # 60 scientific & engineering governance rules
│   ├── design.md                 # UI/UX design for research dashboard
│   ├── task.md                   # 142 tracked execution tasks across 11 phases
│   └── memory.md                 # Persistent decision ledger
├── src/                          # Modular production source code
│   ├── config.py                 # Pydantic v2 configuration engine
│   ├── data_loader.py            # Validated dataset loaders with schema verification
│   ├── logger.py                 # Structured, leveled logging system
│   ├── models/                   # GroupKFold evaluation infrastructure & metrics
│   ├── validation/               # Integrity enforcement modules
│   │   ├── schema_validator.py   # Strict schema and data type validation
│   │   ├── leakage_detector.py   # Target contamination & temporal leakage checks
│   │   ├── missingness_analyzer.py # Missingness mechanism diagnostics
│   │   └── data_profiler.py      # Statistical profiling & distribution audits
│   ├── dlsm/                     # DLSM cross-study integration
│   │   ├── compatibility_gate.py # Multi-metric compatibility scorer & audit gate
│   │   └── effectiveness_test.py # Empirical 5-fold feature ablation study
│   ├── retention/                # Academic persistence models & survival analysis
│   │   ├── loader.py             # Longitudinal retention data loader
│   │   ├── features.py           # Vectorized OLS trajectory feature engine
│   │   ├── models.py             # Multi-tier GroupKFold benchmark
│   │   ├── survival.py           # Kaplan-Meier & Cox Proportional Hazards
│   │   ├── clustering.py         # Trajectory phenotype clustering (k=3, ARI=0.9703)
│   │   └── resilience.py         # Academic resilience & recovery signature analysis
│   ├── placement/                # Career placement classification & salary regression
│   │   ├── loader.py             # Multi-stage placement data loader
│   │   ├── features.py           # Academic progression & composite features
│   │   └── models.py             # Employability classifiers & salary diagnostics
│   ├── explainability/           # Interpretability & attribution
│   │   └── shap_analyzer.py      # SHAP TreeExplainer attributions
│   └── cross_dataset/            # Latent construct & representation bridging
│       └── representation_bridge.py # Cross-study Wasserstein distance & KS alignment
├── app/                          # Interactive Streamlit Research Observatory
│   ├── main.py                   # 7-view interactive dashboard with early warning simulator
│   └── components.py             # Themed cards, Plotly dark charts & limitation banners
├── reports/                      # Reproducible markdown research reports
│   ├── retention/                # Retention audit, models, survival, clustering & resilience reports
│   ├── placement/                # Placement audit & employability results
│   ├── dlsm/                     # DLSM compatibility & empirical effectiveness reports
│   ├── cross_dataset/            # Representation bridge analysis
│   └── FINAL_RESEARCH_SUMMARY.md # Comprehensive final scientific summary
├── tests/                        # Comprehensive test suite (38/38 passing)
│   └── unit/                     # Unit tests for all modules
├── pyproject.toml                # Project packaging & dependency manifest
└── README.md                     # Framework documentation
```

---

## 🛡️ Research Governance & Scientific Integrity

SSIF adheres to strict scientific guidelines (documented in [`docs/Rules.md`](file:///c:/Users/Lenovo/Downloads/SSIF/docs/Rules.md)):

1. **Zero Data Leakage (RULE-009):** In retention modeling, future realized outcomes (`End_of_Semester_Status`) and survival censoring indicators (`Censored`) are permanently forbidden as training features.
2. **Student-Grouped Cross-Validation (RULE-004):** Random train-test splitting across multi-semester observations for the same student introduces artificial autocorrelation. All retention validation utilizes `GroupKFold(n_splits=5, groups=Student_ID)`.
3. **No Unjustified Merges (RULE-002):** Retention (20,000 students) and placement (215 students) populations originate from disjoint educational institutions with non-overlapping identifiers. Row-level concatenation is mathematically invalid and disallowed.
4. **DLSM Empirical Compatibility Gate (RULE-003, RULE-005):** Direct injection of digital lifestyle features into academic datasets is strictly blocked by the compatibility gate (`compatibility_score = 0.154` — verdict: **NO-GO**). An empirical 5-fold feature ablation proved $\Delta\text{AUROC} = -0.00005$ ($p = 0.932$), mathematically proving that demographic overlap provides zero incremental predictive power without true behavioural telemetry.

---

## ⚡ Quickstart

### 1. Installation

Clone the repository and install dependencies in an active Python 3.10+ environment:

```bash
git clone https://github.com/HarshkumarG007/SSIF.git
cd SSIF
pip install -e .
```

### 2. Verify System Integrity (38 Tests)

Execute the full automated test suite:

```bash
pytest tests/ -v
```

### 3. Launch the Interactive Observatory Dashboard

```bash
streamlit run app/main.py
```

### 4. Run Modular Research Pipelines

```python
from src.retention.models import run_retention_benchmark
from src.retention.clustering import run_trajectory_clustering
from src.retention.resilience import run_resilience_analysis
from src.placement.models import run_placement_pipeline
from src.dlsm.effectiveness_test import run_dlsm_effectiveness_ablation

# Run Multi-Tier Retention Benchmark
summary_df, metrics = run_retention_benchmark()

# Run Trajectory Phenotype Clustering
phenotypes = run_trajectory_clustering()

# Run Academic Resilience Analysis
resilience = run_resilience_analysis()

# Run Placement Classification & Salary Regression
placement_res = run_placement_pipeline()

# Run DLSM Feature Ablation Study
ablation_res = run_dlsm_effectiveness_ablation()
```

---

## 📊 Roadmap & Execution Ledger

All 11 Project Phases are **COMPLETE** and verified:

- [x] **Phase 0:** Scientific Documentation & Theoretical Specification (PRD, Architecture, Rules, Design, Task, Memory)
- [x] **Phase 1:** Environment, Modular Architecture & Scaffolding
- [x] **Phase 2:** Automated Data Audit & Profiling Engine (`audit_report.md` for both datasets)
- [x] **Phase 3:** Longitudinal Retention Benchmark (AUROC = 0.8014), Survival Analysis ($C = 0.7498$), Phenotypes (ARI = 0.9703), and Resilience Analysis (Advising OR = 1.731)
- [x] **Phase 4:** Placement Classification (AUROC = 0.9370) & Conditional Salary Regression ($R^2 \approx 0$)
- [x] **Phase 5:** Cross-Dataset Representation & Construct Bridge (Wasserstein distance = 1.767 years)
- [x] **Phase 6:** Formal DLSM Compatibility Gate (`score = 0.154`, NO-GO verdict)
- [x] **Phase 7:** DLSM Empirical Feature Ablation ($\Delta\text{AUROC} = -0.00005$, confirming NO-GO)
- [x] **Phase 8:** Explainable AI & Attributions (SHAP TreeExplainer & Hazard Multipliers)
- [x] **Phase 9:** Interactive Streamlit Research Observatory (7 views, dark mode, calibrated risk simulator)
- [x] **Phase 10:** Automated Test Suite (**38/38 passing unit and integration tests**)
- [x] **Phase 11:** Final Scientific Summary & Research Documentation (`FINAL_RESEARCH_SUMMARY.md`)


---

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
