# Student Success Intelligence Framework (SSIF)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-19%2F19%20passing-brightgreen.svg)]()
[![Code Architecture](https://img.shields.io/badge/architecture-modular-orange.svg)]()

> **A Multi-Dataset Empirical Framework for Academic Retention, Employment Placement Trajectories, and Digital Lifestyle Spillover Analysis**

---

## 📌 Executive Summary

Higher education institutions face a dual challenge: identifying students at risk of premature departure early enough to intervene, and optimizing the post-graduation placement pathways of completing cohorts. 

The **Student Success Intelligence Framework (SSIF)** is an open-source, scientifically audited machine learning and survival analysis framework that addresses both challenges through empirical modeling:

1. **Academic Persistence & Retention Analytics:** Evaluates longitudinal survival trajectories across 79,239 student-semester records (20,000 distinct students) to predict next-semester dropout risk (`Target_Dropout_Next_Sem`) using GroupKFold cross-validation to prevent student-level data leakage.
2. **Employment Placement & Salary Forecasting:** Analyzes multi-stage academic performance (secondary, higher secondary, undergraduate degree, MBA specialization, and prior work experience) to predict employment selection status and post-graduation remuneration.
3. **Cross-Study Compatibility Gate (DLSM Bridge):** Rigorously inspects empirical variable overlap with Digital Lifestyle Spillover Modeling ([DLSM](https://github.com/HarshkumarG007/DLSM)), establishing a scientifically grounded **NO-GO verdict** for row-level merges while enabling representation-level construct alignment between student digital habits and academic persistence risk.

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
│   ├── validation/               # Integrity enforcement modules
│   │   ├── schema_validator.py   # Strict schema and data type validation
│   │   └── leakage_detector.py   # Target contamination & temporal leakage checks
│   ├── dlsm/                     # DLSM cross-study integration
│   │   └── compatibility_gate.py # Multi-metric compatibility scorer & audit gate
│   ├── retention/                # Academic persistence models & survival analysis
│   ├── placement/                # Career placement classification & salary regression
│   ├── cross_dataset/            # Latent construct & representation bridging
│   └── visualization/            # Research visualization & Streamlit dashboard
├── tests/                        # Comprehensive test suite (unit, integration, regression)
│   └── unit/
│       └── test_schema_validator.py # 19 passing unit tests for data integrity
├── pyproject.toml                # Project packaging & dependency manifest
└── README.md                     # Framework documentation
```

---

## 🛡️ Research Governance & Scientific Integrity

SSIF adheres to strict scientific guidelines (documented in [`docs/Rules.md`](file:///c:/Users/Lenovo/Downloads/SSIF/docs/Rules.md)):

1. **Zero Data Leakage (RULE-009):** In retention modeling, future realized outcomes (`End_of_Semester_Status`) and survival censoring indicators (`Censored`) are permanently forbidden as training features.
2. **Student-Grouped Cross-Validation (RULE-004):** Random train-test splitting across multi-semester observations for the same student introduces artificial autocorrelation. All retention validation utilizes `GroupKFold(n_splits=5, groups=Student_ID)`.
3. **No Unjustified Merges (RULE-002):** Retention (20,000 students) and placement (215 students) populations originate from disjoint educational institutions with non-overlapping identifiers. Row-level concatenation is mathematically invalid and disallowed.
4. **DLSM Empirical Compatibility Gate (RULE-003):** Direct injection of digital lifestyle features into academic datasets is strictly blocked by the compatibility gate (`compatibility_score = 0.154` — verdict: **NO-GO**). Instead, representation-level latent constructs bridge the domains.

---

## ⚡ Quickstart

### 1. Installation

Clone the repository and install dependencies in an active Python 3.10+ environment:

```bash
git clone https://github.com/HarshkumarG007/SSIF.git
cd SSIF
pip install -e .
```

### 2. Verify System Integrity

Execute the automated test suite to verify schema validators, leakage detectors, and compatibility gate functions:

```bash
pytest tests/ -v
```

### 3. Load & Audit Datasets

```python
from src.data_loader import load_retention, load_placement
from src.dlsm.compatibility_gate import check_dlsm_compatibility

# Load validated dataframes
df_retention = load_retention()
print("Retention Panel:", df_retention.shape)

df_placement = load_placement()
print("Placement Cohort:", df_placement.shape)

# Run the DLSM Compatibility Gate
report = check_dlsm_compatibility()
print(f"DLSM Compatibility Verdict: {report.verdict} (Score: {report.compatibility_score:.3f})")
```

---

## 📊 Roadmap & Execution Ledger

Implementation progress is tracked in real-time in [`docs/task.md`](file:///c:/Users/Lenovo/Downloads/SSIF/docs/task.md) and [`docs/memory.md`](file:///c:/Users/Lenovo/Downloads/SSIF/docs/memory.md):

- [x] **Phase 0:** Scientific Documentation & Theoretical Specification (PRD, Architecture, Rules, Design, Task, Memory)
- [x] **Phase 1:** Environment, Modular Architecture & Integrity Suite (19/19 Tests Passing)
- [ ] **Phase 2:** Automated Data Audit & Profiling Engine
- [ ] **Phase 3:** Longitudinal Academic Persistence & Trajectory Engineering
- [ ] **Phase 4:** Survival Analysis (Kaplan-Meier, Cox PH, Time-to-Dropout)
- [ ] **Phase 5:** Placement Prediction & Salary Estimation Models
- [ ] **Phase 6:** Explainable AI & Feature Attribution (SHAP, Counterfactuals)
- [ ] **Phase 7:** Fair ML & Disparate Impact Auditing
- [ ] **Phase 8:** Cross-Dataset Representation & DLSM Construct Bridge
- [ ] **Phase 9:** Interactive Streamlit Research Dashboard
- [ ] **Phase 10:** Automated Research Report Generation

---

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
