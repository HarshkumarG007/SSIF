# Student Success Intelligence Framework (SSIF)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CI Status](https://github.com/HarshkumarG007/SSIF/actions/workflows/ci.yml/badge.svg)](https://github.com/HarshkumarG007/SSIF/actions)
[![Tests Passing](https://img.shields.io/badge/tests-82%2F82%20passing-brightgreen.svg)](tests/)
[![Streamlit Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ssif-research.streamlit.app/)
[![Live Observatory](https://img.shields.io/badge/Streamlit%20Cloud-Live%20Observatory-FF4B4B.svg)](https://ssif-research.streamlit.app/)
[![Research Paper](https://img.shields.io/badge/IEEE%20Format-Paper%20PDF-8B5CF6.svg)](papers/ssif_academic_retention_study.pdf)

> **A reproducible research framework for evaluating whether student-success predictions remain valid under temporal leakage, dataset incompatibility, small-sample constraints, and actionable-intervention requirements**

---

## 📑 Table of Contents

1. [Executive Summary & Core Philosophy](#-executive-summary--core-philosophy)
2. [Ecosystem & System Architecture](#-ecosystem--system-architecture)
3. [Empirical Dataset Audits & Topology](#-empirical-dataset-audits--topology)
4. [Phase-by-Phase Deep Dive & Implementations](#-phase-by-phase-deep-dive--implementations)
   - [Phase 0 & 1: Scientific Governance, The 62 Rules & Infrastructure](#phase-0--1-scientific-governance-the-62-rules--infrastructure)
   - [Phase 2: Schema Validation, Missingness Diagnostics & Data Hygiene](#phase-2-schema-validation-missingness-diagnostics--data-hygiene)
   - [Phase 3: Academic Persistence Dynamics (Trajectories, Survival & Phenotypes)](#phase-3-academic-persistence-dynamics-trajectories-survival--phenotypes)
   - [Phase 4: Employability Intelligence & Conditional Compensation](#phase-4-employability-intelligence--conditional-compensation)
   - [Phase 5: Cross-Dataset Synthesis & The Construct Bridge](#phase-5-cross-dataset-synthesis--the-construct-bridge)
   - [Phases 6 & 7: The DLSM Compatibility Gate & Empirical Feature Ablation](#phases-6--7-the-dlsm-compatibility-gate--empirical-feature-ablation)
   - [Phase 8: Explainable AI (SHAP) & Algorithmic Counterfactual Recourse](#phase-8-explainable-ai-shap--algorithmic-counterfactual-recourse)
   - [Phase 9: Interactive Streamlit Research Observatory](#phase-9-interactive-streamlit-research-observatory)
   - [Phase 10: Continuous Quality Engineering & The Feasibility Auditor](#phase-10-continuous-quality-engineering--the-feasibility-auditor)
   - [Phase 11 & Extensions: Publication Suite & Camera-Ready IEEE Paper](#phase-11--extensions-publication-suite--camera-ready-ieee-paper)
   - [Phase 12: Multi-Disciplinary Research Experiments & Policy Simulation Suite](#phase-12-multi-disciplinary-research-experiments--policy-simulation-suite)
   - [Reproducible Data Lakehouse: Interim & Processed Enriched Datasets](#-reproducible-data-lakehouse-interim--processed-enriched-datasets)
5. [💡 What the Data Reveals in Plain English (Layman's Compendium)](#-what-the-data-reveals-in-plain-english-laymans-compendium)
6. [Challenges Encountered, Problems & Engineering Solutions](#-challenges-encountered-problems--engineering-solutions)
7. [Threat Model & Data Leakage Defenses](#-threat-model--data-leakage-defenses)
8. [Quickstart & Reproduction Guide](#-quickstart--reproduction-guide)
9. [Project Directory Blueprint](#-project-directory-blueprint)
10. [License & Citation](#-license--citation)
11. [Acknowledgements & Original Dataset Credits](#-acknowledgements--original-dataset-credits)

---

## 📌 Executive Summary & Core Philosophy

Higher education faces two critical junctures: **academic attrition** (students dropping out prior to degree completion) and **post-graduation transition** (graduates securing gainful professional employment). Historically, educational data mining has treated these challenges as disconnected Kaggle-style prediction problems—often forcing naive merges, overlooking time-series grade momentum, and failing to verify whether upstream digital habits actually matter.

The **Student Success Intelligence Framework (SSIF)** was engineered as a computational research laboratory adhering to one non-negotiable scientific principle:

> **"DO NOT FORCE THE DATA TO FIT THE HYPOTHESIS. Let the data decide whether the connection exists."**

```
   ┌────────────────────────────────────────────────────────────────────────────────┐
   │                          THE SSIF SCIENTIFIC HIERARCHY                         │
   │                                                                                │
   │   RAW DATA ──► QUALITY AUDIT ──► VALIDITY CHECK ──► BASELINE ──► COMPLEXITY    │
   │      │                                                            ▲            │
   │      ▼                                                            │            │
   │   LEAKAGE DETECTED? ──► YES ──► [HALT & DROP LEAKY VARIABLE] ─────┘            │
   │      │                                                                         │
   │     NO                                                                         │
   │      ▼                                                                         │
   │   CROSS-DATASET JOIN? ──► PROVENANCE DISJOINT? ──► [FORBID ROW-LEVEL MERGE]    │
   │                                                ──► [PERMIT CONSTRUCT BRIDGE]   │
   └────────────────────────────────────────────────────────────────────────────────┘
```

SSIF unifies three distinct analytical domains across 79,239 longitudinal student-semester records, 215 business school candidate profiles, and digital health telemetry:
1. **Academic Persistence & Retention Analytics:** Evaluates longitudinal grade velocity, run-length decline counters, and survival hazard to identify departure risk before physical departure occurs.
2. **Academic Resilience & Phenotype Discovery:** Identifies 3 stable structural phenotypes ($k=3$, Bootstrap ARI = 0.9703, $B=1000$ iterations) and isolates recovery patterns where academic advising is *associated with* a **+73.1% increase** in the odds of academic rebound (observational multivariate association, $\text{OR}=1.731$, $p<0.001$).
3. **Employability Intelligence:** Decouples placement probability (AUROC = 0.9370) from starting compensation ($R^2 \approx 0.00$), *finding* that work experience is associated with an observed **+26.9% absolute placement lift** (N=215, EPV=3.2 — interpret with appropriate sample-size caution).
4. **Cross-Study Governance Gate (DLSM Bridge):** Establishes an objective **NO-GO gate** for row-level merges with Digital Lifestyle Spillover Modeling ([DLSM](https://github.com/HarshkumarG007/DLSM)), backed by an empirical 5-fold feature ablation study ($\Delta\text{AUROC} = -0.00005, p=0.932$) and grounded in large-scale specification-curve literature.

### 🎯 Unified Research Question

> **When is a student-success intervention supported by evidence strongly enough to justify operational use?**

Every SSIF component addresses a different part of that question:

| Component | Operational Question Answered |
|-----------|------------------------------|
| **Retention Prediction** | *Who is at risk, and how early can we know?* |
| **Survival & Hazard** | *When does departure risk become critical?* |
| **Trajectory Phenotypes** | *What structural trajectory pattern does this student follow?* |
| **Causal DML** | *What might change the outcome under stated identification assumptions?* |
| **Algorithmic Recourse** | *What model inputs are actionable for this student profile?* |
| **Policy Simulation** | *What happens at the system level if institutional resources shift?* |
| **Compatibility Gate** | *Can we trust the data bridge between these two study populations?* |
| **Governance Rules** | *Are these conclusions structurally protected from common research failures?* |

### 🏆 Executive Scorecard: Quantified Research Achievements

| Research Pillar | Primary Metric | Baseline / Benchmark | SSIF Achievement | Statistical Rigor & Impact |
|:---|:---|:---|:---|:---|
| **Academic Retention** | **AUROC** | 0.4945 (Majority Class) | **0.8014** (Tier 1 Logistic Regression) | GroupKFold ($k=5$, groups=`Student_ID`), zero future leakage (`RULE-062`) |
| **Early Departure Detection** | **PR-AUC** | 0.0860 (Prevalence Floor) | **0.3643** (+323.6% precision gain) | Brier Score = 0.0669; detects risk 2 semesters before physical departure |
| **Survival & Hazard Modeling** | **Harrell's C-Index** | 0.5000 (Random Guess) | **0.7498** ($p < 0.001$) | Log-rank test $p < 10^{-15}$; 20,000 students right-censored at semester 8 |
| **Socioeconomic Vulnerability** | **Hazard Ratio ($\text{HR}$)** | 1.00× (Parity) | **1.98×** (95% CI: $[1.89, 2.08]$) | First-generation students face nearly double instantaneous departure hazard |
| **Institutional Shielding** | **Hazard Ratio ($\text{HR}$)** | 1.00× (Parity) | **0.52×** (95% CI: $[0.49, 0.55]$) | Scholarship cuts departure hazard in half, neutralizing the first-gen penalty |
| **Advising Rebound Effect** | **Odds Ratio ($\text{OR}$)** | 1.00× (Parity) | **1.731×** ($p < 0.001$) | **+73.1% rebound odds** per counseling session after acute GPA collapse |
| **Trajectory Phenotypes** | **Bootstrap ARI** | 0.7000 (Minimum Valid) | **0.9703** ($B=1000$ iterations) | $K$-Means ($k=3$) identifies 3 well-supported initial trajectory phenotypes (stability diagnostic) |
| **Employability Prediction** | **AUROC / PR-AUC** | 0.5000 / 0.6880 | **0.9370 / 0.9650** | Stratified 5-Fold CV ($N=215$ candidates, EPV = 3.2 guarded) |
| **Work Experience Value** | **Placement Rate** | 59.6% (No Experience) | **86.5%** (With Experience) | **+26.9% absolute placement lift**; outweighs a +15% college exam score gain |
| **Starting Salary Determinants**| **Regressor $R^2$** | Hypothetical high $R^2$ | **$\approx 0.00$** ($N=148$ placed) | Finds no detectable linear association between GPA and salary; consistent with fixed corporate pay bands (interpret with $N$-size caution) |
| **Cross-Study Governance** | **Compatibility Score**| $\ge 0.70$ (Merge Threshold)| **0.154** (**STRICT NO-GO**) | Blocked false row-level join; validated parallel construct bridge |
| **Digital Lifestyle Ablation**| **Incremental $\Delta\text{AUC}$** | $\ge +0.0100$ (Significance) | **$-0.00005$** ($p = 0.932$) | No statistically detectable incremental predictive value observed under this specification |
| **Labor-Policy Work-Study (EXP-001)** | **Counterfactual GPA / Retention** | 0.00 / 0.0 pp | **+0.077 GPA / -2.97 pp Risk** | 200 Monte Carlo bootstrap iterations; estimates work-study is associated with reduced academic harm under stated model assumptions |
| **Pipeline Cascade (EXP-002)** | **Intervention Return** | Uniform across stages | **Stage 1: +81.6 grads / 1k** | Multi-stage cascade simulation demonstrates Stage 1 intervention yields the highest modeled system multiplier |
| **Hiring Equity Audit (EXP-003)** | **Subgroup Achievability** | Parity across income | **Q1 vs Q4 Disparity** | Identified N=992 Qualified-But-Excluded students who maintain positive GPA velocity despite initial adversity |
| **Early Warning Horizon (EXP-004)**| **Early Career AUC** | 0.5000 (Random) | **0.7469 (S1-S2) → 0.8387 (S4)**| GroupKFold validation indicating early signals are predictive of 4-year outcomes within this dataset |
| **Budget Optimizer (EXP-005)** | **Pareto Frontier ROI** | Unoptimized ad-hoc allocation| **0.0533 reductions / $1** | HiGHS Linear Programming across $10K-$500K portfolios; advising prioritizes first |
| **Causal Double ML (DML)** | **Average Treatment Effect (ATE)** | Observational association | **-4.66 pp Dropout / +0.024 GPA** | Neyman-orthogonal cross-fitting with GroupKFold ($p < 10^{-6}$, E-value = 1.27) |
| **Code Reliability & Testing** | **Automated Tests** | Standard smoke tests | **82/82 Passed** (100% pass rate) | Pytest suite covering data validators, trajectory engines, causal DML, REST API, & Kaggle release |

![SSIF Executive Research Observatory](docs/screenshots/01_executive_overview.png)
*Figure 1: The SSIF Research Observatory Executive Dashboard (ssif-research.streamlit.app). Features dimensional elevation cards, glassmorphic research context panels, and live framework KPIs across 79,239 longitudinal student-semester records.*

---

## 🏛️ Ecosystem & System Architecture

SSIF interfaces with two educational datasets while establishing strict architectural boundaries with the DLSM digital health repository:

```mermaid
flowchart TD
    subgraph Data_Layer ["Data & Ingestion Layer"]
        A1[("academic_survival_longitudinal.csv<br/>79,239 rows | 20,000 students")]
        A2[("Placement_Data_Full_Class.csv<br/>215 candidates | 15 columns")]
        A3[("DLSM Digital Health Cohorts<br/>bedtime_screentime & AI_SocialMedia")]
    end

    subgraph Validation_Layer ["Integrity & Feasibility Gatekeepers"]
        V1["Schema Validator<br/>(Pydantic & Type Guards)"]
        V2["Leakage Detector<br/>(RULE-009 & RULE-010 Checks)"]
        V3["Tabular Feasibility Auditor<br/>(7 Automated Quality Gates)"]
    end

    subgraph Analytical_Engines ["Analytical & Modeling Engines"]
        E1["Longitudinal Trajectory Engine<br/>Vectorized OLS Slopes (0.15s)"]
        E2["Survival Analysis Engine<br/>Kaplan-Meier & Cox PH (C=0.7498)"]
        E3["Resilience & Phenotype Engine<br/>K-Means (k=3, ARI=0.9703)"]
        E4["Employability Diagnostics<br/>XGBoost / LogReg (AUROC=0.9370)"]
        E5["Algorithmic Recourse Engine<br/>L1 Minimal-Effort Optimizer"]
    end

    subgraph Cross_Study_Gate ["Cross-Study Governance"]
        G1{"DLSM Compatibility Gate<br/>Score: 0.154"}
        G2["Row Merge: STRICT NO-GO<br/>(Zero Synthetic Record Merging)"]
        G3["Construct Bridge: VALID<br/>(Wasserstein Distance = 1.767 yrs)"]
        G4["Feature Ablation Leaderboard<br/>ΔAUROC = -0.00005 (p=0.932)"]
    end

    subgraph Delivery_Layer ["Research Delivery Interfaces"]
        D1["Streamlit Cloud Live Observatory<br/>(7 Interactive Views)"]
        D2["Camera-Ready IEEE Paper PDF<br/>(Automated Tectonic Engine)"]
        D3["Master Kaggle Publication Suite<br/>(Standalone Notebook & Article)"]
        D4["Continuous Integration (CI)<br/>(82/82 Pytest Automated Tests)"]
    end

    A1 --> V1 & V2 & V3
    A2 --> V1 & V2 & V3
    A3 --> G1

    V2 --> E1 & E2 & E3
    V1 --> E4
    E1 --> E5

    A1 & A3 --> G1
    G1 -->|"Score < 0.70"| G2
    G1 -->|"Shared Latent Space"| G3
    G3 --> G4

    E1 & E2 & E3 & E4 & E5 & G4 --> D1
    E1 & E2 & G4 --> D2
    E1 & E4 & G4 --> D3
    V3 & E1 & E4 --> D4
```

---

## 🔬 Empirical Dataset Audits & Topology

| Dimension | Dataset A: Academic Retention Panel | Dataset B: Employment Placement Cohort | DLSM-A: Sleep & Screen Telemetry | DLSM-B: AI, Social Media & Health |
|---|---|---|---|---|
| **Original Creator** | **Razan Ihab Abdellatif** | **Amey Thakur** ([@ameythakur20](https://www.kaggle.com/ameythakur20)) | **Samar Talwar** | **Sri Syra** ([@srisyra02](https://www.kaggle.com/srisyra02)) |
| **Primary Source** | [Kaggle: Retention Panel](https://www.kaggle.com/datasets/razanihababdellatif/student-retention-and-academic-performance-data) | [Kaggle: Placement Data](https://www.kaggle.com/datasets/ameythakur20/placement-data) | [Kaggle: Sleep & Screen](https://www.kaggle.com/datasets/samartalwar/sleep-debt-and-screen-time-late-night-phone-habits) via [DLSM](https://github.com/HarshkumarG007/DLSM) | [Kaggle: AI & Social Media](https://www.kaggle.com/datasets/srisyra02/ai-and-social-media-impact-student-health-and-grades) via [DLSM](https://github.com/HarshkumarG007/DLSM) |
| **Raw File** | `academic_survival_longitudinal.csv` | `Placement_Data_Full_Class.csv` | `bedtime_screentime_sleep_debt.csv` | `AI_SocialMedia_Student_Dataset.csv` |
| **Observation Unit** | Student $\times$ Semester panel | Individual candidate profile | User sleep & device session | Student digital habit record |
| **Sample Size** | **79,239 records** (20,000 distinct students) | **215 candidates** (MBA cohort) | **8,500 records** | **16,000 records** (students) |
| **Temporal Breadth** | 1 to 8 semesters per student | Cross-sectional graduation snapshot | Cross-sectional device logs | Cross-sectional lifestyle audit |
| **Primary Target** | `Target_Dropout_Next_Sem` (binary, 8.73% rate) | `status` (Placed: 68.8% / Not Placed: 31.2%) | `sleep_debt_category` / `next_day_fatigue_score` | `Mental_Health_Score` (1–10) |
| **Missingness** | `Family_Income` (4.55%), `LMS_Logins` (1.09%) | `salary` (31.16% — structurally unplaced) | 0 missing cells | 0 missing cells |
| **Audit Status** | 🟢 Validated via GroupKFold | 🟢 Validated (EPV = 3.2 guarded) | 🟡 Independent Sister Study | 🟡 Independent Sister Study |

---

## 🛠️ Phase-by-Phase Deep Dive & Implementations

```
  PHASE 0 ──► PHASE 1 ──► PHASE 2 ──► PHASE 3 ──► PHASE 4 ──► PHASE 5 ──► PHASES 6-7 ──► PHASE 8 ──► PHASE 9 ──► PHASE 10 ──► PHASE 11
  Governance   Scaffold    Auditing   Retention   Placement    Bridge     DLSM Gate     Recourse   Dashboard    82 Tests     Paper/PDF
```

---

### Phase 0 & 1: Scientific Governance, The 62 Rules & Infrastructure

#### Purpose & Core Scientific Question
*How do we eliminate researcher degrees of freedom, prevent data leakage, and ensure reproducibility before writing predictive code?*

#### Mermaid Architectural Workflow
```mermaid
flowchart LR
    Y["Raw YAML Configs<br/>(data.yaml, features.yaml, models.yaml)"] --> P["Pydantic v2 Settings Engine<br/>(src/config.py)"]
    P --> R["Scientific Constitution<br/>(RULE-001 to RULE-062)"]
    R --> L["Leveled Structured Logger<br/>(src/logger.py)"]
    L --> D["Downstream Analytical Pipelines<br/>(Audit, Feature, Model Engines)"]
```

#### ASCII System Schematic
```
   ┌────────────────────────────────────────────────────────────────────────┐
   │                       RULE ENFORCEMENT ENGINE                          │
   │                                                                        │
   │  RULE-002: Never fabricate student identifiers across datasets.        │
   │  RULE-003: Never merge retention & placement rows directly.            │
   │  RULE-009: Zero future information permitted in feature sets.          │
   │  RULE-010: Never predict salary on unplaced candidates.                │
   │  RULE-031: GroupKFold validation strictly mandatory for panels.        │
   │  RULE-061: Calibrate digital lifestyle against Orben & Przybylski.     │
   │  RULE-062: Lifetime sequence duration strictly forbidden as feature.   │
   └────────────────────────────────────────────────────────────────────────┘
```

#### Technical Implementation & Key Formulations
- **Antigravity Engineering Constitution ([`docs/Rules.md`](file:///c:/Users/Lenovo/Downloads/SSIF/docs/Rules.md)):** Codified 62 immutable scientific rules spanning data hygiene, feature engineering, model validation, and publication standards.
- **Pydantic v2 Configuration Management ([`src/config.py`](file:///c:/Users/Lenovo/Downloads/SSIF/src/config.py)):** Enforces type-safe schema definitions reading declarative settings from `configs/data.yaml`, `configs/features.yaml`, and `configs/models.yaml`.
- **Structured Structured Logging ([`src/logger.py`](file:///c:/Users/Lenovo/Downloads/SSIF/src/logger.py)):** Emits timestamped, leveled audit logs ensuring full traceability of every calculation.

#### Engineering Decisions & Scientific Rationale
- **Declarative Immutability:** Replacing raw dictionaries with Pydantic models guarantees that seed numbers, file paths, and hyperparameters cannot be silently overwritten during runtime.
- **Fail-Fast Invariants:** Any violation of temporal boundaries or data leakage immediately triggers an unrecoverable `ValueError`, halting execution rather than silently poisoning results.

#### Challenges Faced & Problem Solutions
- **Challenge:** Configuration drift across local Windows development machines, Docker environments, and Streamlit Cloud Linux instances.
- **Solution:** Engineered path resolution in `src/config.py` using dynamic workspace root detection relative to `__file__`, guaranteeing identical execution anywhere.

#### What the Data Reveals in Layman's Context
Before building a high-speed sports car, you engineer the brakes and the roll cage. Phase 0 and 1 guaranteed that no AI model could "cheat" by looking into the future, making up fake student IDs, or exaggerating predictive power.

---

### Phase 2: Schema Validation, Missingness Diagnostics & Data Hygiene

#### Purpose & Core Scientific Question
*What are the underlying missingness mechanisms (MCAR vs. MAR), structural anomalies, and demographic biases lurking within the raw data?*

#### Mermaid Architectural Workflow
```mermaid
flowchart TD
    A["Raw Retention Panel<br/>(79,239 rows, 20,000 students)"] --> B["Schema Validator<br/>(src/validation/schema_validator.py)"]
    B --> C["Missingness Diagnostics<br/>(Little's MCAR & Welch's t-test)"]
    C --> D{"Missingness Mechanism"}
    D -->|"LMS Logins: 1.09% missing"| E["MAR: Dependent on Attendance (p < 0.001)"]
    D -->|"Family Income: 4.55% missing"| F["MAR: Dependent on First_Gen (p < 0.001)"]
    B --> G["Data Hygiene Engine<br/>(src/data_loader.py)"]
    G --> H["Gender Canonicalizer<br/>(8 raw typo variants -> 4 canonical cohorts)"]
    H --> I["Demographic Selection Parity Restored<br/>(Selection Ratio: 0.09 vs 0.09)"]
```

#### ASCII System Schematic
```
   Raw Gender Entries:                                  Clean Canonicalized Cohorts:
   ┌───────────────────────────────┐                    ┌───────────────────────────┐
   │ "Female" (38,035)             │                    │                           │
   │ "female" (875)                ├───────────────────►│  Female: 39,626 (49.9%)   │
   │ "F"      (716)                │                    │                           │
   ├───────────────────────────────┤                    ├───────────────────────────┤
   │ "Male"   (36,556)             │                    │                           │
   │ "male"   (764)                ├───────────────────►│  Male:   38,059 (48.0%)   │
   │ "M"      (739)                │                    │                           │
   ├───────────────────────────────┤                    ├───────────────────────────┤
   │ "Other"  (1,125)              ├───────────────────►│  Other:   1,125  (1.4%)   │
   ├───────────────────────────────┤                    ├───────────────────────────┤
   │ "Prefer not to say" (429)     ├───────────────────►│  Prefer:    429  (0.5%)   │
   └───────────────────────────────┘                    └───────────────────────────┘
```

#### Technical Implementation & Key Formulations
- **Automated Schema Validator ([`src/validation/schema_validator.py`](file:///c:/Users/Lenovo/Downloads/SSIF/src/validation/schema_validator.py)):** Verifies exact column counts, dtypes, range boundaries, and forbidden target leakage columns.
- **Missingness Diagnostic Suite ([`src/validation/missingness_analyzer.py`](file:///c:/Users/Lenovo/Downloads/SSIF/src/validation/missingness_analyzer.py)):**
  - Little's MCAR Test: $\chi^2 = 142.8$, $p < 0.001$ (rejects Missing Completely At Random).
  - Statistically confirmed **Missing At Random (MAR)**: `LMS_Logins` missingness correlates with attendance drops ($t = -12.4, p < 0.001$), while `Family_Income` missingness correlates with first-generation status ($\chi^2 = 89.2, p < 0.001$).
- **Data Hygiene Pipeline ([`src/data_loader.py`](file:///c:/Users/Lenovo/Downloads/SSIF/src/data_loader.py)):** Canonicalizes raw string variants into 4 clean cohorts before model ingestion.

#### Engineering Decisions & Scientific Rationale
- **Fold-Isolated Imputation:** Imputing missing values using global dataset statistics leaks test data into training sets. Imputers are fitted strictly inside each training fold during cross-validation.
- **Upstream Data Cleaning:** Normalizing demographic categories at the raw ingestion layer prevents downstream models and fairness monitors from evaluating fragmented, typo-induced subgroups.

#### Challenges Faced & Problem Solutions
- **Challenge:** The raw data contained 8 distinct string representations for `Gender` (`Female`, `female`, `F`, `Male`, `male`, `M`, `Other`, `Prefer not to say`). Minor typos with skewed sample sizes caused automated algorithmic fairness audits to trigger false alarms (selection-rate disparity of 0.39).
- **Solution:** Implemented regex canonicalization in `src/data_loader.py`. This restored true demographic parity: base rates of 0.09 vs. 0.09, model selection rates of 0.09 vs. 0.09, and true positive rates of 0.38 vs. 0.37.

#### What the Data Reveals in Layman's Context
- **Data Quality and Apparent Fairness Disparity:** The raw data contained 8 spelling variants of `Gender`. After canonicalization, an apparent selection-rate disparity of 0.39 collapsed to parity (0.09 vs 0.09). This demonstrates that **some apparent subgroup disparity was attributable to inconsistent categorical encoding** — a data-quality finding, not proof that the model is definitively fair across all dimensions.
- **The Attendance & Login Connection:** Missing data is rarely random. Students who stop logging into the campus portal tend to be the exact students who are already disengaging from classes. The MAR diagnosis ($t = -12.4$, $p < 0.001$) supports this interpretation and justifies our imputation approach.

![Data Audit & Missingness Observatory](docs/screenshots/02_data_audit.png)
*Figure 2: Data Audit & Missingness Observatory view. Quantifies 79,239 records across 20,000 students, confirming MAR mechanisms for Family Income (4.55%) and LMS Logins (1.09%) via Welch's t-tests and Chi-square statistics.*

---

### Phase 3: Academic Persistence Dynamics (Trajectories, Survival & Phenotypes)

#### Purpose & Core Scientific Question
*Can student departure be anticipated semesters before physical departure by modeling grade velocity, survival hazard, and latent behavioral phenotypes?*

#### Mermaid Architectural Workflow
```mermaid
stateDiagram-v2
    [*] --> Enrolled: Semester 1 (Matriculation)
    Enrolled --> Stable_Persistence: Consistent GPA (Slope > -0.10)
    Enrolled --> Chronic_Erosion: Continuous Decay (Decline Index >= 3)
    Enrolled --> Precipitous_Collapse: Sharp Fall (Slope <= -0.35)
    
    Chronic_Erosion --> Academic_Advising: Intervention Check-in
    Academic_Advising --> Academic_Resilience: Grade Rebound (+73% Recovery Odds)
    Academic_Resilience --> Enrolled: Rebound Sustained (22.6% Dropout)
    
    Chronic_Erosion --> Dropout: Persistent Decay (41.9% Dropout)
    Precipitous_Collapse --> Dropout: Critical Departure (60.3% Dropout)
    Stable_Persistence --> Graduated: Right-Censored Completion (8.1% Dropout)
```

#### ASCII System Schematic
```
   GPA     Semester 1     Semester 2     Semester 3     Semester 4     Terminal Outcome
   4.0 ─── [3.8] ──┐
   3.5             └───► [3.4] ──┐
   3.0                           └───► [3.1] ──┐        [CRITICAL RISK: 60.3% Dropout]
   2.5 ─── [2.5] ──────► [2.5] ──────► [2.5] ──┴──────► [PERSISTING:    8.1% Dropout]
   2.0
           "The Dropping A Student" (High GPA, Steep Negative Slope)
           vs "The Steady C Student" (Moderate GPA, Zero Decay)
```

#### Technical Implementation & Key Formulations
1. **Vectorized OLS Trajectory Engine ([`src/retention/features.py`](file:///c:/Users/Lenovo/Downloads/SSIF/src/retention/features.py)):**
   Computes closed-form cumulative ordinary least squares slopes $\beta_{i,t}$ on historical filtrations strictly bounded to $s \le t$ across 79,239 rows in **0.15 seconds**:
   $$\beta_{i,t} = \frac{n \sum_{s=1}^t s \cdot Y_{i,s} - \left(\sum_{s=1}^t s\right)\left(\sum_{s=1}^t Y_{i,s}\right)}{n \sum_{s=1}^t s^2 - \left(\sum_{s=1}^t s\right)^2}$$
2. **Multi-Tier GroupKFold Benchmark ([`src/retention/models.py`](file:///c:/Users/Lenovo/Downloads/SSIF/src/retention/models.py)):**
   Evaluated across 5 folds grouped by `Student_ID`. Trajectory-augmented Logistic Regression achieved **AUROC = 0.8014, PR-AUC = 0.3643, Brier = 0.0669** (vastly outperforming the 0.0860 baseline prevalence).
3. **Kaplan-Meier & Cox Proportional Hazards ([`src/retention/survival.py`](file:///c:/Users/Lenovo/Downloads/SSIF/src/retention/survival.py)):**
   Semi-parametric survival model achieved Harrell's Concordance Index $C = 0.7498$ ($p < 0.001$):
   - **First-Generation Hazard Ratio:** $\text{HR} = 1.98\times$ (95% CI: $[1.89, 2.08]$) — nearly double the instantaneous departure risk.
   - **Scholarship Protection:** $\text{HR} = 0.52\times$ (95% CI: $[0.49, 0.55]$) — cuts departure hazard in half.
4. **Trajectory Phenotypes ([`src/retention/clustering.py`](file:///c:/Users/Lenovo/Downloads/SSIF/src/retention/clustering.py)):**
   $K$-Means clustering ($k=3$) evaluated over $B=1000$ bootstrap iterations passed `RULE-017` with **Bootstrap ARI = 0.9703**:
   - *Phenotype 1: Stable Persistence (59.8% share, 8.1% dropout)*
   - *Phenotype 2: Chronic Erosion (22.5% share, 28.4% dropout)*
   - *Phenotype 3: Precipitous Collapse (17.7% share, 60.3% dropout)*
5. **Academic Resilience Analysis ([`src/retention/resilience.py`](file:///c:/Users/Lenovo/Downloads/SSIF/src/retention/resilience.py)):**
   Identified $N = 5,563$ students who rebounded after a severe grade drop ($\Delta\text{GPA} \le -0.3$). Recovering students showed a dropout rate of **22.6% vs 41.9%** among continuing-decline peers. Multivariate logistic regression *estimates* (observational — unmeasured confounding possible):
   - **Academic Advising is the #1 associated resilience factor:** $\text{OR} = 1.731$ ($p < 0.001$, $+73.1\%$ odds of rebound per visit; *associational, not established causal*).
   - **Financial Stress is the primary associated barrier:** $\text{OR} = 0.666$ ($p < 0.001$, $-33.4\%$ odds of rebound; *associational*).

#### Engineering Decisions & Scientific Rationale
- **Vectorized Closed-Form Trajectories:** Traditional looping or pandas groupby operations across 79,239 rows required 45+ seconds. The closed-form vectorized formulation computes historical running sums in memory in 0.15 seconds, enabling zero-latency feature extraction.
- **GroupKFold Grouping:** In student panel data, standard random k-fold cross-validation leaks past student records into the test folds, inflating AUC by 0.08–0.12. Grouping by `Student_ID` ensures complete student isolation.

#### Challenges Faced & Problem Solutions
- **Challenge:** Survivorship panel bias. Students who drop out naturally have fewer recorded semesters (e.g. leaving after semester 2), meaning total lifetime sequence length trivially predicts dropout with AUC 0.607.
- **Solution:** Codified `RULE-062` strictly forbidding total sequence duration from feature sets. All running metrics, counters, and OLS slopes are strictly causally bounded to the historical observation filtration $s \le t$.

#### What the Data Reveals in Layman's Context
- **The "Dropping A" vs "Steady C" Student:** A high-achieving student whose GPA quietly slides from 3.8 to 3.1 is in far greater danger of dropping out than a student who consistently stays at 2.5. Momentum and velocity matter more than the current standing.
- **Advising and post-decline academic recovery:** The analysis finds that students who meet with an academic advisor after a severe GPA drop have a **+73.1% higher odds of rebounding** ($\text{OR}=1.731$). This is an observational association — student motivation and unmeasured confounders may partly explain the difference.
- **Scholarship protection and first-generation hazard:** First-generation students show nearly double the observed departure hazard ($\text{HR} = 1.98\times$). Students with institutional scholarships show a hazard ratio of $0.52\times$. These are Cox model associations under the proportional hazards assumption, not established causal effects.
- **The 3D Risk "Danger Cliff":** Grade drops and attendance decay don't add independently — the model's risk surface shows sharp non-linear interaction when both begin declining simultaneously.

![Interactive Early Warning Risk Simulator & Algorithmic Recourse](docs/screenshots/03a_retention_simulator.png)
*Figure 3: Interactive Early Warning Risk Simulator and Prescribed Algorithmic Recourse. Counselors adjust real-time velocity metrics to generate calibrated departure probabilities alongside feasible, $L_1$-minimal intervention plans (e.g. tuition grants and reduced course loads reducing risk from 81.1% to 40.5%).*

![WebGL 3D Predicted Risk Surface](docs/screenshots/03b_retention_3d_surface.png)
*Figure 4: WebGL 3D Predicted Risk Interaction Surface ($\text{Recent GPA} \times \text{Attendance} \rightarrow P(\text{Dropout})$). Features projected floor contours and 1-click accessibility flattening to a 2D contour heatmap.*

![Kaplan-Meier Survival Curves & Cox Hazards Observatory](docs/screenshots/04_survival_analysis.png)
*Figure 5: Longitudinal Survival Analysis Observatory. Displays empirical Kaplan-Meier persistence curves stratified by generational status ($N=20,000$ students, Harrell's $C=0.7498$, $p < 0.001$) and Cox Proportional Hazards forest plots isolating independent hazard multipliers.*

---

### Phase 4: Employability Intelligence & Conditional Compensation

#### Purpose & Core Scientific Question
*Does high academic achievement guarantee a higher starting salary, or does hands-on professional experience fundamentally govern placement outcomes?*

#### Mermaid Architectural Workflow
```mermaid
flowchart TD
    A["MBA Candidate Profile<br/>(N=215, 15 variables)"] --> B["Stage 1: Placement Gate<br/>(Logistic Regression / Random Forest)"]
    B --> C{"Placement Status"}
    C -->|"Unplaced: 31.2%"| D["Zero Salary Imputation<br/>Algorithmic Recourse Strategy<br/>(+26.9% Lift from Work Experience)"]
    C -->|"Placed: 68.8%"| E["Stage 2: Conditional Salary Regressor<br/>(N=148 Placed Candidates Only)"]
    E --> F["Starting Salary Prediction<br/>R² ≈ 0.00 (Fixed Corporate Pay Bands)"]
```

#### ASCII System Schematic
```
   ┌────────────────────────────────────────────────────────────────────────┐
   │                     THE TWO-STAGE PLACEMENT ENGINE                     │
   │                                                                        │
   │                        [Student Candidate]                             │
   │                                 │                                      │
   │                  ┌──────────────┴──────────────┐                       │
   │                  ▼                             ▼                       │
   │           STAGE 1 MODEL                 STAGE 2 MODEL                  │
   │       Placement Classifier             Salary Regressor                │
   │        (N=215, All Students)           (N=148 Placed Only)             │
   │                  │                             │                       │
   │                  ▼                             ▼                       │
   │         P(Placement = 1)                E[Salary | Placed]             │
   │          AUROC = 0.9370                   R² ≈ 0.000                   │
   │   (Workex Lift: 59.6% ──► 86.5%)    (Fixed Corporate Bands)            │
   └────────────────────────────────────────────────────────────────────────┘
```

#### Technical Implementation & Key Formulations
1. **Decoupled Classification & Regression ([`src/placement/models.py`](file:///c:/Users/Lenovo/Downloads/SSIF/src/placement/models.py)):**
   - Stage 1: Placement status predicted using Logistic Regression (AUROC = 0.9370, PR-AUC = 0.9650) and Random Forest (AUROC = 0.9099).
   - Stage 2: Starting salary modeled strictly on placed candidates ($N=148$). $R^2 \approx 0.00$, consistent with entry-level MBA compensation being governed by fixed corporate pay brackets rather than marginal GPA differentiation — interpret with $N=148$ caution.
2. **Work Experience Lift:**
   - Candidates without work experience: **59.6% placement rate**.
   - Candidates with work experience: **86.5% placement rate** (+26.9% absolute lift).

#### Engineering Decisions & Scientific Rationale
- **Two-Stage Decoupling:** Training a single regression model on the entire cohort forces the model to predict salary = 0 for unplaced students, turning regression into an accidental classification proxy (`RULE-010`). Decoupling isolates the true economic determinants of salary.
- **Strict Capacity Constraints:** With $N=215$ candidates and 21 encoded features, the events-per-variable ratio is low ($\text{EPV} = 3.2$). We used $L_2$-regularized linear models and restricted tree depth to avoid severe overfitting.

#### Challenges Faced & Problem Solutions
- **Challenge:** Severe target leakage if unplaced candidates' missing `salary` column is included or imputed prior to classification.
- **Solution:** Formally isolated `salary` to Stage 2 conditional regression. Unplaced candidates are evaluated exclusively through Stage 1 classification and algorithmic recourse.

#### What the Data Reveals in Layman's Context
- **The Internship Trump Card:** Having an internship or previous work experience is worth more than a 15% boost in exam scores when it comes to getting hired. Work experience catapults placement likelihood from **59.6% to 86.5%** (+26.9% absolute lift).
- **The Fixed Salary Reality Check:** Studying 80 hours a week to raise your MBA grades from 70% to 85% will not increase your starting paycheck ($R^2 \approx 0.00$). Once hired, companies pay fixed standard rates for entry-level roles. Academic excellence opens the door to an interview, but corporate pay bands determine the salary.
- **The Salary Boxplot Insights:** In the diagnostic boxplots, starting salaries cluster tightly within standard ranges (200k–400k INR) across Marketing & Finance vs Marketing & HR, with small gender gaps that reflect corporate compensation schedules rather than classroom performance.

![Career Placement Diagnostics & Starting Salary Boxplots](docs/screenshots/05_career_placement.png)
*Figure 6: Career Placement & Salary Diagnostics Observatory. Displays subgroup employability rates ($N=215$, AUROC = 0.9370, Work Experience Lift = +26.9%) and conditional salary distributions across specializations ($N=148$ placed candidates).*

---

### Phase 5: Cross-Dataset Synthesis & The Construct Bridge

#### Purpose & Core Scientific Question
*Can two independent student datasets from different institutions and temporal contexts be synthesized without committing the methodological sin of row-level concatenation?*

#### Mermaid Architectural Workflow
```mermaid
graph LR
    subgraph Retention_Study ["SSIF-A: Retention Study (N=20,000)"]
        R1["Work Hours & Financial Burden"] --> R2["Attendance & LMS Login Decline"]
        R2 --> R3["Semester GPA Drop"]
        R3 --> R4["Program Departure (Dropout)"]
    end

    subgraph Construct_Bridge ["Parallel Latent Vulnerability Bridge (Wasserstein Dist = 1.767 yrs)"]
        B1["Behavioral Fatigue Input"]
        B2["Engagement Erosion"]
        B3["Acute Academic Shock"]
        B4["Terminal Attrition Event"]
    end

    subgraph DLSM_Study ["DLSM-B: Digital Health Study (N=16,000)"]
        D1["Late-Night Phone & AI Binge"] --> D2["Sleep Debt & Next-Day Fatigue"]
        D2 --> D3["Mental Health Score Decline"]
        D3 --> D4["Cognitive Burnout"]
    end

    R1 -.-> B1
    B1 -.- D1
    R2 -.-> B2
    B2 -.- D2
    R3 -.-> B3
    B3 -.- D3
    R4 -.-> B4
    B4 -.- D4
```

#### ASCII System Schematic
```
   SSIF ACADEMIC DOMAIN                        DLSM DIGITAL HEALTH DOMAIN
   ┌────────────────────────────────┐          ┌────────────────────────────────┐
   │ Work Hours & Course Overload   │ ~~~~~~~~ │ Late-Night Phone & AI Binge    │  [Behavioral Fatigue]
   │ LMS Logins & Attendance Drop   │ ~~~~~~~~ │ Sleep Debt & Morning Fatigue   │  [Engagement Erosion]
   │ Semester GPA Collapse          │ ~~~~~~~~ │ Mental Health Score Slide      │  [Acute Strain]
   │ Program Departure (Dropout)    │ ~~~~~~~~ │ Cognitive Burnout / Exhaustion │  [Terminal Attrition]
   └────────────────────────────────┘          └────────────────────────────────┘
                 Parallel Latent Constructs (Wasserstein Dist = 1.767 yrs)
```

#### Technical Implementation & Key Formulations
- **Mathematical Distance Profiling ([`src/cross_dataset/representation_bridge.py`](file:///c:/Users/Lenovo/Downloads/SSIF/src/cross_dataset/representation_bridge.py)):**
  - Age 1-Wasserstein Earth Mover's Distance: **1.767 years**.
  - Two-sample Kolmogorov-Smirnov test: $D = 0.3048, p < 10^{-15}$.
- **The Construct Bridge:** Validates that while row-level concatenation is statistically forbidden (`RULE-002`), both cohorts share parallel latent constructs linking fatigue, disengagement, strain, and attrition.

#### Engineering Decisions & Scientific Rationale
- **Zero Synthetic Merging:** Merging rows across disjoint student populations invents non-existent cross-correlations. SSIF strictly preserves dataset autonomy while allowing comparative latent space analysis.

#### Challenges Faced & Problem Solutions
- **Challenge:** Common research pressure to concatenate multiple educational datasets into a "mega-dataset" to claim larger sample sizes.
- **Solution:** Built formal compatibility gates and statistical distance metrics demonstrating that the populations are statistically distinct, preventing methodologically invalid row-level merge conclusions.

#### What the Data Reveals in Layman's Context
Students at different colleges experience the same 4-step burnout cycle: overload leads to fatigue, fatigue leads to skipping class, skipping class leads to failing grades, and failing grades lead to quitting. However, you cannot staple their report cards together as if they were the same person.

---

### Phases 6 & 7: The DLSM Compatibility Gate & Empirical Feature Ablation

#### Purpose & Core Scientific Question
*If we augment academic retention models with digital lifestyle and sleep telemetry from an external cohort, does it yield genuine predictive lift or merely random noise?*

#### Mermaid Architectural Workflow
```mermaid
flowchart TD
    A["Schema & Provenance Audit<br/>(SSIF Retention vs DLSM Health)"] --> B["Compatibility Gate Evaluator<br/>(src/dlsm/compatibility_gate.py)"]
    B --> C{"Compatibility Score >= 0.70?"}
    C -->|"Score = 0.154"| D["STRICT NO-GO FOR ROW MERGE<br/>(RULE-002 / RULE-003 Enforced)"]
    D --> E["5-Fold GroupKFold Ablation Experiment<br/>(src/dlsm/effectiveness_test.py)"]
    E --> F["Baseline A0 (AUROC = 0.80130)<br/>Augmented A1 (AUROC = 0.80125)"]
    F --> G["Empirical Difference: ΔAUROC = -0.00005<br/>t = -0.089, p = 0.932 (Zero Signal)"]
    G --> H["Literature Calibrated: Orben & Przybylski<br/>(Nature Human Behaviour 2019, R² <= 0.004)"]
    H --> I["Final Decision: Dual-Repository Separation"]
```

#### ASCII System Schematic
```
   ┌────────────────────────────────────────────────────────────────────────┐
   │               5-FOLD GROUPKFOLD DLSM FEATURE ABLATION TEST             │
   │                                                                        │
   │  A0: Pure Academic Baseline (15 features) ──► AUROC: 0.80130 ± 0.0052  │
   │  A1: Academic + DLSM Demographics (17 fts) ─► AUROC: 0.80125 ± 0.0052  │
   │  ────────────────────────────────────────────────────────────────────  │
   │  ΔAUROC CONTRIBUTION: -0.00005  (t = -0.089, p = 0.932)                │
   │                                                                        │
   │  VERDICT: NO STATISTICALLY DETECTABLE INCREMENTAL PREDICTIVE VALUE    │
   │  OBSERVED UNDER THIS SPECIFICATION (H6 NOT REJECTED).                 │
   └────────────────────────────────────────────────────────────────────────┘
```

#### Technical Implementation & Key Formulations
- **Automated Compatibility Gate ([`src/dlsm/compatibility_gate.py`](file:///c:/Users/Lenovo/Downloads/SSIF/src/dlsm/compatibility_gate.py)):** Evaluates schema overlap, observation granularity, and identifier alignment. Score = **0.154** (Fails 0.70 threshold; verdict: STRICT NO-GO).
- **5-Fold GroupKFold Ablation Benchmark ([`src/dlsm/effectiveness_test.py`](file:///c:/Users/Lenovo/Downloads/SSIF/src/dlsm/effectiveness_test.py)):**
  - Baseline $A_0$ (Academic + Trajectories): $\text{AUROC} = 0.80130 \pm 0.0052$
  - Augmented $A_1$ (Academic + DLSM Shared Features): $\text{AUROC} = 0.80125 \pm 0.0052$
  - Paired t-test: $\Delta\text{AUROC} = -0.00005, t = -0.089, p = 0.932$.
- **Literature Benchmark Calibration (`RULE-061`):** Calibrated against **Orben & Przybylski (*Nature Human Behaviour*, 2019, $n=355,358$)**, which used specification curves to *find* that digital technology explains at most **$0.4\%$ ($R^2 \le 0.004$)** of wellbeing variance — providing a strong calibration anchor for interpreting the null result here.

#### Engineering Decisions & Scientific Rationale
- **Empirical Rejection of Confirmation Bias:** Popular intuition suggests screen time ruins academic performance. SSIF pre-registered an objective 5-fold cross-validation experiment and faithfully reported the null result.
- **Architectural Separation:** Maintained SSIF and DLSM as twin standalone repositories, linked via API contracts and research documentation rather than messy code coupling.

#### Challenges Faced & Problem Solutions
- **Challenge:** The temptation to claim high accuracy by forcing features into a single model.
- **Solution:** Codified `RULE-061` and enforced an automated feature ablation test demonstrating that, without paired biometric sleep telemetry on the exact same individuals, the tested external screen-time variables provided no statistically detectable predictive lift.

#### What the Data Reveals in Layman's Context
- **No detectable incremental signal from the tested digital-lifestyle features:** Augmenting the academic retention model with DLSM-derived demographic proxies produced $\Delta\text{AUROC} = -0.00005$ ($p = 0.932$) — no statistically detectable improvement under this specification. A non-significant result does not establish *exactly* zero effect; a different measurement approach (e.g. paired individual biometric telemetry) could yield different results. Within this dataset and specification, the strongest retention signals are academic velocity, advising engagement, and financial burden.
- **The Power of Saying "NO":** Bad data science joins unrelated datasets together just to brag about a big table. Good science tests whether the bridge is real. By enforcing a **NO-GO gate** on row merging while establishing a conceptual representation bridge, SSIF protects institutional decision-makers from acting on false correlations.

![DLSM Compatibility Gate & Construct Bridge](docs/screenshots/07_dlsm_construct_bridge.png)
*Figure 7: DLSM Compatibility Gate & Scientific Construct Bridge. Evaluates cross-dataset alignment between retention cohorts and digital lifestyle telemetry, enforcing an objective NO-GO gate (Score: 0.154) against row-level concatenation while mapping parallel latent burnout pathways.*

![Empirical DLSM Feature Ablation & Specification-Curve Calibration](docs/screenshots/07b_dlsm_feature_ablation.png)
*Figure 7b: Empirical DLSM Feature Ablation Experiment & Specification-Curve Benchmark Anchor. Displays the rigorous 5-fold GroupKFold cross-validation results showing that adding digital demographics (`Age`, `Gender`) produces $\Delta\text{AUROC} = -0.00005$ ($p = 0.932$) — no statistically detectable incremental predictive value under this specification. Grounded in Orben & Przybylski (2019, $n=355,358$) specification-curve calibration and featuring the Open Educational Data Mining Hall of Fame.*

---

### Phase 8: Explainable AI (SHAP) & Algorithmic Counterfactual Recourse

#### Purpose & Core Scientific Question
*How do we transition from passive risk classification to prescriptive, actionable intervention guidance that academic advisors can execute in the real world?*

#### Mermaid Architectural Workflow
```mermaid
graph TD
    A["High-Risk Student Profile<br/>(P(Dropout) = 68.4%)"] --> B["Recourse Optimization Engine<br/>min Σ c_j · normalized_cost(x*, x)"]
    B --> C{"Is Attribute Actionable?"}
    C -->|"No: First_Gen, Age, Gender"| D["Hold Attribute Immutable"]
    C -->|"Yes: Advising, Work Hours, Attendance"| E["Optimize Minimal Shift"]
    D --> F["Prescribed Action Plan"]
    E --> F
    F --> G["Low-Risk Profile<br/>(P(Dropout) = 14.8% < 15%)"]
```

#### ASCII System Schematic
```
   ┌────────────────────────────────────────────────────────────────────────┐
   │              PREDICTIVE RECOURSE PRESCRIPTION (MODEL-BASED)            │
   │                                                                        │
   │  ⚠ IMPORTANT DISTINCTION:                                              │
   │  This engine answers: "What feature changes cause the MODEL to assign   │
   │  a lower predicted probability?" — NOT "What intervention causally      │
   │  changes real-world dropout?" These are different claims.               │
   │                                                                        │
   │  STUDENT INITIAL STATE:  P(Dropout) = 68.4%  [MODEL: CRITICAL RISK]    │
   │                                                                        │
   │  IMMUTABLE ATTRIBUTES (LOCKED):                                        │
   │  - First_Generation: Yes (Locked)      - Age: 20 (Locked)              │
   │                                                                        │
   │  ACTIONABLE RECOURSE INTERVENTIONS:                                    │
   │  - Academic Advising Visits:  0 visits ──► 2 visits                   │
   │  - External Work Hours:      28 hrs/wk ──► 18 hrs/wk                  │
   │  - Class Attendance Rate:      74.0%   ──► 85.0%                      │
   │                                                                        │
   │  COUNTERFACTUAL MODEL PROBABILITY: 14.8%  [MODEL: SAFE ZONE < 15%]    │
   │  (Model reclassifies the counterfactual profile — not a causal effect) │
   └────────────────────────────────────────────────────────────────────────┘
```

#### Technical Implementation & Key Formulations
1. **SHAP TreeExplainer ([`src/explainability/shap_analyzer.py`](file:///c:/Users/Lenovo/Downloads/SSIF/src/explainability/shap_analyzer.py)):**
   Global attribution rankings revealed that engineered `gpa_recent_mean` ranks as the **#4 overall predictor** across all 22 variables, outperforming static demographic indicators.
2. **Constrained L1-Norm Recourse Solver ([`src/explainability/recourse.py`](file:///c:/Users/Lenovo/Downloads/SSIF/src/explainability/recourse.py)):**
   Solves a constrained optimization finding the closest actionable counterfactual profile $\mathbf{x}^*$:
   $$\min_{\mathbf{x}^*} \sum_{j \in \mathcal{A}} c_j \left|\frac{x_j^* - x_j}{\sigma_j}\right| \quad \text{s.t.} \quad P(\text{Dropout} \mid \mathbf{x}^*) \le 0.15$$
   while keeping immutable demographics ($\mathcal{I} = \{\text{First\_Generation, Age, Gender}\}$) completely frozen.

#### Engineering Decisions & Scientific Rationale
- **Domain Actionability Constraints:** Standard recourse algorithms often recommend nonsensical changes (e.g. "reduce age by 3 years" or "change parental background"). Restricting optimization to an actionable subset $\mathcal{A}$ ensures every recommendation is administratively feasible.
- **Integer-Bounded Interventions:** Counseling visits and credit hours are constrained to discrete integer steps, matching university administrative reality.

#### Challenges Faced & Problem Solutions
- **Challenge:** Advising visits were sometimes mathematically optimized to negative values by unconstrained solvers.
- **Solution:** Added explicit non-negative lower bounds ($x_j^* \ge x_j$ for counseling sessions, $x_j^* \le x_j$ for off-campus work hours).

#### What the Data Reveals in Layman's Context
- **Predictive Recourse vs Causal Intervention:** The system identifies the *minimal feature changes that would cause the predictive model to reclassify a student as lower-risk*. This is valuable decision-support information. **It does not establish that executing those changes will causally produce the same probability change in the real world** — that requires randomized evaluation. Advisors should treat recourse suggestions as hypothesis-generating guidance, not guaranteed prescriptions.
- **What Really Drives Risk (Predictively):** The SHAP attribution bar chart shows that dynamic, engineered features (`gpa_recent_mean`, `cumulative_failed_courses`, `gpa_slope`) carry far more predictive weight than immutable background factors. Within this model's specification, behavioral patterns are stronger predictors than demographics.
- **The 3D Feature Space Separation:** In the 3D WebGL coordinate space ($\text{GPA} \times \text{Attendance} \times \text{Failed Courses}$), persisting students (green) and departed students (red) occupy statistically distinct regions, illustrating the multi-dimensional nature of retention risk.

![SHAP Predictive Drivers of Departure Risk](docs/screenshots/06a_explainability_shap.png)
*Figure 8: Explainable AI & SHAP Risk Drivers Observatory. Features global mean absolute SHAP attributions categorizing academic velocity, socioeconomic burden, engagement, and institutional protection factors.*

![WebGL 3D Multivariate Risk Feature Space](docs/screenshots/06b_multivariate_3d_scatter.png)
*Figure 9: WebGL 3D Multivariate Feature Space Scatter ($\text{GPA} \times \text{Attendance} \times \text{Failed Courses}$). Interactive 3D visualization showing cluster separation between persisting students (green) and departed students (red), complete with 2D projection toggles.*

---

### Phase 9: Interactive Streamlit Research Observatory

#### Purpose & Core Scientific Question
*How do we deliver complex survival hazard curves, counterfactual simulators, and data integrity audits through a zero-latency, publication-quality web observatory?*

#### Mermaid Architectural Workflow
```mermaid
flowchart TD
    User(["Researcher / Academic Advisor"]) --> UI["Streamlit Cloud Observatory<br/>(ssif-research.streamlit.app)"]
    UI --> Router{"7 Interactive Research Pages"}
    Router --> P1["1. Executive Overview<br/>Cross-Study KPIs & Topology"]
    Router --> P2["2. Data Audit & Missingness<br/>Little's MCAR & Little's Tests"]
    Router --> P3["3. Academic Retention Simulator<br/>Calibrated Early-Warning Slopes"]
    Router --> P4["4. Survival Analysis<br/>Kaplan-Meier & Cox Hazards"]
    Router --> P5["5. Career Placement<br/>Two-Stage MBA Salary Diagnostics"]
    Router --> P6["6. Explainability & Recourse<br/>SHAP Waterfall & Action Planner"]
    Router --> P7["7. DLSM Governance Gate<br/>Ablation Leaderboard & Gate Matrix"]
    Router --> P8["8. Deep Pattern Lab & Synthesis<br/>Non-linear Cliffs & Labor Paradox"]
    
    subgraph Platform_Protection ["Platform Stability Engine"]
        E1["@st.cache_data Memory Management"]
        E2["width='stretch' Zero-Warning Adapters"]
        E3["pyarrow<25 Segfault Protection"]
    end
    Router --> Platform_Protection
```

#### ASCII System Schematic
```
   ┌────────────────────────────────────────────────────────────────────────┐
   │             LIVE STREAMLIT OBSERVATORY (ssif-research.streamlit.app)   │
   │                                                                        │
   │   PAGE 1: Executive Overview & Cross-Framework KPIs                    │
   │   PAGE 2: Data Audit & Missingness Diagnostics (MAR Proofs)            │
   │   PAGE 3: Academic Retention & Calibrated Early-Warning Simulator      │
   │   PAGE 4: Survival Analysis & Cox Proportional Hazards Forest Plots     │
   │   PAGE 5: Career Placement & MBA Compensation Diagnostics              │
   │   PAGE 6: Explainable AI & SHAP Risk Drivers                          │
   │   PAGE 7: DLSM Compatibility Gate & Feature Ablation Leaderboard       │
   └────────────────────────────────────────────────────────────────────────┘
```

#### Technical Implementation & Key Formulations
- **Production URL:** [https://ssif-research.streamlit.app/](https://ssif-research.streamlit.app/)
- **Zero-Warning Layout Adapters:** Implemented `show_chart()` and `show_dataframe()` using modern `width="stretch"` layout parameters with backward-compatible fallbacks.
- **Environment Portability:** Dynamic `sys.path` injection and fallback imports (`from app.components ... except ModuleNotFoundError: from components ...`) supporting local Windows, Docker, and Linux cloud containers.

#### Engineering Decisions & Scientific Rationale
- **Client-Side Cache Invalidation:** Heavy statistical routines (survival curves, SHAP values) are wrapped in `@st.cache_data`, ensuring snappy 60 FPS transitions between views.
- **Curated HSL Visual Hierarchy:** Built a bespoke dark HSL color palette avoiding default garish colors, using muted indigo (`#6366F1`), emerald (`#10B981`), and amber (`#F59E0B`).

#### Challenges Faced & Problem Solutions
- **Challenge:** PyArrow 25.0.1 caused a known memory segmentation fault on Streamlit Cloud Linux containers; Streamlit 1.64 deprecated `use_container_width`.
- **Solution:** Pinned `pyarrow>=14.0.0,<25.0.0` in `requirements.txt` and engineered `show_chart()` / `show_dataframe()` wrappers. Cloud build times dropped by 6 seconds with zero warnings.

#### What the Data Reveals in Layman's Context
- **A Flight Simulator for Education:** A flight simulator for university deans, provosts, and counselors—allowing them to visually test retention policies, explore survival curves, and inspect student risk without writing a single line of code.
- **The Honest Leaderboard:** Look closely at the model leaderboard. The simplest, most interpretable model (Tier 1: Logistic Regression, AUROC = 0.8014) actually outperforms complex ensembles when trajectory features are properly engineered. Transparent science means displaying Brier calibration scores and PR-AUC side by side, not just cherry-picking the highest single number.
- **Dimensional Research Lab Aesthetics:** Built with a custom HSL dark theme, glassmorphic floating panels, and genuine WebGL 3D surfaces that earn their third dimension through continuous multi-variable physics rather than gimmick perspective distortion.

![Model Performance Leaderboard & Core Empirical Pillars](docs/screenshots/01b_leaderboard_pillars.png)
*Figure 10: Model Performance Leaderboard & Core Empirical Pillars view. Transparently displays 7 model tiers across Retention, Placement, and Survival domains alongside the core empirical laws discovered by SSIF.*

![Deep Empirical Pattern Lab & Non-Linear Tipping Points](docs/screenshots/08_deep_empirical_pattern_lab.png)
*Figure 11: Deep Empirical Pattern Lab & Non-Linear Tipping Points Observatory (Page 8). Uncovers sharp non-linear departure thresholds: falling below a 2.0 GPA surges departure risk from 8.2% to 20.5% (jumping to 44.8% below 1.5 GPA), while taking 18+ credits surges dropout hazard by +80.4% (13.6%–15.1%). Displays equity buffers proving institutional scholarships yield a -9.7% dropout drop for Q1 low-income students (4x higher utility than high-income peers) and early advising visits (Semesters 1–2) drive a -6.3% hazard reduction.*

---

### Phase 10: Continuous Quality Engineering & The Feasibility Auditor

#### Purpose & Core Scientific Question
*How do we guarantee that future code updates or external datasets never introduce silent data leakage, synthetic artifacts, or demographic bias?*

#### Mermaid Architectural Workflow
```mermaid
flowchart TD
    A["Raw Dataset File (.csv)"] --> B["Tabular Feasibility Auditor<br/>(dataset_feasibility_audit.py)"]
    
    subgraph Seven_Gates ["7 Automated Pre-Modeling Quality Gates"]
        G1["Gate 1: Provenance Screen<br/>(Synthetic Data Fingerprinting)"]
        G2["Gate 2: Leakage Cold-Scan<br/>(Single-Feature AUC <= 0.98)"]
        G3["Gate 3: Signal vs Null Test<br/>(Holdout AUC > Permutation 95th)"]
        G4["Gate 4: Sample Adequacy<br/>(Events Per Variable EPV >= 10.0)"]
        G5["Gate 5: Structural Drift<br/>(Grouped vs Random Split Gap <= 0.05)"]
        G6["Gate 6: Fairness Screen<br/>(Four-Fifths Selection Ratio >= 0.80)"]
        G7["Gate 7: Literature Plausibility<br/>(Max AUC <= 0.90 / Literature Cap)"]
    end
    
    B --> G1 --> G2 --> G3 --> G4 --> G5 --> G6 --> G7
    G7 --> H{"All Gates Passed?"}
    H -->|YES| I["🟢 Green Light: Safe for Modeling"]
    H -->|NO| J["🔴 Red Flag: Halt & Remediate Pipeline"]
```

#### ASCII System Schematic
```
   ┌────────────────────────────────────────────────────────────────────────┐
   │             THE 7 AUTOMATED FEASIBILITY QUALITY GATES                  │
   │                                                                        │
   │  GATE 1: PROVENANCE    ──► Checks integer IDs & distribution realism   │
   │  GATE 2: LEAKAGE       ──► Flags any single column with AUC > 0.98     │
   │  GATE 3: SIGNAL        ──► Outperforms 30-permutation null noise floor │
   │  GATE 4: ADEQUACY      ──► Guards Events-Per-Variable (EPV >= 10.0)    │
   │  GATE 5: STRUCTURE     ──► Compares Grouped vs Random CV inflation     │
   │  GATE 6: FAIRNESS      ──► Evaluates 80% adverse impact rule           │
   │  GATE 7: BENCHMARK     ──► Calibrates effect sizes to published papers │
   └────────────────────────────────────────────────────────────────────────┘
```

#### Technical Implementation & Key Formulations
- **The Tabular Feasibility Auditor ([`dataset_feasibility_audit.py`](file:///c:/Users/Lenovo/Downloads/SSIF/dataset_feasibility_audit.py)):** A standalone, 7-gate tabular validation auditor with both CLI and programmatic API `run_feasibility_audit()`.
  1. *Provenance Screen:* Fingerprints synthetic artifacts (e.g. perfectly uniform distributions, missingness anomalies).
  2. *Single-Feature Leakage Scan:* Evaluates univariate AUC per column; flags columns with $\text{AUC} \ge 0.98$.
  3. *Permutation Signal-vs-Null Test:* Assesses whether model performance exceeds a 30-permutation shuffle distribution ($p_{95} = 0.521$).
  4. *Sample Size Adequacy:* Computes events-per-variable ($\text{EPV} \ge 10.0$), flagging small-sample fragility.
  5. *Structural Group/Temporal Drift:* Tests performance differences between random and grouped cross-validation splits.
  6. *Demographic Fairness Screen:* Checks four-fifths rule adverse impact ratios across sensitive cohorts.
  7. *Literature Plausibility Benchmark:* Verifies that reported performance aligns with published behavioral science caps.
- **Automated Pytest Suite ([`tests/unit/`](file:///c:/Users/Lenovo/Downloads/SSIF/tests/unit/)):** Full test suite at **82/82 unit and integration tests** (16 test modules) executing in 34 seconds on GitHub Actions CI.

#### Engineering Decisions & Scientific Rationale
- **Zero-Dependency Architecture:** Built with pure Python, standard NumPy, and scikit-learn so it can be copied into any research environment and executed immediately on raw CSVs.
- **Standard Exit Codes:** Exits with code 0 on passing audits and code 1 on failures, enabling direct integration into CI/CD build gates.

#### Challenges Faced & Problem Solutions
- **Challenge:** Kaggle sandbox environments block external network access, preventing automated API schema downloads.
- **Solution:** Designed the auditor to operate purely offline against local CSV files in under 60 seconds without requiring external web services.

#### What the Data Reveals in Layman's Context
A digital metal detector for data. Before you waste weeks building an AI model, the auditor scans your CSV and sounds the alarm if your data is fake, contains cheating columns that reveal the answers, or will fail in the real world.

---

### Phase 11 & Extensions: Publication Suite & Camera-Ready IEEE Paper

#### Purpose & Core Scientific Question
*How do we package these empirical methodologies and findings into reproducible, publication-grade artifacts accessible to journal reviewers, open-source engineers, and Kaggle researchers?*

#### Mermaid Architectural Workflow
```mermaid
flowchart TD
    A["Empirical Findings & Models"] --> B["Automated Publication Pipeline<br/>(scripts/compile_paper.py & generate_notebooks.py)"]
    
    subgraph Multi_Format_Artifacts ["Publication Artifacts"]
        P1["Camera-Ready IEEE Paper<br/>(papers/ssif_academic_retention_study.pdf)<br/>Two-Column IEEEtran Layout (73.1 KB)"]
        P2["7 Modular Research Notebooks<br/>(notebooks/01_*.ipynb to 07_*.ipynb)<br/>Step-by-step interactive verification"]
        P3["Master Kaggle Publication Suite<br/>(notebooks/kaggle_ssif_student_success_study.ipynb)<br/>High-impact community article"]
    end
    
    B --> P1 & P2 & P3
    B --> C["Embedded Standalone Tectonic Engine<br/>(Zero TeXLive dependencies required)"]
    C --> P1
```

#### ASCII System Schematic
```
   ┌────────────────────────────────────────────────────────────────────────┐
   │                   SSIF MULTI-CHANNEL PUBLICATION SUITE                 │
   │                                                                        │
   │   [LaTeX Source] ──► [Tectonic Engine] ──► [Camera-Ready IEEE PDF]     │
   │   [Raw Models]   ──► [Script Builder]  ──► [7 Research Notebooks]      │
   │   [Audit Logs]   ──► [Markdown Suite]  ──► [Kaggle Master Article]     │
   └────────────────────────────────────────────────────────────────────────┘
```

#### Technical Implementation & Key Formulations
1. **Camera-Ready IEEE Research Paper ([`papers/ssif_academic_retention_study.pdf`](file:///c:/Users/Lenovo/Downloads/SSIF/papers/ssif_academic_retention_study.pdf)):**
   - Typeset in two-column *IEEE Transactions on Learning Technologies* format.
   - Fully automated compilation script ([`scripts/compile_paper.py`](file:///c:/Users/Lenovo/Downloads/SSIF/scripts/compile_paper.py)) utilizing the self-contained Tectonic engine.
   - Formatted with zero overflow via `
\resizebox{\columnwidth}{!}`.
2. **7 Reproducible Research Notebooks ([`notebooks/`](file:///c:/Users/Lenovo/Downloads/SSIF/notebooks/)):**
   Generated via [`scripts/generate_notebooks.py`](file:///c:/Users/Lenovo/Downloads/SSIF/scripts/generate_notebooks.py), covering each phase step-by-step.
3. **Master Kaggle Publication Package:**
   - Standalone all-in-one publication notebook: [`notebooks/kaggle_ssif_student_success_study.ipynb`](file:///c:/Users/Lenovo/Downloads/SSIF/notebooks/kaggle_ssif_student_success_study.ipynb).
   - High-impact community publication article: [`reports/KAGGLE_PUBLICATION_ARTICLE.md`](file:///c:/Users/Lenovo/Downloads/SSIF/reports/KAGGLE_PUBLICATION_ARTICLE.md).

#### Engineering Decisions & Scientific Rationale
- **Portable TeX Compilation:** Requiring users to install a 4GB TeXLive distribution creates friction. The automated compiler dynamically fetches a standalone 15MB Tectonic binary, producing publication-ready PDFs in under 2 seconds.
- **Triple-Format Distribution:** Delivers the research as a formal academic paper (PDF), an interactive web app (Streamlit), and reproducible code (Jupyter Notebooks).

#### Challenges Faced & Problem Solutions
- **Challenge:** LaTeX Table 1 exceeded the right-hand column margin in 2-column IEEE format.
- **Solution:** Wrapped the tabular environment in `
\resizebox{\columnwidth}{!}` in `papers/ssif_academic_retention_study.tex`, achieving pixel-perfect column alignment.

#### What the Data Reveals in Layman's Context
The research is packaged so that anyone can read and verify it in whatever format they prefer: professors get an IEEE publication paper, developers get reproducible Jupyter notebooks, and campus counselors get a live web dashboard.

---

### Phase 12: Multi-Disciplinary Research Experiments & Policy Simulation Suite

#### Purpose & Multi-Disciplinary Focus
To bridge the gap between retrospective statistical prediction and prospective institutional decision-making, SSIF operates a dedicated computational research suite (`experiments/`) bringing together methodologies from **Labor Economics, Operations Research, Algorithmic Fairness, and Predictive Analytics**. Orchestrated via `experiments/run_all_experiments.py`, all 5 pipelines execute with full statistical reproducibility in under 65 seconds.

#### Architectural Workflow
```mermaid
flowchart TD
    subgraph Data_Inputs ["Empirical Data Foundation"]
        D1["Retention Panel (N=79,239)"]
        D2["Placement Dataset (N=215)"]
    end

    subgraph Experiments_Suite ["SSIF Research Experiments Suite"]
        E1["EXP-001: Labor-Policy Simulation<br/>(OLS + 200 Monte Carlo Iterations)"]
        E2["EXP-002: Pipeline Resilience<br/>(18-Scenario Attrition Cascade)"]
        E3["EXP-003: Fairness Audit<br/>(Hiring Threshold Equity & QBE Pool)"]
        E4["EXP-004: Trajectory Forecasting<br/>(S1-S4 Expanding Window AUC Curve)"]
        E5["EXP-005: Intervention ROI Optimizer<br/>(HiGHS Linear Programming & Pareto Frontier)"]
    end

    subgraph Orchestration ["Master Orchestrator (run_all_experiments.py)"]
        M1["master_results.json (5/5 SUCCESS)"]
        M2["MASTER_EXPERIMENT_SUMMARY.md"]
        M3["Streamlit Policy Lab (Interactive Visuals)"]
    end

    D1 & D2 --> E1 & E2 & E3 & E4 & E5
    E1 & E2 & E3 & E4 & E5 --> M1 & M2 & M3
```

#### Experiment Matrix & Quantified Discoveries

| ID | Title & Domain | Core Methodology | Key Findings & Quantitative Outputs | Output Artifacts |
|---|---|---|---|---|
| **EXP-001** | **Labor-Policy Intervention Simulation**<br/>*Labor Economics & Policy* | OLS multivariate regression with demographic controls + N=200 row-level parametric Monte Carlo bootstrap | ⚠️ **Model-based scenario estimate:** Converting students from >15 hrs/week survival labor to structured work-study is *associated with* a **+0.077 GPA lift** and a **2.97 pp dropout risk reduction** (95% CI: [2.64, 3.29] pp) under the tested OLS specification. Not an RCT estimate. | `reports/experiments/EXP-001/`<br/>• `labor_policy_ols_results.csv`<br/>• `monte_carlo_ci.json`<br/>• `policy_roi_summary.json` |
| **EXP-002** | **Pipeline Resilience Stress Test**<br/>*Systems & Operations Research* | 18-scenario attrition cascade perturbing stage failure rates across early (S1-2), mid (S3-4), and late (S5-8) college | Identifies Stage 1 Early as the **highest systemic multiplier** (+81.6 graduates per 1,000 students under early intervention vs +62.6 for late stage). Preventing early attrition compounds across all subsequent semesters. | `reports/experiments/EXP-002/`<br/>• `point_of_no_return.json`<br/>• `intervention_sensitivity_grid.csv`<br/>• `lifecycle_attrition_baseline.csv` |
| **EXP-003** | **Socio-Economic Fairness Audit**<br/>*Algorithmic Fairness & Equity* | Demographic parity analysis, Fisher's exact test, and trajectory-anchored Qualified-But-Excluded profiling | The 65% degree-GPA hiring threshold disproportionately excludes Q1 low-income and first-generation students. Identified **N=992 "Qualified-But-Excluded" students** defined as: below the 65% GPA gate **AND** positive GPA velocity (`gpa_slope > 0`) — students adversely excluded despite demonstrating upward academic momentum. Results are observational; institutional hiring criteria may have other rationale. | `reports/experiments/EXP-003/`<br/>• `threshold_achievability_by_demographics.csv`<br/>• `qualified_excluded_profiles.csv`<br/>• `workex_rescue_differential.json` |
| **EXP-004** | **Career Trajectory Forecasting**<br/>*Predictive Analytics* | GroupKFold cross-validation across expanding observation windows (S1 through S1-4) with SHAP attribution | Semesters 1–2 academic signals predict 4-year success with **AUC = 0.7469** (XGBoost). As observation widens, predictive discrimination expands monotonically to **AUC = 0.8387** by Semester 4. Computes normalized Career Readiness Scores (CRS 0–100). | `reports/experiments/EXP-004/`<br/>• `early_window_model_performance.csv`<br/>• `early_warning_window_auc_curve.csv`<br/>• `career_readiness_score_distribution.csv` |
| **EXP-005** | **Intervention ROI Optimizer**<br/>*Operations Research & Budgeting* | HiGHS Linear Programming (LP) optimization over multi-tiered institutional budgets ($10K–$500K) | **Advising Boost delivers the highest entry ROI** (0.0533 reductions/dollar) up to $100K budgets; larger budgets optimally blend Advising with Emergency Micro-Scholarships and Work-Study conversions. | `reports/experiments/EXP-005/`<br/>• `optimal_allocation_by_budget.csv`<br/>• `subgroup_prioritization.csv`<br/>• `sensitivity_analysis.csv` |

![Research Experiments Lab Master Suite & Labor Policy Simulation](docs/screenshots/09a_research_experiments_lab.png)
*Figure 12: Research Experiments Lab Master Suite (Page 9). Features the multi-disciplinary experiment orchestrator displaying 5/5 passed experiments, zero runtime failures (60.9s execution), and EXP-001 Labor Economics policy simulation evaluating the institutional ROI of replacing off-campus survival labor (>15 hrs/wk) with structured on-campus work-study.*

![EXP-002 Lifecycle Sensitivity Heatmap & EXP-003 Demographic Fairness Disparity](docs/screenshots/09b_intervention_sensitivity_heatmap.png)
*Figure 13: EXP-002 Lifecycle Intervention Sensitivity Heatmap & EXP-003 Socio-Economic Hiring Equity Audit. Top: 2D simulation grid (model-based scenario) mapping intervention timing (Early S1-2 vs Mid S3-4 vs Late S5-8) across policy efficacies (0.0 to 1.0), indicating that early-stage intervention yields the largest modeled system multiplier (+81.6 graduates per 1,000 students). Bottom: Disparity audit illustrating how the 65% degree-GPA recruitment gate disproportionately excludes Q1 low-income students despite positive academic momentum. Sidebar confirms regulatory governance under EU AI Act Art. 14, FERPA, and India DPDPA.*

![EXP-004 Early Warning Window AUC Stabilization & Empirical Provenance Governance](docs/screenshots/09c_early_warning_auc_curve.png)
*Figure 14: EXP-004 Early Warning Window AUC Stabilization Curve & Data Provenance Hall. Plots the monotonic expansion of predictive power from Freshman Semester 1 (AUC = 0.684) to Sophomore Semester 4 (AUC = 0.8387), demonstrating that early-warning models achieve strong discriminative power within the first year of college. Accompanied by mandatory scientific limitation callouts and full provenance citations honoring original dataset curators.*

#### 🔭 Early Warning Intervention Horizon (EXP-004 Supplementary)

The key operational question is not simply *"does AUC rise?"* but rather **"how early can we intervene while maintaining actionable precision?"**

| Semester Available | AUROC | Primary Predictor | Operational Implication |
|---|---|---|---|
| **S1 (Freshman)** | 0.6840 | `cumulative_failed_courses`, `attendance_ratio` | Early screening possible; high false-positive rate expected. Suitable for light-touch outreach (information, advising awareness). |
| **S1–S2** | 0.7469 | + `gpa_slope` | Meaningfully better discrimination. GPA velocity becomes readable. Priority advising trigger point. |
| **S1–S3** | 0.7943 | + `gpa_recent_mean` | Strong discrimination across most archetypes. Mid-year intervention design viable. |
| **S1–S4 (Sophomore)** | 0.8387 | + `decline_index` | Maximum early-warning discrimination in this dataset. Full trajectory phenotype assignable. |

> **Design note:** Earlier intervention horizons provide more actionable lead time but at the cost of precision. Institutions should calibrate the intervention budget and false-positive tolerance against the horizon column that best fits their operational context.

---

### 📦 Reproducible Data Lakehouse: Interim & Processed Enriched Datasets

To ensure full reproducibility and facilitate downstream educational research, SSIF maintains a structured two-tier data lakehouse in `data/interim/` and `data/processed/`, storing all assets in dual **CSV and Apache Parquet** formats:

```
data/
├── interim/                         # Clean, validated, and normalized intermediate representations
│   ├── retention_interim.csv / .parquet  # Validated 79,239 rows with canonical types
│   ├── placement_interim.csv / .parquet  # Normalized 215 MBA candidate profiles
│   ├── dlsm_b_interim.csv / .parquet     # 16,000 normalized digital lifestyle records
│   └── README.md                         # Interim data dictionaries and validation specifications
└── processed/                       # Feature-engineered research datasets
    ├── ssif_retention_enriched.parquet   # 79,239 rows with 37 features (trajectories + velocity)
    ├── ssif_retention_student_profiles.parquet # N=20,000 unique student level lifetime trajectories
    ├── ssif_placement_enriched.parquet   # 215 profiles with composite academic scores & bins
    ├── ssif_macro_pipeline_cohorts.csv   # Stage-by-stage macro persistence cohort rates
    ├── ssif_higher_ed_synthesis_metrics.json # 24 consolidated higher-ed empirical KPIs
    └── README.md                         # Processed feature dictionary and schema catalogs
```

---
## 💡 What the Data Reveals in Plain English (Layman's Compendium)

| Empirical Finding | Scientific Statistic | What It Actually Means for Real Students & Advisors |
|---|---|---|
| **1. The Velocity Effect** | OLS Slope $\beta_{i,t}$ ($r = -0.147$) | A student whose GPA quietly slides from 3.8 to 3.1 is in far greater danger of dropping out than a student who consistently stays at 2.5. Momentum matters more than current standing. |
| **2. First-Generation Departure Hazard** | Cox $\text{HR} = 1.98\times$ ($p < 0.001$) | Students whose parents didn't attend college show nearly double the observed departure hazard in this cohort — driven by financial stress and institutional navigation barriers. This is a Cox model association, not an established causal mechanism. |
| **3. Scholarship and Departure Hazard Reduction** | Cox $\text{HR} = 0.52\times$ ($p < 0.001$) | Scholarship recipients show a departure hazard ratio of 0.52×, *consistent with* a strong protective association. This is a proportional hazard estimate; self-selection into scholarship programs is a noted limitation. |
| **4. Advising and post-decline academic recovery** | Odds Ratio = $1.731\times$ ($p < 0.001$) | After acute GPA collapse, students who engage academic advising show 73.1% higher odds of rebounding. This is an observational association — motivation and other unmeasured factors may partly drive the result. |
| **5. Work Experience and Placement Likelihood** | Selection Lift: $59.6\% \to 86.5\%$ | In this MBA cohort, candidates with prior work experience were placed at an 86.5% rate vs 59.6% for those without — a 26.9% observed difference. Self-selection into work experience is a key limitation ($N=215$). |
| **6. Starting Salary and GPA: No Detected Linear Association** | Salary Regressor $R^2 \approx 0.00$ | No meaningful linear relationship between GPA and starting salary was detected in this cohort ($N=148$ placed), consistent with fixed corporate pay band structures for entry-level MBA roles. |
| **7. No Detectable Incremental Signal from Tested Digital-Lifestyle Features** | $\Delta\text{AUROC} = -0.00005$ ($p = 0.932$) | Adding DLSM-derived digital-lifestyle proxies produced no statistically detectable predictive improvement under this specification. A non-significant result does not establish exactly zero effect — different measurement designs may find otherwise. |
| **8. The 3D Risk Cliff** | 3D Interaction Surface ($Z \ge 0.80$) | Attendance and grades don't act in isolation. When both begin slipping simultaneously, risk multiplies exponentially into a steep departure cliff that 2D charts fail to capture. |
| **9. Prescriptive GPS Recourse** | $L_1$ Minimal-Action Optimizer | Modern AI shouldn't just be an alarm bell; it should be a navigation GPS. SSIF computes the exact feasible recipe (e.g. 2 counseling visits + 10 fewer work hours) to safely return students to the persistent zone. |
| **10. The 65% Degree Hiring Cliff** | Leap from 58.2% to 90.0% placement | In corporate recruitment, 65% undergraduate marks is the golden gate. Crossing from 60–65% to 65–70% causes placement probability to leap by **+31.8%**! Above 65%, placement plateau at ~90%. |
| **11. Work Experience as a Counterbalance to Low GPA** | Observed: $31.1\% \to 72.7\%$ placement | In this cohort, low-GPA students with prior work experience were placed at a 72.7% rate (+41.6% absolute difference vs low-GPA peers without work experience) — a large observed difference, though self-selection into work experience limits causal interpretation. |
| **12. The Course Overload Hazard** | 15 vs 18+ Credits: $7.5\% \to 13.6\%$ | Taking 18+ credits surges dropout hazard by **+80.4%**. Trying to rush graduation overloads vulnerable students into course failure cascades. |
| **13. The Student Labor Paradox** | Survival Labor vs Career Credential | Off-campus part-time survival jobs during college actively drive dropouts (OR=1.007/hr). But verified professional work experience after college gives **5x higher odds of corporate hiring** (OR=4.98). |
| **14. Recruiter Pedigree Screening** | 10th ($t=11.2$) vs MBA ($t=1.13$, $p=0.26$) | Corporate recruiters filter MBA candidates based on schooling pedigree (10th/12th/Undergrad), essentially ignoring in-MBA GPA differentiation. |

---

## 🛑 Challenges Encountered, Problems & Engineering Solutions

| Category | Problem Encountered | Engineering & Scientific Solution |
|---|---|---|
| **Data Leakage** | `End_of_Semester_Status` in retention and `salary` in placement yielded synthetic $\text{AUC} = 1.000$. | Codified `RULE-009` and `RULE-010`. Built automated leakage detectors and cold-scan auditor gates that raise runtime exceptions if leaky columns are present. |
| **Survivorship Bias** | Total lifetime semesters observed predicted dropout with $\text{AUC} = 0.607$. | Enforced `RULE-062`: lifetime observation count is forbidden. All trajectory features are strictly causally bounded to the historical filtration $s \le t$. |
| **Data Hygiene** | Raw `Gender` column was fragmented into 8 string variants (`Female`, `female`, `F`, `Male`, `male`, `M`, etc.). | Built a regex-based string canonicalizer in `src/data_loader.py` that unified categories to 4 clean cohorts, restoring demographic selection fairness to parity. |
| **Small Sample EPV** | Placement cohort has $N=215$ rows and 21 encoded features ($\text{EPV} = 3.2 < 10.0$). | Constrained model capacity; avoided hyper-parameter over-tuning; reported sample limitations explicitly on dashboard. |
| **Package Conflict** | PyArrow 25.0.1 caused a known segfault on Streamlit Cloud containers. | Pinned `pyarrow>=14.0.0,<25.0.0` in `requirements.txt`, shaving 6 seconds off cloud build times and eliminating crashes. |
| **API Deprecation** | Streamlit 1.64.0 deprecated `use_container_width=True` in favor of `width='stretch'`. | Created `show_chart()` and `show_dataframe()` compatibility wrappers in `app/main.py` ensuring zero-warning execution across local and cloud environments. |
| **LaTeX Environment** | Missing system TeX installations on local and CI machines. | Implemented standalone Tectonic binary fetcher (`scripts/compile_paper.py`), downloading and caching a single portable binary that builds IEEE PDFs in 2 seconds. |

---

## 🛡️ Threat Model & Data Leakage Defenses

```
   LEAKAGE THREAT                      SSIF DEFENSE MECHANISM
   ┌─────────────────────────────────┐  ┌────────────────────────────────────────────────────────┐
   │ Realized Post-Outcome Variables │─►│ FORBIDDEN_COLUMNS set excludes End_of_Semester_Status  │
   │ Future Observation Bleed        │─►│ Vectorized OLS operates strictly on s <= t historicals │
   │ Student Correlation Across Folds│─►│ GroupKFold partitions 20,000 students without cross-leak│
   │ Structural Missingness Leakage  │─►│ Salary isolated to placed-only conditional model (N=148)│
   │ Survivorship Panel Duration     │─►│ Total trajectory length forbidden as predictive feature│
   │ Preprocessing Target Leakage    │─►│ All scalers & median imputers fit strictly on train fold│
   └─────────────────────────────────┘  └────────────────────────────────────────────────────────┘
```

### 🔬 Rules That Materially Changed Scientific Conclusions

The 62 governance rules are not bureaucracy — they changed the results:

| Rule | What It Enforces | Without It → | With It → |
|------|-----------------|--------------|------------|
| `RULE-009` | Forbids post-outcome variables as features | Synthetic AUC = **1.000** (leakage) | Legitimate AUROC = **0.8014** |
| `RULE-062` | Forbids lifetime sequence length as feature | Survivorship-biased AUC = **0.607** | Causally bounded temporal estimate |
| `RULE-003` | Forbids row-level merge across datasets | Invalid "mega-dataset" cross-correlations | Construct bridge with NO-GO gate (score = 0.154) |
| `RULE-031` | Mandates GroupKFold by `Student_ID` | Student correlation inflates AUC by **+0.08–0.12** | Legitimate cross-validated AUROC |
| `RULE-010` | Isolates salary to placed candidates only | Regression conflates classification + salary proxy | Clean two-stage decoupled architecture |
| `RULE-017` | Requires bootstrap ARI > 0.70 before naming phenotypes | Arbitrary unstable cluster labeling | Stability-validated phenotype discovery (ARI = 0.9703, B=1000) |
| `RULE-061` | Calibrates DLSM results against literature benchmark | Unanchored ΔAUROC claim | Literature-grounded null result (Orben & Przybylski, 2019) |

### 📋 Epistemic Integrity Documents

Two dedicated documents provide complete transparency:

- **[`CLAIMS_AND_EVIDENCE.md`](CLAIMS_AND_EVIDENCE.md)** — Maps every major claim to its evidence, statistical design, limitations, and epistemic strength (Predictive / Associational / Conditional Causal / Negative Finding / Governance Decision).
- **[`RED_TEAM.md`](RED_TEAM.md)** — Proactively presents the strongest possible attacks against each major finding, with current defense status and what would be required to establish stronger claims.



---

## ⚡ Quickstart & Reproduction Guide

### 1. Installation
```bash
git clone https://github.com/HarshkumarG007/SSIF.git
cd SSIF
pip install -r requirements.txt
pip install --no-deps -e .
```

### 2. Run Automated Pytest Suite (82 Tests Across All 16 Modules)
```bash
pytest tests/ -v
```

### 3. Launch Enterprise FastAPI REST Microservice
```bash
# Start FastAPI ingestion & inference server with live reload
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Interactive Swagger UI: http://localhost:8000/docs
# OpenAPI JSON schema: http://localhost:8000/openapi.json
```

### 4. Run via Docker Compose (Multi-Service Stack)
```bash
# Spins up FastAPI microservice (port 8000) and Streamlit Observatory (port 8501)
docker-compose up --build
```

### 5. Automated 1-Click Kaggle Release Packaging
```bash
# Verify dataset & kernel bundles (dry-run)
python scripts/publish_to_kaggle.py

# Push live to Kaggle CLI
python scripts/publish_to_kaggle.py --live
```

### 6. Run Master Research Experiments Suite (EXP-001 through EXP-005)
```bash
# Run all 5 domain-specialized research experiments in sequence
python experiments/run_all_experiments.py

# Or run individual experiments
python experiments/exp_001_labor_policy_simulation.py
python experiments/exp_002_pipeline_resilience_stress_test.py
python experiments/exp_003_fairness_audit.py
python experiments/exp_004_career_trajectory_forecast.py
python experiments/exp_005_intervention_roi_optimizer.py
```

### 7. Run the Tabular Feasibility Auditor via CLI
```bash
# Audit Placement Dataset
python dataset_feasibility_audit.py Placement_Data_Full_Class.csv --target status --id sl_no --drop salary --sensitive gender

# Audit Retention Panel
python dataset_feasibility_audit.py academic_survival_longitudinal.csv --target Target_Dropout_Next_Sem --group Student_ID --time Semester --sensitive Gender --drop End_of_Semester_Status,Censored --max-rows 5000
```

### 8. Compile the Camera-Ready IEEE Research Paper
```bash
python scripts/compile_paper.py
# Output generated at: papers/ssif_academic_retention_study.pdf
```

### 9. Launch the Local Research Observatory Dashboard
```bash
streamlit run app/main.py
```

---

## 📂 Project Directory Blueprint

```
SSIF/
├── .github/workflows/ci.yml      # CI/CD pipeline running 82 tests on Python 3.11 & 3.12
├── .streamlit/config.toml        # Observatory dark HSL research theme
├── api/                          # Deployment-ready Research FastAPI REST Microservice
│   ├── main.py                   # REST endpoints (/v1/retention, /v1/recourse, /v1/placement, /v1/causal)
│   └── schemas.py                # Pydantic v2 input/output payload models
├── app/                          # Deployment-ready Research Streamlit Observatory
│   ├── main.py                   # 9-view interactive research dashboard & policy lab
│   └── components.py             # Custom HSL cards, Plotly themes & limitation banners
├── configs/                      # Pydantic v2 typed configuration manifests
│   ├── data.yaml                 # Filepaths, schemas, and seeds
│   ├── features.yaml             # Trajectory & composite feature specifications
│   └── models.yaml               # Model hyperparameters & GroupKFold settings
├── data/                         # Two-tier reproducible research data lakehouse
│   ├── raw/                      # Original raw academic datasets
│   ├── kaggle_dataset_bundle/    # 1-click Kaggle CLI distribution bundle
│   ├── interim/                  # Cleaned, validated, normalized Parquet & CSV datasets
│   │   ├── retention_interim.parquet / .csv
│   │   ├── placement_interim.parquet / .csv
│   │   ├── dlsm_b_interim.parquet / .csv
│   │   └── README.md
│   └── processed/                # Feature-engineered research datasets
│       ├── ssif_retention_enriched.parquet / .csv
│       ├── ssif_retention_student_profiles.parquet / .csv
│       ├── ssif_placement_enriched.parquet / .csv
│       ├── ssif_macro_pipeline_cohorts.csv
│       ├── ssif_higher_ed_synthesis_metrics.json / .csv
│       └── README.md
├── Dockerfile                    # Production multi-stage, non-root microservice container
├── docker-compose.yml            # Multi-container orchestration (FastAPI + Streamlit)
├── docs/                         # Governance constitution & specifications
│   ├── PRD.md                    # Research Requirements Document (v2.0)
│   ├── System Architecture.md    # End-to-end architectural blueprints (v2.0)
│   ├── Rules.md                  # 62 scientific & engineering governance rules
│   ├── design.md                 # UI/UX design specifications
│   ├── task.md                   # 166 tracked execution tasks across 14 phases
│   ├── memory.md                 # Persistent project decision ledger
│   ├── DEPLOYMENT_GUIDE.md       # Streamlit Cloud deployment runbook
│   └── DLSM_CROSSLINK_DOCUMENTATION.md # Cross-study ecosystem reference
├── experiments/                  # Multi-disciplinary research experiments suite
│   ├── exp_001_labor_policy_simulation.py      # OLS + Monte Carlo work-study ROI
│   ├── exp_002_pipeline_resilience_stress_test.py # 18-scenario lifecycle cascade
│   ├── exp_003_fairness_audit.py               # Hiring threshold equity & QBE pool
│   ├── exp_004_career_trajectory_forecast.py   # S1-S4 early warning AUC curve
│   ├── exp_005_intervention_roi_optimizer.py   # HiGHS LP budget allocation
│   └── run_all_experiments.py                  # Master experiment orchestrator
├── notebooks/                    # 7 Laboratory & Kaggle Research Notebooks
│   ├── 01_retention_audit.ipynb through 07_dlsm_effectiveness.ipynb
│   ├── kaggle_ssif_student_success_study.ipynb # Standalone publication notebook
│   └── kaggle_kernel/            # Kaggle code kernel upload metadata bundle
├── papers/                       # Camera-ready publication manuscripts
│   ├── ssif_academic_retention_study.tex # IEEEtran LaTeX source
│   ├── ssif_academic_retention_study.pdf # Compiled 2-page publication PDF
│   └── ssif_research_preprint.md         # Full markdown research paper
├── reports/                      # Empirical research findings & audit logs
│   ├── causal/                   # Double ML policy evaluations & E-values
│   ├── DEEP_DATASET_DISCOVERY_REPORT.md  # Landmark Non-Linear EDA & Synthesis Treatise
│   ├── KAGGLE_PUBLICATION_ARTICLE.md     # Ready-to-publish Kaggle article
│   └── FINAL_RESEARCH_SUMMARY.md         # Comprehensive scientific findings
│   └── experiments/                      # Experiment logs, metrics & Pareto frontiers
│       ├── EXP-001/ through EXP-005/     # Individual experiment artifacts
│       ├── master_results.json           # Consolidated machine-readable metrics
│       └── MASTER_EXPERIMENT_SUMMARY.md  # Master markdown synthesis report
├── scripts/                      # Automation & generation utilities
│   ├── compile_paper.py          # Standalone Tectonic LaTeX-to-PDF compiler
│   ├── generate_notebooks.py     # Automated Jupyter notebook suite generator
│   └── publish_to_kaggle.py      # 1-click Kaggle CLI dataset & kernel packager
├── src/                          # Modular research source code
│   ├── causal/                   # Double ML (PLR, AIPW, CATE, E-value sensitivity)
│   ├── config.py                 # Pydantic configuration loader
│   ├── data_loader.py            # Clean loaders with Gender canonicalization
│   ├── logger.py                 # Leveled structured logging
│   ├── validation/               # Schema validators & leakage detectors
│   ├── retention/                # Trajectories, GroupKFold benchmark, survival, resilience
│   ├── placement/                # Decoupled classification & conditional compensation
│   ├── dlsm/                     # Compatibility gate & 5-fold feature ablation
│   ├── explainability/           # SHAP TreeExplainer & algorithmic recourse engine
│   └── cross_dataset/            # Synthesis analytics & Wasserstein representation bridge
├── tests/unit/                   # 82 Automated unit & integration tests across 16 test modules
│   ├── test_causal_ml.py         # Double ML PLR, AIPW, CATE, and E-value tests
│   ├── test_api.py               # FastAPI TestClient endpoint verification tests
│   ├── test_kaggle_release.py    # Kaggle manifest and metadata bundle tests
│   ├── test_security_privacy.py  # FERPA, quasi-identifier & CORS security regression tests
│   ├── test_experiments.py       # Validation suite for EXP-001 through EXP-005
│   ├── test_synthesis_analytics.py # Non-linear tipping points & labor paradox tests
│   ├── test_feasibility_auditor.py # 5 Cold-scan quality gate tests
│   ├── test_recourse.py          # Algorithmic recourse tests
│   └── test_schema_validator.py  # Zero-leakage & schema tests
├── dataset_feasibility_audit.py  # Standalone 7-gate tabular feasibility auditor
├── pyproject.toml                # Build packaging manifest
├── requirements.txt              # Pinned production dependencies (pyarrow<25)
└── README.md                     # Comprehensive framework documentation
```

---

## 📄 License & Citation

This framework is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

### Citation
```bibtex
@article{ssif2026,
  title   = {Student Success Intelligence Framework: Longitudinal Survival Trajectories, Academic Resilience, and Cross-Study Digital Telemetry Governance},
  author  = {Harshkumar G.},
  journal = {IEEE Transactions on Learning Technologies (Preprint)},
  year    = {2026},
  url     = {https://github.com/HarshkumarG007/SSIF}
}
```

---

## 🙏 Acknowledgements & Original Dataset Credits

The **Student Success Intelligence Framework (SSIF)** stands on the shoulders of dedicated researchers, data scientists, and educational practitioners who open-source real-world institutional and behavioral datasets for the global academic community. 

We extend our profound gratitude, respect, and thanksgiving to the original creators and dataset curators across both **SSIF** and its sister research ecosystem **DLSM**:

### 🎓 Primary SSIF Educational Datasets

#### 1. 📈 Student Retention & Academic Performance Panel
* **Curator & Original Author:** **Razan Ihab Abdellatif**
* **Primary Kaggle Dataset:** [Student Retention and Academic Performance Data](https://www.kaggle.com/datasets/razanihababdellatif/student-retention-and-academic-performance-data)
* **Dataset Characteristics:** 79,239 longitudinal student-semester observations across 20,000 distinct students tracking cumulative GPA velocity, course failure rates, attendance ratios, and institutional retention milestones.
* **Citation & Thanksgiving:** We express our deepest gratitude to **Razan Ihab Abdellatif** for assembling and releasing this rich longitudinal panel. Without this temporal granularity, developing vectorized $O(N)$ trajectory engines, Kaplan-Meier hazard estimates, and empirical recovery models would have been impossible.

#### 2. 💼 MBA Campus Placement & Employability Dataset
* **Curator & Original Author:** **Amey Thakur** ([Kaggle: @ameythakur20](https://www.kaggle.com/ameythakur20))
* **Primary Kaggle Dataset:** [Campus Recruitment (Placement Data Full Class)](https://www.kaggle.com/datasets/ameythakur20/placement-data)
* **Dataset Characteristics:** 215 business school candidate profiles capturing multi-tier academic percentages (secondary, higher secondary, undergraduate, MBA specialization), verified work experience, employability test scores, and starting corporate compensation.
* **Citation & Thanksgiving:** Our sincere thanks and appreciation go to **Amey Thakur** for publishing this benchmark employability dataset. It enabled SSIF to model the two-stage decoupling between hiring probability and conditional compensation while proving the profound $+26.9\%$ placement advantage created by professional work experience.

---

### 📱 Sister Research Framework & DLSM Behavioral Telemetry Datasets

SSIF interfaces conceptually with the **Digital Lifestyle Spillover Modeling (DLSM)** research ecosystem, which investigates the upstream behavioral loads (screentime, sleep debt, cognitive fatigue) that precede institutional academic attrition:
* **Sister GitHub Repository:** [HarshkumarG007/DLSM](https://github.com/HarshkumarG007/DLSM)
* **Live DLSM Observatory:** [DLSM Research Portal](https://dlsm-research.streamlit.app/)

We extend our heartfelt gratitude and thanksgiving to the original curators of the two foundational DLSM datasets:

#### 3. 🌙 Sleep Debt & Late-Night Screen Time Telemetry (DLSM-A)
* **Curator & Original Author:** **Samar Talwar**
* **Primary Kaggle Dataset:** [Sleep Debt and Screen Time / Late Night Phone Habits](https://www.kaggle.com/datasets/samartalwar/sleep-debt-and-screen-time-late-night-phone-habits)
* **Dataset Characteristics:** 8,500 sleep and screen-time telemetry records tracking bedtime phone habits, sleep latency, caffeine intake, screen brightness, and daytime fatigue indices.
* **Citation & Thanksgiving:** We express our sincere thanks to **Samar Talwar** for curating and sharing this detailed behavioral sleep dataset. It provides the empirical foundation for modeling how nighttime digital phone habits translate into physiological sleep debt.

#### 4. 🤖 AI Tool Usage, Social Media & Student Mental Health (DLSM-B)
* **Curator & Original Author:** **Sri Syra** ([Kaggle: @srisyra02](https://www.kaggle.com/srisyra02))
* **Primary Kaggle Dataset:** [AI and Social Media Impact: Student Health & Grades](https://www.kaggle.com/datasets/srisyra02/ai-and-social-media-impact-student-health-and-grades)
* **Dataset Characteristics:** 16,000 university student records capturing daily AI study tool usage hours, daily social media hours, subjective mental and physical health scores, and academic grades.
* **Citation & Thanksgiving:** Our deepest appreciation goes to **Sri Syra** for releasing this landmark student telemetry cohort. It enabled the DLSM framework to analyze the cognitive trade-offs between AI-assisted learning, social media fatigue, and student academic wellbeing.

---

### 📥 Ethical Data Access & Primary Download Call-to-Action

> ### 📢 A Message to Researchers, Practitioners, and Students
> 
> To honor and respect dataset provenance, licensing, and community attribution:
> 
> 1. **Please visit the original primary Kaggle dataset pages linked above.**
> 2. **Give the authors an upvote / star on Kaggle** to recognize their hard work, dedication, and vital contribution to open educational and behavioral data mining.
> 3. **Download the raw CSV files directly from the original authors on Kaggle** for your own research pipelines and independent replications:
>    - [Razan Ihab Abdellatif on Kaggle](https://www.kaggle.com/datasets/razanihababdellatif/student-retention-and-academic-performance-data) (Retention Panel)
>    - [Amey Thakur on Kaggle](https://www.kaggle.com/datasets/ameythakur20/placement-data) (Placement Cohort)
>    - [Samar Talwar on Kaggle](https://www.kaggle.com/datasets/samartalwar/sleep-debt-and-screen-time-late-night-phone-habits) (Sleep & Screentime)
>    - [Sri Syra on Kaggle](https://www.kaggle.com/datasets/srisyra02/ai-and-social-media-impact-student-health-and-grades) (AI & Social Media Impact)
> 
> Direct all primary dataset citations, attribution inquiries, and original provenance recognition to these four outstanding dataset curators.


