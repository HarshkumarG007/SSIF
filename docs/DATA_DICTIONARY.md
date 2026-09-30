# SSIF Comprehensive Data Dictionary & Schema Specification

**Student Success Intelligence Framework (SSIF)**  
**Version:** 2.0.0  
**Standards Compliance:** FERPA (20 U.S.C. § 1232g), EU AI Act Annex III, India DPDP Act (2023), Scikit-Learn Strict Leakage Protocol  

---

## 1. Overview & Institutional Data Architecture

The Student Success Intelligence Framework operates over two distinct institutional data panels with structurally separated governance boundaries:

1. **SSIF-A: Longitudinal Academic Persistence Panel (`academic_survival_longitudinal.csv`)**  
   - **Sample Size:** 79,239 student-semester records across 20,000 unique students ($N=20,000$).
   - **Structure:** Multi-period longitudinal observational panel (1 to 8 semesters per student).
   - **Primary Objective:** Calibrated early warning of prospective academic departure (`Target_Dropout_Next_Sem`) and causal evaluation of institutional financial/advising interventions.

2. **SSIF-B: Cross-Sectional Employability Cohort (`Placement_Data_Full_Class.csv`)**  
   - **Sample Size:** 215 MBA candidates ($N=215$, 148 placed, 67 unplaced).
   - **Structure:** Cross-sectional post-graduate institutional outcomes.
   - **Primary Objective:** Regularized employability diagnostic classification and conditional post-graduation compensation analysis.

> [!IMPORTANT]
> **Boundary Isolation Gate (RULE-001 & RULE-002):**  
> Direct row-merging, joint cross-training, or pooling between Dataset A and Dataset B is strictly prohibited by the `DLSMCompatibilityGate`. They represent non-overlapping populations and distinct institutional contexts.

---

## 2. Dataset A: Academic Persistence Panel (`academic_survival_longitudinal.csv`)

### 2.1 Column Summary Matrix

| Variable Name | Storage Type | Domain / Range | Role in Pipeline | Missing Rate | FERPA / Privacy Level |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Student_ID` | `string` (object) | `STU_00001` .. `STU_20000` | Group Identifier (CV) | 0.0% | Pseudonymized Identifier |
| `Age` | `int64` | [17, 35] | Static Demographic Feature | 0.0% | Protected Demographic |
| `Gender` | `string` (object) | `{"Female", "Male", "Non-Binary"}` | Fairness Audit / Covariate | 0.0% | Sensitive Characteristic |
| `First_Generation` | `int64` | `{0, 1}` | Socioeconomic Covariate | 0.0% | Protected Background |
| `Family_Income` | `float64` | [0.0, 250000.0] | Socioeconomic Covariate | 4.55% (3,604) | Highly Confidential |
| `Household_Size` | `int64` | [1, 10] | Socioeconomic Covariate | 0.0% | Confidential |
| `Housing_Status` | `string` (object) | `{"On-Campus", "Off-Campus", "With Family"}` | Environmental Feature | 0.0% | Operational Metadata |
| `Scholarship` | `int64` | `{0, 1}` | Policy Treatment Leaver | 0.0% | Institutional Award Record |
| `Tuition_Base` | `int64` | [5000, 35000] | Institutional Parameter | 0.0% | Financial Record |
| `Semester` | `int64` | [1, 8] | Temporal Sequence Index | 0.0% | Academic Progress Anchor |
| `Course_Load` | `int64` | [3, 21] | Academic Intensity Feature | 0.0% | Enrolled Credit Hours |
| `Work_Hours` | `float64` | [0.0, 60.0] | Structural Stressor Feature | 0.0% | Employment Burden |
| `Emergency_Expense` | `float64` | [0.0, 5000.0] | Financial Shock Feature | 0.0% | Acute Vulnerability |
| `Sem_GPA` | `float64` | [0.00, 4.00] | Performance History Feature | 0.0% | Academic Record |
| `Attendance` | `float64` | [0.0, 100.0] | Engagement History Feature | 0.0% | LMS/Classroom Record |
| `LMS_Logins` | `float64` | [0.0, 300.0] | Engagement History Feature | 1.09% (867) | Telemetry Record |
| `Advising_Visits` | `int64` | [0, 15] | Intervention / Support Feature | 0.0% | Institutional Support Log |
| `Failed_Courses` | `int64` | [0, 6] | Academic Shock Feature | 0.0% | Academic Standing Record |
| `Financial_Stress` | `float64` | [1.0, 5.0] | Psychosocial Survey Feature | 0.0% | Self-Reported Survey |
| `Target_Dropout_Next_Sem` | `int64` | `{0, 1}` | **Primary Binary Target** | 0.0% | Administrative Outcome |
| `End_of_Semester_Status` | `string` (object) | `{"Enrolled", "Dropout", "Graduated"}` | **FORBIDDEN (Leakage Gate)** | 0.0% | Current Semester Outcome |
| `Censored` | `int64` | `{0, 1}` | **Survival Target / Censoring** | 0.0% | Survival Event Indicator |

### 2.2 In-Depth Semantic Definitions & Governance Constraints

- **`Student_ID`:** Unique institutional synthetic key. Strictly used to group longitudinal folds via `GroupKFold(groups=Student_ID)` (RULE-004) to prevent intra-student cross-split data contamination.
- **`Family_Income`:** Annual household income in USD. Contains 4.55% missingness. Diagnosed as Missing At Random (MAR); imputed strictly using training fold median or modeled with an explicit missingness indicator `Family_Income_is_missing` (RULE-007).
- **`Semester`:** Monotonically increasing enrollment semester index. Crucial for causal time-series splits: trajectory metrics for semester $t$ are computed strictly using history where $\tau \le t$ (RULE-009).
- **`Target_Dropout_Next_Sem`:** Binary indicator ($y=1$ if the student departs the institution prior to completing semester $t+1$; $y=0$ if the student persists or graduates). Overall panel departure prevalence is ~12.4%.
- **`End_of_Semester_Status` (RULE-010 Strict Stop):** Administrative resolution for current semester $t$. **Strictly prohibited as a predictor feature.** Inclusion causes 100% label leakage (AUROC $\to 1.0000$) because a status of `"Dropout"` perfectly reveals $y=1$. Enforced automatically by `check_retention_leakage()`.
- **`Censored` (RULE-010 Strict Stop):** Right-censoring indicator for survival analysis ($C=1$ if student is still enrolled or graduated at study termination without departure; $C=0$ if departure event was observed). Forbidden in classification feature sets.

---

## 3. Dataset B: Placement Cohort (`Placement_Data_Full_Class.csv`)

### 3.1 Column Summary Matrix

| Variable Name | Storage Type | Domain / Range | Role in Pipeline | Missing Rate | Governance & EPV Note |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `sl_no` | `int64` | [1, 215] | Candidate Row ID | 0.0% | Dropped during modeling |
| `gender` | `string` (object) | `{"M", "F"}` | Fairness Audit / Covariate | 0.0% | Sensitive characteristic |
| `ssc_p` | `float64` | [40.89, 89.00] | Secondary School Percentage (10th) | 0.0% | Quasi-identifier (binned) |
| `ssc_b` | `string` (object) | `{"Central", "Others"}` | School Board Category | 0.0% | Secondary educational board |
| `hsc_p` | `float64` | [37.00, 97.70] | Higher Secondary Percentage (12th) | 0.0% | Quasi-identifier (binned) |
| `hsc_b` | `string` (object) | `{"Central", "Others"}` | School Board Category | 0.0% | High school educational board |
| `hsc_s` | `string` (object) | `{"Commerce", "Science", "Arts"}` | Pre-University Specialization | 0.0% | Academic stream |
| `degree_p` | `float64` | [50.00, 91.00] | Undergraduate Degree Percentage | 0.0% | Key academic predictor |
| `degree_t` | `string` (object) | `{"Comm&Mgmt", "Sci&Tech", "Others"}` | Undergraduate Discipline | 0.0% | Technical vs management |
| `workex` | `string` (object) | `{"Yes", "No"}` | Professional Experience Flag | 0.0% | Highest-leverage feature |
| `etest_p` | `float64` | [50.00, 98.00] | Employability Test Percentage | 0.0% | Cognitive / aptitude score |
| `specialisation` | `string` (object) | `{"Mkt&Fin", "Mkt&HR"}` | MBA Concentration Track | 0.0% | Corporate placement track |
| `mba_p` | `float64` | [51.21, 77.89] | MBA Degree Cumulative Percentage | 0.0% | Academic predictor |
| `status` | `string` (object) | `{"Placed", "Not Placed"}` | **Primary Classification Target** | 0.0% | Placed=1 (N=148), Not Placed=0 (N=67) |
| `salary` | `float64` | [200000, 940000] | **Conditional Regression Target** | 31.16% (67) | **FORBIDDEN in classification** |

### 3.2 In-Depth Semantic Definitions & Statistical Limits

- **Sample Size Limitation ($N=215$):**  
  With only 67 events in the minority class ("Not Placed"), the Events Per Variable (EPV) limit restricts classification models to at most 6 degrees of freedom (RULE-025) to prevent over-parameterization.
- **`status`:** Mapped to binary indicator: $y=1$ if `"Placed"`, $y=0$ if `"Not Placed"`. Baseline placement rate is 68.84% (majority class baseline AUROC = 0.5000, Accuracy = 68.84%).
- **`salary` (Structural Missingness & Leakage Gate):**  
  Annual compensation offered in INR. Exactly 67 rows have `NaN` because unplaced students receive no salary offer.
  - **Leakage Rule (RULE-010):** `salary` is strictly prohibited in placement classification. Its presence indicates 100% label leakage ($salary > 0 \implies Placed$).
  - **Regression Rule (RULE-026):** Salary regression is conducted exclusively on the subset $status == "Placed"$ ($N=148$).
- **Quasi-Identifier Anonymization (SEC-02):**  
  To prevent student deanonymization via combinations of high-precision grades, percentages (`ssc_p`, `hsc_p`, `mba_p`) are binned into 5% intervals before public export.

---

## 4. Engineered Feature Dictionary

### 4.1 Longitudinal Trajectory Features (`src/retention/features.py`)

All trajectory features for student $i$ at semester $t$ are derived exclusively from records $\{\tau \le t\}$:

| Feature Name | Type | Mathematical Formulation | Epistemic Intent |
| :--- | :--- | :--- | :--- |
| `n_prior_semesters` | `int` | $t$ | Student academic age / cohort persistence |
| `is_single_semester`| `int` | $\mathbb{I}(t == 1)$ | Cold-start flag for first-term entrants |
| `gpa_slope` | `float` | $\frac{\sum (\tau - \bar{\tau})(GPA_\tau - \overline{GPA})}{\sum (\tau - \bar{\tau})^2}$ | Ordinary Least Squares linear trajectory trend |
| `gpa_velocity` | `float` | $GPA_t - GPA_{t-1}$ | Immediate 1-semester momentum shift |
| `gpa_volatility` | `float` | $\text{StdDev}(GPA_{1 \dots t})$ | Performance instability index |
| `gpa_recent_mean` | `float` | $\text{Mean}(GPA_{\max(1, t-1) \dots t})$ | Short-term smoothed baseline |
| `attendance_slope`| `float` | $\frac{\sum (\tau - \bar{\tau})(\text{Att}_\tau - \overline{\text{Att}})}{\sum (\tau - \bar{\tau})^2}$ | Engagement decay or recovery vector |
| `attendance_delta`| `float` | $\text{Att}_t - \text{Att}_{t-1}$ | Acute disengagement shock indicator |
| `lms_slope` | `float` | $\text{OLS slope of LMS logins}$ | Digital engagement trend |
| `lms_delta` | `float` | $\text{LMS}_t - \text{LMS}_{t-1}$ | Platform activity change |
| `cumulative_failed_courses` | `int` | $\sum_{\tau=1}^t \text{Failed}_\tau$ | Accumulating academic deficit burden |
| `cumulative_advising_visits`| `int` | $\sum_{\tau=1}^t \text{Advising}_\tau$ | Cumulative institutional touchpoints |
| `decline_index` | `int` | Count of consecutive drops in GPA | Acute spiral early detection |
| `recovery_index`| `int` | Count of consecutive gains in GPA | Resilience and bounce-back measure |

### 4.2 Employability Composite Features (`src/placement/features.py`)

| Feature Name | Formulation | Interpretation |
| :--- | :--- | :--- |
| `academic_progression` | $hsc\_p - ssc\_p$ | High school academic growth trajectory |
| `degree_deviation` | $degree\_p - \overline{degree\_p}$ | Performance relative to cohort undergraduate baseline |
| `composite_academic_score` | $0.2 \cdot ssc + 0.3 \cdot hsc + 0.5 \cdot degree$ | Weighted long-term academic standing |

---

## 5. Missing Data & Imputation Protocols

1. **Pre-Outcome Separation:** Imputation parameters (e.g., median, scaling coefficients) are computed **strictly within the training fold** during cross-validation (`Pipeline` or `GroupKFold` loop). No test-fold information is ever used to calculate median or mean replacements.
2. **Missingness-as-a-Feature (Optional Flag):**  
   For datasets with informative non-response (e.g. `Family_Income`), the pipeline supports generating explicit indicator variables:
   $$\text{Income\_Missing}_i = \mathbb{I}(\text{Family\_Income}_i = \text{NaN})$$
   allowing models to distinguish between low-income families and non-disclosing families without bias.
3. **Cold-Start Trajectory Imputation:**  
   For first-semester students ($t=1$), trajectory derivatives (`gpa_slope`, `gpa_velocity`) are structurally undefined. The pipeline imputes these features to neutral default `0.0` and activates the binary flag `is_single_semester = 1`, preserving model calibration across both freshmen and seniors.
