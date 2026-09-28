# Academic Persistence (Retention) Dataset Audit Report
**Dataset:** `academic_survival_longitudinal.csv` (SSIF-A)  
**Generated:** 2026-09-29  
**Evaluation:** Pre-modeling Data Quality & Governance Audit (RULE-001, RULE-009, RULE-014)  

## 1. Executive Summary
```text
Dataset: SSIF-A: Academic Persistence (Retention)
  Rows: 79,239
  Columns: 22
  Missing cells: 4,471 (0.26%)
  Duplicate rows: 0
  Unique Student_ID: 20,000
  [WARNING] Quality warnings:
    - Column 'Family_Income' has 4.5% missing values (moderate)
    - Column 'Tuition_Base' has 19.1% IQR outliers — investigate
    - Column 'LMS_Logins' has 1.1% missing values (moderate)
    - Class imbalance: dropout rate = 8.7% — use class_weight='balanced' or SMOTE (training only)
    - 3,404 students have only 1 semester — trajectory features will be NaN for these
```

## 2. Target Variable & Panel Dynamics
- **Primary Target:** `Target_Dropout_Next_Sem` (0 = Persisted, 1 = Dropped out)
- **Overall Dropout Rate:** 8.73% (6,917 events across 79,239 observations)
- **Cohort Size:** 20,000 unique students observed across semesters 1 to 8
- **Realized Status (`End_of_Semester_Status`):**
  - `Enrolled`: 70,354 (88.79%)
  - `Dropped_Out`: 6,917 (8.73%)
  - `Graduated`: 1,968 (2.48%)

## 3. Missingness Mechanism Analysis
```text
=== Missingness Report: SSIF-A: Retention Panel ===
Total Rows: 79,239 | Missing Cells: 4,471 (0.26%)

Column Diagnostics:
  - Family_Income: 3,604 missing (4.55%) | Diagnosis: MAR | Strategy: Fit MedianImputer or IterativeImputer on training folds ONLY (RULE-007, RULE-008).
    Evidence: Statistically associated with: First_Generation, Scholarship, Semester
  - LMS_Logins: 867 missing (1.09%) | Diagnosis: MAR | Strategy: Fit MedianImputer or IterativeImputer on training folds ONLY (RULE-007, RULE-008).
    Evidence: Statistically associated with: Household_Size, Attendance
```

## 4. Column Statistical Profiles
| Column                  | Type    |   Missing | Missing%   |   Unique | Mean      | Std       | Min      | Median    | Max       | Skew   | Constant?   | Outliers%   |
|:------------------------|:--------|----------:|:-----------|---------:|:----------|:----------|:---------|:----------|:----------|:-------|:------------|:------------|
| Student_ID              | object  |         0 | 0.0%       |    20000 | —         | —         | —        | —         | —         | —      | OK          | —           |
| Age                     | int64   |         0 | 0.0%       |       19 | 19.492    | 2.139     | 17.00    | 19.000    | 37.00     | 1.34   | OK          | 1.8%        |
| Gender                  | object  |         0 | 0.0%       |        8 | —         | —         | —        | —         | —         | —      | OK          | —           |
| First_Generation        | int64   |         0 | 0.0%       |        2 | 0.328     | 0.469     | 0.00     | 0.000     | 1.00      | 0.73   | OK          | —           |
| Family_Income           | float64 |      3604 | 4.5%       |      296 | 61518.384 | 39661.167 | 3000.00  | 52000.000 | 488000.00 | 2.11   | OK          | 4.7%        |
| Household_Size          | int64   |         0 | 0.0%       |        6 | 3.188     | 1.333     | 1.00     | 3.000     | 6.00      | 0.20   | OK          | —           |
| Housing_Status          | object  |         0 | 0.0%       |        3 | —         | —         | —        | —         | —         | —      | OK          | —           |
| Scholarship             | int64   |         0 | 0.0%       |        2 | 0.278     | 0.448     | 0.00     | 0.000     | 1.00      | 0.99   | OK          | —           |
| Tuition_Base            | int64   |         0 | 0.0%       |        3 | 24775.111 | 10794.899 | 15000.00 | 25000.000 | 45000.00  | 0.95   | OK          | 19.1%       |
| Semester                | int64   |         0 | 0.0%       |        8 | 3.146     | 1.946     | 1.00     | 3.000     | 8.00      | 0.73   | OK          | —           |
| Course_Load             | int64   |         0 | 0.0%       |        5 | 14.889    | 2.307     | 9.00     | 15.000    | 21.00     | 0.01   | OK          | 1.8%        |
| Work_Hours              | float64 |         0 | 0.0%       |      401 | 16.936    | 10.724    | 0.00     | 16.500    | 40.00     | 0.21   | OK          | —           |
| Emergency_Expense       | float64 |         0 | 0.0%       |     2629 | 200.497   | 724.401   | 0.00     | 0.000     | 3998.00   | 3.69   | OK          | —           |
| Sem_GPA                 | float64 |         0 | 0.0%       |      329 | 2.691     | 0.423     | 0.44     | 2.700     | 3.96      | -0.32  | OK          | 1.9%        |
| Attendance              | float64 |         0 | 0.0%       |      366 | 85.082    | 6.416     | -2.00    | 85.200    | 105.00    | -2.67  | OK          | 0.5%        |
| LMS_Logins              | float64 |       867 | 1.1%       |      104 | 66.376    | 12.731    | 12.00    | 66.000    | 116.00    | -0.01  | OK          | 0.8%        |
| Advising_Visits         | int64   |         0 | 0.0%       |        4 | 0.478     | 0.672     | 0.00     | 0.000     | 3.00      | 1.26   | OK          | 0.9%        |
| Failed_Courses          | int64   |         0 | 0.0%       |        6 | 0.652     | 0.786     | 0.00     | 0.000     | 5.00      | 1.13   | OK          | 2.5%        |
| Financial_Stress        | float64 |         0 | 0.0%       |      160 | 2.578     | 1.818     | 1.00     | 2.100     | 18.60     | 1.85   | OK          | 2.9%        |
| Target_Dropout_Next_Sem | int64   |         0 | 0.0%       |        2 | 0.087     | 0.282     | 0.00     | 0.000     | 1.00      | 2.92   | OK          | —           |
| End_of_Semester_Status  | object  |         0 | 0.0%       |        3 | —         | —         | —        | —         | —         | —      | OK          | —           |
| Censored                | int64   |         0 | 0.0%       |        2 | 0.140     | 0.347     | 0.00     | 0.000     | 1.00      | 2.07   | OK          | —           |

## 5. Data Leakage & Feature Integrity Check
- **Leakage Audit Status:** PASSED
- **Target Contamination Risk:** 0 forbidden features detected
- **Clean Baseline Predictors (18 features):** `['Age', 'Gender', 'First_Generation', 'Family_Income', 'Household_Size', 'Housing_Status', 'Scholarship', 'Tuition_Base', 'Semester', 'Course_Load', 'Work_Hours', 'Emergency_Expense', 'Sem_GPA', 'Attendance', 'LMS_Logins', 'Advising_Visits', 'Failed_Courses', 'Financial_Stress']`

## 6. Actionable Preprocessing Recommendations
1. **GroupKFold Strategy:** Group by `Student_ID` (k=5) to prevent multi-semester student data leakage (RULE-014).
2. **Imputation:** Impute `Family_Income` and `LMS_Logins` inside the CV pipeline (fit on train fold only) using MedianImputer (RULE-007).
3. **Class Imbalance:** Apply `class_weight='balanced'` or calibrated decision thresholds to accommodate the 8.73% dropout prevalence.
4. **Trajectory Handling:** For 3,404 single-semester students, flag missing slope features with an indicator or use static fallbacks.