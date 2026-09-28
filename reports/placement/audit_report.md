# Academic Placement (Employability) Dataset Audit Report
**Dataset:** `Placement_Data_Full_Class.csv` (SSIF-B)  
**Generated:** 2026-09-29  
**Evaluation:** Pre-modeling Data Quality & Governance Audit (RULE-001, RULE-006, RULE-009)  

## 1. Executive Summary
```text
Dataset: SSIF-B: Academic Placement (Employability)
  Rows: 215
  Columns: 15
  Missing cells: 67 (2.08%)
  Duplicate rows: 0
  Unique sl_no: 215
  [WARNING] Quality warnings:
    - Column 'salary' has 31.2% missing values (HIGH)
    - Column 'salary' has 10.1% IQR outliers — investigate
    - N=215 — SMALL DATASET. All results are exploratory only.
    - Class balance: Placed=68.8%, Not Placed=31.2%
    - Salary: 67 structurally missing (all 'Not Placed') — model salary on N=148 placed students ONLY
```

## 2. Target Distributions
- **Classification Target (`status`):**
  - `Placed`: 148 (68.84%)
  - `Not Placed`: 67 (31.16%)
- **Regression Target (`salary`):**
  - Observed for 148 placed students
  - Median salary: INR 265,000 (Mean: INR 288,655)
  - Structurally unobserved for 67 unplaced students

## 3. Missingness Mechanism Analysis
```text
=== Missingness Report: SSIF-B: Placement Cohort ===
Total Rows: 215 | Missing Cells: 67 (2.08%)

Column Diagnostics:
  - salary: 67 missing (31.16%) | Diagnosis: Structural/MNAR | Strategy: Filter subset for placed candidates when predicting salary; do not impute.
    Evidence: 100% of missing salary corresponds to unplaced candidates. Structural missingness by design.
```

## 4. Column Statistical Profiles
| Column         | Type    |   Missing | Missing%   |   Unique | Mean       | Std       | Min       | Median     | Max       | Skew   | Constant?   | Outliers%   |
|:---------------|:--------|----------:|:-----------|---------:|:-----------|:----------|:----------|:-----------|:----------|:-------|:------------|:------------|
| sl_no          | int64   |         0 | 0.0%       |      215 | 108.000    | 62.209    | 1.00      | 108.000    | 215.00    | 0.00   | OK          | —           |
| gender         | object  |         0 | 0.0%       |        2 | —          | —         | —         | —          | —         | —      | OK          | —           |
| ssc_p          | float64 |         0 | 0.0%       |      103 | 67.303     | 10.827    | 40.89     | 67.000     | 89.40     | -0.13  | OK          | —           |
| ssc_b          | object  |         0 | 0.0%       |        2 | —          | —         | —         | —          | —         | —      | OK          | —           |
| hsc_p          | float64 |         0 | 0.0%       |       97 | 66.333     | 10.898    | 37.00     | 65.000     | 97.70     | 0.16   | OK          | 3.7%        |
| hsc_b          | object  |         0 | 0.0%       |        2 | —          | —         | —         | —          | —         | —      | OK          | —           |
| hsc_s          | object  |         0 | 0.0%       |        3 | —          | —         | —         | —          | —         | —      | OK          | —           |
| degree_p       | float64 |         0 | 0.0%       |       89 | 66.370     | 7.359     | 50.00     | 66.000     | 91.00     | 0.24   | OK          | 0.5%        |
| degree_t       | object  |         0 | 0.0%       |        3 | —          | —         | —         | —          | —         | —      | OK          | —           |
| workex         | object  |         0 | 0.0%       |        2 | —          | —         | —         | —          | —         | —      | OK          | —           |
| etest_p        | float64 |         0 | 0.0%       |      100 | 72.101     | 13.276    | 50.00     | 71.000     | 98.00     | 0.28   | OK          | —           |
| specialisation | object  |         0 | 0.0%       |        2 | —          | —         | —         | —          | —         | —      | OK          | —           |
| mba_p          | float64 |         0 | 0.0%       |      205 | 62.278     | 5.833     | 51.21     | 62.000     | 77.89     | 0.31   | OK          | —           |
| status         | object  |         0 | 0.0%       |        2 | —          | —         | —         | —          | —         | —      | OK          | —           |
| salary         | float64 |        67 | 31.2%      |       45 | 288655.405 | 93457.452 | 200000.00 | 265000.000 | 940000.00 | 3.57   | OK          | 10.1%       |

## 5. Data Leakage & Feature Integrity Check
- **Leakage Audit Status:** PASSED
- **Salary as Feature for Placement:** Strictly forbidden (salary only exists after placement).
- **Clean Baseline Predictors (12 features):** `['gender', 'ssc_p', 'ssc_b', 'hsc_p', 'hsc_b', 'hsc_s', 'degree_p', 'degree_t', 'workex', 'etest_p', 'specialisation', 'mba_p']`

## 6. Actionable Preprocessing Recommendations
1. **Sample Size Awareness:** N=215 is an exploratory sample. Use repeated StratifiedKFold (k=5, 10 repeats) and regularization to prevent overfitting.
2. **Dual Model Architecture:** Model 1 = Binary Classifier (`Placed` vs `Not Placed`); Model 2 = Salary Regressor trained exclusively on placed candidates (`N=148`).
3. **Categorical Encoding:** One-hot encode MBA specialization, degree stream, and work experience.