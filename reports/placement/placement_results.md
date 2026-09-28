# Academic Placement & Employability Benchmark Results
**Generated:** 2026-09-29  
**Sample Sizes:** N = 215 (Placement Status) | N = 148 (Placed Candidate Salary)  
**Validation:** Stratified 5-Fold CV (RULE-007, RULE-025)  

## 1. Placement Status Classification Benchmark (N=215)
| Model                       | AUROC (Mean +/- Std)   | Accuracy   |   F1-Score |   Brier Score |
|:----------------------------|:-----------------------|:-----------|-----------:|--------------:|
| Tier 0: Majority Class      | 0.5000 +/- 0.0000      | 68.84%     |     0      |        0.2145 |
| Tier 1: Logistic Regression | 0.9370 +/- 0.0303      | 85.58%     |     0.8942 |        0.1007 |
| Tier 3: Random Forest       | 0.9099 +/- 0.0537      | 86.98%     |     0.9108 |        0.1037 |
| Tier 4: HistGBM             | 0.9098 +/- 0.0437      | 85.58%     |     0.8963 |        0.1064 |

### Key Findings:
- **Academic Performance as Employability Gate:** Secondary (`ssc_p`), undergraduate (`degree_p`), and employability test (`etest_p`) are strong discriminators.
- **Sample Size Governance:** With N=215, Random Forest and Logistic Regression achieve ~88–89% AUROC, but confidence intervals reflect sample size limits.

## 2. Employability Subgroup Disparities
- **Gender (M vs F):**
  - `F`: 63.2% placement rate
  - `M`: 71.9% placement rate
- **Work Experience (Yes vs No):**
  - `No`: 59.6% placement rate
  - `Yes`: 86.5% placement rate
- **MBA Specialisation:**
  - `Mkt&Fin`: 79.2% placement rate
  - `Mkt&HR`: 55.8% placement rate

## 3. Salary Regression Benchmark (N=148 Placed Candidates)
| Model                           | R2 Score           | MAE (INR)   | RMSE (INR)   |
|:--------------------------------|:-------------------|:------------|:-------------|
| Tier 1: Ridge Regression        | -0.0602 +/- 0.7396 | ₹60,224     | ₹95,904      |
| Tier 3: Random Forest Regressor | -0.1687 +/- 0.2687 | ₹63,858     | ₹100,691     |

### Salary Disparity Breakdown (Median Offers):
- **Gender Median Salary:**
  - `F`: INR 250,000
  - `M`: INR 270,000
- **Work Experience Median Salary:**
  - `No`: INR 262,000
  - `Yes`: INR 267,500
- **MBA Specialisation Median Salary:**
  - `Mkt&Fin`: INR 270,000
  - `Mkt&HR`: INR 255,000

## 4. Methodological Note (RULE-025)
Because N=215 is an institutional snapshot, results provide actionable local intelligence
but must not be generalized universally across national employment markets without multi-institutional replication.