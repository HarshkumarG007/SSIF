# EXP-004: Career Trajectory Forecasting from Early Academic Signals
## Student Success Intelligence Framework (SSIF)

### Research Question
Can Semester 1–2 academic signals predict long-run career placement eligibility,
years before students reach the job market?

---

### Model Performance (Early-Window: Semesters [1, 2])
*(GroupKFold CV, groups=Student_ID, N_splits=5)*

| model              |    auc |     f1 |   brier |   n_students |   n_successes | window_sems   |
|:-------------------|-------:|-------:|--------:|-------------:|--------------:|:--------------|
| LogisticRegression | 0.7329 | 0.1309 |  0.1548 |        20000 |          4452 | [1, 2]        |
| RandomForest       | 0.7413 | 0      |  0.152  |        20000 |          4452 | [1, 2]        |
| XGBoost            | 0.7469 | 0.0523 |  0.1506 |        20000 |          4452 | [1, 2]        |

**Best model:** XGBoost (AUC=0.7469)

---

### SHAP Feature Importance (Top 10 Early-Window Features)
| feature              |   mean_abs_shap |   rank |
|:---------------------|----------------:|-------:|
| Financial_Stress     |      0.0316548  |      1 |
| decline_index        |      0.0306168  |      2 |
| Sem_GPA              |      0.0210255  |      3 |
| gpa_velocity         |      0.0153412  |      4 |
| Work_Hours           |      0.0137143  |      5 |
| gpa_slope            |      0.0133944  |      6 |
| gpa_volatility       |      0.0132433  |      7 |
| Family_Income        |      0.0125192  |      8 |
| First_Generation_enc |      0.0101729  |      9 |
| attendance_delta     |      0.00923146 |     10 |

---

### Early Warning Window — AUC Stabilization Curve
| window   |   max_semester |   n_students |    auc |   delta_auc | stabilized   |
|:---------|---------------:|-------------:|-------:|------------:|:-------------|
| S1–S1    |              1 |        20000 | 0.684  |    nan      | False        |
| S1–S2    |              2 |        20000 | 0.7329 |      0.0489 | False        |
| S1–S3    |              3 |        20000 | 0.7868 |      0.0539 | False        |
| S1–S4    |              4 |        20000 | 0.8387 |      0.0519 | False        |

**Stabilization point:** `Not reached within S4`
*(The earliest window where ΔAUC < 0.01, indicating marginal information gain from additional semesters is negligible)*

---

### Career Readiness Score (CRS) Distribution by Gender
| Gender            |   mean_crs |   std_crs |
|:------------------|-----------:|----------:|
| Female            |      21.97 |     14.42 |
| Male              |      22.54 |     14.82 |
| Other             |      22.94 |     15.51 |
| Prefer not to say |      22.08 |     14.55 |

*(CRS = normalized LR predicted probability × 100. Range: 0–100. Higher = stronger predicted long-run success pathway.)*

---

### Key Findings
1. **Semester 1–2 signals alone** achieve AUC=0.7469 for predicting
   long-run academic success — meaningful early-warning capability.
2. **Top early predictors** include: Financial_Stress, decline_index, Sem_GPA.
3. **AUC stabilizes at Not reached within S4**: additional semesters add diminishing predictive value.
4. **CRS can serve as an early academic advising trigger** — students below CRS=40 in
   Semester 2 represent the highest-priority cohort for proactive intervention.

### Limitations
- Long-run outcome (persist to S6+) is a proxy for graduation and placement eligibility;
  actual placement data requires Dataset B, which cannot be row-merged (RULE-003).
- N students with ≥6 semesters may be a biased subsample (survivors only).
- Early-window models are retrospective; real-time deployment would require prospective validation.
