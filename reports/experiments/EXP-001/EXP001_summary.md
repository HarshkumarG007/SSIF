# EXP-001: Labor-Policy Intervention Simulation
## Student Success Intelligence Framework (SSIF)

### Research Question
What is the net semester-GPA and dropout-risk benefit of converting students from
unstructured survival labor (>15 hrs/week off-campus) to institutional work-study
(≤10 hrs/week, on-campus structured)?

---

### Dataset & Cohort
- **Total student-semesters analyzed:** 79,239
- **High-hours cohort (>15 hrs/week):** 43,268 student-semesters
- **Intervention scenario:** Reduce Work_Hours → 10 hrs, Financial_Stress → ×0.8

---

### OLS Model: GPA Impact of Work Hours
| Parameter | Estimate | p-value |
|-----------|----------|---------|
| Work_Hours coefficient | -0.0027 GPA points/hr | 0.0000 |
| Interpretation | **statistically significant** | — |

> A coefficient of -0.0027 means each additional work hour per week is associated
> with a 0.0027-point change in semester GPA, controlling for financial stress,
> scholarship, attendance, failed courses, gender, and first-generation status.

---

### Counterfactual Simulation (Bootstrap N=200)
| Metric | Mean | 95% CI |
|--------|------|--------|
| GPA lift under intervention | +0.077 pts | [0.073, 0.080] |
| Dropout risk reduction | −2.97 pp | [2.64, 3.29] |

---

### Policy ROI Analysis
| Metric | Value |
|--------|-------|
| Eligible students (high-hours cohort) | 13,776 |
| Expected dropouts prevented | 408.8 |
| Total program cost | $44,083,200 |
| Cost per retained student-semester | $107,847.05 |
| Tuition revenue protected | $1,839,405 |
| **Net institutional ROI** | **$-42,243,795** |
| **ROI ratio** | **0.04×** |

---

### Limitations
- Observational data: OLS estimates reflect correlations, not randomized effects.
- "Financial stress relief" factor of 0.8 is a conservative assumption; actual
  relief may be higher or lower depending on wage rates and family need.
- N=43,268 high-hours rows span repeated student observations (within-student correlation
  handled via cluster-robust SEs by Student_ID).
- ROI calculation uses NCES 2023 national averages; local costs may differ significantly.

### References
- NCES 2023 Federal Work-Study Program Statistics: https://nces.ed.gov/programs/digest/
- Darolia (2014): Working (and studying) day and night. *Journal of Labor Economics*, 32(3).
- Bound & Turner (2011): Coming to college: Cohort size and college enrollment. *AEJ: Applied*.
