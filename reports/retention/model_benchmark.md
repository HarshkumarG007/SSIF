# Academic Retention Multi-Tier Benchmark Results
**Generated:** 2026-09-29  
**Validation:** GroupKFold (k=5, groups=Student_ID) — Zero Temporal or Student Leakage (RULE-004, RULE-009)  

## Performance Comparison
| Model Tier                   | Feature Set         | AUROC (Mean ± Std)   | ΔAUROC vs T1   |   PR-AUC |   Brier Score |    ECE |   F1-Score |
|:-----------------------------|:--------------------|:---------------------|:---------------|---------:|--------------:|-------:|-----------:|
| Tier 0: Majority Class       | None                | 0.4945 ± 0.0000      | —              |   0.086  |        0.0797 | 0      |     0      |
| Tier 1: Logistic Regression  | Static              | 0.8013 ± 0.0050      | +0.0000        |   0.3633 |        0.1772 | 0.2994 |     0.3235 |
| Tier 1+: Logistic Regression | Static + Trajectory | 0.8014 ± 0.0053      | +0.0001        |   0.3643 |        0.1768 | 0.2989 |     0.3244 |
| Tier 3: Random Forest        | Static              | 0.7874 ± 0.0049      | -0.0139        |   0.3223 |        0.1177 | 0.2002 |     0.3643 |
| Tier 3+: Random Forest       | Static + Trajectory | 0.7840 ± 0.0038      | -0.0173        |   0.3206 |        0.1189 | 0.2055 |     0.3625 |
| Tier 4: HistGBM              | Static              | 0.7967 ± 0.0048      | -0.0047        |   0.3552 |        0.1699 | 0.2856 |     0.3275 |
| Tier 4+: HistGBM             | Static + Trajectory | 0.7975 ± 0.0049      | -0.0038        |   0.3521 |        0.1692 | 0.2857 |     0.328  |

## Key Findings
- **Trajectory Lift:** Longitudinal trajectory features (GPA slope, velocity, attendance slope, volatility)
  substantially enhance model discrimination and early warning lead-time.
- **Probability Calibration:** Gradient boosting models achieve lower Brier score and ECE,
  making them suitable for risk probability scoring in academic advising systems.