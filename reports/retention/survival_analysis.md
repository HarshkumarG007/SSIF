# Academic Retention: Longitudinal Survival & Hazard Analysis
**Generated:** 2026-09-29  
**Sample:** 20,000 Students | 6,917 Dropout Events (34.6%) | 13,083 Censored (65.4%)  
**Evaluation:** Kaplan-Meier Non-Parametric Estimator + Cox Proportional Hazards (RULE-009, RULE-016)  

## 1. Overall Model Discrimination: Harrell's C-index = `0.7498`
A concordance index of 0.75 indicates strong discriminative capacity to rank student time-to-dropout.

## 2. Cox Proportional Hazards — Hazard Ratio Summary
| Covariate        |   Coefficient |   Hazard_Ratio |   HR_Lower_95 |   HR_Upper_95 |      p_value |     z_score |
|:-----------------|--------------:|---------------:|--------------:|--------------:|-------------:|------------:|
| Financial_Stress |   0.203965    |       1.22625  |      1.2133   |      1.23934  | 3.26306e-310 |  37.6501    |
| Sem_GPA          |  -0.909848    |       0.402585 |      0.381437 |      0.424907 | 1.72621e-239 | -33.0469    |
| First_Generation |   0.685538    |       1.98484  |      1.89121  |      2.0831   | 3.61996e-170 |  27.8065    |
| Scholarship      |  -0.650782    |       0.521638 |      0.490475 |      0.554781 | 3.02501e-95  | -20.7065    |
| Family_Income    |  -5.99768e-06 |       0.999994 |      0.999993 |      0.999995 | 1.7427e-36   | -12.6151    |
| Attendance       |  -0.0124465   |       0.987631 |      0.984505 |      0.990766 | 1.40535e-14  |  -7.69587   |
| Gender           |  -0.0226814   |       0.977574 |      0.932395 |      1.02494  | 0.347473     |  -0.939502  |
| Age              |  -0.000393942 |       0.999606 |      0.988742 |      1.01059  | 0.943674     |  -0.0706531 |

### Key Epidemiological Interpretations
1. **First-Generation Status (HR = 1.98, 95% CI: [1.89, 2.08], p < 0.001):** First-generation college students experience **nearly double the rate of dropout** at any semester compared to continuing-generation peers.
2. **Scholarship Buffer (HR = 0.52, 95% CI: [0.49, 0.55], p < 0.001):** Institutional scholarship support **reduces dropout hazard by 48%**, acting as a critical protective factor.
3. **Academic GPA (HR = 0.40, 95% CI: [0.38, 0.42], p < 0.001):** Each 1.0 GPA increment **reduces the instantaneous dropout hazard by 60%**.
4. **Financial Stress (HR = 1.23, 95% CI: [1.21, 1.24], p < 0.001):** Every unit increase in financial stress increases dropout hazard by **23%**.

## 3. Kaplan-Meier Cumulative Persistence Probability
- **Semester 1:** 93.3% cumulative persistence probability
- **Semester 2:** 85.1% cumulative persistence probability
- **Semester 3:** 76.8% cumulative persistence probability
- **Semester 4:** 69.6% cumulative persistence probability
- **Semester 5:** 62.7% cumulative persistence probability
- **Semester 6:** 56.8% cumulative persistence probability
- **Semester 7:** 51.1% cumulative persistence probability
- **Semester 8:** 46.3% cumulative persistence probability

## 4. Stratified Log-Rank Tests
- **First_Generation (First-Gen vs Continuing-Gen):** Chi-Square = `481.45` — Statistically Significant (p < 0.001)
- **Scholarship (Scholarship vs No Scholarship):** Chi-Square = `245.43` — Statistically Significant (p < 0.001)
- **Gender (Male vs Female):** Chi-Square = `0.55` — p = 0.4590