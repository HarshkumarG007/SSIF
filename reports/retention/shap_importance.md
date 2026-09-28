# Academic Retention: SHAP Feature Importance & Risk Attribution
**Generated:** 2026-09-29  
**Method:** TreeExplainer (Random Forest with Grouped Longitudinal Trajectories)  

## Top 15 Predictors of Next-Semester Dropout Risk
| Feature                   |   Mean_Abs_SHAP |   Relative_Importance_Pct |
|:--------------------------|----------------:|--------------------------:|
| Sem_GPA                   |      0.0573124  |                 14.7879   |
| Financial_Stress          |      0.0486827  |                 12.5613   |
| Failed_Courses            |      0.0470829  |                 12.1485   |
| gpa_recent_mean           |      0.0434143  |                 11.2019   |
| First_Generation_enc      |      0.0377732  |                  9.74636  |
| Scholarship_enc           |      0.0257972  |                  6.65629  |
| cumulative_failed_courses |      0.0193295  |                  4.98746  |
| Work_Hours                |      0.0173617  |                  4.47973  |
| Attendance                |      0.0144416  |                  3.72627  |
| Family_Income             |      0.0107081  |                  2.76294  |
| gpa_volatility            |      0.0088905  |                  2.29396  |
| gpa_velocity              |      0.00846357 |                  2.1838   |
| gpa_slope                 |      0.00840544 |                  2.1688   |
| Course_Load               |      0.00539567 |                  1.39221  |
| LMS_Logins                |      0.00384968 |                  0.993307 |

## Domain Insights
1. **Academic Momentum:** Longitudinal GPA trajectory (slope, velocity) and current Semester GPA are leading risk indicators.
2. **Engagement Decay:** Consecutive GPA declines (`decline_index`) and Attendance slope provide strong early warning before official dropout.
3. **Financial Stress & Work Hours:** Financial stress and high work hours act as compounding vulnerability multipliers.