# DLSM Compatibility Gate Report

**Generated:** 2026-09-29  
**Scientific Principle:** RULE-004 — Never claim DLSM compatibility without empirical evidence.

## ❌ Gate: SSIF-A (Retention) ↔ DLSM-B (AI & Social Media Student Health)
**Verdict:** `IntegrationVerdict.NO_GO` | **Compatibility Score:** 0%
**Row Merge Permitted:** ❌ NO (RULE-003)
**Representation Bridge Permitted:** ✅ YES

### Variable Mapping
| DLSM Variable             | SSIF Column   | Status   | Transformation                                            | Notes                                                                                 |
|:--------------------------|:--------------|:---------|:----------------------------------------------------------|:--------------------------------------------------------------------------------------|
| Daily_Social_Media_Hours  | ABSENT        | ABSENT   | —                                                         | No social media variable in retention panel                                           |
| Daily_AI_Tool_Usage_Hours | ABSENT        | ABSENT   | —                                                         | No AI usage variable in retention panel                                               |
| Sleep_Hours               | ABSENT        | ABSENT   | —                                                         | No sleep variable in retention panel                                                  |
| Physical_Activity_Hours   | ABSENT        | ABSENT   | —                                                         | No physical activity in retention panel                                               |
| Mental_Health_Score       | ABSENT        | ABSENT   | —                                                         | No mental health outcome in retention panel                                           |
| Physical_Health_Score     | ABSENT        | ABSENT   | —                                                         | No physical health in retention panel                                                 |
| Age                       | Age           | DIRECT   | —                                                         | Direct mapping — same meaning                                                         |
| Gender                    | Gender        | DIRECT   | —                                                         | Direct mapping — same meaning                                                         |
| Education_Level           | Semester      | PROXY    | Semester→Education_Level: 1-4=Undergraduate, 5-8=Advanced | Semester is a proxy for academic progression stage — not identical to Education_Level |

### Scientific Rationale
Core DLSM-B behavioral variables absent from SSIF-A: ['Daily_Social_Media_Hours', 'Daily_AI_Tool_Usage_Hours', 'Sleep_Hours', 'Physical_Activity_Hours']. Compatibility score: 0.00 (0/4 core vars). Verdict: NO-GO for direct integration. PERMITTED: Representation-level comparison — both datasets contain student populations at similar life stages; latent construct comparison is scientifically defensible without claiming these are the same individuals. Future work: Collect Sleep_Hours, Daily_Social_Media_Hours, Daily_AI_Tool_Usage_Hours, Physical_Activity_Hours alongside retention variables to enable full integration.

## ❌ Gate: SSIF-B (Placement) ↔ DLSM-A (Bedtime Screen Time & Sleep Debt)
**Verdict:** `IntegrationVerdict.NO_GO` | **Compatibility Score:** 0%
**Row Merge Permitted:** ❌ NO (RULE-003)
**Representation Bridge Permitted:** ❌ NO

### Variable Mapping
| DLSM Variable            | SSIF Column   | Status   | Transformation   | Notes                                 |
|:-------------------------|:--------------|:---------|:-----------------|:--------------------------------------|
| bedtime_phone_minutes    | ABSENT        | ABSENT   | —                | No bedtime variable in placement data |
| screen_brightness_pct    | ABSENT        | ABSENT   | —                | No screen variable in placement data  |
| blue_light_filter_active | ABSENT        | ABSENT   | —                | Not measured                          |
| caffeine_post_5pm_mg     | ABSENT        | ABSENT   | —                | Not measured                          |
| total_sleep_hours        | ABSENT        | ABSENT   | —                | Not measured                          |
| sleep_latency_min        | ABSENT        | ABSENT   | —                | Not measured                          |
| physical_activity_min    | ABSENT        | ABSENT   | —                | Not measured                          |
| gender                   | gender        | DIRECT   | —                | Direct mapping                        |
| age                      | ABSENT        | ABSENT   | —                | Age not in placement dataset          |

### Scientific Rationale
All DLSM-A core behavioral variables absent from SSIF-B (Placement). Compatibility score: 0.00. Verdict: NO-GO. No representation-level bridge recommended given minimal variable overlap.

---
## Key Finding

**Neither SSIF dataset can be directly scored by DLSM** because the core behavioral
variables (sleep hours, daily social/AI media hours, screen time, physical activity)
are absent from both `academic_survival_longitudinal.csv` and `Placement_Data_Full_Class.csv`.

**What IS permitted:**
- SSIF-A (Retention) ↔ DLSM-B: Representation-level comparison of student populations
  on shared demographic dimensions (Age, Gender, Education stage)
- Study of whether academic persistence latent structure mirrors digital-lifestyle latent structure
  *without claiming these are the same students*

**Future data required for full integration:**
Collect `Sleep_Hours`, `Daily_Social_Media_Hours`, `Daily_AI_Tool_Usage_Hours`,
`Physical_Activity_Hours` alongside all retention variables for the same students.

---

## 🏛️ Dataset Provenance & Curators Thanksgiving

1. **SSIF Retention Dataset (SSIF-A):**
   - **Curator:** **Razan Ihab Abdellatif**
   - **Kaggle URL:** [Student Retention and Academic Performance Data](https://www.kaggle.com/datasets/razanihababdellatif/student-retention-and-academic-performance-data)
2. **SSIF Placement Dataset (SSIF-B):**
   - **Curator:** **Amey Thakur** ([Kaggle: @ameythakur20](https://www.kaggle.com/ameythakur20))
   - **Kaggle URL:** [Campus Recruitment (Placement Data Full Class)](https://www.kaggle.com/datasets/ameythakur20/placement-data)
3. **DLSM Sleep & Screentime (DLSM-A):**
   - **Curator:** **Samar Talwar**
   - **Kaggle URL:** [Sleep Debt and Screen Time / Late Night Phone Habits](https://www.kaggle.com/datasets/samartalwar/sleep-debt-and-screen-time-late-night-phone-habits)
4. **DLSM AI & Social Media (DLSM-B):**
   - **Curator:** **Sri Syra** ([Kaggle: @srisyra02](https://www.kaggle.com/srisyra02))
   - **Kaggle URL:** [AI and Social Media Impact: Student Health & Grades](https://www.kaggle.com/datasets/srisyra02/ai-and-social-media-impact-student-health-and-grades)
5. **DLSM Sister Framework:** [HarshkumarG007/DLSM](https://github.com/HarshkumarG007/DLSM)

> 📢 *We gratefully acknowledge the above creators and encourage researchers to star their datasets on Kaggle and download primary CSVs directly from their author profiles.*