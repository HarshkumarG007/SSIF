# SSIF Data Lakehouse & Dataset Catalog
## Student Success Intelligence Framework — Reproducible Data Lakehouse

This directory houses the Medallion-style three-tier data lakehouse powering the **Student Success Intelligence Framework (SSIF)** and its empirical bridge with the **Digital Lifestyle Spillover Modeling (DLSM)** research ecosystem.

---

## 🏛️ Lakehouse Architecture (Medallion Standard)

```
data/
├── raw/                              # Bronze Tier: Immutable raw CSVs
│   ├── retention/academic_survival_longitudinal.csv
│   ├── placement/Placement_Data_Full_Class.csv
│   └── dlsm_b/AI_SocialMedia_Student_Dataset.csv
│
├── interim/                          # Silver Tier: Cleaned & type-sanitized
│   ├── retention_interim.csv / .parquet
│   ├── placement_interim.csv / .parquet
│   └── dlsm_b_interim.csv / .parquet
│
├── processed/                        # Gold Tier: Analytical feature matrices
│   ├── ssif_retention_longitudinal_enriched.csv / .parquet
│   ├── ssif_retention_student_profiles.csv / .parquet
│   ├── ssif_placement_enriched.csv / .parquet
│   ├── ssif_macro_pipeline_cohorts.csv
│   └── ssif_higher_ed_synthesis_metrics.json / .csv
│
└── DATASET_METRICS_CATALOG.json      # Machine-readable lakehouse manifest
```

### Tier Definitions
1. **`data/raw/` (Bronze Tier):** Exact source datasets. Strict read-only governance; never edited in place.
2. **`data/interim/` (Silver Tier):** Strict schema-validated datasets with canonicalized demographic strings, sanitized missing values, and zero data leakage. Dual-stored in `.csv` and compressed `.parquet`.
3. **`data/processed/` (Gold Tier):** Fully enriched analytical matrices containing OLS trajectory slopes, decline run-lengths, career readiness indicators, and macro pipeline stage aggregations ready for downstream modeling.

---

## 🙏 Primary Dataset Provenance, Credits & Curators Thanksgiving

The SSIF research initiative stands upon the invaluable contributions of the academic data science community. We extend our deepest gratitude, respect, and thanksgiving to the original creators and curators of the datasets:

### 🎓 1. Student Retention & Academic Performance Panel (SSIF-A)
* **Curator & Original Author:** **Razan Ihab Abdellatif**
* **Primary Kaggle Dataset:** [Student Retention and Academic Performance Data](https://www.kaggle.com/datasets/razanihababdellatif/student-retention-and-academic-performance-data)
* **Sample Dimensions:** 79,239 longitudinal student-semester records across 20,000 unique students (Semesters 1 through 8).
* **Key Observations:** GPA momentum, course failure counts, attendance percentage, advising visits, financial aid, and realized dropout indicators.
* **Thanksgiving Note:** We are profoundly grateful to **Razan Ihab Abdellatif** for open-sourcing this rich multi-semester panel. Its temporal structure made rigorous zero-leakage trajectory modeling, Kaplan-Meier hazard curves, and empirical persistence analysis possible.

### 💼 2. MBA Campus Recruitment & Employability Cohort (SSIF-B)
* **Curator & Original Author:** **Amey Thakur** ([Kaggle: @ameythakur20](https://www.kaggle.com/ameythakur20))
* **Primary Kaggle Dataset:** [Campus Recruitment (Placement Data Full Class)](https://www.kaggle.com/datasets/ameythakur20/placement-data)
* **Sample Dimensions:** 215 business school candidate profiles with complete multi-tier academic history.
* **Key Observations:** Secondary (10th), higher secondary (12th), undergraduate degree percentages, MBA specialization, standardized employability scores, prior work experience, placement status, and corporate salary offers.
* **Thanksgiving Note:** Our sincere thanks go to **Amey Thakur** for providing this landmark placement benchmark. It enabled SSIF to model the two-stage decoupling between hiring probability and conditional salary, revealing the critical role of prior professional work experience (+26.9 pp placement lift).

### 🌙 3. Sleep Debt & Screen Time / Late-Night Phone Habits (DLSM-A)
* **Curator & Original Author:** **Samar Talwar**
* **Primary Kaggle Dataset:** [Sleep Debt and Screen Time / Late Night Phone Habits](https://www.kaggle.com/datasets/samartalwar/sleep-debt-and-screen-time-late-night-phone-habits)
* **Sample Dimensions:** 8,500 lifestyle telemetry records.
* **Key Observations:** Bedtime screen usage minutes, device brightness, post-5pm caffeine intake, sleep latency, total sleep hours, and next-day fatigue indices.
* **Thanksgiving Note:** Heartfelt thanks to **Samar Talwar** for curating and sharing this detailed behavioral dataset. It serves as the empirical anchor for understanding how late-night digital habits translate into physiological sleep debt within the sister DLSM framework.

### 🤖 4. AI Tool Usage, Social Media & Student Mental Health (DLSM-B)
* **Curator & Original Author:** **Sri Syra** ([Kaggle: @srisyra02](https://www.kaggle.com/srisyra02))
* **Primary Kaggle Dataset:** [AI and Social Media Impact: Student Health & Grades](https://www.kaggle.com/datasets/srisyra02/ai-and-social-media-impact-student-health-and-grades)
* **Sample Dimensions:** 16,000 university student records.
* **Key Observations:** Daily AI study tool usage hours, daily social media hours, subjective mental and physical health ratings, and academic grade bands.
* **Thanksgiving Note:** Our deepest appreciation goes to **Sri Syra** for releasing this large-scale student telemetry cohort, allowing DLSM to explore the non-linear inflection points between productive AI study assistance and digital cognitive fatigue.

### 🔗 Sister Research Ecosystem: DLSM
* **GitHub Repository:** [HarshkumarG007/DLSM](https://github.com/HarshkumarG007/DLSM)
* **Live DLSM Portal:** [https://dlsm-research.streamlit.app/](https://dlsm-research.streamlit.app/)
* **Cross-Study Relationship:** DLSM models upstream behavioral lifestyle loads (sleep debt, screentime, mental health); SSIF models downstream institutional persistence and career placement.

---

## 📢 Ethical Data Access & Primary Download Call-to-Action

> ### ⚠️ Important Notice for Researchers and Replicators
> 
> To honor academic data governance, licensing, and community attribution:
> 
> 1. **Do not use third-party or reshuffled mirrors of these datasets.**
> 2. **Please visit the original primary Kaggle dataset pages linked above.**
> 3. **Give the creators an upvote / star on Kaggle** to recognize their generous contributions to open educational and behavioral data science:
>    - [Razan Ihab Abdellatif on Kaggle](https://www.kaggle.com/datasets/razanihababdellatif/student-retention-and-academic-performance-data)
>    - [Amey Thakur on Kaggle](https://www.kaggle.com/datasets/ameythakur20/placement-data)
>    - [Samar Talwar on Kaggle](https://www.kaggle.com/datasets/samartalwar/sleep-debt-and-screen-time-late-night-phone-habits)
>    - [Sri Syra on Kaggle](https://www.kaggle.com/datasets/srisyra02/ai-and-social-media-impact-student-health-and-grades)
> 4. **Download the raw CSV files directly from the original Kaggle creators** into your local `data/raw/` directory.
> 
> All primary dataset recognition, licensing inquiries, and original provenance belong to these four outstanding curators.
