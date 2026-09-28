# 🎓 Student Success Intelligence: Longitudinal Survival, Academic Resilience & Digital Habit Spillover
### A Publication-Ready Community Article & Kaggle Notebook Write-up

**Author:** Harshkumar G.  
**Open Source Repository:** [HarshkumarG007/SSIF](https://github.com/HarshkumarG007/SSIF)  
**Related Project:** [Digital Lifestyle Spillover Modeling (DLSM)](https://github.com/HarshkumarG007/DLSM) | [Live DLSM App](https://dlsm-research.streamlit.app/)  
**License:** Apache 2.0  

---

## 🚀 The Core Problem: Why Most Student Dropout Models Fail

In traditional academic analytics, institutions build models on static semester snapshots. A student with a 2.8 GPA who was previously at 3.8 is treated identically to a student with a 2.8 GPA who rose from 1.8. 

Furthermore, standard tabular train-test splits (like 80/20 random splits) suffer from **severe temporal and within-student leakage**: observations from the same student appear in both train and test sets, artificially inflating cross-validation performance while failing completely in production.

In this research, we introduce the **Student Success Intelligence Framework (SSIF)**, built across two public Kaggle datasets:
1. **[Student Retention and Academic Performance Panel](https://www.kaggle.com/datasets/razanihababdellatif/student-retention-and-academic-performance-data):** 79,239 longitudinal records across 20,000 students over up to 8 semesters.
2. **[MBA Placement & Employability Dataset](https://www.kaggle.com/datasets/ameythakur20/placement-data):** 215 candidates tracking multi-stage academic history and career placement.
3. **[Digital Lifestyle Telemetry (DLSM)](https://github.com/HarshkumarG007/DLSM):** Investigating whether digital habits (AI tool usage, sleep debt, screentime) can or should be merged with academic records.

---

## 🧠 Breakthrough 1: Vectorized OLS Trajectories in 0.15s (Zero Leakage)

Rather than using rolling windows with future leakage, we developed a closed-form cumulative ordinary least squares (OLS) regression engine:

$$\text{GPA Slope}_t = \frac{n \sum_{s=1}^t (s \cdot \text{GPA}_s) - \left(\sum s\right)\left(\sum \text{GPA}_s\right)}{n \sum s^2 - \left(\sum s\right)^2}$$

Because all calculations for student $i$ at semester $t$ utilize strictly $s \le t$, it preserves **strict temporal causality (RULE-009)**. Running in 0.15 seconds over 79,239 rows, it computes:
- Cumulative GPA slope & volatility
- GPA velocity ($\Delta\text{GPA}$)
- Consecutive semester decline index
- Trajectory recovery index

Students with a declining GPA trajectory ($\text{slope} < -0.2$) suffer a **16.03% departure rate**, compared to **7.08%** for students with stable or improving trajectories.

---

## ⏱️ Breakthrough 2: Time-to-Event Survival Analysis (Cox PH $C = 0.7498$)

Dropout is a time-to-event process with right-censoring (graduating students are not failures; they are censored observations). Using Kaplan-Meier and Cox Proportional Hazards:

| Factor | Hazard Ratio (HR) | 95% Confidence Interval | p-value | Institutional Interpretation |
|---|:---:|:---:|:---:|---|
| **First-Generation Status** | **1.98×** | [1.89, 2.08] | $p < 0.001$ | First-gen students face **nearly double the instantaneous departure hazard** at every semester. |
| **Institutional Scholarship** | **0.52×** | [0.49, 0.55] | $p < 0.001$ | Scholarship funding **cuts departure hazard by 48%** — the strongest protective buffer. |
| **Academic GPA** | **0.40×** | [0.38, 0.42] | $p < 0.001$ | Each 1.0 GPA increase cuts dropout hazard by **60%**. |
| **Financial Stress** | **1.23×** | [1.21, 1.24] | $p < 0.001$ | Each point of financial stress multiplies departure hazard by **1.23×**. |

---

## 🛡️ Breakthrough 3: Academic Resilience & The Advising Window ($N=5,563$)

What separates students who rebound from academic shocks from those who drop out?
We isolated $N=5,563$ students who exhibited a **recovery signature** ($\Delta\text{GPA} \le -0.3$ shock followed by rebound):
- **Recovery cohort dropout rate:** **22.6%**
- **Continuing decline dropout rate:** **41.9%** (an absolute risk reduction of 19.3%).

Multivariate Logistic Regression revealed:
- **Academic Advising is the top resilience booster ($\text{OR} = 1.731$, $p < 0.001$):** Each advising visit per semester increases the odds of recovery by **+73.1%**.
- **Financial Stress is the primary barrier ($\text{OR} = 0.666$, $p < 0.001$):** High stress cuts recovery odds by **33.4%**, explaining why advice alone fails without emergency financial relief.

---

## 🚫 Breakthrough 4: The DLSM Integration Truth — Why Row Merging Is Forbidden

A common impulse in data science is to merge datasets row-by-row to "enrich" features. We tested whether DLSM digital telemetry can be merged into academic records:
1. **Schema Overlap:** Only `Age` and `Gender` are shared. Core digital telemetry (`Sleep_Hours`, `Daily_Social_Media_Hours`, `Daily_AI_Tool_Usage_Hours`) is absent from academic records.
2. **Empirical Ablation Test:** We ran a 5-Fold GroupKFold ablation comparing pure academic baseline (A0) against academic + DLSM demographics (A1):
   - A0 AUROC: **0.80130**
   - A1 AUROC: **0.80125**
   - **$\Delta\text{AUROC} = -0.00005$ ($p = 0.932$, null effect).**

**Verdict:** Row-level merging is a **NO-GO**. However, a **representation-level construct bridge** is scientifically valid: digital lifestyle fatigue and academic fatigue represent parallel pathways leading to higher education burnout.

---

## 💼 Placement Reality: What Actually Predicts MBA Hiring?

Analyzing the 215 MBA candidate dataset with Stratified 5-Fold Cross-Validation:
- **Employability Classification:** Logistic Regression achieved **AUROC = 0.9370**.
- **Prior Work Experience Lift:** Candidates with work experience achieved an **86.5% placement rate**, vs **59.6%** for those without.
- **Salary Reality ($R^2 \approx 0$):** Starting salary offers among placed students are uncorrelated with fine GPA differences; starting remuneration is governed by rigid corporate pay bands.

---

## 💻 Explore the Code & Interactive Observatory

- **GitHub Repository:** [HarshkumarG007/SSIF](https://github.com/HarshkumarG007/SSIF)
- **Kaggle Master Notebook:** `notebooks/kaggle_ssif_student_success_study.ipynb`
- **Streamlit Research Observatory:** Run `streamlit run app/main.py`
