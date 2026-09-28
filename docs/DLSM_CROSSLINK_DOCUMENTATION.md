# Cross-Linking Documentation: DLSM ↔ SSIF Research Ecosystem
### Mutual Integration Reference between Digital Lifestyle Telemetry & Academic Persistence

**SSIF Repository:** [https://github.com/HarshkumarG007/SSIF](https://github.com/HarshkumarG007/SSIF)  
**DLSM Repository:** [https://github.com/HarshkumarG007/DLSM](https://github.com/HarshkumarG007/DLSM)  
**Live DLSM Observatory:** [https://dlsm-research.streamlit.app/](https://dlsm-research.streamlit.app/)  
**Live SSIF Observatory:** [https://ssif-research.streamlit.app/](https://ssif-research.streamlit.app/) (or local: `http://localhost:8502`)  

---

## 📌 Context & Motivation

The **Digital Lifestyle Spillover Modeling (DLSM)** project investigates the impact of late-night screentime, sleep debt, and AI study habits on cognitive fatigue and student mental health. 

The **Student Success Intelligence Framework (SSIF)** investigates longitudinal academic persistence, departure hazard (Cox Proportional Hazards $C=0.7498$), and career placement across 79,239 student semesters.

Together, these two open-source frameworks form a **unified computational education intelligence ecosystem**:
1. **DLSM** captures the **upstream behavioral inputs** (screen habits, sleep debt, cognitive fatigue).
2. **SSIF** captures the **downstream institutional outcomes** (GPA momentum, course failure, dropout hazard, career placement).

---

## 📝 README Section to Add to DLSM (`C:\Users\Lenovo\Downloads\DLSM\README.md`)

Add the following section to the bottom of the DLSM repository README:

```markdown
---

## 🎓 Sister Research Framework: SSIF (Student Success Intelligence)

DLSM is designed to interface conceptually with the **Student Success Intelligence Framework (SSIF)**:
- **Repository:** [HarshkumarG007/SSIF](https://github.com/HarshkumarG007/SSIF)
- **Live Observatory:** [SSIF Research Portal](https://ssif-research.streamlit.app/)

### Cross-Study Empirical Gate Summary
In rigorous empirical audits across 79,239 longitudinal academic records and 16,000 digital lifestyle records:
- **Row-Level Merging:** Evaluated as **NO-GO** (Direct overlap score: 0.154). An empirical 5-Fold GroupKFold ablation proved $\Delta\text{AUROC} = -0.00005$ ($p = 0.932$), demonstrating that demographic overlap alone provides zero predictive lift without direct behavioral telemetry.
- **Representation-Level Construct Bridge:** Evaluated as **VALID**. Demographic alignment achieves a Wasserstein distance of $1.767$ years. DLSM's fatigue score and SSIF's attendance/GPA decay represent parallel structural mechanisms in student attrition.
- **Future Unified Cohort Blueprint:** A protocol for prospective multi-modal data capture tracking synchronous bedtime telemetry, LMS activity timestamps, and semester credit completion on shared student identifiers.
```

---

## 🔬 Cross-Study Theoretical Mapping

| Construct Dimension | DLSM (Digital Lifestyle Layer) | SSIF (Academic Persistence Layer) | Joint Policy Recommendation |
|---|---|---|---|
| **Behavioral Input** | Late-night screen time, AI study bingeing | Course overload, high external work hours | Early habit coaching & workload advisory |
| **Fatigue Mechanism** | Sleep debt, daytime fatigue index | Attendance erosion, LMS login drop | Early warning alert triggered by attendance dip |
| **Academic Shock** | Mental health score decline | Semester GPA drop ($\Delta\text{GPA} \le -0.3$) | Academic advising check-in (OR = 1.731 recovery boost) |
| **Structural Buffer** | Sleep hygiene habits, digital detox | Institutional scholarship, tuition relief | Emergency financial relief cuts hazard by 48% |
| **Terminal Outcome** | Cognitive burnout | Academic departure (Dropout) | Unified early retention intervention |

---

## ⚖️ The Empirical Calibration Ceiling: Orben & Przybylski (2019, n=355,358)

A critical scientific principle governs both DLSM and SSIF: **effect-size plausibility benchmarking**.

In a landmark, pre-registered study of digital technology use and adolescent wellbeing using specification-curve analysis across $n = 355,358$ individuals, Orben & Przybylski (*Nature Human Behaviour*, 2019) demonstrated that:
- **Digital technology use explains at most $0.4\%$ ($R^2 \le 0.004$) of the variance in adolescent wellbeing.**
- The association between digital screen exposure and wellbeing is comparable to the association between eating potatoes and wellbeing, and smaller than the negative association of wearing corrective eyewear.
- In contrast, physiological drivers such as **getting adequate sleep** and **eating breakfast regularly** exhibited orders-of-magnitude stronger associations with student wellbeing.

### Methodological Implications for SSIF and DLSM:
1. **Plausibility Gate:** Any model or claimed synthetic feature asserting massive, deterministic direct causal links between screen hours and academic dropout (e.g. $R^2 > 0.10$ or single-variable $\text{AUC} > 0.85$) represents either target leakage or a synthetic generator artifact, not an authentic human behavioral phenomenon.
2. **Shared Feasibility Tooling (`dataset_feasibility_audit.py`):** Rather than forcing synthetic data merges across disjoint populations, the true architectural bridge between DLSM and SSIF is **shared pre-modeling validation tooling**. The automated 7-gate feasibility auditor scans raw CSV files for synthetic fingerprints, post-outcome leakage (`salary`, `End_of_Semester_Status`), events-per-variable adequacy, and specification-curve plausibility before modeling commences.

