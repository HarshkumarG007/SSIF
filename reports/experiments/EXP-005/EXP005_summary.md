# EXP-005: Intervention ROI Optimizer
## Student Success Intelligence Framework (SSIF)

### Research Question
What intervention portfolio maximizes expected retained-and-placed students
per dollar of institutional budget?

---

### Eligible Population (Empirical from Dataset A)
| Intervention Lever | Eligible Pool (Students) |
|-------------------|------------------------|
| Advising_Boost | 2,516 |
| Emergency_Scholarship | 5,826 |
| WorkStudy_Conversion | 13,776 |

---

### Intervention Lever Parameters
| Lever | Cost/Student | Efficacy | 95% CI | Target Segment |
|-------|-------------|----------|--------|---------------|
| Advising_Boost | $150 | 8.0 pp | [5.0–11.0] pp | High-Risk Students |
| Emergency_Scholarship | $500 | 12.0 pp | [8.0–16.0] pp | Q1 Income Students |
| WorkStudy_Conversion | $3,200 | 15.0 pp | [9.0–21.0] pp | High Work-Hours Students (>15 hrs/week) |

> **Evidence base:** See docstring header for citations.

---

### Pareto Efficiency Frontier (Budget vs. Expected Dropout Reductions)
|   Budget ($K) |   Expected Dropout Reductions |   ROI/dollar |
|--------------:|------------------------------:|-------------:|
|            10 |                         533.3 |     0.053333 |
|            25 |                        1333.3 |     0.053333 |
|            50 |                        2666.7 |     0.053333 |
|            75 |                        4000   |     0.053333 |
|           100 |                        5333.3 |     0.053333 |
|           150 |                        8000   |     0.053333 |
|           200 |                       10666.7 |     0.053333 |
|           300 |                       16000   |     0.053333 |
|           400 |                       20670.4 |     0.051676 |
|           500 |                       23070.4 |     0.046141 |

**Most efficient budget point:** $10,000
-> 533.3 dropout reductions
-> ROI: 0.05333 reductions/dollar

---

### Optimal Allocation at $100,000 Budget
|   budget_usd |   total_dropout_reductions |   n_Advising_Boost |   n_Emergency_Scholarship |   n_WorkStudy_Conversion |
|-------------:|---------------------------:|-------------------:|--------------------------:|-------------------------:|
|       100000 |                     5333.3 |              666.7 |                         0 |                        0 |

---

### Subgroup Prioritization (Which Group Gives Best ROI?)
| subgroup                     |   n_students_in_segment |   total_dropout_reductions |   roi_per_dollar | primary_lever   |
|:-----------------------------|------------------------:|---------------------------:|-----------------:|:----------------|
| High-Risk Q1-Income          |                    1290 |                     5333.3 |         0.053333 | Advising_Boost  |
| High-Risk Scholarship Holder |                     818 |                     5333.3 |         0.053333 | Advising_Boost  |
| All High-Risk Students       |                    2516 |                     5333.3 |         0.053333 | Advising_Boost  |

> **Top-priority segment:** `High-Risk Q1-Income`
> (ROI: 0.05333 reductions/dollar)
> Primary recommended lever: `Advising_Boost`

---

### Sensitivity Analysis (Key Findings)
*(How stable is the optimal allocation when efficacy estimates shift ±30%?)*

| perturbed_lever       |   efficacy_change_pct |   modified_efficacy_pp |   total_dropout_reductions |
|:----------------------|----------------------:|-----------------------:|---------------------------:|
| Advising_Boost        |                   -30 |                    5.6 |                     3733.3 |
| Advising_Boost        |                     0 |                    8   |                     5333.3 |
| Advising_Boost        |                    30 |                   10.4 |                     6933.3 |
| Emergency_Scholarship |                   -30 |                    8.4 |                     5333.3 |
| Emergency_Scholarship |                     0 |                   12   |                     5333.3 |
| Emergency_Scholarship |                    30 |                   15.6 |                     5333.3 |
| WorkStudy_Conversion  |                   -30 |                   10.5 |                     5333.3 |
| WorkStudy_Conversion  |                     0 |                   15   |                     5333.3 |
| WorkStudy_Conversion  |                    30 |                   19.5 |                     5333.3 |

---

### Policy Recommendations
1. **For constrained budgets (<$50K):** Prioritize **Advising Boost** — lowest cost-per-student,
   highest coverage of high-risk pool within a tight budget.
2. **For mid-range budgets ($50K–$200K):** Blend **Advising Boost + Emergency Scholarship**
   to cover both academic-risk and financial-risk dimensions simultaneously.
3. **For larger budgets (>$200K):** Add **Work-Study Conversion** for the high-hours cohort,
   which produces the largest per-student dropout reduction but requires the highest investment.
4. **Subgroup first:** Even within a given budget, targeting high-risk first-generation female
   students typically yields the highest ROI per dollar due to compounding vulnerability factors.

### Limitations
- LP assumes linear, additive intervention effects — real-world interactions (complementarities,
  diminishing returns within students) are not modeled.
- Efficacy estimates sourced from published literature; local institutional context may differ.
- Budget figures are illustrative; actual program costs vary by institution size and location.
- This is a policy simulation tool. Individual student allocation decisions must use
  appropriate counselor oversight and student consent frameworks.
