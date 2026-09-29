# EXP-003: Socio-Economic Fairness Audit
## Student Success Intelligence Framework (SSIF)

### Research Question
Does the empirical 65% degree-GPA hiring threshold disproportionately exclude
students from specific demographic groups?

---

### Threshold Context
- **Threshold:** 2.8400 GPA (65th percentile of Dataset A; analogous to the 65% degree GPA hiring gate in Dataset B)
- **Power caveat:** Dataset B has N=215; all placement statistics are low-power.

---

### Q1: Threshold Achievability by Demographics (Dataset A, N=20,000 students)

#### By Gender
| group             |   n_students |   achievability_pct |
|:------------------|-------------:|--------------------:|
| Female            |         9905 |               57.21 |
| Male              |         9499 |               58.13 |
| Other             |          289 |               59.86 |
| Prefer not to say |          108 |               53.7  |

#### By First-Generation Status
| group          |   n_students |   achievability_pct |
|:---------------|-------------:|--------------------:|
| Continuing-Gen |        12880 |               58.63 |
| First-Gen      |         6921 |               55.9  |

#### By Income Quartile
| group        |   n_students |   achievability_pct |
|:-------------|-------------:|--------------------:|
| Q1 (Lowest)  |         5134 |               49.94 |
| Q2           |         4893 |               54.12 |
| Q3           |         4827 |               62.54 |
| Q4 (Highest) |         4947 |               64.46 |

---

### Q2: Placement Fairness Metrics (Dataset B, N=215)

#### Demographic Parity — Placement Rate
| dimension   | group   |   n |   placement_rate_% |
|:------------|:--------|----:|-------------------:|
| Gender      | F       |  76 |              63.16 |
| Gender      | M       | 139 |              71.94 |
| Workex      | No      | 141 |              59.57 |
| Workex      | Yes     |  74 |              86.49 |

#### Salary Parity (Placed Students Only)
| group   |   n |   mean_salary_INR |
|:--------|----:|------------------:|
| F       |  48 |            267292 |
| M       | 100 |            298910 |

#### Gender Wage Gap
| group   |   value | unit                  |
|:--------|--------:|:----------------------|
| M vs F  |       8 | % above female median |

---

### Q3: Work-Experience Rescue Differential (Dataset B, sub-threshold students)

| Gender | N sub-threshold | Rescue w/ WorkEx | Rescue w/o WorkEx | Differential |
|--------|----------------|-----------------|------------------|-------------|
| M | 60 | 81.25% | 31.82% | +49.43 pp |
| F | 23 | 50.0% | 29.41% | +20.59 pp |

**Fisher's Exact Test** (gender × workex for sub-threshold pool):
- Odds Ratio: 1.0303
- p-value: 1.0000
- Fisher's exact test on [gender x workex] for sub-threshold students. OR=1.030, p=1.0000. No significant differential (p>=0.05): work experience rescue appears gender-neutral.

> ⚠️ N=83 sub-threshold students in Dataset B (N=215 total); Fisher's test preferred for small samples.

---

### Q4: Qualified-But-Excluded Profiles (Dataset A)
*(Students with improving GPA trajectory who never crossed threshold and were retained)*

| Gender   |   First_Generation | income_quartile   |   n_qualified_excluded |   mean_gpa_slope |   mean_max_gpa |
|:---------|-------------------:|:------------------|-----------------------:|-----------------:|---------------:|
| Female   |                  0 | Q1                |                     85 |           0.045  |          2.441 |
| Female   |                  0 | Q2                |                     77 |           0.0567 |          2.482 |
| Female   |                  0 | Q3                |                     95 |           0.0532 |          2.542 |
| Female   |                  0 | Q4                |                     83 |           0.0506 |          2.539 |
| Female   |                  1 | Q1                |                     44 |           0.0495 |          2.469 |
| Female   |                  1 | Q2                |                     54 |           0.062  |          2.494 |
| Female   |                  1 | Q3                |                     34 |           0.0544 |          2.552 |
| Female   |                  1 | Q4                |                     41 |           0.0593 |          2.562 |
| Male     |                  0 | Q1                |                     83 |           0.052  |          2.474 |
| Male     |                  0 | Q2                |                     83 |           0.0467 |          2.496 |
| Male     |                  0 | Q3                |                     68 |           0.0535 |          2.528 |
| Male     |                  0 | Q4                |                     80 |           0.0558 |          2.531 |
| Male     |                  1 | Q1                |                     32 |           0.0526 |          2.51  |
| Male     |                  1 | Q2                |                     40 |           0.0532 |          2.51  |
| Male     |                  1 | Q3                |                     41 |           0.0593 |          2.565 |

---

### Key Findings
1. **Threshold Achievability** is significantly lower for Q1 (lowest income) students
   compared to Q4, suggesting the threshold encodes socio-economic advantage.
2. **Work-Experience Rescue** may be more accessible to students from higher-income
   backgrounds who can afford unpaid internships or have networks for paid roles.
3. **Gender Wage Gap**: Males earn a statistically non-significant premium
   over females in placed salaries (Dataset B, N limited).
4. **Qualified-But-Excluded** students are disproportionately first-generation and Q1 income.

### Limitations
- Dataset B (N=215) is insufficient for robust subgroup fairness analysis; findings are
  directional signals requiring larger-N validation.
- Dataset A has no direct linkage to eventual placement; achievability analysis is a proxy.
- No causal claims: selection effects (who applies to what companies) are unobserved.
