# SSIF Scientific Methodology & Governance Specification

**Student Success Intelligence Framework (SSIF)**  
**Version:** 2.0.0  
**Research Standard:** Executable Scientific Governance & Evidence-Bounded Inference  

---

## 1. Unified Research Question & Scientific Governance

### 1.1 The Central Institutional Inquiry

> **Can longitudinal educational indicators, evaluated under strict causal and temporal boundary conditions, produce calibrated early warnings and actionable policy recourse without succumbing to data leakage, small-sample overfitting, or unverified causal overclaims?**

SSIF addresses this question by shifting the paradigm from ad-hoc predictive modeling to **Executable Scientific Governance**. Rather than treating machine learning models as black-box predictors, SSIF implements 62 programmatically enforced rules that structurally prevent methodological flaws at runtime.

### 1.2 The 62-Rule Governance Architecture

The governance layer is organized across 5 core research domains:

```
┌─────────────────────────────────────────────────────────────────┐
│                    SSIF GOVERNANCE LAYER                        │
├─────────────────┬─────────────────┬──────────────┬──────────────┤
│ Boundary & Joins│ Temporal Leaks  │ Model Tiering│ Uncertainty  │
│ (RULE-001..003) │ (RULE-007..011) │ (RULE-004..6)│ (RULE-016..26)
└─────────────────┴─────────────────┴──────────────┴──────────────┘
```

1. **Boundary & Cross-Dataset Isolation (RULE-001 to RULE-003):**  
   - `RULE-001`: SSIF-A (Longitudinal Retention, $N=79,239$) and SSIF-B (Placement, $N=215$) belong to distinct institutions and populations.
   - `RULE-002`: No fabricated or shared student keys across datasets.
   - `RULE-003`: Row merging or joint training across datasets is prohibited; compatibility is strictly evaluated via `DLSMCompatibilityGate` representation testing.
2. **Temporal Causality & Zero Leakage (RULE-007 to RULE-011):**  
   - `RULE-007`: Preprocessing (scaling, imputation, encoding) is fitted exclusively on training folds.
   - `RULE-008`: No test fold summary statistics are ever accessed during inference.
   - `RULE-009`: All trajectory calculations for semester $t$ use records with $\tau \le t$.
   - `RULE-010`: Post-outcome variables (`End_of_Semester_Status`, `Censored`, `salary`) are strictly forbidden as predictor features.
3. **Complexity Justification & Model Hierarchy (RULE-004 to RULE-006):**  
   - `RULE-004`: Longitudinal retention models must be validated using `GroupKFold(groups=Student_ID)`.
   - `RULE-005`: All benchmarks report sample counts ($N$), positive event counts, and baseline class prevalence.
   - `RULE-006`: Every complex architecture (Random Forest, Gradient Boosting) must be benchmarked against a Tier 0 majority-class prevalence baseline and Tier 1 regularized linear model.
4. **Statistical Uncertainty & Evidence Boundedness (RULE-016 to RULE-026):**  
   - `RULE-016`: Always report PR-AUC alongside AUROC for imbalanced targets, accompanied by cross-fold variance ($\pm \sigma$).
   - `RULE-020`: Probability calibration (Brier Score, ECE, MCE) must be reported for operational intervention models.
   - `RULE-025`: Placement cohort ($N=215$, minority class $N=67$) models must strictly constrain degrees of freedom ($\text{DoF} \le 6$) to maintain Events Per Variable ($\text{EPV} \ge 10$).

---

## 2. Temporal Causality & Zero-Leakage Validation Pipeline

Standard machine learning pipelines in higher education frequently suffer from "future leakage" (e.g., using cumulative 4-year GPA or final semester status to predict 2nd-year dropout). SSIF eliminates this vulnerability structurally:

```
Raw Multi-Semester Records
           │
           ▼
[Temporal Ordering: Sort (Student_ID, Semester)]
           │
           ▼
[Strict Causal History Slice: Semester <= t]
           │
           ▼
[Feature Engineering: OLS Slopes, Deltas, Indices]
           │
           ▼
[Automated Leakage Audit: check_retention_leakage()]
           │
     Pass / Fail Gate
           │
           ▼
[GroupKFold Partitioning: Student_ID Disjoint]
```

- **Leakage Auditing (`src/validation/leakage_detector.py`):**  
  Every candidate feature matrix is screened against forbidden variable registries. If any feature exhibits correlation $\ge 0.999$ with the target or matches post-outcome variable signatures, execution halts immediately with a `ValueError`.
- **Group-Level Partitioning (`src/models/base.py`):**  
  Random train/test splits inadvertently place semester 1 of Student A in the training set and semester 2 of Student A in the test set, creating memorization leakage. SSIF strictly enforces `GroupKFold(n_splits=5)` grouped on `Student_ID`.

---

## 3. Tiered Model Complexity & Probability Calibration

### 3.1 The 5-Tier Evaluation Hierarchy

| Tier | Model Architecture | Specification | Epistemic Purpose |
| :--- | :--- | :--- | :--- |
| **Tier 0** | Majority Class Baseline | Predicts empirical prevalence $\bar{y}$ | Ground floor benchmark (AUROC = 0.5000) |
| **Tier 1** | Logistic Regression | $L_2$ Regularized, balanced weights | Interpretable linear benchmark |
| **Tier 2** | ElasticNet / Feature-Constrained | Sparsity-inducing regularization | Parameter-efficient linear model |
| **Tier 3** | Random Forest | 100 trees, max depth 6 | Non-linear interaction baseline |
| **Tier 4** | HistGradientBoosting / LightGBM | Early stopping, depth 4 | High-capacity non-linear model |
| **Tier +** | Trajectory-Augmented | Static features + Longitudinal trends | Quantifies empirical value of temporal data ($\Delta \text{AUC}$) |

### 3.2 Probability Calibration Diagnostics

For student support advising, high AUROC alone is insufficient: an advisor allocating emergency scholarships needs to know whether a predicted 70% departure probability truly corresponds to 7 out of 10 students departing. SSIF evaluates calibration using three metrics:

1. **Brier Score Loss:**  
   $$\text{Brier} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$$
2. **Expected Calibration Error (ECE):**  
   Partitions predictions into $M=10$ equal-width probability bins $B_m$:
   $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
3. **Maximum Calibration Error (MCE):**  
   Measures worst-case bin divergence:
   $$\text{MCE} = \max_{m \in \{1 \dots M\}} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
4. **Post-Hoc Calibration:**  
   Uncalibrated ensembles are calibrated via Platt scaling (logistic sigmoid) or Isotonic regression fitted strictly on cross-validation folds using `CalibratedClassifierCV`.

---

## 4. Causal Machine Learning: Double Machine Learning & Sensitivity

### 4.1 Robinson Partially Linear Model (PLR)

To measure the true causal impact of institutional policy interventions (e.g., awarding a scholarship $T$) on academic outcomes $Y$ (departure risk or GPA), SSIF uses the Double Machine Learning (DML) framework:

$$Y = D \cdot \theta_0 + g_0(X) + U, \quad \mathbb{E}[U \mid D, X] = 0$$
$$D = m_0(X) + V, \quad \mathbb{E}[V \mid X] = 0$$

- **Nuisance Estimation:** Machine learning estimators (Random Forest or Gradient Boosting) estimate conditional outcome $\hat{\ell}(X) = \mathbb{E}[Y \mid X]$ and propensity score $\hat{m}(X) = \mathbb{E}[D \mid X]$.
- **Cross-Fitting:** Out-of-fold cross-fitting eliminates regularization bias.
- **Orthogonal Neyman Score:**
  $$\hat{\theta}_0 = \frac{\sum_{i} (D_i - \hat{m}(X_i))(Y_i - \hat{\ell}(X_i))}{\sum_{i} (D_i - \hat{m}(X_i))^2}$$

### 4.2 Sensitivity Analysis via E-Values

Because observational datasets cannot rule out unmeasured confounders (e.g., intrinsic motivation), SSIF computes the **E-value** for every causal claim:

$$\text{E-value} = \text{RR} + \sqrt{\text{RR}(\text{RR} - 1)}$$

The E-value establishes the minimum strength of association that an unmeasured confounder must have with both the treatment and the outcome to explain away the observed effect.

---

## 5. Algorithmic Counterfactual Recourse Optimization

Predicting failure without offering recourse is unhelpful for students. SSIF implements an algorithmic recourse solver (`src/explainability/recourse.py`):

1. **Optimization Objective:** Find the minimal-effort perturbation $\delta^*$ such that:
   $$\delta^* = \arg\min_{\delta \in \mathcal{A}} \|\delta\|_1 \quad \text{s.t.} \quad f(x + \delta) \le \tau_{\text{target}}$$
2. **Actionability Constraints ($\mathcal{A}$):**
   - Non-actionable immutable features (Age, First-Gen, Enrolled Semester) cannot be modified ($\delta_j = 0$).
   - Directional constraints: GPA slope and attendance can only be improved, not degraded.
   - Bounded levers: Work hours can be reduced by at most 15 hrs/week; financial stress can be relieved via emergency funding.
3. **Predictive Disclaimers:**  
   Every counterfactual action plan is explicitly flagged: **"PREDICTIVE COUNTERFACTUAL WARNING: This is a predictive-model counterfactual, not a guaranteed causal intervention. Model associations do not guarantee deterministic individual outcomes."**

---

## 6. Small-Sample EPV Regularization & Uncertainty Bounds

### 6.1 Constrained Degrees of Freedom ($N=215$)

The placement dataset represents an authentic institutional reality: small cohort scale ($N=215$, 148 placed, 67 unplaced).
- Standard logistic regression over 15+ features yields severe overfitting ($\text{EPV} < 4.5$).
- SSIF restricts candidate predictors to at most 5 high-leverage covariates (`degree_p`, `etest_p`, `ssc_p`, `workex`, `specialisation`), guaranteeing $\text{EPV} \ge 13.4$.

### 6.2 Uncertainty Intervals in API Output

Rather than returning overconfident 4-decimal point estimates, the production API calculates 95% confidence intervals reflecting sample scale:
$$\text{SE}_{\text{logit}} \approx \sqrt{\frac{1}{N \cdot p (1-p)}}$$
$$\text{CI}_{95} = \left[ \sigma(\text{logit} - 1.96 \cdot \text{SE}), \sigma(\text{logit} + 1.96 \cdot \text{SE}) \right]$$

---

## 7. The 5-Phase Empirical Experimentation Suite

| Experiment | Focus Area | Methodology | Artifacts Generated |
| :--- | :--- | :--- | :--- |
| **EXP-001** | Labor-Policy Simulation | OLS + Monte Carlo bootstrap ($N=200$) simulating conversion of survival employment to work-study. | GPA lift (+0.077), Risk reduction (-2.97 pp, 95% CI: [2.64, 3.29]). |
| **EXP-002** | Pipeline Resilience | 18 perturbation stress tests across S1 to S8 college lifecycle stages. | Identified Early S1 intervention as the highest systemic multiplier (+81.6 graduates / 1K students). |
| **EXP-003** | Socio-Economic Fairness | 65% Degree GPA hiring gate disparity analysis & small-cell suppression ($N < 5$). | Identified $N=992$ "Qualified-But-Excluded" high-potential resilient candidates. |
| **EXP-004** | Trajectory Forecasting | S1–S2 early signal evaluation with GroupKFold cross-validation. | S1–S2 signals alone predict career tier with $\text{AUC}=0.7469$, expanding to $0.8387$ by S4. |
| **EXP-005** | Intervention ROI | HiGHS Simplex Linear Programming optimization under resource budgets ($10K–$500K). | Advising delivers top entry ROI (0.0533/dollar) up to $100K; micro-scholarships blend at scale. |
