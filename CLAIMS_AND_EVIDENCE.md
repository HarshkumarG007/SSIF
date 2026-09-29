# CLAIMS AND EVIDENCE — SSIF Epistemic Register

> **Purpose:** This document maps every major SSIF claim to its supporting evidence, statistical design,
> known limitations, and epistemic strength category. It is intended to be read by reviewers, collaborators,
> and anyone wishing to understand exactly what the data can and cannot establish.
>
> **Core principle:** A claim is only as strong as its weakest identification assumption.
> This register makes those assumptions explicit.

---

## 1. Academic Retention & Survival

| # | Claim | Evidence | Statistical Design | Limitations | Epistemic Strength |
|---|-------|----------|--------------------|-------------|-------------------|
| R-01 | GPA trajectory slope is negatively correlated with dropout (r = −0.147) | OLS slopes on 79,239 student-semester rows | Vectorized causal filtration s ≤ t; no future leakage | Association only; financial crisis or stress may drive both GPA decline and dropout | **Predictive / Associational** |
| R-02 | Logistic Regression achieves AUROC = 0.8014 for retention prediction | 5-fold GroupKFold CV on 20,000 students | GroupKFold by Student_ID; train/test student isolation | Dataset-specific; external validity unknown; no calibration curve shown | **Predictive (internal)** |
| R-03 | First-generation students show HR ≈ 1.98× departure hazard | Cox PH model on 20,000 students | Semi-parametric Cox PH; log-rank p < 10^-15 | PH assumption may not hold; self-selection into first-gen status | **Associational (Cox model)** |
| R-04 | Scholarship recipients show HR ≈ 0.52× departure hazard | Cox PH model on 20,000 students | Same Cox PH model | Self-selection into scholarships is a major confound | **Associational (Cox model)** |
| R-05 | Advising associated with +73.1% higher recovery odds (OR = 1.731) after GPA collapse | Multivariate logistic regression on N=5,563 rebounders | Logistic regression with 10 covariates | Unmeasured motivation confound; selection bias likely | **Associational (observational)** |
| R-06 | Financial stress associated with −33.4% recovery odds (OR = 0.666) | Same multivariate regression | Same design | Reverse causality possible | **Associational (observational)** |

---

## 2. Trajectory Phenotypes & Clustering

| # | Claim | Evidence | Statistical Design | Limitations | Epistemic Strength |
|---|-------|----------|--------------------|-------------|-------------------|
| C-01 | Three distinct academic phenotypes exist (Stable, Erosion, Collapse) | K-Means (k=3) on 5 trajectory features | Silhouette score across k=2–5; B=1000 bootstrap ARI | May not replicate in other institutions; K-Means assumes spherical clusters | **Exploratory / Descriptive** |
| C-02 | Bootstrap ARI = 0.9703 confirms centroid stability | 1000 bootstrap resamples vs reference clustering | Bootstrap with replacement; 10 initializations per resample | ARI measures centroid stability, not construct validity | **Stability diagnostic** |

---

## 3. Placement & Employability

| # | Claim | Evidence | Statistical Design | Limitations | Epistemic Strength |
|---|-------|----------|--------------------|-------------|-------------------|
| P-01 | Logistic Regression AUROC = 0.9370 for placement prediction | Stratified 5-Fold CV; N=215 MBA candidates | Stratified k-fold; EPV=3.2 guarded | N=215 small; AUROC CI ≈ ±0.05–0.10; single-institution cohort | **Predictive (internal, low N)** |
| P-02 | Work experience associated with 86.5% vs 59.6% placement (+26.9% lift) | Observed group difference; N=215 | Subgroup comparison | Self-selection: students who obtain work experience differ systematically — no matching | **Observational group difference** |
| P-03 | Starting salary R^2 ≈ 0.00 across GPA features; N=148 placed candidates | 5-Fold CV Ridge + RF regression | Standard k-fold regression | Very small N; zero R^2 consistent with fixed pay bands but also with insufficient power | **Negative finding (small N)** |

---

## 4. DLSM Compatibility & Feature Ablation

| # | Claim | Evidence | Statistical Design | Limitations | Epistemic Strength |
|---|-------|----------|--------------------|-------------|-------------------|
| D-01 | DLSM compatibility score = 0.154 (below 0.70 threshold → NO-GO) | Automated gate scoring schema overlap, granularity, identifier alignment | Rule-based gate | Threshold 0.70 is a domain-chosen heuristic, not statistically derived | **Governance decision** |
| D-02 | ΔAUROC = −0.00005 (p = 0.932) when adding DLSM demographic proxies | 5-fold GroupKFold ablation: baseline vs augmented model | Paired t-test on fold AUCs | Non-significant ≠ exactly zero effect; only Age/Gender tested, not paired biometric telemetry | **Negative evidence (specification-constrained)** |

---

## 5. Cross-Dataset Compatibility

| # | Claim | Evidence | Statistical Design | Limitations | Epistemic Strength |
|---|-------|----------|--------------------|-------------|-------------------|
| X-01 | Row-level merge is invalid (Wasserstein distance = 1.767 yrs) | KS test D=0.3048, p < 10^-15 on age distributions | Wasserstein distance + KS test | Distributional mismatch ≠ construct invalidity | **Statistically supported governance decision** |
| X-02 | A construct-level bridge is scientifically valid | Shared 4-stage latent vulnerability sequence identified | Conceptual mapping | Construct names do not guarantee measurement invariance | **Conceptual / Exploratory** |

---

## 6. Research Experiments (EXP-001–005)

| # | Experiment | Claim | Design | Limitation | Epistemic Strength |
|---|-----------|-------|--------|------------|-------------------|
| E-01 | EXP-001 | Work-study conversion associated with +0.077 GPA lift and −2.97 pp dropout risk | OLS + 200 Monte Carlo bootstrap | Observational; not an RCT; work-study assignment not random | **Simulated policy estimate (associational)** |
| E-02 | EXP-002 | Stage 1 (early) intervention yields highest system multiplier (+81.6 grads/1k) | 18-scenario lifecycle cascade simulation | Linear stage transition probabilities assumed — a strong simplification | **Simulation model output** |
| E-03 | EXP-003 | 65% GPA gate disproportionately excludes Q1 students; N=992 Qualified-But-Excluded | Fisher's exact test + trajectory-anchor profiling | Observational equity audit; hiring policies may have other rationale | **Equity audit finding** |
| E-04 | EXP-004 | Semester 1–2 signals: AUC=0.7469; expands to AUC=0.8387 by Semester 4 | GroupKFold expanding-window AUC | Within-dataset; may not replicate in cohorts with less early-semester variance | **Predictive (internal)** |
| E-05 | EXP-005 | Advising delivers highest entry ROI (0.0533 risk-reductions / dollar) | HiGHS LP optimization | ROI depends on assumed cost and effect-size inputs drawn from observational data | **Model-based optimization under assumptions** |

---

## 7. Causal Double Machine Learning (Under Stated Identification Assumptions)

| # | Claim | Evidence | Statistical Design | Identification Assumptions | Epistemic Strength |
|---|-------|----------|--------------------|---------------------------|-------------------|
| ML-01 | Scholarship → ATE = −4.66 pp dropout | DoubleMLPLR with GroupKFold cross-fitting; p < 10^-6; E-value = 1.27 | Neyman-orthogonal; k=5 GroupKFold | **(1) No unobserved confounders conditional on covariates; (2) Correct nuisance model spec; (3) Overlap/positivity.** None guaranteed in observational data. | **Conditional causal estimate (under stated assumptions)** |
| ML-02 | Scholarship → ATE = +0.024 GPA lift | Same DoubleMLPLR design | Same cross-fitting | Same identification assumptions | **Conditional causal estimate (under stated assumptions)** |
| ML-03 | E-value = 1.27 for dropout ATE | VanderWeele & Ding E-value | Closed-form E-value | E-value quantifies required confounding strength; does not rule out confounding | **Sensitivity diagnostic** |

---

## 8. Algorithmic Recourse (Predictive Model Counterfactuals)

| # | Claim | Evidence | Statistical Design | Limitations | Epistemic Strength |
|---|-------|----------|--------------------|-------------|-------------------|
| REC-01 | Recourse plan reduces model-predicted dropout from 68.4% → 14.8% | Constrained L1 optimization against trained logistic classifier | Domain-constrained optimization; immutable variables locked | **This is a predictive-model counterfactual, not a causal intervention.** Real-world outcomes may differ from model predictions. Advisors must treat recourse as hypothesis-generating guidance. | **Predictive recourse (model-based, not causal)** |

---

## 9. Fairness & Data Quality

| # | Claim | Evidence | Statistical Design | Limitations | Epistemic Strength |
|---|-------|----------|--------------------|-------------|-------------------|
| F-01 | Some apparent gender selection-rate disparity was caused by inconsistent encoding | Selection ratio: 0.39 → parity after canonicalization | Before/after with 8 variants → 4 canonical cohorts | Establishes encoding problem; does not establish comprehensive model fairness | **Data-quality finding** |

---

## 10. Epistemic Strength Legend

| Strength Label | Meaning |
|---|---|
| **Predictive (internal)** | Model achieves the stated metric within this specific dataset's CV procedure. External validity untested. |
| **Associational (observational)** | Statistical association under stated model; unmeasured confounders may explain the association. |
| **Conditional causal estimate** | Causal interpretation valid only under stated identification assumptions (no unobserved confounders, correct model, overlap). |
| **Negative finding** | No statistically detectable effect under this specification; does not rule out non-zero true effect. |
| **Simulation / Model output** | Result depends on model structure and input assumptions; not directly empirically derived. |
| **Exploratory / Descriptive** | Pattern described in data; no inferential claim intended. |
| **Stability diagnostic** | Validates computational stability; not construct validity. |
| **Governance decision** | Rule-based threshold enforced; threshold value is a domain choice, not a statistical derivation. |

---

> **Last Updated:** 2026-09-30
> **Related files:** [`RED_TEAM.md`](RED_TEAM.md) · [`docs/Rules.md`](docs/Rules.md) · [`src/causal/double_ml.py`](src/causal/double_ml.py)
