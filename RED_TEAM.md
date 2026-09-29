# RED TEAM — SSIF Adversarial Claim Assessment

> **Purpose:** This document proactively identifies the strongest possible attacks against each major
> SSIF finding and records the current status of each. It is designed to be read by hostile peer reviewers,
> statistical methodologists, and data scientists attempting to falsify SSIF's claims.
>
> **Epistemological stance:** Proactively publishing attack vectors against your own findings is
> a marker of scientific credibility. SSIF does not claim immunity from criticism — it claims
> to have thought harder than average about what could be wrong.

---

## Category 1: Causal Overinterpretation

---

### Claim: "Academic advising increases recovery odds."

**Strongest attack:**
> Students who seek academic advising are self-selected: they may already be more motivated,
> aware of their options, or have better support networks. The observed OR = 1.731 may largely
> reflect selection into advising rather than any causal effect of advising itself.
> A student randomized to receive advising who would not have sought it voluntarily may
> experience a much smaller or zero effect.

**Status:** ⚠️ NOT ESTABLISHED CAUSALLY
**Current defense:** Cox PH + multivariate logistic regression with 10 covariates;
Double ML E-value (1.18) quantifies required confounding strength.
**What would be needed:** Randomized controlled trial (RCT) or quasi-experimental design
(regression discontinuity, instrumental variable).
**Honest label:** Observational association with unmeasured confounding acknowledged.

---

### Claim: "Institutional scholarships cut dropout hazard in half."

**Strongest attack:**
> Scholarship recipients are selected — often by academic merit, financial need screening,
> or institutional priority. The scholarship group may differ from non-recipients in
> unmeasured dimensions (motivation, institutional belonging, part-time employment status).
> HR = 0.52 is a Cox PH association, not a causal estimate unless selection into
> scholarship is orthogonal to residual hazard after controlling for covariates.

**Status:** ⚠️ ASSOCIATIONAL ONLY (Cox PH model association)
**Current defense:** Cox PH with multiple covariates including financial stress, first-gen status;
DML ATE = −4.66 pp under identification assumptions.
**What would be needed:** Natural experiment (e.g., scholarship lottery cutoff) or IV design.
**Honest label:** Conditional causal estimate under stated DML assumptions; Cox HR is observational.

---

### Claim: "Work experience improves placement."

**Strongest attack:**
> Students who obtain work experience self-select. They may be more professionally proactive,
> have stronger networks, or belong to social strata with better access to internships.
> The 26.9% lift reflects group-level differences, not the causal effect of assigning
> an otherwise-identical student to work experience.

**Status:** ⚠️ OBSERVED GROUP DIFFERENCE, NOT CAUSAL ESTIMATE
**Current defense:** Subgroup analysis reported with explicit EPV limitation (N=215, EPV=3.2);
no causal language used in claims.
**What would be needed:** Propensity score matching, IV (e.g., mandatory internship policy),
or RCT design.
**Honest label:** Observational group difference with self-selection confound acknowledged.

---

### Claim: "DML ATE = −4.66 pp scholarship effect on dropout (under stated assumptions)."

**Strongest attack:**
> DML does not eliminate unobserved confounding. If there exists an unmeasured variable U
> (e.g. institutional belonging, family stability, peer network) that jointly determines
> scholarship eligibility and dropout risk, the DML estimate will be biased.
> The E-value of 1.27 means confounding of moderate strength could explain the result.

**Status:** ⚠️ CONDITIONAL CAUSAL ESTIMATE — identification assumptions may not hold
**Current defense:** Neyman-orthogonal cross-fitting eliminates regularization bias;
E-value sensitivity analysis published; GroupKFold prevents student-level leakage.
**What would be needed:** Valid instrumental variable for scholarship assignment (e.g. budget cutoff RD).
**Honest label:** "Under the assumption of no unobserved confounders conditional on the 10 included covariates,
the estimated ATE is −4.66 pp."

---

## Category 2: Predictive Recourse vs Real-World Intervention

---

### Claim: "Algorithmic recourse reduces dropout risk from 68.4% to 14.8%."

**Strongest attack:**
> The recourse engine produces a counterfactual feature vector that the predictive model
> scores as 14.8%. This does not mean that implementing those changes in the real world
> will produce a 14.8% dropout probability. Predictive models capture correlation structure;
> they do not automatically identify causal mechanisms. If the model is wrong about the
> causal pathway (e.g. advising works only for motivated students), the recourse is misleading.

**Status:** ⚠️ PREDICTIVE-MODEL COUNTERFACTUAL — not a causal intervention guarantee
**Current defense:** Distinction between "predictive recourse" and "causal intervention" is now
explicitly documented in the code comments, dashboard, and README.
**What would be needed:** Randomized evaluation of recourse-prescribed interventions.
**Honest label:** "The model assigns 14.8% probability to the counterfactual feature vector.
Real-world outcome may differ."

---

## Category 3: Null Result Interpretation

---

### Claim: "Digital lifestyle telemetry adds zero predictive signal (ΔAUROC = −0.00005)."

**Strongest attack:**
> The null result (p = 0.932) is specific to this specification: DLSM demographic proxies
> (Age, Gender) added to SSIF academic features. It does not rule out that:
> (a) paired individual-level real-time screen-time data would add signal;
> (b) different feature engineering of DLSM variables would yield lift;
> (c) measurement error in the DLSM proxies is attenuating a real effect.
> Absence of evidence ≠ evidence of absence.

**Status:** ⚠️ SPECIFICATION-CONSTRAINED NULL RESULT
**Current defense:** Explicitly calibrated against Orben & Przybylski (2019, n=355,358)
specification-curve literature; paired t-test confidence intervals reported; null result
stated as "no detectable incremental value under this specification" not "zero effect."
**What would be needed:** Paired individual biometric telemetry on the same students over the same period.
**Honest label:** "No statistically detectable incremental predictive value was observed under this specification."

---

## Category 4: Sample Size & External Validity

---

### Claim: "Placement AUROC = 0.9370 indicates strong employability prediction."

**Strongest attack:**
> With N=215 and EPV=3.2, the 95% confidence interval around AUROC=0.9370 is approximately
> [0.88, 0.99] — too wide to treat as a precise estimate. A spectacular AUROC at N=215 is
> consistent with overfitting despite CV, or genuine predictability due to very few
> decision-relevant variables in a homogeneous MBA cohort. The result may not generalize
> beyond this specific business school program.

**Status:** ⚠️ LOW-N RESULT — interpret with sample-size caution
**Current defense:** EPV explicitly reported; L2 regularization applied; confidence interval
reasoning mentioned in documentation; bootstrap CI not yet formally computed.
**Recommended addition:** Bootstrap 95% CI for AUROC; calibration curve.
**Honest label:** "AUROC = 0.9370 (5-fold CV; N=215; interpret with sample-size caution — CI ≈ ±0.05–0.10)."

---

### Claim: "Bootstrap ARI = 0.9703 confirms stable trajectory phenotypes."

**Strongest attack (before fix):**
> B=15 bootstrap iterations is far too small for a strong stability claim.
> A slightly different subsample could yield a substantially different ARI.

**Status:** ✅ ADDRESSED
**Fix applied:** Bootstrap iterations increased from B=15 → B=1000. ARI=0.9703 now evaluated
over 1,000 resamples, providing a credible stability estimate.
**Remaining limitation:** Bootstrap ARI measures centroid stability, not construct validity.
Phenotypes may not replicate across different institutions.

---

## Category 5: Overfitted Scope

---

### Attack: "SSIF is an elaborate engineering wrapper around public benchmark datasets with many exploratory hypotheses."

**Elements with ammunition:**
- The datasets are public Kaggle releases of uncertain provenance and institutional specificity.
- External validity language (e.g., "higher education") is broader than the evidence supports.
- Many hypotheses explored; multiple-comparison correction not uniformly applied.
- N=215 placement cohort is very small for the conclusions drawn.

**Elements that defeat the attack:**
- Explicit temporal leakage prevention (GroupKFold, RULE-062) prevents the most common ML inflation.
- Negative findings (DLSM null result, salary R² ≈ 0.00) are reported, not hidden.
- CLAIMS_AND_EVIDENCE.md explicitly states epistemic strength for each claim.
- 62 scientific governance rules codified and machine-enforced.
- DML causal engine with E-value sensitivity analysis.
- Cross-dataset compatibility gate that produces a NO-GO decision.

**Verdict:** Attack has some legitimate ammunition. The correct response is **tightening claims
and maintaining explicit limitation language**, not adding more features.

---

## Governance Rule Impact Map

The following rules materially changed results (not just bureaucracy):

| Rule | What It Does | Without It | With It |
|------|-------------|------------|---------|
| `RULE-009` | Forbids post-outcome variables as features | Synthetic AUC ≈ 1.000 (leakage) | Legitimate AUROC = 0.8014 |
| `RULE-062` | Forbids lifetime sequence length as feature | Survivorship-biased AUC ≈ 0.607 | Causal temporal estimate |
| `RULE-003` | Forbids row-level merge across datasets | Invalid "mega-dataset" conclusions | Construct bridge with NO-GO gate |
| `RULE-031` | Mandates GroupKFold by Student_ID | Student-correlation inflated AUC +0.08–0.12 | Legitimate cross-validated AUROC |
| `RULE-010` | Isolates salary to placed candidates only | Regression conflates classification + salary | Clean two-stage decoupling |
| `RULE-017` | Requires bootstrap ARI > 0.70 before naming phenotypes | Arbitrary cluster labeling | Stability-validated phenotype discovery |
| `RULE-061` | Calibrates DLSM results against Orben & Przybylski (2019) | Unanchored ΔAUROC claim | Literature-calibrated null result |

> **The governance system is not bureaucracy. It changes scientific conclusions.**

---

> **Last Updated:** 2026-09-30
> **Related files:** [`CLAIMS_AND_EVIDENCE.md`](CLAIMS_AND_EVIDENCE.md) · [`docs/Rules.md`](docs/Rules.md)
