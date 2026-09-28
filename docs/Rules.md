# Rules.md — Engineering Constitution
# Student Success Intelligence Framework (SSIF)

**Version:** 1.0  
**Status:** Active & Binding — applies to all agents, contributors, and AI coding assistants  
**Authority:** Overridden only by a documented exception in memory.md signed by Lead Researcher

---

## PART I — ABSOLUTE SCIENTIFIC RULES (Rules 001–015)

These rules protect the scientific validity of the project. They may NEVER be violated without explicit lead researcher approval and a documented exception in memory.md.

---

### RULE-001: Data Before Hypothesis
Inspect the actual dataset schemas before implementing any model. The confirmed empirical schemas in PRD.md are ground truth. Do not assume column names from Kaggle descriptions.

### RULE-002: Never Fabricate Student Identifiers
Student_ID values in the retention dataset (STU_00001 to STU_20000) and sl_no in placement (1–215) are real dataset identifiers. Never create synthetic linkage keys that pretend to join these two populations. These are different people.

### RULE-003: Never Row-Merge Retention and Placement
The following code patterns are FORBIDDEN in any production module:
```python
# FORBIDDEN
pd.merge(retention_df, placement_df, ...)
pd.concat([retention_df, placement_df])
retention_df.join(placement_df, ...)
```
Cross-dataset analysis must use representation-level comparison only.

### RULE-004: Never Claim DLSM Compatibility Without Evidence
DLSM requires: social_media_hours, AI_usage_hours, sleep_hours, physical_activity, bedtime_phone_minutes, screen_brightness. These variables are ABSENT from both SSIF datasets. The DLSM integration verdict is NO-GO unless new datasets with these variables are added.

### RULE-005: Run the DLSM Compatibility Gate Before Any Integration
Even if future datasets are added, run src/dlsm/compatibility_gate.py and document the output in memory.md before any DLSM feature engineering.

### RULE-006: Baseline Before Complexity
Establish a naive baseline (majority class predictor or mean predictor) and a logistic regression baseline before implementing Random Forest, XGBoost, or any advanced model. Complexity must demonstrate improvement over the simpler model.

### RULE-007: Preprocessing Fitted on Training Data Only
All preprocessing (imputation, scaling, encoding, feature selection) must be fit exclusively on training fold data. Transformations are then applied to test folds. Use sklearn.pipeline.Pipeline to enforce this.
```python
# CORRECT
pipe = Pipeline([('imputer', SimpleImputer()), ('model', LogisticRegression())])
pipe.fit(X_train, y_train)
pipe.predict(X_test)

# WRONG — leaks test set statistics
imputer.fit(X_all)  # FORBIDDEN
```

### RULE-008: Never Impute Test Data Using Test Set Statistics
The imputer must be fit on train data, then transform both train and test. No exceptions.

### RULE-009: No Future Information in Features
In the retention panel, Semester N data must never include any information from Semester N+1 or later as a feature. `Target_Dropout_Next_Sem` is the label; `End_of_Semester_Status` at the same semester row is the realized outcome — using both simultaneously is leakage.

### RULE-010: Post-Outcome Variables Are Forbidden as Predictors
In retention: `End_of_Semester_Status` describes what happened at end of semester. `Target_Dropout_Next_Sem` is the forward-looking label. These must not both appear in the feature set simultaneously. The leakage detector must flag this.

### RULE-011: Never Silently Drop Missing Observations
All imputation and exclusion decisions must be logged. Every dropped row must have a documented reason. The counts must match validation report totals.
- Family_Income: 3,604 missing → document imputation strategy chosen
- LMS_Logins: 867 missing → document imputation strategy chosen

### RULE-012: Log Every Data-Cleaning Decision
All transformations, drops, imputations, and encodings must be logged to a cleaning log. The log is part of the experimental artifact.

### RULE-013: Never Remove Outliers Solely to Improve Model Error
Outliers may be removed only if they are verified data errors (e.g., Age < 10, GPA > 4.0) with documentation. Never remove an outlier merely because it reduces RMSE or improves AUROC.

### RULE-014: Use GroupKFold for Retention Dataset
The retention dataset has multiple rows per student (longitudinal panel). A student's semesters must NEVER be split across train and test folds. Always use `GroupKFold(groups=df['Student_ID'])` or equivalent.

### RULE-015: SHAP Is Not Causal Evidence
SHAP values represent model attribution, not causal importance. All SHAP interpretations must use phrasing such as:
- ✅ "The model relied strongly on Sem_GPA"
- ✅ "Sem_GPA had high SHAP attribution"
- ❌ "Sem_GPA caused dropout"
- ❌ "Financial_Stress leads to dropout"

---

## PART II — STATISTICAL VALIDITY RULES (Rules 016–025)

### RULE-016: Predictive Accuracy ≠ Causal Proof
A model with AUROC=0.85 does not establish that the top features cause the outcome. Observational data establishes association, not causation. All research outputs must respect this distinction.

### RULE-017: Never Force Clusters When Stability Is Poor
Cluster analysis must report stability metrics: Silhouette score, Davies-Bouldin, bootstrap ARI. If bootstrap ARI < 0.70, the clustering result must be reported as unstable and not interpreted as meaningful.

### RULE-018: Never Name a Component Without Construct Validity
Engineered PCA/FA components must not be named "Student Success," "Academic Risk," or "Burnout Index" unless the naming is validated through construct validity analysis (loading interpretation + factor stability). Prefer descriptive names tied to actual variables.

### RULE-019: Null Results Are Valid Scientific Results
If trajectory features do not improve AUROC, report it. If DLSM does not add incremental value, report it. If clusters are unstable, report it. Null results are not failures — they are evidence.

### RULE-020: Report Uncertainty for All Major Results
All major metrics must include 95% confidence intervals. Acceptable methods: bootstrap CI, cross-validation standard deviation, or parametric CI where assumptions hold. Never report a point estimate alone for primary results.

### RULE-021: Preserve Dataset Provenance
Every dataframe must carry metadata about its origin:
- Which raw file it came from
- Which preprocessing steps were applied
- Which version it represents (timestamp or hash)

### RULE-022: All Experiments Must Be Reproducible
Every model training run must be logged to MLflow with: random_seed=42, dataset_version, feature_version, model_params, all metrics, artifact paths. Re-running with the same parameters must produce the same results.

### RULE-023: Every Major Model Must Have an Explainability Artifact
Tree-based models: SHAP summary plot + at least 3 local waterfall plots saved as artifacts. Linear models: coefficient plot with confidence intervals.

### RULE-024: Every Research Claim Must Be Traceable to an Experiment
No statement in any report or dashboard may claim a result not backed by a logged MLflow experiment run with a documented experiment_id.

### RULE-025: If Data Cannot Answer a Question, Say So
If the placement dataset (N=215) is too small to produce stable interaction analysis, the code must output: "INSUFFICIENT SAMPLE: This analysis requires N > 500 for stability. Results are exploratory only." Never present an underpowered result as definitive.

---

## PART III — MACHINE LEARNING ENGINEERING RULES (Rules 026–040)

### RULE-026: Prefer Simpler Models When Performance Is Comparable
If logistic regression achieves AUROC=0.78 and XGBoost achieves AUROC=0.80, the difference (0.02) must be evaluated for practical significance. If not meaningful, use logistic regression. Complexity must be justified.

### RULE-027: Never Hide Negative or Null Results
Negative experiment results must be logged to MLflow and recorded in memory.md. No selective reporting. No deleting failed experiments.

### RULE-028: Never Manufacture Statistical Significance
Do not run multiple tests hoping for p < 0.05, then report only significant ones. Correct for multiple comparisons (Bonferroni, FDR) when testing multiple hypotheses simultaneously.

### RULE-029: Never Alter DLSM Methodology Without Documentation
If any component of DLSM logic is modified (e.g., DLL formula, feature engineering), the change must be documented in memory.md with: what was changed, why, and what validation was performed.

### RULE-030: Scientific Validity > Engineering Convenience
When there is a conflict between making the code cleaner/simpler and maintaining scientific validity, scientific validity wins. Always.

### RULE-031: Retention Validation Must Always Use GroupKFold
GroupKFold(n_splits=5, groups=Student_ID) is mandatory for all retention model evaluation. Stratified K-Fold without grouping is FORBIDDEN for this dataset.

### RULE-032: Salary Regression Is a Separate Model
Never merge placement probability and expected salary into a single model or a single score. They are different prediction targets and must be evaluated separately.

### RULE-033: Imbalanced Classes Must Be Handled Explicitly
Dropout rate in retention: ~8.73% of rows. Handle with: class_weight='balanced', stratified sampling, or SMOTE (training data only, never test data). Report metrics on minority class explicitly (recall, precision, F1 for dropout=1).

### RULE-034: Hyperparameter Tuning Must Not Leak Test Information
All hyperparameter search (GridSearch, RandomizedSearch, Optuna) must operate inside cross-validation on training folds only. The test/holdout set is never touched during tuning.

### RULE-035: No Feature Selection Using Test Data
Feature importance, correlation filters, and selection methods must be computed on training data only and applied to test data without re-fitting.

### RULE-036: Calibrate Classifiers
All classifiers must have calibration evaluated using: Calibration curve (reliability diagram), Brier score, Expected Calibration Error. A well-discriminating but poorly calibrated model must be reported as such.

### RULE-037: Trajectory Features Are Only Valid If Temporal Ordering Is Preserved
Trajectory slope, velocity, and acceleration features must be computed in semester order (1 → 8). Never sort by GPA or any other variable before computing trajectory.

### RULE-038: Survival Analysis Requires Censoring to Be Respected
KM estimator, Cox model, and Random Survival Forest must receive both event and censoring flags. `Censored=1` rows must be treated as right-censored (event not observed), not as non-events.

### RULE-039: Do Not Over-Interpret Survival Results with Few Late Events
At Semester 8, only 2,172 students remain. Survival estimates at late time points have wide confidence bands. These must be plotted with CIs and not interpreted as precise estimates.

### RULE-040: Always Report AUC-ROC AND AUC-PR
For imbalanced classification (dropout: ~8.73%), AUC-ROC alone is misleading. Always report both AUC-ROC and AUC-PR (Precision-Recall). Prefer PR-AUC as the primary metric for dropout detection.

---

## PART IV — CODE QUALITY RULES (Rules 041–050)

### RULE-041: PEP 8 Compliance
All Python code must pass `ruff check .` without errors. Style is enforced by the linter, not by manual review.

### RULE-042: Type Hints Are Mandatory
All function signatures must include type hints. No bare `def f(x):` without type annotation.
```python
# CORRECT
def compute_trajectory_slope(df: pd.DataFrame, student_id: str) -> float:

# WRONG
def compute_trajectory_slope(df, student_id):
```

### RULE-043: Docstrings for All Public Functions
Every public function and class must have a docstring explaining: purpose, parameters, returns, raises (if applicable), and notes about scientific assumptions.

### RULE-044: Hard-Coded Paths Are Forbidden
Use `configs/data.yaml` and the `src/config.py` loader. Never hardcode:
```python
# WRONG
df = pd.read_csv("C:/Users/Lenovo/Downloads/SSIF/academic_survival_longitudinal.csv")

# CORRECT
df = pd.read_csv(cfg.data.retention_path)
```

### RULE-045: Small Functions, Single Responsibility
Functions must not exceed 50 lines. If a function grows beyond 50 lines, decompose it. Each function does one thing.

### RULE-046: Tests Must Run After Any Meaningful Change
After modifying any `src/` module, run `pytest tests/` and confirm passing. Do not commit code with failing tests.

### RULE-047: No Circular Imports
Module dependency graph must be a DAG. `retention/` must not import from `placement/` and vice versa. Both may import from `validation/`, `models/`, `statistics/`, `explainability/`.

### RULE-048: Log Level Discipline
- DEBUG: internal computation steps
- INFO: major pipeline milestones (e.g., "Retention baseline training complete")
- WARNING: potential data quality issues (e.g., "867 missing LMS_Logins — imputing")
- ERROR: validation failures that halt the pipeline
- CRITICAL: reserved for data corruption or security events

### RULE-049: No print() in Production Code
All output must use the Python logging module. `print()` is only acceptable in notebooks and temporary scripts.

### RULE-050: Memory.md Must Be Updated After Significant Work
After completing any Phase in task.md, the agent must update memory.md with: current status, decisions made, experiment results, limitations discovered, and next task. Memory must remain the single source of truth for project state.

---

## PART V — AI AGENT BEHAVIOR RULES (Rules 051–060)

### RULE-051: Read Before Writing
Before modifying any existing file, read its current contents. Never overwrite a file without understanding what is already there.

### RULE-052: Read PRD.md and memory.md Before Starting Any Task
At the start of each session, read: `docs/PRD.md` (requirements), `docs/memory.md` (current state), relevant task from `docs/task.md`. Do not proceed without this context.

### RULE-053: Make the Smallest Correct Change
Do not refactor or rewrite components unrelated to the current task. Avoid scope creep. If a refactor is needed, create a separate task for it.

### RULE-054: Explain Major Architectural Decisions
Before implementing any significant structural change, output an explanation of what is being changed, why, and what the alternatives were. Record the decision in memory.md.

### RULE-055: Never Rewrite Working Systems Without Reason
If a component passes tests and produces valid scientific output, do not rewrite it merely for aesthetic improvement. "Working is better than clean" in research pipelines.

### RULE-056: Inspect DLSM Repository Before Touching DLSM-Adjacent Code
Before writing any code in `src/dlsm/`, review: the DLSM feature_dictionary.yaml, the DLSM README compatibility notes, and the current NO-GO verdict in PRD.md.

### RULE-057: Report After Every Significant Task
After completing a task, output a structured report:
```
Task Completed: [TASK-ID]
Files Modified: [list]
Tests Status: [passing/failing]
Key Finding: [one sentence]
Scientific Impact: [what this changes scientifically]
Next Task: [TASK-ID]
Limitations Discovered: [any new ones]
```

### RULE-058: Do Not Assume Kaggle Descriptions Are Accurate
The actual CSV schemas are the ground truth. The Kaggle dataset description for Dataset B (DLSM) incorrectly implied academic grades were present — the actual schema has no grades column. Always verify from the file.

### RULE-059: Surface Conflicts — Never Resolve Them Silently
If two documents (e.g., PRD.md and task.md) contradict each other, or if the actual data contradicts a documented assumption, surface the conflict immediately. Do not silently resolve it by choosing one side.

### RULE-060: The Vibe Coding Lifecycle Must Not Be Skipped
We do not jump from "idea" to "code." The lifecycle is: Research → PRD → Architecture → Rules → Design → Tasks → Setup → Develop → Test → Review → QA → Deploy. Each stage must be completed before the next begins.

### RULE-061: Literature Plausibility Calibration (Orben & Przybylski Ceiling)
Any observational model or feature claiming direct cross-domain impact between digital lifestyle metrics and academic attrition/mental health outcomes exceeding the pre-registered specification-curve ceiling ($R^2 \le 0.004$ / $0.4\%$, Orben & Przybylski 2019, $n=355,358$) must be treated as synthetic generation artifact or data leakage until confirmed via prospective causal tracking.

### RULE-062: Panel Survivorship Bias Guard
In longitudinal survival panels, total lifetime observation counts or whole-trajectory duration (e.g., total semesters observed) must never enter feature matrices predicting semester-level dropout. All trajectory features must be causally bounded strictly to the historical filtration $\mathcal{F}_{i,t} = \{s \le t\}$.

---

*This Rules.md is the engineering constitution for SSIF.*  
*Violations must be documented in memory.md under RULE VIOLATIONS.*  
*No rule may be silently suspended.*

