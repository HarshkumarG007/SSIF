"""
missingness_analyzer.py — Missingness Mechanism & Pattern Analyzer
Student Success Intelligence Framework (SSIF)

Analyzes missingness patterns to determine whether data is:
  - MCAR (Missing Completely At Random)
  - MAR (Missing At Random — depends on observed covariates)
  - MNAR / Structural (Missing Not At Random — e.g. unplaced salary)

RULE-007: Preprocessing fitted on training data only.
RULE-008: Never impute test data using test set statistics.
RULE-012: Log every data-cleaning decision.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.logger import get_module_logger

logger = get_module_logger("validation.missingness")


@dataclass
class ColumnMissingnessSummary:
    column: str
    n_missing: int
    pct_missing: float
    mechanism_diagnosis: str  # "MCAR", "MAR", "Structural/MNAR", "None"
    evidence: str
    correlated_predictors: list[tuple[str, float]] = field(default_factory=list)  # (col, p_value)
    recommended_strategy: str = ""


@dataclass
class MissingnessReport:
    dataset_name: str
    total_rows: int
    total_missing_cells: int
    overall_pct_missing: float
    column_summaries: list[ColumnMissingnessSummary] = field(default_factory=list)
    co_occurrence_matrix: dict[str, dict[str, int]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"=== Missingness Report: {self.dataset_name} ===",
            f"Total Rows: {self.total_rows:,} | Missing Cells: {self.total_missing_cells:,} ({self.overall_pct_missing:.2f}%)",
        ]
        if not self.column_summaries:
            lines.append("No missing values found across any columns. Complete dataset.")
        else:
            lines.append("\nColumn Diagnostics:")
            for s in self.column_summaries:
                lines.append(
                    f"  - {s.column}: {s.n_missing:,} missing ({s.pct_missing:.2f}%) | "
                    f"Diagnosis: {s.mechanism_diagnosis} | Strategy: {s.recommended_strategy}"
                )
                if s.evidence:
                    lines.append(f"    Evidence: {s.evidence}")
        if self.warnings:
            lines.append("\nWarnings:")
            for w in self.warnings:
                lines.append(f"  [!] {w}")
        return "\n".join(lines)


def analyze_column_mar(
    df: pd.DataFrame,
    missing_col: str,
    candidate_predictors: list[str] | None = None,
    alpha: float = 0.01,
) -> tuple[str, str, list[tuple[str, float]]]:
    """
    Test whether missingness in `missing_col` is significantly associated with observed covariates.
    Uses Welch's t-test for numeric predictors and Chi-square test for categorical predictors.
    """
    missing_indicator = df[missing_col].isna().astype(int)
    if missing_indicator.sum() == 0:
        return "None", "No missing values", []

    if candidate_predictors is None:
        candidate_predictors = [c for c in df.columns if c != missing_col]

    significant_predictors = []

    for pred in candidate_predictors:
        if pred not in df.columns:
            continue
        try:
            if pd.api.types.is_numeric_dtype(df[pred]):
                vals_missing = df.loc[missing_indicator == 1, pred].dropna()
                vals_observed = df.loc[missing_indicator == 0, pred].dropna()
                if len(vals_missing) >= 10 and len(vals_observed) >= 10:
                    stat, pval = stats.ttest_ind(vals_missing, vals_observed, equal_var=False)
                    if pval < alpha:
                        significant_predictors.append((pred, float(pval)))
            else:
                contingency = pd.crosstab(missing_indicator, df[pred])
                if contingency.size > 0 and (contingency.values > 0).all():
                    stat, pval, _, _ = stats.chi2_contingency(contingency)
                    if pval < alpha:
                        significant_predictors.append((pred, float(pval)))
        except Exception:
            continue

    if significant_predictors:
        evidence = f"Statistically associated with: {', '.join(p[0] for p in significant_predictors[:3])}"
        return "MAR", evidence, significant_predictors
    else:
        evidence = "No significant covariate associations found (p >= alpha). Consistent with MCAR."
        return "MCAR", evidence, []


def analyze_missingness(
    df: pd.DataFrame,
    dataset_name: str,
    alpha: float = 0.01,
) -> MissingnessReport:
    """
    Generate comprehensive missingness report for any dataset.
    """
    total_rows = len(df)
    missing_counts = df.isna().sum()
    missing_cols = missing_counts[missing_counts > 0].index.tolist()
    total_missing_cells = int(missing_counts.sum())
    overall_pct = (total_missing_cells / (total_rows * len(df.columns)) * 100) if total_rows > 0 else 0.0

    report = MissingnessReport(
        dataset_name=dataset_name,
        total_rows=total_rows,
        total_missing_cells=total_missing_cells,
        overall_pct_missing=overall_pct,
    )

    if not missing_cols:
        return report

    # Co-occurrence matrix
    if len(missing_cols) > 1:
        co_occ = {}
        for c1 in missing_cols:
            co_occ[c1] = {}
            for c2 in missing_cols:
                co_occ[c1][c2] = int((df[c1].isna() & df[c2].isna()).sum())
        report.co_occurrence_matrix = co_occ

    for col in missing_cols:
        n_missing = int(missing_counts[col])
        pct_missing = float(n_missing / total_rows * 100)

        # Domain-specific structural diagnosis
        if dataset_name.lower().find("placement") >= 0 and col == "salary":
            if "status" in df.columns:
                not_placed_mask = df["status"] == "Not Placed"
                if (df.loc[not_placed_mask, "salary"].isna()).all() and (not df.loc[~not_placed_mask, "salary"].isna().any()):
                    diag = "Structural/MNAR"
                    ev = "100% of missing salary corresponds to unplaced candidates. Structural missingness by design."
                    strat = "Filter subset for placed candidates when predicting salary; do not impute."
                    sig = [("status", 0.0)]
                else:
                    diag, ev, sig = analyze_column_mar(df, col, alpha=alpha)
                    strat = "Conditional model"
            else:
                diag, ev, sig = analyze_column_mar(df, col, alpha=alpha)
                strat = "Investigate target conditional relationship"
        else:
            diag, ev, sig = analyze_column_mar(df, col, alpha=alpha)
            if diag == "MAR":
                strat = "Fit MedianImputer or IterativeImputer on training folds ONLY (RULE-007, RULE-008)."
            else:
                strat = "Fit SimpleImputer(strategy='median') on training folds ONLY."

        report.column_summaries.append(
            ColumnMissingnessSummary(
                column=col,
                n_missing=n_missing,
                pct_missing=pct_missing,
                mechanism_diagnosis=diag,
                evidence=ev,
                correlated_predictors=sig,
                recommended_strategy=strat,
            )
        )

    logger.info(
        "[Missingness] Analyzed %s: %d missing cols, overall %.2f%% missing",
        dataset_name, len(missing_cols), overall_pct
    )
    return report
