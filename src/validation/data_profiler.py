"""
data_profiler.py — Automated Data Profiler
Student Success Intelligence Framework (SSIF)

Generates a comprehensive statistical profile of any DataFrame.
Output is used for the dashboard Data Audit page and audit reports.

RULE-001: Inspect schemas before implementing models.
RULE-012: Log every data-cleaning decision.
RULE-020: Report uncertainty for all major results.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.logger import get_module_logger

logger = get_module_logger("validation.profiler")


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    n_total: int
    n_missing: int
    pct_missing: float
    n_unique: int
    # Numeric stats (None for categoricals)
    mean: float | None = None
    std: float | None = None
    min_val: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    max_val: float | None = None
    skewness: float | None = None
    # Categorical stats
    top_values: dict[str, int] = field(default_factory=dict)
    # Quality flags
    is_constant: bool = False
    has_infinite: bool = False
    outlier_pct: float = 0.0


@dataclass
class DatasetProfile:
    name: str
    n_rows: int
    n_cols: int
    total_missing_cells: int
    pct_missing_overall: float
    duplicate_rows: int
    columns: list[ColumnProfile] = field(default_factory=list)
    quality_warnings: list[str] = field(default_factory=list)
    # For longitudinal data
    id_col: str | None = None
    n_unique_ids: int | None = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert column profiles to a summary DataFrame."""
        rows = []
        for c in self.columns:
            rows.append({
                "Column": c.name,
                "Type": c.dtype,
                "Missing": c.n_missing,
                "Missing%": f"{c.pct_missing:.1f}%",
                "Unique": c.n_unique,
                "Mean": f"{c.mean:.3f}" if c.mean is not None else "—",
                "Std": f"{c.std:.3f}" if c.std is not None else "—",
                "Min": f"{c.min_val:.2f}" if c.min_val is not None else "—",
                "Median": f"{c.p50:.3f}" if c.p50 is not None else "—",
                "Max": f"{c.max_val:.2f}" if c.max_val is not None else "—",
                "Skew": f"{c.skewness:.2f}" if c.skewness is not None else "—",
                "Constant?": "⚠️ YES" if c.is_constant else "OK",
                "Outliers%": f"{c.outlier_pct:.1f}%" if c.outlier_pct > 0 else "—",
            })
        return pd.DataFrame(rows)

    def summary_text(self) -> str:
        lines = [
            f"Dataset: {self.name}",
            f"  Rows: {self.n_rows:,}",
            f"  Columns: {self.n_cols}",
            f"  Missing cells: {self.total_missing_cells:,} ({self.pct_missing_overall:.2f}%)",
            f"  Duplicate rows: {self.duplicate_rows:,}",
        ]
        if self.n_unique_ids is not None:
            lines.append(f"  Unique {self.id_col}: {self.n_unique_ids:,}")
        if self.quality_warnings:
            lines.append("  ⚠️  Quality warnings:")
            for w in self.quality_warnings:
                lines.append(f"    - {w}")
        return "\n".join(lines)


def _profile_column(series: pd.Series, n_top: int = 5) -> ColumnProfile:
    """Build a ColumnProfile for a single Series."""
    n_total = len(series)
    n_missing = series.isna().sum()
    pct_missing = (n_missing / n_total * 100) if n_total > 0 else 0.0
    n_unique = series.nunique(dropna=True)
    is_constant = n_unique <= 1

    profile = ColumnProfile(
        name=series.name,
        dtype=str(series.dtype),
        n_total=n_total,
        n_missing=int(n_missing),
        pct_missing=round(pct_missing, 4),
        n_unique=int(n_unique),
        is_constant=is_constant,
    )

    if pd.api.types.is_numeric_dtype(series):
        clean = series.dropna()
        has_inf = np.isinf(clean).any() if len(clean) > 0 else False
        clean = clean.replace([np.inf, -np.inf], np.nan).dropna()
        profile.has_infinite = bool(has_inf)

        if len(clean) > 0:
            profile.mean = float(clean.mean())
            profile.std = float(clean.std())
            profile.min_val = float(clean.min())
            profile.p25 = float(clean.quantile(0.25))
            profile.p50 = float(clean.median())
            profile.p75 = float(clean.quantile(0.75))
            profile.max_val = float(clean.max())
            profile.skewness = float(clean.skew())

            # IQR-based outlier detection
            iqr = profile.p75 - profile.p25
            if iqr > 0:
                lo = profile.p25 - 1.5 * iqr
                hi = profile.p75 + 1.5 * iqr
                n_outliers = ((clean < lo) | (clean > hi)).sum()
                profile.outlier_pct = float(n_outliers / len(clean) * 100)
    else:
        # Categorical: top N value counts
        vc = series.value_counts(dropna=True).head(n_top)
        profile.top_values = {str(k): int(v) for k, v in vc.items()}

    return profile


def profile_dataset(
    df: pd.DataFrame,
    name: str,
    id_col: str | None = None,
    quality_checks: bool = True,
) -> DatasetProfile:
    """
    Generate a full statistical profile of a DataFrame.

    Args:
        df: Input DataFrame to profile.
        name: Human-readable name (used in reports).
        id_col: Optional ID column to count unique entities.
        quality_checks: If True, generate quality warnings.

    Returns:
        DatasetProfile with column-level statistics and quality flags.
    """
    logger.info("[Profiler] Profiling dataset: %s (%d rows, %d cols)", name, len(df), len(df.columns))

    total_missing = int(df.isna().sum().sum())
    pct_missing_overall = (total_missing / (len(df) * len(df.columns)) * 100) if len(df) > 0 else 0.0
    duplicate_rows = int(df.duplicated().sum())

    profile = DatasetProfile(
        name=name,
        n_rows=len(df),
        n_cols=len(df.columns),
        total_missing_cells=total_missing,
        pct_missing_overall=round(pct_missing_overall, 4),
        duplicate_rows=duplicate_rows,
        id_col=id_col,
        n_unique_ids=int(df[id_col].nunique()) if id_col and id_col in df.columns else None,
    )

    # Profile each column
    for col in df.columns:
        profile.columns.append(_profile_column(df[col]))

    # Quality warnings
    if quality_checks:
        warnings = []
        for c in profile.columns:
            if c.is_constant:
                warnings.append(f"Column '{c.name}' is constant — zero variance")
            if c.has_infinite:
                warnings.append(f"Column '{c.name}' contains infinite values")
            if c.pct_missing > 10.0:
                warnings.append(f"Column '{c.name}' has {c.pct_missing:.1f}% missing values (HIGH)")
            elif c.pct_missing > 1.0:
                warnings.append(f"Column '{c.name}' has {c.pct_missing:.1f}% missing values (moderate)")
            if c.outlier_pct > 5.0:
                warnings.append(f"Column '{c.name}' has {c.outlier_pct:.1f}% IQR outliers — investigate")
        if duplicate_rows > 0:
            warnings.append(f"{duplicate_rows:,} fully duplicate rows detected")
        profile.quality_warnings = warnings

    logger.info(
        "[Profiler] Done: %d missing cells (%.2f%%), %d warnings",
        total_missing, pct_missing_overall, len(profile.quality_warnings)
    )
    return profile


def profile_retention(df: pd.DataFrame) -> DatasetProfile:
    """Profile the retention dataset with domain-specific checks."""
    p = profile_dataset(df, "SSIF-A: Academic Persistence (Retention)", id_col="Student_ID")

    # Domain-specific: class balance check
    if "Target_Dropout_Next_Sem" in df.columns:
        dropout_rate = df["Target_Dropout_Next_Sem"].mean() * 100
        if dropout_rate < 15:
            p.quality_warnings.append(
                f"Class imbalance: dropout rate = {dropout_rate:.1f}% "
                "— use class_weight='balanced' or SMOTE (training only)"
            )

    # Domain-specific: temporal coverage
    if "Semester" in df.columns and "Student_ID" in df.columns:
        sem_counts = df.groupby("Student_ID")["Semester"].count()
        single_sem = (sem_counts == 1).sum()
        if single_sem > 0:
            p.quality_warnings.append(
                f"{single_sem:,} students have only 1 semester — trajectory features will be NaN for these"
            )

    return p


def profile_placement(df: pd.DataFrame) -> DatasetProfile:
    """Profile the placement dataset with domain-specific checks."""
    p = profile_dataset(df, "SSIF-B: Academic Placement (Employability)", id_col="sl_no")

    if "status" in df.columns:
        placed_rate = (df["status"] == "Placed").mean() * 100
        p.quality_warnings.append(
            f"N={len(df)} — SMALL DATASET. All results are exploratory only."
        )
        p.quality_warnings.append(
            f"Class balance: Placed={placed_rate:.1f}%, Not Placed={100-placed_rate:.1f}%"
        )

    if "salary" in df.columns:
        salary_missing = df["salary"].isna().sum()
        p.quality_warnings.append(
            f"Salary: {salary_missing} structurally missing (all 'Not Placed') — "
            "model salary on N=148 placed students ONLY"
        )

    return p


def profile_dlsm_a(df: pd.DataFrame) -> DatasetProfile:
    """Profile DLSM Dataset A."""
    return profile_dataset(df, "DLSM-A: Bedtime Screen Time & Sleep Debt", id_col="user_id")


def profile_dlsm_b(df: pd.DataFrame) -> DatasetProfile:
    """Profile DLSM Dataset B."""
    p = profile_dataset(df, "DLSM-B: AI & Social Media Student Health", id_col="Student_ID")
    p.quality_warnings.append(
        "DIFFERENT POPULATION from SSIF data — representation bridge permitted, row merge FORBIDDEN"
    )
    return p
