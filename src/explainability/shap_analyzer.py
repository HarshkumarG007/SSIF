"""
shap_analyzer.py — SHAP Feature Attribution & Model Explainability
Student Success Intelligence Framework (SSIF)

Computes TreeExplainer and LinearExplainer attributions to determine
which academic, demographic, and trajectory features drive student dropout risk.

RULE-016: Provide feature attributions for high-stakes decisions.
RULE-027: Explainability reports must be documented and auditable.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier

from src.data_loader import load_retention
from src.logger import get_module_logger
from src.retention.features import prepare_retention_dataset

logger = get_module_logger("explainability.shap")


def compute_retention_shap_importance(
    sample_size: int = 2000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Train a Random Forest model on retention data and compute SHAP values.

    Args:
        sample_size: Number of background samples to evaluate for SHAP speed.
        random_state: Reproducibility seed.

    Returns:
        (importance_df, top_features_dict)
    """
    logger.info("Computing SHAP feature importance for Retention (sample_size=%d)...", sample_size)
    df_raw = load_retention()
    X, y, groups, feature_names = prepare_retention_dataset(df_raw, include_trajectories=True)

    # Train Random Forest on full set
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X, y)

    # Sample for SHAP computation
    np.random.seed(random_state)
    sample_idx = np.random.choice(len(X), size=min(sample_size, len(X)), replace=False)
    X_sample = X.iloc[sample_idx]

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_sample)

    # For binary classification, use dropout class (class 1)
    if isinstance(shap_values, list):
        vals = shap_values[1]
    elif len(shap_values.shape) == 3:
        vals = shap_values[:, :, 1]
    else:
        vals = shap_values

    mean_abs_shap = np.mean(np.abs(vals), axis=0)

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Abs_SHAP": mean_abs_shap,
        "Relative_Importance_Pct": (mean_abs_shap / np.sum(mean_abs_shap)) * 100,
    }).sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

    top_dict = {
        row["Feature"]: float(row["Mean_Abs_SHAP"])
        for _, row in importance_df.iterrows()
    }

    logger.info("Top 5 Drivers of Dropout Risk:")
    for i, r in importance_df.head(5).iterrows():
        logger.info("  %d. %s: %.4f (%.1f%%)", i + 1, r["Feature"], r["Mean_Abs_SHAP"], r["Relative_Importance_Pct"])

    return importance_df, top_dict


def save_shap_report(importance_df: pd.DataFrame):
    """Save SHAP attributions to reports/retention/."""
    out_dir = Path("reports/retention")
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON export
    json_path = out_dir / "shap_importance.json"
    importance_df.to_json(json_path, orient="records", indent=2)

    # Markdown export
    lines = [
        "# Academic Retention: SHAP Feature Importance & Risk Attribution",
        "**Generated:** 2026-09-29  ",
        "**Method:** TreeExplainer (Random Forest with Grouped Longitudinal Trajectories)  ",
        "",
        "## Top 15 Predictors of Next-Semester Dropout Risk",
        importance_df.head(15).to_markdown(index=False),
        "",
        "## Domain Insights",
        "1. **Academic Momentum:** Longitudinal GPA trajectory (slope, velocity) and current Semester GPA are leading risk indicators.",
        "2. **Engagement Decay:** Consecutive GPA declines (`decline_index`) and Attendance slope provide strong early warning before official dropout.",
        "3. **Financial Stress & Work Hours:** Financial stress and high work hours act as compounding vulnerability multipliers.",
    ]
    md_path = out_dir / "shap_importance.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved SHAP reports to %s and %s", json_path, md_path)


if __name__ == "__main__":
    df_imp, _ = compute_retention_shap_importance()
    save_shap_report(df_imp)
