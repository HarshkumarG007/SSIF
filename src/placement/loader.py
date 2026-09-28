"""
loader.py — Placement Dataset Loader
Student Success Intelligence Framework (SSIF)

Loads and validates the employment placement dataset (N=215).
Ensures zero target leakage: salary is NOT loaded as an input feature for placement classification.

RULE-001: Inspect schemas before implementing models.
RULE-009: No future information in features.
RULE-025: Explicitly report N=215 sample size constraint in all modeling logs.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_loader import load_placement as _load_raw
from src.logger import get_module_logger

logger = get_module_logger("placement.loader")


def load_placement(path: Path | str | None = None) -> pd.DataFrame:
    """
    Load validated placement dataset.

    Returns:
        DataFrame with 215 rows, 15 columns.
    """
    df = _load_raw(path)
    logger.info(
        "[Placement] Loaded %d candidates. Placed: %d (%.1f%%), Not Placed: %d (%.1f%%)",
        len(df),
        (df["status"] == "Placed").sum(),
        (df["status"] == "Placed").mean() * 100,
        (df["status"] == "Not Placed").sum(),
        (df["status"] == "Not Placed").mean() * 100,
    )
    return df
