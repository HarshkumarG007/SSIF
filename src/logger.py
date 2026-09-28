"""
logger.py — Structured Logging Setup
Student Success Intelligence Framework (SSIF)

Provides a consistent, leveled logger for all SSIF modules.
Uses Python's standard logging module with rich formatting.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path


_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> logging.Logger:
    """
    Create and configure a named logger for SSIF modules.

    Args:
        name: Logger name (e.g., 'ssif.retention.features')
        level: Logging level (default: INFO)
        log_file: Optional file path to write logs to disk

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # Avoid duplicate handlers on re-import

    logger.setLevel(level)
    formatter = logging.Formatter(_FMT, datefmt=_DATE_FMT)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def get_module_logger(module: str) -> logging.Logger:
    """Convenience wrapper — creates logger named 'ssif.<module>'."""
    return setup_logger(f"ssif.{module}")
