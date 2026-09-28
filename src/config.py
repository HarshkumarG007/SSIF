"""
config.py — Pydantic v2 Configuration Loader
Student Success Intelligence Framework (SSIF)

Loads and validates all YAML configuration files.
Enforces data types and paths at startup.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, field_validator, model_validator
from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class ColumnConfig(BaseModel):
    retention_id: str = "Student_ID"
    retention_target_binary: str = "Target_Dropout_Next_Sem"
    retention_target_multiclass: str = "End_of_Semester_Status"
    retention_censored: str = "Censored"
    retention_time: str = "Semester"
    placement_id: str = "sl_no"
    placement_target_classification: str = "status"
    placement_target_regression: str = "salary"
    dlsm_b_target: str = "Mental_Health_Score"


class ValidationConfig(BaseModel):
    random_state: int = 42
    test_size: float = 0.20
    n_splits: int = 5


class DataConfig(BaseModel):
    retention_path: str
    placement_path: str
    dlsm_a_path: str
    dlsm_b_path: str
    dlsm_root: str
    processed_dir: str = "data/processed"
    interim_dir: str = "data/interim"
    reports_dir: str = "reports"

    @field_validator("retention_path", "placement_path")
    @classmethod
    def path_must_exist(cls, v: str) -> str:
        p = Path(v)
        if not p.is_absolute():
            # Try resolving relative to project root
            p = Path.cwd() / v
        if not p.exists():
            raise ValueError(f"Data file not found: {p}")
        return str(p)


class DLSMCompatibilityConfig(BaseModel):
    retention_verdict: str = "NO-GO"
    placement_verdict: str = "NO-GO"
    reason: str
    shared_demographics: list[str] = ["Age", "Gender"]
    representation_bridge_allowed: bool = True
    row_merge_forbidden: bool = True


# ---------------------------------------------------------------------------
# Main Config
# ---------------------------------------------------------------------------

class SSIFConfig(BaseModel):
    """
    Master configuration object for the Student Success Intelligence Framework.
    
    Load via: cfg = load_config()
    Access via: cfg.data.retention_path, cfg.validation.random_state, etc.
    """
    data: DataConfig
    validation: ValidationConfig = ValidationConfig()
    dlsm_compatibility: DLSMCompatibilityConfig
    project_root: Path = Path.cwd()
    random_state: int = 42


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(config_dir: Optional[Path] = None) -> SSIFConfig:
    """
    Load and validate the SSIF configuration from YAML files.

    Args:
        config_dir: Path to configs/ directory. Defaults to <cwd>/configs/

    Returns:
        Validated SSIFConfig object.

    Raises:
        FileNotFoundError: If any required config file is missing.
        ValueError: If configuration values fail Pydantic validation.
    """
    if config_dir is None:
        config_dir = Path.cwd() / "configs"

    config_dir = Path(config_dir)
    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    def _load_yaml(name: str) -> dict[str, Any]:
        path = config_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    data_cfg = _load_yaml("data.yaml")

    return SSIFConfig(
        data=DataConfig(**data_cfg["data"]),
        validation=ValidationConfig(**data_cfg.get("validation", {})),
        dlsm_compatibility=DLSMCompatibilityConfig(
            **data_cfg.get("dlsm_compatibility", {})
        ),
        project_root=config_dir.parent,
    )


# ---------------------------------------------------------------------------
# Singleton for dashboard / scripts
# ---------------------------------------------------------------------------

_config_cache: Optional[SSIFConfig] = None


def get_config(config_dir: Optional[Path] = None) -> SSIFConfig:
    """Return cached config or load fresh one."""
    global _config_cache
    if _config_cache is None:
        _config_cache = load_config(config_dir)
    return _config_cache
