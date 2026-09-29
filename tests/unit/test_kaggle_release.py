"""
test_kaggle_release.py — Unit Tests for Kaggle Release Package & Metadata
Student Success Intelligence Framework (SSIF)
"""
import json
from pathlib import Path
import pytest

from scripts.publish_to_kaggle import (
    prepare_kaggle_dataset_bundle,
    prepare_kaggle_kernel_bundle,
    DATASET_SLUG,
    KERNEL_SLUG,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_kaggle_dataset_bundle_metadata():
    """Verify dataset metadata structure and licensing."""
    bundle_dir = prepare_kaggle_dataset_bundle()
    meta_file = bundle_dir / "dataset-metadata.json"
    assert meta_file.exists(), "dataset-metadata.json missing!"

    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    assert meta["id"] == DATASET_SLUG
    assert "SSIF" in meta["title"]
    assert len(meta["licenses"]) > 0
    assert any("CC-BY-SA" in lic["name"] for lic in meta["licenses"])
    assert "Razan Ihab Abdellatif" in meta["description"]
    assert "Amey Thakur" in meta["description"]

    # Verify essential dataset files are bundled
    assert (bundle_dir / "academic_survival_longitudinal.csv").exists()
    assert (bundle_dir / "Placement_Data_Full_Class.csv").exists()
    assert (bundle_dir / "NOTICE").exists()


def test_kaggle_kernel_bundle_metadata():
    """Verify kernel metadata and notebook existence."""
    kernel_dir = prepare_kaggle_kernel_bundle()
    meta_file = kernel_dir / "kernel-metadata.json"
    assert meta_file.exists(), "kernel-metadata.json missing!"

    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    assert meta["id"] == KERNEL_SLUG
    assert meta["language"] == "python"
    assert meta["kernel_type"] == "notebook"
    assert DATASET_SLUG in meta["dataset_sources"]

    nb_file = kernel_dir / "kaggle_ssif_student_success_study.ipynb"
    assert nb_file.exists(), "Kernel notebook file missing!"
