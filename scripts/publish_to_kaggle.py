"""
publish_to_kaggle.py — Automated 1-Click Kaggle Release Pipeline
Student Success Intelligence Framework (SSIF)

Automates:
  1. Packaging verified Bronze & Gold dataset artifacts with schema dictionaries.
  2. Generating dataset-metadata.json and kernel-metadata.json.
  3. Validating Kaggle API authentication credentials (~/.kaggle/kaggle.json).
  4. Executing or dry-running `kaggle datasets version/create` and `kaggle kernels push`.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
KAGGLE_DIR = REPO_ROOT / "data" / "kaggle_dataset_bundle"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
KERNEL_DIR = NOTEBOOKS_DIR / "kaggle_kernel"

DATASET_SLUG = "harshkumarg007/ssif-student-success-intelligence-framework"
KERNEL_SLUG = "harshkumarg007/ssif-longitudinal-academic-persistence-study"


def prepare_kaggle_dataset_bundle() -> Path:
    """Prepares directory containing datasets and Kaggle dataset-metadata.json."""
    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Copy core datasets
    files_to_bundle = [
        REPO_ROOT / "academic_survival_longitudinal.csv",
        REPO_ROOT / "Placement_Data_Full_Class.csv",
        REPO_ROOT / "NOTICE",
        REPO_ROOT / "LICENSE",
        REPO_ROOT / "data" / "DATASET_METRICS_CATALOG.json",
    ]

    for src in files_to_bundle:
        if src.exists():
            shutil.copy2(src, KAGGLE_DIR / src.name)

    # 2. Write dataset-metadata.json
    metadata = {
        "title": "Student Success Intelligence Framework (SSIF) Datasets",
        "id": DATASET_SLUG,
        "licenses": [{"name": "CC-BY-SA-4.0"}],
        "keywords": [
            "education",
            "higher-education",
            "machine-learning",
            "retention",
            "survival-analysis",
            "explainable-ai",
            "causal-inference",
        ],
        "description": (
            "A Multi-Dataset Computational Research Laboratory for Longitudinal Academic Persistence, "
            "Employability Phenotypes, Survival Hazards, and Algorithmic Counterfactual Recourse.\n\n"
            "SSIF Primary Datasets:\n"
            "- Academic Retention Panel: 79,239 records across 20,000 students (Curator: Razan Ihab Abdellatif)\n"
            "- Campus Placement Cohort: 215 MBA candidates (Curator: Amey Thakur)\n\n"
            "Full project documentation & code: https://github.com/HarshkumarG007/SSIF"
        ),
    }

    meta_path = KAGGLE_DIR / "dataset-metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[OK] Prepared Kaggle dataset bundle at: {KAGGLE_DIR}")
    return KAGGLE_DIR


def prepare_kaggle_kernel_bundle() -> Path:
    """Prepares directory containing the publication notebook and kernel-metadata.json."""
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)

    # Copy notebook
    src_nb = NOTEBOOKS_DIR / "kaggle_ssif_student_success_study.ipynb"
    dest_nb = KERNEL_DIR / "kaggle_ssif_student_success_study.ipynb"
    if src_nb.exists():
        shutil.copy2(src_nb, dest_nb)

    # Write kernel-metadata.json
    kernel_meta = {
        "id": KERNEL_SLUG,
        "title": "SSIF: Longitudinal Academic Persistence & Employability Study",
        "code_file": "kaggle_ssif_student_success_study.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": False,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [DATASET_SLUG],
        "keywords": ["education", "data-science", "xgboost", "survival-analysis", "causal-inference"],
    }

    meta_path = KERNEL_DIR / "kernel-metadata.json"
    meta_path.write_text(json.dumps(kernel_meta, indent=2), encoding="utf-8")
    print(f"[OK] Prepared Kaggle kernel bundle at: {KERNEL_DIR}")
    return KERNEL_DIR


def check_kaggle_auth() -> bool:
    """Verifies Kaggle API credentials."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    has_env = "KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ
    if kaggle_json.exists() or has_env:
        return True
    return False


def run_kaggle_release(dry_run: bool = True):
    print("=" * 60)
    print("SSIF 1-CLICK KAGGLE RELEASE AUTOMATION")
    print("=" * 60)

    dataset_bundle = prepare_kaggle_dataset_bundle()
    kernel_bundle = prepare_kaggle_kernel_bundle()

    has_auth = check_kaggle_auth()
    print(f"[AUTH] Kaggle CLI credentials detected: {has_auth}")

    if not has_auth:
        print("\n[NOTE] No ~/.kaggle/kaggle.json or KAGGLE_KEY env var detected.")
        print("To push directly to Kaggle, place your kaggle.json in ~/.kaggle/ or set env variables.")
        print("Kaggle tokens can be generated at: https://www.kaggle.com/settings -> 'Create New Token'")

    if dry_run or not has_auth:
        print("\n[DRY RUN] Generated all publication packages successfully.")
        print("To publish live, execute:")
        print(f"  kaggle datasets create -p {dataset_bundle}")
        print(f"  kaggle kernels push -p {kernel_bundle}")
    else:
        print("\n[DISPATCH] Publishing dataset to Kaggle...")
        subprocess.run(["kaggle", "datasets", "version", "-p", str(dataset_bundle), "-m", "Automated SSIF v2.0 Release"], check=False)
        print("[DISPATCH] Publishing kernel to Kaggle...")
        subprocess.run(["kaggle", "kernels", "push", "-p", str(kernel_bundle)], check=False)

    print("\n[SUCCESS] Kaggle Release Package Ready!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSIF 1-Click Kaggle Release CLI")
    parser.add_argument("--live", action="store_true", help="Execute live Kaggle API push instead of dry-run")
    args = parser.parse_args()

    run_kaggle_release(dry_run=not args.live)
