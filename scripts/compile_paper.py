"""
compile_paper.py — Automated LaTeX to PDF Compiler for SSIF Research Paper
Student Success Intelligence Framework (SSIF)

Fetches the standalone Tectonic LaTeX engine (if not installed) and compiles
papers/ssif_academic_retention_study.tex to camera-ready IEEEtran PDF.
"""
from __future__ import annotations

import hashlib
import io
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = REPO_ROOT / "papers"
TEX_FILE = PAPERS_DIR / "ssif_academic_retention_study.tex"
PDF_FILE = PAPERS_DIR / "ssif_academic_retention_study.pdf"
TOOLS_DIR = REPO_ROOT / "tools"
TECTONIC_EXE = TOOLS_DIR / "tectonic.exe"

TECTONIC_URL = "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64-pc-windows-msvc.zip"
# Cryptographic SHA-256 hash of tectonic-0.17.0-x86_64-pc-windows-msvc.zip (SEC-03)
TECTONIC_ZIP_SHA256 = "f61ce51f0b0ade1015b7de7ef368541c5424e9756ecbd0d7af97d6d48030845f"


def ensure_tectonic() -> Path:
    """Ensure tectonic executable is available, downloading standalone binary with SHA-256 verification."""
    # Check if tectonic is on system PATH
    system_tectonic = shutil.which("tectonic")
    if system_tectonic:
        print(f"[OK] Found system tectonic at: {system_tectonic}")
        return Path(system_tectonic)

    if TECTONIC_EXE.exists():
        print(f"[OK] Found local tectonic at: {TECTONIC_EXE}")
        return TECTONIC_EXE

    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[FETCH] Downloading standalone Tectonic binary from {TECTONIC_URL}...")
    headers = {"User-Agent": "SSIF-Research-Compiler/1.0"}
    req = urllib.request.Request(TECTONIC_URL, headers=headers)
    with urllib.request.urlopen(req) as resp:
        zip_bytes = resp.read()

    # Cryptographic integrity check (CWE-494)
    actual_hash = hashlib.sha256(zip_bytes).hexdigest()
    if actual_hash != TECTONIC_ZIP_SHA256:
        raise ValueError(
            f"SECURITY ERROR: Cryptographic hash mismatch on Tectonic archive!\n"
            f"Expected SHA-256: {TECTONIC_ZIP_SHA256}\n"
            f"Actual SHA-256:   {actual_hash}"
        )
    print(f"[OK] SHA-256 integrity verified: {actual_hash[:16]}...")

    print(f"[EXTRACT] Extracting {len(zip_bytes) / 1024 / 1024:.1f} MB archive...")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for member in zf.namelist():
            if member.endswith("tectonic.exe") or member == "tectonic.exe":
                with zf.open(member) as source, open(TECTONIC_EXE, "wb") as target:
                    shutil.copyfileobj(source, target)
                break

    if not TECTONIC_EXE.exists():
        raise RuntimeError("Failed to extract tectonic.exe from downloaded archive.")

    print(f"[OK] Tectonic installed to {TECTONIC_EXE}")
    return TECTONIC_EXE


def compile_paper() -> bool:
    """Compile LaTeX document to PDF using Tectonic."""
    if not TEX_FILE.exists():
        print(f"[FAIL] TeX file not found: {TEX_FILE}")
        return False

    compiler = ensure_tectonic()
    print(f"[COMPILE] Compiling {TEX_FILE.name} with Tectonic...")
    
    cmd = [
        str(compiler),
        str(TEX_FILE),
        "--outdir",
        str(PAPERS_DIR),
        "--print",
    ]
    
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    
    if result.returncode != 0:
        print("[FAIL] Compilation error encountered:")
        print(result.stderr or result.stdout)
        return False

    if PDF_FILE.exists():
        size_kb = PDF_FILE.stat().st_size / 1024
        print(f"[SUCCESS] Camera-ready PDF generated successfully: {PDF_FILE} ({size_kb:.1f} KB)")
        return True
    else:
        print("[FAIL] Output PDF was not created.")
        return False


if __name__ == "__main__":
    success = compile_paper()
    sys.exit(0 if success else 1)
