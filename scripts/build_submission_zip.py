"""
Automated Submission Packager for FraudShield.

Creates FraudShield_submission.zip containing code, documentation, models,
notebooks, presentation, and reports while strictly excluding:
- .venv / virtualenvs
- node_modules
- data/raw
- .git
- __pycache__ and .pytest_cache
- .zip archives and backup folders
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_ZIP = ROOT_DIR / "FraudShield_submission.zip"

EXCLUDE_DIRS = {
    ".venv",
    "venv",
    "env",
    "node_modules",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".idea",
    ".vscode",
    "archive",
    "legacy_streamlit",
    "FraudShield_backup_v4",
    "FraudShield_backup_v5",
    "FraudShield_backup_v6",
    "FraudShield_backup_v7",
    "raw",  # data/raw
}

EXCLUDE_FILENAMES = {
    "train.parquet",
    "val.parquet",
    "test.parquet",
}

EXCLUDE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".zip",
    ".log",
    ".tmp",
}


def should_exclude(path: Path) -> bool:
    rel = path.relative_to(ROOT_DIR)
    parts = rel.parts

    # Check directory exclusions
    for part in parts:
        if part in EXCLUDE_DIRS:
            return True

    # Check specific filename exclusions
    if path.name in EXCLUDE_FILENAMES:
        return True

    # Check extension exclusions
    if path.suffix.lower() in EXCLUDE_EXTENSIONS:
        return True

    # Specifically exclude data/raw
    if len(parts) >= 2 and parts[0] == "data" and parts[1] == "raw":
        return True

    return False


def build_zip() -> None:
    print(f"Creating submission package: {OUTPUT_ZIP} ...")
    count = 0
    total_bytes = 0

    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for root, dirs, files in os.walk(ROOT_DIR):
            root_path = Path(root)

            # Filter dirs in-place to avoid descending into excluded directories
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]

            for f in files:
                file_path = root_path / f
                if should_exclude(file_path):
                    continue

                arcname = file_path.relative_to(ROOT_DIR)
                zf.write(file_path, arcname)
                count += 1
                total_bytes += file_path.stat().st_size

    zip_size_mb = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
    print(f"Submission archive complete:")
    print(f"  • Files packed : {count}")
    print(f"  • Uncompressed : {total_bytes / (1024 * 1024):.2f} MB")
    print(f"  • Archive size : {zip_size_mb:.2f} MB")
    print(f"  • Destination  : {OUTPUT_ZIP}")


if __name__ == "__main__":
    build_zip()
