#!/usr/bin/env python3
"""Automated quality fixer for common issues."""
import os
import re
from pathlib import Path


def fix_trailing_whitespace(file_path):
    """Remove trailing whitespace from file."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    fixed_lines = [line.rstrip() + "\n" if line.endswith("\n") else line.rstrip() for line in lines]

    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(fixed_lines)


def fix_imports_order(file_path):
    """Move imports to top of file (basic fix)."""
    # This is handled better by isort, skip for now


def main():
    """Fix common issues in Python files."""
    repo_root = Path(__file__).resolve().parents[2]

    # Files with trailing whitespace issues
    files_to_fix = [
        "audit_tiff_usage.py",
        "diagnose_tiff_quality.py",
        "fix_float_tiffs.py",
        "fix_tiff_16bit.py",
        "fix_tiff_loading.py",
        "fix_tiff_saving.py",
        "maximum_quality_pipeline.py",
        "process_750picacho_proper_16bit.py",
        "run_unified_pipeline.py",
        "save_tiff_correctly.py",
        "tiff_quality_optimizer.py",
        "ultimate_quality_pipeline.py",
        "verify_tiff_implementation.py",
        "convert_all_tiffs_to_16bit.py",
        "tests/test_unified_luxury_pipeline.py",
        "examples/unified_luxury_pipeline_examples.py",
        ".backup_local/conservative_enhance_greatroom_final.py",
        ".backup_local/conservative_enhance_greatroom_v4.py",
        ".backup_local/conservative_enhance_greatroom_v5.py",
        ".backup_local/conservative_enhance_greatroom_v6.py",
        ".backup_local/conservative_enhance_greatroom_v7.py",
        ".backup_local/conservative_enhance_greatroom_v8.py",
        ".backup_local/conservative_enhance_pool.py",
        ".backup_local/conservative_enhance_pool_v2.py",
        ".backup_local/conservative_enhance_pool_v3.py",
    ]

    for file_rel in files_to_fix:
        file_path = repo_root / file_rel
        if file_path.exists():
            print(f"Fixing: {file_rel}")
            fix_trailing_whitespace(file_path)
        else:
            print(f"Skip (not found): {file_rel}")


if __name__ == "__main__":
    main()
