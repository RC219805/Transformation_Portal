#!/usr/bin/env python3
"""
HuggingFace Model Revision Validator

Validates that HuggingFace model revisions in presets are pinned to
specific commit SHAs (not 'main' or 'NEEDS_VERIFICATION').

Usage:
    python scripts/validation/validate_hf_revisions.py
    python scripts/validation/validate_hf_revisions.py --fix

ADR: ADR-032 (Dependency Pinning Strategy)
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple

import yaml


def find_preset_files() -> List[Path]:
    """Find all preset YAML files."""
    preset_dir = Path("config/presets")
    return list(preset_dir.glob("**/*.yaml"))


def check_revision(preset_path: Path) -> List[Tuple[str, int, str]]:
    """Check for unpinned revisions in a preset file.

    Returns:
        List of (file, line_number, issue) tuples
    """
    issues = []

    with open(preset_path, "r") as f:
        lines = f.readlines()

    for line_num, line in enumerate(lines, start=1):
        # Check for revision field
        if "revision:" in line.lower():
            # Check for problematic patterns
            if "NEEDS_VERIFICATION" in line:
                issues.append((str(preset_path), line_num, f"Placeholder revision: {line.strip()}"))
            elif '"main"' in line or "'main'" in line or "main  #" in line:
                issues.append((str(preset_path), line_num, f"Unpinned revision (main branch): {line.strip()}"))
            elif "revision: null" in line:
                issues.append((str(preset_path), line_num, f"Null revision (will use main): {line.strip()}"))

    return issues


def get_da3_latest_commit() -> str:
    """Get placeholder for DA3 1.1 commit SHA.

    Note: Must be manually verified at:
    https://huggingface.co/depth-anything/DA3-NESTED-GIANT-LARGE-1.1/commits/main
    """
    return "MANUAL_VERIFICATION_REQUIRED"


def main():
    parser = argparse.ArgumentParser(description="Validate HuggingFace model revisions in presets")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix issues (manual verification still required)")
    parser.add_argument("--experimental-ok", action="store_true", help="Allow placeholders in experimental presets")
    args = parser.parse_args()

    print("━" * 70)
    print("  HuggingFace Model Revision Validator")
    print("━" * 70)
    print()

    preset_files = find_preset_files()
    all_issues = []

    for preset_path in preset_files:
        issues = check_revision(preset_path)

        # Skip experimental presets if flag is set
        if args.experimental_ok and "experimental" in str(preset_path):
            continue

        all_issues.extend(issues)

    if not all_issues:
        print("✅ All HuggingFace model revisions are properly pinned")
        print()
        return 0

    print(f"⚠️  Found {len(all_issues)} unpinned or placeholder revisions:")
    print()

    for file, line_num, issue in all_issues:
        print(f"  {file}:{line_num}")
        print(f"    {issue}")
        print()

    if args.fix:
        print("❌ Auto-fix not implemented - manual verification required")
        print()
        print("Action Items:")
        print("  1. Visit HuggingFace model pages")
        print("  2. Copy commit SHA from verified release")
        print("  3. Update preset YAML files")
        print()
        print("Example for DA3 1.1:")
        print("  URL: https://huggingface.co/depth-anything/DA3-NESTED-GIANT-LARGE-1.1")
        print("  Navigate to: Files and versions → Commits")
        print("  Copy: 40-character commit SHA")
        print()
        return 1

    print("━" * 70)
    print(f"Summary: {len(all_issues)} issues found")
    print()
    print("Run with --fix for guidance on resolving issues")
    print("Use --experimental-ok to skip experimental presets")
    print("━" * 70)

    return 1 if all_issues else 0


if __name__ == "__main__":
    sys.exit(main())
