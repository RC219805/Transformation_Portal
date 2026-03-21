#!/usr/bin/env python3
"""Retrofit pytest markers to test files per ADR-044.

This is a thin CLI wrapper that delegates to the reusable logic in
transformation_portal.dev.test_markers.

Usage:
    # Dry-run (show what would be changed)
    python scripts/validation/retrofit_test_markers.py --dry-run

    # Apply changes
    python scripts/validation/retrofit_test_markers.py --apply

    # Apply to specific directory
    python scripts/validation/retrofit_test_markers.py --apply tests/attestation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Import all reusable logic from the src package
from transformation_portal.dev.test_markers import process_file

if TYPE_CHECKING:
    from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for retrofit_test_markers.

    Args:
        argv: Command line arguments. If None, uses sys.argv.

    Returns:
        Exit code (0 for success).
    """
    parser = argparse.ArgumentParser(description="Retrofit pytest markers to test files per ADR-044")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files",
    )
    mode_group.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes to files",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["tests"],
        help="Paths to process (default: tests/)",
    )

    args = parser.parse_args(argv)

    dry_run = args.dry_run

    # Collect test files
    test_files: list[Path] = []
    for path_str in args.paths:
        path = Path(path_str)
        if path.is_file():
            test_files.append(path)
        elif path.is_dir():
            test_files.extend(path.rglob("test_*.py"))

    # Process files
    modified_count = 0
    skipped_reasons: dict[str, int] = {}

    for file_path in sorted(test_files):
        modified, reason = process_file(file_path, dry_run=dry_run)
        if modified:
            modified_count += 1
            print(f"{'[DRY-RUN] ' if dry_run else ''}Modified: {file_path}")
            print(f"  -> {reason}")
        else:
            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1

    # Summary
    print()
    print("=" * 60)
    print(f"{'DRY-RUN ' if dry_run else ''}SUMMARY")
    print("=" * 60)
    print(f"Total files scanned: {len(test_files)}")
    print(f"Files {'would be ' if dry_run else ''}modified: {modified_count}")
    print()
    print("Skipped files by reason:")
    for reason, count in sorted(skipped_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
