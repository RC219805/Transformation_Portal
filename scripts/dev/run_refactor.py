#!/usr/bin/env python3
"""Run the auto-refactoring engine to deduplicate semantic code.

This script scans the codebase for semantically duplicate code and
optionally refactors it by extracting canonical implementations
to a shared module.

Usage:
    # Preview changes (default)
    python scripts/dev/run_refactor.py

    # Execute refactoring
    python scripts/dev/run_refactor.py --execute

    # Dry run with verbose output
    python scripts/dev/run_refactor.py --dry-run --verbose

    # Specify custom paths
    python scripts/dev/run_refactor.py --root src --shared src/mypackage/shared

Options:
    --root: Root directory to scan (default: src)
    --shared: Shared module directory (default: src/transformation_portal/shared)
    --execute: Actually perform refactoring (default is preview only)
    --dry-run: Show what would be done without doing it
    --verbose: Show detailed output
    --batch-size: Process in batches of N duplicate groups
    --target-hash: Only refactor files with this specific hash
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Auto-refactor to eliminate semantic code duplication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("src"),
        help="Root directory to scan (default: src)",
    )
    parser.add_argument(
        "--shared",
        type=Path,
        default=Path("src/transformation_portal/shared"),
        help="Shared module directory",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute refactoring (default is preview only)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without doing it",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Process in batches (0 = all at once)",
    )
    parser.add_argument(
        "--target-hash",
        type=str,
        help="Only refactor files with this specific hash",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Additional patterns to exclude",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    # Validate paths
    if not args.root.exists():
        print(f"Error: Root directory not found: {args.root}", file=sys.stderr)
        return 1

    # Import after path setup
    from transformation_portal.dev.refactor_engine import (
        AutoRefactorEngine,
        IncrementalRefactor,
    )

    # Create engine
    exclude_patterns = [
        "**/test_*.py",
        "**/*_test.py",
        "**/conftest.py",
        "**/__pycache__/**",
        "**/shared/**",  # Don't refactor the shared module itself
    ]
    exclude_patterns.extend(args.exclude)

    engine = AutoRefactorEngine(
        root=args.root,
        shared_module=args.shared,
        exclude_patterns=exclude_patterns,
    )

    # Build plan
    print(f"Scanning {args.root} for semantic duplicates...")
    plan = engine.build_plan()

    if not plan.duplicates:
        print("✅ No semantic duplicates found")
        return 0

    # Show preview
    print()
    print(plan.summary())
    print()

    if not args.execute:
        print(engine.preview(plan))
        print()
        print("💡 Run with --execute to perform refactoring")
        print("💡 Run with --dry-run to see detailed changes")
        return 0

    # Execute refactoring
    dry_run = args.dry_run

    if args.target_hash:
        # Refactor specific hash only
        incremental = IncrementalRefactor(engine)
        result = incremental.refactor_by_hash(plan, args.target_hash, dry_run=dry_run)
        results = [result]

    elif args.batch_size > 0:
        # Refactor in batches
        incremental = IncrementalRefactor(engine)
        results = incremental.refactor_batch(plan, batch_size=args.batch_size, dry_run=dry_run)

    else:
        # Refactor all at once
        result = engine.execute(plan, dry_run=dry_run)
        results = [result]

    # Show results
    print()
    for i, result in enumerate(results):
        if len(results) > 1:
            print(f"Batch {i + 1}:")
        print(result.summary())
        print()

    # Check for errors
    all_success = all(r.success for r in results)
    total_errors = sum(len(r.errors) for r in results)

    if total_errors > 0:
        print(f"⚠️  {total_errors} error(s) occurred")
        return 1

    if all_success:
        if dry_run:
            print("✅ Dry run completed (no changes made)")
        else:
            print("✅ Refactoring completed successfully")
        return 0
    else:
        print("❌ Refactoring failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
