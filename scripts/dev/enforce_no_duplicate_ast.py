#!/usr/bin/env python3
"""Enforce no duplicate semantic code in the repository.

This script scans the codebase for semantically duplicate code
and fails if any duplicates are found. It's designed to be used
as a pre-commit hook or CI check.

Usage:
    python scripts/dev/enforce_no_duplicate_ast.py [--allow-functions] [--allow-classes]

Exit codes:
    0: No duplicates found (or only allowed types)
    1: Duplicates found

Options:
    --allow-functions: Don't fail on duplicate functions (only files)
    --allow-classes: Don't fail on duplicate classes (only files)
    --report-only: Report duplicates but don't fail
    --exclude: Glob patterns to exclude (can be repeated)
    --min-nodes: Minimum AST node count to consider (default: 10)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from transformation_portal.dev.deduplicate import deduplicate_repo


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enforce no duplicate semantic code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="src",
        help="Root directory to scan (default: src)",
    )
    parser.add_argument(
        "--allow-functions",
        action="store_true",
        help="Don't fail on duplicate functions",
    )
    parser.add_argument(
        "--allow-classes",
        action="store_true",
        help="Don't fail on duplicate classes",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Report duplicates but don't fail",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Glob patterns to exclude (can be repeated)",
    )
    parser.add_argument(
        "--min-nodes",
        type=int,
        default=10,
        help="Minimum AST node count to consider (default: 10)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output report as JSON",
    )

    args = parser.parse_args()
    root = Path(args.root)

    if not root.exists():
        print(f"Error: Directory not found: {root}", file=sys.stderr)
        return 1

    # Default exclusions
    exclude_patterns = args.exclude or []
    exclude_patterns.extend(
        [
            "**/test_*.py",  # Test files often have intentional duplication
            "**/*_test.py",
            "**/conftest.py",
            "**/__pycache__/**",
        ]
    )

    print(f"Scanning {root} for semantic duplicates...")

    report = deduplicate_repo(
        root,
        exclude_patterns=exclude_patterns,
        include_functions=not args.allow_functions,
        include_classes=not args.allow_classes,
    )

    # Filter by minimum node count
    filtered_file_dups = []
    for group in report.file_duplicates:
        # Check if any file in group meets minimum
        from transformation_portal.dev.ast_index import ASTEquivalenceIndex

        index = ASTEquivalenceIndex()
        for path in group.files:
            entry = index.add_file(path)
            if entry and entry.node_count >= args.min_nodes:
                filtered_file_dups.append(group)
                break

    report.file_duplicates = filtered_file_dups

    if args.json:
        print(report.to_json())
    else:
        print(report.summary())

    # Determine if we should fail
    has_duplicates = False

    if report.file_duplicates:
        print(f"\n❌ Found {len(report.file_duplicates)} duplicate file group(s)")
        has_duplicates = True

    if report.function_duplicates and not args.allow_functions:
        print(f"\n❌ Found {len(report.function_duplicates)} duplicate function group(s)")
        has_duplicates = True

    if report.class_duplicates and not args.allow_classes:
        print(f"\n❌ Found {len(report.class_duplicates)} duplicate class group(s)")
        has_duplicates = True

    if has_duplicates:
        if args.report_only:
            print("\n⚠️  Duplicates found (report-only mode, not failing)")
            return 0
        else:
            print("\n💡 Consider refactoring to eliminate duplicates")
            return 1
    else:
        print("\n✅ No semantic duplicates found")
        return 0


if __name__ == "__main__":
    sys.exit(main())
