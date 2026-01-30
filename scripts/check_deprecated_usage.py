#!/usr/bin/env python3
"""Check for deprecated API usage in source code (for CI).

This script scans Python files for usage of deprecated depth modules and
fails with non-zero exit code if any are found. Used in CI to prevent
new code from using deprecated APIs.

Usage:
    python scripts/check_deprecated_usage.py src/
    python scripts/check_deprecated_usage.py --strict src/  # Exit 1 if found
"""

import argparse
import sys
from pathlib import Path

# Reuse logic from migration script
try:
    from migrate_to_depth_canonical import (
        find_python_files,
        scan_file_for_deprecated_imports,
    )
except ImportError:
    # If not importable, define minimal versions inline
    import ast

    def find_python_files(root_path: Path):
        """Find all Python files in the given directory tree."""
        if root_path.is_file() and root_path.suffix == ".py":
            return [root_path]

        python_files = []
        for pattern in ["**/*.py"]:
            python_files.extend(root_path.glob(pattern))

        # Exclude venv, __pycache__, etc.
        excluded_patterns = ["venv", "__pycache__", ".git", "build", "dist", "*.egg-info"]
        filtered_files = []
        for f in python_files:
            if not any(pattern in str(f) for pattern in excluded_patterns):
                filtered_files.append(f)

        return filtered_files

    def scan_file_for_deprecated_imports(file_path: Path):
        """Scan a Python file for deprecated import statements."""
        deprecated_modules = [
            "transformation_portal.depth",
            "transformation_portal.lux_depth_v3",
            "transformation_portal.depth_intelligence",
        ]

        deprecated_imports = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            for module in deprecated_modules:
                if module in content and "import" in content:
                    # Simple check - found deprecated import
                    deprecated_imports.append((file_path, module))

        except Exception:
            pass

        return deprecated_imports


def main():
    parser = argparse.ArgumentParser(description="Check for deprecated API usage (CI)")
    parser.add_argument("path", type=Path, help="Path to scan")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if deprecated usage found",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Exclude paths matching pattern (can be repeated)",
    )

    args = parser.parse_args()

    if not args.path.exists():
        print(f"Error: Path '{args.path}' does not exist", file=sys.stderr)
        return 1

    # Add default exclusions
    exclusions = args.exclude + [
        "src/transformation_portal/depth/__init__.py",  # Allow in deprecated modules themselves
        "src/transformation_portal/lux_depth_v3/__init__.py",
        "src/transformation_portal/depth_intelligence/__init__.py",
        "tests/test_deprecation_warnings.py",  # Allow in deprecation tests
        "scripts/migrate_to_depth_canonical.py",  # Allow in migration script
        "scripts/check_deprecated_usage.py",  # Allow in this script
    ]

    # Find Python files
    python_files = find_python_files(args.path)

    # Filter out excluded paths
    filtered_files = []
    for py_file in python_files:
        should_exclude = False
        for exclusion in exclusions:
            if exclusion in str(py_file):
                should_exclude = True
                break
        if not should_exclude:
            filtered_files.append(py_file)

    print(f"Checking {len(filtered_files)} Python files for deprecated API usage...")

    # Scan for deprecated imports
    all_deprecated_imports = []
    for py_file in filtered_files:
        deprecated_imports = scan_file_for_deprecated_imports(py_file)
        all_deprecated_imports.extend(deprecated_imports)

    if not all_deprecated_imports:
        print("✅ No deprecated API usage found in new code!")
        return 0
    else:
        print(f"\n⚠️  Found {len(all_deprecated_imports)} uses of deprecated APIs:\n")
        for imp in all_deprecated_imports:
            if isinstance(imp, tuple):
                print(f"  {imp[0]}: {imp[1]}")
            else:
                print(f"  {imp.file_path}:{imp.line_number} - {imp.module_name}")

        print("\n💡 Migration guide: docs/migration/depth_v2_migration.md")
        print("💡 Run: python scripts/migrate_to_depth_canonical.py --migrate src/")

        if args.strict:
            return 1
        else:
            return 0


if __name__ == "__main__":
    sys.exit(main())
