#!/usr/bin/env python3
"""Migration script for transitioning to depth_canonical module.

This script helps migrate code from deprecated depth modules to the new
depth_canonical module. It can scan, report, and optionally auto-migrate
deprecated import statements.

Usage:
    # Scan for deprecated imports
    python scripts/migrate_to_depth_canonical.py --scan src/

    # Generate migration report
    python scripts/migrate_to_depth_canonical.py --report src/ > migration_report.txt

    # Dry run (show what would change)
    python scripts/migrate_to_depth_canonical.py --dry-run src/

    # Auto-migrate (creates backups)
    python scripts/migrate_to_depth_canonical.py --migrate src/
"""

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class DeprecatedImport:
    """Represents a deprecated import statement."""

    file_path: Path
    line_number: int
    original_line: str
    suggested_replacement: str
    module_name: str


# Mapping of deprecated modules to canonical replacements
DEPRECATED_MODULES = {
    "transformation_portal.depth": "transformation_portal.depth_canonical",
    "transformation_portal.lux_depth_v3": "transformation_portal.depth_canonical",
    "transformation_portal.depth_intelligence": "transformation_portal.depth_canonical",
}

# Mapping of deprecated class names to canonical names
CLASS_MAPPINGS = {
    # depth/ module mappings
    "ArchitecturalDepthPipeline": "DepthPipeline",
    "DepthConfig": "UnifiedDepthConfig",
    # lux_depth_v3/ module mappings
    "generate_pbr_maps": "generate_pbr_maps",  # Same name, different module
    "DA3InferenceEngine": "ModelRegistry",  # Conceptual mapping
    # depth_intelligence/ module mappings
    "DepthEstimator": "ModelRegistry",  # Conceptual mapping
}


def find_python_files(root_path: Path) -> List[Path]:
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


def scan_file_for_deprecated_imports(file_path: Path) -> List[DeprecatedImport]:
    """Scan a Python file for deprecated import statements."""
    deprecated_imports = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            lines = content.split("\n")

        # Try AST parsing first (more robust)
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in DEPRECATED_MODULES:
                            line_num = node.lineno
                            original_line = lines[line_num - 1].strip()
                            suggested = original_line.replace(
                                alias.name, DEPRECATED_MODULES[alias.name]
                            )
                            deprecated_imports.append(
                                DeprecatedImport(
                                    file_path=file_path,
                                    line_number=line_num,
                                    original_line=original_line,
                                    suggested_replacement=suggested,
                                    module_name=alias.name,
                                )
                            )

                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module in DEPRECATED_MODULES:
                        line_num = node.lineno
                        original_line = lines[line_num - 1].strip()

                        # Build suggested replacement
                        canonical_module = DEPRECATED_MODULES[node.module]
                        imports = [alias.name for alias in node.names]

                        # Map class names to canonical names
                        mapped_imports = []
                        for imp in imports:
                            mapped_name = CLASS_MAPPINGS.get(imp, imp)
                            if imp != mapped_name:
                                mapped_imports.append(f"{mapped_name} as {imp}")
                            else:
                                mapped_imports.append(imp)

                        suggested = f"from {canonical_module} import {', '.join(mapped_imports)}"

                        deprecated_imports.append(
                            DeprecatedImport(
                                file_path=file_path,
                                line_number=line_num,
                                original_line=original_line,
                                suggested_replacement=suggested,
                                module_name=node.module,
                            )
                        )

        except SyntaxError:
            # Fall back to regex if AST parsing fails
            for line_num, line in enumerate(lines, start=1):
                for old_module, new_module in DEPRECATED_MODULES.items():
                    if old_module in line and ("import" in line or "from" in line):
                        suggested = line.replace(old_module, new_module)
                        deprecated_imports.append(
                            DeprecatedImport(
                                file_path=file_path,
                                line_number=line_num,
                                original_line=line.strip(),
                                suggested_replacement=suggested.strip(),
                                module_name=old_module,
                            )
                        )

    except Exception as e:
        print(f"Warning: Could not scan {file_path}: {e}", file=sys.stderr)

    return deprecated_imports


def generate_report(deprecated_imports: List[DeprecatedImport]) -> str:
    """Generate a human-readable migration report."""
    if not deprecated_imports:
        return "✅ No deprecated imports found. Migration not needed."

    # Group by file
    by_file: Dict[Path, List[DeprecatedImport]] = {}
    for imp in deprecated_imports:
        if imp.file_path not in by_file:
            by_file[imp.file_path] = []
        by_file[imp.file_path].append(imp)

    report_lines = ["Migration Report", "=" * 80, ""]

    # Summary
    report_lines.append(f"Deprecated Imports Found: {len(deprecated_imports)}")
    report_lines.append(f"Files Affected: {len(by_file)}")
    report_lines.append("")

    # Details by file
    for file_path, imports in sorted(by_file.items()):
        report_lines.append(f"File: {file_path}")
        for imp in sorted(imports, key=lambda x: x.line_number):
            report_lines.append(f"  Line {imp.line_number}:")
            report_lines.append(f"    Old: {imp.original_line}")
            report_lines.append(f"    New: {imp.suggested_replacement}")
        report_lines.append("")

    # Summary statistics
    report_lines.append("Summary:")
    report_lines.append(f"  - {len(by_file)} files need migration")
    report_lines.append(f"  - {len(deprecated_imports)} import statements to update")

    # Estimate effort (1 min per file)
    effort_minutes = len(by_file)
    report_lines.append(f"  - Estimated effort: {effort_minutes} minutes")
    report_lines.append("")
    report_lines.append("Run with --migrate to automatically apply these changes.")

    return "\n".join(report_lines)


def migrate_file(file_path: Path, deprecated_imports: List[DeprecatedImport], dry_run: bool = False) -> bool:
    """Migrate a single file, replacing deprecated imports.

    Args:
        file_path: Path to the file to migrate
        deprecated_imports: List of deprecated imports in this file
        dry_run: If True, don't actually modify files

    Returns:
        True if migration was successful, False otherwise
    """
    if not deprecated_imports:
        return True

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Create backup if not dry run
        if not dry_run:
            backup_path = file_path.with_suffix(file_path.suffix + ".bak")
            with open(backup_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
            print(f"  Created backup: {backup_path}")

        # Apply replacements (in reverse line order to preserve line numbers)
        for imp in sorted(deprecated_imports, key=lambda x: x.line_number, reverse=True):
            line_idx = imp.line_number - 1
            if line_idx < len(lines):
                indent = len(lines[line_idx]) - len(lines[line_idx].lstrip())
                lines[line_idx] = " " * indent + imp.suggested_replacement + "\n"

        # Write modified content
        if dry_run:
            print(f"  [DRY RUN] Would update {len(deprecated_imports)} imports")
        else:
            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
            print(f"  ✅ Migrated {len(deprecated_imports)} imports")

        return True

    except Exception as e:
        print(f"  ❌ Error migrating {file_path}: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Migrate code to depth_canonical module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to Python file or directory to scan",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Scan for deprecated imports and print summary",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate detailed migration report",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files",
    )
    parser.add_argument(
        "--migrate",
        action="store_true",
        help="Auto-migrate files (creates .bak backups)",
    )

    args = parser.parse_args()

    if not args.path.exists():
        print(f"Error: Path '{args.path}' does not exist", file=sys.stderr)
        return 1

    # Default to scan if no mode specified
    if not any([args.scan, args.report, args.dry_run, args.migrate]):
        args.scan = True

    # Find Python files
    python_files = find_python_files(args.path)
    print(f"Scanning {len(python_files)} Python files...", file=sys.stderr)

    # Scan for deprecated imports
    all_deprecated_imports = []
    for py_file in python_files:
        deprecated_imports = scan_file_for_deprecated_imports(py_file)
        all_deprecated_imports.extend(deprecated_imports)

    # Handle different modes
    if args.scan:
        if not all_deprecated_imports:
            print("✅ No deprecated imports found!")
            return 0
        else:
            print(f"\n⚠️  Found {len(all_deprecated_imports)} deprecated imports in {len(set(imp.file_path for imp in all_deprecated_imports))} files")
            print("\nRun with --report for detailed information")
            print("Run with --migrate to automatically fix")
            return 1

    elif args.report:
        print(generate_report(all_deprecated_imports))
        return 0

    elif args.dry_run or args.migrate:
        if not all_deprecated_imports:
            print("✅ No deprecated imports found!")
            return 0

        # Group by file
        by_file: Dict[Path, List[DeprecatedImport]] = {}
        for imp in all_deprecated_imports:
            if imp.file_path not in by_file:
                by_file[imp.file_path] = []
            by_file[imp.file_path].append(imp)

        # Migrate each file
        print(f"\n{'DRY RUN: ' if args.dry_run else ''}Migrating {len(by_file)} files...\n")
        success_count = 0
        for file_path, imports in sorted(by_file.items()):
            print(f"{file_path}:")
            if migrate_file(file_path, imports, dry_run=args.dry_run):
                success_count += 1

        print(f"\n{'Would migrate' if args.dry_run else 'Migrated'} {success_count}/{len(by_file)} files successfully")

        if not args.dry_run:
            print("\n⚠️  Backups created with .bak extension")
            print("💡 Run tests to verify migration: pytest tests/")

        return 0 if success_count == len(by_file) else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
