#!/usr/bin/env python3
"""Retrofit pytest markers to test files per ADR-044.

This script adds appropriate pytest markers to test files based on their
directory location, following the conventions in ADR-044 Section 4.

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
import ast
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

# Directory -> required marker mapping (ADR-044)
DIRECTORY_MARKERS: dict[str, list[str]] = {
    "unit": ["unit"],
    "smoke": ["unit"],  # smoke tests map to unit per ADR-044
    "security": ["security"],
    "integration": ["integration"],
    "benchmarks": ["benchmark"],
    "stress": ["stress", "slow"],
    "golden": ["golden"],
    # Default for other directories
    "_default": ["unit"],
}

# Directories that should NOT be auto-tagged (fixtures, data, etc.)
SKIP_DIRECTORIES: frozenset[str] = frozenset(
    {
        "fixtures",
        "data",
        "baselines",
        "__pycache__",
    }
)

# Pattern to detect existing pytestmark
PYTESTMARK_PATTERN = re.compile(r"^pytestmark\s*=", re.MULTILINE)

# Pattern to detect existing @pytest.mark on module level
MODULE_MARKER_PATTERN = re.compile(r"^@pytest\.mark\.\w+", re.MULTILINE)

# Pattern to detect imports
IMPORT_PYTEST_PATTERN = re.compile(r"^import pytest\s*$|^from pytest import", re.MULTILINE)


def has_test_functions(content: str) -> bool:
    """Check if file contains test functions."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            return True
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name.startswith("test_"):
                    return True
    return False


def has_existing_module_markers(content: str) -> bool:
    """Check if file already has module-level pytest markers."""
    return bool(PYTESTMARK_PATTERN.search(content)) or bool(MODULE_MARKER_PATTERN.search(content))


def has_class_or_function_markers(content: str) -> bool:
    """Check if file has @pytest.mark on classes or functions."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Attribute):
                    if isinstance(decorator.value, ast.Attribute):
                        if isinstance(decorator.value.value, ast.Name):
                            if decorator.value.value.id == "pytest" and decorator.value.attr == "mark":
                                return True
                elif isinstance(decorator, ast.Call):
                    if isinstance(decorator.func, ast.Attribute):
                        if isinstance(decorator.func.value, ast.Attribute):
                            if isinstance(decorator.func.value.value, ast.Name):
                                if decorator.func.value.value.id == "pytest" and decorator.func.value.attr == "mark":
                                    return True
    return False


def get_directory_marker(file_path: Path) -> list[str]:
    """Determine markers based on directory."""
    parts = file_path.parts
    for part in reversed(parts):
        if part in DIRECTORY_MARKERS:
            return DIRECTORY_MARKERS[part]
    return DIRECTORY_MARKERS["_default"]


def add_pytest_import(content: str) -> str:
    """Add 'import pytest' after other imports."""
    lines = content.split("\n")

    # Find last import line
    last_import_idx = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            last_import_idx = i

    if last_import_idx >= 0:
        # Insert after last import
        lines.insert(last_import_idx + 1, "import pytest")
    else:
        # No imports found, add at beginning after docstring
        insert_idx = 0
        if lines and (lines[0].startswith('"""') or lines[0].startswith("'''")):
            quote = lines[0][:3]
            for i, line in enumerate(lines):
                if i > 0 and quote in line:
                    insert_idx = i + 1
                    break
        lines.insert(insert_idx, "import pytest")

    return "\n".join(lines)


def add_pytestmark(content: str, markers: list[str]) -> str:
    """Add pytestmark declaration to file content."""
    lines = content.split("\n")

    # Build the marker line
    if len(markers) == 1:
        marker_line = f"pytestmark = pytest.mark.{markers[0]}"
    else:
        marker_parts = ", ".join(f"pytest.mark.{m}" for m in markers)
        marker_line = f"pytestmark = [{marker_parts}]"

    # Find insertion point (after imports, before first non-import code)
    insert_idx = 0
    in_docstring = False
    docstring_quote = None
    past_imports = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Handle docstrings
        if not in_docstring:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                docstring_quote = stripped[:3]
                if stripped.count(docstring_quote) >= 2:
                    # Single-line docstring
                    insert_idx = i + 1
                    continue
                in_docstring = True
                continue
        else:
            if docstring_quote and docstring_quote in stripped:
                in_docstring = False
                insert_idx = i + 1
            continue

        # Handle imports and from imports
        if stripped.startswith("import ") or stripped.startswith("from "):
            insert_idx = i + 1
            past_imports = True
            continue

        # Handle __future__ imports specially (must come first)
        if "from __future__" in stripped:
            insert_idx = i + 1
            continue

        # Handle blank lines after imports
        if past_imports and not stripped:
            insert_idx = i + 1
            continue

        # Handle comments
        if stripped.startswith("#"):
            if past_imports:
                # Comments after imports - good insertion point
                insert_idx = i
                break
            continue

        # First real code - insert before it
        if stripped and not stripped.startswith("#"):
            if not past_imports:
                # No imports found, insert after docstring
                pass
            break

    # Insert marker line with appropriate spacing
    if insert_idx < len(lines) and lines[insert_idx].strip():
        # Add blank line before if inserting before code
        lines.insert(insert_idx, "")
        insert_idx += 1
    lines.insert(insert_idx, marker_line)
    if insert_idx + 1 < len(lines) and lines[insert_idx + 1].strip():
        # Add blank line after if there's code following
        lines.insert(insert_idx + 1, "")

    return "\n".join(lines)


def process_file(file_path: Path, dry_run: bool = True) -> tuple[bool, str]:
    """Process a single test file.

    Returns:
        (modified, reason) tuple
    """
    if not file_path.name.startswith("test_"):
        return False, "not a test file"

    # Skip fixture/data directories
    for part in file_path.parts:
        if part in SKIP_DIRECTORIES:
            return False, f"in skip directory: {part}"

    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        return False, f"read error: {e}"

    # Skip if no test functions
    if not has_test_functions(content):
        return False, "no test functions"

    # Skip if already has module-level markers
    if has_existing_module_markers(content):
        return False, "already has module-level markers"

    # Skip if has class/function markers (defer manual review)
    if has_class_or_function_markers(content):
        return False, "has class/function markers (manual review needed)"

    # Ensure pytest is imported - now we handle this
    if not IMPORT_PYTEST_PATTERN.search(content):
        content = add_pytest_import(content)

    # Get appropriate markers
    markers = get_directory_marker(file_path)

    # Add markers
    new_content = add_pytestmark(content, markers)

    if not dry_run:
        file_path.write_text(new_content, encoding="utf-8")

    return True, f"added {', '.join(f'@pytest.mark.{m}' for m in markers)}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Retrofit pytest markers to test files per ADR-044")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files",
    )
    parser.add_argument(
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

    if not args.dry_run and not args.apply:
        print("ERROR: Must specify --dry-run or --apply", file=sys.stderr)
        return 1

    dry_run = args.dry_run or not args.apply

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
