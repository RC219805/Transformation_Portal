"""Test marker utilities for ADR-044 test marker enforcement.

This module provides functions for adding pytest markers to test files
based on directory conventions defined in ADR-044. These utilities are
designed to be imported by both:
- The CLI script (scripts/validation/retrofit_test_markers.py)
- Test suites that validate marker behavior

Key functions:
- add_pytest_import(): Insert 'import pytest' after other imports
- add_pytestmark(): Add pytestmark declaration to a file
- process_file(): Full file-level transformation
- has_test_functions(): Check if file contains test functions
- has_existing_module_markers(): Check for existing module markers
- has_class_or_function_markers(): Check for class/function markers
- get_directory_marker(): Determine markers based on directory path
"""

from __future__ import annotations

import argparse
import ast
import re
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
    """Check if file contains test functions.

    Args:
        content: Python source code as a string.

    Returns:
        True if the file contains functions starting with 'test_'.
    """
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
    """Check if file already has module-level pytest markers.

    Note: MODULE_MARKER_PATTERN currently overmatches and will also match
    decorators on classes/functions that start at column 0. This is a known
    limitation documented for future cleanup.

    Args:
        content: Python source code as a string.

    Returns:
        True if the file has pytestmark= or @pytest.mark at module level.
    """
    return bool(PYTESTMARK_PATTERN.search(content)) or bool(MODULE_MARKER_PATTERN.search(content))


def has_class_or_function_markers(content: str) -> bool:
    """Check if file has @pytest.mark on classes or functions.

    Args:
        content: Python source code as a string.

    Returns:
        True if any class or function has pytest.mark decorators.
    """
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
    """Determine markers based on directory.

    Args:
        file_path: Path to the test file.

    Returns:
        List of marker names for the file based on its directory location.
    """
    parts = file_path.parts
    for part in reversed(parts):
        if part in DIRECTORY_MARKERS:
            return DIRECTORY_MARKERS[part]
    return DIRECTORY_MARKERS["_default"]


# Pattern to detect PEP 263 encoding declarations (# -*- coding: utf-8 -*- or # coding=utf-8)
_ENCODING_PATTERN = re.compile(r"coding[:=]\s*([-\w.]+)")


def _is_encoding_line(line: str) -> bool:
    r"""Check if a line is a PEP 263 encoding declaration.

    Per PEP 263, encoding declarations must match the regex:
    ``coding[:=]\s*([-\w.]+)``

    Common formats:
    - # -*- coding: utf-8 -*-
    - # coding=utf-8
    - # vim: set fileencoding=utf-8 :

    Args:
        line: A source line to check.

    Returns:
        True if the line is a valid encoding declaration.
    """
    return line.startswith("#") and bool(_ENCODING_PATTERN.search(line))


def _count_special_header_lines(lines: list[str]) -> int:
    """Count shebang and encoding declaration lines at the start of a file.

    Python requires that shebang (#!/...) be on line 1 (if present), and
    encoding declarations (# -*- coding: ... or # coding=...) be on line 1
    or 2. This function identifies how many such lines exist at the start.

    Args:
        lines: List of source lines.

    Returns:
        Number of special header lines (0, 1, or 2).
    """
    count = 0

    # Check for shebang on line 1
    if lines and lines[0].startswith("#!"):
        count = 1
        # Check for encoding declaration on line 2
        if len(lines) > 1 and _is_encoding_line(lines[1]):
            count = 2
    elif lines and _is_encoding_line(lines[0]):
        # No shebang, check for encoding on line 1
        count = 1

    return count


def add_pytest_import(content: str) -> str:
    """Add 'import pytest' after other imports.

    Uses AST parsing to reliably find the last import statement and determine
    the correct insertion point. Preserves shebang lines and encoding
    declarations at the start of the file.

    Args:
        content: Python source code as a string.

    Returns:
        Modified source with 'import pytest' added after last import,
        or after module docstring if no imports exist, or after
        shebang/encoding lines if neither imports nor docstring exist.
    """
    lines = content.split("\n")
    source = "\n".join(lines)

    # Count special header lines that must be preserved at the top
    special_header_count = _count_special_header_lines(lines)

    # AST-based detection for import/docstring positions.
    # insert_idx is updated through the try-else block:
    #   - Default: after special headers (shebang/encoding)
    #   - SyntaxError fallback: after last detected import line
    #   - AST success: after last import, or docstring, or headers
    last_import_line: int | None = None
    docstring_end_line: int | None = None
    insert_idx = special_header_count

    try:
        module = ast.parse(source)
    except SyntaxError:
        # Fall back to line-based scanning if the file is not valid Python.
        in_multiline_import = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if in_multiline_import:
                if ")" in stripped:
                    in_multiline_import = False
                    insert_idx = i + 1
                continue
            if stripped.startswith("import ") or stripped.startswith("from "):
                insert_idx = i + 1
                if "(" in stripped and ")" not in stripped:
                    in_multiline_import = True
    else:
        # Record module docstring end line, if present.
        if module.body:
            first_node = module.body[0]
            if isinstance(first_node, ast.Expr):
                expr_value = first_node.value
                if isinstance(expr_value, ast.Constant) and isinstance(expr_value.value, str):
                    docstring_end_line = getattr(first_node, "end_lineno", first_node.lineno)

        # Find the last top-level import or from-import.
        for node in module.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                end_lineno = getattr(node, "end_lineno", node.lineno)
                if last_import_line is None or end_lineno > last_import_line:
                    last_import_line = end_lineno

        if last_import_line is not None:
            # Insert after the last import (1-based to 0-based index).
            insert_idx = last_import_line
        elif docstring_end_line is not None:
            # No imports; insert after module docstring.
            insert_idx = docstring_end_line
        # else: insert_idx remains at special_header_count (after special headers)

    # Insert the import
    lines.insert(insert_idx, "import pytest")

    return "\n".join(lines)


def add_pytestmark(content: str, markers: list[str]) -> str:
    """Add pytestmark declaration to file content.

    Uses AST parsing to reliably find the last import statement and insert
    the pytestmark after all imports are complete.

    Args:
        content: Python source code as a string.
        markers: List of marker names to add.

    Returns:
        Modified source with pytestmark declaration added.

    Note: The double-blank-line spacing reflects current implementation shape,
    not a timeless contract. The spacing may change in future refactors.
    """
    lines = content.split("\n")

    # Build the marker line
    if len(markers) == 1:
        marker_line = f"pytestmark = pytest.mark.{markers[0]}"
    else:
        marker_parts = ", ".join(f"pytest.mark.{m}" for m in markers)
        marker_line = f"pytestmark = [{marker_parts}]"

    # Determine insertion point using AST to avoid splitting import blocks.
    source = "\n".join(lines)
    last_import_line: int | None = None
    docstring_end_line: int | None = None
    insert_idx = 0

    try:
        module = ast.parse(source)
    except SyntaxError:
        # Fall back to line-based scanning if the file is not valid Python.
        # Find last import line manually
        in_multiline_import = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if in_multiline_import:
                if ")" in stripped:
                    in_multiline_import = False
                    insert_idx = i + 1
                continue
            if stripped.startswith("import ") or stripped.startswith("from "):
                insert_idx = i + 1
                if "(" in stripped and ")" not in stripped:
                    in_multiline_import = True
    else:
        # Record module docstring end line, if present.
        if module.body:
            first_node = module.body[0]
            if isinstance(first_node, ast.Expr):
                expr_value = first_node.value
                if isinstance(expr_value, ast.Constant) and isinstance(expr_value.value, str):
                    docstring_end_line = getattr(first_node, "end_lineno", first_node.lineno)

        # Find the last top-level import or from-import.
        for node in module.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                end_lineno = getattr(node, "end_lineno", node.lineno)
                if last_import_line is None or end_lineno > last_import_line:
                    last_import_line = end_lineno

        if last_import_line is not None:
            # Insert after the last import (1-based to 0-based index).
            insert_idx = last_import_line
        elif docstring_end_line is not None:
            # No imports; insert after module docstring.
            insert_idx = docstring_end_line
        else:
            # No imports or docstring; insert at the top.
            insert_idx = 0

    # Skip over any blank lines immediately following the chosen insertion line.
    while 0 <= insert_idx < len(lines) and not lines[insert_idx].strip():
        insert_idx += 1

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

    Args:
        file_path: Path to the test file to process.
        dry_run: If True, don't write changes to disk.

    Returns:
        (modified, reason) tuple where modified is True if the file was
        (or would be) modified, and reason explains the action taken.
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
    """CLI entry point for retrofit_test_markers.

    This function can be called directly for testing or via the CLI wrapper.

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


__all__ = [
    # Constants
    "DIRECTORY_MARKERS",
    "SKIP_DIRECTORIES",
    "PYTESTMARK_PATTERN",
    "MODULE_MARKER_PATTERN",
    "IMPORT_PYTEST_PATTERN",
    # Detection functions
    "has_test_functions",
    "has_existing_module_markers",
    "has_class_or_function_markers",
    "get_directory_marker",
    # Transformation functions
    "add_pytest_import",
    "add_pytestmark",
    "process_file",
    # CLI
    "main",
]
