#!/usr/bin/env python3
"""Detect raw Path.write_text() calls that bypass formatting.

This script scans the codebase for direct write_text() calls to Python
files that don't use the canonical write_formatted() utility. Such calls
can introduce unformatted code into the repository.

Usage:
    python scripts/dev/assert_no_raw_writes.py

    # Check specific directory:
    python scripts/dev/assert_no_raw_writes.py src/

Allowed patterns:
- write_formatted() calls (canonical)
- write_text() to non-Python files (safe)
- write_text() in test fixtures (allowed)
- write_text() in the formatting module itself (bootstrap)

Exit codes:
    0: No violations found
    1: Violations found (forbidden raw writes)
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Iterator

# Files that are allowed to use raw write_text()
ALLOWED_FILES = {
    # The formatting module itself needs raw writes
    "src/transformation_portal/dev/formatting.py",
    # Test files may use raw writes for fixtures
}

# Patterns that indicate allowed write_text() usage
ALLOWED_PATTERNS = [
    # Writing to temp files
    "tmp",
    "temp",
    # Writing non-Python files
    ".json",
    ".yaml",
    ".yml",
    ".txt",
    ".md",
    ".csv",
]


class RawWriteVisitor(ast.NodeVisitor):
    """AST visitor to detect raw write_text() calls."""

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.violations: list[tuple[int, str]] = []

    def visit_Call(self, node: ast.Call) -> None:
        """Check for write_text() method calls."""
        # Check if this is a method call
        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr

            if method_name == "write_text":
                # Check if this looks like it's writing Python code
                if self._is_suspicious_write(node):
                    self.violations.append((node.lineno, f"Raw write_text() call at line {node.lineno}"))

        # Continue visiting child nodes
        self.generic_visit(node)

    def _is_suspicious_write(self, node: ast.Call) -> bool:
        """Determine if a write_text() call might be writing Python code."""
        # Get the source of what's being written to
        try:
            # Check if target ends with .py
            if isinstance(node.func, ast.Attribute):
                value = node.func.value
                # Look for patterns like path.write_text() where path might be .py
                source = ast.unparse(value)
                if any(ext in source.lower() for ext in [".py", "python"]):
                    return True

                # Check if any argument mentions .py
                for arg in node.args:
                    arg_source = ast.unparse(arg)
                    if ".py" in arg_source:
                        return True

        except Exception:
            pass

        return False


def check_file(filepath: Path) -> list[tuple[int, str]]:
    """Check a single file for raw write_text() violations.

    Args:
        filepath: Path to the Python file to check.

    Returns:
        List of (line_number, message) tuples for violations found.
    """
    # Skip allowed files
    rel_path = str(filepath)
    if any(allowed in rel_path for allowed in ALLOWED_FILES):
        return []

    # Skip test files (they may need raw writes for fixtures)
    if "test_" in filepath.name or filepath.name.startswith("test"):
        return []

    try:
        content = filepath.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(filepath))

        visitor = RawWriteVisitor(filepath)
        visitor.visit(tree)

        return visitor.violations

    except SyntaxError as e:
        print(f"Syntax error in {filepath}: {e}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"Error processing {filepath}: {e}", file=sys.stderr)
        return []


def scan_directory(directory: Path) -> Iterator[tuple[Path, list[tuple[int, str]]]]:
    """Scan a directory for raw write_text() violations.

    Args:
        directory: Directory to scan.

    Yields:
        Tuples of (filepath, violations) for files with violations.
    """
    for filepath in directory.rglob("*.py"):
        violations = check_file(filepath)
        if violations:
            yield filepath, violations


def main() -> int:
    """Main entry point.

    Returns:
        0 if no violations found, 1 otherwise.
    """
    # Determine directories to scan
    if len(sys.argv) > 1:
        directories = [Path(arg) for arg in sys.argv[1:]]
    else:
        directories = [Path("src")]

    total_violations = 0

    for directory in directories:
        if not directory.exists():
            print(f"Directory not found: {directory}", file=sys.stderr)
            continue

        print(f"Scanning {directory}...")

        for filepath, violations in scan_directory(directory):
            for line_no, message in violations:
                print(f"  {filepath}:{line_no}: {message}")
                total_violations += 1

    if total_violations > 0:
        print(f"\n✗ Found {total_violations} potential raw write_text() violation(s)")
        print("  Use write_formatted() from transformation_portal.dev.formatting instead")
        return 1
    else:
        print("✓ No raw write_text() violations found")
        return 0


if __name__ == "__main__":
    sys.exit(main())
