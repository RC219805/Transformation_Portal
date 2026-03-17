#!/usr/bin/env python3
"""Auto-format changed Python files before commit.

This script is designed to be used as a pre-commit hook to ensure
all staged Python files are properly formatted before they enter
the repository.

Usage:
    python scripts/dev/auto_format_changed.py

    # Or via pre-commit:
    pre-commit run auto-format

This implements the "zero-diff formatting" principle where:
- Code is formatted at write-time, not as a post-hoc fix
- CI becomes verification-only (no fixing in CI)
- No human involvement required for formatting
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Repository formatting standards
BLACK_LINE_LENGTH = 127


def get_staged_python_files() -> list[Path]:
    """Get list of staged Python files."""
    result = subprocess.run(
        ["git", "diff", "--name-only", "--cached", "--diff-filter=ACMR"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [Path(f) for f in result.stdout.strip().splitlines() if f.endswith((".py", ".pyi")) and Path(f).exists()]


def get_changed_python_files() -> list[Path]:
    """Get list of changed (unstaged) Python files."""
    result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=ACMR"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [Path(f) for f in result.stdout.strip().splitlines() if f.endswith((".py", ".pyi")) and Path(f).exists()]


def format_files(files: list[Path]) -> bool:
    """Format the given files with Black and isort.

    Returns:
        True if formatting was successful, False otherwise.
    """
    if not files:
        return True

    file_strs = [str(f) for f in files]
    success = True

    # Apply Black formatting
    try:
        subprocess.run(
            ["black", f"--line-length={BLACK_LINE_LENGTH}", *file_strs],
            check=True,
        )
        print(f"✓ Black formatted {len(files)} file(s)")
    except subprocess.CalledProcessError as e:
        print(f"✗ Black formatting failed: {e}", file=sys.stderr)
        success = False
    except FileNotFoundError:
        print("⚠ Black not found - skipping Black formatting", file=sys.stderr)

    # Apply isort import ordering
    try:
        subprocess.run(
            ["isort", *file_strs],
            check=True,
        )
        print(f"✓ isort formatted {len(files)} file(s)")
    except subprocess.CalledProcessError as e:
        print(f"✗ isort failed: {e}", file=sys.stderr)
        success = False
    except FileNotFoundError:
        print("⚠ isort not found - skipping import sorting", file=sys.stderr)

    return success


def re_stage_files(files: list[Path]) -> None:
    """Re-stage files after formatting."""
    if not files:
        return

    file_strs = [str(f) for f in files]
    subprocess.run(["git", "add", *file_strs], check=True)
    print(f"✓ Re-staged {len(files)} file(s)")


def main() -> int:
    """Main entry point.

    Returns:
        0 on success, 1 on failure.
    """
    # Get staged files
    staged_files = get_staged_python_files()

    if not staged_files:
        print("No staged Python files to format")
        return 0

    print(f"Found {len(staged_files)} staged Python file(s)")

    # Format files
    if not format_files(staged_files):
        return 1

    # Re-stage formatted files
    re_stage_files(staged_files)

    return 0


if __name__ == "__main__":
    sys.exit(main())
