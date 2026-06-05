#!/usr/bin/env python3
"""
Automated linting fixes for CI/CD compliance.
Fixes trailing whitespace, unnecessary f-strings, and other common issues.
"""

import re
import sys
from pathlib import Path


def fix_trailing_whitespace(content: str) -> str:
    """Remove trailing whitespace from lines."""
    lines = content.splitlines(keepends=True)
    fixed_lines = [line.rstrip() + ("\n" if line.endswith("\n") else "") for line in lines]
    return "".join(fixed_lines)


def fix_unnecessary_fstrings(content: str) -> str:
    """Convert f-strings without interpolation to regular strings."""
    # Pattern: f"text without {variables}"
    # Replace with: "text without {variables}"
    pattern = r'f(["\'])((?:(?!\1)[^{])*)\1'

    def replacement(match):
        quote = match.group(1)
        text = match.group(2)
        # Only replace if there are no braces
        if "{" not in text and "}" not in text:
            return f"{quote}{text}{quote}"
        return match.group(0)

    return re.sub(pattern, replacement, content)


def process_file(filepath: Path) -> bool:
    """Process a single Python file. Returns True if changes were made."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            original_content = f.read()

        content = original_content

        # Apply fixes
        content = fix_trailing_whitespace(content)
        content = fix_unnecessary_fstrings(content)

        # Only write if changes were made
        if content != original_content:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Fixed: {filepath}")
            return True

        return False

    except Exception as e:
        print(f"Error processing {filepath}: {e}", file=sys.stderr)
        return False


def main():
    """Main function to process all Python files."""
    repo_root = Path(__file__).resolve().parents[2]

    # Directories to exclude
    exclude_dirs = {"deprecated", "src/transformation_portal", ".venv", "__pycache__", ".git", ".github"}

    # Find all Python files
    python_files = []
    for py_file in repo_root.rglob("*.py"):
        # Skip excluded directories
        if any(excluded in py_file.parts for excluded in exclude_dirs):
            continue
        python_files.append(py_file)

    print(f"Processing {len(python_files)} Python files...")

    fixed_count = 0
    for py_file in python_files:
        if process_file(py_file):
            fixed_count += 1

    print(f"\nFixed {fixed_count} files")

    if fixed_count > 0:
        print("\nRun 'git dif' to review changes before committing.")


if __name__ == "__main__":
    main()
