#!/usr/bin/env python3
"""Verify that no Python files import basicsr package.

This script scans the repository for any basicsr imports in Python files
and exits with error code 1 if any are found.
"""

import sys
from pathlib import Path


def main():
    """Scan for basicsr imports in Python files."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    python_files = list(repo_root.glob("**/*.py"))

    # Exclude virtual environments and deprecated code
    python_files = [
        f for f in python_files
        if not any(part in f.parts for part in ['.venv', 'venv', 'venv_py311', 'deprecated'])
    ]

    violations = []

    for py_file in python_files:
        try:
            content = py_file.read_text(encoding='utf-8')
            for line_num, line in enumerate(content.splitlines(), start=1):
                # Check for basicsr imports
                if 'import basicsr' in line or 'from basicsr' in line:
                    violations.append((py_file, line_num, line.strip()))
        except Exception:
            # Skip files that can't be read
            continue

    if violations:
        print("❌ ERROR: Found forbidden 'basicsr' imports in the codebase:")
        for file_path, line_num, line in violations:
            rel_path = file_path.relative_to(repo_root)
            print(f"  {rel_path}:{line_num}: {line}")
        return 1

    print("✅ No forbidden 'basicsr' imports found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
