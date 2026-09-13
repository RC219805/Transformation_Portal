#!/usr/bin/env python3
"""Compatibility CLI for the canonical Unicode-control security checker."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validation.check_unicode_controls import check_file as _check_file  # noqa: E402


def check_file(filepath):
    """Preserve the legacy result tuple while failing closed on read errors."""
    issues = _check_file(Path(filepath))
    return (False, "; ".join(issues)) if issues else (True, None)


def main():
    """Check all provided files for security issues."""
    if len(sys.argv) < 2:
        print("✅ No files to check")
        return 0

    files_to_check = sys.argv[1:]
    issues_found = []

    for filepath in files_to_check:
        passed, error = check_file(filepath)
        if not passed:
            issues_found.append((filepath, error))

    if issues_found:
        print("❌ Security issues found:")
        for filepath, error in issues_found:
            print(f"  {filepath}: {error}")
        return 1

    print(f"✅ Checked {len(files_to_check)} files - no security issues found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
