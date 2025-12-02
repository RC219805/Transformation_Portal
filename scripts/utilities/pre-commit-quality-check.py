#!/usr/bin/env python3
"""
Pre-Commit Quality Control System
Ensures code quality before commits to prevent CI failures
"""

import subprocess
import sys
from pathlib import Path


def run_flake8_critical():
    """Run flake8 for critical errors only"""
    print("🔍 Running flake8 (critical errors only)...")
    result = subprocess.run(
        ['flake8', '.', '--count', '--select=E9,F63,F7,F82', '--show-source', '--statistics',
         '--exclude=.venv,deprecated,src/transformation_portal,scripts'],
        capture_output=True,
        text=True,
        check=False
    )

    if result.returncode != 0:
        print("❌ Flake8 found critical errors:")
        print(result.stdout)
        print(result.stderr)
        return False

    print("✅ Flake8: No critical errors")
    return True


def check_undefined_names():
    """Check for common undefined name errors"""
    print("🔍 Checking for undefined names...")
    result = subprocess.run(
        ['flake8', '.', '--select=F821', '--exclude=.venv,deprecated,src/transformation_portal,scripts'],
        capture_output=True,
        text=True,
        check=False
    )

    if result.returncode != 0:
        print("⚠️  Found undefined names (F821):")
        print(result.stdout)
        return False

    print("✅ No undefined names")
    return True


def check_markdown_count():
    """Ensure root markdown files don't exceed limit"""
    print("🔍 Checking markdown file count...")
    root = Path('.')
    md_files = list(root.glob('*.md'))

    if len(md_files) > 10:
        print(f"❌ Too many markdown files in root ({len(md_files)} > 10):")
        for f in md_files:
            print(f"  - {f.name}")
        print("\nMove documentation files to docs/ directory")
        return False

    print(f"✅ Markdown count OK ({len(md_files)}/10)")
    return True


def check_trailing_whitespace():
    """Check for excessive trailing whitespace"""
    print("🔍 Checking for trailing whitespace...")

    result = subprocess.run(
        ['git', 'diff', '--cached', '--name-only', '--diff-filter=ACM'],
        capture_output=True,
        text=True,
        check=False
    )

    if result.returncode != 0:
        return True  # No staged files

    staged_files = [f for f in result.stdout.strip().split('\n') if f.endswith('.py')]
    issues_found = False

    for filepath in staged_files:
        if not Path(filepath).exists():
            continue

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        trailing_lines = []
        for i, line in enumerate(lines, 1):
            if line.rstrip() != line.rstrip('\n'):
                trailing_lines.append(i)

        if len(trailing_lines) > 5:
            print(f"⚠️  {filepath}: {len(trailing_lines)} lines with trailing whitespace")
            issues_found = True

    if issues_found:
        print("Consider running: autopep8 --in-place --select=W291,W293 <files>")
    else:
        print("✅ Trailing whitespace OK")

    return True  # Non-blocking


def check_import_order():
    """Check for common import issues"""
    print("🔍 Checking imports...")

    result = subprocess.run(
        ['flake8', '.', '--select=E402,F401', '--exclude=.venv,deprecated,src/transformation_portal,scripts'],
        capture_output=True,
        text=True,
        check=False
    )

    if result.returncode != 0:
        print("⚠️  Import issues found:")
        print(result.stdout)
        return False

    print("✅ Imports OK")
    return True


def run_quick_tests():
    """Run fast tests to catch obvious breakage"""
    print("🔍 Running quick tests...")

    result = subprocess.run(
        ['pytest', '-x', '-v', '--tb=short',
         'tests/test_format_utils.py::TestNormalizeExtension',
         'tests/test_error_handling.py::TestFileValidation'],
        capture_output=True,
        text=True,
        check=False
    )

    if result.returncode != 0:
        print("❌ Quick tests failed:")
        print(result.stdout[-1000:])  # Last 1000 chars
        return False

    print("✅ Quick tests passed")
    return True


def main():
    """Run all quality checks"""
    print("=" * 60)
    print("PRE-COMMIT QUALITY CHECK")
    print("=" * 60)

    checks = [
        ("Critical Errors (flake8)", run_flake8_critical, True),
        ("Undefined Names", check_undefined_names, True),
        ("Markdown Count", check_markdown_count, True),
        ("Trailing Whitespace", check_trailing_whitespace, False),  # Non-blocking
        ("Import Order", check_import_order, False),  # Non-blocking
        ("Quick Tests", run_quick_tests, False),  # Non-blocking for speed
    ]

    failures = []
    warnings = []

    for name, check_fn, blocking in checks:
        try:
            passed = check_fn()
            if not passed:
                if blocking:
                    failures.append(name)
                else:
                    warnings.append(name)
        except Exception as e:
            print(f"⚠️  Error running {name}: {e}")
            if blocking:
                failures.append(name)

    print("\n" + "=" * 60)

    if failures:
        print(f"❌ COMMIT BLOCKED - {len(failures)} critical issue(s):")
        for f in failures:
            print(f"  - {f}")
        print("\nFix these issues before committing.")
        return 1

    if warnings:
        print(f"⚠️  {len(warnings)} warning(s) found (non-blocking):")
        for w in warnings:
            print(f"  - {w}")
        print("\nConsider addressing these before pushing to main.")

    if not failures and not warnings:
        print("✅ ALL CHECKS PASSED - Ready to commit!")

    return 0


if __name__ == '__main__':
    sys.exit(main())
