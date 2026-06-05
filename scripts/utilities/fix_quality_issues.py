#!/usr/bin/env python3
"""
Comprehensive quality fix script for Transformation Portal.
Addresses all linting issues automatically where possible.
"""

import shlex
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report results.

    Security Note: Uses shlex.split() instead of shell=True to prevent
    command injection vulnerabilities (SEC-001). This safely parses the
    command string into a list of arguments without invoking a shell.
    """
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    # Convert string to list for safe subprocess execution (no shell injection risk)
    cmd_list = shlex.split(cmd) if isinstance(cmd, str) else cmd
    result = subprocess.run(cmd_list, capture_output=True, text=True, check=False)
    if result.stdout:
        print(result.stdout)
    if result.stderr and result.returncode != 0:
        print("STDERR:", result.stderr)
    return result.returncode


def main():
    """Fix quality issues."""
    repo_root = Path(__file__).resolve().parents[2]

    # 1. Fix trailing whitespace
    print("\n🔧 Fixing trailing whitespace...")
    files_to_fix = []
    for pattern in ["*.py"]:
        files_to_fix.extend(repo_root.glob(f"**/{pattern}"))

    # Filter out excluded paths
    excluded = {".venv", "deprecated", "src/transformation_portal", ".backup_local", ".local_backup"}
    files_to_fix = [f for f in files_to_fix if not any(ex in str(f) for ex in excluded)]

    for file_path in files_to_fix:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Remove trailing whitespace
            lines = content.split("\n")
            fixed_lines = [line.rstrip() for line in lines]
            fixed_content = "\n".join(fixed_lines)

            if fixed_content != content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(fixed_content)
                print(f"✓ Fixed: {file_path.relative_to(repo_root)}")
        except Exception as e:
            print(f"✗ Error fixing {file_path}: {e}")

    # 2. Run autopep8 for line length
    print("\n🔧 Running autopep8 for line length...")
    run_command(
        "autopep8 --in-place --max-line-length=127 --select=E501 "
        "--exclude=deprecated,src/transformation_portal,.venv,.backup_local,.local_backup "
        "--recursive .",
        "Auto-fixing line length issues",
    )

    # 3. Run flake8 to verify
    print("\n✅ Verifying with flake8...")
    flake8_result = run_command(
        "python3 -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics "
        "--exclude=deprecated/,src/transformation_portal/,.venv/",
        "Critical errors check",
    )

    # 4. Check imports order
    print("\n🔧 Checking import order...")
    run_command(
        "python3 -m isort --check-only --profile=black --line-length=127 "
        "--skip deprecated --skip src/transformation_portal --skip .venv "
        "--skip .backup_local --skip .local_backup .",
        "Import order check",
    )

    print("\n" + "=" * 60)
    if flake8_result == 0:
        print("✅ All critical issues fixed!")
        return 0
    else:
        print("⚠️  Some issues remain - check output above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
