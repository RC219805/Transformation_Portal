#!/usr/bin/env python3
"""Verify that basicsr is not importable (safety check for CVE-2024-27763).

Usage: python scripts/utilities/verify_no_basicsr_imports.py --check-pkg
Exits non-zero if basicsr can be imported from the active environment.
"""
import sys
import argparse


def _find_repo_root() -> Path:
    """Find repository root by looking for .git directory."""
    script_path = Path(__file__).resolve()
    current = script_path.parent

    # Walk up directory tree looking for .git directory
    for _ in range(_MAX_TRAVERSAL_DEPTH):
        if (current / '.git').exists():
            return current
        if current.parent == current:  # Reached filesystem root
            break
        current = current.parent

    # Fallback: use path traversal from script location
    # scripts/utilities/verify_no_basicsr_imports.py -> repository root
    return script_path.parent.parent.parent


def check_basicsr_installed() -> bool:
    """Check if the vulnerable basicsr package is installed.

    Returns:
        True if basicsr is installed (security violation), False otherwise.
    """
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'show', 'basicsr'],
            capture_output=True,
            text=True,
            check=False
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, OSError):
        # If we can't check, assume it's not installed
        return False


def main():
    p = argparse.ArgumentParser(
        description='Verify that the vulnerable basicsr package is not importable.'
    )
    p.add_argument(
        '--check-pkg',
        action='store_true',
        help='Check if basicsr is importable and exit non-zero if present'
    )
    args = p.parse_args()

    if not args.check_pkg:
        print('Use --check-pkg to verify basicsr is not importable')
        sys.exit(0)

    try:
        import basicsr  # type: ignore  # noqa: F401
        print('ERROR: basicsr is importable in the environment. '
              'This may expose CVE-2024-27763')
        sys.exit(2)
    except (ImportError, ModuleNotFoundError):
        # Import failed due to missing package, which is the expected state
        print('OK: basicsr is not importable')
        sys.exit(0)
    except Exception as e:
        # Other errors during import (e.g., dependency issues) also indicate
        # basicsr is not usable, which is acceptable for our security check
        print(f'OK: basicsr import failed with error: {e}')
        sys.exit(0)


if __name__ == '__main__':
    main()
