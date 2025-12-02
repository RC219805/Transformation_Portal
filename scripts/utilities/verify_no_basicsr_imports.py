#!/usr/bin/env python3
"""Verify that basicsr is not importable (safety check for CVE-2024-27763).

Usage: python scripts/utilities/verify_no_basicsr_imports.py --check-pkg
Exits non-zero if basicsr can be imported from the active environment.
"""
import sys
import argparse


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
    except ImportError:
        # Import failed, which is the expected state
        print('OK: basicsr is not importable')
        sys.exit(0)


if __name__ == '__main__':
    main()
