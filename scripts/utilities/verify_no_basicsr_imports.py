#!/usr/bin/env python3
"""Verify that basicsr is not importable (safety check for CVE-2024-27763).

Usage: python scripts/utilities/verify_no_basicsr_imports.py --check-pkg
Exits non-zero if basicsr can be imported from the active environment.
"""
import sys
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--check-pkg', action='store_true', help='Check if basicsr is importable and exit non-zero if present')
    args = p.parse_args()

    try:
        import basicsr  # type: ignore
        print('ERROR: basicsr is importable in the environment. This may expose CVE-2024-27763')
        sys.exit(2)
    except Exception:
        # Import failed, which is the expected state
        print('OK: basicsr is not importable')
        sys.exit(0)


if __name__ == '__main__':
    main()
