#!/usr/bin/env python3
"""Enforce pytest marker requirements for test functions.

This is a thin CLI wrapper that delegates to the reusable logic in
transformation_portal.dev.check_test_markers.

Policy (ADR-044):
- All test functions must have at least one pytest marker.
- Tests in specific directories have required markers by convention.
- Pre-commit hook blocks unmarked tests from being added.

Usage:
    # Pre-commit mode (validate specific files)
    python scripts/validation/check_test_markers.py tests/test_foo.py tests/test_bar.py

    # Full audit mode (scan entire tests/ directory)
    python scripts/validation/check_test_markers.py --audit

    # Show detailed report
    python scripts/validation/check_test_markers.py --audit --verbose
"""

from __future__ import annotations

# Import all logic from the src package
from transformation_portal.dev.check_test_markers import main

if __name__ == "__main__":
    raise SystemExit(main())
