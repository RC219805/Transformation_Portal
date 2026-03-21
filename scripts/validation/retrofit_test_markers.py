#!/usr/bin/env python3
"""Retrofit pytest markers to test files per ADR-044.

This is a thin CLI wrapper that delegates to the reusable logic in
transformation_portal.dev.test_markers.

Usage:
    # Dry-run (show what would be changed)
    python scripts/validation/retrofit_test_markers.py --dry-run

    # Apply changes
    python scripts/validation/retrofit_test_markers.py --apply

    # Apply to specific directory
    python scripts/validation/retrofit_test_markers.py --apply tests/attestation
"""

from __future__ import annotations

import sys

# Import all logic from the src package
from transformation_portal.dev.test_markers import main

if __name__ == "__main__":
    sys.exit(main())
