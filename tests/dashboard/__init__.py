"""Behavioral test suite for the ``transformation_portal.dashboard`` package.

The dashboard package is ~9.6k LOC of pure-Python (FastAPI/pydantic/sqlite,
no ML) that historically had no dedicated ``tests/dashboard/`` directory and
no per-package coverage floor. These suites establish a deterministic,
offline behavioral baseline against the most self-contained seams so a
conservative cold-zone floor can be set per
``docs/testing/COLD_ZONE_COVERAGE_PROGRAM.md``.
"""
