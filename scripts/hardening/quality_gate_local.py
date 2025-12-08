#!/usr/bin/env python3
"""
Local-only quality gate driver (CI workflow is provided separately).

This script is intentionally lightweight: it checks for the presence of a golden
set and a baseline report, then exits with non-zero if thresholds are violated.

Wire this into your real validator/metrics once validation modules land, per
PRODUCTION_VALIDATION_GUIDE.
"""

from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    baseline = Path("golden_baseline/baseline_report.json")
    current = Path("current_test/current_report.json")

    if not baseline.exists() or not current.exists():
        print("Golden gate skipped (missing baseline_report.json or current_report.json).")
        return 0

    try:
        b = json.loads(baseline.read_text())
        c = json.loads(current.read_text())
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in report file: {e}")
        return 1

    b_score = float(b.get("composite_score", 0.0))
    c_score = float(c.get("composite_score", 0.0))

    # Default tolerance per guide example (0.05).
    if c_score < b_score - 0.05:
        print(f"FAIL: composite_score regressed. baseline={b_score:.4f} current={c_score:.4f}")
        return 2

    print(f"OK: composite_score. baseline={b_score:.4f} current={c_score:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
