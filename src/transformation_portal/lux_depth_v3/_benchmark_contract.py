"""Internal helpers for Lux Depth V3 benchmark baseline enforcement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def write_benchmark_metrics(path: Path, payload: Dict[str, Any]) -> None:
    """Write benchmark metrics as deterministic JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_benchmark_metrics(path: Path) -> Dict[str, Any]:
    """Load a benchmark metrics JSON fixture."""
    return json.loads(path.read_text(encoding="utf-8"))


def assert_regression_within_tolerance(
    *,
    label: str,
    measured_value: float,
    baseline_value: float,
    tolerance_fraction: float,
    unit: str,
) -> None:
    """Assert a metric does not regress beyond an allowed tolerance."""
    if baseline_value <= 0:
        raise AssertionError(f"{label} baseline must be positive, got {baseline_value!r}")

    allowed_limit = baseline_value * (1.0 + tolerance_fraction)
    if measured_value > allowed_limit:
        tolerance_pct = tolerance_fraction * 100.0
        raise AssertionError(
            f"{label} regression: measured {measured_value:.3f}{unit} exceeds "
            f"baseline {baseline_value:.3f}{unit} +{tolerance_pct:.0f}% "
            f"(limit {allowed_limit:.3f}{unit})"
        )
