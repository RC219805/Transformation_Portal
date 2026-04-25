#!/usr/bin/env python3
"""Emit a governed Depth Pro / DA3 backend benchmark report.

The default mode is offline-safe: it validates the evalset and writes a
comparison report with assets marked ``not_executed``. The Python API accepts
a runner callable for tests and for future live-execution wiring.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from transformation_portal.evals.apex_visual import build_depth_backend_benchmark_report


def _bool_flag(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean-like value, got {value!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Emit a governed depth backend comparison report.")
    parser.add_argument("--evalset", required=True, help="Evalset directory or evalset.json path.")
    parser.add_argument("--backends", required=True, help="Comma-separated backend ids, e.g. da3-metric,depth_pro.")
    parser.add_argument("--quality-tier", default="apex", help="Quality tier recorded in the report.")
    parser.add_argument("--output-dir", required=True, help="Directory for depth_backend_comparison_report.json.")
    parser.add_argument(
        "--emit-comparison-report",
        choices=("on", "off"),
        default="on",
        help="Compatibility flag; reports are always persisted for deterministic audit evidence.",
    )
    parser.add_argument("--non-commercial-ok", type=_bool_flag, default=False)
    parser.add_argument("--accept-apple-depth-pro-research-license", type=_bool_flag, default=False)
    args = parser.parse_args()

    backends = [item.strip() for item in args.backends.split(",") if item.strip()]
    report = build_depth_backend_benchmark_report(
        Path(args.evalset),
        backends=backends,
        quality_tier=args.quality_tier,
        output_dir=Path(args.output_dir),
        non_commercial_ok=args.non_commercial_ok,
        accept_depth_pro_license=args.accept_apple_depth_pro_research_license,
    )
    if args.emit_comparison_report == "on":
        print(Path(args.output_dir) / "depth_backend_comparison_report.json")
    blocked = [item["backend"] for item in report["backends"] if item["status"] == "license_blocked"]
    if blocked:
        print("License-blocked backend(s): " + ", ".join(blocked))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
