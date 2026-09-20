#!/usr/bin/env python3
"""Measure declared depth references and named candidates without running models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from transformation_portal.lux_depth_v5.evaluation import EvaluationError, write_report  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="Frozen tp.depth.quality.manifest.v1 JSON")
    parser.add_argument("--output", required=True, type=Path, help="New canonical report path; never overwritten")
    args = parser.parse_args(argv)
    try:
        report = write_report(args.manifest, args.output)
    except (EvaluationError, OSError) as exc:
        print(f"Depth evaluation incomplete: {exc}", file=sys.stderr)
        return 1
    print(f"{args.output}: {len(report['scenes'])} scenes; production acceptance not established")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
