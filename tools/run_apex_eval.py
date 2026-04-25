#!/usr/bin/env python3
"""Run offline-safe APEX visual evalset validation/reporting."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transformation_portal.evals.apex_visual import build_apex_eval_report, parse_candidate_outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Emit an APEX visual quality eval report.")
    parser.add_argument("--evalset", required=True, help="Evalset directory or evalset.json path.")
    parser.add_argument("--output-dir", required=True, help="Directory for apex_eval_report.json.")
    parser.add_argument(
        "--candidate-output",
        action="append",
        default=[],
        metavar="CANDIDATE:ASSET_ID=PATH",
        help="Optional candidate output image for visible-delta metrics. May be repeated.",
    )
    parser.add_argument(
        "--emit-report",
        choices=("on", "off"),
        default="on",
        help="Compatibility flag; off validates inputs but does not suppress process exit status.",
    )
    args = parser.parse_args()

    try:
        candidate_outputs = parse_candidate_outputs(args.candidate_output)
        report = build_apex_eval_report(
            Path(args.evalset),
            output_dir=Path(args.output_dir),
            candidate_outputs=candidate_outputs,
        )
    except (OSError, ValueError) as exc:
        print(f"APEX eval error: {exc}", file=sys.stderr)
        return 2
    if args.emit_report == "on":
        print(report["report_path"])
    missing = [item["asset_id"] for item in report["assets"] if item["asset_status"]["status"] not in {"ready"}]
    if missing:
        print("APEX eval report emitted with non-ready assets: " + ", ".join(missing))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
