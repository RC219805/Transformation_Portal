#!/usr/bin/env python3
"""Run offline-safe APEX visual evalset validation/reporting."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transformation_portal.evals.apex_visual import build_apex_eval_report, parse_candidate_outputs
from transformation_portal.evals.apex_evidence_bundle import build_apex_evidence_bundle, parse_candidate_evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Emit an APEX visual quality eval report.")
    parser.add_argument("--evalset", required=True, help="Evalset directory or evalset.json path.")
    parser.add_argument("--output-dir", required=True, help="Directory for apex_eval_report.json.")
    parser.add_argument("--asset-root", default=None, help="Optional external asset root for evalset asset paths.")
    parser.add_argument(
        "--candidate-output",
        action="append",
        default=[],
        metavar="CANDIDATE:ASSET_ID=PATH",
        help="Optional candidate output image for APEX metric evaluation. May be repeated.",
    )
    parser.add_argument(
        "--emit-report",
        choices=("on", "off"),
        default="on",
        help="Compatibility flag; off validates inputs but does not suppress process exit status.",
    )
    parser.add_argument(
        "--candidate-evidence",
        action="append",
        default=[],
        metavar="CANDIDATE:ASSET_ID=PATH",
        help="Optional candidate telemetry JSON for evidence bundles. May be repeated.",
    )
    parser.add_argument(
        "--run-scope-asset-id",
        action="append",
        default=[],
        help="Optional asset id limiting evidence promotion scope. May be repeated.",
    )
    parser.add_argument(
        "--synthetic-data",
        choices=("on", "off"),
        default="off",
        help="Mark evidence bundle inputs as synthetic plumbing evidence.",
    )
    parser.add_argument(
        "--emit-evidence-bundle",
        choices=("on", "off"),
        default="off",
        help="Emit evidence_bundle.json alongside the APEX eval report.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        candidate_outputs = parse_candidate_outputs(args.candidate_output)
        candidate_evidence = parse_candidate_evidence(args.candidate_evidence)
        report = build_apex_eval_report(
            Path(args.evalset),
            output_dir=Path(args.output_dir),
            candidate_outputs=candidate_outputs,
            asset_root=args.asset_root,
        )
        if args.emit_evidence_bundle == "on" or candidate_evidence:
            resolved_output_dir = Path(str(report["report_path"])).parent
            report_repo_root = report.get("evalset", {}).get("repo_root")
            build_apex_evidence_bundle(
                report,
                output_dir=resolved_output_dir,
                candidate_evidence=candidate_evidence,
                run_scope_asset_ids=args.run_scope_asset_id,
                synthetic_data=args.synthetic_data == "on",
                repo_root=Path(str(report_repo_root)) if report_repo_root else None,
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
