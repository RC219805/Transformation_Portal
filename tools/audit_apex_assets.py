#!/usr/bin/env python3
"""Audit APEX evalset asset readiness and canonical eligibility."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transformation_portal.evals.apex_visual import build_apex_eval_report
from transformation_portal.ingest.canonical_json import dump_json


def _bool_flag(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean-like value, got {value!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit APEX evalset assets.")
    parser.add_argument("--evalset", required=True, help="Evalset directory or evalset.json path.")
    parser.add_argument("--asset-root", default=None, help="Optional external asset root for evalset asset paths.")
    parser.add_argument(
        "--output-dir",
        default="output/apex_asset_audit",
        help="Directory for apex_asset_audit_report.json.",
    )
    parser.add_argument(
        "--require-canonical",
        type=_bool_flag,
        default=False,
        help="Fail when no canonical assets are eligible or canonical assets are missing/invalid.",
    )
    args = parser.parse_args()

    try:
        report = build_apex_eval_report(
            Path(args.evalset),
            output_dir=Path(args.output_dir),
            asset_root=args.asset_root,
        )
    except (OSError, ValueError) as exc:
        print(f"APEX asset audit error: {exc}", file=sys.stderr)
        return 3

    audit_path = Path(args.output_dir)
    if not audit_path.is_absolute():
        audit_path = Path.cwd() / audit_path
    audit_path.mkdir(parents=True, exist_ok=True)
    out_file = audit_path / "apex_asset_audit_report.json"
    with out_file.open("w", encoding="utf-8") as handle:
        dump_json(report, handle, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(out_file)

    checksum_mismatches = [
        item["asset_id"] for item in report["assets"] if item["asset_status"]["status"] == "checksum_mismatch"
    ]
    if checksum_mismatches:
        print("Checksum mismatch asset(s): " + ", ".join(checksum_mismatches), file=sys.stderr)
        return 1

    if args.require_canonical:
        canonical_count = int(report["evalset"].get("canonical_scoring_eligible_count") or 0)
        canonical_failures = [
            item["asset_id"]
            for item in report["assets"]
            if item.get("asset_role") == "canonical_apex_reference" and not item.get("canonical_scoring_eligible")
        ]
        if canonical_count == 0 or canonical_failures:
            if canonical_count == 0:
                print("No canonical scoring eligible assets found.", file=sys.stderr)
            if canonical_failures:
                print("Canonical asset(s) missing or invalid: " + ", ".join(canonical_failures), file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
