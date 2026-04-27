#!/usr/bin/env python3
"""Run offline-safe APEX visual evalset validation/reporting."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from transformation_portal.evals.apex_visual import (
    build_apex_eval_report,
    parse_candidate_masks,
    parse_candidate_outputs,
)
from transformation_portal.evals.apex_evidence_bundle import (
    build_apex_evidence_bundle,
    derive_materials_v3_evidence_from_manifest,
    parse_candidate_evidence,
)


# Restrict candidate / asset_id to characters that are safe to embed in a
# filename without risk of path traversal or unintended subdirectory creation.
# `parse_candidate_evidence` does not validate these components, so the CLI
# layer enforces the contract before constructing any filesystem path.
_SAFE_PATH_COMPONENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _ensure_safe_path_component(role: str, value: str) -> str:
    """Reject candidate / asset_id strings that would be unsafe as a filename.

    Raises:
        ValueError when the value is empty, contains path separators, contains
        ``..`` traversal segments, or contains any character outside the safe
        set ``[A-Za-z0-9._-]``.
    """
    if not value or value in {".", ".."} or not _SAFE_PATH_COMPONENT_RE.fullmatch(value):
        raise ValueError(
            f"Unsafe {role} {value!r}; must match {_SAFE_PATH_COMPONENT_RE.pattern} "
            "(letters, digits, '.', '_', '-')."
        )
    return value


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
        "--candidate-mask",
        action="append",
        default=[],
        metavar="CANDIDATE:ASSET_ID=PATH",
        help="Optional candidate Materials V3 mask NPZ for mask-aware APEX metrics. May be repeated.",
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
        "--candidate-evidence-from-manifest",
        action="append",
        default=[],
        metavar="CANDIDATE:ASSET_ID=PATH",
        help=(
            "Derive candidate telemetry JSON directly from a per-image manifest. "
            "PATH points at the manifest written by EnhanceOrchestrator; the "
            "tool renders MaterialsV3Metadata into the evidence schema and "
            "writes <output-dir>/derived_evidence/<candidate>__<asset_id>.evidence.json. "
            "CANDIDATE and ASSET_ID must match [A-Za-z0-9._-]+ (no path "
            "separators or '..' segments). Explicit --candidate-evidence wins "
            "for the same candidate+asset_id. May be repeated."
        ),
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
        candidate_masks = parse_candidate_masks(args.candidate_mask)
        candidate_evidence = parse_candidate_evidence(args.candidate_evidence)
        manifest_evidence_sources = parse_candidate_evidence(args.candidate_evidence_from_manifest)
        report = build_apex_eval_report(
            Path(args.evalset),
            output_dir=Path(args.output_dir),
            candidate_outputs=candidate_outputs,
            candidate_masks=candidate_masks,
            asset_root=args.asset_root,
        )
        if manifest_evidence_sources:
            resolved_output_dir = Path(str(report["report_path"])).parent
            derived_dir = resolved_output_dir / "derived_evidence"
            derived_dir.mkdir(parents=True, exist_ok=True)
            derived_dir_resolved = derived_dir.resolve()
            for candidate, asset_paths in manifest_evidence_sources.items():
                _ensure_safe_path_component("candidate", candidate)
                candidate_slot = candidate_evidence.setdefault(candidate, {})
                for asset_id, manifest_path in asset_paths.items():
                    _ensure_safe_path_component("asset_id", asset_id)
                    if asset_id in candidate_slot:
                        # Explicit --candidate-evidence wins; skip derivation.
                        continue
                    derived = derive_materials_v3_evidence_from_manifest(manifest_path)
                    derived_path = derived_dir / f"{candidate}__{asset_id}.evidence.json"
                    # Defense-in-depth: even after the regex check, confirm the
                    # resolved write target is inside the derived-evidence dir.
                    if derived_dir_resolved not in derived_path.resolve().parents:
                        raise ValueError(
                            f"Refusing to write derived evidence outside {derived_dir_resolved}: "
                            f"{derived_path}"
                        )
                    derived_path.write_text(json.dumps(derived, sort_keys=True), encoding="utf-8")
                    candidate_slot[asset_id] = derived_path
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
