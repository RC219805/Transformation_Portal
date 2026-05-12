#!/usr/bin/env python3
"""Generate the cold-zone coverage baseline from Cobertura coverage.xml."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

try:
    from scripts.ci.cobertura_xml import CoberturaFile, load_cobertura_files
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from cobertura_xml import CoberturaFile, load_cobertura_files  # type: ignore[no-redef]


@dataclass(frozen=True)
class ColdZoneTarget:
    filename: str
    marker_lane: str


COLD_ZONE_TARGETS: tuple[ColdZoneTarget, ...] = (
    ColdZoneTarget("src/transformation_portal/streaming/stages.py", "unit"),
    ColdZoneTarget("src/transformation_portal/depth/tools.py", "unit"),
    ColdZoneTarget("src/transformation_portal/plugins/loader.py", "security"),
    ColdZoneTarget("src/transformation_portal/vlm/quality_validator.py", "unit"),
    ColdZoneTarget("src/transformation_portal/vlm/scene_analyzer.py", "unit"),
    ColdZoneTarget("src/transformation_portal/vlm/llava.py", "ml"),
    ColdZoneTarget("src/transformation_portal/stage_graph/policy.py", "unit"),
    ColdZoneTarget("src/transformation_portal/stage_graph/stages/depth.py", "unit"),
    ColdZoneTarget("src/transformation_portal/spatial_ai/reconstruction/", "unit"),
)


def _percent(covered: int, valid: int) -> float | None:
    if valid == 0:
        return None
    return round(100.0 * covered / valid, 2)


def _format_percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}%"


def _format_ranges(ranges: tuple[str, ...]) -> str:
    return ", ".join(ranges) if ranges else "-"


def _is_target_match(filename: str, target: ColdZoneTarget) -> bool:
    if target.filename.endswith("/"):
        return filename.startswith(target.filename)
    return filename == target.filename


def _combine_files(target: ColdZoneTarget, files: tuple[CoberturaFile, ...]) -> dict[str, Any]:
    matches = [coverage_file for coverage_file in files if _is_target_match(coverage_file.filename, target)]
    covered_lines = sum(coverage_file.covered_lines for coverage_file in matches)
    valid_lines = sum(coverage_file.valid_lines for coverage_file in matches)
    covered_branches = sum(coverage_file.branch_covered for coverage_file in matches)
    valid_branches = sum(coverage_file.branch_valid for coverage_file in matches)
    missed_branches = sum(coverage_file.missed_branch_count for coverage_file in matches)
    missed_ranges: list[str] = []
    for coverage_file in matches:
        if target.filename.endswith("/"):
            prefix = coverage_file.filename.removeprefix(target.filename)
            missed_ranges.extend(f"{prefix}:{line_range}" for line_range in coverage_file.missed_line_ranges)
        else:
            missed_ranges.extend(coverage_file.missed_line_ranges)

    return {
        "file": target.filename,
        "recommended_marker_lane": target.marker_lane,
        "line_coverage_percent": _percent(covered_lines, valid_lines),
        "lines_covered": covered_lines,
        "lines_valid": valid_lines,
        "branch_coverage_percent": _percent(covered_branches, valid_branches),
        "branches_covered": covered_branches,
        "branches_valid": valid_branches,
        "missed_line_ranges": missed_ranges,
        "missed_branch_count": missed_branches,
    }


def build_report_payload(coverage_xml: Path, *, baseline_date: str) -> dict[str, Any]:
    files = load_cobertura_files(coverage_xml)
    rows = [_combine_files(target, files) for target in COLD_ZONE_TARGETS]
    return {
        "baseline_date": baseline_date,
        "source": str(coverage_xml),
        "targets": rows,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# Cold-Zone Coverage Baseline ({payload['baseline_date']})",
        "",
        f"Generated from `{payload['source']}`.",
        "",
        "| File | Marker Lane | Lines | Line Coverage | Branches | Branch Coverage | Missed Lines | Missed Branches |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in payload["targets"]:
        line_ratio = f"{row['lines_covered']}/{row['lines_valid']}"
        branch_ratio = f"{row['branches_covered']}/{row['branches_valid']}"
        lines.append(
            "| `{file}` | `{lane}` | {line_ratio} | {line_pct} | {branch_ratio} | {branch_pct} | {missed_lines} | {missed_branches} |".format(
                file=row["file"],
                lane=row["recommended_marker_lane"],
                line_ratio=line_ratio,
                line_pct=_format_percent(row["line_coverage_percent"]),
                branch_ratio=branch_ratio,
                branch_pct=_format_percent(row["branch_coverage_percent"]),
                missed_lines=_format_ranges(tuple(row["missed_line_ranges"])),
                missed_branches=row["missed_branch_count"],
            )
        )
    lines.extend(
        [
            "",
            "This baseline is informational. PR 0 adds no coverage floors; ratchets land only after review.",
            "",
        ]
    )
    return "\n".join(lines)


def _infer_baseline_date(markdown_out: Path | None) -> str:
    if markdown_out is not None:
        match = re.search(r"(\d{4}-\d{2}-\d{2})", markdown_out.name)
        if match:
            return match.group(1)
    return date.today().isoformat()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coverage_xml", type=Path, help="Path to Cobertura coverage.xml produced by pytest-cov.")
    parser.add_argument("--markdown-out", type=Path, help="Optional Markdown report output path.")
    parser.add_argument("--json-out", type=Path, help="Optional JSON report output path.")
    parser.add_argument("--baseline-date", help="Baseline ISO date. Defaults to date inferred from markdown output.")
    args = parser.parse_args(argv)

    if not args.coverage_xml.is_file():
        print(
            f"cold_zone_report: coverage.xml not found at {args.coverage_xml}. " "Run make coverage-report first.",
            file=sys.stderr,
        )
        return 2

    baseline_date = args.baseline_date or _infer_baseline_date(args.markdown_out)
    payload = build_report_payload(args.coverage_xml, baseline_date=baseline_date)
    markdown = render_markdown(payload)

    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown, encoding="utf-8")
    else:
        print(markdown)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    sys.exit(main())
