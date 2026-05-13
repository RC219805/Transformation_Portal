#!/usr/bin/env python3
"""Report coverage evidence for cold-zone files touched by a PR.

This script supports the Cold-Zone Coverage Program touched-file rule. It does
not enforce per-file percentage floors; package floors remain the automated
ratchet. Instead, it fails closed when a touched cold-zone source file is not
present in ``coverage.xml``, because reviewers cannot verify whether coverage
decreased without measurement data.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from scripts.ci.cobertura_xml import CoberturaFile, load_cobertura_files
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from cobertura_xml import CoberturaFile, load_cobertura_files  # type: ignore[no-redef]


COLD_ZONE_PREFIXES: tuple[str, ...] = (
    "src/transformation_portal/plugins/",
    "src/transformation_portal/stage_graph/",
    "src/transformation_portal/vlm/",
    "src/transformation_portal/depth/",
    "src/transformation_portal/streaming/",
    "src/transformation_portal/spatial_ai/reconstruction/",
)


@dataclass(frozen=True)
class TouchedFileCoverage:
    """Coverage facts for one touched cold-zone source file."""

    filename: str
    covered_lines: int
    valid_lines: int
    covered_branches: int
    valid_branches: int
    missed_line_ranges: tuple[str, ...]
    missed_branch_count: int

    @classmethod
    def from_cobertura_file(cls, coverage_file: CoberturaFile) -> "TouchedFileCoverage":
        return cls(
            filename=coverage_file.filename,
            covered_lines=coverage_file.covered_lines,
            valid_lines=coverage_file.valid_lines,
            covered_branches=coverage_file.branch_covered,
            valid_branches=coverage_file.branch_valid,
            missed_line_ranges=coverage_file.missed_line_ranges,
            missed_branch_count=coverage_file.missed_branch_count,
        )

    @property
    def line_percentage(self) -> float | None:
        if self.valid_lines == 0:
            return None
        return 100.0 * self.covered_lines / self.valid_lines

    @property
    def branch_percentage(self) -> float | None:
        if self.valid_branches == 0:
            return None
        return 100.0 * self.covered_branches / self.valid_branches


@dataclass(frozen=True)
class TouchedReport:
    """Report payload for touched cold-zone coverage evidence."""

    compare_ref: str
    touched_files: tuple[str, ...]
    measured_files: tuple[TouchedFileCoverage, ...]
    missing_files: tuple[str, ...]


def _normalize_repo_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    if normalized.startswith("/"):
        normalized = normalized[1:]
    return normalized


def _is_cold_zone_source(path: str) -> bool:
    normalized = _normalize_repo_path(path)
    return normalized.endswith(".py") and any(normalized.startswith(prefix) for prefix in COLD_ZONE_PREFIXES)


def _coverage_index(files: Iterable[CoberturaFile]) -> dict[str, CoberturaFile]:
    indexed: dict[str, CoberturaFile] = {}
    for coverage_file in files:
        indexed[coverage_file.filename] = coverage_file
        if coverage_file.filename.startswith("src/"):
            indexed[coverage_file.filename.removeprefix("src/")] = coverage_file
    return indexed


def collect_changed_files(compare_ref: str, *, repo_root: Path) -> tuple[str, ...]:
    """Return files changed relative to ``compare_ref``.

    Prefer a three-dot diff so local runs compare against the merge base. Shallow
    CI checkouts can lack that merge base even after fetching ``origin/main``;
    in that case, fall back to a direct tree diff against the fetched base ref.
    """
    result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=ACMRT", f"{compare_ref}...HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    message = result.stderr.strip() or result.stdout.strip() or f"git diff exited {result.returncode}"
    if result.returncode != 0 and "no merge base" in message.lower():
        result = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=ACMRT", compare_ref, "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        message = result.stderr.strip() or result.stdout.strip() or f"git diff exited {result.returncode}"

    if result.returncode != 0:
        raise RuntimeError(message)
    return tuple(_normalize_repo_path(line) for line in result.stdout.splitlines() if line.strip())


def build_report(coverage_xml: Path, changed_files: Iterable[str], *, compare_ref: str) -> TouchedReport:
    measured_by_path = _coverage_index(load_cobertura_files(coverage_xml))
    touched = tuple(sorted({_normalize_repo_path(path) for path in changed_files if _is_cold_zone_source(path)}))

    measured: list[TouchedFileCoverage] = []
    missing: list[str] = []
    for filename in touched:
        coverage_file = measured_by_path.get(filename)
        if coverage_file is None:
            missing.append(filename)
            continue
        measured.append(TouchedFileCoverage.from_cobertura_file(coverage_file))

    return TouchedReport(
        compare_ref=compare_ref,
        touched_files=touched,
        measured_files=tuple(measured),
        missing_files=tuple(missing),
    )


def _format_percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}%"


def _format_ranges(ranges: tuple[str, ...]) -> str:
    return ", ".join(ranges) if ranges else "-"


def render_report(report: TouchedReport) -> str:
    """Render a terminal-friendly touched-file coverage table."""
    lines = [
        "Cold-zone touched-file coverage report",
        f"Compare ref: {report.compare_ref}",
        f"Touched cold-zone source files: {len(report.touched_files)}",
    ]

    if not report.touched_files:
        lines.append("No cold-zone source files changed relative to the compare ref.")
        return "\n".join(lines)

    header = (
        f"{'File':<72}  {'Lines':>12}  {'Line %':>8}  "
        f"{'Branches':>12}  {'Branch %':>8}  {'Missed Lines':<24}  {'Missed Branches':>15}"
    )
    lines.extend(["", header, "-" * len(header)])
    for row in report.measured_files:
        line_ratio = f"{row.covered_lines}/{row.valid_lines}"
        branch_ratio = f"{row.covered_branches}/{row.valid_branches}"
        lines.append(
            f"{row.filename:<72}  {line_ratio:>12}  {_format_percent(row.line_percentage):>8}  "
            f"{branch_ratio:>12}  {_format_percent(row.branch_percentage):>8}  "
            f"{_format_ranges(row.missed_line_ranges):<24}  {row.missed_branch_count:>15}"
        )

    if report.missing_files:
        lines.extend(["", "Missing coverage entries:"])
        for filename in report.missing_files:
            lines.append(f"  - {filename}")

    lines.extend(
        [
            "",
            "Reviewer note: this report supplies evidence for the touched-file rule; "
            "new untested lines or branches still require PR-body justification.",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "coverage_xml",
        nargs="?",
        default="coverage.xml",
        type=Path,
        help="Path to Cobertura coverage.xml produced by pytest-cov.",
    )
    parser.add_argument(
        "--compare-ref",
        default="origin/main",
        help="Git ref used as the base for touched-file detection.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root used for git diff execution.",
    )
    args = parser.parse_args(argv)

    if not args.coverage_xml.is_file():
        print(
            f"check_cold_zone_touched_files: coverage.xml not found at {args.coverage_xml}. "
            "This script must run after pytest-cov produces the report.",
            file=sys.stderr,
        )
        return 2

    try:
        changed_files = collect_changed_files(args.compare_ref, repo_root=args.repo_root)
    except RuntimeError as exc:
        print(
            f"check_cold_zone_touched_files: unable to diff against {args.compare_ref}: {exc}",
            file=sys.stderr,
        )
        return 2

    report = build_report(args.coverage_xml, changed_files, compare_ref=args.compare_ref)
    print(render_report(report))

    if report.missing_files:
        print(
            "check_cold_zone_touched_files: touched cold-zone files are missing from coverage.xml.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
