#!/usr/bin/env python3
"""Dry-run branch-coverage reporting for governed package prefixes.

PR 0 intentionally does not enforce branch floors yet. This companion to
``check_per_package_coverage.py`` reads the same Cobertura ``coverage.xml`` and
can enforce branch floors once they are configured, but CI wires it in
``--dry-run`` mode until the cold-zone baseline is reviewed.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

try:
    from scripts.ci.cobertura_xml import _matches_prefix, load_cobertura_files
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from cobertura_xml import _matches_prefix, load_cobertura_files  # type: ignore[no-redef]


@dataclass(frozen=True)
class BranchFloor:
    """A branch-coverage gate over a path prefix."""

    prefix: str
    floor: float
    exclude_prefixes: tuple[str, ...] = field(default_factory=tuple)


# PR 0 only wires branch reporting. Actual floors land after baseline review.
BRANCH_FLOORS: tuple[BranchFloor, ...] = ()


@dataclass(frozen=True)
class BranchResult:
    prefix: str
    floor: float
    covered: int
    valid: int

    @property
    def percentage(self) -> float:
        if self.valid == 0:
            return 0.0
        return 100.0 * self.covered / self.valid

    @property
    def passed(self) -> bool:
        if self.valid == 0:
            return False
        return self.percentage >= self.floor


def aggregate(coverage_xml: Path, floors: Iterable[BranchFloor]) -> list[BranchResult]:
    files = load_cobertura_files(coverage_xml)
    results: list[BranchResult] = []
    for floor_spec in floors:
        covered = 0
        valid = 0
        normalized_prefix = floor_spec.prefix.replace("\\", "/")
        normalized_excludes = tuple(p.replace("\\", "/") for p in floor_spec.exclude_prefixes)
        for coverage_file in files:
            if not _matches_prefix(coverage_file.filename, normalized_prefix):
                continue
            if any(_matches_prefix(coverage_file.filename, excl) for excl in normalized_excludes):
                continue
            covered += coverage_file.branch_covered
            valid += coverage_file.branch_valid
        results.append(BranchResult(prefix=floor_spec.prefix, floor=floor_spec.floor, covered=covered, valid=valid))
    return results


def render_table(results: list[BranchResult], *, dry_run: bool) -> str:
    header = f"{'Package':<60}  {'Branches':>14}  {'Coverage':>10}  {'Floor':>8}  Status"
    lines = [header, "-" * len(header)]
    for result in results:
        ratio = f"{result.covered}/{result.valid}" if result.valid else "0/0"
        pct = f"{result.percentage:6.2f}%" if result.valid else "  N/A  "
        floor = f"{result.floor:5.1f}%"
        if result.passed:
            status = "PASS"
        else:
            status = "DRY-RUN" if dry_run else "FAIL"
        lines.append(f"{result.prefix:<60}  {ratio:>14}  {pct:>10}  {floor:>8}  {status}")
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
        "--dry-run",
        action="store_true",
        help="Report branch floor status without failing on floor misses.",
    )
    args = parser.parse_args(argv)

    if not args.coverage_xml.is_file():
        print(
            f"check_per_package_branch_coverage: coverage.xml not found at {args.coverage_xml}. "
            "This script must run after pytest-cov produces the report.",
            file=sys.stderr,
        )
        return 2

    if not BRANCH_FLOORS:
        print("No per-package branch coverage floors configured.")
        if args.dry_run:
            print("Dry-run branch coverage check completed; no floors enforced.")
        return 0

    results = aggregate(args.coverage_xml, BRANCH_FLOORS)
    print(render_table(results, dry_run=args.dry_run))

    failures = [result for result in results if not result.passed]
    if failures and not args.dry_run:
        print()
        print("Per-package branch coverage check FAILED for the following prefixes:", file=sys.stderr)
        for result in failures:
            if result.valid == 0:
                print(
                    f"  - {result.prefix} matched 0 branch-covered source files. "
                    "Did the package move or does it have no measured branches?",
                    file=sys.stderr,
                )
            else:
                print(
                    f"  - {result.prefix}: {result.percentage:.2f}% < floor {result.floor:.1f}% "
                    f"({result.covered}/{result.valid} branches)",
                    file=sys.stderr,
                )
        return 1

    if failures and args.dry_run:
        print()
        print("Dry-run only: branch floor misses were reported but not enforced.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
