#!/usr/bin/env python3
"""Enforce per-package line-coverage floors against a Cobertura coverage.xml.

The repo-wide ``--cov-fail-under`` in ``.github/workflows/build.yml`` is a
single global floor. That number is dominated by large modules and hides
gaps in small but governed surfaces (run-card validators, tp.* contract
modules, lux_depth_v3 core seams). This script ratchets per-package
floors on top of that single global gate so a regression on, say,
``lux_depth_v3/validators/`` cannot be masked by coverage gains
elsewhere in the repo.

The script reads the Cobertura ``coverage.xml`` produced by ``pytest-cov``
(no second test run, no recomputation), aggregates ``lines-covered`` /
``lines-valid`` over all source files whose Cobertura ``filename`` matches
a configured prefix, and exits non-zero if any prefix is below its floor.

Each package floor is independent: the report lists ALL prefixes (covered
lines / total lines / percentage / floor / pass-fail) so contributors can
see at a glance where they stand, not just the first failure.

Floors here MUST only ratchet upward over time. If a refactor genuinely
moves coverage down, raise the matter in review and adjust intentionally
— do not silently lower a floor to make CI green.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


# Per-package floors. Keep the list small and focused on governed surfaces.
# Floors are line-coverage percentages (0-100). They are *floors*, not targets.
PACKAGE_FLOORS: tuple[tuple[str, float], ...] = (
    # tp.crypto / tp.merkle / tp.phase4 — contract & evidence chain
    ("src/tp/", 60.0),
    # Run-card validators are the binding contract surface for governed deliverables.
    ("src/transformation_portal/lux_depth_v3/validators/", 80.0),
    # Lux Depth V3 core orchestrator seams (config_resolver, pipeline_coordinator,
    # execution_engine, artifact_manager, manifest, provenance, run_card_contract,
    # io_atomic, reconstruction_manifest). Backends and the very large
    # segmentation_backend live alongside but are excluded by the deeper
    # validators/ floor above and treated under the global floor.
    ("src/transformation_portal/lux_depth_v3/", 50.0),
)


@dataclass(frozen=True)
class PackageResult:
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
        # If a prefix matches zero source files we treat it as a configuration
        # error (typo / moved package) — fail loud rather than silently pass.
        if self.valid == 0:
            return False
        return self.percentage >= self.floor


def _normalize_filename(raw: str) -> str:
    """Cobertura filenames may be relative or absolute; normalize to forward slashes."""
    return raw.replace("\\", "/").lstrip("./")


def _iter_class_elements(root: ET.Element) -> Iterable[ET.Element]:
    # coverage.py emits Cobertura with packages > package > classes > class.
    for cls in root.iter("class"):
        yield cls


def aggregate(coverage_xml: Path, floors: Iterable[tuple[str, float]]) -> List[PackageResult]:
    tree = ET.parse(coverage_xml)
    root = tree.getroot()

    classes = list(_iter_class_elements(root))
    results: list[PackageResult] = []
    for prefix, floor in floors:
        covered = 0
        valid = 0
        normalized_prefix = prefix.replace("\\", "/")
        for cls in classes:
            filename = _normalize_filename(cls.attrib.get("filename", ""))
            if not filename.startswith(normalized_prefix):
                # Some coverage configurations strip the leading "src/" — try
                # matching against the package path with that segment removed.
                stripped = normalized_prefix.removeprefix("src/")
                if not stripped or not filename.startswith(stripped):
                    continue
            lines = cls.find("lines")
            if lines is None:
                continue
            for line in lines.findall("line"):
                valid += 1
                hits = int(line.attrib.get("hits", "0"))
                if hits > 0:
                    covered += 1
        results.append(PackageResult(prefix=prefix, floor=floor, covered=covered, valid=valid))
    return results


def render_table(results: List[PackageResult]) -> str:
    header = f"{'Package':<60}  {'Lines':>14}  {'Coverage':>10}  {'Floor':>8}  Status"
    lines = [header, "-" * len(header)]
    for r in results:
        ratio = f"{r.covered}/{r.valid}" if r.valid else "0/0"
        pct = f"{r.percentage:6.2f}%" if r.valid else "  N/A  "
        floor = f"{r.floor:5.1f}%"
        status = "PASS" if r.passed else "FAIL"
        lines.append(f"{r.prefix:<60}  {ratio:>14}  {pct:>10}  {floor:>8}  {status}")
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
    args = parser.parse_args(argv)

    if not args.coverage_xml.is_file():
        print(
            f"check_per_package_coverage: coverage.xml not found at {args.coverage_xml}. "
            "This script must run after pytest-cov produces the report.",
            file=sys.stderr,
        )
        return 2

    results = aggregate(args.coverage_xml, PACKAGE_FLOORS)
    print(render_table(results))

    failures = [r for r in results if not r.passed]
    if failures:
        print()
        print("Per-package coverage check FAILED for the following prefixes:", file=sys.stderr)
        for r in failures:
            if r.valid == 0:
                print(
                    f"  - {r.prefix} matched 0 covered source files. "
                    "Did the package move? Update PACKAGE_FLOORS.",
                    file=sys.stderr,
                )
            else:
                print(
                    f"  - {r.prefix}: {r.percentage:.2f}% < floor {r.floor:.1f}% "
                    f"({r.covered}/{r.valid} lines)",
                    file=sys.stderr,
                )
        print(
            "\nAdd tests for the failing package or, if intentional, "
            "raise the matter in review before lowering the floor.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
