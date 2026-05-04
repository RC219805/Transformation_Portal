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

A floor may declare ``exclude_prefixes`` so a parent rollup does NOT
double-count files that already have their own stricter nested floor.
For example, ``lux_depth_v3/`` excludes ``lux_depth_v3/validators/``
because the validators directory has its own 80% floor and including it
in the parent 50% rollup would let high validator coverage mask
regressions in the rest of ``lux_depth_v3/``.

Floors here MUST only ratchet upward over time. If a refactor genuinely
moves coverage down, raise the matter in review and adjust intentionally
— do not silently lower a floor to make CI green.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List


@dataclass(frozen=True)
class PackageFloor:
    """A coverage gate over a path prefix, optionally excluding nested prefixes."""

    prefix: str
    floor: float
    exclude_prefixes: tuple[str, ...] = field(default_factory=tuple)


# Per-package floors. Keep the list small and focused on governed surfaces.
# Floors are line-coverage percentages (0-100). They are *floors*, not targets.
PACKAGE_FLOORS: tuple[PackageFloor, ...] = (
    # tp.crypto / tp.merkle / tp.phase4 — contract & evidence chain
    PackageFloor("src/tp/", 60.0),
    # Run-card validators are the binding contract surface for governed deliverables.
    PackageFloor("src/transformation_portal/lux_depth_v3/validators/", 80.0),
    # Lux Depth V3 core orchestrator seams (config_resolver, pipeline_coordinator,
    # execution_engine, artifact_manager, manifest, provenance, run_card_contract,
    # io_atomic, reconstruction_manifest). Validators have their own stricter
    # floor above; explicitly exclude them so a high validator percentage
    # cannot mask a regression in the rest of lux_depth_v3.
    PackageFloor(
        "src/transformation_portal/lux_depth_v3/",
        50.0,
        exclude_prefixes=("src/transformation_portal/lux_depth_v3/validators/",),
    ),
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
    """Normalize a Cobertura ``filename`` to a repo-relative ``src/...`` form.

    Cobertura emits filenames in whichever form the coverage configuration
    produces — relative (``src/tp/x.py``), bare-style (``./src/tp/x.py``),
    or absolute (``/home/runner/work/repo/src/tp/x.py``). All three must
    normalize to the same string so prefix matching is reliable.

    Strategy:
    1. Backslashes → forward slashes (Windows-style runners).
    2. Strip a single leading ``./`` if present.
    3. If the result is still absolute, find the *last* occurrence of
       ``/src/`` and slice from there, dropping the absolute prefix.
       Falling back to the last occurrence (rather than the first) is
       the safe move when the repo path itself contains ``/src`` —
       e.g. ``/work/my-src-repo/src/tp/x.py``.
    """
    norm = raw.replace("\\", "/")
    if norm.startswith("./"):
        norm = norm[2:]
    if norm.startswith("/"):
        marker = "/src/"
        idx = norm.rfind(marker)
        if idx != -1:
            # Skip the leading slash so the result starts with "src/".
            norm = norm[idx + 1 :]
    return norm


def _matches_prefix(filename: str, prefix: str) -> bool:
    """Return True if filename is matched by prefix (with src/-stripped fallback)."""
    if filename.startswith(prefix):
        return True
    # Some coverage configurations strip the leading "src/" — try
    # matching against the package path with that segment removed.
    stripped = prefix.removeprefix("src/")
    return bool(stripped) and filename.startswith(stripped)


def _iter_class_elements(root: ET.Element) -> Iterable[ET.Element]:
    # coverage.py emits Cobertura with packages > package > classes > class.
    for cls in root.iter("class"):
        yield cls


def aggregate(coverage_xml: Path, floors: Iterable[PackageFloor]) -> List[PackageResult]:
    tree = ET.parse(coverage_xml)
    root = tree.getroot()

    classes = list(_iter_class_elements(root))
    results: list[PackageResult] = []
    for floor_spec in floors:
        covered = 0
        valid = 0
        normalized_prefix = floor_spec.prefix.replace("\\", "/")
        normalized_excludes = tuple(p.replace("\\", "/") for p in floor_spec.exclude_prefixes)
        for cls in classes:
            filename = _normalize_filename(cls.attrib.get("filename", ""))
            if not _matches_prefix(filename, normalized_prefix):
                continue
            if any(_matches_prefix(filename, excl) for excl in normalized_excludes):
                continue
            lines = cls.find("lines")
            if lines is None:
                continue
            for line in lines.findall("line"):
                valid += 1
                hits = int(line.attrib.get("hits", "0"))
                if hits > 0:
                    covered += 1
        results.append(PackageResult(prefix=floor_spec.prefix, floor=floor_spec.floor, covered=covered, valid=valid))
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
                    f"  - {r.prefix} matched 0 covered source files. " "Did the package move? Update PACKAGE_FLOORS.",
                    file=sys.stderr,
                )
            else:
                print(
                    f"  - {r.prefix}: {r.percentage:.2f}% < floor {r.floor:.1f}% " f"({r.covered}/{r.valid} lines)",
                    file=sys.stderr,
                )
        print(
            "\nAdd tests for the failing package or, if intentional, " "raise the matter in review before lowering the floor.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
