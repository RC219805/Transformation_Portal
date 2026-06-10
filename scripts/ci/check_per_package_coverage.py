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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List

try:
    from scripts.ci.cobertura_xml import _matches_prefix, load_cobertura_files
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from cobertura_xml import _matches_prefix, load_cobertura_files  # type: ignore[no-redef]


@dataclass(frozen=True)
class PackageFloor:
    """A coverage gate over a path prefix, optionally excluding nested prefixes."""

    prefix: str
    floor: float
    exclude_prefixes: tuple[str, ...] = field(default_factory=tuple)


# Per-package floors. Keep the list small and focused on governed surfaces.
# Floors are line-coverage percentages (0-100). They are *floors*, not
# targets — set conservatively below the measured baseline so any
# regression fails CI, then ratchet upward in a follow-up as coverage
# improves. NEVER silently lower a floor to make CI green; raise the
# matter in review and adjust intentionally.
PACKAGE_FLOORS: tuple[PackageFloor, ...] = (
    # tp.crypto / tp.merkle / tp.phase4 — contract & evidence chain.
    # Conservative starter; tp.* is a small surface so coverage is volatile.
    PackageFloor("src/tp/", 40.0),
    # Run-card validators are the binding contract surface for governed
    # deliverables. The existing test_verify_run_card_integrity.py is
    # comprehensive (~30 cases / 941 LOC) so 70% should comfortably hold
    # — ratchet upward once a CI run confirms.
    PackageFloor("src/transformation_portal/lux_depth_v3/validators/", 70.0),
    # Lux Depth V3 core orchestrator seams (config_resolver, pipeline_coordinator,
    # execution_engine, artifact_manager, manifest, provenance, run_card_contract,
    # io_atomic, reconstruction_manifest). Conservative 30% starter — large
    # surface (~10k LOC) where coverage is heavier in some seams than others.
    # Validators have their own stricter floor above; explicitly exclude
    # them so a high validator percentage cannot mask a regression in the
    # rest of lux_depth_v3.
    PackageFloor(
        "src/transformation_portal/lux_depth_v3/",
        30.0,
        exclude_prefixes=("src/transformation_portal/lux_depth_v3/validators/",),
    ),
    # Cold-Zone Coverage Program stability ratchets. These floors were
    # raised after repeated stable CI runs and a fresh 2026-05-13 required
    # CI coverage snapshot. Prefixes with cross-lane variance keep extra
    # headroom until the next measured ratchet.
    PackageFloor("src/transformation_portal/plugins/", 48.0),
    PackageFloor("src/transformation_portal/stage_graph/", 74.0),
    PackageFloor("src/transformation_portal/vlm/", 69.0),
    PackageFloor("src/transformation_portal/depth/", 57.0),
    PackageFloor("src/transformation_portal/streaming/", 53.0),
    PackageFloor("src/transformation_portal/spatial_ai/reconstruction/", 42.0),
    # FastAPI origin. app.py is already measured in the core-tier coverage step
    # (build.yml runs `--cov=app`), and a 2026-06-06 core-lane snapshot put it at
    # ~83.8% line coverage, but until now nothing FLOORED it — the most
    # security-critical hardening surface (allowed-root validation, API-key /
    # trusted-host enforcement, request limits, pipeline allowlists) could
    # silently regress. Conservative starter well below the measured value to
    # absorb cross-lane variance; ratchet upward after a confirming CI run.
    PackageFloor("app.py", 76.0),
    # Governed paid-pilot durable-state backends. These three packages are the
    # first-class JobRepository / QueueBroker / ArtifactStore Protocol surfaces
    # (memory + Postgres/Redis/S3 implementations). The Postgres/Redis/S3 paths
    # only get full exercise behind the opt-in live-service contract gates, so
    # the core-lane rollup leans on the in-memory/local implementations. Floors
    # set below the 2026-06-06 core-lane snapshot (storage 68.0%, queue 67.6%,
    # artifact_store 65.9%) so the in-memory/local contract coverage cannot
    # silently regress; raise once the live-service lanes are folded in.
    PackageFloor("src/transformation_portal/orchestrator/storage/", 60.0),
    PackageFloor("src/transformation_portal/orchestrator/queue/", 58.0),
    PackageFloor("src/transformation_portal/orchestrator/artifact_store/", 58.0),
    # Performance ledger CLI (pure-Python SQLite). Behavioral tests in
    # tests/test_metrics_ledger.py took this from ~30% to ~98.8% line coverage
    # on 2026-06-06; this file-level floor locks the gain in. The rest of
    # metrics/ stays unfloored (several siblings are ML/metric backends).
    PackageFloor("src/transformation_portal/metrics/ledger.py", 90.0),
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


def aggregate(coverage_xml: Path, floors: Iterable[PackageFloor]) -> List[PackageResult]:
    files = load_cobertura_files(coverage_xml)
    results: list[PackageResult] = []
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
            valid += coverage_file.valid_lines
            covered += coverage_file.covered_lines
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
