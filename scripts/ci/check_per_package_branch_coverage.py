#!/usr/bin/env python3
"""Enforce branch-coverage floors for governed package prefixes.

This companion to ``check_per_package_coverage.py`` reads the same Cobertura
``coverage.xml`` and enforces conservative branch floors for the cold-zone
package set. ``--dry-run`` remains available for local floor proposals, but CI
uses enforcing mode.
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


# Cold-Zone Coverage Program branch ratchets. These floors were raised after
# repeated stable CI runs and a fresh 2026-05-13 required CI coverage snapshot.
# Prefixes with cross-lane variance keep extra headroom until the next measured
# ratchet.
BRANCH_FLOORS: tuple[BranchFloor, ...] = (
    BranchFloor("src/transformation_portal/plugins/", 40.0),
    BranchFloor("src/transformation_portal/stage_graph/", 63.0),
    BranchFloor("src/transformation_portal/vlm/", 55.0),
    BranchFloor("src/transformation_portal/depth/", 42.0),
    BranchFloor("src/transformation_portal/streaming/", 29.0),
    BranchFloor("src/transformation_portal/spatial_ai/reconstruction/", 47.0),
    # FastAPI origin + governed paid-pilot durable-state backends. Measured in
    # the 2026-06-06 core-lane snapshot at app.py 75.3%, orchestrator/storage/
    # 51.4%, orchestrator/queue/ 43.8%, orchestrator/artifact_store/ 54.7%
    # branch coverage. Conservative starters below those values; ratchet upward
    # after a confirming required-CI run. See check_per_package_coverage.py for
    # the matching line-coverage floors and rationale.
    BranchFloor("app.py", 71.0),
    BranchFloor("src/transformation_portal/orchestrator/storage/", 48.0),
    BranchFloor("src/transformation_portal/orchestrator/queue/", 40.0),
    BranchFloor("src/transformation_portal/orchestrator/artifact_store/", 50.0),
    # Performance ledger CLI — branch coverage reached ~93% on 2026-06-06 via
    # tests/test_metrics_ledger.py; conservative floor locks it in. See the
    # matching line floor in check_per_package_coverage.py.
    BranchFloor("src/transformation_portal/metrics/ledger.py", 90.0),
    # ComfyUI workflow builder — branch coverage reached 100% on 2026-06-06 via
    # tests/test_comfyui_workflow_builder.py. Conservative floor locks it in.
    BranchFloor("src/transformation_portal/comfyui/workflow_builder.py", 92.0),
    # ComfyUI workflow templates — pure declarative builders and serialization
    # only. Custom variant-list tests cover the non-default target/strength
    # branches; no executor, torch, ML, or network runtime path is invoked.
    BranchFloor("src/transformation_portal/comfyui/workflow_templates.py", 96.0),
    # Universal hardening wrapper — branch coverage reached 100% on 2026-06-06
    # via the input-validation path tests. Conservative package floor.
    BranchFloor("src/transformation_portal/hardening/", 95.0),
    # CAS store — branch coverage reached ~89% on 2026-06-06 via the lifecycle
    # tests. Conservative floor below that (residual misses are race/IO-fault).
    BranchFloor("src/transformation_portal/storage/cas_store.py", 85.0),
    # Orchestrator worker runner — branch coverage reached 100% on 2026-06-06
    # via the unit tests. Conservative floor.
    BranchFloor("src/transformation_portal/orchestrator/worker.py", 95.0),
    # Dashboard node-state store — branch coverage reached ~97% on 2026-06-11
    # via the lifecycle tests. Conservative file-level floor.
    BranchFloor("src/transformation_portal/dashboard/node_state_store.py", 90.0),
    # Dashboard execution manager — branch coverage reached ~82% on 2026-06-11
    # via the lifecycle/cancellation tests. Conservative floor (residual misses
    # are the legacy inline-trim and post-loop cancellation edge branches).
    BranchFloor("src/transformation_portal/dashboard/execution_manager.py", 75.0),
    # Dashboard time-travel router — branch coverage reached 100% on 2026-06-11
    # via the TestClient route-branch tests. Conservative floor.
    BranchFloor("src/transformation_portal/dashboard/time_travel.py", 90.0),
    # Dashboard studio inspector — branch coverage reached 100% on 2026-06-11
    # (only two branches; the rest is HTML). Conservative floor.
    BranchFloor("src/transformation_portal/dashboard/studio_inspector.py", 80.0),
    # Dashboard node-inspection router — branch coverage reached 100% on
    # 2026-06-11 via the TestClient route tests. Conservative floor.
    BranchFloor("src/transformation_portal/dashboard/node_api.py", 90.0),
    # Dashboard experiment API — branch coverage reached 100% on 2026-06-11
    # via the Python-API + router tests. Conservative floor.
    BranchFloor("src/transformation_portal/dashboard/experiment_api.py", 90.0),
    # Remaining dashboard routers + server, covered 2026-06-11. Branch floors
    # are conservative below the measured offline branch coverage; gpu_api and
    # optimization_api carry inherent offline ceilings (no GPU / stubbed
    # optimizer), so their floors are set accordingly.
    BranchFloor("src/transformation_portal/dashboard/server.py", 66.0),
    BranchFloor("src/transformation_portal/dashboard/gpu_api.py", 36.0),
    BranchFloor("src/transformation_portal/dashboard/dag_api.py", 90.0),
    BranchFloor("src/transformation_portal/dashboard/artifact_api.py", 76.0),
    BranchFloor("src/transformation_portal/dashboard/artifact_preview.py", 85.0),
    BranchFloor("src/transformation_portal/dashboard/optimization_api.py", 72.0),
    BranchFloor("src/transformation_portal/dashboard/rl_api.py", 65.0),
    BranchFloor("src/transformation_portal/dashboard/dag_editor_api.py", 68.0),
    BranchFloor("src/transformation_portal/dashboard/execution_api.py", 72.0),
    # Package-level branch floor for the whole dashboard package (~87% measured
    # 2026-06-11); guards future unfloored additions.
    BranchFloor("src/transformation_portal/dashboard/", 75.0),
    # Luxury video master grader — branch coverage reached ~84% on 2026-06-11
    # via the build_config + main() control-flow tests. Conservative floor.
    BranchFloor("src/transformation_portal/processors/luxury_video_master_grader.py", 76.0),
)

# If a future branch temporarily clears floors, dry-run mode still reports the
# package-level branch baselines needed for review.
DRY_RUN_BRANCH_PREFIXES: tuple[str, ...] = (
    "src/transformation_portal/plugins/",
    "src/transformation_portal/stage_graph/",
    "src/transformation_portal/vlm/",
    "src/transformation_portal/depth/",
    "src/transformation_portal/streaming/",
    "src/transformation_portal/spatial_ai/reconstruction/",
    "app.py",
    "src/transformation_portal/orchestrator/storage/",
    "src/transformation_portal/orchestrator/queue/",
    "src/transformation_portal/orchestrator/artifact_store/",
    "src/transformation_portal/metrics/ledger.py",
    "src/transformation_portal/comfyui/workflow_builder.py",
    "src/transformation_portal/comfyui/workflow_templates.py",
    "src/transformation_portal/hardening/",
    "src/transformation_portal/storage/cas_store.py",
    "src/transformation_portal/orchestrator/worker.py",
    "src/transformation_portal/dashboard/node_state_store.py",
    "src/transformation_portal/dashboard/execution_manager.py",
    "src/transformation_portal/dashboard/time_travel.py",
    "src/transformation_portal/dashboard/studio_inspector.py",
    "src/transformation_portal/dashboard/node_api.py",
    "src/transformation_portal/dashboard/experiment_api.py",
    "src/transformation_portal/dashboard/server.py",
    "src/transformation_portal/dashboard/gpu_api.py",
    "src/transformation_portal/dashboard/dag_api.py",
    "src/transformation_portal/dashboard/artifact_api.py",
    "src/transformation_portal/dashboard/artifact_preview.py",
    "src/transformation_portal/dashboard/optimization_api.py",
    "src/transformation_portal/dashboard/rl_api.py",
    "src/transformation_portal/dashboard/dag_editor_api.py",
    "src/transformation_portal/dashboard/execution_api.py",
    "src/transformation_portal/dashboard/",
    "src/transformation_portal/processors/luxury_video_master_grader.py",
)


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


def aggregate_prefixes(coverage_xml: Path, prefixes: Iterable[str]) -> list[BranchResult]:
    return aggregate(coverage_xml, (BranchFloor(prefix, 0.0) for prefix in prefixes))


def render_table(results: list[BranchResult], *, dry_run: bool, enforce_floors: bool = True) -> str:
    header = f"{'Package':<60}  {'Branches':>14}  {'Coverage':>10}  {'Floor':>8}  Status"
    lines = [header, "-" * len(header)]
    for result in results:
        ratio = f"{result.covered}/{result.valid}" if result.valid else "0/0"
        pct = f"{result.percentage:6.2f}%" if result.valid else "  N/A  "
        floor = f"{result.floor:5.1f}%" if enforce_floors else "  N/A  "
        if not enforce_floors:
            status = "DRY-RUN" if dry_run else "INFO"
        elif result.passed:
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
            results = aggregate_prefixes(args.coverage_xml, DRY_RUN_BRANCH_PREFIXES)
            print(render_table(results, dry_run=True, enforce_floors=False))
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
