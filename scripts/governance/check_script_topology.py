#!/usr/bin/env python3
"""Validate governed script placement and compatibility-wrapper contracts."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

COMPATIBILITY_WRAPPERS = {
    "scripts/create_board_textures.py": (
        "scripts/utilities/create_board_textures.py",
        "from scripts.utilities.create_board_textures import main",
    ),
    "scripts/download_depth_models.py": (
        "scripts/setup/download_depth_models.py",
        "from scripts.setup.download_depth_models import",
    ),
    "scripts/install_models.py": (
        "scripts/setup/install_models.py",
        "from scripts.setup.install_models import",
    ),
    "scripts/install_models_auto.py": (
        "scripts/setup/install_models_auto.py",
        "from scripts.setup.install_models_auto import",
    ),
    "scripts/run_aerial_enhancement.py": (
        "scripts/pipelines/run_aerial_enhancement.py",
        "from scripts.pipelines.run_aerial_enhancement import",
    ),
    "scripts/synthetic_viewer.py": (
        "src/transformation_portal/perceptual/synthetic_viewer.py",
        "from transformation_portal.perceptual.synthetic_viewer import",
    ),
    "scripts/visualize_material_assignments.py": (
        "scripts/utilities/visualize_material_assignments.py",
        "from scripts.utilities.visualize_material_assignments import",
    ),
}

RETIRED_ORGANIZER_PATHS = {
    "archive/.organize_docs.sh": "archive/scripts/legacy-organization/organize_docs_root_legacy.sh",
    "scripts/install_models_old_backup.py": "archive/scripts/legacy-organization/install_models_old_backup.py",
    "scripts/organize_outputs.sh": "archive/scripts/legacy-organization/organize_outputs.sh",
    "scripts/organize_remaining.sh": "archive/scripts/legacy-organization/organize_remaining.sh",
    "scripts/organize_root_files.sh": "archive/scripts/legacy-organization/organize_root_files.sh",
    "scripts/organize_scripts.sh": "archive/scripts/legacy-organization/organize_scripts.sh",
}

ALLOWED_SCRIPT_DOCS = {
    "scripts/README.md",
    "scripts/README_QUALITY_CONTROL.md",
    "scripts/QUICKSTART_QUALITY.md",
    "scripts/TEST_V2_INTEGRATION_README.md",
}


@dataclass(frozen=True)
class TopologyViolation:
    """Single script-topology violation."""

    path: str
    reason: str
    suggestion: str


def _git_ls_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git ls-files failed")
    return sorted(line.strip() for line in result.stdout.splitlines() if line.strip())


def _read_repo_text(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def validate_script_topology(
    tracked_paths: Iterable[str],
    *,
    read_text: Callable[[str], str] = _read_repo_text,
) -> list[TopologyViolation]:
    """Return deterministic script-topology violations for tracked paths."""
    tracked = set(tracked_paths)
    violations: list[TopologyViolation] = []

    for path, destination in sorted(RETIRED_ORGANIZER_PATHS.items()):
        if path in tracked:
            violations.append(
                TopologyViolation(
                    path=path,
                    reason="retired broad-mutating organization helper remains active",
                    suggestion=f"move to {destination}",
                )
            )

    for path in sorted(p for p in tracked if p.startswith("scripts/") and Path(p).parent == Path("scripts")):
        if path in ALLOWED_SCRIPT_DOCS:
            continue
        suffix = Path(path).suffix.lower()
        if suffix in {".md", ".txt"}:
            violations.append(
                TopologyViolation(
                    path=path,
                    reason="historical script report is stored in active scripts root",
                    suggestion="move historical evidence to docs/historical/script-audits/",
                )
            )

    for wrapper, (canonical, import_marker) in sorted(COMPATIBILITY_WRAPPERS.items()):
        wrapper_present = wrapper in tracked
        canonical_present = canonical in tracked
        if wrapper_present and not canonical_present:
            violations.append(
                TopologyViolation(
                    path=wrapper,
                    reason="compatibility wrapper has no tracked canonical implementation",
                    suggestion=f"restore canonical implementation at {canonical}",
                )
            )
            continue
        if canonical_present and not wrapper_present:
            violations.append(
                TopologyViolation(
                    path=canonical,
                    reason="canonical implementation is missing its public compatibility wrapper",
                    suggestion=f"restore wrapper at {wrapper}",
                )
            )
            continue
        if not wrapper_present:
            continue
        try:
            wrapper_text = read_text(wrapper)
        except OSError as exc:
            violations.append(
                TopologyViolation(
                    path=wrapper,
                    reason=f"compatibility wrapper could not be read: {exc}",
                    suggestion="restore a readable wrapper file",
                )
            )
            continue
        if import_marker not in wrapper_text:
            violations.append(
                TopologyViolation(
                    path=wrapper,
                    reason="compatibility wrapper does not delegate to canonical implementation",
                    suggestion=f"import and delegate to {canonical}",
                )
            )

    return sorted(violations, key=lambda item: item.path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate governed script topology.")
    parser.add_argument("--verbose", action="store_true", help="Print extra pass/fail detail.")
    args = parser.parse_args()

    try:
        violations = validate_script_topology(_git_ls_files())
    except RuntimeError as exc:
        print(f"Unable to collect tracked paths: {exc}", file=sys.stderr)
        return 2

    if violations:
        print("Script topology violations detected:")
        for violation in violations:
            print(f"  - {violation.path}")
            print(f"    reason: {violation.reason}")
            print(f"    suggested: {violation.suggestion}")
        return 1

    if args.verbose:
        print("Script topology check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
