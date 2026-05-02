#!/usr/bin/env python3
"""Enforce documentation placement and retention structure under docs/."""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
from dataclasses import dataclass

ALLOWED_DOCS_ROOT_FILES = {"README.md"}
ALLOWED_DOCS_TOP_LEVEL_DIRS = {
    "750_picacho",
    "_archive",
    "analysis",
    "apex",
    "api",
    "architecture",
    "archive",
    "brand",
    "ci",
    "ci_cd",
    "cli",
    "compliance",
    "contracts",
    "decisions",
    "deliverables",
    "deployment",
    "deprecation",
    "depth_model",
    "depth_pipeline",
    "development",
    "examples",
    "fixes",
    "governance",
    "guides",
    "historical",
    "implementation",
    "implementation_notes",
    "incidents",
    "investigations",
    "materials",
    "migration",
    "operations",
    "optimization",
    "performance",
    "pipeline",
    "pipeline_docs",
    "pr_archive",
    "pr_reports",
    "pr_summaries",
    "processing",
    "project-status",
    "projects",
    "quality_analysis",
    "quick_references",
    "reference",
    "reports",
    "runtimes",
    "schemas",
    "session_summaries",
    "sessions",
    "spatial_ai",
    "status",
    "summaries",
    "testing",
    "validation",
    "verification",
    "version_history",
    "visual_review",
    "workflow",
    "workflows",
}
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_LEGACY_ALLOWLIST_PATH = REPO_ROOT / "scripts" / "governance" / "docs_structure_legacy_allowlist.txt"


@dataclass(frozen=True)
class DocChange:
    """Single changed documentation path with its git status."""

    status: str
    path: str


def _run_git(args: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _parse_name_status_output(output: str) -> list[DocChange]:
    changes: list[DocChange] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if not parts:
            continue
        status = parts[0][:1]
        if status not in {"A", "C", "M", "R"}:
            continue
        if status in {"R", "C"}:
            if len(parts) < 3:
                continue
            path = parts[2]
        else:
            if len(parts) < 2:
                continue
            path = parts[1]
        changes.append(DocChange(status=status, path=path.replace("\\", "/")))
    return changes


def _changed_docs_files() -> tuple[list[DocChange] | None, list[str]]:
    commands = [
        ["diff", "--name-status", "--diff-filter=ACMR", "--cached", "--", "docs"],
        ["diff", "--name-status", "--diff-filter=ACMR", "--", "docs"],
        ["diff", "--name-status", "--diff-filter=ACMR", "HEAD^..HEAD", "--", "docs"],
        ["show", "--name-status", "--diff-filter=ACMR", "--pretty=format:", "HEAD", "--", "docs"],
    ]
    errors: list[str] = []
    had_success = False

    for cmd in commands:
        code, output, stderr = _run_git(cmd)
        if code != 0:
            cmd_text = "git " + " ".join(cmd)
            detail = stderr.strip() or f"exit {code}"
            errors.append(f"{cmd_text}: {detail}")
            continue
        had_success = True
        changes = _parse_name_status_output(output)
        if changes:
            return changes, []

    if had_success:
        return [], []

    return None, errors


def _all_docs_files() -> list[str]:
    code, output, _stderr = _run_git(["ls-files", "--", "docs"])
    if code == 0:
        return sorted(line.replace("\\", "/") for line in output.splitlines() if line.strip())

    docs_root = REPO_ROOT / "docs"
    if not docs_root.exists():
        return []

    return sorted(str(path.relative_to(REPO_ROOT)).replace("\\", "/") for path in docs_root.rglob("*") if path.is_file())


def _all_docs_changes() -> list[DocChange]:
    return [DocChange(status="A", path=path) for path in _all_docs_files()]


def _load_legacy_allowlist(path: pathlib.Path) -> set[str]:
    if not path.exists():
        return set()

    allowlist: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        allowlist.add(line.replace("\\", "/"))
    return allowlist


def _display_path(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _root_violation(change: DocChange) -> bool:
    normalized = change.path.replace("\\", "/")
    parts = pathlib.PurePosixPath(normalized).parts
    if len(parts) < 2 or parts[0] != "docs":
        return False

    if len(parts) == 2:
        return parts[1] not in ALLOWED_DOCS_ROOT_FILES

    return parts[1] not in ALLOWED_DOCS_TOP_LEVEL_DIRS


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate documentation placement rules.")
    parser.add_argument(
        "--changed-only",
        action="store_true",
        help="Scan only documentation files from the current git diff, even in CI.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Scan all files under docs/ instead of only the current git diff.",
    )
    parser.add_argument(
        "--legacy-allowlist",
        default=str(DEFAULT_LEGACY_ALLOWLIST_PATH),
        help="Path to a newline-delimited allowlist of legacy docs/ topology violations for full-repo scans.",
    )
    args = parser.parse_args()

    ci_mode = os.getenv("CI", "").strip().lower() == "true"
    scan_all = args.all or (ci_mode and not args.changed_only)
    legacy_allowlist = _load_legacy_allowlist(pathlib.Path(args.legacy_allowlist))

    if scan_all:
        candidates = _all_docs_changes()
    else:
        candidates, errors = _changed_docs_files()
        if candidates is None:
            if ci_mode:
                print("Unable to determine changed docs files in CI; failing closed.")
                for error in errors:
                    print(f"  - {error}")
                return 2
            print("Unable to determine changed docs files; falling back to full docs scan.")
            for error in errors:
                print(f"  - {error}")
            candidates = _all_docs_changes()

    if not candidates:
        print("No documentation files to validate.")
        return 0

    topology_violations = [change for change in candidates if _root_violation(change)]
    known_legacy_violations: list[DocChange] = []
    if scan_all and legacy_allowlist:
        known_legacy_violations = [change for change in topology_violations if change.path in legacy_allowlist]
        topology_violations = [change for change in topology_violations if change.path not in legacy_allowlist]

        stale_allowlist_entries = sorted(legacy_allowlist - {change.path for change in known_legacy_violations})
        if stale_allowlist_entries:
            print("Legacy docs allowlist contains paths that are no longer present:")
            for path in stale_allowlist_entries:
                print(f"  - {path}")
            print("Update the allowlist so repo-wide docs validation reflects the current baseline.")
            return 1

    if topology_violations:
        print("Documentation structure violations detected:")
        print("Allowed docs topology:")
        print("  - docs/README.md")
        print("  - docs/<approved-top-level-dir>/...")
        print("Approved top-level dirs under docs/:")
        for directory in sorted(ALLOWED_DOCS_TOP_LEVEL_DIRS):
            print(f"  - {directory}")
        print("Violations:")
        for change in topology_violations:
            print(f"  - [{change.status}] {change.path}")
        return 1

    if known_legacy_violations:
        print(
            "Documentation structure check passed with "
            f"{len(known_legacy_violations)} grandfathered legacy docs/ root file(s)."
        )
        print(f"Legacy baseline: {_display_path(pathlib.Path(args.legacy_allowlist))}")
        print("A dedicated cleanup PR is still required to remove the remaining debt.")
        return 0

    print(f"Documentation structure check passed ({len(candidates)} file(s) scanned).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
