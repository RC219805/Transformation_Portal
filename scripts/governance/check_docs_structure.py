#!/usr/bin/env python3
"""Enforce documentation placement and retention structure under docs/."""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import subprocess
import sys
from dataclasses import dataclass

KEYWORD_RE = re.compile(r"(SUMMARY|REPORT|COMPLETE|STATUS)", re.IGNORECASE)
ALLOWED_PREFIXES = ("docs/historical/", "docs/pr_archive/")
ALLOWED_DOCS_ROOT_FILES = {"README.md"}
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


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
    docs_root = REPO_ROOT / "docs"
    if not docs_root.exists():
        return []

    return sorted(str(path.relative_to(REPO_ROOT)).replace("\\", "/") for path in docs_root.rglob("*") if path.is_file())


def _all_docs_changes() -> list[DocChange]:
    return [DocChange(status="A", path=path) for path in _all_docs_files()]


def _root_violation(change: DocChange) -> bool:
    normalized = change.path.replace("\\", "/")
    parts = pathlib.PurePosixPath(normalized).parts
    if len(parts) != 2 or parts[0] != "docs":
        return False

    if parts[1] in ALLOWED_DOCS_ROOT_FILES:
        return False

    return change.status in {"A", "C", "R"}


def _keyword_violation(path_str: str) -> bool:
    normalized = path_str.replace("\\", "/")
    name = pathlib.PurePosixPath(normalized).name
    if not KEYWORD_RE.search(name):
        return False
    return not normalized.startswith(ALLOWED_PREFIXES)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate documentation placement rules.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Scan all files under docs/ instead of only the current git diff.",
    )
    args = parser.parse_args()

    if args.all:
        candidates = _all_docs_changes()
    else:
        candidates, errors = _changed_docs_files()
        if candidates is None:
            if os.getenv("CI", "").strip().lower() == "true":
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

    root_violations = [change for change in candidates if _root_violation(change)]
    keyword_violations = [change.path for change in candidates if _keyword_violation(change.path)]

    if root_violations or keyword_violations:
        print("Documentation structure violations detected:")
        if root_violations:
            print("New docs root files are restricted to docs/README.md:")
            for change in root_violations:
                print(f"  - [{change.status}] {change.path}")
        if keyword_violations:
            print("Files with SUMMARY/REPORT/COMPLETE/STATUS must be archived:")
            for path in keyword_violations:
                print(f"  - {path}")
            print("Allowed prefixes: docs/historical/ and docs/pr_archive/.")
        return 1

    print(f"Documentation structure check passed ({len(candidates)} file(s) scanned).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
