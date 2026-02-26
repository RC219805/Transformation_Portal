#!/usr/bin/env python3
"""Enforce documentation placement rules for summary/report/status artifacts."""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

KEYWORD_RE = re.compile(r"(SUMMARY|REPORT|COMPLETE|STATUS)", re.IGNORECASE)
ALLOWED_PREFIXES = ("docs/historical/", "docs/pr_archive/")
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _run_git(args: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout


def _changed_docs_files() -> list[str]:
    commands = [
        ["diff", "--name-only", "--diff-filter=ACMR", "--cached", "--", "docs"],
        ["diff", "--name-only", "--diff-filter=ACMR", "--", "docs"],
        ["diff", "--name-only", "--diff-filter=ACMR", "HEAD^..HEAD", "--", "docs"],
        ["show", "--name-only", "--diff-filter=ACMR", "--pretty=format:", "HEAD", "--", "docs"],
    ]

    for cmd in commands:
        code, output = _run_git(cmd)
        if code != 0:
            continue
        paths = sorted({line.strip() for line in output.splitlines() if line.strip()})
        if paths:
            return paths

    return []


def _all_docs_files() -> list[str]:
    docs_root = REPO_ROOT / "docs"
    if not docs_root.exists():
        return []

    return sorted(str(path.relative_to(REPO_ROOT)).replace("\\", "/") for path in docs_root.rglob("*") if path.is_file())


def _is_violation(path_str: str) -> bool:
    normalized = path_str.replace("\\", "/")
    name = pathlib.PurePosixPath(normalized).name
    if not KEYWORD_RE.search(name):
        return False
    return not normalized.startswith(ALLOWED_PREFIXES)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate documentation placement for summary/report/status artifacts.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Scan all files under docs/ instead of only the current git diff.",
    )
    args = parser.parse_args()

    candidates = _all_docs_files() if args.all else _changed_docs_files()
    if not candidates:
        print("No documentation files to validate.")
        return 0

    violations = [path for path in candidates if _is_violation(path)]

    if violations:
        print("Documentation structure violations detected:")
        for path in violations:
            print(f"  - {path}")
        print(
            "\nFiles with SUMMARY/REPORT/COMPLETE/STATUS in the filename must be under "
            "docs/historical/ or docs/pr_archive/."
        )
        return 1

    print(f"Documentation structure check passed ({len(candidates)} file(s) scanned).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
