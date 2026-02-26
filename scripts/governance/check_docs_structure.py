#!/usr/bin/env python3
"""Enforce documentation placement and retention structure under docs/."""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import subprocess
import sys

KEYWORD_RE = re.compile(r"(SUMMARY|REPORT|COMPLETE|STATUS)", re.IGNORECASE)
ALLOWED_PREFIXES = ("docs/historical/", "docs/pr_archive/")
ALLOWED_DOCS_ROOT_FILES = {"README.md"}
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _run_git(args: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _changed_docs_files() -> tuple[list[str] | None, list[str]]:
    commands = [
        ["diff", "--name-only", "--diff-filter=ACMR", "--cached", "--", "docs"],
        ["diff", "--name-only", "--diff-filter=ACMR", "--", "docs"],
        ["diff", "--name-only", "--diff-filter=ACMR", "HEAD^..HEAD", "--", "docs"],
        ["show", "--name-only", "--diff-filter=ACMR", "--pretty=format:", "HEAD", "--", "docs"],
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
        paths = sorted({line.strip() for line in output.splitlines() if line.strip()})
        if paths:
            return paths, []

    if had_success:
        return [], []

    return None, errors


def _all_docs_files() -> list[str]:
    docs_root = REPO_ROOT / "docs"
    if not docs_root.exists():
        return []

    return sorted(str(path.relative_to(REPO_ROOT)).replace("\\", "/") for path in docs_root.rglob("*") if path.is_file())


def _root_violation(path_str: str) -> bool:
    normalized = path_str.replace("\\", "/")
    parts = pathlib.PurePosixPath(normalized).parts
    if len(parts) != 2 or parts[0] != "docs":
        return False

    return parts[1] not in ALLOWED_DOCS_ROOT_FILES


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
        candidates = _all_docs_files()
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
            candidates = _all_docs_files()

    if not candidates:
        print("No documentation files to validate.")
        return 0

    root_violations = [path for path in candidates if _root_violation(path)]
    keyword_violations = [path for path in candidates if _keyword_violation(path)]

    if root_violations or keyword_violations:
        print("Documentation structure violations detected:")
        if root_violations:
            print("Root docs files are restricted to docs/README.md:")
            for path in root_violations:
                print(f"  - {path}")
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
