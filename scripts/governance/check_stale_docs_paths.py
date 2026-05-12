#!/usr/bin/env python3
"""Detect stale root-level docs path references in changed files."""

from __future__ import annotations

import pathlib
import re
import subprocess

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
IGNORE_DIRS = {"__pycache__"}
IGNORE_SUFFIXES = {".pyc"}
# Test files may contain strings referencing fictional docs paths as test fixtures.
# Exclude them from stale docs path detection.
IGNORE_PATH_PATTERNS = {"tests/"}  # Paths starting with these are skipped
ARCHIVE_DOC_PREFIXES = ("docs/historical/", "docs/pr_archive/")
INTENTIONAL_MISSING_REF_TERMS = (
    "broken reference",
    "broken documentation reference",
    "did not exist",
    "does not exist",
    "missing target",
    "target does not exist",
)
PATH_PATTERN = re.compile(
    "".join(
        [
            r"(?<![A-Za-z0-9_/-])",
            r"(?:\./|(?:\.\./)+)?docs/",
            r"([A-Za-z0-9_-]+\.(?:md|txt|csv))",
            r"(?![A-Za-z0-9_/-])",
        ]
    ),
    re.IGNORECASE,
)


def _run_git(args: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _parse_name_only_output(output: str) -> list[pathlib.Path]:
    normalized = set()
    for line in output.splitlines():
        stripped = line.strip()
        if stripped:
            normalized.add(stripped.replace("\\", "/"))
    return [REPO_ROOT / path for path in sorted(normalized)]


def _changed_files() -> tuple[list[pathlib.Path] | None, list[str]]:
    commands = [
        [
            "diff",
            "--name-only",
            "--diff-filter=ACMRTUXB",
            "origin/main...HEAD",
        ],
        ["diff", "--name-only", "--diff-filter=ACMRTUXB", "--cached"],
        ["diff", "--name-only", "--diff-filter=ACMRTUXB"],
        ["diff", "--name-only", "--diff-filter=ACMRTUXB", "HEAD^..HEAD"],
        [
            "show",
            "--name-only",
            "--diff-filter=ACMRTUXB",
            "--pretty=format:",
            "HEAD",
        ],
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
        changed = _parse_name_only_output(output)
        if changed:
            return changed, []

    if had_success:
        return [], []

    return None, errors


def _should_skip(path: pathlib.Path, relative_path: pathlib.Path | None = None) -> bool:
    if any(part in IGNORE_DIRS for part in path.parts):
        return True
    if path.suffix.lower() in IGNORE_SUFFIXES:
        return True
    # Skip test files that may contain fictional docs paths as test fixtures
    if relative_path is not None:
        rel_str = str(relative_path).replace("\\", "/")
        if any(rel_str.startswith(pattern) for pattern in IGNORE_PATH_PATTERNS):
            return True
    return False


def _read_text_if_probably_text(path: pathlib.Path) -> str | None:
    try:
        payload = path.read_bytes()
    except OSError:
        return None

    if b"\0" in payload:
        return None

    return payload.decode("utf-8", errors="ignore")


def _allows_intentional_missing_ref(
    line: str,
    relative_path: pathlib.Path | None,
) -> bool:
    """Allow archived evidence to quote a path that was broken historically."""
    if relative_path is None:
        return False

    rel_str = str(relative_path).replace("\\", "/")
    if not rel_str.startswith(ARCHIVE_DOC_PREFIXES):
        return False

    lowered = line.lower()
    return any(term in lowered for term in INTENTIONAL_MISSING_REF_TERMS)


def _find_stale_refs(path: pathlib.Path, relative_path: pathlib.Path | None = None) -> list[str]:
    if not path.exists() or not path.is_file() or _should_skip(path, relative_path):
        return []

    text = _read_text_if_probably_text(path)
    if text is None:
        return []

    stale = set()
    for line in text.splitlines():
        for match in PATH_PATTERN.findall(line):
            target = REPO_ROOT / "docs" / match
            if target.exists() or _allows_intentional_missing_ref(line, relative_path):
                continue
            stale.add(f"docs/{match}")
    return sorted(stale)


def main() -> int:
    changed_files, errors = _changed_files()
    if changed_files is None:
        message = "".join(
            [
                "Unable to determine changed files for stale docs-path ",
                "validation.",
            ]
        )
        print(message)
        for error in errors:
            print(f"  - {error}")
        return 2

    if not changed_files:
        print("No changed files to scan.")
        return 0

    findings = []
    for path in changed_files:
        relative = path.relative_to(REPO_ROOT)
        for stale_ref in _find_stale_refs(path, relative):
            findings.append((str(relative), stale_ref))

    if findings:
        print("Stale documentation path references detected:")
        for file_path, stale_ref in sorted(set(findings)):
            print(f"  - {file_path}: references missing {stale_ref}")
        return 1

    print("No stale docs path references detected in changed files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
