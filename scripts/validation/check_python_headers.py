#!/usr/bin/env python3
"""Check Python header lines for invalid encoding-cookie-like text.

Policy:
- A shebang is allowed on line 1.
- A valid PEP 263 encoding comment is allowed on line 1 or 2.
- Header lines 1-2 must not contain ``coding:`` or ``encoding:`` tokens in a
  cookie-like form unless they are valid encoding comments.
"""

from __future__ import annotations

import argparse
import codecs
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HEADER_COOKIE_PATTERN = re.compile(r"\b(?:coding|encoding)\s*[:=]\s*([-\w.]+)")
PEP263_PATTERN = re.compile(r"coding[:=]\s*([-\w.]+)")


def iter_python_files(roots: Iterable[Path]) -> list[Path]:
    """Return Python source files under the supplied roots."""
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            if root.suffix == ".py":
                files.append(root)
            continue
        files.extend(path for path in root.rglob("*.py") if path.is_file())
    return sorted({path.resolve() for path in files})


def iter_tracked_python_files(project_root: Path = PROJECT_ROOT) -> list[Path]:
    """Return tracked Python files from git."""
    result = subprocess.run(
        ["git", "ls-files", "--", "*.py"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(f"git ls-files failed: {stderr or 'unknown error'}")
    files = [project_root / line for line in result.stdout.splitlines() if line.strip()]
    return sorted(path.resolve() for path in files)


def _read_header_lines(path: Path) -> list[str]:
    """Return the first two source lines, preserving trailing whitespace."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="replace")
    return text.splitlines()[:2]


def _is_valid_encoding_comment(line: str) -> bool:
    stripped = line.lstrip()
    if not stripped.startswith("#"):
        return False
    match = PEP263_PATTERN.search(stripped)
    if match is None:
        return False
    try:
        codecs.lookup(match.group(1))
    except LookupError:
        return False
    return True


def _header_line_violation(path: Path, line_number: int, line: str) -> str | None:
    match = HEADER_COOKIE_PATTERN.search(line)
    if match is None:
        return None
    if _is_valid_encoding_comment(line):
        return None
    token = match.group(0)
    return f"{path}:{line_number}: header contains cookie-like text {token!r}; " "use a valid PEP 263 encoding comment instead"


def find_violations(roots: Iterable[Path] | None = None) -> list[str]:
    """Return invalid Python header declarations."""
    scan_paths = iter_python_files(roots) if roots is not None else iter_tracked_python_files()
    violations: list[str] = []
    for path in scan_paths:
        header_lines = _read_header_lines(path)
        for index, line in enumerate(header_lines, start=1):
            if index == 1 and line.startswith("#!"):
                continue
            violation = _header_line_violation(path, index, line)
            if violation is not None:
                violations.append(violation)
    return violations


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Check Python source headers for invalid encoding-cookie-like text")
    parser.add_argument(
        "paths",
        nargs="*",
        help="Optional file or directory paths to scan instead of tracked Python files",
    )
    args = parser.parse_args()

    roots = tuple(Path(path) for path in args.paths) if args.paths else None
    violations = find_violations(roots)
    if not violations:
        print("OK: Python header declarations are valid")
        return 0

    print("Python header declaration violations detected:")
    for violation in violations:
        print(f"  - {violation}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
