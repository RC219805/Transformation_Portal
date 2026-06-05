#!/usr/bin/env python3
"""Check for bidirectional Unicode and other format-control characters."""

from __future__ import annotations

import argparse
import subprocess
import sys
import unicodedata
from pathlib import Path

BIDI_CHARS = {
    "\u202A": "LEFT-TO-RIGHT EMBEDDING",
    "\u202B": "RIGHT-TO-LEFT EMBEDDING",
    "\u202C": "POP DIRECTIONAL FORMATTING",
    "\u202D": "LEFT-TO-RIGHT OVERRIDE",
    "\u202E": "RIGHT-TO-LEFT OVERRIDE",
    "\u2066": "LEFT-TO-RIGHT ISOLATE",
    "\u2067": "RIGHT-TO-LEFT ISOLATE",
    "\u2068": "FIRST STRONG ISOLATE",
    "\u2069": "POP DIRECTIONAL ISOLATE",
}

TEXT_SUFFIXES = {".py", ".yml", ".yaml", ".md"}


def check_file(filepath: Path) -> list[str]:
    """Return Unicode-control issues for a single text file."""
    issues: list[str] = []
    try:
        content = filepath.read_text(encoding="utf-8")

        for line_num, line in enumerate(content.splitlines(), start=1):
            for col_num, char in enumerate(line, start=1):
                if char in BIDI_CHARS:
                    issues.append(
                        f"{filepath}:{line_num}:{col_num}: "
                        f"Bidirectional Unicode U+{ord(char):04X} ({BIDI_CHARS[char]})"
                    )
                elif unicodedata.category(char) == "Cf":
                    name = unicodedata.name(char, "UNKNOWN")
                    issues.append(
                        f"{filepath}:{line_num}:{col_num}: "
                        f"Format control character U+{ord(char):04X} ({name})"
                    )
    except (OSError, UnicodeDecodeError) as exc:
        issues.append(f"{filepath}: Error reading file: {exc}")

    return issues


def _is_supported_text_path(filepath: Path) -> bool:
    return filepath.suffix.lower() in TEXT_SUFFIXES


def _staged_files() -> list[Path]:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git diff --cached failed")

    return [
        Path(f)
        for f in result.stdout.strip().split("\n")
        if f and _is_supported_text_path(Path(f))
    ]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check text files for bidirectional Unicode and other format-control characters."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Text files to check. When omitted, staged Python, YAML, and Markdown files are checked.",
    )
    return parser.parse_args(argv)


def _candidate_files(paths: list[Path]) -> list[Path]:
    return [path for path in paths if _is_supported_text_path(path)]


def main(argv: list[str] | None = None) -> int:
    """Check explicit files or, when no files are supplied, staged text files."""
    args = _parse_args(argv)
    try:
        files = _candidate_files(args.paths) if args.paths else _staged_files()
    except RuntimeError as exc:
        print(f"Error getting staged files: {exc}", file=sys.stderr)
        return 1

    if not files:
        return 0

    all_issues: list[str] = []
    for filepath in files:
        if not filepath.exists():
            continue
        all_issues.extend(check_file(filepath))

    if all_issues:
        print("Found dangerous Unicode control characters:", file=sys.stderr)
        for issue in all_issues:
            print(f"  {issue}", file=sys.stderr)
        print("\nThese characters can enable 'Trojan Source' attacks.", file=sys.stderr)
        print("Please remove them before committing.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
