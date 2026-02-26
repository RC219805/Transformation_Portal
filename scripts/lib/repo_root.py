#!/usr/bin/env python3
"""Deterministic repository root discovery using dual anchors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

ANCHOR_FILE = "pyproject.toml"
ANCHOR_DIR = Path(".github") / "workflows"


class RepoRootError(RuntimeError):
    """Raised when repository root discovery fails validation."""


def _has_required_anchors(candidate: Path) -> bool:
    return (candidate / ANCHOR_FILE).is_file() and (candidate / ANCHOR_DIR).is_dir()


def _walk_up(start: Path) -> Iterable[Path]:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    yield current
    yield from current.parents


def resolve_repo_root(start: Path | None = None, repo: Path | None = None) -> Path:
    """Resolve repository root by explicit override or dual-anchor upward walk."""
    if repo is not None:
        candidate = repo.expanduser().resolve()
        if not _has_required_anchors(candidate):
            raise RepoRootError(f"Invalid --repo path: {candidate} " f"(expected {ANCHOR_FILE} and {ANCHOR_DIR.as_posix()}/).")
        return candidate

    origin = (start or Path(__file__)).resolve()
    for candidate in _walk_up(origin):
        if _has_required_anchors(candidate):
            return candidate

    raise RepoRootError(
        f"Unable to locate repository root from {origin}. " f"Expected both {ANCHOR_FILE} and {ANCHOR_DIR.as_posix()}/."
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resolve repository root using dual anchors.")
    parser.add_argument("--repo", type=Path, help="Explicit repository path override.")
    parser.add_argument(
        "--start",
        type=Path,
        default=Path(__file__),
        help="Start path for upward search when --repo is not provided.",
    )
    parser.add_argument("--print", action="store_true", help="Print resolved root path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        root = resolve_repo_root(start=args.start, repo=args.repo)
    except RepoRootError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.print:
        print(root)
    else:
        print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
