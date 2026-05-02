#!/usr/bin/env python3
"""Validate markdown links that target headings in related docs."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)\s]+)\)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True)
class HeadingReference:
    source: Path
    target: Path
    heading_prefix: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_sources() -> tuple[Path, ...]:
    root = _repo_root()
    return (
        root / "docs" / "fixes" / "BINARY_FILE_BEST_PRACTICES.md",
        root / "docs" / "deliverables" / "QUICK_WINS.md",
        root / "docs" / "analysis" / "TODO_INVENTORY.md",
    )


def _default_heading_references() -> tuple[HeadingReference, ...]:
    root = _repo_root()
    binary_best_practices = root / "docs" / "fixes" / "BINARY_FILE_BEST_PRACTICES.md"
    return (
        HeadingReference(
            source=binary_best_practices,
            target=root / "docs" / "deliverables" / "QUICK_WINS.md",
            heading_prefix="QW-3:",
        ),
        HeadingReference(
            source=binary_best_practices,
            target=root / "docs" / "analysis" / "TODO_INVENTORY.md",
            heading_prefix="4.1 Binary File Cleanup",
        ),
    )


def _heading_texts(path: Path) -> list[str]:
    headings: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        match = HEADING_RE.match(raw_line)
        if match:
            headings.append(match.group(2).strip())
    return headings


def _slugify_heading(heading: str) -> str:
    text = heading.strip().lower()
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"[^\w\s.-]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def _heading_slugs(path: Path) -> set[str]:
    counts: dict[str, int] = {}
    slugs: set[str] = set()
    for heading in _heading_texts(path):
        base = _slugify_heading(heading)
        if not base:
            continue
        count = counts.get(base, 0)
        counts[base] = count + 1
        slugs.add(base if count == 0 else f"{base}-{count}")
    return slugs


def _iter_markdown_heading_links(source: Path) -> Iterable[tuple[Path, str]]:
    text = source.read_text(encoding="utf-8")
    for match in MARKDOWN_LINK_RE.finditer(text):
        href = match.group(1).strip()
        if href.startswith(("http://", "https://", "mailto:", "#")):
            continue
        if "#" not in href:
            continue
        raw_target, raw_anchor = href.split("#", 1)
        if not raw_target.endswith(".md") or not raw_anchor:
            continue
        target = (source.parent / raw_target).resolve()
        yield target, raw_anchor


def _validate_markdown_links(sources: Iterable[Path]) -> list[str]:
    failures: list[str] = []
    slug_cache: dict[Path, set[str]] = {}
    for source in sources:
        if not source.is_file():
            failures.append(f"{source}: source file is missing")
            continue
        for target, anchor in _iter_markdown_heading_links(source):
            if not target.is_file():
                failures.append(f"{source}: heading link target is missing: {target}")
                continue
            slugs = slug_cache.setdefault(target, _heading_slugs(target))
            if anchor not in slugs:
                failures.append(f"{source}: {target} has no heading anchor #{anchor}")
    return failures


def _validate_named_references(references: Iterable[HeadingReference]) -> list[str]:
    failures: list[str] = []
    heading_cache: dict[Path, list[str]] = {}
    for reference in references:
        if not reference.source.is_file():
            failures.append(f"{reference.source}: source file is missing")
            continue
        if not reference.target.is_file():
            failures.append(f"{reference.source}: heading reference target is missing: {reference.target}")
            continue
        headings = heading_cache.setdefault(reference.target, _heading_texts(reference.target))
        if not any(heading.startswith(reference.heading_prefix) for heading in headings):
            failures.append(
                f"{reference.source}: {reference.target} has no heading starting with {reference.heading_prefix!r}"
            )
    return failures


def check(paths: Iterable[Path]) -> list[str]:
    sources = tuple(path.resolve() for path in paths) or _default_sources()
    failures = _validate_markdown_links(sources)
    if not paths:
        failures.extend(_validate_named_references(_default_heading_references()))
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate markdown heading links in related documentation.")
    parser.add_argument("paths", nargs="*", type=Path, help="Markdown source files to scan; defaults to TODO/QW closure docs.")
    args = parser.parse_args(argv)

    failures = check(args.paths)
    if failures:
        print("Doc heading link validation failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print("doc heading links: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
