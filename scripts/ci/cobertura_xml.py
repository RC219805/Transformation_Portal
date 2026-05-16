"""Shared Cobertura XML parsing helpers for CI coverage tooling."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_CONDITION_COVERAGE_RE = re.compile(r"\((?P<covered>\d+)\s*/\s*(?P<valid>\d+)\)")


@dataclass(frozen=True)
class CoberturaLine:
    """Line coverage and branch coverage facts for one source line."""

    number: int
    hits: int
    branch_covered: int = 0
    branch_valid: int = 0
    missing_branches: tuple[str, ...] = ()

    @property
    def covered(self) -> bool:
        return self.hits > 0

    @property
    def missed_branch_count(self) -> int:
        return max(self.branch_valid - self.branch_covered, 0)


@dataclass(frozen=True)
class CoberturaFile:
    """Resolved repo-relative coverage facts for one source file."""

    filename: str
    lines: tuple[CoberturaLine, ...]

    @property
    def valid_lines(self) -> int:
        return len(self.lines)

    @property
    def covered_lines(self) -> int:
        return sum(1 for line in self.lines if line.covered)

    @property
    def line_percentage(self) -> float:
        if self.valid_lines == 0:
            return 0.0
        return 100.0 * self.covered_lines / self.valid_lines

    @property
    def branch_valid(self) -> int:
        return sum(line.branch_valid for line in self.lines)

    @property
    def branch_covered(self) -> int:
        return sum(line.branch_covered for line in self.lines)

    @property
    def branch_percentage(self) -> float:
        if self.branch_valid == 0:
            return 0.0
        return 100.0 * self.branch_covered / self.branch_valid

    @property
    def missed_branch_count(self) -> int:
        return sum(line.missed_branch_count for line in self.lines)

    @property
    def missed_line_ranges(self) -> tuple[str, ...]:
        missed = [line.number for line in self.lines if not line.covered]
        return compress_ranges(missed)


def _normalize_filename(raw: str) -> str:
    """Normalize a Cobertura ``filename`` to a repo-relative ``src/...`` form."""
    norm = raw.replace("\\", "/")
    if norm.startswith("./"):
        norm = norm[2:]
    if norm.startswith("/"):
        marker = "/src/"
        idx = norm.rfind(marker)
        if idx != -1:
            norm = norm[idx + 1 :]
    return norm


def _source_relative_prefix(source_text: str) -> str | None:
    """Convert an absolute Cobertura ``<source>`` path to its ``src/...`` form."""
    norm = source_text.replace("\\", "/").rstrip("/")
    marker = "/src/"
    idx = norm.rfind(marker)
    if idx != -1:
        return norm[idx + 1 :]
    if norm.endswith("/src"):
        return "src"
    if norm == "src":
        return "src"
    return None


def _resolve_class_path(
    class_filename: str,
    source_roots: Iterable[Path],
    source_prefixes: Iterable[str],
) -> str | None:
    """Resolve a Cobertura class filename to its canonical repo-relative path."""
    norm = _normalize_filename(class_filename)
    source_pairs = list(zip(source_roots, source_prefixes))
    for root, prefix in source_pairs:
        if (root / norm).is_file():
            return f"{prefix.rstrip('/')}/{norm}"
    for _, prefix in source_pairs:
        repo_relative = f"{prefix.rstrip('/')}/{norm}"
        if Path(repo_relative).is_file():
            return repo_relative
    return norm if norm else None


def _matches_prefix(filename: str, prefix: str) -> bool:
    """Return True if filename is matched by prefix, with a src-stripped fallback."""
    if filename.startswith(prefix):
        return True
    stripped = prefix.removeprefix("src/")
    return bool(stripped) and filename.startswith(stripped)


def _iter_class_elements(root: ET.Element) -> Iterable[ET.Element]:
    for cls in root.iter("class"):
        yield cls


def _collect_sources(root: ET.Element) -> tuple[list[Path], list[str]]:
    """Read ``<sources>`` and return absolute roots plus src-relative prefixes."""
    roots: list[Path] = []
    prefixes: list[str] = []
    seen: set[str] = set()
    for src in root.findall("sources/source"):
        if not src.text:
            continue
        prefix = _source_relative_prefix(src.text)
        if not prefix or prefix in seen:
            continue
        seen.add(prefix)
        roots.append(Path(src.text))
        prefixes.append(prefix)
    return roots, prefixes


def _parse_int(value: str | None, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _parse_branch_counts(line: ET.Element) -> tuple[int, int]:
    condition_coverage = line.attrib.get("condition-coverage", "")
    match = _CONDITION_COVERAGE_RE.search(condition_coverage)
    if match:
        return _parse_int(match.group("covered")), _parse_int(match.group("valid"))

    conditions = line.find("conditions")
    if conditions is None:
        return 0, 0

    covered = 0
    valid = 0
    for condition in conditions.findall("condition"):
        valid += 1
        coverage = condition.attrib.get("coverage", "").rstrip("%")
        try:
            if float(coverage) > 0.0:
                covered += 1
        except ValueError:
            continue
    return covered, valid


def _parse_missing_branches(line: ET.Element) -> tuple[str, ...]:
    raw = line.attrib.get("missing-branches", "")
    if not raw:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _parse_lines(cls: ET.Element) -> tuple[CoberturaLine, ...]:
    lines = cls.find("lines")
    if lines is None:
        return ()

    parsed: list[CoberturaLine] = []
    for line in lines.findall("line"):
        branch_covered, branch_valid = _parse_branch_counts(line)
        parsed.append(
            CoberturaLine(
                number=_parse_int(line.attrib.get("number")),
                hits=_parse_int(line.attrib.get("hits")),
                branch_covered=branch_covered,
                branch_valid=branch_valid,
                missing_branches=_parse_missing_branches(line),
            )
        )
    return tuple(parsed)


def load_cobertura_files(coverage_xml: Path) -> tuple[CoberturaFile, ...]:
    """Load Cobertura classes resolved to repo-relative file paths."""
    tree = ET.parse(coverage_xml)
    root = tree.getroot()
    source_roots, source_prefixes = _collect_sources(root)

    files: list[CoberturaFile] = []
    for cls in _iter_class_elements(root):
        filename = _resolve_class_path(cls.attrib.get("filename", ""), source_roots, source_prefixes)
        if filename is None:
            continue
        files.append(CoberturaFile(filename=filename, lines=_parse_lines(cls)))
    return tuple(files)


def compress_ranges(numbers: Iterable[int]) -> tuple[str, ...]:
    """Compress sorted or unsorted positive line numbers into display ranges."""
    sorted_numbers = sorted({number for number in numbers if number > 0})
    if not sorted_numbers:
        return ()

    ranges: list[str] = []
    start = previous = sorted_numbers[0]
    for number in sorted_numbers[1:]:
        if number == previous + 1:
            previous = number
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = number
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return tuple(ranges)
