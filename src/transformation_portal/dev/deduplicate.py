"""Code deduplication engine for semantic duplicate detection.

This module provides tools for detecting and reporting duplicate
code at the semantic level, enabling:
- Codebase compression
- DRY enforcement
- Refactoring suggestions

The deduplication engine uses AST hashing to detect code that
has the same semantic structure, even if it differs syntactically.

Usage:
    from transformation_portal.dev.deduplicate import deduplicate_repo, DuplicationReport

    report = deduplicate_repo(Path("src"))
    print(f"Found {len(report.file_duplicates)} duplicate file groups")
    print(f"Found {len(report.function_duplicates)} duplicate function groups")

    for group in report.file_duplicates:
        print(f"Keep: {group.canonical}")
        print(f"Remove: {group.duplicates}")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from transformation_portal.dev.ast_index import ASTEquivalenceIndex, DuplicateGroup

logger = logging.getLogger(__name__)


@dataclass
class FunctionDuplicate:
    """A group of duplicate functions."""

    hash: str
    """The shared function hash."""

    locations: list[tuple[Path, str]]
    """List of (path, function_name) tuples."""

    canonical_path: Path
    """Path containing the canonical version."""

    canonical_name: str
    """Name of the canonical function."""


@dataclass
class ClassDuplicate:
    """A group of duplicate classes."""

    hash: str
    """The shared class hash."""

    locations: list[tuple[Path, str]]
    """List of (path, class_name) tuples."""

    canonical_path: Path
    """Path containing the canonical version."""

    canonical_name: str
    """Name of the canonical class."""


@dataclass
class DuplicationReport:
    """Report of all duplications found in a codebase."""

    root: Path
    """Root directory that was scanned."""

    total_files: int
    """Total number of files scanned."""

    unique_files: int
    """Number of files with unique content."""

    file_duplicates: list[DuplicateGroup]
    """Groups of duplicate files."""

    function_duplicates: list[FunctionDuplicate]
    """Groups of duplicate functions."""

    class_duplicates: list[ClassDuplicate]
    """Groups of duplicate classes."""

    bytes_duplicated: int = 0
    """Total bytes of duplicated code."""

    def to_dict(self) -> dict:
        """Convert report to a dictionary."""
        return {
            "root": str(self.root),
            "total_files": self.total_files,
            "unique_files": self.unique_files,
            "file_duplicates": [
                {"hash": g.hash[:16], "canonical": str(g.canonical), "duplicates": [str(p) for p in g.duplicates]}
                for g in self.file_duplicates
            ],
            "function_duplicates": [
                {
                    "hash": f.hash[:16],
                    "canonical": f"{f.canonical_path}::{f.canonical_name}",
                    "locations": [f"{p}::{n}" for p, n in f.locations],
                }
                for f in self.function_duplicates
            ],
            "class_duplicates": [
                {
                    "hash": c.hash[:16],
                    "canonical": f"{c.canonical_path}::{c.canonical_name}",
                    "locations": [f"{p}::{n}" for p, n in c.locations],
                }
                for c in self.class_duplicates
            ],
            "bytes_duplicated": self.bytes_duplicated,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        """Get a human-readable summary of the report."""
        lines = [
            f"Deduplication Report: {self.root}",
            f"=" * 50,
            f"Total files scanned: {self.total_files}",
            f"Unique files: {self.unique_files}",
            f"Duplicate file groups: {len(self.file_duplicates)}",
            f"Duplicate function groups: {len(self.function_duplicates)}",
            f"Duplicate class groups: {len(self.class_duplicates)}",
            f"Bytes duplicated: {self.bytes_duplicated:,}",
        ]

        if self.file_duplicates:
            lines.append("")
            lines.append("File Duplicates:")
            for group in self.file_duplicates[:10]:  # Limit output
                lines.append(f"  Hash: {group.hash[:16]}")
                lines.append(f"    Canonical: {group.canonical}")
                for dup in group.duplicates[:3]:
                    lines.append(f"    Duplicate: {dup}")
                if len(group.duplicates) > 3:
                    lines.append(f"    ... and {len(group.duplicates) - 3} more")

        if self.function_duplicates:
            lines.append("")
            lines.append("Function Duplicates:")
            for func in self.function_duplicates[:10]:
                lines.append(f"  Hash: {func.hash[:16]}")
                lines.append(f"    Canonical: {func.canonical_path}::{func.canonical_name}")
                for path, name in func.locations[:3]:
                    if path != func.canonical_path or name != func.canonical_name:
                        lines.append(f"    Duplicate: {path}::{name}")

        return "\n".join(lines)


def deduplicate_repo(
    root: Path,
    *,
    exclude_patterns: Optional[list[str]] = None,
    include_functions: bool = True,
    include_classes: bool = True,
) -> DuplicationReport:
    """Analyze a repository for semantic duplicates.

    This function scans a directory tree, computes AST hashes for
    all Python files, and identifies semantic duplicates at the
    file, function, and class levels.

    Args:
        root: Root directory to scan.
        exclude_patterns: Glob patterns to exclude (e.g., ["**/test_*.py"]).
        include_functions: If True, detect duplicate functions.
        include_classes: If True, detect duplicate classes.

    Returns:
        DuplicationReport with all found duplicates.

    Example:
        report = deduplicate_repo(Path("src"))
        print(report.summary())

        # Save report
        Path("duplicates.json").write_text(report.to_json())
    """
    exclude_patterns = exclude_patterns or []

    # Build index
    index = ASTEquivalenceIndex()
    index.build(root, exclude_patterns=exclude_patterns)

    # Get file duplicates
    file_duplicates = index.find_duplicates()

    # Get function duplicates
    function_duplicates = []
    if include_functions:
        for func_hash, locations in index.find_duplicate_functions().items():
            # Select canonical (prefer shorter paths)
            sorted_locs = sorted(locations, key=lambda loc: (len(str(loc[0])), str(loc[0]), loc[1]))
            canonical_path, canonical_name = sorted_locs[0]

            function_duplicates.append(
                FunctionDuplicate(
                    hash=func_hash,
                    locations=locations,
                    canonical_path=canonical_path,
                    canonical_name=canonical_name,
                )
            )

    # Get class duplicates
    class_duplicates = []
    if include_classes:
        for class_hash, locations in index.find_duplicate_classes().items():
            sorted_locs = sorted(locations, key=lambda loc: (len(str(loc[0])), str(loc[0]), loc[1]))
            canonical_path, canonical_name = sorted_locs[0]

            class_duplicates.append(
                ClassDuplicate(
                    hash=class_hash,
                    locations=locations,
                    canonical_path=canonical_path,
                    canonical_name=canonical_name,
                )
            )

    # Calculate bytes duplicated
    bytes_duplicated = 0
    for group in file_duplicates:
        for dup_path in group.duplicates:
            entry = index.entries.get(dup_path)
            if entry:
                bytes_duplicated += entry.size_bytes

    stats = index.get_stats()

    return DuplicationReport(
        root=root,
        total_files=stats["total_files"],
        unique_files=stats["unique_hashes"],
        file_duplicates=file_duplicates,
        function_duplicates=function_duplicates,
        class_duplicates=class_duplicates,
        bytes_duplicated=bytes_duplicated,
    )


def suggest_refactoring(report: DuplicationReport) -> list[str]:
    """Generate refactoring suggestions based on duplication report.

    Args:
        report: Deduplication report to analyze.

    Returns:
        List of refactoring suggestions as strings.
    """
    suggestions = []

    # File-level suggestions
    for group in report.file_duplicates:
        if len(group.duplicates) >= 2:
            suggestions.append(
                f"Consider consolidating {len(group.duplicates) + 1} identical files. "
                f"Keep '{group.canonical}' and import from there."
            )

    # Function-level suggestions
    for func in report.function_duplicates:
        if len(func.locations) >= 3:
            suggestions.append(
                f"Function '{func.canonical_name}' is duplicated in {len(func.locations)} locations. "
                f"Consider extracting to a shared module."
            )

    # Class-level suggestions
    for cls in report.class_duplicates:
        if len(cls.locations) >= 2:
            suggestions.append(
                f"Class '{cls.canonical_name}' is duplicated in {len(cls.locations)} locations. "
                f"Consider using inheritance or composition."
            )

    return suggestions
