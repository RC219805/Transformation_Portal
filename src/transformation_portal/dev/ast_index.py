"""AST equivalence index for tracking semantic duplicates.

This module provides an index that tracks semantic equivalence
across a codebase, enabling:
- Duplicate logic detection
- Code reuse suggestions
- DRY enforcement at the semantic level

The index maps AST hashes to file paths, allowing detection of
files (or functions/classes) that have the same semantic structure.

Usage:
    from transformation_portal.dev.ast_index import ASTEquivalenceIndex

    index = ASTEquivalenceIndex()
    index.build(Path("src"))

    duplicates = index.find_duplicates()
    for hash, files in duplicates.items():
        print(f"Duplicate code in: {files}")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from transformation_portal.dev.ast_hash import (
    ASTHashResult,
    compute_ast_hash_safe,
    compute_class_hash,
    compute_function_hash,
)

logger = logging.getLogger(__name__)


@dataclass
class IndexEntry:
    """Entry in the AST equivalence index."""

    path: Path
    """Path to the source file."""

    hash: str
    """AST hash of the file."""

    node_count: int
    """Number of AST nodes."""

    size_bytes: int
    """File size in bytes."""

    functions: dict[str, str] = field(default_factory=dict)
    """Map of function names to their hashes."""

    classes: dict[str, str] = field(default_factory=dict)
    """Map of class names to their hashes."""


@dataclass
class DuplicateGroup:
    """A group of files with identical semantic structure."""

    hash: str
    """The shared AST hash."""

    files: list[Path]
    """Files with this hash."""

    canonical: Path
    """The canonical (preferred) file."""

    duplicates: list[Path]
    """The duplicate files (excluding canonical)."""


class ASTEquivalenceIndex:
    """Index for tracking semantic equivalence across a codebase.

    This index computes AST hashes for all Python files and tracks
    which files have the same semantic structure.

    Example:
        index = ASTEquivalenceIndex()
        index.build(Path("src"))

        # Find all duplicates
        for group in index.find_duplicates():
            print(f"Keep: {group.canonical}")
            print(f"Remove: {group.duplicates}")

        # Check if a file has duplicates
        dups = index.get_duplicates_of(Path("src/foo.py"))
    """

    def __init__(self):
        self.entries: dict[Path, IndexEntry] = {}
        self.hash_index: dict[str, list[Path]] = {}
        self.function_index: dict[str, list[tuple[Path, str]]] = {}
        self.class_index: dict[str, list[tuple[Path, str]]] = {}

    def add_file(self, path: Path) -> Optional[IndexEntry]:
        """Add a file to the index.

        Args:
            path: Path to the Python file.

        Returns:
            IndexEntry for the file, or None if it couldn't be indexed.
        """
        if not path.exists() or path.suffix != ".py":
            return None

        try:
            source = path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("Could not read %s: %s", path, e)
            return None

        result = compute_ast_hash_safe(source)

        if not result.is_valid:
            logger.debug("Skipping invalid file %s: %s", path, result.error)
            return None

        # Create entry
        entry = IndexEntry(
            path=path,
            hash=result.hash,
            node_count=result.node_count,
            size_bytes=len(source.encode("utf-8")),
        )

        # Index functions and classes
        entry.functions = self._extract_function_hashes(source)
        entry.classes = self._extract_class_hashes(source)

        # Store in indices
        self.entries[path] = entry
        self.hash_index.setdefault(result.hash, []).append(path)

        # Index individual functions
        for func_name, func_hash in entry.functions.items():
            self.function_index.setdefault(func_hash, []).append((path, func_name))

        # Index individual classes
        for class_name, class_hash in entry.classes.items():
            self.class_index.setdefault(class_hash, []).append((path, class_name))

        return entry

    def _extract_function_hashes(self, source: str) -> dict[str, str]:
        """Extract hashes for all functions in the source."""
        import ast

        result = {}
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_hash = compute_function_hash(source, node.name)
                    if func_hash:
                        result[node.name] = func_hash
        except Exception:
            pass
        return result

    def _extract_class_hashes(self, source: str) -> dict[str, str]:
        """Extract hashes for all classes in the source."""
        import ast

        result = {}
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_hash = compute_class_hash(source, node.name)
                    if class_hash:
                        result[node.name] = class_hash
        except Exception:
            pass
        return result

    def build(self, root: Path, *, exclude_patterns: Optional[list[str]] = None) -> int:
        """Build the index from a directory.

        Args:
            root: Root directory to scan.
            exclude_patterns: Glob patterns to exclude.

        Returns:
            Number of files indexed.
        """
        exclude_patterns = exclude_patterns or []
        count = 0

        for path in root.rglob("*.py"):
            # Skip excluded patterns
            skip = False
            for pattern in exclude_patterns:
                if path.match(pattern):
                    skip = True
                    break
            if skip:
                continue

            if self.add_file(path):
                count += 1

        logger.info("Indexed %d files from %s", count, root)
        return count

    def find_duplicates(self) -> list[DuplicateGroup]:
        """Find all groups of duplicate files.

        Returns:
            List of DuplicateGroup objects for files with identical AST.
        """
        groups = []

        for hash_val, files in self.hash_index.items():
            if len(files) > 1:
                # Select canonical file (prefer shorter paths, then alphabetical)
                sorted_files = sorted(files, key=lambda p: (len(str(p)), str(p)))
                canonical = sorted_files[0]
                duplicates = sorted_files[1:]

                groups.append(
                    DuplicateGroup(
                        hash=hash_val,
                        files=files.copy(),
                        canonical=canonical,
                        duplicates=duplicates,
                    )
                )

        return groups

    def find_duplicate_functions(self) -> dict[str, list[tuple[Path, str]]]:
        """Find all groups of duplicate functions.

        Returns:
            Dict mapping function hash to list of (path, function_name) tuples.
        """
        return {h: locs for h, locs in self.function_index.items() if len(locs) > 1}

    def find_duplicate_classes(self) -> dict[str, list[tuple[Path, str]]]:
        """Find all groups of duplicate classes.

        Returns:
            Dict mapping class hash to list of (path, class_name) tuples.
        """
        return {h: locs for h, locs in self.class_index.items() if len(locs) > 1}

    def get_duplicates_of(self, path: Path) -> list[Path]:
        """Get all duplicates of a specific file.

        Args:
            path: Path to check.

        Returns:
            List of paths with the same AST hash (excluding the input).
        """
        entry = self.entries.get(path)
        if not entry:
            return []

        all_with_hash = self.hash_index.get(entry.hash, [])
        return [p for p in all_with_hash if p != path]

    def get_stats(self) -> dict:
        """Get statistics about the index.

        Returns:
            Dict with index statistics.
        """
        duplicate_groups = self.find_duplicates()
        duplicate_functions = self.find_duplicate_functions()
        duplicate_classes = self.find_duplicate_classes()

        return {
            "total_files": len(self.entries),
            "unique_hashes": len(self.hash_index),
            "duplicate_file_groups": len(duplicate_groups),
            "duplicate_files": sum(len(g.duplicates) for g in duplicate_groups),
            "duplicate_function_groups": len(duplicate_functions),
            "duplicate_class_groups": len(duplicate_classes),
            "total_functions": sum(len(e.functions) for e in self.entries.values()),
            "total_classes": sum(len(e.classes) for e in self.entries.values()),
        }

    def save(self, path: Path) -> None:
        """Save the index to a JSON file.

        Args:
            path: Path to save the index.
        """
        data = {
            "entries": {
                str(p): {
                    "hash": e.hash,
                    "node_count": e.node_count,
                    "size_bytes": e.size_bytes,
                    "functions": e.functions,
                    "classes": e.classes,
                }
                for p, e in self.entries.items()
            },
        }

        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        """Load the index from a JSON file.

        Args:
            path: Path to load the index from.
        """
        data = json.loads(path.read_text(encoding="utf-8"))

        self.entries = {}
        self.hash_index = {}
        self.function_index = {}
        self.class_index = {}

        for path_str, entry_data in data["entries"].items():
            file_path = Path(path_str)
            entry = IndexEntry(
                path=file_path,
                hash=entry_data["hash"],
                node_count=entry_data["node_count"],
                size_bytes=entry_data["size_bytes"],
                functions=entry_data.get("functions", {}),
                classes=entry_data.get("classes", {}),
            )

            self.entries[file_path] = entry
            self.hash_index.setdefault(entry.hash, []).append(file_path)

            for func_name, func_hash in entry.functions.items():
                self.function_index.setdefault(func_hash, []).append((file_path, func_name))

            for class_name, class_hash in entry.classes.items():
                self.class_index.setdefault(class_hash, []).append((file_path, class_name))
