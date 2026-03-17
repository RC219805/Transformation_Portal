"""Auto-refactoring engine for semantic code deduplication.

This module provides a closed-loop code optimizer that:
- Detects duplicate semantic code via AST hashing
- Selects canonical implementations
- Extracts to shared modules
- Rewrites imports and call sites
- Enforces semantic reuse

Pipeline:
    SCAN → HASH → GROUP → SELECT CANONICAL → EXTRACT → REWRITE → VALIDATE

This is effectively a compiler pass + linker for the repository,
ensuring minimal representation and enforced reuse.

Usage:
    from transformation_portal.dev.refactor_engine import AutoRefactorEngine

    engine = AutoRefactorEngine(
        root=Path("src"),
        shared_module=Path("src/transformation_portal/shared"),
    )

    plan = engine.build_plan()
    print(f"Found {len(plan.duplicates)} duplicate groups")

    # Execute refactoring (with dry_run for safety)
    result = engine.execute(plan, dry_run=True)
"""

from __future__ import annotations

import ast
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from transformation_portal.dev.ast_hash import compute_ast_hash, compute_function_hash
from transformation_portal.dev.ast_index import ASTEquivalenceIndex
from transformation_portal.dev.ast_normalize import canonicalize_code

logger = logging.getLogger(__name__)


@dataclass
class SymbolInfo:
    """Information about an extracted symbol."""

    name: str
    """Symbol name (function or class name)."""

    kind: str
    """Symbol kind: 'function', 'async_function', or 'class'."""

    hash: str
    """AST hash of the symbol."""

    source: str
    """Canonical source code of the symbol."""


@dataclass
class RefactorPlan:
    """Plan for refactoring duplicate code."""

    canonical_map: dict[str, Path] = field(default_factory=dict)
    """Map from AST hash to canonical file path."""

    duplicates: dict[str, list[Path]] = field(default_factory=dict)
    """Map from AST hash to list of duplicate file paths."""

    symbols: dict[str, list[SymbolInfo]] = field(default_factory=dict)
    """Map from AST hash to extracted symbols."""

    affected_files: set[Path] = field(default_factory=set)
    """Set of files that will be modified."""

    def summary(self) -> str:
        """Get a summary of the refactoring plan."""
        lines = [
            "Refactoring Plan",
            "=" * 50,
            f"Duplicate groups: {len(self.duplicates)}",
            f"Files to refactor: {len(self.affected_files)}",
            f"Symbols to extract: {sum(len(s) for s in self.symbols.values())}",
        ]

        if self.duplicates:
            lines.append("")
            lines.append("Duplicate Groups:")
            for hash_val, files in list(self.duplicates.items())[:5]:
                canonical = self.canonical_map.get(hash_val)
                lines.append(f"  Hash: {hash_val[:12]}")
                lines.append(f"    Canonical: {canonical}")
                for f in files[:3]:
                    lines.append(f"    Duplicate: {f}")
                if len(files) > 3:
                    lines.append(f"    ... and {len(files) - 3} more")

        return "\n".join(lines)


@dataclass
class RefactorResult:
    """Result of executing a refactoring plan."""

    success: bool
    """Whether refactoring completed successfully."""

    files_created: list[Path] = field(default_factory=list)
    """New files created in shared module."""

    files_modified: list[Path] = field(default_factory=list)
    """Files modified with new imports."""

    files_deleted: list[Path] = field(default_factory=list)
    """Files deleted (replaced with imports)."""

    errors: list[str] = field(default_factory=list)
    """Error messages for any failures."""

    dry_run: bool = False
    """Whether this was a dry run."""

    def summary(self) -> str:
        """Get a summary of the refactoring result."""
        status = "DRY RUN" if self.dry_run else ("SUCCESS" if self.success else "FAILED")
        lines = [
            f"Refactoring Result: {status}",
            "=" * 50,
            f"Files created: {len(self.files_created)}",
            f"Files modified: {len(self.files_modified)}",
            f"Files deleted: {len(self.files_deleted)}",
        ]

        if self.errors:
            lines.append(f"Errors: {len(self.errors)}")
            for err in self.errors[:5]:
                lines.append(f"  - {err}")

        return "\n".join(lines)


class AutoRefactorEngine:
    """Semantic deduplication and refactoring engine.

    This engine implements a closed-loop optimizer that:
    1. Scans codebase for semantic duplicates
    2. Selects canonical implementations
    3. Extracts to shared modules
    4. Rewrites duplicates to import from shared
    5. Validates semantic equivalence

    Example:
        engine = AutoRefactorEngine(
            root=Path("src"),
            shared_module=Path("src/mypackage/shared"),
        )

        # Build plan (no modifications)
        plan = engine.build_plan()

        # Execute with dry run first
        result = engine.execute(plan, dry_run=True)

        # Execute for real
        result = engine.execute(plan, dry_run=False)
    """

    def __init__(
        self,
        root: Path,
        shared_module: Path,
        *,
        exclude_patterns: Optional[list[str]] = None,
        min_symbols: int = 1,
        package_name: Optional[str] = None,
    ):
        """Initialize the refactoring engine.

        Args:
            root: Root directory to scan for duplicates.
            shared_module: Directory for extracted shared modules.
            exclude_patterns: Glob patterns to exclude from scanning.
            min_symbols: Minimum number of symbols to consider a file for extraction.
            package_name: Package name for imports (auto-detected if not provided).
        """
        self.root = root
        self.shared_module = shared_module
        self.exclude_patterns = exclude_patterns or [
            "**/test_*.py",
            "**/*_test.py",
            "**/conftest.py",
            "**/__pycache__/**",
        ]
        self.min_symbols = min_symbols
        self.package_name = package_name or self._detect_package_name()

    def _detect_package_name(self) -> str:
        """Detect the package name from the shared module path."""
        # Convert path to module notation
        parts = self.shared_module.relative_to(self.root).parts
        return ".".join(parts)

    # --------------------------------------------------
    # Phase 1: Build refactoring plan
    # --------------------------------------------------
    def build_plan(self) -> RefactorPlan:
        """Build a refactoring plan by scanning for duplicates.

        Returns:
            RefactorPlan with canonical selections and duplicate mappings.
        """
        index = ASTEquivalenceIndex()
        index.build(self.root, exclude_patterns=self.exclude_patterns)

        plan = RefactorPlan()

        # Process file-level duplicates
        for group in index.find_duplicates():
            hash_val = group.hash
            canonical = self._select_canonical(group.files)

            plan.canonical_map[hash_val] = canonical
            plan.duplicates[hash_val] = [f for f in group.files if f != canonical]
            plan.affected_files.update(group.files)

            # Extract symbols from canonical file
            symbols = self._extract_symbols(canonical)
            if symbols:
                plan.symbols[hash_val] = symbols

        logger.info("Built refactoring plan: %d duplicate groups", len(plan.duplicates))
        return plan

    def _select_canonical(self, files: list[Path]) -> Path:
        """Select the canonical file from a group of duplicates.

        Selection criteria (in order):
        1. Shortest path (prefer less nested)
        2. Lexicographic order (deterministic)

        Args:
            files: List of duplicate file paths.

        Returns:
            The selected canonical file path.
        """
        return sorted(files, key=lambda p: (len(str(p)), str(p)))[0]

    def _extract_symbols(self, file: Path) -> list[SymbolInfo]:
        """Extract symbols (functions, classes) from a file.

        Args:
            file: Path to the Python file.

        Returns:
            List of SymbolInfo for each top-level symbol.
        """
        try:
            source = file.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except Exception as e:
            logger.warning("Could not parse %s: %s", file, e)
            return []

        symbols = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                # Extract function source
                func_source = ast.unparse(node)
                func_hash = compute_ast_hash(func_source)

                symbols.append(
                    SymbolInfo(
                        name=node.name,
                        kind="function",
                        hash=func_hash,
                        source=func_source,
                    )
                )

            elif isinstance(node, ast.AsyncFunctionDef):
                func_source = ast.unparse(node)
                func_hash = compute_ast_hash(func_source)

                symbols.append(
                    SymbolInfo(
                        name=node.name,
                        kind="async_function",
                        hash=func_hash,
                        source=func_source,
                    )
                )

            elif isinstance(node, ast.ClassDef):
                class_source = ast.unparse(node)
                class_hash = compute_ast_hash(class_source)

                symbols.append(
                    SymbolInfo(
                        name=node.name,
                        kind="class",
                        hash=class_hash,
                        source=class_source,
                    )
                )

        return symbols

    # --------------------------------------------------
    # Phase 2: Execute refactoring
    # --------------------------------------------------
    def execute(self, plan: RefactorPlan, *, dry_run: bool = True) -> RefactorResult:
        """Execute the refactoring plan.

        Args:
            plan: The refactoring plan to execute.
            dry_run: If True, don't actually modify files.

        Returns:
            RefactorResult with details of changes made.
        """
        result = RefactorResult(success=True, dry_run=dry_run)

        if not plan.duplicates:
            logger.info("No duplicates to refactor")
            return result

        try:
            # Phase 2a: Create shared module structure
            self._materialize_shared_module(plan, result, dry_run)

            # Phase 2b: Rewrite duplicate files
            self._rewrite_duplicates(plan, result, dry_run)

            # Phase 2c: Validate (verify hashes preserved)
            if not dry_run:
                self._validate_refactoring(plan, result)

        except Exception as e:
            logger.error("Refactoring failed: %s", e)
            result.success = False
            result.errors.append(str(e))

        return result

    def _materialize_shared_module(self, plan: RefactorPlan, result: RefactorResult, dry_run: bool) -> None:
        """Create the shared module with canonical implementations.

        Args:
            plan: Refactoring plan with canonical mappings.
            result: Result object to update.
            dry_run: If True, don't create files.
        """
        if not dry_run:
            self.shared_module.mkdir(parents=True, exist_ok=True)

            # Create __init__.py
            init_path = self.shared_module / "__init__.py"
            if not init_path.exists():
                init_content = '"""Shared canonical implementations (auto-generated)."""\n'
                init_path.write_text(init_content, encoding="utf-8")
                result.files_created.append(init_path)

        for hash_val, canonical_file in plan.canonical_map.items():
            module_name = f"_{hash_val[:12]}"
            target = self.shared_module / f"{module_name}.py"

            if dry_run:
                logger.info("Would create: %s (from %s)", target, canonical_file)
                result.files_created.append(target)
            else:
                try:
                    source = canonical_file.read_text(encoding="utf-8")
                    canonical_code = canonicalize_code(source)

                    # Add header comment
                    header = f'"""Canonical implementation (hash: {hash_val[:16]}).\n\nSource: {canonical_file}\n"""\n\n'
                    target.write_text(header + canonical_code, encoding="utf-8")

                    result.files_created.append(target)
                    logger.info("Created: %s", target)

                except Exception as e:
                    result.errors.append(f"Failed to create {target}: {e}")

    def _rewrite_duplicates(self, plan: RefactorPlan, result: RefactorResult, dry_run: bool) -> None:
        """Rewrite duplicate files to import from shared module.

        Args:
            plan: Refactoring plan with duplicate mappings.
            result: Result object to update.
            dry_run: If True, don't modify files.
        """
        for hash_val, files in plan.duplicates.items():
            module_name = f"_{hash_val[:12]}"

            # Get symbols to re-export
            symbols = plan.symbols.get(hash_val, [])
            symbol_names = [s.name for s in symbols]

            for file in files:
                if dry_run:
                    logger.info("Would rewrite: %s → import from %s", file, module_name)
                    result.files_modified.append(file)
                else:
                    try:
                        self._rewrite_file_to_import(file, module_name, symbol_names)
                        result.files_modified.append(file)
                        logger.info("Rewrote: %s", file)
                    except Exception as e:
                        result.errors.append(f"Failed to rewrite {file}: {e}")

    def _rewrite_file_to_import(self, path: Path, module_name: str, symbol_names: list[str]) -> None:
        """Rewrite a file to import from the shared module.

        This replaces the file content with imports that re-export
        the canonical symbols, preserving the public API.

        Args:
            path: Path to the file to rewrite.
            module_name: Name of the shared module.
            symbol_names: Names of symbols to re-export.
        """
        lines = [
            f'"""Re-export from canonical shared module.',
            f"",
            f"This file has been refactored to use the canonical implementation",
            f"from {self.package_name}.{module_name}",
            f'"""',
            f"",
            f"from {self.package_name}.{module_name} import *  # noqa: F401,F403",
        ]

        # Add explicit re-exports for better IDE support
        if symbol_names:
            lines.append("")
            lines.append("# Explicit re-exports for IDE support")
            lines.append("__all__ = [")
            for name in symbol_names:
                lines.append(f'    "{name}",')
            lines.append("]")

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _validate_refactoring(self, plan: RefactorPlan, result: RefactorResult) -> None:
        """Validate that refactoring preserved semantics.

        This checks that the shared module files have the expected
        AST hashes after canonicalization.

        Args:
            plan: The refactoring plan.
            result: Result object to update with validation errors.
        """
        for hash_val, canonical_file in plan.canonical_map.items():
            module_name = f"_{hash_val[:12]}"
            shared_file = self.shared_module / f"{module_name}.py"

            if not shared_file.exists():
                result.errors.append(f"Shared module not found: {shared_file}")
                continue

            try:
                # Compute hash of shared module (excluding header comment)
                source = shared_file.read_text(encoding="utf-8")

                # Remove docstring for hash comparison
                tree = ast.parse(source)
                if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
                    tree.body = tree.body[1:]

                shared_source = ast.unparse(tree)
                shared_hash = compute_ast_hash(shared_source)

                if shared_hash != hash_val:
                    result.errors.append(f"Hash mismatch for {shared_file}: expected {hash_val[:12]}, got {shared_hash[:12]}")

            except Exception as e:
                result.errors.append(f"Validation failed for {shared_file}: {e}")

    # --------------------------------------------------
    # Utility methods
    # --------------------------------------------------
    def rollback(self, result: RefactorResult) -> None:
        """Rollback changes made by a refactoring execution.

        Args:
            result: The RefactorResult from a previous execution.
        """
        if result.dry_run:
            logger.info("Nothing to rollback (was dry run)")
            return

        # Delete created files
        for path in result.files_created:
            if path.exists():
                path.unlink()
                logger.info("Deleted: %s", path)

        # Note: Cannot restore modified files without backup
        if result.files_modified:
            logger.warning("Cannot restore %d modified files (no backup)", len(result.files_modified))

    def preview(self, plan: RefactorPlan) -> str:
        """Generate a preview of what the refactoring will do.

        Args:
            plan: The refactoring plan.

        Returns:
            Human-readable preview string.
        """
        lines = [plan.summary(), "", "Actions:"]

        for hash_val, canonical_file in plan.canonical_map.items():
            module_name = f"_{hash_val[:12]}"
            lines.append(f"  CREATE: {self.shared_module / module_name}.py")
            lines.append(f"    Source: {canonical_file}")

            for dup in plan.duplicates.get(hash_val, []):
                lines.append(f"  REWRITE: {dup}")

        return "\n".join(lines)


class IncrementalRefactor:
    """Incremental refactoring for large codebases.

    This class provides methods for refactoring in smaller batches,
    useful for very large codebases or gradual migration.
    """

    def __init__(self, engine: AutoRefactorEngine):
        self.engine = engine

    def refactor_by_hash(self, plan: RefactorPlan, target_hash: str, *, dry_run: bool = True) -> RefactorResult:
        """Refactor only files matching a specific hash.

        Args:
            plan: Full refactoring plan.
            target_hash: The AST hash to refactor.
            dry_run: If True, don't modify files.

        Returns:
            RefactorResult for this specific hash.
        """
        # Create a filtered plan
        filtered_plan = RefactorPlan()

        if target_hash in plan.canonical_map:
            filtered_plan.canonical_map[target_hash] = plan.canonical_map[target_hash]
            filtered_plan.duplicates[target_hash] = plan.duplicates.get(target_hash, [])
            filtered_plan.symbols[target_hash] = plan.symbols.get(target_hash, [])
            filtered_plan.affected_files = set(filtered_plan.duplicates[target_hash])
            filtered_plan.affected_files.add(filtered_plan.canonical_map[target_hash])

        return self.engine.execute(filtered_plan, dry_run=dry_run)

    def refactor_batch(self, plan: RefactorPlan, batch_size: int = 5, *, dry_run: bool = True) -> list[RefactorResult]:
        """Refactor in batches of a specified size.

        Args:
            plan: Full refactoring plan.
            batch_size: Number of duplicate groups per batch.
            dry_run: If True, don't modify files.

        Returns:
            List of RefactorResult for each batch.
        """
        results = []
        hashes = list(plan.duplicates.keys())

        for i in range(0, len(hashes), batch_size):
            batch_hashes = hashes[i : i + batch_size]

            # Create batch plan
            batch_plan = RefactorPlan()
            for h in batch_hashes:
                batch_plan.canonical_map[h] = plan.canonical_map[h]
                batch_plan.duplicates[h] = plan.duplicates[h]
                batch_plan.symbols[h] = plan.symbols.get(h, [])
                batch_plan.affected_files.update(batch_plan.duplicates[h])
                batch_plan.affected_files.add(batch_plan.canonical_map[h])

            result = self.engine.execute(batch_plan, dry_run=dry_run)
            results.append(result)

            if not result.success:
                logger.warning("Batch %d failed, stopping", i // batch_size)
                break

        return results
