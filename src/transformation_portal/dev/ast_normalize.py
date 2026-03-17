"""AST normalization utilities for canonical code emission.

This module provides the high-level API for normalizing Python source code
through AST canonicalization followed by deterministic code emission.

Pipeline:
    SOURCE CODE → AST PARSE → CANONICAL TRANSFORM → AST UNPARSE → CANONICAL CODE

Usage:
    from transformation_portal.dev.ast_normalize import canonicalize_code

    normalized = canonicalize_code('''
    x = {"b": 2, "a": 1}
    foo(z=3, a=1, b=2)
    ''')

    # Result:
    # x = {"a": 1, "b": 2}
    # foo(a=1, b=2, z=3)
"""

from __future__ import annotations

import ast
import hashlib
import logging
from dataclasses import dataclass
from typing import Optional

from transformation_portal.dev.ast_canonicalizer import Canonicalizer, canonicalize_ast

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CanonicalResult:
    """Result of code canonicalization."""

    source: str
    """Original source code."""

    canonical: str
    """Canonicalized source code."""

    transformations: tuple[str, ...]
    """List of transformations that were applied."""

    ast_hash: str
    """SHA256 hash of the canonical AST structure."""

    is_valid: bool
    """Whether the source was valid Python."""

    error: Optional[str] = None
    """Error message if parsing failed."""


def canonicalize_code(source: str, *, aggressive: bool = False) -> str:
    """Parse source code, normalize AST, and emit canonical code.

    This is the primary API for code canonicalization. It:
    1. Parses the source code into an AST
    2. Applies canonical transformations (dict sorting, kwarg sorting, etc.)
    3. Emits deterministic code via ast.unparse()

    Args:
        source: Python source code to canonicalize.
        aggressive: If True, apply more aggressive normalizations.

    Returns:
        Canonicalized Python source code.

    Raises:
        SyntaxError: If the source code has invalid syntax.

    Example:
        >>> canonicalize_code('x = {"b": 2, "a": 1}')
        'x = {"a": 1, "b": 2}'

        >>> canonicalize_code('foo(z=3, a=1)')
        'foo(a=1, z=3)'
    """
    tree = ast.parse(source)
    tree, _ = canonicalize_ast(tree, aggressive=aggressive)
    return ast.unparse(tree)


def canonicalize_code_safe(source: str, *, aggressive: bool = False) -> CanonicalResult:
    """Safely canonicalize code, returning a result object.

    Unlike canonicalize_code(), this function never raises exceptions.
    Instead, it returns a CanonicalResult with error information.

    Args:
        source: Python source code to canonicalize.
        aggressive: If True, apply more aggressive normalizations.

    Returns:
        CanonicalResult with canonical code or error information.
    """
    try:
        tree = ast.parse(source)
        tree, transformations = canonicalize_ast(tree, aggressive=aggressive)
        canonical = ast.unparse(tree)

        # Compute AST hash for content-addressable identification
        ast_hash = compute_ast_hash(tree)

        return CanonicalResult(
            source=source,
            canonical=canonical,
            transformations=tuple(transformations),
            ast_hash=ast_hash,
            is_valid=True,
        )

    except SyntaxError as e:
        return CanonicalResult(
            source=source,
            canonical=source,  # Return original on error
            transformations=(),
            ast_hash="",
            is_valid=False,
            error=f"Syntax error: {e}",
        )

    except Exception as e:
        logger.warning("Unexpected error during canonicalization: %s", e)
        return CanonicalResult(
            source=source,
            canonical=source,
            transformations=(),
            ast_hash="",
            is_valid=False,
            error=f"Unexpected error: {e}",
        )


def compute_ast_hash(tree: ast.AST) -> str:
    """Compute a content-addressable hash of an AST.

    This hash represents the semantic structure of the code,
    ignoring formatting differences. Two pieces of code that
    produce the same AST hash are semantically equivalent.

    Args:
        tree: The AST to hash.

    Returns:
        SHA256 hex digest of the AST structure.
    """
    # Use ast.dump() for structural representation
    # sort_keys=True is implicit in ast.dump() for dicts
    dump = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(dump.encode("utf-8")).hexdigest()


def are_semantically_equivalent(code1: str, code2: str) -> bool:
    """Check if two code snippets are semantically equivalent.

    This compares the canonical AST structure of both code snippets,
    ignoring formatting and stylistic differences.

    Args:
        code1: First code snippet.
        code2: Second code snippet.

    Returns:
        True if both snippets have the same semantic structure.

    Example:
        >>> are_semantically_equivalent(
        ...     'x = {"a": 1, "b": 2}',
        ...     'x = {"b": 2, "a": 1}'
        ... )
        True
    """
    try:
        tree1 = ast.parse(code1)
        tree2 = ast.parse(code2)

        tree1, _ = canonicalize_ast(tree1)
        tree2, _ = canonicalize_ast(tree2)

        return compute_ast_hash(tree1) == compute_ast_hash(tree2)

    except SyntaxError:
        return False


def normalize_module(source: str, *, sort_imports: bool = True, aggressive: bool = False) -> str:
    """Normalize a complete Python module.

    This applies additional module-level normalizations:
    - Groups and sorts imports (if sort_imports=True)
    - Applies standard canonicalization

    Args:
        source: Python module source code.
        sort_imports: If True, sort and group imports at the top.
        aggressive: If True, apply aggressive normalizations.

    Returns:
        Normalized module source code.
    """
    tree = ast.parse(source)
    tree, _ = canonicalize_ast(tree, aggressive=aggressive)

    if sort_imports:
        tree = _reorganize_imports(tree)

    return ast.unparse(tree)


def _reorganize_imports(tree: ast.Module) -> ast.Module:
    """Reorganize imports at the top of a module.

    Separates imports into:
    1. Standard library imports
    2. Third-party imports
    3. Local imports

    This is a simplified version - full import sorting should use isort.
    """
    # This is intentionally minimal - defer to isort for full sorting
    # We only ensure imports are at the top and sorted within their group

    imports: list[ast.stmt] = []
    other: list[ast.stmt] = []

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
        else:
            other.append(node)

    # Sort imports by module name
    imports.sort(key=lambda n: (isinstance(n, ast.ImportFrom), getattr(n, "module", "") or "", ast.unparse(n)))

    tree.body = imports + other
    return tree
