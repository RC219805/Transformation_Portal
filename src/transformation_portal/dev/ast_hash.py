"""AST hashing for semantic code identity.

This module computes semantic hashes of Python code, enabling:
- Content-addressable code identification
- Semantic equivalence detection
- Duplicate logic detection across different syntax

Two pieces of code that are semantically identical will produce
the same hash, even if they differ in:
- Formatting and whitespace
- Dict/kwarg ordering (after canonicalization)
- Import ordering

Pipeline:
    SOURCE → CANONICAL AST → DETERMINISTIC DICT → JSON → SHA256

Usage:
    from transformation_portal.dev.ast_hash import compute_ast_hash

    hash1 = compute_ast_hash('x = {"b": 2, "a": 1}')
    hash2 = compute_ast_hash('x = {"a": 1, "b": 2}')
    assert hash1 == hash2  # Same semantic structure
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

from transformation_portal.dev.ast_canonicalizer import canonicalize_ast

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ASTHashResult:
    """Result of AST hashing operation."""

    hash: str
    """SHA256 hash of the semantic AST structure."""

    source_hash: str
    """SHA256 hash of the original source (for comparison)."""

    node_count: int
    """Number of AST nodes in the tree."""

    is_valid: bool
    """Whether the source was valid Python."""

    error: Optional[str] = None
    """Error message if hashing failed."""


def _ast_to_dict(node: Any) -> Any:
    """Convert AST node to a deterministic dict representation.

    This creates a JSON-serializable structure that captures
    the semantic structure of the AST while ignoring:
    - Line numbers and column offsets
    - Formatting details
    - Source location information

    Args:
        node: AST node or primitive value.

    Returns:
        Dict representation suitable for hashing.
    """
    if isinstance(node, ast.AST):
        # Get all fields, sorted for determinism
        fields = {k: _ast_to_dict(v) for k, v in sorted(ast.iter_fields(node))}

        return {
            "_type": node.__class__.__name__,
            **fields,
        }

    elif isinstance(node, list):
        return [_ast_to_dict(item) for item in node]

    elif isinstance(node, (str, int, float, bool, type(None))):
        return node

    elif isinstance(node, bytes):
        return {"_bytes": node.hex()}

    elif isinstance(node, complex):
        return {"_complex": [node.real, node.imag]}

    else:
        # Fallback for unknown types
        return {"_repr": repr(node)}


def _count_nodes(tree: ast.AST) -> int:
    """Count the number of nodes in an AST."""
    count = 1
    for child in ast.walk(tree):
        count += 1
    return count


def compute_ast_hash(source: str, *, canonicalize: bool = True) -> str:
    """Compute semantic hash of Python source code.

    This function:
    1. Parses the source into an AST
    2. Optionally canonicalizes the AST (sorts dicts, kwargs, etc.)
    3. Converts to a deterministic dict representation
    4. Serializes to JSON with sorted keys
    5. Computes SHA256 hash

    Args:
        source: Python source code to hash.
        canonicalize: If True, apply canonicalization before hashing.

    Returns:
        SHA256 hex digest representing the semantic structure.

    Raises:
        SyntaxError: If the source code has invalid syntax.

    Example:
        >>> h1 = compute_ast_hash('x = {"b": 2, "a": 1}')
        >>> h2 = compute_ast_hash('x = {"a": 1, "b": 2}')
        >>> h1 == h2
        True
    """
    tree = ast.parse(source)

    if canonicalize:
        tree, _ = canonicalize_ast(tree)

    ast_dict = _ast_to_dict(tree)

    # Serialize with sorted keys and minimal separators for determinism
    payload = json.dumps(ast_dict, sort_keys=True, separators=(",", ":"))

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_ast_hash_safe(source: str, *, canonicalize: bool = True) -> ASTHashResult:
    """Safely compute AST hash, returning a result object.

    Unlike compute_ast_hash(), this function never raises exceptions.
    Instead, it returns an ASTHashResult with error information.

    Args:
        source: Python source code to hash.
        canonicalize: If True, apply canonicalization before hashing.

    Returns:
        ASTHashResult with hash or error information.
    """
    try:
        tree = ast.parse(source)

        if canonicalize:
            tree, _ = canonicalize_ast(tree)

        ast_dict = _ast_to_dict(tree)
        payload = json.dumps(ast_dict, sort_keys=True, separators=(",", ":"))
        ast_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()

        # Also compute source hash for comparison
        source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()

        return ASTHashResult(
            hash=ast_hash,
            source_hash=source_hash,
            node_count=_count_nodes(tree),
            is_valid=True,
        )

    except SyntaxError as e:
        return ASTHashResult(
            hash="",
            source_hash=hashlib.sha256(source.encode("utf-8")).hexdigest(),
            node_count=0,
            is_valid=False,
            error=f"Syntax error: {e}",
        )

    except Exception as e:
        logger.warning("Unexpected error during AST hashing: %s", e)
        return ASTHashResult(
            hash="",
            source_hash=hashlib.sha256(source.encode("utf-8")).hexdigest(),
            node_count=0,
            is_valid=False,
            error=f"Unexpected error: {e}",
        )


def _hash_definition(node: ast.stmt) -> str:
    """Hash one function or class definition as a stand-alone module.

    The node is canonicalized in place; pass a copy when the tree it belongs
    to is still needed.
    """
    # Create a module containing just this definition
    def_tree, _ = canonicalize_ast(ast.Module(body=[node], type_ignores=[]))
    ast_dict = _ast_to_dict(def_tree)
    payload = json.dumps(ast_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_definition_hashes(source: str) -> tuple[dict[str, str], dict[str, str]]:
    """Hash every function and class definition of ``source`` with a single parse.

    Returns ``(functions, classes)`` mapping a definition key to the hash that
    compute_function_hash()/compute_class_hash() give for it. The key is the
    bare name when that name is defined once in the file; same-named
    definitions (for example ``run`` methods of two classes) are keyed by
    their dotted qualified name (``A.run``, ``B.run``) so they no longer
    collapse into one entry. Definitions are visited in ``ast.walk`` order and
    the first one wins when even the qualified name repeats.

    Raises:
        SyntaxError: If ``source`` is not valid Python.
    """
    import copy

    tree = ast.parse(source)
    qualnames: dict[int, str] = {}
    stack: list[tuple[ast.AST, str]] = [(tree, "")]
    while stack:
        parent, prefix = stack.pop()
        for child in ast.iter_child_nodes(parent):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qualname = f"{prefix}.{child.name}" if prefix else child.name
                qualnames[id(child)] = qualname
                stack.append((child, qualname))
            else:
                stack.append((child, prefix))

    functions_nodes: list[ast.stmt] = []
    class_nodes: list[ast.stmt] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions_nodes.append(node)
        elif isinstance(node, ast.ClassDef):
            class_nodes.append(node)

    def _hash_all(nodes: list[ast.stmt]) -> dict[str, str]:
        name_counts: dict[str, int] = {}
        for node in nodes:
            name = node.name  # type: ignore[attr-defined]
            name_counts[name] = name_counts.get(name, 0) + 1
        hashes: dict[str, str] = {}
        for node in nodes:
            name = node.name  # type: ignore[attr-defined]
            key = name if name_counts[name] == 1 else qualnames[id(node)]
            if key in hashes:
                continue
            try:
                # Copy: canonicalization mutates, and enclosing definitions
                # share their nodes with the definitions nested inside them.
                hashes[key] = _hash_definition(copy.deepcopy(node))
            except Exception as e:  # pragma: no cover - defensive, mirrors compute_function_hash
                logger.warning("Error computing definition hash for %s: %s", key, e)
        return hashes

    return _hash_all(functions_nodes), _hash_all(class_nodes)


def compute_function_hash(source: str, function_name: str) -> Optional[str]:
    """Compute hash of a specific function within source code.

    This extracts and hashes a single function definition,
    useful for detecting duplicate functions across modules.

    Args:
        source: Python source code containing the function.
        function_name: Name of the function to hash.

    Returns:
        SHA256 hash of the function, or None if not found.
    """
    try:
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == function_name:
                    return _hash_definition(node)

        return None

    except Exception as e:
        logger.warning("Error computing function hash: %s", e)
        return None


def compute_class_hash(source: str, class_name: str) -> Optional[str]:
    """Compute hash of a specific class within source code.

    This extracts and hashes a single class definition,
    useful for detecting duplicate classes across modules.

    Args:
        source: Python source code containing the class.
        class_name: Name of the class to hash.

    Returns:
        SHA256 hash of the class, or None if not found.
    """
    try:
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if node.name == class_name:
                    return _hash_definition(node)

        return None

    except Exception as e:
        logger.warning("Error computing class hash: %s", e)
        return None


class AlphaRenamer(ast.NodeTransformer):
    """AST transformer that alpha-renames variables for deeper equivalence.

    This normalizes variable names to detect structural equivalence
    even when variable names differ:

        x = a + b  →  v0 = v1 + v2
        y = c + d  →  v0 = v1 + v2

    Use with caution: this is more aggressive and may produce
    false positives for code that is intentionally different.
    """

    def __init__(self):
        super().__init__()
        self.name_map: dict[str, str] = {}
        self.counter = 0

    def _get_normalized_name(self, original: str) -> str:
        """Get or create a normalized name for a variable."""
        if original not in self.name_map:
            self.name_map[original] = f"v{self.counter}"
            self.counter += 1
        return self.name_map[original]

    def visit_Name(self, node: ast.Name) -> ast.Name:
        """Rename variable references."""
        # Don't rename built-in names
        builtins = {"True", "False", "None", "print", "len", "range", "str", "int", "float", "list", "dict", "set", "tuple"}
        if node.id not in builtins:
            node.id = self._get_normalized_name(node.id)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Rename function definitions."""
        node.name = self._get_normalized_name(node.name)
        self.generic_visit(node)
        return node

    def visit_arg(self, node: ast.arg) -> ast.arg:
        """Rename function arguments."""
        node.arg = self._get_normalized_name(node.arg)
        return node


def compute_structural_hash(source: str) -> str:
    """Compute hash with alpha-renamed variables for deep structural equivalence.

    This hash will be the same for code that has the same structure
    but different variable names:

        def add(x, y): return x + y
        def sum(a, b): return a + b

    Both will produce the same structural hash.

    Args:
        source: Python source code to hash.

    Returns:
        SHA256 hash of the alpha-renamed structure.

    Warning:
        This is aggressive and may produce false positives.
        Use for similarity detection, not identity.
    """
    tree = ast.parse(source)
    tree, _ = canonicalize_ast(tree)

    # Apply alpha-renaming
    renamer = AlphaRenamer()
    tree = renamer.visit(tree)
    ast.fix_missing_locations(tree)

    ast_dict = _ast_to_dict(tree)
    payload = json.dumps(ast_dict, sort_keys=True, separators=(",", ":"))

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
