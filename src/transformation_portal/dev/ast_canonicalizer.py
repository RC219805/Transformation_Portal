"""AST Canonicalizer for deterministic code structure.

This module enforces canonical Python AST structure to eliminate
stylistic drift across code generators (Copilot, scripts, humans).

The canonicalizer applies safe transformations that:
- Preserve semantic equivalence
- Normalize structure deterministically
- Eliminate generator variance

Pipeline:
    RAW CODE → AST PARSE → CANONICAL TRANSFORM → AST → CODE → BLACK → COMMIT

Safety rules (DO NOT normalize):
- Non-constant dict keys (may change semantics)
- Function call order (side effects): dict entries and keyword arguments
  are only reordered when every value is a constant or a plain name
- Dicts containing ``**`` spreads (position decides which value wins)
- ``== None`` comparisons, outside aggressive mode (``__eq__`` overloads)
- Floating-point expressions (precision drift)
- List order with non-constants (semantic meaning)
"""

from __future__ import annotations

import ast
import logging
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=ast.AST)


def _is_side_effect_free(node: ast.expr) -> bool:
    """Return True when evaluating ``node`` cannot run user code (constant or plain name)."""
    return isinstance(node, ast.Constant) or (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load))


class Canonicalizer(ast.NodeTransformer):
    """Enforces canonical Python AST structure.

    Goals:
    - Normalize dict/list/set ordering where safe
    - Normalize imports
    - Normalize keyword arguments
    - Remove redundant constructs

    All transformations preserve semantic equivalence.
    """

    def __init__(self, *, aggressive: bool = False):
        """Initialize the canonicalizer.

        Args:
            aggressive: If True, apply more aggressive normalizations
                       that may change code appearance significantly. These
                       are NOT guaranteed to preserve semantics (e.g.
                       ``x == None`` -> ``x is None`` for objects that
                       overload ``__eq__``).
        """
        super().__init__()
        self.aggressive = aggressive
        self._transformations_applied: list[str] = []

    @property
    def transformations_applied(self) -> list[str]:
        """List of transformations that were applied."""
        return self._transformations_applied.copy()

    # -------------------------
    # Dict normalization
    # -------------------------
    def visit_Dict(self, node: ast.Dict) -> ast.AST:
        """Normalize dict literals by sorting constant string keys."""
        self.generic_visit(node)

        # Only sort if ALL keys are constant strings (safe transformation)
        if not node.keys:
            return node

        try:
            # Sort only when every key is a constant string (a ``None`` key is a
            # ``**`` spread whose position decides which value wins) and every
            # value is side-effect free (values are evaluated left to right).
            if all(isinstance(k, ast.Constant) and isinstance(k.value, str) for k in node.keys) and all(
                _is_side_effect_free(v) for v in node.values
            ):
                items = list(zip(node.keys, node.values))
                items.sort(key=lambda kv: kv[0].value)

                node.keys = [k for k, _ in items]
                node.values = [v for _, v in items]
                self._transformations_applied.append("dict_key_sort")

        except Exception as e:
            logger.debug("Dict normalization skipped: %s", e)

        return node

    # -------------------------
    # Set normalization
    # -------------------------
    def visit_Set(self, node: ast.Set) -> ast.AST:
        """Normalize set literals by sorting constant elements."""
        self.generic_visit(node)

        # Only sort if all elements are constants and no two of them are equal
        # set members: for equal members (``{1, True}``, ``{1, 1.0}``) the first
        # one in source order is the one the set keeps.
        if all(isinstance(el, ast.Constant) for el in node.elts):
            try:
                values = [el.value for el in node.elts]
                if len(set(values)) != len(values):
                    return node
                node.elts = sorted(node.elts, key=lambda x: (type(x.value).__name__, str(x.value)))
                self._transformations_applied.append("set_element_sort")
            except Exception as e:
                logger.debug("Set normalization skipped: %s", e)

        return node

    # -------------------------
    # List normalization (very conservative)
    # -------------------------
    def visit_List(self, node: ast.List) -> ast.AST:
        """Normalize list literals (only in specific safe contexts)."""
        self.generic_visit(node)

        # DO NOT sort lists by default - order is semantic
        # Only sort if explicitly in a context where order doesn't matter
        # (e.g., __all__ declarations)

        return node

    # -------------------------
    # Normalize function definitions
    # -------------------------
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        """Normalize function definitions."""
        self.generic_visit(node)

        # Sort decorators alphabetically (if aggressive mode)
        if self.aggressive and node.decorator_list:
            try:
                node.decorator_list.sort(key=lambda d: ast.unparse(d))
                self._transformations_applied.append("decorator_sort")
            except Exception:
                pass

        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        """Normalize async function definitions."""
        self.generic_visit(node)
        return node

    # -------------------------
    # Normalize class definitions
    # -------------------------
    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        """Normalize class definitions."""
        self.generic_visit(node)

        # Sort base classes alphabetically (if aggressive mode)
        if self.aggressive and node.bases:
            try:
                node.bases.sort(key=lambda b: ast.unparse(b))
                self._transformations_applied.append("base_class_sort")
            except Exception:
                pass

        return node

    # -------------------------
    # Normalize function calls
    # -------------------------
    def visit_Call(self, node: ast.Call) -> ast.AST:
        """Normalize function calls by sorting keyword arguments."""
        self.generic_visit(node)

        # Sort keyword arguments alphabetically, but only when every keyword
        # value (including ``**`` spreads) is side-effect free: argument
        # expressions are evaluated in source order.
        if node.keywords and all(_is_side_effect_free(kw.value) for kw in node.keywords):
            try:
                # Separate **kwargs (arg=None) from regular kwargs
                regular_kwargs = [kw for kw in node.keywords if kw.arg is not None]
                star_kwargs = [kw for kw in node.keywords if kw.arg is None]

                # Sort regular kwargs
                regular_kwargs.sort(key=lambda kw: kw.arg)

                # Reconstruct: regular kwargs first, then **kwargs
                node.keywords = regular_kwargs + star_kwargs
                self._transformations_applied.append("keyword_arg_sort")

            except Exception as e:
                logger.debug("Keyword argument sorting skipped: %s", e)

        return node

    # -------------------------
    # Normalize imports
    # -------------------------
    def visit_Import(self, node: ast.Import) -> ast.AST:
        """Normalize import statements by sorting names."""
        self.generic_visit(node)

        try:
            node.names.sort(key=lambda n: n.name)
            self._transformations_applied.append("import_sort")
        except Exception as e:
            logger.debug("Import sorting skipped: %s", e)

        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST:
        """Normalize from-import statements by sorting names."""
        self.generic_visit(node)

        try:
            node.names.sort(key=lambda n: n.name)
            self._transformations_applied.append("import_from_sort")
        except Exception as e:
            logger.debug("Import-from sorting skipped: %s", e)

        return node

    # -------------------------
    # Normalize comparisons
    # -------------------------
    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        """Normalize comparison expressions."""
        self.generic_visit(node)

        # Normalize `x == None` to `x is None` (PEP 8). Aggressive mode only:
        # objects that overload ``__eq__`` (SQLAlchemy columns, numpy arrays)
        # return a different value for ``==`` than ``is``.
        if self.aggressive and len(node.ops) == 1 and len(node.comparators) == 1:
            op = node.ops[0]
            comp = node.comparators[0]

            if isinstance(comp, ast.Constant) and comp.value is None:
                if isinstance(op, ast.Eq):
                    node.ops = [ast.Is()]
                    self._transformations_applied.append("none_comparison_normalize")
                elif isinstance(op, ast.NotEq):
                    node.ops = [ast.IsNot()]
                    self._transformations_applied.append("none_comparison_normalize")

        return node

    # -------------------------
    # Normalize boolean operations
    # -------------------------
    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        """Normalize boolean operations."""
        self.generic_visit(node)

        # Sort operands in boolean expressions (only for commutative ops with constants)
        if self.aggressive and isinstance(node.op, (ast.And, ast.Or)):
            try:
                # Only sort if all operands are "simple" (names or constants)
                if all(isinstance(v, (ast.Name, ast.Constant)) for v in node.values):
                    node.values.sort(key=lambda v: ast.unparse(v))
                    self._transformations_applied.append("bool_op_sort")
            except Exception:
                pass

        return node

    # -------------------------
    # Normalize subscripts
    # -------------------------
    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        """Normalize subscript expressions."""
        self.generic_visit(node)
        return node

    # -------------------------
    # Normalize f-strings
    # -------------------------
    def visit_JoinedStr(self, node: ast.JoinedStr) -> ast.AST:
        """Normalize f-string expressions."""
        self.generic_visit(node)
        return node

    # -------------------------
    # Normalize annotations
    # -------------------------
    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        """Normalize annotated assignments."""
        self.generic_visit(node)
        return node


def canonicalize_ast(tree: ast.AST, *, aggressive: bool = False) -> tuple[ast.AST, list[str]]:
    """Apply canonical transformations to an AST.

    Args:
        tree: The AST to canonicalize.
        aggressive: If True, apply more aggressive normalizations.

    Returns:
        Tuple of (canonicalized AST, list of transformations applied).
    """
    canonicalizer = Canonicalizer(aggressive=aggressive)
    new_tree = canonicalizer.visit(tree)
    ast.fix_missing_locations(new_tree)
    return new_tree, canonicalizer.transformations_applied
