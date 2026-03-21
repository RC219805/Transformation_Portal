"""Tests for AST canonicalization and hashing."""

from __future__ import annotations

import ast

import pytest

pytestmark = pytest.mark.unit


class TestCanonicalizer:
    """Tests for the AST Canonicalizer."""

    def test_dict_key_sorting(self):
        """Dict with constant string keys should be sorted."""
        from transformation_portal.dev.ast_canonicalizer import Canonicalizer

        source = '{"b": 2, "a": 1, "c": 3}'
        tree = ast.parse(source, mode="eval")

        canonicalizer = Canonicalizer()
        new_tree = canonicalizer.visit(tree)

        result = ast.unparse(new_tree)
        assert result == "{'a': 1, 'b': 2, 'c': 3}"

    def test_dict_non_string_keys_not_sorted(self):
        """Dict with non-string keys should not be sorted."""
        from transformation_portal.dev.ast_canonicalizer import Canonicalizer

        source = "{2: 'b', 1: 'a'}"
        tree = ast.parse(source, mode="eval")

        canonicalizer = Canonicalizer()
        new_tree = canonicalizer.visit(tree)

        result = ast.unparse(new_tree)
        # Order should be preserved for non-string keys
        assert "2" in result and "1" in result

    def test_keyword_argument_sorting(self):
        """Keyword arguments should be sorted alphabetically."""
        from transformation_portal.dev.ast_canonicalizer import Canonicalizer

        source = "foo(z=3, a=1, m=2)"
        tree = ast.parse(source, mode="eval")

        canonicalizer = Canonicalizer()
        new_tree = canonicalizer.visit(tree)

        result = ast.unparse(new_tree)
        assert result == "foo(a=1, m=2, z=3)"

    def test_import_sorting(self):
        """Import names should be sorted."""
        from transformation_portal.dev.ast_canonicalizer import Canonicalizer

        source = "from os import path, getcwd, listdir"
        tree = ast.parse(source)

        canonicalizer = Canonicalizer()
        new_tree = canonicalizer.visit(tree)

        result = ast.unparse(new_tree)
        assert "getcwd" in result
        # Names should appear in sorted order
        assert result.index("getcwd") < result.index("listdir") < result.index("path")

    def test_none_comparison_normalization(self):
        """x == None should become x is None."""
        from transformation_portal.dev.ast_canonicalizer import Canonicalizer

        source = "x == None"
        tree = ast.parse(source, mode="eval")

        canonicalizer = Canonicalizer()
        new_tree = canonicalizer.visit(tree)

        result = ast.unparse(new_tree)
        assert result == "x is None"

    def test_not_none_comparison_normalization(self):
        """x != None should become x is not None."""
        from transformation_portal.dev.ast_canonicalizer import Canonicalizer

        source = "x != None"
        tree = ast.parse(source, mode="eval")

        canonicalizer = Canonicalizer()
        new_tree = canonicalizer.visit(tree)

        result = ast.unparse(new_tree)
        assert result == "x is not None"


class TestCanonicalizeCode:
    """Tests for the canonicalize_code function."""

    def test_basic_canonicalization(self):
        """Basic code should be canonicalized."""
        from transformation_portal.dev.ast_normalize import canonicalize_code

        source = 'x = {"b": 2, "a": 1}'
        result = canonicalize_code(source)

        assert "{'a': 1, 'b': 2}" in result

    def test_keyword_args_canonicalized(self):
        """Keyword arguments should be sorted."""
        from transformation_portal.dev.ast_normalize import canonicalize_code

        source = "foo(z=3, a=1)"
        result = canonicalize_code(source)

        assert result == "foo(a=1, z=3)"

    def test_invalid_syntax_raises(self):
        """Invalid syntax should raise SyntaxError."""
        from transformation_portal.dev.ast_normalize import canonicalize_code

        with pytest.raises(SyntaxError):
            canonicalize_code("def foo( invalid")


class TestCanonicalizeCodeSafe:
    """Tests for the safe canonicalization function."""

    def test_valid_code_returns_result(self):
        """Valid code should return a CanonicalResult."""
        from transformation_portal.dev.ast_normalize import canonicalize_code_safe

        result = canonicalize_code_safe('x = {"b": 2, "a": 1}')

        assert result.is_valid
        assert "{'a': 1, 'b': 2}" in result.canonical
        assert len(result.ast_hash) == 64  # SHA256 hex

    def test_invalid_code_returns_error(self):
        """Invalid code should return result with error."""
        from transformation_portal.dev.ast_normalize import canonicalize_code_safe

        result = canonicalize_code_safe("def foo( invalid")

        assert not result.is_valid
        assert result.error is not None
        assert "Syntax" in result.error


class TestSemanticEquivalence:
    """Tests for semantic equivalence detection."""

    def test_dict_order_equivalence(self):
        """Dicts with same keys in different order are equivalent."""
        from transformation_portal.dev.ast_normalize import are_semantically_equivalent

        code1 = 'x = {"a": 1, "b": 2}'
        code2 = 'x = {"b": 2, "a": 1}'

        assert are_semantically_equivalent(code1, code2)

    def test_kwarg_order_equivalence(self):
        """Functions with same kwargs in different order are equivalent."""
        from transformation_portal.dev.ast_normalize import are_semantically_equivalent

        code1 = "foo(a=1, b=2)"
        code2 = "foo(b=2, a=1)"

        assert are_semantically_equivalent(code1, code2)

    def test_different_code_not_equivalent(self):
        """Different code should not be equivalent."""
        from transformation_portal.dev.ast_normalize import are_semantically_equivalent

        code1 = "x = 1"
        code2 = "x = 2"

        assert not are_semantically_equivalent(code1, code2)

    def test_invalid_syntax_not_equivalent(self):
        """Invalid syntax should return False."""
        from transformation_portal.dev.ast_normalize import are_semantically_equivalent

        assert not are_semantically_equivalent("valid = 1", "invalid syntax (")


class TestASTHash:
    """Tests for AST hashing."""

    def test_same_code_same_hash(self):
        """Identical code should have the same hash."""
        from transformation_portal.dev.ast_hash import compute_ast_hash

        code = "x = 1"
        hash1 = compute_ast_hash(code)
        hash2 = compute_ast_hash(code)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex

    def test_equivalent_code_same_hash(self):
        """Semantically equivalent code should have the same hash."""
        from transformation_portal.dev.ast_hash import compute_ast_hash

        hash1 = compute_ast_hash('x = {"a": 1, "b": 2}')
        hash2 = compute_ast_hash('x = {"b": 2, "a": 1}')

        assert hash1 == hash2

    def test_different_code_different_hash(self):
        """Different code should have different hashes."""
        from transformation_portal.dev.ast_hash import compute_ast_hash

        hash1 = compute_ast_hash("x = 1")
        hash2 = compute_ast_hash("x = 2")

        assert hash1 != hash2

    def test_function_hash(self):
        """Function hash should work for specific functions."""
        from transformation_portal.dev.ast_hash import compute_function_hash

        source = """
def foo():
    return 1

def bar():
    return 2
"""
        foo_hash = compute_function_hash(source, "foo")
        bar_hash = compute_function_hash(source, "bar")

        assert foo_hash is not None
        assert bar_hash is not None
        assert foo_hash != bar_hash

    def test_structural_hash(self):
        """Structural hash should normalize variable names."""
        from transformation_portal.dev.ast_hash import compute_structural_hash

        # These have the same structure but different variable names
        hash1 = compute_structural_hash("x = a + b")
        hash2 = compute_structural_hash("y = c + d")

        assert hash1 == hash2


class TestASTHashSafe:
    """Tests for safe AST hashing."""

    def test_valid_code_returns_result(self):
        """Valid code should return ASTHashResult."""
        from transformation_portal.dev.ast_hash import compute_ast_hash_safe

        result = compute_ast_hash_safe("x = 1")

        assert result.is_valid
        assert len(result.hash) == 64
        assert result.node_count > 0

    def test_invalid_code_returns_error(self):
        """Invalid code should return result with error."""
        from transformation_portal.dev.ast_hash import compute_ast_hash_safe

        result = compute_ast_hash_safe("def invalid(")

        assert not result.is_valid
        assert result.error is not None
