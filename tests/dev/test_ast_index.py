"""Tests for the AST equivalence index (M-26: single parse, no name collapse)."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

from transformation_portal.dev.ast_hash import compute_class_hash, compute_definition_hashes, compute_function_hash
from transformation_portal.dev.ast_index import ASTEquivalenceIndex

pytestmark = pytest.mark.unit

_SOURCE = textwrap.dedent("""
    def top(a, b):
        def inner(x):
            return {"b": x, "a": 2}
        return inner(a) + b


    class Outer:
        flag = {"z": 1, "y": 2}

        def method(self, value):
            return value * 2

        class Inner:
            def deep(self):
                return "deep"


    async def fetch(url):
        return url
    """)


def test_unique_definitions_hash_like_the_per_name_helpers(tmp_path: Path) -> None:
    path = tmp_path / "mod.py"
    path.write_text(_SOURCE, encoding="utf-8")

    entry = ASTEquivalenceIndex().add_file(path)

    assert entry is not None
    assert list(entry.functions) == ["top", "fetch", "inner", "method", "deep"]
    assert list(entry.classes) == ["Outer", "Inner"]
    for name, digest in entry.functions.items():
        assert digest == compute_function_hash(_SOURCE, name)
    for name, digest in entry.classes.items():
        assert digest == compute_class_hash(_SOURCE, name)


def test_same_named_methods_in_different_classes_do_not_collapse(tmp_path: Path) -> None:
    source = textwrap.dedent("""
        def run():
            return 0


        class A:
            def run(self):
                return 1


        class B:
            def run(self):
                return "completely different"
        """)
    path = tmp_path / "methods.py"
    path.write_text(source, encoding="utf-8")

    index = ASTEquivalenceIndex()
    entry = index.add_file(path)

    assert entry is not None
    assert set(entry.functions) == {"run", "A.run", "B.run"}
    # The top-level definition keeps its bare key and its previous hash.
    assert entry.functions["run"] == compute_function_hash(source, "run")
    assert len(set(entry.functions.values())) == 3
    assert {loc for locs in index.function_index.values() for loc in locs} == {
        (path, "run"),
        (path, "A.run"),
        (path, "B.run"),
    }


def test_identical_methods_in_different_classes_are_reported_as_duplicates(tmp_path: Path) -> None:
    source = textwrap.dedent("""
        class A:
            def run(self):
                return 1


        class B:
            def run(self):
                return 1
        """)
    path = tmp_path / "dups.py"
    path.write_text(source, encoding="utf-8")

    index = ASTEquivalenceIndex()
    index.add_file(path)

    groups = list(index.find_duplicate_functions().values())
    assert groups == [[(path, "A.run"), (path, "B.run")]]


def test_add_file_parses_the_source_a_bounded_number_of_times(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = "\n".join(f"def f{i}(a):\n    return a + {i}\n" for i in range(40))
    source += "\n" + "\n".join(f"class C{i}:\n    def m{i}(self):\n        return {i}\n" for i in range(10))
    path = tmp_path / "many.py"
    path.write_text(source, encoding="utf-8")

    calls = {"n": 0}
    real_parse = ast.parse

    def counting_parse(*args, **kwargs):
        calls["n"] += 1
        return real_parse(*args, **kwargs)

    monkeypatch.setattr(ast, "parse", counting_parse)
    entry = ASTEquivalenceIndex().add_file(path)

    assert entry is not None
    assert len(entry.functions) == 50
    assert len(entry.classes) == 10
    # One parse for the file hash, one for all definitions (was one per definition).
    assert calls["n"] <= 2


def test_compute_definition_hashes_rejects_invalid_source() -> None:
    with pytest.raises(SyntaxError):
        compute_definition_hashes("def broken(:\n    pass\n")
