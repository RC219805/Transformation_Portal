"""Tests for the auto-refactoring engine."""

from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_FIXTURE_PACKAGE = "tp_refactor_fixture"

_TILES = '''\
"""Tile helpers."""
from __future__ import annotations

LIMIT = 4


def _clamp(v: int) -> int:
    return max(0, min(LIMIT, v))


def tile(v: int) -> int:
    return _clamp(v) * 2
'''

# Star-imports and explicitly imports (private names included) from a duplicate.
_NAMESPACE_PROBE = f"""
namespace = {{}}
exec("from {_FIXTURE_PACKAGE}.b.tiles import *", namespace)
from {_FIXTURE_PACKAGE}.b.tiles import LIMIT, _clamp, tile
print(sorted(name for name in namespace if name != "__builtins__"), tile(9), _clamp(-3), LIMIT)
"""


def _make_tiles_tree(root: Path, body: str = _TILES, *, packages: bool = True) -> dict[Path, bytes]:
    """Create three identical ``tiles.py`` modules under a throw-away package and snapshot every file."""
    for sub in ("a", "b", "c"):
        directory = root / _FIXTURE_PACKAGE / sub
        directory.mkdir(parents=True)
        if packages:
            (directory / "__init__.py").write_text("")
        (directory / "tiles.py").write_text(body)
    if packages:
        (root / _FIXTURE_PACKAGE / "__init__.py").write_text("")
    return _snapshot(root)


def _snapshot(root: Path) -> dict[Path, bytes]:
    return {path: path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def _run_python(root: Path, code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _engine(root: Path, **kwargs):
    from transformation_portal.dev.refactor_engine import AutoRefactorEngine

    return AutoRefactorEngine(root, root / _FIXTURE_PACKAGE / "shared", **kwargs)


class TestRefactorPlan:
    """Tests for RefactorPlan."""

    def test_empty_plan_summary(self):
        """Empty plan should have valid summary."""
        from transformation_portal.dev.refactor_engine import RefactorPlan

        plan = RefactorPlan()
        summary = plan.summary()

        assert "Refactoring Plan" in summary
        assert "Duplicate groups: 0" in summary

    def test_plan_with_duplicates_summary(self):
        """Plan with duplicates should show them in summary."""
        from transformation_portal.dev.refactor_engine import RefactorPlan

        plan = RefactorPlan()
        plan.canonical_map["abc123"] = Path("src/a.py")
        plan.duplicates["abc123"] = [Path("src/b.py"), Path("src/c.py")]

        summary = plan.summary()

        assert "Duplicate groups: 1" in summary
        assert "abc123" in summary


class TestAutoRefactorEngine:
    """Tests for AutoRefactorEngine."""

    def test_build_plan_empty_directory(self):
        """Building plan for empty directory should return empty plan."""
        from transformation_portal.dev.refactor_engine import AutoRefactorEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shared = root / "shared"

            engine = AutoRefactorEngine(root, shared)
            plan = engine.build_plan()

            assert len(plan.duplicates) == 0

    def test_build_plan_no_duplicates(self):
        """Building plan with no duplicates should return empty plan."""
        from transformation_portal.dev.refactor_engine import AutoRefactorEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shared = root / "shared"

            # Create unique files
            (root / "a.py").write_text("def a(): return 1\n")
            (root / "b.py").write_text("def b(): return 2\n")

            engine = AutoRefactorEngine(root, shared)
            plan = engine.build_plan()

            assert len(plan.duplicates) == 0

    def test_build_plan_with_duplicates(self):
        """Building plan with duplicates should detect them."""
        from transformation_portal.dev.refactor_engine import AutoRefactorEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shared = root / "shared"

            # Create duplicate files
            code = "def foo(): return 1\n"
            (root / "a.py").write_text(code)
            (root / "b.py").write_text(code)

            engine = AutoRefactorEngine(root, shared)
            plan = engine.build_plan()

            assert len(plan.duplicates) == 1

    def test_select_canonical_shortest_path(self):
        """Canonical selection should prefer shorter paths."""
        from transformation_portal.dev.refactor_engine import AutoRefactorEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            engine = AutoRefactorEngine(root, root / "shared")

            files = [Path("src/deep/nested/file.py"), Path("src/file.py"), Path("src/another/file.py")]

            canonical = engine._select_canonical(files)

            assert canonical == Path("src/file.py")

    def test_extract_symbols(self):
        """Symbol extraction should find functions and classes."""
        from transformation_portal.dev.refactor_engine import AutoRefactorEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            engine = AutoRefactorEngine(root, root / "shared")

            code = """
def foo():
    return 1

class Bar:
    pass

async def baz():
    pass
"""
            file = root / "test.py"
            file.write_text(code)

            symbols = engine._extract_symbols(file)

            names = [s.name for s in symbols]
            assert "foo" in names
            assert "Bar" in names
            assert "baz" in names

            kinds = {s.name: s.kind for s in symbols}
            assert kinds["foo"] == "function"
            assert kinds["Bar"] == "class"
            assert kinds["baz"] == "async_function"

    def test_execute_dry_run(self):
        """Dry run should not modify files."""
        from transformation_portal.dev.refactor_engine import AutoRefactorEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shared = root / "shared"

            # Create duplicates
            code = "def foo(): return 1\n"
            (root / "a.py").write_text(code)
            (root / "b.py").write_text(code)

            engine = AutoRefactorEngine(root, shared)
            plan = engine.build_plan()
            result = engine.execute(plan, dry_run=True)

            # Should have planned changes
            assert len(result.files_created) > 0 or len(result.files_modified) > 0

            # But shared directory should not exist
            assert not shared.exists()

            # And files should be unchanged
            assert (root / "a.py").read_text() == code
            assert (root / "b.py").read_text() == code

    def test_execute_creates_shared_module(self):
        """Execution should create shared module with canonical code."""
        from transformation_portal.dev.refactor_engine import AutoRefactorEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shared = root / "shared"

            # Create duplicates
            code = "def foo(): return 1\n"
            (root / "a.py").write_text(code)
            (root / "b.py").write_text(code)

            engine = AutoRefactorEngine(root, shared, package_name="shared")
            plan = engine.build_plan()
            result = engine.execute(plan, dry_run=False)

            # Shared module should exist
            assert shared.exists()
            assert (shared / "__init__.py").exists()

            # Should have created at least one module file
            py_files = list(shared.glob("_*.py"))
            assert len(py_files) >= 1

    def test_preview_shows_actions(self):
        """Preview should show planned actions."""
        from transformation_portal.dev.refactor_engine import AutoRefactorEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shared = root / "shared"

            code = "def foo(): return 1\n"
            (root / "a.py").write_text(code)
            (root / "b.py").write_text(code)

            engine = AutoRefactorEngine(root, shared)
            plan = engine.build_plan()
            preview = engine.preview(plan)

            assert "CREATE" in preview
            assert "REWRITE" in preview


class TestRefactorResult:
    """Tests for RefactorResult."""

    def test_success_summary(self):
        """Successful result should have appropriate summary."""
        from transformation_portal.dev.refactor_engine import RefactorResult

        result = RefactorResult(success=True)
        result.files_created = [Path("a.py")]

        summary = result.summary()

        assert "SUCCESS" in summary
        assert "Files created: 1" in summary

    def test_dry_run_summary(self):
        """Dry run result should indicate it in summary."""
        from transformation_portal.dev.refactor_engine import RefactorResult

        result = RefactorResult(success=True, dry_run=True)

        summary = result.summary()

        assert "DRY RUN" in summary


class TestIncrementalRefactor:
    """Tests for incremental refactoring."""

    def test_refactor_by_hash(self):
        """Should be able to refactor a specific hash only."""
        from transformation_portal.dev.refactor_engine import (
            AutoRefactorEngine,
            IncrementalRefactor,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shared = root / "shared"

            # Create multiple duplicate groups
            (root / "a1.py").write_text("def foo(): return 1\n")
            (root / "a2.py").write_text("def foo(): return 1\n")
            (root / "b1.py").write_text("def bar(): return 2\n")
            (root / "b2.py").write_text("def bar(): return 2\n")

            engine = AutoRefactorEngine(root, shared)
            plan = engine.build_plan()

            # Get first hash
            target_hash = list(plan.duplicates.keys())[0]

            incremental = IncrementalRefactor(engine)
            result = incremental.refactor_by_hash(plan, target_hash, dry_run=True)

            # Should only affect files with that hash
            assert result.dry_run


class TestSafeExecution:
    """A real execution must leave an importable tree or no change at all (audit finding M-22)."""

    def test_rewritten_duplicates_keep_future_import_private_names_and_star_exports(self, tmp_path):
        _make_tiles_tree(tmp_path)
        before = _run_python(tmp_path, _NAMESPACE_PROBE)
        assert before.returncode == 0, before.stderr

        engine = _engine(tmp_path)
        result = engine.execute(engine.build_plan(), dry_run=False)

        assert result.success, result.errors
        assert result.errors == []
        assert len(result.files_modified) == 2
        after = _run_python(tmp_path, _NAMESPACE_PROBE)
        assert after.returncode == 0, after.stderr
        assert after.stdout == before.stdout

        stub = (tmp_path / _FIXTURE_PACKAGE / "b" / "tiles.py").read_text()
        assert "import *" not in stub
        stub_tree = ast.parse(stub)
        assert ast.get_docstring(stub_tree) == "Tile helpers."

    def test_min_symbols_is_honoured(self, tmp_path):
        _make_tiles_tree(tmp_path)

        plan = _engine(tmp_path).build_plan()
        assert len(plan.duplicates) == 1
        assert all(path.name == "tiles.py" for path in plan.canonical_map.values())
        assert not any(path.name == "__init__.py" for files in plan.duplicates.values() for path in files)
        assert any("min_symbols=1" in reason for reason in plan.skipped.values())
        assert "Skipped groups: 1" in plan.summary()

        strict_plan = _engine(tmp_path, min_symbols=3).build_plan()
        assert strict_plan.duplicates == {}
        assert sum("min_symbols=3" in reason for reason in strict_plan.skipped.values()) == 2

    def test_failed_import_check_reports_failure_and_leaves_tree_untouched(self, tmp_path):
        body = "import tp_refactor_missing_dependency\n\n\ndef tile(v):\n    return v * 2\n"
        original = _make_tiles_tree(tmp_path, body)

        engine = _engine(tmp_path)
        result = engine.execute(engine.build_plan(), dry_run=False)

        assert result.success is False
        assert any("Import check failed" in error and "ModuleNotFoundError" in error for error in result.errors)
        assert result.rolled_back is True
        assert "FAILED (rolled back)" in result.summary()
        assert _snapshot(tmp_path) == original
        assert not (tmp_path / _FIXTURE_PACKAGE / "shared").exists()

    def test_import_check_timeout_fails_closed(self, tmp_path, monkeypatch):
        from transformation_portal.dev import refactor_engine

        original = _make_tiles_tree(tmp_path)

        def _timeout(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout"))

        monkeypatch.setattr(refactor_engine.subprocess, "run", _timeout)
        engine = _engine(tmp_path)
        result = engine.execute(engine.build_plan(), dry_run=False)

        assert result.success is False
        assert any("timed out" in error for error in result.errors)
        assert _snapshot(tmp_path) == original

    def test_stale_plan_is_refused_and_rolled_back(self, tmp_path):
        _make_tiles_tree(tmp_path)
        engine = _engine(tmp_path)
        plan = engine.build_plan()

        edited = tmp_path / _FIXTURE_PACKAGE / "c" / "tiles.py"
        edited.write_text(_TILES + "\nEXTRA = 1\n")
        expected = _snapshot(tmp_path)

        result = engine.execute(plan, dry_run=False)

        assert result.success is False
        assert any("changed since the plan was built" in error for error in result.errors)
        assert _snapshot(tmp_path) == expected

    def test_rollback_restores_rewritten_files(self, tmp_path):
        original = _make_tiles_tree(tmp_path)
        engine = _engine(tmp_path)
        result = engine.execute(engine.build_plan(), dry_run=False)
        assert result.success, result.errors
        assert _snapshot(tmp_path) != original

        engine.rollback(result)

        assert result.rolled_back is True
        assert "ROLLED BACK" in result.summary()
        assert _snapshot(tmp_path) == original
        assert not (tmp_path / _FIXTURE_PACKAGE / "shared").exists()
        engine.rollback(result)  # a second rollback is a no-op
        assert _snapshot(tmp_path) == original

    def test_existing_shared_module_is_never_overwritten_or_deleted(self, tmp_path):
        _make_tiles_tree(tmp_path)
        engine = _engine(tmp_path)
        plan = engine.build_plan()
        (hash_val,) = plan.duplicates
        shared = tmp_path / _FIXTURE_PACKAGE / "shared"
        shared.mkdir()
        foreign = shared / f"_{hash_val[:12]}.py"
        foreign.write_text("OTHER = 1\n")
        expected = _snapshot(tmp_path)

        result = engine.execute(plan, dry_run=False)

        assert result.success is False
        assert any("Refusing to overwrite" in error for error in result.errors)
        assert _snapshot(tmp_path) == expected

    def test_second_run_finds_nothing_to_refactor(self, tmp_path):
        _make_tiles_tree(tmp_path, packages=False)
        engine = _engine(tmp_path, min_symbols=0)
        result = engine.execute(engine.build_plan(), dry_run=False)
        assert result.success, result.errors

        second = _engine(tmp_path, min_symbols=0).build_plan()

        assert second.duplicates == {}
        assert second.skipped == {}

    def test_module_without_bindings_still_runs_and_keeps_docstring(self, tmp_path):
        body = '"""Registers on import."""\nprint("loaded")\n'
        (tmp_path / "first.py").write_text(body)
        (tmp_path / "second.py").write_text(body)
        engine = _engine(tmp_path, min_symbols=0)

        result = engine.execute(engine.build_plan(), dry_run=False)

        assert result.success, result.errors
        rewritten = next(path for path in result.files_modified)
        probe = _run_python(tmp_path, f"import {rewritten.stem} as m; print(m.__doc__)")
        assert probe.returncode == 0, probe.stderr
        assert probe.stdout.splitlines() == ["loaded", "Registers on import."]

    @pytest.mark.parametrize(
        ("body", "reason_fragment"),
        [
            ("from .config import LIMIT\n\n\ndef tile(v):\n    return v * LIMIT\n", "relative import"),
            ("from pathlib import Path\n\nDATA = Path(__file__).parent\n\n\ndef tile(v):\n    return v\n", "__file__"),
            ("from os.path import *\n\n\ndef tile(v):\n    return v\n", "star import"),
            ("def tile(v):\n    return v\n\n\nif __name__ == '__main__':\n    tile(1)\n", "If statement"),
            ("try:\n    import json\nexcept ImportError:\n    json = None\n\n\ndef tile(v):\n    return v\n", "Try statement"),
            ("(LIMIT := 4)\n\n\ndef tile(v):\n    return v * LIMIT\n", "assignment expression"),
        ],
    )
    def test_modules_that_cannot_be_relocated_are_skipped(self, tmp_path, body, reason_fragment):
        _make_tiles_tree(tmp_path, body)

        plan = _engine(tmp_path).build_plan()

        assert plan.duplicates == {}
        assert any(reason_fragment in reason for reason in plan.skipped.values()), plan.skipped

    def test_module_exports_cover_every_static_binding(self):
        from transformation_portal.dev.refactor_engine import _module_exports, _relocation_blocker

        source = (
            '"""Doc."""\n'
            "from __future__ import annotations\n"
            "import os.path\n"
            "import json as _json\n"
            "from typing import TYPE_CHECKING, Any as _Any\n"
            "if TYPE_CHECKING:\n"
            "    from collections.abc import Sequence\n"
            "first, (second, *rest) = 1, (2, 3)\n"
            "os.environ['UNUSED'] = 'x'\n"
            "ANNOTATED: int = 1\n"
            "DECLARED_ONLY: int\n"
            "__all__ = ['Klass']\n"
            "__all__ += ['run']\n"
            "class Klass:\n"
            "    inner = 1\n"
            "async def run():\n"
            "    local = 1\n"
        )
        tree = ast.parse(source)

        assert _relocation_blocker(tree) is None
        assert _module_exports(tree) == [
            "ANNOTATED",
            "Klass",
            "TYPE_CHECKING",
            "_Any",
            "__all__",
            "_json",
            "first",
            "os",
            "rest",
            "run",
            "second",
        ]

    @pytest.mark.parametrize(
        "prologue",
        [
            ["#!/usr/bin/env python", "# -*- coding: utf-8 -*-"],
            ["# Copyright notice", "# -*- coding: utf-8 -*-"],
            ["# -*- coding: utf-8 -*-"],
            ["#!/usr/bin/env python"],
            [],
        ],
    )
    def test_shared_module_keeps_shebang_and_encoding_declaration_first(self, tmp_path, prologue):
        source = "".join(line + "\n" for line in prologue) + '"""Doc."""\nfrom __future__ import annotations\n'
        engine = _engine(tmp_path)

        rendered = engine._render_shared_module(source, "f" * 64, tmp_path / "module.py")

        lines = rendered.splitlines()
        assert lines[: len(prologue)] == prologue
        assert lines[len(prologue)].startswith("# Canonical implementation (hash: ffffffffffffffff)")
        assert lines[len(prologue) + 1] == "# Source: module.py"
        assert lines[len(prologue) + 2] == '"""Doc."""'
        compile(rendered, "<rendered>", "exec", dont_inherit=True)
