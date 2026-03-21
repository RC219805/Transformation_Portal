"""Tests for the auto-refactoring engine."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


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
