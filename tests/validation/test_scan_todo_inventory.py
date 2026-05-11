"""Unit tests for the TODO inventory scanner.

Tests cover:
1. Governed and ungoverned TODO detection
2. Abstract NotImplementedError pattern auto-governance (including bare raises)
3. Tokenization/parsing error surfacing
4. Exit code behavior for various scenarios
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from textwrap import dedent
from types import ModuleType

import pytest

# pylint: disable=redefined-outer-name,unnecessary-lambda

# Mark all tests in this module as unit tests (ADR-044)
pytestmark = [
    pytest.mark.unit,
]


def _load_scan_todo_inventory_module() -> ModuleType:
    """Load the script module via file path to avoid import boundary violations.

    This repo's structural rules do not allow tests to import directly from
    scripts.validation.*. Instead, we load the module dynamically by path.
    """
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "validation" / "scan_todo_inventory.py"
    module_name = "scan_todo_inventory_under_test"

    spec = importlib.util.spec_from_file_location(
        module_name,
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    # Register the module in sys.modules BEFORE executing it
    # This is required for dataclasses to work properly in Python 3.12+
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scanner_module() -> ModuleType:
    """Fixture providing the scan_todo_inventory module loaded by file path."""
    return _load_scan_todo_inventory_module()


# ============================================================================
# TEST: Governed vs Ungoverned TODO Detection
# ============================================================================


class TestGovernanceRefExtraction:
    """Tests for governance reference detection in TODOs."""

    def test_extract_phase_ref(self, scanner_module: ModuleType) -> None:
        """Phase references are extracted correctly."""
        refs = scanner_module._extract_governance_refs("# TODO(Phase 3): implement")
        assert "(Phase 3)" in refs

    def test_extract_adr_ref(self, scanner_module: ModuleType) -> None:
        """ADR references are extracted correctly."""
        refs = scanner_module._extract_governance_refs("# TODO(ADR-044): marker")
        assert "(ADR-044)" in refs

    def test_extract_issue_ref(self, scanner_module: ModuleType) -> None:
        """Issue references are extracted correctly."""
        refs = scanner_module._extract_governance_refs("# TODO(#1234): fix bug")
        assert "(#1234)" in refs

    def test_extract_owner_ref(self, scanner_module: ModuleType) -> None:
        """Owner references are extracted correctly."""
        refs = scanner_module._extract_governance_refs("# TODO(@specialist): review")
        assert "(@specialist)" in refs

    def test_no_governance_ref(self, scanner_module: ModuleType) -> None:
        """TODOs without governance refs return empty tuple."""
        refs = scanner_module._extract_governance_refs("# TODO: something to do")
        assert refs == ()


class TestTodoDetectionInPythonFiles:
    """Tests for TODO pattern detection in Python files."""

    def test_governed_todo_detected(self, tmp_path: Path, scanner_module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
        """Governed TODOs are detected and marked as governed."""
        # Monkeypatch PROJECT_ROOT to use tmp_path
        monkeypatch.setattr(scanner_module, "PROJECT_ROOT", tmp_path)

        test_file = tmp_path / "test_governed.py"
        test_file.write_text(dedent("""
            # TODO(Phase 3): implement feature
            def foo():
                pass
        """).strip())

        items, errors = scanner_module._scan_python_file(test_file)

        assert len(errors) == 0
        assert len(items) == 1
        assert items[0].has_governance_ref is True
        assert "(Phase 3)" in items[0].governance_refs

    def test_ungoverned_todo_detected(
        self, tmp_path: Path, scanner_module: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ungoverned TODOs are detected and marked as ungoverned."""
        monkeypatch.setattr(scanner_module, "PROJECT_ROOT", tmp_path)

        test_file = tmp_path / "test_ungoverned.py"
        test_file.write_text(dedent("""
            # TODO: something without a tracking ref
            def bar():
                pass
        """).strip())

        items, errors = scanner_module._scan_python_file(test_file)

        assert len(errors) == 0
        assert len(items) == 1
        assert items[0].has_governance_ref is False
        assert items[0].governance_refs == ()

    def test_fixme_and_hack_detected(
        self, tmp_path: Path, scanner_module: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FIXME and HACK patterns are detected as TODOs."""
        monkeypatch.setattr(scanner_module, "PROJECT_ROOT", tmp_path)

        test_file = tmp_path / "test_other_patterns.py"
        test_file.write_text(dedent("""
            # FIXME(#123): broken edge case
            # HACK(@dev): workaround for issue
            def baz():
                pass
        """).strip())

        items, errors = scanner_module._scan_python_file(test_file)

        assert len(errors) == 0
        assert len(items) == 2
        types = {item.todo_type.value for item in items}
        assert "FIXME" in types
        assert "HACK" in types


class TestExcludedPaths:
    """Tests for generated/ignored path exclusions."""

    def test_next_build_outputs_are_excluded(self, scanner_module: ModuleType) -> None:
        assert scanner_module._is_excluded_path(Path("web/secure-landing/.next/dev/server/chunks/a.js")) is True
        assert scanner_module._is_excluded_path(Path("web/secure-landing/.next-build-verify/server/a.js")) is True
        assert scanner_module._is_excluded_path(Path("web/secure-landing/.next-smoke-123/server/a.js")) is True


# ============================================================================
# TEST: Abstract NotImplementedError Patterns
# ============================================================================


class TestAbstractMethodPatterns:
    """Tests for abstract method NotImplementedError auto-governance."""

    def test_bare_raise_not_implemented_is_auto_governed(self, scanner_module: ModuleType) -> None:
        """Bare raise NotImplementedError (no message) is treated as auto-governed."""
        assert scanner_module._is_abstract_method_pattern("") is True

    def test_abstract_method_message_is_auto_governed(self, scanner_module: ModuleType) -> None:
        """NotImplementedError with abstract method message is auto-governed."""
        assert scanner_module._is_abstract_method_pattern("Subclasses must implement") is True
        assert scanner_module._is_abstract_method_pattern("Override in subclass") is True
        assert scanner_module._is_abstract_method_pattern("Abstract method") is True

    def test_actionable_message_is_not_auto_governed(self, scanner_module: ModuleType) -> None:
        """NotImplementedError with actionable message is not auto-governed."""
        assert scanner_module._is_abstract_method_pattern("Need to add support for X") is False
        assert scanner_module._is_abstract_method_pattern("Implement feature Y") is False

    def test_bare_raise_in_file_is_governed(
        self, tmp_path: Path, scanner_module: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Bare raise NotImplementedError in a file is detected and auto-governed."""
        monkeypatch.setattr(scanner_module, "PROJECT_ROOT", tmp_path)

        test_file = tmp_path / "test_abstract.py"
        test_file.write_text(dedent("""
            class Base:
                def must_override(self):
                    raise NotImplementedError
        """).strip())

        items, errors = scanner_module._scan_python_file(test_file)

        assert len(errors) == 0
        assert len(items) == 1
        assert items[0].todo_type.value == "NotImplementedError"
        assert items[0].has_governance_ref is True  # Auto-governed
        assert items[0].message == "(no message)"

    def test_raise_with_message_in_file(
        self, tmp_path: Path, scanner_module: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """NotImplementedError with message is detected."""
        monkeypatch.setattr(scanner_module, "PROJECT_ROOT", tmp_path)

        test_file = tmp_path / "test_implemented.py"
        test_file.write_text(dedent("""
            class Concrete:
                def todo_feature(self):
                    raise NotImplementedError("Need to implement this feature")
        """).strip())

        items, errors = scanner_module._scan_python_file(test_file)

        assert len(errors) == 0
        assert len(items) == 1
        assert items[0].todo_type.value == "NotImplementedError"
        assert items[0].has_governance_ref is False  # Not auto-governed (actionable message)
        assert "Need to implement this feature" in items[0].message


# ============================================================================
# TEST: Error Surfacing
# ============================================================================


class TestErrorSurfacing:
    """Tests for tokenization/parsing error surfacing."""

    def test_syntax_error_surfaced(self, tmp_path: Path, scanner_module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
        """Syntax errors in Python files are surfaced as scan errors."""
        monkeypatch.setattr(scanner_module, "PROJECT_ROOT", tmp_path)

        test_file = tmp_path / "test_syntax_error.py"
        test_file.write_text(dedent("""
            def broken(:
                pass
        """).strip())

        items, errors = scanner_module._scan_python_file(test_file)

        # Should have a syntax error recorded
        assert len(errors) >= 1
        assert any("syntax error" in err.lower() for err in errors)

    def test_tokenize_error_surfaced(
        self, tmp_path: Path, scanner_module: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Tokenize errors in Python files are surfaced as scan errors."""
        monkeypatch.setattr(scanner_module, "PROJECT_ROOT", tmp_path)

        test_file = tmp_path / "test_tokenize_error.py"
        # Write a file with invalid encoding continuation
        test_file.write_bytes(b'# -*- coding: utf-8 -*-\n"incomplete string\n')

        items, errors = scanner_module._scan_python_file(test_file)

        # Should have a tokenize or syntax error recorded
        assert len(errors) >= 1


# ============================================================================
# TEST: Snapshot Writing
# ============================================================================


class TestSnapshotWriting:
    """Tests for canonical scanner snapshot output."""

    def test_json_payload_includes_governance_compliance(self, scanner_module: ModuleType) -> None:
        """JSON payload includes the same compliance flag as CLI JSON output."""
        result = scanner_module.ScanResult()
        result.files_scanned = 1

        payload = scanner_module._json_payload(result)

        assert payload["summary"]["total"] == 0
        assert payload["governance_compliant"] is True

    def test_write_json_snapshot_stays_under_repo(
        self, tmp_path: Path, scanner_module: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Snapshot writer writes stable JSON under the configured repo root."""
        monkeypatch.setattr(scanner_module, "PROJECT_ROOT", tmp_path)
        payload = {"summary": {"total": 0}, "governance_compliant": True}

        snapshot_path = scanner_module._write_json_snapshot(payload, "docs/analysis/todo_scanner_snapshot.json")

        assert snapshot_path == tmp_path / "docs" / "analysis" / "todo_scanner_snapshot.json"
        assert snapshot_path.read_text(encoding="utf-8").endswith("\n")
        assert json.loads(snapshot_path.read_text(encoding="utf-8")) == payload

    def test_snapshot_path_outside_repo_rejected(
        self, tmp_path: Path, scanner_module: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Snapshot writer rejects paths outside the repository root."""
        monkeypatch.setattr(scanner_module, "PROJECT_ROOT", tmp_path)

        with pytest.raises(ValueError, match="repository root"):
            scanner_module._resolve_snapshot_path("../outside.json")

    def test_write_json_snapshot_is_atomic_on_replace_failure(
        self, tmp_path: Path, scanner_module: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Snapshot writer preserves existing content if the atomic replace fails."""
        monkeypatch.setattr(scanner_module, "PROJECT_ROOT", tmp_path)
        snapshot_dir = tmp_path / "docs" / "analysis"
        snapshot_dir.mkdir(parents=True)
        snapshot_path = snapshot_dir / "todo_scanner_snapshot.json"
        snapshot_path.write_text("existing\n", encoding="utf-8")
        original_replace = Path.replace

        def fail_replace(self: Path, target: Path) -> Path:
            if self.name.startswith(".todo_scanner_snapshot.json."):
                raise OSError("simulated replace failure")
            return original_replace(self, target)

        monkeypatch.setattr(Path, "replace", fail_replace)

        with pytest.raises(OSError, match="simulated replace failure"):
            scanner_module._write_json_snapshot({"summary": {"total": 1}}, snapshot_path.as_posix())

        assert snapshot_path.read_text(encoding="utf-8") == "existing\n"
        assert not list(snapshot_dir.glob(".todo_scanner_snapshot.json.*.tmp"))


# ============================================================================
# TEST: Exit Codes
# ============================================================================


class TestExitCodes:
    """Tests for exit code behavior in governance mode."""

    def test_exit_0_when_no_issues(self, scanner_module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
        """Exit code 0 when scan completes without ungoverned TODOs or errors."""

        class MockArgs:
            json = False
            check_governance = True

        # Create a mock result with no ungoverned items and no errors
        mock_result = scanner_module.ScanResult()
        mock_result.files_scanned = 1

        monkeypatch.setattr(scanner_module, "_parse_args", lambda: MockArgs())
        monkeypatch.setattr(scanner_module, "scan_repository", lambda: mock_result)

        exit_code = scanner_module.main()
        assert exit_code == 0

    def test_exit_1_when_ungoverned_todos(self, scanner_module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
        """Exit code 1 when ungoverned TODOs found in governance mode."""

        class MockArgs:
            json = False
            check_governance = True

        # Create a mock result with ungoverned items
        mock_result = scanner_module.ScanResult()
        mock_result.files_scanned = 1
        mock_result.items.append(
            scanner_module.TodoItem(
                path=Path("test.py"),
                lineno=1,
                todo_type=scanner_module.TodoType.TODO,
                message="ungoverned todo",
                has_governance_ref=False,
                governance_refs=(),
            )
        )

        monkeypatch.setattr(scanner_module, "_parse_args", lambda: MockArgs())
        monkeypatch.setattr(scanner_module, "scan_repository", lambda: mock_result)

        exit_code = scanner_module.main()
        assert exit_code == 1

    def test_exit_2_when_scan_errors_in_governance_mode(
        self, scanner_module: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Exit code 2 when scan errors occur in governance mode (fail closed)."""

        class MockArgs:
            json = False
            check_governance = True

        # Create a mock result with errors
        mock_result = scanner_module.ScanResult()
        mock_result.files_scanned = 1
        mock_result.errors.append("test.py: syntax error: invalid syntax")

        monkeypatch.setattr(scanner_module, "_parse_args", lambda: MockArgs())
        monkeypatch.setattr(scanner_module, "scan_repository", lambda: mock_result)

        exit_code = scanner_module.main()
        assert exit_code == 2

    def test_exit_0_when_errors_outside_governance_mode(
        self, scanner_module: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Exit code 0 when scan errors occur but not in governance mode."""

        class MockArgs:
            json = False
            check_governance = False

        # Create a mock result with errors
        mock_result = scanner_module.ScanResult()
        mock_result.files_scanned = 1
        mock_result.errors.append("test.py: syntax error: invalid syntax")

        monkeypatch.setattr(scanner_module, "_parse_args", lambda: MockArgs())
        monkeypatch.setattr(scanner_module, "scan_repository", lambda: mock_result)

        exit_code = scanner_module.main()
        assert exit_code == 0

    def test_exit_2_and_no_snapshot_when_snapshot_scan_has_errors(
        self, tmp_path: Path, scanner_module: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Snapshot writing fails closed when scan errors are present."""

        class MockArgs:
            json = False
            check_governance = False
            write_snapshot = "docs/analysis/todo_scanner_snapshot.json"

        monkeypatch.setattr(scanner_module, "PROJECT_ROOT", tmp_path)
        mock_result = scanner_module.ScanResult()
        mock_result.files_scanned = 1
        mock_result.errors.append("test.py: syntax error: invalid syntax")

        monkeypatch.setattr(scanner_module, "_parse_args", lambda: MockArgs())
        monkeypatch.setattr(scanner_module, "scan_repository", lambda: mock_result)

        exit_code = scanner_module.main()

        assert exit_code == 2
        assert not (tmp_path / "docs" / "analysis" / "todo_scanner_snapshot.json").exists()


# ============================================================================
# TEST: JS/TS TODO Detection
# ============================================================================


class TestJsTsTodoDetection:
    """Tests for TODO detection in JavaScript/TypeScript files."""

    def test_js_todo_detected(self, tmp_path: Path, scanner_module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
        """TODOs in JS files are detected."""
        monkeypatch.setattr(scanner_module, "PROJECT_ROOT", tmp_path)

        test_file = tmp_path / "test.js"
        test_file.write_text(dedent("""
            // TODO(Phase 2): implement feature
            function foo() {}
        """).strip())

        items = scanner_module._scan_js_ts_file(test_file)

        assert len(items) == 1
        assert items[0].has_governance_ref is True
        assert "(Phase 2)" in items[0].governance_refs

    def test_js_block_comment_todo_detected(
        self, tmp_path: Path, scanner_module: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """TODOs in JS block comments are detected."""
        monkeypatch.setattr(scanner_module, "PROJECT_ROOT", tmp_path)

        test_file = tmp_path / "test.ts"
        test_file.write_text(dedent("""
            /**
             * TODO(ADR-044): document this
             */
            function bar() {}
        """).strip())

        items = scanner_module._scan_js_ts_file(test_file)

        assert len(items) == 1
        assert items[0].has_governance_ref is True
        assert "(ADR-044)" in items[0].governance_refs


# ============================================================================
# TEST: Exclusion Patterns
# ============================================================================


class TestExclusionPatterns:
    """Tests for path and content exclusion."""

    def test_excluded_path_returns_true(self, scanner_module: ModuleType) -> None:
        """Paths matching exclusion patterns are excluded."""
        assert scanner_module._is_excluded_path(Path("docs/foo.py")) is True
        assert scanner_module._is_excluded_path(Path("__pycache__/module.pyc")) is True

    def test_non_excluded_path_returns_false(self, scanner_module: ModuleType) -> None:
        """Paths not matching exclusion patterns are not excluded."""
        assert scanner_module._is_excluded_path(Path("src/module.py")) is False
        assert scanner_module._is_excluded_path(Path("tests/test_foo.py")) is False

    def test_excluded_content_returns_true(self, scanner_module: ModuleType) -> None:
        """Lines with excluded patterns (false positives) are excluded."""
        assert scanner_module._is_excluded_content("TODO_REPLACE = 'secret'") is True

    def test_non_excluded_content_returns_false(self, scanner_module: ModuleType) -> None:
        """Normal TODO comments are not excluded."""
        assert scanner_module._is_excluded_content("# TODO: implement feature") is False
