#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for test_markers module (ADR-044).

This module provides regression protection for the test marker retrofit utilities.
It validates:
- Pure transformation behavior of add_pytest_import() and add_pytestmark()
- File-level orchestration via process_file()
- CLI contract via main()

The tests use exact output comparison for high-risk scenarios to catch subtle
formatting/placement regressions that could break test files.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

# Import from the src package (architectural fix: tests should not import scripts)
from transformation_portal.dev.test_markers import (
    add_pytest_import,
    add_pytestmark,
    get_directory_marker,
    has_class_or_function_markers,
    has_existing_module_markers,
    has_test_functions,
    main,
    process_file,
)

pytestmark = pytest.mark.unit


def _src(text: str) -> str:
    """Dedent and strip leading newlines from inline source text."""
    return textwrap.dedent(text).lstrip("\n")


# =============================================================================
# Pure Transformation Tests: add_pytest_import()
# =============================================================================


class TestAddPytestImport:
    """Tests for add_pytest_import() transformation."""

    def test_handles_multiline_from_import(self) -> None:
        """Validate pytest import is placed AFTER multi-line from imports.

        This is the primary regression test for the bug in PR #1242.
        The import must not be inserted inside the parenthesized import block.
        Uses exact-output assertion for consistency with other high-risk tests.
        """
        content = _src("""
            from pkg.mod import (
                A,
                B,
            )

            def test_example():
                assert True
            """)

        result = add_pytest_import(content)

        # Exact output validation - import pytest must be placed after closing paren
        expected = _src("""
            from pkg.mod import (
                A,
                B,
            )
            import pytest

            def test_example():
                assert True
            """)
        assert result == expected

    def test_handles_simple_import(self) -> None:
        """Validate pytest import is placed after simple import statement."""
        content = _src("""
            import os

            def test_example():
                assert True
            """)

        result = add_pytest_import(content)

        expected = _src("""
            import os
            import pytest

            def test_example():
                assert True
            """)
        assert result == expected

    def test_handles_from_import(self) -> None:
        """Validate pytest import is placed after from import statement."""
        content = _src("""
            from pathlib import Path

            def test_example():
                assert True
            """)

        result = add_pytest_import(content)

        expected = _src("""
            from pathlib import Path
            import pytest

            def test_example():
                assert True
            """)
        assert result == expected

    def test_handles_multiple_imports(self) -> None:
        """Validate pytest import is placed after last import."""
        content = _src("""
            import os
            import sys
            from pathlib import Path

            def test_example():
                assert True
            """)

        result = add_pytest_import(content)

        expected = _src("""
            import os
            import sys
            from pathlib import Path
            import pytest

            def test_example():
                assert True
            """)
        assert result == expected

    @pytest.mark.xfail(
        reason=(
            "add_pytest_import uses line-based docstring handling that fails "
            "to correctly place import after multi-line docstrings. "
            "The intended behavior is import AFTER docstring, not BEFORE."
        ),
        strict=True,
    )
    def test_handles_no_imports_with_docstring(self) -> None:
        """Validate pytest import is placed AFTER module docstring when no imports.

        INTENDED BEHAVIOR: import pytest should be placed after the docstring,
        not before it. The current implementation has a limitation in the
        line-based docstring detection that causes this to fail.

        When the script is fixed to use AST-based docstring detection
        (like add_pytestmark does), this test should pass and the xfail
        marker can be removed.
        """
        content = _src('''
            """Test module for X."""

            def test_example():
                assert True
            ''')

        result = add_pytest_import(content)

        # INTENDED: import pytest should appear AFTER the docstring
        expected = _src('''
            """Test module for X."""
            import pytest

            def test_example():
                assert True
            ''')
        assert result == expected

    def test_handles_no_imports_no_docstring(self) -> None:
        """Validate pytest import is placed at the start when no imports or docstring."""
        content = _src("""
            def test_example():
                assert True
            """)

        result = add_pytest_import(content)

        expected = _src("""
            import pytest
            def test_example():
                assert True
            """)
        assert result == expected

    def test_handles_future_import(self) -> None:
        """Validate pytest import is placed after __future__ and other imports.

        Uses exact-output assertion since __future__ positioning is structurally sensitive.
        """
        content = _src("""
            from __future__ import annotations

            from pkg import x

            def test_example():
                assert True
            """)

        result = add_pytest_import(content)

        # Exact output validation - __future__ must stay first, import pytest after last import
        expected = _src("""
            from __future__ import annotations

            from pkg import x
            import pytest

            def test_example():
                assert True
            """)
        assert result == expected

    def test_handles_multiline_with_trailing_comma(self) -> None:
        """Validate proper handling of multi-line import with trailing comma.

        This uses exact-output assertion to catch any whitespace/placement drift.
        """
        content = _src("""
            from pkg.mod import (
                A,
                B,
                C,
            )

            def test_something():
                pass
            """)

        result = add_pytest_import(content)

        # Exact output validation - import pytest must be after closing paren
        expected = _src("""
            from pkg.mod import (
                A,
                B,
                C,
            )
            import pytest

            def test_something():
                pass
            """)
        assert result == expected


# =============================================================================
# Pure Transformation Tests: add_pytestmark()
# =============================================================================


class TestAddPytestmark:
    """Tests for add_pytestmark() transformation."""

    def test_places_marker_after_multiline_imports(self) -> None:
        """Validate pytestmark is placed after multi-line imports.

        This uses exact-output assertion to catch any whitespace/placement drift.
        NOTE: Double blank lines before pytestmark reflect current implementation
        spacing behavior, not necessarily the ideal long-term formatting contract.
        """
        content = _src("""
            from pkg.mod import (
                A,
                B,
            )
            import pytest

            def test_example():
                assert True
            """)

        result = add_pytestmark(content, ["unit"])

        # Exact output validation - the script adds double newline before pytestmark
        expected = _src("""
            from pkg.mod import (
                A,
                B,
            )
            import pytest


            pytestmark = pytest.mark.unit

            def test_example():
                assert True
            """)
        assert result == expected

    def test_places_marker_after_simple_imports(self) -> None:
        """Validate pytestmark is placed after simple imports.

        Uses exact-output assertion to catch whitespace/placement drift.
        """
        content = _src("""
            import os
            import pytest

            def test_example():
                assert True
            """)

        result = add_pytestmark(content, ["unit"])

        # Exact output validation
        expected = _src("""
            import os
            import pytest


            pytestmark = pytest.mark.unit

            def test_example():
                assert True
            """)
        assert result == expected

    def test_places_marker_after_docstring_and_existing_pytest_import(self) -> None:
        """Validate pytestmark is placed after docstring when import pytest already exists.

        Uses exact-output assertion to catch whitespace/placement drift.
        NOTE: Double blank lines before pytestmark reflect current implementation
        spacing behavior, not necessarily the ideal long-term formatting contract.
        """
        content = _src('''
            """Test module for X."""
            import pytest

            def test_example():
                assert True
            ''')

        result = add_pytestmark(content, ["unit"])

        # Exact output validation
        expected = _src('''
            """Test module for X."""
            import pytest


            pytestmark = pytest.mark.unit

            def test_example():
                assert True
            ''')
        assert result == expected

    def test_places_marker_after_existing_pytest_import(self) -> None:
        """Validate pytestmark placement when only import pytest exists (no docstring).

        Uses exact-output assertion to catch whitespace/placement drift.
        NOTE: Double blank lines before pytestmark reflect current implementation
        spacing behavior, not necessarily the ideal long-term formatting contract.
        """
        content = _src("""
            import pytest

            def test_example():
                assert True
            """)

        result = add_pytestmark(content, ["unit"])

        # Exact output validation
        expected = _src("""
            import pytest


            pytestmark = pytest.mark.unit

            def test_example():
                assert True
            """)
        assert result == expected

    def test_handles_multiple_markers(self) -> None:
        """Validate multiple markers are formatted as a list.

        Uses exact-output assertion to catch whitespace/placement drift.
        """
        content = _src("""
            import pytest

            def test_example():
                assert True
            """)

        result = add_pytestmark(content, ["stress", "slow"])

        # Exact output validation
        expected = _src("""
            import pytest


            pytestmark = [pytest.mark.stress, pytest.mark.slow]

            def test_example():
                assert True
            """)
        assert result == expected

    def test_preserves_import_comments(self) -> None:
        """Validate comments around imports are preserved."""
        content = _src('''
            """module doc"""
            import pytest

            import os
            import sys

            # third-party
            from PIL import Image

            # local imports
            from app.tools import run

            def test_example():
                assert True
            ''')

        result = add_pytestmark(content, ["unit"])

        # Verify comments are preserved (not removed or displaced)
        assert "# third-party" in result
        assert "# local imports" in result
        # Verify pytestmark is placed after imports
        assert "\npytestmark = pytest.mark.unit\n\n" in result


# =============================================================================
# File-Level Orchestration Tests: process_file()
# =============================================================================


class TestProcessFile:
    """Tests for process_file() orchestration."""

    @pytest.mark.xfail(
        reason=(
            "add_pytest_import has a known limitation with docstring-only files. "
            "It places 'import pytest' BEFORE the docstring instead of after. "
            "This is a line-based fallback issue when no imports are found."
        ),
        strict=True,
    )
    def test_handles_module_docstring_without_imports(self, tmp_path: Path) -> None:
        """Validate process_file handles docstring-only files correctly.

        INTENDED BEHAVIOR: The import and pytestmark should appear AFTER the
        docstring, not before it. This is the correct Python style.

        When add_pytest_import is fixed to use AST-based docstring detection,
        this test should pass and the xfail marker can be removed.
        """
        file_path = tmp_path / "test_docstring_case.py"
        file_path.write_text(
            _src('''
                """Test module for X."""

                def test_example():
                    assert True
                '''),
            encoding="utf-8",
        )

        modified, reason = process_file(file_path, dry_run=False)

        assert modified is True
        assert "@pytest.mark.unit" in reason

        result = file_path.read_text(encoding="utf-8")
        # INTENDED exact output: docstring first, then import, then marker
        expected = _src('''
            """Test module for X."""
            import pytest


            pytestmark = pytest.mark.unit

            def test_example():
                assert True
            ''')
        assert result == expected

    def test_preserves_future_import_order(self, tmp_path: Path) -> None:
        """Validate __future__ imports remain first executable statement.

        Uses exact-output assertion to catch any whitespace/placement drift.
        """
        file_path = tmp_path / "test_future_imports.py"
        file_path.write_text(
            _src("""
                from __future__ import annotations

                from pkg import x

                def test_example():
                    assert True
                """),
            encoding="utf-8",
        )

        modified, _ = process_file(file_path, dry_run=False)
        assert modified is True

        result = file_path.read_text(encoding="utf-8")
        # Exact output validation
        expected = _src("""
            from __future__ import annotations

            from pkg import x
            import pytest


            pytestmark = pytest.mark.unit

            def test_example():
                assert True
            """)
        assert result == expected

    def test_handles_mixed_import_groups_with_comments(self, tmp_path: Path) -> None:
        """Validate process_file preserves import group structure with comments.

        Uses exact-output assertion to catch any whitespace/placement drift.
        """
        file_path = tmp_path / "test_import_groups.py"
        file_path.write_text(
            _src('''
                """module doc"""

                import os
                import sys

                # third-party
                from PIL import Image

                # local imports
                from app.tools import run

                def test_example():
                    assert True
                '''),
            encoding="utf-8",
        )

        modified, _ = process_file(file_path, dry_run=False)
        assert modified is True

        result = file_path.read_text(encoding="utf-8")
        # Exact output validation
        expected = _src('''
            """module doc"""

            import os
            import sys

            # third-party
            from PIL import Image

            # local imports
            from app.tools import run
            import pytest


            pytestmark = pytest.mark.unit

            def test_example():
                assert True
            ''')
        assert result == expected

    def test_skips_existing_pytestmark(self, tmp_path: Path) -> None:
        """Validate process_file skips files with existing pytestmark."""
        file_path = tmp_path / "test_existing_marker.py"
        original = _src("""
            import pytest

            pytestmark = pytest.mark.unit

            def test_example():
                assert True
            """)
        file_path.write_text(original, encoding="utf-8")

        modified, reason = process_file(file_path, dry_run=False)

        assert modified is False
        assert reason == "already has module-level markers"
        # Verify file content is unchanged
        assert file_path.read_text(encoding="utf-8") == original

    def test_handles_no_imports(self, tmp_path: Path) -> None:
        """Validate process_file handles files with no imports."""
        file_path = tmp_path / "test_no_imports.py"
        file_path.write_text(
            _src("""
                def test_example():
                    assert True
                """),
            encoding="utf-8",
        )

        modified, _ = process_file(file_path, dry_run=False)
        assert modified is True

        expected = _src("""
            import pytest

            pytestmark = pytest.mark.unit

            def test_example():
                assert True
            """)
        assert file_path.read_text(encoding="utf-8") == expected

    def test_handles_multiline_from_import(self, tmp_path: Path) -> None:
        """Validate full transform of multi-line import file.

        This is the GOLDEN CASE for the PR #1242 bug fix.
        Uses exact-output assertion - this is the primary regression test.
        """
        file_path = tmp_path / "test_multiline_import.py"
        file_path.write_text(
            _src("""
                from pkg.mod import (
                    A,
                    B,
                )

                def test_example():
                    assert True
                """),
            encoding="utf-8",
        )

        modified, reason = process_file(file_path, dry_run=False)

        assert modified is True
        assert "@pytest.mark.unit" in reason

        result = file_path.read_text(encoding="utf-8")
        # EXACT output validation - this is the primary regression test
        expected = _src("""
            from pkg.mod import (
                A,
                B,
            )
            import pytest


            pytestmark = pytest.mark.unit

            def test_example():
                assert True
            """)
        assert result == expected

    def test_does_not_duplicate_existing_pytest_import(self, tmp_path: Path) -> None:
        """Validate process_file doesn't add duplicate import pytest."""
        file_path = tmp_path / "test_existing_import.py"
        file_path.write_text(
            _src("""
                import pytest
                import os

                def test_example():
                    assert True
                """),
            encoding="utf-8",
        )

        modified, _ = process_file(file_path, dry_run=False)
        assert modified is True

        result = file_path.read_text(encoding="utf-8")
        # Count occurrences of "import pytest"
        assert result.count("import pytest") == 1

    def test_skips_file_with_class_function_markers(self, tmp_path: Path) -> None:
        """Validate process_file skips files where decorators are detected.

        Note: The current MODULE_MARKER_PATTERN regex matches @pytest.mark at
        the beginning of any line, which includes function/class decorators.
        This causes the file to be skipped with 'already has module-level markers'.
        This is actually a more conservative (safer) behavior that prevents
        double-marking files that may already have intentional markers.
        """
        file_path = tmp_path / "test_decorated.py"
        original = _src("""
            import pytest

            @pytest.mark.slow
            def test_example():
                assert True
            """)
        file_path.write_text(original, encoding="utf-8")

        modified, reason = process_file(file_path, dry_run=False)

        # File should be skipped (not modified)
        assert modified is False
        # The reason may be 'already has module-level markers' due to regex matching
        # or 'class/function markers' - either is acceptable for skipping
        assert "markers" in reason

    def test_skips_non_test_file(self, tmp_path: Path) -> None:
        """Validate process_file rejects non-test files."""
        file_path = tmp_path / "example_helpers.py"
        file_path.write_text(
            _src("""
                def helper():
                    return True
                """),
            encoding="utf-8",
        )

        modified, reason = process_file(file_path, dry_run=False)

        assert modified is False
        assert reason == "not a test file"

    def test_skips_fixture_directory(self, tmp_path: Path) -> None:
        """Validate process_file skips files in fixture directories."""
        fixtures_dir = tmp_path / "fixtures"
        fixtures_dir.mkdir()
        file_path = fixtures_dir / "test_x.py"
        file_path.write_text(
            _src("""
                def test_example():
                    assert True
                """),
            encoding="utf-8",
        )

        modified, reason = process_file(file_path, dry_run=False)

        assert modified is False
        assert "in skip directory: fixtures" in reason

    def test_skips_data_directory(self, tmp_path: Path) -> None:
        """Validate process_file skips files in data directories."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        file_path = data_dir / "test_x.py"
        file_path.write_text(
            _src("""
                def test_example():
                    assert True
                """),
            encoding="utf-8",
        )

        modified, reason = process_file(file_path, dry_run=False)

        assert modified is False
        assert "in skip directory: data" in reason

    def test_handles_syntax_error_file(self, tmp_path: Path) -> None:
        """Validate process_file gracefully handles syntax errors."""
        file_path = tmp_path / "test_broken.py"
        file_path.write_text(
            _src("""
                def test_example(
                    # Missing closing parenthesis
                """),
            encoding="utf-8",
        )

        modified, reason = process_file(file_path, dry_run=False)

        assert modified is False
        assert reason == "no test functions"

    def test_dry_run_does_not_modify_file(self, tmp_path: Path) -> None:
        """Validate dry_run=True does not modify file."""
        file_path = tmp_path / "test_dry_run.py"
        original = _src("""
            def test_example():
                assert True
            """)
        file_path.write_text(original, encoding="utf-8")

        modified, reason = process_file(file_path, dry_run=True)

        assert modified is True
        assert "@pytest.mark.unit" in reason
        # Verify file was NOT modified
        assert file_path.read_text(encoding="utf-8") == original

    def test_skips_file_without_test_functions(self, tmp_path: Path) -> None:
        """Validate process_file skips files without test functions."""
        file_path = tmp_path / "test_no_tests.py"
        file_path.write_text(
            _src("""
                def helper():
                    return True

                class NotATest:
                    def method(self):
                        pass
                """),
            encoding="utf-8",
        )

        modified, reason = process_file(file_path, dry_run=False)

        assert modified is False
        assert reason == "no test functions"


# =============================================================================
# Directory Marker Determination Tests
# =============================================================================


class TestGetDirectoryMarker:
    """Tests for get_directory_marker()."""

    def test_unit_directory(self, tmp_path: Path) -> None:
        """Validate unit directory gets unit marker."""
        file_path = tmp_path / "unit" / "test_example.py"
        assert get_directory_marker(file_path) == ["unit"]

    def test_smoke_directory(self, tmp_path: Path) -> None:
        """Validate smoke directory gets unit marker per ADR-044."""
        file_path = tmp_path / "smoke" / "test_example.py"
        assert get_directory_marker(file_path) == ["unit"]

    def test_security_directory(self, tmp_path: Path) -> None:
        """Validate security directory gets security marker."""
        file_path = tmp_path / "security" / "test_example.py"
        assert get_directory_marker(file_path) == ["security"]

    def test_integration_directory(self, tmp_path: Path) -> None:
        """Validate integration directory gets integration marker."""
        file_path = tmp_path / "integration" / "test_example.py"
        assert get_directory_marker(file_path) == ["integration"]

    def test_benchmarks_directory(self, tmp_path: Path) -> None:
        """Validate benchmarks directory gets benchmark marker."""
        file_path = tmp_path / "benchmarks" / "test_example.py"
        assert get_directory_marker(file_path) == ["benchmark"]

    def test_stress_directory(self, tmp_path: Path) -> None:
        """Validate stress directory gets both stress and slow markers."""
        file_path = tmp_path / "stress" / "test_example.py"
        assert get_directory_marker(file_path) == ["stress", "slow"]

    def test_golden_directory(self, tmp_path: Path) -> None:
        """Validate golden directory gets golden marker."""
        file_path = tmp_path / "golden" / "test_example.py"
        assert get_directory_marker(file_path) == ["golden"]

    def test_default_directory(self, tmp_path: Path) -> None:
        """Validate unknown directory gets default unit marker."""
        file_path = tmp_path / "unknown" / "test_example.py"
        assert get_directory_marker(file_path) == ["unit"]


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHasTestFunctions:
    """Tests for has_test_functions()."""

    def test_detects_test_function(self) -> None:
        """Validate detection of test_ prefixed function."""
        content = _src("""
            def test_example():
                assert True
            """)
        assert has_test_functions(content) is True

    def test_detects_test_function_in_class(self) -> None:
        """Validate detection of test_ method in class."""
        content = _src("""
            class TestExample:
                def test_method(self):
                    assert True
            """)
        assert has_test_functions(content) is True

    def test_no_test_functions(self) -> None:
        """Validate returns False when no test functions."""
        content = _src("""
            def helper():
                return True
            """)
        assert has_test_functions(content) is False

    def test_handles_syntax_error(self) -> None:
        """Validate returns False for syntax errors."""
        content = _src("""
            def test_broken(
            """)
        assert has_test_functions(content) is False


class TestHasExistingModuleMarkers:
    """Tests for has_existing_module_markers()."""

    def test_detects_pytestmark_assignment(self) -> None:
        """Validate detection of pytestmark = ..."""
        content = _src("""
            import pytest

            pytestmark = pytest.mark.unit
            """)
        assert has_existing_module_markers(content) is True

    def test_detects_module_level_decorator(self) -> None:
        """Validate detection of @pytest.mark at module level."""
        content = _src("""
            import pytest

            @pytest.mark.slow
            """)
        assert has_existing_module_markers(content) is True

    def test_no_markers(self) -> None:
        """Validate returns False when no markers present."""
        content = _src("""
            import pytest

            def test_example():
                assert True
            """)
        assert has_existing_module_markers(content) is False

    def test_module_marker_regex_currently_matches_function_decorators(self) -> None:
        """Document that MODULE_MARKER_PATTERN overmatches function decorators.

        KNOWN LIMITATION: The regex r'^@pytest\\.mark\\.\\w+' matches ANY line
        starting with @pytest.mark, including function/class decorators that are
        indented with zero spaces. This means has_existing_module_markers()
        returns True even when the marker is on a function, not at module level.

        This causes process_file() to skip files with 'already has module-level
        markers' when they actually only have function-level markers.

        This test documents the current behavior. A future fix could:
        1. Use AST to distinguish module-level decorators from function decorators
        2. Or change the regex to require pytestmark = ... (stricter pattern)
        """
        # Content with @pytest.mark on a FUNCTION (not at module level)
        content = _src("""
            import pytest

            @pytest.mark.slow
            def test_example():
                assert True
            """)

        # CURRENT BEHAVIOR: Returns True even though this is a function decorator
        # This is an overmatch - the decorator is on a function, not at module level
        result = has_existing_module_markers(content)
        assert result is True, (
            "Expected True due to MODULE_MARKER_PATTERN overmatch. "
            "If this fails, the regex may have been fixed - update this test."
        )


class TestHasClassOrFunctionMarkers:
    """Tests for has_class_or_function_markers()."""

    def test_detects_function_marker(self) -> None:
        """Validate detection of @pytest.mark on function."""
        content = _src("""
            import pytest

            @pytest.mark.slow
            def test_example():
                assert True
            """)
        assert has_class_or_function_markers(content) is True

    def test_detects_class_marker(self) -> None:
        """Validate detection of @pytest.mark on class."""
        content = _src("""
            import pytest

            @pytest.mark.unit
            class TestExample:
                def test_method(self):
                    pass
            """)
        assert has_class_or_function_markers(content) is True

    def test_detects_parametrize_marker(self) -> None:
        """Validate detection of @pytest.mark.parametrize()."""
        content = _src("""
            import pytest

            @pytest.mark.parametrize("x", [1, 2, 3])
            def test_example(x):
                assert x > 0
            """)
        assert has_class_or_function_markers(content) is True

    def test_no_markers(self) -> None:
        """Validate returns False when no markers present."""
        content = _src("""
            import pytest

            def test_example():
                assert True
            """)
        assert has_class_or_function_markers(content) is False

    def test_handles_syntax_error(self) -> None:
        """Validate returns False for syntax errors."""
        content = _src("""
            def test_broken(
            """)
        assert has_class_or_function_markers(content) is False


# =============================================================================
# CLI Contract Tests: main()
# =============================================================================


class TestMain:
    """Tests for main() CLI contract."""

    def test_rejects_missing_mode(self) -> None:
        """Validate CLI requires --dry-run or --apply."""
        with pytest.raises(SystemExit) as excinfo:
            main([])

        assert excinfo.value.code == 2

    def test_rejects_both_apply_and_dry_run(self) -> None:
        """Validate CLI rejects both --apply and --dry-run together."""
        with pytest.raises(SystemExit) as excinfo:
            main(["--apply", "--dry-run"])

        assert excinfo.value.code == 2

    def test_dry_run_mode_returns_zero(self, tmp_path: Path) -> None:
        """Validate --dry-run returns 0 on success."""
        # Create a test file to process
        file_path = tmp_path / "test_example.py"
        file_path.write_text(
            _src("""
                def test_example():
                    assert True
                """),
            encoding="utf-8",
        )

        result = main(["--dry-run", str(tmp_path)])

        assert result == 0

    def test_apply_mode_returns_zero(self, tmp_path: Path) -> None:
        """Validate --apply returns 0 on success."""
        # Create a test file to process
        file_path = tmp_path / "test_example.py"
        file_path.write_text(
            _src("""
                def test_example():
                    assert True
                """),
            encoding="utf-8",
        )

        result = main(["--apply", str(tmp_path)])

        assert result == 0

    def test_accepts_multiple_paths(self, tmp_path: Path) -> None:
        """Validate CLI accepts multiple paths."""
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()

        file1 = dir1 / "test_a.py"
        file2 = dir2 / "test_b.py"
        file1.write_text("def test_a(): pass", encoding="utf-8")
        file2.write_text("def test_b(): pass", encoding="utf-8")

        result = main(["--dry-run", str(dir1), str(dir2)])

        assert result == 0

    def test_accepts_file_path(self, tmp_path: Path) -> None:
        """Validate CLI accepts a single file path."""
        file_path = tmp_path / "test_single.py"
        file_path.write_text("def test_single(): pass", encoding="utf-8")

        result = main(["--dry-run", str(file_path)])

        assert result == 0

    def test_empty_directory_returns_zero(self, tmp_path: Path) -> None:
        """Validate CLI returns 0 even with empty directory."""
        result = main(["--dry-run", str(tmp_path)])

        assert result == 0

    def test_dry_run_output_includes_prefix(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Validate --dry-run output includes [DRY-RUN] prefix."""
        file_path = tmp_path / "test_example.py"
        file_path.write_text("def test_example(): pass", encoding="utf-8")

        main(["--dry-run", str(tmp_path)])

        captured = capsys.readouterr()
        assert "[DRY-RUN] Modified:" in captured.out

    def test_summary_output_includes_total_files(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Validate summary output includes file count."""
        file_path = tmp_path / "test_example.py"
        file_path.write_text("def test_example(): pass", encoding="utf-8")

        main(["--dry-run", str(tmp_path)])

        captured = capsys.readouterr()
        assert "Total files scanned:" in captured.out
        assert "DRY-RUN SUMMARY" in captured.out

    def test_empty_directory_summary(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Validate summary reports zero files for empty directory."""
        main(["--dry-run", str(tmp_path)])

        captured = capsys.readouterr()
        assert "Total files scanned: 0" in captured.out
        assert "Files would be modified: 0" in captured.out


# =============================================================================
# End-to-End Scenario Tests
# =============================================================================


class TestEndToEndScenarios:
    """End-to-end scenario tests covering full transformations."""

    def test_full_transform_with_future_and_docstring(self, tmp_path: Path) -> None:
        """Validate complete transformation with __future__ import and docstring.

        Uses exact-output assertion to catch any whitespace/placement drift.
        """
        file_path = tmp_path / "test_complete.py"
        file_path.write_text(
            _src('''
                """A test module with all common elements."""

                from __future__ import annotations

                import os
                from pathlib import Path

                def test_example():
                    assert True
                '''),
            encoding="utf-8",
        )

        modified, _ = process_file(file_path, dry_run=False)
        assert modified is True

        result = file_path.read_text(encoding="utf-8")
        # Exact output validation
        expected = _src('''
            """A test module with all common elements."""

            from __future__ import annotations

            import os
            from pathlib import Path
            import pytest


            pytestmark = pytest.mark.unit

            def test_example():
                assert True
            ''')
        assert result == expected

    def test_stress_directory_gets_both_markers(self, tmp_path: Path) -> None:
        """Validate stress directory files get both stress and slow markers."""
        stress_dir = tmp_path / "stress"
        stress_dir.mkdir()
        file_path = stress_dir / "test_stress.py"
        file_path.write_text(
            _src("""
                def test_large_batch():
                    assert True
                """),
            encoding="utf-8",
        )

        modified, reason = process_file(file_path, dry_run=False)
        assert modified is True
        assert "@pytest.mark.stress" in reason
        assert "@pytest.mark.slow" in reason

        result = file_path.read_text(encoding="utf-8")
        assert "pytestmark = [pytest.mark.stress, pytest.mark.slow]" in result

    @pytest.mark.xfail(
        reason=(
            "The script has no explicit shebang/coding-cookie handling. "
            "add_pytest_import does not detect shebang lines (#!/...) or "
            "encoding declarations (# -*- coding: ...). The intended behavior "
            "is: shebang stays line 1, coding cookie stays line 2, then imports."
        ),
        strict=True,
    )
    def test_preserves_shebang_and_encoding(self, tmp_path: Path) -> None:
        """Validate that shebang and encoding declarations are preserved correctly.

        INTENDED BEHAVIOR:
        - Shebang (#!/usr/bin/env python3) must remain on line 1
        - Encoding declaration must remain on line 2
        - import pytest should be placed AFTER these special lines

        When the script is fixed to handle shebang/encoding declarations,
        this test should pass and the xfail marker can be removed.
        """
        file_path = tmp_path / "test_shebang.py"
        file_path.write_text(
            _src('''
                #!/usr/bin/env python3
                # -*- coding: utf-8 -*-
                """Test module."""

                def test_example():
                    assert True
                '''),
            encoding="utf-8",
        )

        modified, _ = process_file(file_path, dry_run=False)
        assert modified is True

        result = file_path.read_text(encoding="utf-8")
        # INTENDED exact output: shebang line 1, encoding line 2, docstring, then imports
        expected = _src('''
            #!/usr/bin/env python3
            # -*- coding: utf-8 -*-
            """Test module."""
            import pytest


            pytestmark = pytest.mark.unit

            def test_example():
                assert True
            ''')
        assert result == expected

    def test_handles_large_multiline_import_block(self, tmp_path: Path) -> None:
        """Validate handling of large multi-line import blocks.

        Uses exact-output assertion to catch any whitespace/placement drift.
        (Renamed from test_handles_complex_multiline_import for precision.)
        """
        file_path = tmp_path / "test_large_import.py"
        file_path.write_text(
            _src("""
                from typing import (
                    TYPE_CHECKING,
                    Any,
                    Callable,
                    Dict,
                    List,
                    Optional,
                    Tuple,
                    Union,
                )

                def test_example():
                    assert True
                """),
            encoding="utf-8",
        )

        modified, _ = process_file(file_path, dry_run=False)
        assert modified is True

        result = file_path.read_text(encoding="utf-8")
        # Exact output validation
        expected = _src("""
            from typing import (
                TYPE_CHECKING,
                Any,
                Callable,
                Dict,
                List,
                Optional,
                Tuple,
                Union,
            )
            import pytest


            pytestmark = pytest.mark.unit

            def test_example():
                assert True
            """)
        assert result == expected
