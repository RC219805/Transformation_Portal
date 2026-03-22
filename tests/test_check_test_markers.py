#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contract tests for the marker audit/enforcement script (ADR-044).

This module validates the semantics of the check_test_markers.py audit script.
These tests pin the behavior that was fixed in #1245 and ensure it cannot drift.

Key contracts tested:
- Builtin markers (skip, skipif) do NOT satisfy category coverage
- Module-level category markers DO satisfy coverage
- Mixed marker lists are detected correctly
- smoke/ path maps to unit marker (not a separate smoke marker)
- Directory-based marker requirements are validated

See: docs/architecture/ADR-044-test-marker-enforcement.md
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

# Import the audit script module directly for testing
# We add the scripts directory to the path temporarily to import it
_SCRIPTS_DIR = Path(__file__).parent.parent / "scripts" / "validation"
sys.path.insert(0, str(_SCRIPTS_DIR))

from check_test_markers import (  # noqa: E402
    BUILTIN_MARKERS,
    CATEGORY_MARKERS,
    DIRECTORY_MARKER_REQUIREMENTS,
    VALID_MARKERS,
    audit_test_directory,
    check_files,
    check_marker_requirements,
    get_directory_type,
    scan_file,
)

sys.path.pop(0)  # Restore original path

pytestmark = pytest.mark.unit


def _src(text: str) -> str:
    """Dedent and strip leading newlines from inline source text."""
    return textwrap.dedent(text).lstrip("\n")


# =============================================================================
# Constants Contract Tests
# =============================================================================


class TestMarkerConstants:
    """Verify the marker constant definitions match ADR-044."""

    def test_builtin_markers_are_frozenset(self) -> None:
        """Builtin markers must be immutable frozenset."""
        assert isinstance(BUILTIN_MARKERS, frozenset)

    def test_category_markers_are_frozenset(self) -> None:
        """Category markers must be immutable frozenset."""
        assert isinstance(CATEGORY_MARKERS, frozenset)

    def test_builtin_markers_content(self) -> None:
        """Verify builtin markers match pytest built-ins."""
        expected = frozenset({"skip", "skipif", "xfail", "parametrize", "usefixtures", "filterwarnings", "timeout"})
        assert BUILTIN_MARKERS == expected

    def test_category_markers_content(self) -> None:
        """Verify category markers match pyproject.toml registration."""
        expected = frozenset(
            {
                "slow",
                "unit",
                "regression",
                "security",
                "integration",
                "ml",
                "golden",
                "stress",
                "benchmark",
            }
        )
        assert CATEGORY_MARKERS == expected

    def test_valid_markers_is_union(self) -> None:
        """VALID_MARKERS should be union of category and builtin."""
        assert VALID_MARKERS == CATEGORY_MARKERS | BUILTIN_MARKERS

    def test_builtin_and_category_are_disjoint(self) -> None:
        """Builtin and category markers must not overlap."""
        assert BUILTIN_MARKERS.isdisjoint(CATEGORY_MARKERS)


class TestDirectoryMarkerRequirements:
    """Verify directory-based marker requirements per ADR-044."""

    def test_smoke_maps_to_unit(self) -> None:
        """smoke/ directory maps to unit marker, not a separate smoke marker."""
        assert "smoke" in DIRECTORY_MARKER_REQUIREMENTS
        assert DIRECTORY_MARKER_REQUIREMENTS["smoke"] == ["unit"]

    def test_stress_requires_both_markers(self) -> None:
        """stress/ directory requires both stress AND slow markers."""
        assert "stress" in DIRECTORY_MARKER_REQUIREMENTS
        required = DIRECTORY_MARKER_REQUIREMENTS["stress"]
        assert "stress" in required
        assert "slow" in required
        assert len(required) == 2

    def test_security_directory(self) -> None:
        """security/ directory requires security marker."""
        assert DIRECTORY_MARKER_REQUIREMENTS["security"] == ["security"]

    def test_benchmarks_directory(self) -> None:
        """benchmarks/ directory requires benchmark marker."""
        assert DIRECTORY_MARKER_REQUIREMENTS["benchmarks"] == ["benchmark"]


# =============================================================================
# Core Semantics: Builtin vs Category Markers
# =============================================================================


class TestBuiltinMarkersDoNotSatisfyCoverage:
    """Critical contract: builtin markers alone do NOT satisfy category coverage.

    This is the most important semantic rule in ADR-044.
    Tests with only @pytest.mark.skip or @pytest.mark.skipif are NOT "marked".
    """

    def test_skip_only_is_violation(self, tmp_path: Path) -> None:
        """@pytest.mark.skip alone is a violation."""
        content = _src("""
            import pytest

            @pytest.mark.skip(reason="WIP")
            def test_example():
                pass
            """)
        test_file = tmp_path / "test_skip_only.py"
        test_file.write_text(content, encoding="utf-8")

        functions = scan_file(test_file)
        assert len(functions) == 1

        violation = check_marker_requirements(functions[0])
        assert violation is not None
        assert "missing required marker" in violation.issue

    def test_skipif_only_is_violation(self, tmp_path: Path) -> None:
        """@pytest.mark.skipif alone is a violation."""
        content = _src("""
            import pytest
            import sys

            @pytest.mark.skipif(sys.platform == "win32", reason="Unix only")
            def test_example():
                pass
            """)
        test_file = tmp_path / "test_skipif_only.py"
        test_file.write_text(content, encoding="utf-8")

        functions = scan_file(test_file)
        assert len(functions) == 1

        violation = check_marker_requirements(functions[0])
        assert violation is not None
        assert "missing required marker" in violation.issue

    def test_xfail_only_is_violation(self, tmp_path: Path) -> None:
        """@pytest.mark.xfail alone is a violation."""
        content = _src("""
            import pytest

            @pytest.mark.xfail(reason="Known bug")
            def test_example():
                pass
            """)
        test_file = tmp_path / "test_xfail_only.py"
        test_file.write_text(content, encoding="utf-8")

        functions = scan_file(test_file)
        assert len(functions) == 1

        violation = check_marker_requirements(functions[0])
        assert violation is not None
        assert "missing required marker" in violation.issue

    def test_parametrize_only_is_violation(self, tmp_path: Path) -> None:
        """@pytest.mark.parametrize alone is a violation."""
        content = _src("""
            import pytest

            @pytest.mark.parametrize("value", [1, 2, 3])
            def test_example(value):
                assert value > 0
            """)
        test_file = tmp_path / "test_parametrize_only.py"
        test_file.write_text(content, encoding="utf-8")

        functions = scan_file(test_file)
        assert len(functions) == 1

        violation = check_marker_requirements(functions[0])
        assert violation is not None
        assert "missing required marker" in violation.issue

    def test_multiple_builtins_still_violation(self, tmp_path: Path) -> None:
        """Multiple builtin markers together still a violation."""
        content = _src("""
            import pytest
            import sys

            @pytest.mark.skipif(sys.platform == "win32", reason="Unix only")
            @pytest.mark.parametrize("n", [1, 2])
            @pytest.mark.timeout(60)
            def test_example(n):
                pass
            """)
        test_file = tmp_path / "test_multi_builtin.py"
        test_file.write_text(content, encoding="utf-8")

        functions = scan_file(test_file)
        assert len(functions) == 1

        violation = check_marker_requirements(functions[0])
        assert violation is not None
        assert "missing required marker" in violation.issue


class TestCategoryMarkersSatisfyCoverage:
    """Category markers properly satisfy coverage requirements."""

    def test_unit_marker_satisfies(self, tmp_path: Path) -> None:
        """@pytest.mark.unit satisfies coverage."""
        content = _src("""
            import pytest

            @pytest.mark.unit
            def test_example():
                pass
            """)
        test_file = tmp_path / "test_unit.py"
        test_file.write_text(content, encoding="utf-8")

        functions = scan_file(test_file)
        assert len(functions) == 1

        violation = check_marker_requirements(functions[0])
        assert violation is None

    def test_ml_marker_satisfies(self, tmp_path: Path) -> None:
        """@pytest.mark.ml satisfies coverage."""
        content = _src("""
            import pytest

            @pytest.mark.ml
            def test_example():
                pass
            """)
        test_file = tmp_path / "test_ml.py"
        test_file.write_text(content, encoding="utf-8")

        functions = scan_file(test_file)
        assert len(functions) == 1

        violation = check_marker_requirements(functions[0])
        assert violation is None

    def test_integration_marker_satisfies(self, tmp_path: Path) -> None:
        """@pytest.mark.integration satisfies coverage."""
        content = _src("""
            import pytest

            @pytest.mark.integration
            def test_example():
                pass
            """)
        test_file = tmp_path / "test_integration.py"
        test_file.write_text(content, encoding="utf-8")

        functions = scan_file(test_file)
        assert len(functions) == 1

        violation = check_marker_requirements(functions[0])
        assert violation is None

    def test_builtin_with_category_satisfies(self, tmp_path: Path) -> None:
        """Category marker with builtin markers satisfies coverage."""
        content = _src("""
            import pytest
            import sys

            @pytest.mark.unit
            @pytest.mark.skipif(sys.platform == "win32", reason="Unix only")
            @pytest.mark.parametrize("n", [1, 2])
            def test_example(n):
                pass
            """)
        test_file = tmp_path / "test_mixed.py"
        test_file.write_text(content, encoding="utf-8")

        functions = scan_file(test_file)
        assert len(functions) == 1

        violation = check_marker_requirements(functions[0])
        assert violation is None


# =============================================================================
# Module-Level Marker Detection
# =============================================================================


class TestModuleLevelMarkerDetection:
    """Module-level pytestmark declarations satisfy coverage for all tests."""

    def test_pytestmark_single_satisfies(self, tmp_path: Path) -> None:
        """pytestmark = pytest.mark.unit satisfies all tests in module."""
        content = _src("""
            import pytest

            pytestmark = pytest.mark.unit

            def test_one():
                pass

            def test_two():
                pass
            """)
        test_file = tmp_path / "test_module.py"
        test_file.write_text(content, encoding="utf-8")

        functions = scan_file(test_file)
        assert len(functions) == 2

        for func in functions:
            violation = check_marker_requirements(func)
            assert violation is None, f"{func.name} should not have violation"

    def test_pytestmark_list_satisfies(self, tmp_path: Path) -> None:
        """pytestmark = [pytest.mark.unit, pytest.mark.skipif(...)] satisfies."""
        content = _src("""
            import pytest
            import sys

            pytestmark = [
                pytest.mark.unit,
                pytest.mark.skipif(sys.platform == "win32", reason="Unix only"),
            ]

            def test_example():
                pass
            """)
        test_file = tmp_path / "test_module_list.py"
        test_file.write_text(content, encoding="utf-8")

        functions = scan_file(test_file)
        assert len(functions) == 1

        violation = check_marker_requirements(functions[0])
        assert violation is None

    def test_pytestmark_list_only_builtins_violation(self, tmp_path: Path) -> None:
        """pytestmark = [pytest.mark.skipif(...)] alone is a violation."""
        content = _src("""
            import pytest
            import sys

            pytestmark = [
                pytest.mark.skipif(sys.platform == "win32", reason="Unix only"),
                pytest.mark.timeout(60),
            ]

            def test_example():
                pass
            """)
        test_file = tmp_path / "test_module_builtins.py"
        test_file.write_text(content, encoding="utf-8")

        functions = scan_file(test_file)
        assert len(functions) == 1

        violation = check_marker_requirements(functions[0])
        assert violation is not None
        assert "missing required marker" in violation.issue

    def test_pytestmark_mixed_category_and_builtin(self, tmp_path: Path) -> None:
        """pytestmark with category + builtins satisfies coverage."""
        content = _src("""
            import pytest
            import sys

            pytestmark = [
                pytest.mark.ml,
                pytest.mark.skipif(not sys.version_info >= (3, 11), reason="3.11+"),
                pytest.mark.slow,
            ]

            def test_one():
                pass

            def test_two():
                pass
            """)
        test_file = tmp_path / "test_mixed_module.py"
        test_file.write_text(content, encoding="utf-8")

        functions = scan_file(test_file)
        assert len(functions) == 2

        for func in functions:
            violation = check_marker_requirements(func)
            assert violation is None


# =============================================================================
# Smoke Path Mapping Tests
# =============================================================================


class TestSmokePathMapping:
    """Verify smoke/ path maps to unit marker requirement."""

    def test_smoke_directory_type_detected(self, tmp_path: Path) -> None:
        """Files in smoke/ are detected as smoke directory type."""
        test_file = tmp_path / "tests" / "smoke" / "test_example.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("# placeholder", encoding="utf-8")

        dir_type = get_directory_type(test_file)
        assert dir_type == "smoke"

    def test_smoke_requires_unit_marker(self, tmp_path: Path) -> None:
        """smoke/ directory requires unit marker (not a smoke marker)."""
        # Create file in smoke directory
        smoke_dir = tmp_path / "tests" / "smoke"
        smoke_dir.mkdir(parents=True)
        test_file = smoke_dir / "test_smoke.py"

        # File with unit marker - should pass
        content = _src("""
            import pytest

            pytestmark = pytest.mark.unit

            def test_example():
                pass
            """)
        test_file.write_text(content, encoding="utf-8")

        functions = scan_file(test_file)
        assert len(functions) == 1

        violation = check_marker_requirements(functions[0])
        assert violation is None

    def test_smoke_without_unit_is_violation(self, tmp_path: Path) -> None:
        """smoke/ without unit marker is a violation."""
        smoke_dir = tmp_path / "tests" / "smoke"
        smoke_dir.mkdir(parents=True)
        test_file = smoke_dir / "test_smoke.py"

        # File without unit marker
        content = _src("""
            import pytest

            pytestmark = pytest.mark.ml

            def test_example():
                pass
            """)
        test_file.write_text(content, encoding="utf-8")

        functions = scan_file(test_file)
        assert len(functions) == 1

        violation = check_marker_requirements(functions[0])
        assert violation is not None
        assert "smoke" in violation.issue
        assert "unit" in violation.issue


# =============================================================================
# Class-Level Marker Detection
# =============================================================================


class TestClassLevelMarkerDetection:
    """Class-level markers propagate to all test methods."""

    def test_class_marker_satisfies_methods(self, tmp_path: Path) -> None:
        """@pytest.mark.unit on class satisfies all methods."""
        content = _src("""
            import pytest

            @pytest.mark.unit
            class TestExample:
                def test_one(self):
                    pass

                def test_two(self):
                    pass
            """)
        test_file = tmp_path / "test_class.py"
        test_file.write_text(content, encoding="utf-8")

        functions = scan_file(test_file)
        assert len(functions) == 2

        for func in functions:
            violation = check_marker_requirements(func)
            assert violation is None

    def test_class_builtin_only_violation(self, tmp_path: Path) -> None:
        """@pytest.mark.skipif on class alone is a violation."""
        content = _src("""
            import pytest
            import sys

            @pytest.mark.skipif(sys.platform == "win32", reason="Unix only")
            class TestExample:
                def test_one(self):
                    pass
            """)
        test_file = tmp_path / "test_class_builtin.py"
        test_file.write_text(content, encoding="utf-8")

        functions = scan_file(test_file)
        assert len(functions) == 1

        violation = check_marker_requirements(functions[0])
        assert violation is not None

    def test_module_marker_combines_with_class(self, tmp_path: Path) -> None:
        """Module-level marker combines with class-level markers."""
        content = _src("""
            import pytest

            pytestmark = pytest.mark.unit

            @pytest.mark.slow
            class TestExample:
                def test_one(self):
                    pass
            """)
        test_file = tmp_path / "test_module_class.py"
        test_file.write_text(content, encoding="utf-8")

        functions = scan_file(test_file)
        assert len(functions) == 1

        # Should have both unit and slow markers
        func = functions[0]
        assert "unit" in func.markers
        assert "slow" in func.markers

        violation = check_marker_requirements(func)
        assert violation is None


# =============================================================================
# Pre-commit Mode Tests
# =============================================================================


class TestPreCommitMode:
    """Test the check_files() function used in pre-commit mode."""

    def test_passes_marked_file(self, tmp_path: Path) -> None:
        """Pre-commit passes file with proper markers."""
        test_file = tmp_path / "test_marked.py"
        content = _src("""
            import pytest

            pytestmark = pytest.mark.unit

            def test_example():
                pass
            """)
        test_file.write_text(content, encoding="utf-8")

        violations = check_files([test_file])
        assert len(violations) == 0

    def test_fails_unmarked_file(self, tmp_path: Path) -> None:
        """Pre-commit fails file without markers."""
        test_file = tmp_path / "test_unmarked.py"
        content = _src("""
            def test_example():
                pass
            """)
        test_file.write_text(content, encoding="utf-8")

        violations = check_files([test_file])
        assert len(violations) == 1
        assert "missing required marker" in violations[0].issue

    def test_fails_builtin_only_file(self, tmp_path: Path) -> None:
        """Pre-commit fails file with only builtin markers."""
        test_file = tmp_path / "test_builtin_only.py"
        content = _src("""
            import pytest

            pytestmark = pytest.mark.skip(reason="WIP")

            def test_example():
                pass
            """)
        test_file.write_text(content, encoding="utf-8")

        violations = check_files([test_file])
        assert len(violations) == 1

    def test_skips_non_test_files(self, tmp_path: Path) -> None:
        """Pre-commit skips files not matching test_*.py."""
        # conftest.py should be skipped
        conftest = tmp_path / "conftest.py"
        conftest.write_text("# conftest", encoding="utf-8")

        # helpers.py should be skipped
        helpers = tmp_path / "helpers.py"
        helpers.write_text("def helper(): pass", encoding="utf-8")

        violations = check_files([conftest, helpers])
        assert len(violations) == 0

    def test_skips_nonexistent_files(self, tmp_path: Path) -> None:
        """Pre-commit gracefully handles nonexistent files."""
        nonexistent = tmp_path / "test_does_not_exist.py"
        violations = check_files([nonexistent])
        assert len(violations) == 0


# =============================================================================
# Audit Mode Tests
# =============================================================================


class TestAuditMode:
    """Test the audit_test_directory() function."""

    def test_audit_empty_directory(self, tmp_path: Path) -> None:
        """Audit of empty directory returns clean report."""
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        report = audit_test_directory(tests_dir)

        assert report.total_test_functions == 0
        assert report.marked_functions == 0
        assert report.unmarked_functions == 0
        assert report.coverage_percent == 100.0

    def test_audit_all_marked(self, tmp_path: Path) -> None:
        """Audit with all marked files returns 100% coverage."""
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        test_file = tests_dir / "test_marked.py"
        test_file.write_text(
            _src("""
            import pytest

            pytestmark = pytest.mark.unit

            def test_one():
                pass

            def test_two():
                pass
            """),
            encoding="utf-8",
        )

        report = audit_test_directory(tests_dir)

        assert report.total_test_functions == 2
        assert report.marked_functions == 2
        assert report.unmarked_functions == 0
        assert report.coverage_percent == 100.0
        assert len(report.violations) == 0

    def test_audit_mixed_coverage(self, tmp_path: Path) -> None:
        """Audit with mixed coverage calculates correctly."""
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        # Marked file
        marked_file = tests_dir / "test_marked.py"
        marked_file.write_text(
            _src("""
            import pytest

            pytestmark = pytest.mark.unit

            def test_one():
                pass
            """),
            encoding="utf-8",
        )

        # Unmarked file
        unmarked_file = tests_dir / "test_unmarked.py"
        unmarked_file.write_text(
            _src("""
            def test_two():
                pass
            """),
            encoding="utf-8",
        )

        report = audit_test_directory(tests_dir)

        assert report.total_test_functions == 2
        assert report.marked_functions == 1
        assert report.unmarked_functions == 1
        assert report.coverage_percent == 50.0
        assert len(report.violations) == 1

    def test_audit_counts_marker_distribution(self, tmp_path: Path) -> None:
        """Audit tracks marker distribution."""
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        test_file = tests_dir / "test_markers.py"
        test_file.write_text(
            _src("""
            import pytest

            @pytest.mark.unit
            def test_one():
                pass

            @pytest.mark.ml
            def test_two():
                pass

            @pytest.mark.unit
            @pytest.mark.slow
            def test_three():
                pass
            """),
            encoding="utf-8",
        )

        report = audit_test_directory(tests_dir)

        assert report.marker_counts.get("unit", 0) == 2
        assert report.marker_counts.get("ml", 0) == 1
        assert report.marker_counts.get("slow", 0) == 1


# =============================================================================
# Edge Cases and Complex Scenarios
# =============================================================================


class TestEdgeCases:
    """Edge cases and complex scenarios."""

    def test_async_test_function(self, tmp_path: Path) -> None:
        """Async test functions are detected correctly."""
        content = _src("""
            import pytest

            @pytest.mark.unit
            async def test_async():
                pass
            """)
        test_file = tmp_path / "test_async.py"
        test_file.write_text(content, encoding="utf-8")

        functions = scan_file(test_file)
        assert len(functions) == 1
        assert "unit" in functions[0].markers

    def test_nested_class(self, tmp_path: Path) -> None:
        """Only top-level Test classes are scanned (pytest behavior)."""
        content = _src("""
            import pytest

            @pytest.mark.unit
            class TestOuter:
                def test_outer(self):
                    pass

                class TestNested:
                    def test_nested(self):
                        pass
            """)
        test_file = tmp_path / "test_nested.py"
        test_file.write_text(content, encoding="utf-8")

        functions = scan_file(test_file)
        # Only outer test method is detected (nested classes not visited)
        assert len(functions) == 1
        assert functions[0].name == "TestOuter.test_outer"

    def test_marker_with_call_syntax(self, tmp_path: Path) -> None:
        """Markers with call syntax are detected: @pytest.mark.unit()."""
        content = _src("""
            import pytest

            @pytest.mark.unit()
            def test_example():
                pass
            """)
        test_file = tmp_path / "test_call_syntax.py"
        test_file.write_text(content, encoding="utf-8")

        functions = scan_file(test_file)
        assert len(functions) == 1
        assert "unit" in functions[0].markers

    def test_marker_with_arguments(self, tmp_path: Path) -> None:
        """Markers with arguments are detected."""
        content = _src("""
            import pytest

            @pytest.mark.skipif(False, reason="always run")
            @pytest.mark.unit
            def test_example():
                pass
            """)
        test_file = tmp_path / "test_marker_args.py"
        test_file.write_text(content, encoding="utf-8")

        functions = scan_file(test_file)
        assert len(functions) == 1
        assert "unit" in functions[0].markers
        assert "skipif" in functions[0].markers

    def test_syntax_error_file(self, tmp_path: Path) -> None:
        """Syntax error files return empty function list."""
        content = _src("""
            def test_broken(
            """)
        test_file = tmp_path / "test_syntax_error.py"
        test_file.write_text(content, encoding="utf-8")

        functions = scan_file(test_file)
        assert len(functions) == 0

    def test_method_not_starting_with_test(self, tmp_path: Path) -> None:
        """Methods not starting with test_ are not counted."""
        content = _src("""
            import pytest

            @pytest.mark.unit
            class TestExample:
                def test_actual(self):
                    pass

                def helper_method(self):
                    pass

                def setUp(self):
                    pass
            """)
        test_file = tmp_path / "test_non_test_methods.py"
        test_file.write_text(content, encoding="utf-8")

        functions = scan_file(test_file)
        assert len(functions) == 1
        assert functions[0].name == "TestExample.test_actual"


# =============================================================================
# Directory Type Detection Tests
# =============================================================================


class TestDirectoryTypeDetection:
    """Test get_directory_type() function."""

    def test_unit_directory(self, tmp_path: Path) -> None:
        """Files in tests/unit/ detected as unit."""
        path = tmp_path / "tests" / "unit" / "test_example.py"
        assert get_directory_type(path) == "unit"

    def test_smoke_directory(self, tmp_path: Path) -> None:
        """Files in tests/smoke/ detected as smoke."""
        path = tmp_path / "tests" / "smoke" / "test_example.py"
        assert get_directory_type(path) == "smoke"

    def test_security_directory(self, tmp_path: Path) -> None:
        """Files in tests/security/ detected as security."""
        path = tmp_path / "tests" / "security" / "test_example.py"
        assert get_directory_type(path) == "security"

    def test_integration_directory(self, tmp_path: Path) -> None:
        """Files in tests/integration/ detected as integration."""
        path = tmp_path / "tests" / "integration" / "test_example.py"
        assert get_directory_type(path) == "integration"

    def test_benchmarks_directory(self, tmp_path: Path) -> None:
        """Files in tests/benchmarks/ detected as benchmarks."""
        path = tmp_path / "tests" / "benchmarks" / "test_example.py"
        assert get_directory_type(path) == "benchmarks"

    def test_stress_directory(self, tmp_path: Path) -> None:
        """Files in tests/stress/ detected as stress."""
        path = tmp_path / "tests" / "stress" / "test_example.py"
        assert get_directory_type(path) == "stress"

    def test_root_level_test(self, tmp_path: Path) -> None:
        """Root-level test files return None for directory type."""
        path = tmp_path / "tests" / "test_root.py"
        assert get_directory_type(path) is None

    def test_unknown_directory(self, tmp_path: Path) -> None:
        """Unknown subdirectory returns None."""
        path = tmp_path / "tests" / "unknown_dir" / "test_example.py"
        assert get_directory_type(path) is None


# =============================================================================
# Stress Directory Requirements Tests
# =============================================================================


class TestStressDirectoryRequirements:
    """Stress directory requires BOTH stress AND slow markers."""

    def test_stress_without_slow_is_violation(self, tmp_path: Path) -> None:
        """stress/ with only stress marker is violation (missing slow)."""
        stress_dir = tmp_path / "tests" / "stress"
        stress_dir.mkdir(parents=True)
        test_file = stress_dir / "test_stress.py"

        content = _src("""
            import pytest

            pytestmark = pytest.mark.stress

            def test_example():
                pass
            """)
        test_file.write_text(content, encoding="utf-8")

        functions = scan_file(test_file)
        assert len(functions) == 1

        violation = check_marker_requirements(functions[0])
        assert violation is not None
        assert "slow" in violation.issue

    def test_stress_with_both_markers_passes(self, tmp_path: Path) -> None:
        """stress/ with both stress AND slow markers passes."""
        stress_dir = tmp_path / "tests" / "stress"
        stress_dir.mkdir(parents=True)
        test_file = stress_dir / "test_stress.py"

        content = _src("""
            import pytest

            pytestmark = [pytest.mark.stress, pytest.mark.slow]

            def test_example():
                pass
            """)
        test_file.write_text(content, encoding="utf-8")

        functions = scan_file(test_file)
        assert len(functions) == 1

        violation = check_marker_requirements(functions[0])
        assert violation is None
