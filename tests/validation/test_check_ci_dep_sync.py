"""
Unit tests for the CI dependency sync check script.

Tests cover:
1. Symmetric drift detection (missing deps + unwanted deps)
2. Robust normalization (case, underscore/hyphen, version specifiers, comments, extras)
3. File handling (missing files raise errors)
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from textwrap import dedent

import pytest

# Import the module under test
from scripts.validation.check_ci_dep_sync import (
    CI_TOOLS,
    CORE_TEST_DEPS,
    TEST_RUNNER_PATTERN,
    extract_packages,
)


# ============================================================================
# TEST: Robust Normalization (Reviewer Point 2)
# ============================================================================


class TestExtractPackagesNormalization:
    """Tests for robust package name extraction and normalization."""

    def test_empty_file(self, tmp_path: Path) -> None:
        """Empty file returns empty set."""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text("")
        assert extract_packages(req_file) == set()

    def test_comment_only_file(self, tmp_path: Path) -> None:
        """Files with only comments return empty set."""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text(dedent("""
            # This is a comment
            # Another comment
        """).strip())
        assert extract_packages(req_file) == set()

    def test_whitespace_only_lines(self, tmp_path: Path) -> None:
        """Whitespace-only lines are skipped."""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text(dedent("""
            pytest
            
               
            httpx
        """).strip())
        packages = extract_packages(req_file)
        assert packages == {"pytest", "httpx"}

    def test_inline_comments(self, tmp_path: Path) -> None:
        """Inline comments after package names are handled correctly.
        
        Note: Current implementation extracts package name before comment
        using regex that matches alphanumeric/dots/underscores/hyphens.
        """
        req_file = tmp_path / "requirements.txt"
        req_file.write_text(dedent("""
            pytest>=8.0  # testing framework
            httpx>=0.28  # async HTTP client
        """).strip())
        packages = extract_packages(req_file)
        assert "pytest" in packages
        assert "httpx" in packages

    def test_version_specifiers_stripped(self, tmp_path: Path) -> None:
        """Version specifiers are removed from package names."""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text(dedent("""
            pytest>=8.0,<10
            httpx==0.28.1
            hypothesis~=6.0
            jsonschema>=4.21.0,<5
        """).strip())
        packages = extract_packages(req_file)
        assert packages == {"pytest", "httpx", "hypothesis", "jsonschema"}

    def test_case_insensitive_normalization(self, tmp_path: Path) -> None:
        """Package names are normalized to lowercase per PEP 503."""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text(dedent("""
            PyYAML>=6.0
            OpenCV-Python>=4.8.0
            HTTPX>=0.28
        """).strip())
        packages = extract_packages(req_file)
        assert "pyyaml" in packages
        assert "opencv-python" in packages
        assert "httpx" in packages

    def test_underscore_to_hyphen_normalization(self, tmp_path: Path) -> None:
        """Underscores are normalized to hyphens per PEP 503."""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text(dedent("""
            opencv_python>=4.8.0
            pytest_cov>=4.0
            pytest_asyncio>=0.21
        """).strip())
        packages = extract_packages(req_file)
        assert "opencv-python" in packages
        assert "pytest-cov" in packages
        assert "pytest-asyncio" in packages

    def test_dot_to_hyphen_normalization(self, tmp_path: Path) -> None:
        """Dots in package names are normalized to hyphens per PEP 503."""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text(dedent("""
            zope.interface>=5.0
            ruamel.yaml>=0.18
        """).strip())
        packages = extract_packages(req_file)
        assert "zope-interface" in packages
        assert "ruamel-yaml" in packages

    def test_extras_syntax_handling(self, tmp_path: Path) -> None:
        """Package names with extras are extracted correctly (extras stripped)."""
        req_file = tmp_path / "requirements.txt"
        # Note: Current regex captures up to the first non-alphanumeric/dot/underscore/hyphen,
        # so brackets from extras are not included
        req_file.write_text(dedent("""
            uvicorn[standard]>=0.25.0
            httpx[http2]>=0.28
        """).strip())
        packages = extract_packages(req_file)
        assert "uvicorn" in packages
        assert "httpx" in packages
        # Extras should NOT be in the package name
        assert not any("[" in p for p in packages)

    def test_environment_markers_handling(self, tmp_path: Path) -> None:
        """Package names with environment markers are extracted correctly."""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text(dedent("""
            pyobjc-core>=10.0 ; sys_platform == 'darwin'
            pywin32>=306 ; sys_platform == 'win32'
        """).strip())
        packages = extract_packages(req_file)
        assert "pyobjc-core" in packages
        assert "pywin32" in packages
        # Markers should NOT be in the package name
        assert not any(";" in p for p in packages)

    def test_r_include_lines_skipped(self, tmp_path: Path) -> None:
        """Lines starting with -r are skipped (include directives)."""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text(dedent("""
            -r requirements.txt
            -r ../base.txt
            pytest>=8.0
        """).strip())
        packages = extract_packages(req_file)
        assert packages == {"pytest"}

    def test_mixed_normalization_scenarios(self, tmp_path: Path) -> None:
        """Complex file with mixed scenarios is handled correctly."""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text(dedent("""
            # Core dependencies
            -r requirements.txt
            
            PyYAML>=6.0  # YAML parsing
            opencv_python>=4.8.0,<5
            ZOPE.Interface>=5.0 ; python_version >= '3.10'
            
            # Testing
            pytest>=8.0
            pytest-cov>=4.0
        """).strip())
        packages = extract_packages(req_file)
        expected = {
            "pyyaml",
            "opencv-python",
            "zope-interface",
            "pytest",
            "pytest-cov",
        }
        assert packages == expected


# ============================================================================
# TEST: File Not Found Handling (Fail Fast)
# ============================================================================


class TestExtractPackagesFileMissing:
    """Tests for fail-fast behavior when files are missing."""

    def test_missing_file_raises_error(self, tmp_path: Path) -> None:
        """Missing file raises FileNotFoundError instead of returning empty set."""
        missing_file = tmp_path / "nonexistent.txt"
        with pytest.raises(FileNotFoundError) as exc_info:
            extract_packages(missing_file)
        assert "Required requirements file not found" in str(exc_info.value)
        assert "nonexistent.txt" in str(exc_info.value)


# ============================================================================
# TEST: Pattern Matching for Test Runners and CI Tools
# ============================================================================


class TestPatternMatching:
    """Tests for the pattern constants used to detect package categories."""

    @pytest.mark.parametrize("pkg", [
        "pytest",
        "pytest-cov",
        "pytest-asyncio",
        "pytest-xdist",
        "pytest-json-report",
        "pytest-rerunfailures",
        "httpx",
        "hypothesis",
    ])
    def test_test_runner_pattern_matches_test_deps(self, pkg: str) -> None:
        """TEST_RUNNER_PATTERN matches all test framework packages."""
        assert TEST_RUNNER_PATTERN.match(pkg), f"Should match test runner: {pkg}"

    @pytest.mark.parametrize("pkg", [
        "bandit",
        "safety",
        "build",
        "twine",
        "tox",
        "pypdf",
        "jsonschema",
        "pyyaml",
        "opencv-python",
    ])
    def test_test_runner_pattern_does_not_match_non_test_deps(self, pkg: str) -> None:
        """TEST_RUNNER_PATTERN does NOT match non-test packages."""
        assert not TEST_RUNNER_PATTERN.match(pkg), f"Should NOT match non-test: {pkg}"

    def test_ci_tools_contains_expected_packages(self) -> None:
        """CI_TOOLS frozenset contains expected CI pipeline tools."""
        expected = {"bandit", "safety", "build", "twine", "tox", "pypdf"}
        assert CI_TOOLS == expected

    def test_core_test_deps_contains_expected_packages(self) -> None:
        """CORE_TEST_DEPS frozenset contains expected test framework packages."""
        expected = {
            "pytest",
            "pytest-cov",
            "pytest-asyncio",
            "pytest-json-report",
            "pytest-xdist",
            "hypothesis",
            "httpx",
        }
        assert CORE_TEST_DEPS == expected


# ============================================================================
# TEST: Symmetric Drift Detection (Reviewer Point 1)
# ============================================================================


class TestSymmetricDriftDetection:
    """Tests verifying drift detection catches both unwanted AND missing deps."""

    @pytest.fixture
    def fake_repo(self, tmp_path: Path) -> Path:
        """Create a minimal fake repo structure with requirements files."""
        # Create directories
        repo = tmp_path / "repo"
        repo.mkdir()
        requirements_dir = repo / "requirements"
        requirements_dir.mkdir()

        # Create the script path structure (for Path(__file__).parents[2] resolution)
        scripts_dir = repo / "scripts" / "validation"
        scripts_dir.mkdir(parents=True)

        return repo

    def test_detects_test_runners_in_ci_in(self, fake_repo: Path) -> None:
        """Detects test runners incorrectly placed in ci.in (unwanted deps)."""
        # Setup: test runner in ci.in (wrong)
        root_ci = fake_repo / "requirements-ci.txt"
        root_ci.write_text("pytest>=8.0\n")

        nested_ci = fake_repo / "requirements" / "ci.in"
        nested_ci.write_text("pytest-asyncio>=0.21\n")  # WRONG: should be in dev.in

        nested_dev = fake_repo / "requirements" / "dev.in"
        nested_dev.write_text("pytest>=8.0\n")

        # Extract packages
        nested_ci_packages = extract_packages(nested_ci)
        test_deps_in_nested_ci = {
            p for p in nested_ci_packages if TEST_RUNNER_PATTERN.match(p)
        }

        # Should detect pytest-asyncio in ci.in
        assert test_deps_in_nested_ci == {"pytest-asyncio"}

    def test_detects_ci_tools_in_root_ci(self, fake_repo: Path) -> None:
        """Detects CI tools incorrectly placed in root requirements-ci.txt."""
        # Setup: CI tool in root (wrong)
        root_ci = fake_repo / "requirements-ci.txt"
        root_ci.write_text("pytest>=8.0\nbandit>=1.7\n")  # bandit is WRONG here

        nested_ci = fake_repo / "requirements" / "ci.in"
        nested_ci.write_text("safety>=2.3\n")

        nested_dev = fake_repo / "requirements" / "dev.in"
        nested_dev.write_text("pytest>=8.0\n")

        # Extract packages
        root_ci_packages = extract_packages(root_ci)
        ci_tools_in_root = root_ci_packages & CI_TOOLS

        # Should detect bandit in root
        assert ci_tools_in_root == {"bandit"}

    def test_detects_missing_core_test_deps_in_dev_in(self, fake_repo: Path) -> None:
        """Detects missing test deps from dev.in that exist in root requirements-ci.txt.
        
        This is the key "missing deps" check (symmetric drift detection).
        """
        # Setup: core test deps in root but NOT in dev.in
        root_ci = fake_repo / "requirements-ci.txt"
        root_ci.write_text(dedent("""
            pytest>=8.0
            pytest-cov>=4.0
            pytest-asyncio>=0.21
            httpx>=0.28
        """).strip())

        nested_ci = fake_repo / "requirements" / "ci.in"
        nested_ci.write_text("bandit>=1.7\n")

        # dev.in is MISSING pytest-asyncio and httpx
        nested_dev = fake_repo / "requirements" / "dev.in"
        nested_dev.write_text(dedent("""
            pytest>=8.0
            pytest-cov>=4.0
        """).strip())

        # Extract packages
        root_ci_packages = extract_packages(root_ci)
        nested_dev_packages = extract_packages(nested_dev)

        # Calculate missing
        root_test_deps = root_ci_packages & CORE_TEST_DEPS
        dev_test_deps = nested_dev_packages & CORE_TEST_DEPS
        missing_in_dev = root_test_deps - dev_test_deps

        # Should detect pytest-asyncio and httpx missing from dev.in
        assert missing_in_dev == {"pytest-asyncio", "httpx"}

    def test_no_false_positives_when_synced(self, fake_repo: Path) -> None:
        """No drift detected when files are properly synced."""
        # Setup: everything correctly placed
        root_ci = fake_repo / "requirements-ci.txt"
        root_ci.write_text(dedent("""
            pytest>=8.0
            pytest-cov>=4.0
            pytest-asyncio>=0.21
            httpx>=0.28
            hypothesis>=6.0
        """).strip())

        nested_ci = fake_repo / "requirements" / "ci.in"
        nested_ci.write_text(dedent("""
            bandit>=1.7
            safety>=2.3
            build>=1.0
            twine>=4.0
        """).strip())

        nested_dev = fake_repo / "requirements" / "dev.in"
        nested_dev.write_text(dedent("""
            pytest>=8.0
            pytest-cov>=4.0
            pytest-asyncio>=0.21
            httpx>=0.28
            hypothesis>=6.0
            black==26.3.1
            isort>=5.13
        """).strip())

        # Extract packages
        root_ci_packages = extract_packages(root_ci)
        nested_ci_packages = extract_packages(nested_ci)
        nested_dev_packages = extract_packages(nested_dev)

        # Check 1: No test runners in ci.in
        test_deps_in_nested_ci = {
            p for p in nested_ci_packages if TEST_RUNNER_PATTERN.match(p)
        }
        assert test_deps_in_nested_ci == set()

        # Check 2: No CI tools in root
        ci_tools_in_root = root_ci_packages & CI_TOOLS
        assert ci_tools_in_root == set()

        # Check 3: All core test deps in dev.in
        root_test_deps = root_ci_packages & CORE_TEST_DEPS
        dev_test_deps = nested_dev_packages & CORE_TEST_DEPS
        missing_in_dev = root_test_deps - dev_test_deps
        assert missing_in_dev == set()


# ============================================================================
# TEST: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and unusual input."""

    def test_duplicate_packages_deduplicated(self, tmp_path: Path) -> None:
        """Duplicate packages in file result in single entry."""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text(dedent("""
            pytest>=8.0
            pytest>=7.0
            pytest>=8.0
        """).strip())
        packages = extract_packages(req_file)
        assert packages == {"pytest"}

    def test_normalized_duplicates_merged(self, tmp_path: Path) -> None:
        """Packages that normalize to same name are merged."""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text(dedent("""
            PyYAML>=6.0
            pyyaml>=5.0
            PYYAML>=7.0
        """).strip())
        packages = extract_packages(req_file)
        assert packages == {"pyyaml"}

    def test_package_starting_with_number_handled(self, tmp_path: Path) -> None:
        """Packages starting with numbers are extracted (though rare)."""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text(dedent("""
            3to2>=1.0
            2to3>=0.1
        """).strip())
        packages = extract_packages(req_file)
        assert "3to2" in packages
        assert "2to3" in packages
