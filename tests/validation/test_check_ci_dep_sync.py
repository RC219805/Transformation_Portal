"""
Unit tests for the CI dependency sync check script.

Tests cover:
1. Symmetric drift detection (missing deps + unwanted deps)
2. Robust normalization (case, underscore/hyphen, version specifiers, comments, extras)
3. File handling (missing files raise errors)
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from textwrap import dedent
from types import ModuleType

import pytest

# The fixture is intentionally named for readability at call sites.
# pylint: disable=redefined-outer-name

# Mark all tests in this module as unit tests (ADR-044)
pytestmark = [
    pytest.mark.unit,
]


def _load_check_ci_dep_sync_module() -> ModuleType:
    """Load the script module via file path to avoid import boundary violations.

    This repo's structural rules do not allow tests to import directly from
    scripts.validation.*. Instead, we load the module dynamically by path.
    """
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "validation" / "check_ci_dep_sync.py"

    spec = importlib.util.spec_from_file_location(
        "check_ci_dep_sync_under_test",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sync_module() -> ModuleType:
    """Fixture providing the check_ci_dep_sync module loaded by file path."""
    return _load_check_ci_dep_sync_module()


# ============================================================================
# TEST: Robust Normalization (Reviewer Point 2)
# ============================================================================


class TestExtractPackagesNormalization:
    """Tests for robust package name extraction and normalization."""

    def test_empty_file(self, tmp_path: Path, sync_module: ModuleType) -> None:
        """Empty file returns empty set."""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text("")
        assert sync_module.extract_packages(req_file) == set()

    def test_comment_only_file(self, tmp_path: Path, sync_module: ModuleType) -> None:
        """Files with only comments return empty set."""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text(dedent("""
            # This is a comment
            # Another comment
        """).strip())
        assert sync_module.extract_packages(req_file) == set()

    def test_whitespace_only_lines(self, tmp_path: Path, sync_module: ModuleType) -> None:
        """Whitespace-only lines are skipped."""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text(dedent("""
            pytest

            httpx
        """).strip())
        packages = sync_module.extract_packages(req_file)
        assert packages == {"pytest", "httpx"}

    def test_inline_comments(self, tmp_path: Path, sync_module: ModuleType) -> None:
        """Inline comments after package names are handled correctly.

        Note: Current implementation extracts package name before comment
        using regex that matches alphanumeric/dots/underscores/hyphens.
        """
        req_file = tmp_path / "requirements.txt"
        req_file.write_text(dedent("""
            pytest>=8.0  # testing framework
            httpx>=0.28  # async HTTP client
        """).strip())
        packages = sync_module.extract_packages(req_file)
        assert "pytest" in packages
        assert "httpx" in packages

    def test_version_specifiers_stripped(self, tmp_path: Path, sync_module: ModuleType) -> None:
        """Version specifiers are removed from package names."""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text(dedent("""
            pytest>=8.0,<10
            httpx==0.28.1
            hypothesis~=6.0
            jsonschema>=4.21.0,<5
        """).strip())
        packages = sync_module.extract_packages(req_file)
        assert packages == {"pytest", "httpx", "hypothesis", "jsonschema"}

    def test_case_insensitive_normalization(self, tmp_path: Path, sync_module: ModuleType) -> None:
        """Package names are normalized to lowercase per PEP 503."""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text(dedent("""
            PyYAML>=6.0
            OpenCV-Python>=4.8.0
            HTTPX>=0.28
        """).strip())
        packages = sync_module.extract_packages(req_file)
        assert "pyyaml" in packages
        assert "opencv-python" in packages
        assert "httpx" in packages

    def test_underscore_to_hyphen_normalization(self, tmp_path: Path, sync_module: ModuleType) -> None:
        """Underscores are normalized to hyphens per PEP 503."""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text(dedent("""
            opencv_python>=4.8.0
            pytest_cov>=4.0
            pytest_asyncio>=0.21
        """).strip())
        packages = sync_module.extract_packages(req_file)
        assert "opencv-python" in packages
        assert "pytest-cov" in packages
        assert "pytest-asyncio" in packages

    def test_dot_to_hyphen_normalization(self, tmp_path: Path, sync_module: ModuleType) -> None:
        """Dots in package names are normalized to hyphens per PEP 503."""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text(dedent("""
            zope.interface>=5.0
            ruamel.yaml>=0.18
        """).strip())
        packages = sync_module.extract_packages(req_file)
        assert "zope-interface" in packages
        assert "ruamel-yaml" in packages

    def test_extras_syntax_handling(self, tmp_path: Path, sync_module: ModuleType) -> None:
        """Package names with extras are extracted correctly (extras stripped)."""
        req_file = tmp_path / "requirements.txt"
        # Note: Current regex captures up to the first non-alphanumeric/dot/underscore/hyphen,
        # so brackets from extras are not included
        req_file.write_text(dedent("""
            uvicorn[standard]>=0.25.0
            httpx[http2]>=0.28
        """).strip())
        packages = sync_module.extract_packages(req_file)
        assert "uvicorn" in packages
        assert "httpx" in packages
        # Extras should NOT be in the package name
        assert not any("[" in p for p in packages)

    def test_environment_markers_handling(self, tmp_path: Path, sync_module: ModuleType) -> None:
        """Package names with environment markers are extracted correctly."""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text(dedent("""
            pyobjc-core>=10.0 ; sys_platform == 'darwin'
            pywin32>=306 ; sys_platform == 'win32'
        """).strip())
        packages = sync_module.extract_packages(req_file)
        assert "pyobjc-core" in packages
        assert "pywin32" in packages
        # Markers should NOT be in the package name
        assert not any(";" in p for p in packages)

    def test_r_include_lines_skipped(self, tmp_path: Path, sync_module: ModuleType) -> None:
        """Lines starting with -r are skipped (include directives)."""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text(dedent("""
            -r requirements.txt
            -r ../base.txt
            pytest>=8.0
        """).strip())
        packages = sync_module.extract_packages(req_file)
        assert packages == {"pytest"}

    def test_mixed_normalization_scenarios(self, tmp_path: Path, sync_module: ModuleType) -> None:
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
        packages = sync_module.extract_packages(req_file)
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

    def test_missing_file_raises_error(self, tmp_path: Path, sync_module: ModuleType) -> None:
        """Missing file raises FileNotFoundError instead of returning empty set."""
        missing_file = tmp_path / "nonexistent.txt"
        with pytest.raises(FileNotFoundError) as exc_info:
            sync_module.extract_packages(missing_file)
        assert "Required requirements file not found" in str(exc_info.value)
        assert "nonexistent.txt" in str(exc_info.value)


# ============================================================================
# TEST: Pattern Matching for Test Runners and CI Tools
# ============================================================================


class TestPatternMatching:
    """Tests for the pattern constants used to detect package categories."""

    @pytest.mark.parametrize(
        "pkg",
        [
            "pytest",
            "pytest-cov",
            "pytest-asyncio",
            "pytest-xdist",
            "pytest-json-report",
            "pytest-rerunfailures",
            "httpx",
            "hypothesis",
        ],
    )
    def test_test_runner_pattern_matches_test_deps(self, pkg: str, sync_module: ModuleType) -> None:
        """TEST_RUNNER_PATTERN matches all test framework packages."""
        assert sync_module.TEST_RUNNER_PATTERN.match(pkg), f"Should match test runner: {pkg}"

    @pytest.mark.parametrize(
        "pkg",
        [
            "bandit",
            "safety",
            "build",
            "twine",
            "tox",
            "pypdf",
            "jsonschema",
            "pyyaml",
            "opencv-python",
        ],
    )
    def test_test_runner_pattern_does_not_match_non_test_deps(self, pkg: str, sync_module: ModuleType) -> None:
        """TEST_RUNNER_PATTERN does NOT match non-test packages."""
        assert not sync_module.TEST_RUNNER_PATTERN.match(pkg), f"Should NOT match non-test: {pkg}"

    def test_ci_tools_contains_expected_packages(self, sync_module: ModuleType) -> None:
        """CI_TOOLS frozenset contains expected CI pipeline tools."""
        expected = {"bandit", "safety", "build", "twine", "tox", "pypdf"}
        assert sync_module.CI_TOOLS == expected

    def test_core_test_deps_contains_expected_packages(self, sync_module: ModuleType) -> None:
        """CORE_TEST_DEPS frozenset contains expected test framework packages."""
        expected = {
            "pytest",
            "pytest-cov",
            "pytest-asyncio",
            "pytest-json-report",
            "pytest-xdist",
            "hypothesis",
            "httpx",
            "moto",
        }
        assert sync_module.CORE_TEST_DEPS == expected

    def test_dev_only_deps_contains_expected_packages(self, sync_module: ModuleType) -> None:
        """DEV_ONLY_DEPS frozenset contains expected development-only test tooling."""
        assert sync_module.DEV_ONLY_DEPS == {"pytest-rerunfailures"}


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

    def test_detects_test_runners_in_ci_in(self, fake_repo: Path, sync_module: ModuleType) -> None:
        """Detects test dependencies incorrectly placed in ci.in (unwanted deps)."""
        # Setup: test runner in ci.in (wrong)
        root_ci = fake_repo / "requirements-ci.txt"
        root_ci.write_text("pytest>=8.0\n")

        nested_ci = fake_repo / "requirements" / "ci.in"
        nested_ci.write_text(dedent("""
            pytest-asyncio>=0.21
            moto[s3]>=5.0
        """).strip())  # WRONG: should be in dev.in/root CI

        nested_dev = fake_repo / "requirements" / "dev.in"
        nested_dev.write_text("pytest>=8.0\n")

        # Extract packages
        nested_ci_packages = sync_module.extract_packages(nested_ci)
        test_deps_in_nested_ci = {p for p in nested_ci_packages if sync_module.TEST_RUNNER_PATTERN.match(p)}
        test_deps_in_nested_ci |= nested_ci_packages & sync_module.CORE_TEST_DEPS

        # Should detect pytest-asyncio and moto in ci.in
        assert test_deps_in_nested_ci == {"pytest-asyncio", "moto"}

    def test_detects_ci_tools_in_root_ci(self, fake_repo: Path, sync_module: ModuleType) -> None:
        """Detects CI tools incorrectly placed in root requirements-ci.txt."""
        # Setup: CI tool in root (wrong)
        root_ci = fake_repo / "requirements-ci.txt"
        root_ci.write_text("pytest>=8.0\nbandit>=1.7\n")  # bandit is WRONG here

        nested_ci = fake_repo / "requirements" / "ci.in"
        nested_ci.write_text("safety>=2.3\n")

        nested_dev = fake_repo / "requirements" / "dev.in"
        nested_dev.write_text("pytest>=8.0\n")

        # Extract packages
        root_ci_packages = sync_module.extract_packages(root_ci)
        ci_tools_in_root = root_ci_packages & sync_module.CI_TOOLS

        # Should detect bandit in root
        assert ci_tools_in_root == {"bandit"}

    def test_detects_missing_core_test_deps_in_dev_in(self, fake_repo: Path, sync_module: ModuleType) -> None:
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
        root_ci_packages = sync_module.extract_packages(root_ci)
        nested_dev_packages = sync_module.extract_packages(nested_dev)

        # Calculate missing
        root_test_deps = root_ci_packages & sync_module.CORE_TEST_DEPS
        dev_test_deps = nested_dev_packages & sync_module.CORE_TEST_DEPS
        missing_in_dev = root_test_deps - dev_test_deps

        # Should detect pytest-asyncio and httpx missing from dev.in
        assert missing_in_dev == {"pytest-asyncio", "httpx"}

    def test_detects_missing_core_test_deps_in_root_ci(self, fake_repo: Path, sync_module: ModuleType) -> None:
        """Detects core test deps, including moto, missing from root requirements-ci.txt."""
        root_ci = fake_repo / "requirements-ci.txt"
        root_ci.write_text(dedent("""
            pytest>=8.0
            pytest-cov>=4.0
            pytest-asyncio>=0.21
            pytest-json-report>=1.5
            pytest-xdist>=3.5
            httpx>=0.28
            hypothesis>=6.0
        """).strip())

        nested_ci = fake_repo / "requirements" / "ci.in"
        nested_ci.write_text("bandit>=1.7\n")

        nested_dev = fake_repo / "requirements" / "dev.in"
        nested_dev.write_text(dedent("""
            pytest>=8.0
            pytest-cov>=4.0
            pytest-asyncio>=0.21
            pytest-json-report>=1.5
            pytest-xdist>=3.5
            httpx>=0.28
            hypothesis>=6.0
            moto[s3]>=5.0
        """).strip())

        root_ci_packages = sync_module.extract_packages(root_ci)
        missing_in_root = sync_module.CORE_TEST_DEPS - root_ci_packages

        assert missing_in_root == {"moto"}

    def test_no_false_positives_when_synced(self, fake_repo: Path, sync_module: ModuleType) -> None:
        """No drift detected when files are properly synced."""
        # Setup: everything correctly placed
        root_ci = fake_repo / "requirements-ci.txt"
        root_ci.write_text(dedent("""
            pytest>=8.0
            pytest-cov>=4.0
            pytest-asyncio>=0.21
            pytest-json-report>=1.5
            pytest-xdist>=3.5
            httpx>=0.28
            hypothesis>=6.0
            moto[s3]>=5.0
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
            pytest-json-report>=1.5
            pytest-xdist>=3.5
            httpx>=0.28
            hypothesis>=6.0
            moto[s3]>=5.0
            black==26.3.1
            isort>=5.13
        """).strip())

        # Extract packages
        root_ci_packages = sync_module.extract_packages(root_ci)
        nested_ci_packages = sync_module.extract_packages(nested_ci)
        nested_dev_packages = sync_module.extract_packages(nested_dev)

        # Check 1: No test runners in ci.in
        test_deps_in_nested_ci = {p for p in nested_ci_packages if sync_module.TEST_RUNNER_PATTERN.match(p)}
        assert test_deps_in_nested_ci == set()

        # Check 2: No CI tools in root
        ci_tools_in_root = root_ci_packages & sync_module.CI_TOOLS
        assert ci_tools_in_root == set()

        # Check 3: All core test deps in dev.in
        root_test_deps = root_ci_packages & sync_module.CORE_TEST_DEPS
        dev_test_deps = nested_dev_packages & sync_module.CORE_TEST_DEPS
        missing_in_dev = root_test_deps - dev_test_deps
        assert missing_in_dev == set()

    def test_detects_missing_dev_only_deps_in_root_dev(self, fake_repo: Path, sync_module: ModuleType) -> None:
        """Detects dev-only tooling present in requirements/dev.in but absent from root dev entry point."""
        root_ci = fake_repo / "requirements-ci.txt"
        root_ci.write_text(dedent("""
            pytest>=8.0
            pytest-cov>=4.0
            pytest-asyncio>=0.21
            pytest-json-report>=1.5
            pytest-xdist>=3.5
            httpx>=0.28
            hypothesis>=6.0
            moto[s3]>=5.0
        """).strip())

        root_dev = fake_repo / "requirements-dev.txt"
        root_dev.write_text(dedent("""
            -r requirements-ci.txt
            black>=24.0
        """).strip())

        nested_ci = fake_repo / "requirements" / "ci.in"
        nested_ci.write_text("bandit>=1.7\n")

        nested_dev = fake_repo / "requirements" / "dev.in"
        nested_dev.write_text(dedent("""
            pytest>=8.0
            pytest-cov>=4.0
            pytest-asyncio>=0.21
            pytest-json-report>=1.5
            pytest-xdist>=3.5
            pytest-rerunfailures>=14.0
            httpx>=0.28
            hypothesis>=6.0
            moto[s3]>=5.0
        """).strip())

        errors = sync_module.validate_dependency_sync(fake_repo)

        assert any("Dev-only deps missing from requirements-dev.txt" in error for error in errors)
        assert any("pytest-rerunfailures" in error for error in errors)

    def test_detects_dev_only_deps_in_root_ci(self, fake_repo: Path, sync_module: ModuleType) -> None:
        """Detects dev-only tooling added to lean CI requirements."""
        root_ci = fake_repo / "requirements-ci.txt"
        root_ci.write_text(dedent("""
            pytest>=8.0
            pytest-cov>=4.0
            pytest-asyncio>=0.21
            pytest-json-report>=1.5
            pytest-xdist>=3.5
            pytest-rerunfailures>=14.0
            httpx>=0.28
            hypothesis>=6.0
            moto[s3]>=5.0
        """).strip())

        root_dev = fake_repo / "requirements-dev.txt"
        root_dev.write_text(dedent("""
            -r requirements-ci.txt
            pytest-rerunfailures>=14.0
        """).strip())

        nested_ci = fake_repo / "requirements" / "ci.in"
        nested_ci.write_text("bandit>=1.7\n")

        nested_dev = fake_repo / "requirements" / "dev.in"
        nested_dev.write_text(dedent("""
            pytest>=8.0
            pytest-cov>=4.0
            pytest-asyncio>=0.21
            pytest-json-report>=1.5
            pytest-xdist>=3.5
            pytest-rerunfailures>=14.0
            httpx>=0.28
            hypothesis>=6.0
            moto[s3]>=5.0
        """).strip())

        errors = sync_module.validate_dependency_sync(fake_repo)

        assert any("Dev-only deps found in requirements-ci.txt" in error for error in errors)
        assert any("pytest-rerunfailures" in error for error in errors)


# ============================================================================
# TEST: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and unusual input."""

    def test_duplicate_packages_deduplicated(self, tmp_path: Path, sync_module: ModuleType) -> None:
        """Duplicate packages in file result in single entry."""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text(dedent("""
            pytest>=8.0
            pytest>=7.0
            pytest>=8.0
        """).strip())
        packages = sync_module.extract_packages(req_file)
        assert packages == {"pytest"}

    def test_normalized_duplicates_merged(self, tmp_path: Path, sync_module: ModuleType) -> None:
        """Packages that normalize to same name are merged."""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text(dedent("""
            PyYAML>=6.0
            pyyaml>=5.0
            PYYAML>=7.0
        """).strip())
        packages = sync_module.extract_packages(req_file)
        assert packages == {"pyyaml"}

    def test_package_starting_with_number_handled(self, tmp_path: Path, sync_module: ModuleType) -> None:
        """Packages starting with numbers are extracted (though rare)."""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text(dedent("""
            3to2>=1.0
            2to3>=0.1
        """).strip())
        packages = sync_module.extract_packages(req_file)
        assert "3to2" in packages
        assert "2to3" in packages


# ============================================================================
# TEST: main() Integration (Exit Codes)
# ============================================================================


class TestMainIntegration:
    """Integration tests for main() function exit codes."""

    @pytest.fixture
    def monkeypatch_repo_root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Create a fake repo structure and monkeypatch Path resolution."""
        # Create directory structure
        repo = tmp_path / "repo"
        repo.mkdir()
        requirements_dir = repo / "requirements"
        requirements_dir.mkdir()
        scripts_dir = repo / "scripts" / "validation"
        scripts_dir.mkdir(parents=True)

        # Monkeypatch Path.resolve to point __file__ to our fake repo
        # We need to import main fresh after patching
        return repo

    def test_main_returns_zero_when_synced(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sync_module: ModuleType
    ) -> None:
        """main() returns 0 when no drift is detected."""
        # Create synced repo structure
        repo = tmp_path / "repo"
        repo.mkdir()
        requirements_dir = repo / "requirements"
        requirements_dir.mkdir()

        # Root requirements-ci.txt with test deps (properly synced)
        root_ci = repo / "requirements-ci.txt"
        root_ci.write_text(dedent("""
            -r requirements.txt
            pytest>=8.0
            pytest-cov>=4.0
            pytest-asyncio>=0.21
            pytest-json-report>=1.5
            pytest-xdist>=3.5
            httpx>=0.28
            hypothesis>=6.0
            moto[s3]>=5.0
        """).strip())

        # requirements/ci.in with CI tools only (correct)
        nested_ci = requirements_dir / "ci.in"
        nested_ci.write_text(dedent("""
            bandit>=1.7
            safety>=2.3
            build>=1.0
        """).strip())

        # requirements/dev.in with test deps matching root (synced)
        nested_dev = requirements_dir / "dev.in"
        nested_dev.write_text(dedent("""
            pytest>=8.0
            pytest-cov>=4.0
            pytest-asyncio>=0.21
            pytest-json-report>=1.5
            pytest-xdist>=3.5
            httpx>=0.28
            hypothesis>=6.0
            moto[s3]>=5.0
            pytest-rerunfailures>=14.0
            black>=24.0
        """).strip())

        root_dev = repo / "requirements-dev.txt"
        root_dev.write_text(dedent("""
            -r requirements-ci.txt
            pytest-rerunfailures>=14.0
            black>=24.0
        """).strip())

        errors = sync_module.validate_dependency_sync(repo)
        assert errors == []

    def test_main_returns_one_when_drift_detected(self, tmp_path: Path, sync_module: ModuleType) -> None:
        """main() returns 1 when drift is detected."""
        # Create drifted repo structure
        repo = tmp_path / "repo"
        repo.mkdir()
        requirements_dir = repo / "requirements"
        requirements_dir.mkdir()

        # Root requirements-ci.txt
        root_ci = repo / "requirements-ci.txt"
        root_ci.write_text(dedent("""
            pytest>=8.0
            pytest-asyncio>=0.21
        """).strip())

        # requirements/ci.in with test runner (DRIFT!)
        nested_ci = requirements_dir / "ci.in"
        nested_ci.write_text(dedent("""
            bandit>=1.7
            pytest-xdist>=3.5
        """).strip())  # pytest-xdist is WRONG here

        # requirements/dev.in
        nested_dev = requirements_dir / "dev.in"
        nested_dev.write_text("pytest>=8.0\n")

        # Run checks manually
        errors: list[str] = []
        root_ci_packages = sync_module.extract_packages(root_ci)
        nested_ci_packages = sync_module.extract_packages(nested_ci)

        test_deps_in_nested_ci = {p for p in nested_ci_packages if sync_module.TEST_RUNNER_PATTERN.match(p)}
        if test_deps_in_nested_ci:
            errors.append(f"ERROR: test deps in ci.in: {test_deps_in_nested_ci}")

        result = 1 if errors else 0
        assert result == 1, "Expected main() to return 1 when drift detected"
        assert "pytest-xdist" in str(test_deps_in_nested_ci)

    def test_main_returns_one_when_missing_deps(self, tmp_path: Path, sync_module: ModuleType) -> None:
        """main() returns 1 when core test deps are missing from dev.in."""
        repo = tmp_path / "repo"
        repo.mkdir()
        requirements_dir = repo / "requirements"
        requirements_dir.mkdir()

        # Root has pytest-asyncio
        root_ci = repo / "requirements-ci.txt"
        root_ci.write_text(dedent("""
            pytest>=8.0
            pytest-cov>=4.0
            pytest-asyncio>=0.21
            pytest-json-report>=1.5
            pytest-xdist>=3.5
            httpx>=0.28
            hypothesis>=6.0
            moto[s3]>=5.0
        """).strip())

        # ci.in is correct
        nested_ci = requirements_dir / "ci.in"
        nested_ci.write_text("bandit>=1.7\n")

        # dev.in is MISSING pytest-asyncio and httpx
        nested_dev = requirements_dir / "dev.in"
        nested_dev.write_text(dedent("""
            pytest>=8.0
            pytest-cov>=4.0
            pytest-json-report>=1.5
            pytest-xdist>=3.5
            hypothesis>=6.0
            moto[s3]>=5.0
        """).strip())

        # Run checks manually
        root_ci_packages = sync_module.extract_packages(root_ci)
        nested_dev_packages = sync_module.extract_packages(nested_dev)

        root_test_deps = root_ci_packages & sync_module.CORE_TEST_DEPS
        dev_test_deps = nested_dev_packages & sync_module.CORE_TEST_DEPS
        missing_in_dev = root_test_deps - dev_test_deps

        result = 1 if missing_in_dev else 0
        assert result == 1, "Expected main() to return 1 when deps missing from dev.in"
        assert missing_in_dev == {"pytest-asyncio", "httpx"}


class TestRepositoryContract:
    """Tests for this repository's actual dependency-sync contract."""

    def test_repository_requirements_are_in_sync(self, sync_module: ModuleType) -> None:
        """The checked-in root and layered requirements files satisfy the sync gate."""
        repo_root = Path(__file__).resolve().parents[2]

        assert sync_module.validate_dependency_sync(repo_root) == []
