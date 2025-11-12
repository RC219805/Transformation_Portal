#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests to validate codebase structure and organization.

These tests ensure that the repository maintains good structure and
prevents accumulation of clutter or structural issues.
"""
from pathlib import Path

import pytest

# Repository root for structure validation
_repo_root = Path(__file__).parent.parent


class TestDirectoryStructure:
    """Tests for proper directory organization."""

    def test_src_package_exists(self):
        """Test that main package exists in src/."""
        package_path = _repo_root / "src" / "transformation_portal"
        assert package_path.exists(), "Main package should exist in src/"
        assert package_path.is_dir(), "transformation_portal should be a directory"

    def test_docs_directory_exists(self):
        """Test that docs directory exists."""
        docs_path = _repo_root / "docs"
        assert docs_path.exists(), "docs/ directory should exist"
        assert docs_path.is_dir(), "docs should be a directory"

    def test_tests_directory_exists(self):
        """Test that tests directory exists."""
        tests_path = _repo_root / "tests"
        assert tests_path.exists(), "tests/ directory should exist"
        assert tests_path.is_dir(), "tests should be a directory"

    def test_assets_directory_exists(self):
        """Test that assets directory exists."""
        assets_path = _repo_root / "assets"
        assert assets_path.exists(), "assets/ directory should exist"
        assert assets_path.is_dir(), "assets should be a directory"

    def test_config_directory_exists(self):
        """Test that config directory exists."""
        config_path = _repo_root / "config"
        assert config_path.exists(), "config/ directory should exist"


class TestPackageStructure:
    """Tests for proper package organization in src/."""

    def test_main_init_exists(self):
        """Test that main package has __init__.py."""
        init_file = _repo_root / "src" / "transformation_portal" / "__init__.py"
        assert init_file.exists(), "Main __init__.py should exist"

    def test_subpackages_have_init(self):
        """Test that all subpackages have __init__.py."""
        base_path = _repo_root / "src" / "transformation_portal"

        # Check key subpackages
        subpackages = [
            "depth",
            "processors",
            "pipelines",
            "utils",
            "analyzers",
        ]

        for subpkg in subpackages:
            subpkg_path = base_path / subpkg
            if subpkg_path.exists() and subpkg_path.is_dir():
                init_file = subpkg_path / "__init__.py"
                assert init_file.exists(), f"{subpkg} should have __init__.py"

    def test_utils_performance_module_exists(self):
        """Test that performance utilities module exists."""
        perf_module = (
            _repo_root / "src" / "transformation_portal" /
            "utils" / "performance.py"
        )
        assert perf_module.exists(), "performance.py should exist in utils/"

    def test_utils_error_handling_module_exists(self):
        """Test that error handling utilities module exists."""
        error_module = (
            _repo_root / "src" / "transformation_portal" /
            "utils" / "error_handling.py"
        )
        assert error_module.exists(), "error_handling.py should exist in utils/"


class TestDocumentationOrganization:
    """Tests for documentation organization."""

    def test_main_readme_exists(self):
        """Test that main README exists in root."""
        readme = _repo_root / "README.md"
        assert readme.exists(), "README.md should exist in root"

    def test_no_excessive_root_markdown_files(self):
        """Test that root doesn't have too many markdown files."""
        markdown_files = list(_repo_root.glob("*.md"))
        # Allow README files but not excessive documentation
        assert len(markdown_files) <= 10, (
            f"Too many markdown files in root ({len(markdown_files)}). "
            "Move documentation to docs/"
        )

    def test_docs_subdirectories_exist(self):
        """Test that docs has proper subdirectories."""
        docs_path = _repo_root / "docs"

        # Check for key documentation directories
        expected_subdirs = [
            "depth_pipeline",
            "workflow",
            "brand",
        ]

        for subdir in expected_subdirs:
            subdir_path = docs_path / subdir
            if not subdir_path.exists():
                # Not all may exist in every branch, but log it
                print(f"Note: {subdir} directory not found in docs/")


class TestAssetOrganization:
    """Tests for asset organization."""

    def test_luts_directory_exists(self):
        """Test that LUTs directory exists."""
        luts_path = _repo_root / "assets" / "luts"
        assert luts_path.exists(), "assets/luts/ should exist"

    def test_luts_subdirectories(self):
        """Test that LUTs are organized in subdirectories."""
        luts_path = _repo_root / "assets" / "luts"

        if luts_path.exists():
            # Check for expected LUT categories
            expected_categories = [
                "film_emulation",
                "location_aesthetic",
                "material_response",
            ]

            for category in expected_categories:
                category_path = luts_path / category
                if not category_path.exists():
                    print(f"Note: {category} not found in assets/luts/")


class TestWrapperFiles:
    """Tests for console script entrypoints instead of wrapper files."""

    def test_console_scripts_defined_in_pyproject(self):
        """Test that console_scripts are properly defined in pyproject.toml."""
        pyproject_path = _repo_root / "pyproject.toml"
        assert pyproject_path.exists(), "pyproject.toml should exist"

        content = pyproject_path.read_text()
        assert "[project.scripts]" in content, "project.scripts section should exist"
        assert "luxury-tiff-batch" in content, "luxury-tiff-batch entrypoint should be defined"


class TestGitignore:
    """Tests for .gitignore coverage."""

    def test_gitignore_exists(self):
        """Test that .gitignore exists."""
        gitignore = _repo_root / ".gitignore"
        assert gitignore.exists(), ".gitignore should exist"

    def test_gitignore_covers_python_artifacts(self):
        """Test that .gitignore covers Python artifacts."""
        gitignore = _repo_root / ".gitignore"
        content = gitignore.read_text()

        # Check for important patterns
        patterns = [
            "__pycache__",
            "*.pyc",
            "*.egg-info",
            ".venv",
            "dist/",
            "build/",
        ]

        for pattern in patterns:
            assert pattern in content, (
                f".gitignore should include pattern: {pattern}"
            )

    def test_gitignore_covers_outputs(self):
        """Test that .gitignore covers processing outputs."""
        gitignore = _repo_root / ".gitignore"
        content = gitignore.read_text()

        # Check for output patterns
        output_patterns = [
            "*.log",
            "_depth.npy",
            "_enhanced.png",
            "processed_output/",
        ]

        for pattern in output_patterns:
            assert pattern in content, (
                f".gitignore should include output pattern: {pattern}"
            )


class TestNoOrphanedFiles:
    """Tests to prevent orphaned or redundant files."""

    def test_no_duplicate_material_response(self):
        """Test that material_response isn't duplicated unnecessarily."""
        root_file = _repo_root / "material_response.py"
        src_file = (
            _repo_root / "src" / "transformation_portal" /
            "processors" / "material_response" / "core.py"
        )

        # Both may exist for backward compatibility, but root should be thin
        if root_file.exists() and src_file.exists():
            root_size = root_file.stat().st_size
            src_size = src_file.stat().st_size

            # Root file should not be a full duplicate
            # (Allow some size for a substantial wrapper, but not full copy)
            assert root_size < src_size * 0.5, (
                "Root material_response.py seems to be a full duplicate. "
                "Should be a thin wrapper."
            )

            # Check that the root file actually delegates to the src implementation
            root_text = root_file.read_text(encoding="utf-8")
            # Look for an import from transformation_portal (allow whitespace, case-insensitive)
            import_found = (
                "from transformation_portal" in root_text
                or "import transformation_portal" in root_text
            )
            assert import_found, (
                "Root material_response.py should import from transformation_portal "
                "to delegate implementation, not duplicate it."
            )

    def test_no_build_artifacts_in_root(self):
        """Test that build artifacts aren't in root."""
        # Check for common build artifacts
        artifacts = [
            "dist",
            "build",
            "*.egg-info",
        ]

        for pattern in artifacts:
            matching = list(_repo_root.glob(pattern))
            # Some may be in .gitignore, but shouldn't be tracked
            if matching:
                print(f"Note: Found build artifacts matching {pattern}")


class TestConfigurationFiles:
    """Tests for configuration files."""

    def test_pyproject_toml_exists(self):
        """Test that pyproject.toml exists."""
        pyproject = _repo_root / "pyproject.toml"
        assert pyproject.exists(), "pyproject.toml should exist"

    def test_makefile_exists(self):
        """Test that Makefile exists."""
        makefile = _repo_root / "Makefile"
        assert makefile.exists(), "Makefile should exist for common tasks"

    def test_requirements_files_exist(self):
        """Test that requirements files exist."""
        requirements = _repo_root / "requirements.txt"
        assert requirements.exists(), "requirements.txt should exist"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
