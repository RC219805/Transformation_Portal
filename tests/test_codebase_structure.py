#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests to validate codebase structure and organization.

These tests ensure that the repository maintains good structure and
prevents accumulation of clutter or structural issues.
"""

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

# Pytest markers
pytestmark = [
    pytest.mark.unit,
]

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
        perf_module = _repo_root / "src" / "transformation_portal" / "utils" / "performance.py"
        assert perf_module.exists(), "performance.py should exist in utils/"

    def test_utils_error_handling_module_exists(self):
        """Test that error handling utilities module exists."""
        error_module = _repo_root / "src" / "transformation_portal" / "utils" / "error_handling.py"
        assert error_module.exists(), "error_handling.py should exist in utils/"


class TestDocumentationOrganization:
    """Tests for documentation organization."""

    def test_main_readme_exists(self):
        """Test that main README exists in root."""
        readme = _repo_root / "README.md"
        assert readme.exists(), "README.md should exist in root"

    def test_no_excessive_root_markdown_files(self):
        """Test that tracked root paths conform to the canonical placement policy."""
        result = subprocess.run(
            ["./scripts/setup/pre-commit-check.sh", "--all"],
            cwd=_repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr

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

    def test_duplicate_data_luts_removed(self):
        """LUTs have one canonical repo location."""
        assert not (_repo_root / "data" / "luts").exists(), "data/luts duplicates assets/luts and must stay removed"

    def test_board_material_textures_are_under_assets(self):
        """Board material textures live with other repo assets."""
        texture_path = _repo_root / "assets" / "textures" / "board_materials"
        assert texture_path.is_dir(), "assets/textures/board_materials/ should exist"
        assert not (_repo_root / "textures").exists(), "root textures/ should not be recreated"

        expected_textures = {
            "plaster_marmorino_westwood_beige.png",
            "stone_bokara_coastal.png",
            "cladding_sculptform_warm.png",
            "screens_grey_gum.png",
            "equitone_lt85.png",
            "bison_weathered_ipe.png",
            "dark_bronze_anodized.png",
            "louvretec_powder_white.png",
        }
        actual_textures = {path.name for path in texture_path.glob("*.png")}
        assert expected_textures <= actual_textures


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
            assert pattern in content, f".gitignore should include pattern: {pattern}"

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
            assert pattern in content, f".gitignore should include output pattern: {pattern}"

    def test_gitignore_covers_frontdoor_temp_dirs(self):
        """Test that .gitignore covers frontdoor transient build directories."""
        gitignore = _repo_root / ".gitignore"
        content = gitignore.read_text()

        temp_dir_patterns = [
            "web/secure-landing/.next/",
            "web/secure-landing/.next-build-verify/",
            "web/secure-landing/.next-smoke-*/",
            "web/secure-landing/.next-codex-*/",
        ]

        for pattern in temp_dir_patterns:
            assert pattern in content, f".gitignore should include frontdoor temp dir pattern: {pattern}"


class TestNoOrphanedFiles:
    """Tests to prevent orphaned or redundant files."""

    def test_retired_generated_root_directories_absent(self):
        """Generated/demo roots should not be committed as top-level directories."""
        retired_roots = [
            "dashboard",
            "linear_ingest_demo",
            "projects",
            "test_sky_fix",
        ]
        present_roots = [path for path in retired_roots if (_repo_root / path).exists()]
        assert not present_roots, f"Retired root directories should stay removed: {present_roots}"

    def test_board_texture_generator_has_canonical_utility_module(self):
        """Root board-texture script delegates to the canonical utility module."""
        wrapper = _repo_root / "scripts" / "create_board_textures.py"
        canonical = _repo_root / "scripts" / "utilities" / "create_board_textures.py"

        wrapper_text = wrapper.read_text(encoding="utf-8")
        canonical_text = canonical.read_text(encoding="utf-8")

        assert "from scripts.utilities.create_board_textures import main" in wrapper_text
        assert 'DEFAULT_OUTPUT_DIR = REPO_ROOT / "assets" / "textures" / "board_materials"' in canonical_text

    def test_sky_comparison_requires_explicit_inputs_and_ignored_default_output(self):
        """The sky comparison tool no longer depends on removed root investigation files."""
        script = _repo_root / "tools" / "investigations" / "materials_v3" / "create_sky_comparison.py"

        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=_repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stdout + result.stderr
        assert "--before BEFORE" in result.stdout
        assert "--after AFTER" in result.stdout
        assert "output/materials_v3/sky_fix_comparison.jpg" in result.stdout
        assert "test_sky_fix" not in result.stdout

    def test_no_duplicate_material_response(self):
        """Test that material_response isn't duplicated unnecessarily."""
        root_file = _repo_root / "material_response.py"
        src_file = _repo_root / "src" / "transformation_portal" / "processors" / "material_response" / "core.py"

        # Both may exist for backward compatibility, but root should be thin
        if root_file.exists() and src_file.exists():
            root_size = root_file.stat().st_size
            src_size = src_file.stat().st_size

            # Root file should not be a full duplicate
            # (Allow some size for a substantial wrapper, but not full copy)
            assert root_size < src_size * 0.5, (
                "Root material_response.py seems to be a full duplicate. " "Should be a thin wrapper."
            )

            # Check that the root file actually delegates to the src implementation
            root_text = root_file.read_text(encoding="utf-8")
            # Look for an import from transformation_portal (allow whitespace, case-insensitive)
            import_found = "from transformation_portal" in root_text or "import transformation_portal" in root_text
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

    def test_no_nested_git_worktrees_in_root(self):
        """Fail if a copied repo/worktree lands directly under repo root."""
        nested_git_roots = []

        for child in _repo_root.iterdir():
            if not child.is_dir() or child.name == ".git":
                continue
            if (child / ".git").exists():
                nested_git_roots.append(child.relative_to(_repo_root).as_posix())

        assert not nested_git_roots, (
            "Unexpected nested git repositories/worktrees found in repo root. "
            "Remove copied worktrees so this checkout remains the single source of truth:\n"
            + "\n".join(sorted(nested_git_roots))
        )

    def test_no_exact_duplicate_python_modules_between_src_and_scripts(self):
        """Keep `scripts/` wrappers thin and prevent copy/paste module drift."""
        src_root = _repo_root / "src" / "transformation_portal"
        scripts_root = _repo_root / "scripts"

        if not src_root.exists() or not scripts_root.exists():
            return

        src_hashes = {}
        for src_file in src_root.rglob("*.py"):
            if src_file.is_file():
                digest = hashlib.sha1(src_file.read_bytes()).hexdigest()
                src_hashes.setdefault(digest, []).append(src_file.relative_to(_repo_root))

        duplicates = []
        for script_file in scripts_root.rglob("*.py"):
            if not script_file.is_file():
                continue
            digest = hashlib.sha1(script_file.read_bytes()).hexdigest()
            if digest in src_hashes:
                for src_match in src_hashes[digest]:
                    duplicates.append(f"{script_file.relative_to(_repo_root)} == {src_match}")

        assert not duplicates, (
            "Exact duplicate Python implementations found between scripts/ and src/. "
            "Convert script copies to thin wrappers that import canonical src modules.\n" + "\n".join(sorted(duplicates))
        )


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
