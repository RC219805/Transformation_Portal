"""Tests for CLI module.

Note: These tests assume the package is installed in development mode.
Run `pip install -e .` from the repository root before running tests.

Coverage Target: 60% of CLI module
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

# Pytest markers
pytestmark = [
    pytest.mark.unit,
]


class TestCLIImport:
    """Tests for CLI module import."""

    def test_cli_module_imports(self):
        """Test that CLI module can be imported."""
        from transformation_portal.cli import analyze_cli, main, process_cli, render_cli

        # Verify functions are callable
        assert callable(render_cli)
        assert callable(process_cli)
        assert callable(analyze_cli)
        assert callable(main)

    def test_cli_apps_exist(self):
        """Test that CLI apps are defined."""
        from transformation_portal.cli import analyze_app, app, process_app, render_app

        # Verify apps are typer instances
        assert app is not None
        assert render_app is not None
        assert process_app is not None
        assert analyze_app is not None

    def test_cli_exports(self):
        """Test that CLI module exports expected symbols."""
        from transformation_portal import cli

        # Check __all__ exports
        assert hasattr(cli, "__all__")
        expected_exports = [
            "app",
            "render_app",
            "process_app",
            "analyze_app",
            "render_cli",
            "process_cli",
            "analyze_cli",
            "main",
            "version",
            "info",
        ]

        for export in expected_exports:
            assert export in cli.__all__, f"Missing export: {export}"

    def test_pipeline_app_exists(self):
        """Test that pipeline app is defined."""
        from transformation_portal.cli import pipeline_app

        assert pipeline_app is not None


class TestCLIFunctions:
    """Tests for CLI functions."""

    def test_render_cli_callable(self):
        """Test that render_cli is callable."""
        from transformation_portal.cli import render_cli

        assert callable(render_cli)

    def test_process_cli_callable(self):
        """Test that process_cli is callable."""
        from transformation_portal.cli import process_cli

        assert callable(process_cli)

    def test_analyze_cli_callable(self):
        """Test that analyze_cli is callable."""
        from transformation_portal.cli import analyze_cli

        assert callable(analyze_cli)


class TestCheckModuleAvailability:
    """Tests for check_module_availability function."""

    def test_available_module(self):
        """Test check with available module."""
        from transformation_portal.cli import check_module_availability

        # This module should always exist
        result = check_module_availability("pathlib", "pathlib")
        assert result is True

    def test_unavailable_module(self):
        """Test check with unavailable module."""
        import typer

        from transformation_portal.cli import check_module_availability

        with pytest.raises(typer.Exit) as exc_info:
            check_module_availability("nonexistent_module_xyz", "Test Module")

        assert exc_info.value.exit_code == 1


class TestVersionCommand:
    """Tests for version command."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_version_command(self, runner):
        """Test version command output."""
        from transformation_portal.cli import app

        result = runner.invoke(app, ["version"])

        assert result.exit_code == 0
        assert "Transformation Portal" in result.stdout


class TestInfoCommand:
    """Tests for info command."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_info_command(self, runner):
        """Test info command output."""
        from transformation_portal.cli import app

        result = runner.invoke(app, ["info"])

        assert result.exit_code == 0
        assert "Python:" in result.stdout
        assert "Dependencies:" in result.stdout


class TestRenderCommands:
    """Tests for render subcommands."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_render_lux_no_args(self, runner):
        """Test render lux without arguments shows error."""
        from transformation_portal.cli import render_app

        result = runner.invoke(render_app, ["lux"])

        # Should fail without required options
        assert result.exit_code != 0

    def test_render_lux_missing_input(self, runner, temp_workspace):
        """Test render lux with nonexistent input file."""
        from transformation_portal.cli import render_app

        result = runner.invoke(
            render_app,
            [
                "lux",
                "--input",
                "/nonexistent/image.jpg",
                "--output",
                str(temp_workspace["output_dir"]),
            ],
        )

        assert result.exit_code == 1
        # Error message may be in stdout or output; just verify exit code

    def test_render_depth_no_args(self, runner):
        """Test render depth without arguments shows error."""
        from transformation_portal.cli import render_app

        result = runner.invoke(render_app, ["depth"])

        # Should fail without required options
        assert result.exit_code != 0

    def test_render_depth_missing_input(self, runner, temp_workspace):
        """Test render depth with nonexistent input file."""
        from transformation_portal.cli import render_app

        result = runner.invoke(
            render_app,
            [
                "depth",
                "--input",
                "/nonexistent/image.jpg",
                "--output",
                str(temp_workspace["output_dir"]),
            ],
        )

        assert result.exit_code == 1


class TestProcessCommands:
    """Tests for process subcommands."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_process_material_no_args(self, runner):
        """Test process material without arguments shows error."""
        from transformation_portal.cli import process_app

        result = runner.invoke(process_app, ["material"])

        # Should fail without required options
        assert result.exit_code != 0

    def test_process_material_missing_input(self, runner, temp_workspace):
        """Test process material with nonexistent input file."""
        from transformation_portal.cli import process_app

        result = runner.invoke(
            process_app,
            [
                "material",
                "--input",
                "/nonexistent/image.tiff",
                "--output",
                str(temp_workspace["output_dir"] / "output.tiff"),
            ],
        )

        assert result.exit_code == 1

    def test_process_video_no_args(self, runner):
        """Test process video without arguments shows error."""
        from transformation_portal.cli import process_app

        result = runner.invoke(process_app, ["video"])

        # Should fail without required options
        assert result.exit_code != 0

    def test_process_tif_no_args(self, runner):
        """Test process tif without arguments shows error."""
        from transformation_portal.cli import process_app

        result = runner.invoke(process_app, ["tif"])

        # Should fail without required options
        assert result.exit_code != 0

    def test_process_tif_missing_input_dir(self, runner, temp_workspace):
        """Test process tif with nonexistent input directory."""
        from transformation_portal.cli import process_app

        result = runner.invoke(
            process_app,
            [
                "tif",
                "--input",
                "/nonexistent/directory",
                "--output",
                str(temp_workspace["output_dir"]),
            ],
        )

        assert result.exit_code == 1


class TestAnalyzeCommands:
    """Tests for analyze subcommands."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_analyze_philosophy_no_args(self, runner):
        """Test analyze philosophy with default path."""
        from transformation_portal.cli import analyze_app

        # May fail due to missing module, but shouldn't crash
        result = runner.invoke(analyze_app, ["philosophy"])

        # Either succeeds or fails gracefully (module not found)
        assert result.exit_code in (0, 1)

    def test_analyze_philosophy_missing_path(self, runner):
        """Test analyze philosophy with nonexistent path."""
        from transformation_portal.cli import analyze_app

        result = runner.invoke(
            analyze_app,
            ["philosophy", "--path", "/nonexistent/path"],
        )

        assert result.exit_code == 1

    def test_analyze_decay_missing_path(self, runner):
        """Test analyze decay with nonexistent path."""
        from transformation_portal.cli import analyze_app

        result = runner.invoke(
            analyze_app,
            ["decay", "--path", "/nonexistent/path"],
        )

        assert result.exit_code == 1

    def test_analyze_workflow_missing_path(self, runner):
        """Test analyze workflow with nonexistent path."""
        from transformation_portal.cli import analyze_app

        result = runner.invoke(
            analyze_app,
            ["workflow", "--path", "/nonexistent/workflows"],
        )

        assert result.exit_code == 1


class TestPipelineCommands:
    """Tests for pipeline subcommands."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_pipeline_process_no_args(self, runner):
        """Test pipeline process without arguments shows error."""
        from transformation_portal.cli import pipeline_app

        result = runner.invoke(pipeline_app, ["process"])

        # Should fail without required options
        assert result.exit_code != 0

    def test_pipeline_process_missing_recipe(self, runner, temp_workspace):
        """Test pipeline process with nonexistent recipe file."""
        from transformation_portal.cli import pipeline_app

        result = runner.invoke(
            pipeline_app,
            [
                "process",
                "--input",
                "*.jpg",
                "--output",
                str(temp_workspace["output_dir"]),
                "--recipe",
                "/nonexistent/recipe.yaml",
            ],
        )

        assert result.exit_code == 1

    def test_pipeline_list_recipes_missing_dir(self, runner):
        """Test pipeline list-recipes with nonexistent directory."""
        from transformation_portal.cli import pipeline_app

        result = runner.invoke(
            pipeline_app,
            ["list-recipes", "--dir", "/nonexistent/recipes"],
        )

        assert result.exit_code == 1

    def test_pipeline_validate_recipe_missing_file(self, runner):
        """Test pipeline validate-recipe with nonexistent file."""
        from transformation_portal.cli import pipeline_app

        result = runner.invoke(
            pipeline_app,
            ["validate-recipe", "/nonexistent/recipe.yaml"],
        )

        assert result.exit_code == 1

    def test_pipeline_validate_recipe_valid(self, runner, temp_workspace):
        """Test pipeline validate-recipe with valid recipe file."""
        from transformation_portal.cli import pipeline_app

        # Create valid recipe file
        recipe_path = temp_workspace["root"] / "recipe.yaml"
        recipe_path.write_text(
            """
name: Test Recipe
stages:
  - depth_estimation
  - color_grading
"""
        )

        result = runner.invoke(
            pipeline_app,
            ["validate-recipe", str(recipe_path)],
        )

        assert result.exit_code == 0
        assert "valid" in result.stdout.lower()

    def test_pipeline_validate_recipe_invalid(self, runner, temp_workspace):
        """Test pipeline validate-recipe with invalid recipe file."""
        from transformation_portal.cli import pipeline_app

        # Create invalid recipe file (missing required fields)
        recipe_path = temp_workspace["root"] / "recipe.yaml"
        recipe_path.write_text(
            """
description: No name or stages
"""
        )

        result = runner.invoke(
            pipeline_app,
            ["validate-recipe", str(recipe_path)],
        )

        assert result.exit_code == 1

    def test_pipeline_validate_recipe_verbose(self, runner, temp_workspace):
        """Test pipeline validate-recipe with verbose flag."""
        from transformation_portal.cli import pipeline_app

        # Create valid recipe file
        recipe_path = temp_workspace["root"] / "recipe.yaml"
        recipe_path.write_text(
            """
name: Test Recipe
description: A test recipe
stages:
  - depth_estimation
output:
  format: png
"""
        )

        result = runner.invoke(
            pipeline_app,
            ["validate-recipe", str(recipe_path), "--verbose"],
        )

        assert result.exit_code == 0
        assert "Recipe Information" in result.stdout


class TestMainApp:
    """Tests for main app."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_main_no_args(self, runner):
        """Test main app with no arguments shows help."""
        from transformation_portal.cli import app

        result = runner.invoke(app, [])

        # Should show help or no_args_is_help behavior
        assert result.exit_code == 0 or "Usage:" in result.stdout

    def test_main_help(self, runner):
        """Test main app with help flag."""
        from transformation_portal.cli import app

        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "render" in result.stdout.lower()
        assert "process" in result.stdout.lower()
        assert "analyze" in result.stdout.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
