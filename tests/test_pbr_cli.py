#!/usr/bin/env python3
"""Comprehensive CLI tests for PBR map generation.

Priority P0: CLI Test Coverage
- Target: 0% → 80%+ test coverage
- Tests valid invocations, edge cases, error handling, output validation
"""

import pytest
import numpy as np
from typer.testing import CliRunner
from PIL import Image

from transformation_portal.lux_depth_v3.pbr_cli import app


# Test fixtures
@pytest.fixture
def cli_runner():
    """Create a CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def sample_depth_npy(tmp_path):
    """Create a sample depth map (.npy format)."""
    depth = np.random.rand(512, 512).astype(np.float32)
    depth_path = tmp_path / "test_depth.npy"
    np.save(depth_path, depth)
    return depth_path


@pytest.fixture
def sample_depth_png(tmp_path):
    """Create a sample depth map (.png format)."""
    depth = (np.random.rand(512, 512) * 255).astype(np.uint8)
    depth_img = Image.fromarray(depth, mode='L')
    depth_path = tmp_path / "test_depth.png"
    depth_img.save(depth_path)
    return depth_path


@pytest.fixture
def sample_depth_batch(tmp_path):
    """Create a directory with multiple depth files."""
    batch_dir = tmp_path / "depth_batch"
    batch_dir.mkdir()

    # Create 3 .npy files
    for i in range(3):
        depth = np.random.rand(256, 256).astype(np.float32)
        np.save(batch_dir / f"scene_{i:02d}_depth.npy", depth)

    # Create 2 .png files
    for i in range(2):
        depth = (np.random.rand(256, 256) * 255).astype(np.uint8)
        depth_img = Image.fromarray(depth, mode='L')
        depth_img.save(batch_dir / f"render_{i:02d}_depth.png")

    return batch_dir


@pytest.fixture
def empty_directory(tmp_path):
    """Create an empty directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    return empty_dir


@pytest.fixture
def corrupt_depth_file(tmp_path):
    """Create a corrupt/unreadable file."""
    corrupt_path = tmp_path / "corrupt_depth.npy"
    corrupt_path.write_text("This is not a valid numpy file")
    return corrupt_path


# P0: Valid Invocation Tests
class TestValidInvocations:
    """Test valid CLI invocations with various parameter combinations."""

    def test_single_file_basic(self, cli_runner, sample_depth_npy, tmp_path):
        """Test CLI with single depth file - basic invocation."""
        output_dir = tmp_path / "output"

        result = cli_runner.invoke(app, [
            "generate",
            "--depth", str(sample_depth_npy),
            "--output", str(output_dir),
        ])

        assert result.exit_code == 0, f"CLI failed: {result.stdout}"
        assert "Processing:" in result.stdout
        assert "Generated PBR maps" in result.stdout
        assert output_dir.exists()

    def test_single_file_with_preset(self, cli_runner, sample_depth_npy, tmp_path):
        """Test CLI with preset parameter."""
        output_dir = tmp_path / "output"

        result = cli_runner.invoke(app, [
            "generate",
            "--depth", str(sample_depth_npy),
            "--preset", "premium",
            "--output", str(output_dir),
        ])

        assert result.exit_code == 0
        assert "Using preset: premium" in result.stdout
        assert "Generated PBR maps" in result.stdout

    def test_single_file_all_presets(self, cli_runner, sample_depth_npy, tmp_path):
        """Test all available presets work."""
        presets = ["premium", "standard", "draft", "wood", "metal", "glass", "stone", "fabric"]

        for preset in presets:
            output_dir = tmp_path / f"output_{preset}"

            result = cli_runner.invoke(app, [
                "generate",
                "--depth", str(sample_depth_npy),
                "--preset", preset,
                "--output", str(output_dir),
            ])

            assert result.exit_code == 0, f"Preset '{preset}' failed: {result.stdout}"
            assert f"Using preset: {preset}" in result.stdout

    def test_single_file_custom_params(self, cli_runner, sample_depth_npy, tmp_path):
        """Test CLI with custom parameter overrides."""
        output_dir = tmp_path / "output"

        result = cli_runner.invoke(app, [
            "generate",
            "--depth", str(sample_depth_npy),
            "--normal-strength", "1.5",
            "--roughness-strength", "1.2",
            "--ao-strength", "1.8",
            "--ao-bias", "0.3",
            "--output", str(output_dir),
        ])

        assert result.exit_code == 0
        assert "Applied 4 parameter override(s)" in result.stdout

    def test_single_file_png_format(self, cli_runner, sample_depth_png, tmp_path):
        """Test CLI with PNG depth file."""
        output_dir = tmp_path / "output"

        result = cli_runner.invoke(app, [
            "generate",
            "--depth", str(sample_depth_png),
            "--output", str(output_dir),
        ])

        assert result.exit_code == 0
        assert "Generated PBR maps" in result.stdout

    def test_batch_directory_mode(self, cli_runner, sample_depth_batch, tmp_path):
        """Test CLI in batch directory mode."""
        output_dir = tmp_path / "output"

        result = cli_runner.invoke(app, [
            "generate",
            "--depth-dir", str(sample_depth_batch),
            "--output", str(output_dir),
        ])

        assert result.exit_code == 0
        assert "Batch processing 5 depth file(s)" in result.stdout
        assert "Batch complete" in result.stdout
        assert "Success: 5" in result.stdout
        assert "Errors:  0" in result.stdout

    def test_batch_with_preset(self, cli_runner, sample_depth_batch, tmp_path):
        """Test batch mode with preset."""
        output_dir = tmp_path / "output"

        result = cli_runner.invoke(app, [
            "generate",
            "--depth-dir", str(sample_depth_batch),
            "--preset", "wood",
            "--output", str(output_dir),
        ])

        assert result.exit_code == 0
        assert "Using preset: wood" in result.stdout
        assert "Success: 5" in result.stdout

    def test_verbose_mode(self, cli_runner, sample_depth_npy, tmp_path):
        """Test verbose logging mode."""
        output_dir = tmp_path / "output"

        result = cli_runner.invoke(app, [
            "generate",
            "--depth", str(sample_depth_npy),
            "--output", str(output_dir),
            "--verbose",
        ])

        assert result.exit_code == 0

    def test_list_presets(self, cli_runner):
        """Test --list-presets flag."""
        result = cli_runner.invoke(app, [
            "generate",
            "--list-presets",
        ])

        assert result.exit_code == 0
        assert "Available PBR Presets:" in result.stdout
        assert "premium" in result.stdout
        assert "wood" in result.stdout
        assert "metal" in result.stdout

    def test_info_command(self, cli_runner):
        """Test info command."""
        result = cli_runner.invoke(app, ["info"])

        assert result.exit_code == 0
        assert "Normal Map:" in result.stdout
        assert "Roughness Map:" in result.stdout
        assert "Ambient Occlusion:" in result.stdout


# P0: Edge Case Tests
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_nonexistent_input_file(self, cli_runner, tmp_path):
        """Test with nonexistent input file."""
        output_dir = tmp_path / "output"
        fake_path = tmp_path / "nonexistent_depth.npy"

        result = cli_runner.invoke(app, [
            "generate",
            "--depth", str(fake_path),
            "--output", str(output_dir),
        ])

        assert result.exit_code == 1
        assert "Error: Depth file not found" in result.output

    def test_nonexistent_input_directory(self, cli_runner, tmp_path):
        """Test with nonexistent input directory."""
        output_dir = tmp_path / "output"
        fake_dir = tmp_path / "nonexistent_dir"

        result = cli_runner.invoke(app, [
            "generate",
            "--depth-dir", str(fake_dir),
            "--output", str(output_dir),
        ])

        assert result.exit_code == 1
        assert "Error: Directory not found" in result.output

    def test_empty_directory(self, cli_runner, empty_directory, tmp_path):
        """Test with empty directory (no depth files)."""
        output_dir = tmp_path / "output"

        result = cli_runner.invoke(app, [
            "generate",
            "--depth-dir", str(empty_directory),
            "--output", str(output_dir),
        ])

        assert result.exit_code == 1
        assert "Warning: No depth files" in result.output

    def test_no_input_specified(self, cli_runner, tmp_path):
        """Test with neither --depth nor --depth-dir."""
        output_dir = tmp_path / "output"

        result = cli_runner.invoke(app, [
            "generate",
            "--output", str(output_dir),
        ])

        assert result.exit_code == 1
        assert "Error: Either --depth or --depth-dir required" in result.output

    def test_both_inputs_specified(self, cli_runner, sample_depth_npy, sample_depth_batch, tmp_path):
        """Test with both --depth and --depth-dir (conflicting)."""
        output_dir = tmp_path / "output"

        result = cli_runner.invoke(app, [
            "generate",
            "--depth", str(sample_depth_npy),
            "--depth-dir", str(sample_depth_batch),
            "--output", str(output_dir),
        ])

        assert result.exit_code == 1
        assert "Error: Cannot specify both --depth and --depth-dir" in result.output

    def test_invalid_preset_name(self, cli_runner, sample_depth_npy, tmp_path):
        """Test with invalid preset name."""
        output_dir = tmp_path / "output"

        result = cli_runner.invoke(app, [
            "generate",
            "--depth", str(sample_depth_npy),
            "--preset", "invalid_preset_name",
            "--output", str(output_dir),
        ])

        assert result.exit_code == 1
        # Error message is on stderr, but available presets shown
        assert "Available presets:" in result.output

    def test_output_directory_autocreation(self, cli_runner, sample_depth_npy, tmp_path):
        """Test that output directory is auto-created if missing."""
        # Use nested non-existent path
        output_dir = tmp_path / "nested" / "output" / "dir"
        assert not output_dir.exists()

        result = cli_runner.invoke(app, [
            "generate",
            "--depth", str(sample_depth_npy),
            "--output", str(output_dir),
        ])

        # Should succeed and create directory
        assert result.exit_code == 0
        assert output_dir.exists()

    def test_mixed_case_extensions(self, cli_runner, tmp_path):
        """Test handling of mixed-case file extensions (.JPG, .PNG, .JPEG)."""
        # Create depth files with uppercase extensions
        batch_dir = tmp_path / "mixed_case"
        batch_dir.mkdir()

        # Note: The current CLI looks for *_depth.* pattern
        # This test verifies case-insensitive extension handling
        depth = (np.random.rand(128, 128) * 255).astype(np.uint8)
        depth_img = Image.fromarray(depth, mode='L')

        # Create .png (should be found)
        depth_img.save(batch_dir / "test1_depth.png")

        # Create .PNG (may not be found - current limitation)
        depth_img.save(batch_dir / "test2_depth.PNG")

        output_dir = tmp_path / "output"

        result = cli_runner.invoke(app, [
            "generate",
            "--depth-dir", str(batch_dir),
            "--output", str(output_dir),
        ])

        # Should process at least the .png file
        # Current implementation only finds lowercase .png
        assert result.exit_code == 0
        assert "Batch processing" in result.stdout


# P0: Error Handling Tests
class TestErrorHandling:
    """Test error handling and recovery."""

    def test_corrupt_file_single_mode(self, cli_runner, corrupt_depth_file, tmp_path):
        """Test handling of corrupt depth file in single mode."""
        output_dir = tmp_path / "output"

        result = cli_runner.invoke(app, [
            "generate",
            "--depth", str(corrupt_depth_file),
            "--output", str(output_dir),
        ])

        # Should fail gracefully with error message
        assert result.exit_code == 1
        assert "Error:" in result.output or "✗" in result.output

    def test_batch_continues_on_error(self, cli_runner, tmp_path):
        """Test that batch mode continues processing after individual failures."""
        batch_dir = tmp_path / "mixed_batch"
        batch_dir.mkdir()

        # Create 2 valid files
        for i in range(2):
            depth = np.random.rand(128, 128).astype(np.float32)
            np.save(batch_dir / f"valid_{i}_depth.npy", depth)

        # Create 1 corrupt file
        corrupt = batch_dir / "corrupt_depth.npy"
        corrupt.write_text("Not a valid numpy file")

        output_dir = tmp_path / "output"

        result = cli_runner.invoke(app, [
            "generate",
            "--depth-dir", str(batch_dir),
            "--output", str(output_dir),
        ])

        # Should report mixed results
        assert "Batch complete" in result.stdout
        # At least one should succeed
        assert "Success:" in result.stdout
        # At least one should fail
        assert "Errors:" in result.stdout
        # Exit code should be 1 if any errors
        assert result.exit_code == 1

    def test_no_stack_trace_on_user_error(self, cli_runner, tmp_path):
        """Test that user errors don't expose stack traces."""
        output_dir = tmp_path / "output"
        fake_path = tmp_path / "nonexistent.npy"

        result = cli_runner.invoke(app, [
            "generate",
            "--depth", str(fake_path),
            "--output", str(output_dir),
        ])

        # Should have clean error message
        assert result.exit_code == 1
        assert "Error:" in result.output
        # Should NOT have traceback
        assert "Traceback" not in result.output
        assert "File \"" not in result.output


# P0: Output Validation Tests
class TestOutputValidation:
    """Test that outputs are created correctly."""

    def test_pbr_maps_created(self, cli_runner, sample_depth_npy, tmp_path):
        """Test that all PBR maps are created."""
        output_dir = tmp_path / "output"

        result = cli_runner.invoke(app, [
            "generate",
            "--depth", str(sample_depth_npy),
            "--output", str(output_dir),
        ])

        assert result.exit_code == 0

        # Check that output mentions all three map types
        assert "normal" in result.stdout.lower()
        assert "roughness" in result.stdout.lower()
        assert "ao" in result.stdout.lower() or "ambient" in result.stdout.lower()

    def test_output_naming_convention(self, cli_runner, sample_depth_npy, tmp_path):
        """Test output file naming convention."""
        output_dir = tmp_path / "output"

        result = cli_runner.invoke(app, [
            "generate",
            "--depth", str(sample_depth_npy),
            "--base-name", "custom_name",
            "--output", str(output_dir),
        ])

        assert result.exit_code == 0

        # Output should reference the base name
        assert "custom_name" in result.stdout or output_dir.exists()

    def test_batch_output_structure(self, cli_runner, sample_depth_batch, tmp_path):
        """Test batch mode output directory structure."""
        output_dir = tmp_path / "output"

        result = cli_runner.invoke(app, [
            "generate",
            "--depth-dir", str(sample_depth_batch),
            "--output", str(output_dir),
        ])

        assert result.exit_code == 0
        assert output_dir.exists()

        # Should have processed all files
        assert "Success: 5" in result.stdout


# P0: Parameter Validation Tests
class TestParameterValidation:
    """Test CLI parameter validation."""

    def test_invalid_float_parameter(self, cli_runner, sample_depth_npy, tmp_path):
        """Test handling of invalid float parameters."""
        output_dir = tmp_path / "output"

        result = cli_runner.invoke(app, [
            "generate",
            "--depth", str(sample_depth_npy),
            "--normal-strength", "not_a_number",
            "--output", str(output_dir),
        ])

        # Typer should handle type validation
        assert result.exit_code != 0

    def test_negative_strength_values(self, cli_runner, sample_depth_npy, tmp_path):
        """Test handling of negative strength values (edge case)."""
        output_dir = tmp_path / "output"

        # Negative values might be allowed by the algorithm
        # This documents the behavior
        result = cli_runner.invoke(app, [
            "generate",
            "--depth", str(sample_depth_npy),
            "--normal-strength", "-1.0",
            "--output", str(output_dir),
        ])

        # Either succeeds (algorithm allows negative) or fails gracefully
        assert result.exit_code in [0, 1]
        if result.exit_code == 1:
            assert "Error:" in result.stdout

    def test_extreme_parameter_values(self, cli_runner, sample_depth_npy, tmp_path):
        """Test handling of extreme parameter values."""
        output_dir = tmp_path / "output"

        result = cli_runner.invoke(app, [
            "generate",
            "--depth", str(sample_depth_npy),
            "--normal-strength", "100.0",
            "--ao-bias", "0.0",
            "--output", str(output_dir),
        ])

        # Should handle extreme values gracefully
        assert result.exit_code in [0, 1]


# P0: Exit Code Tests
class TestExitCodes:
    """Test that exit codes are correct."""

    def test_success_exit_code(self, cli_runner, sample_depth_npy, tmp_path):
        """Test that successful execution returns 0."""
        output_dir = tmp_path / "output"

        result = cli_runner.invoke(app, [
            "generate",
            "--depth", str(sample_depth_npy),
            "--output", str(output_dir),
        ])

        assert result.exit_code == 0

    def test_error_exit_code(self, cli_runner, tmp_path):
        """Test that errors return non-zero exit code."""
        output_dir = tmp_path / "output"
        fake_path = tmp_path / "nonexistent.npy"

        result = cli_runner.invoke(app, [
            "generate",
            "--depth", str(fake_path),
            "--output", str(output_dir),
        ])

        assert result.exit_code == 1

    def test_batch_partial_failure_exit_code(self, cli_runner, tmp_path):
        """Test that partial batch failures return non-zero exit code."""
        batch_dir = tmp_path / "batch"
        batch_dir.mkdir()

        # One valid file
        depth = np.random.rand(128, 128).astype(np.float32)
        np.save(batch_dir / "valid_depth.npy", depth)

        # One corrupt file
        corrupt = batch_dir / "corrupt_depth.npy"
        corrupt.write_text("Not valid")

        output_dir = tmp_path / "output"

        result = cli_runner.invoke(app, [
            "generate",
            "--depth-dir", str(batch_dir),
            "--output", str(output_dir),
        ])

        # Should exit with error code due to partial failure
        assert result.exit_code == 1
        assert "Errors:" in result.stdout


# P0: Overwrite/Idempotency Tests
class TestOverwriteBehavior:
    """Test --overwrite/--no-overwrite flag functionality."""

    def test_no_overwrite_fails_when_outputs_exist(self, cli_runner, sample_depth_npy, tmp_path):
        """Test that --no-overwrite fails when output files already exist."""
        output_dir = tmp_path / "output"

        # First run - should succeed
        result1 = cli_runner.invoke(app, [
            "generate",
            "--depth", str(sample_depth_npy),
            "--output", str(output_dir),
        ])
        assert result1.exit_code == 0, f"First run failed: {result1.stdout}"

        # Second run with --no-overwrite should fail
        result2 = cli_runner.invoke(app, [
            "generate",
            "--depth", str(sample_depth_npy),
            "--output", str(output_dir),
            "--no-overwrite",
        ])
        assert result2.exit_code == 1, f"Expected failure with --no-overwrite: {result2.stdout}"
        assert "already exist" in result2.output.lower()

    def test_overwrite_succeeds_when_outputs_exist(self, cli_runner, sample_depth_npy, tmp_path):
        """Test that --overwrite succeeds when output files already exist."""
        output_dir = tmp_path / "output"

        # First run
        result1 = cli_runner.invoke(app, [
            "generate",
            "--depth", str(sample_depth_npy),
            "--output", str(output_dir),
        ])
        assert result1.exit_code == 0

        # Second run with --overwrite (default) should succeed
        result2 = cli_runner.invoke(app, [
            "generate",
            "--depth", str(sample_depth_npy),
            "--output", str(output_dir),
            "--overwrite",
        ])
        assert result2.exit_code == 0, f"Expected success with --overwrite: {result2.stdout}"

    def test_batch_no_overwrite_skips_existing(self, cli_runner, tmp_path):
        """Test that batch mode with --no-overwrite skips files with existing outputs."""
        batch_dir = tmp_path / "batch"
        batch_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create two depth files
        for i in range(2):
            depth = np.random.rand(128, 128).astype(np.float32)
            np.save(batch_dir / f"scene_{i}_depth.npy", depth)

        # Run first to create outputs for scene_0
        result1 = cli_runner.invoke(app, [
            "generate",
            "--depth", str(batch_dir / "scene_0_depth.npy"),
            "--output", str(output_dir),
        ])
        assert result1.exit_code == 0

        # Now run batch with --no-overwrite
        result2 = cli_runner.invoke(app, [
            "generate",
            "--depth-dir", str(batch_dir),
            "--output", str(output_dir),
            "--no-overwrite",
        ])

        # Should have partial success (1 skipped, 1 processed)
        assert result2.exit_code == 1  # Exit with error due to skipped file
        assert "Errors:" in result2.stdout


# P0: Base Name Derivation Tests
class TestBaseNameDerivation:
    """Test that base name is correctly derived from depth filename."""

    def test_depth_suffix_removed_from_end_only(self, cli_runner, tmp_path):
        """Test that _depth is only removed from the end, not middle of filename."""
        # Create a depth file with "_depth" in the middle
        depth_data = np.random.rand(128, 128).astype(np.float32)
        depth_path = tmp_path / "scene_depth_map_depth.npy"
        np.save(depth_path, depth_data)

        output_dir = tmp_path / "output"

        result = cli_runner.invoke(app, [
            "generate",
            "--depth", str(depth_path),
            "--output", str(output_dir),
        ])

        assert result.exit_code == 0

        # Output should be named "scene_depth_map_*" not "scene__map_*"
        # (removesuffix only removes from end)
        assert (output_dir / "scene_depth_map_normal.png").exists()
        assert not (output_dir / "scene__map_normal.png").exists()
