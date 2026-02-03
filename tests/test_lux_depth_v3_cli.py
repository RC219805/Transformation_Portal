#!/usr/bin/env python3
"""Tests for lux_depth_v3 CLI.

Tests cover:
- CLI argument parsing and validation
- Quality tier configuration
- Preset resolution
- Depth backend selection
- License validation
- Feature toggle handling
"""

import json
import pytest
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock
from PIL import Image

from transformation_portal.lux_depth_v3.cli import (
    app,
    QualityTier,
    _parse_bool_flag,
    _resolve_preset,
    _resolve_model_variant,
    _apply_quality_tier,
)
from transformation_portal.lux_depth_v3.config import EnhanceConfig, ModelVariant, Preset



@pytest.fixture
def cli_runner():
    """Create a CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def sample_image_dir(tmp_path):
    """Create a directory with sample images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    # Create 3 sample images
    for i in range(3):
        img = Image.new('RGB', (512, 512), color=(i*80, i*80, i*80))
        img.save(img_dir / f"image_{i:02d}.jpg")

    return img_dir


@pytest.fixture
def empty_dir(tmp_path):
    """Create an empty directory."""
    empty = tmp_path / "empty"
    empty.mkdir()
    return empty


class TestCLIInfo:
    """Test the info command."""

    def test_info_command(self, cli_runner):
        """Test that info command displays pipeline information."""
        result = cli_runner.invoke(app, ["info"])

        assert result.exit_code == 0
        assert "Quality Tiers:" in result.stdout
        assert "apex" in result.stdout
        assert "Presets:" in result.stdout
        assert "Depth Backends:" in result.stdout


class TestHelperFunctions:
    """Test helper functions used by the CLI."""

    def test_parse_bool_flag_true(self):
        """Test parsing of true boolean flags."""
        assert _parse_bool_flag("true") is True
        assert _parse_bool_flag("True") is True
        assert _parse_bool_flag("TRUE") is True
        assert _parse_bool_flag("yes") is True
        assert _parse_bool_flag("on") is True
        assert _parse_bool_flag("1") is True

    def test_parse_bool_flag_false(self):
        """Test parsing of false boolean flags."""
        assert _parse_bool_flag("false") is False
        assert _parse_bool_flag("False") is False
        assert _parse_bool_flag("no") is False
        assert _parse_bool_flag("off") is False
        assert _parse_bool_flag("0") is False

    def test_resolve_preset_standard(self):
        """Test resolving standard presets."""
        assert _resolve_preset("premium") == Preset.LUXURY_ESTATE
        assert _resolve_preset("luxury_estate") == Preset.LUXURY_ESTATE
        assert _resolve_preset("architectural_interior") == Preset.ARCHITECTURAL_INTERIOR
        assert _resolve_preset("architectural_exterior") == Preset.ARCHITECTURAL_EXTERIOR
        assert _resolve_preset("default") == Preset.DEFAULT

    def test_resolve_preset_research_model(self):
        """Test resolving research model presets."""
        # Research presets should map to luxury estate
        preset = _resolve_preset("depth-anything-v3.1-research-m4")
        assert preset == Preset.LUXURY_ESTATE

    def test_resolve_model_variant_depth_pro(self):
        """Test that depth_pro backend returns None for model variant."""
        variant = _resolve_model_variant("premium", "depth_pro")
        assert variant is None

    def test_resolve_model_variant_da3(self):
        """Test DA3 model variant resolution."""
        variant = _resolve_model_variant("premium", None)
        assert variant == ModelVariant.METRIC_LARGE

    def test_apply_quality_tier_draft(self):
        """Test quality tier application for draft."""
        config = EnhanceConfig()
        config = _apply_quality_tier(config, QualityTier.DRAFT)

        assert config.model_variant == ModelVariant.METRIC_SMALL
        assert config.pbr_normal_strength == 0.8
        assert config.save_float_depth is False

    def test_apply_quality_tier_standard(self):
        """Test quality tier application for standard."""
        config = EnhanceConfig()
        config = _apply_quality_tier(config, QualityTier.STANDARD)

        assert config.model_variant == ModelVariant.METRIC_BASE
        assert config.pbr_normal_strength == 1.0
        assert config.save_float_depth is True

    def test_apply_quality_tier_premium(self):
        """Test quality tier application for premium."""
        config = EnhanceConfig()
        config = _apply_quality_tier(config, QualityTier.PREMIUM)

        assert config.model_variant == ModelVariant.METRIC_LARGE
        assert config.pbr_normal_strength == 1.2
        assert config.save_float_depth is True

    def test_apply_quality_tier_apex(self):
        """Test quality tier application for APEX."""
        config = EnhanceConfig()
        config = _apply_quality_tier(config, QualityTier.APEX)

        assert config.model_variant == ModelVariant.METRIC_LARGE
        assert config.pbr_normal_strength == 1.5
        assert config.pbr_normal_blur_radius == 0
        assert config.save_float_depth is True
        assert config.verify_depth_writes is True
        assert config.enable_manifest_cache is True


class TestCLIValidation:
    """Test CLI input validation."""

    def test_missing_input_dir(self, cli_runner, tmp_path):
        """Test that missing input directory is caught."""
        result = cli_runner.invoke(app, [
            "process",
            "--input-dir", str(tmp_path / "nonexistent"),
            "--output-dir", str(tmp_path / "output"),
        ])

        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()

    def test_empty_input_dir(self, cli_runner, empty_dir, tmp_path):
        """Test that empty input directory is caught."""
        result = cli_runner.invoke(app, [
            "process",
            "--input-dir", str(empty_dir),
            "--output-dir", str(tmp_path / "output"),
        ])

        assert result.exit_code == 1
        assert "No images found" in result.stdout

    def test_depth_pro_requires_non_commercial(self, cli_runner, sample_image_dir, tmp_path):
        """Test that depth_pro backend requires non-commercial acknowledgment."""
        result = cli_runner.invoke(app, [
            "process",
            "--input-dir", str(sample_image_dir),
            "--output-dir", str(tmp_path / "output"),
            "--depth-backend", "depth_pro",
            "--non-commercial-ok", "false",
        ])

        assert result.exit_code == 1
        assert "research-only" in result.stdout.lower()

    def test_depth_pro_requires_license_acceptance(self, cli_runner, sample_image_dir, tmp_path):
        """Test that depth_pro backend requires explicit license acceptance."""
        result = cli_runner.invoke(app, [
            "process",
            "--input-dir", str(sample_image_dir),
            "--output-dir", str(tmp_path / "output"),
            "--depth-backend", "depth_pro",
            "--non-commercial-ok", "true",
            "--accept-apple-depth-pro-research-license", "false",
        ])

        assert result.exit_code == 1
        assert "license acceptance" in result.stdout.lower()


class TestCLIProcessCommand:
    """Test the process command with mocked orchestrator."""

    @patch('transformation_portal.lux_depth_v3.cli.EnhanceOrchestrator')
    def test_basic_processing(self, mock_orchestrator_class, cli_runner, sample_image_dir, tmp_path):
        """Test basic image processing with mocked orchestrator."""
        # Setup mock
        mock_orchestrator = MagicMock()
        mock_orchestrator.enhance_image.return_value = {}
        mock_orchestrator_class.return_value = mock_orchestrator

        output_dir = tmp_path / "output"

        result = cli_runner.invoke(app, [
            "process",
            "--input-dir", str(sample_image_dir),
            "--output-dir", str(output_dir),
            "--preset", "premium",
            "--quality-tier", "apex",
        ])

        assert result.exit_code == 0
        assert "Processing Complete" in result.stdout
        assert "3/3" in result.stdout or "Successful: 3" in result.stdout
        assert mock_orchestrator.enhance_image.call_count == 3

    @patch('transformation_portal.lux_depth_v3.cli.EnhanceOrchestrator')
    def test_quality_tier_apex(self, mock_orchestrator_class, cli_runner, sample_image_dir, tmp_path):
        """Test APEX quality tier configuration."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.enhance_image.return_value = {}
        mock_orchestrator_class.return_value = mock_orchestrator

        result = cli_runner.invoke(app, [
            "process",
            "--input-dir", str(sample_image_dir),
            "--output-dir", str(tmp_path / "output"),
            "--quality-tier", "apex",
            "--limit", "1",
        ])

        assert result.exit_code == 0

        # Check that orchestrator was initialized with APEX settings
        call_args = mock_orchestrator_class.call_args
        config = call_args[0][0]

        # APEX should enable PBR and depth caching
        assert config.generate_pbr is True
        assert config.enable_depth_cache is True
        assert config.pbr_normal_strength == 1.5

    @patch('transformation_portal.lux_depth_v3.cli.EnhanceOrchestrator')
    def test_pbr_toggle_on(self, mock_orchestrator_class, cli_runner, sample_image_dir, tmp_path):
        """Test PBR toggle set to ON."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.enhance_image.return_value = {}
        mock_orchestrator_class.return_value = mock_orchestrator

        result = cli_runner.invoke(app, [
            "process",
            "--input-dir", str(sample_image_dir),
            "--output-dir", str(tmp_path / "output"),
            "--pbr", "on",
            "--limit", "1",
        ])

        assert result.exit_code == 0
        config = mock_orchestrator_class.call_args[0][0]
        assert config.generate_pbr is True

    @patch('transformation_portal.lux_depth_v3.cli.EnhanceOrchestrator')
    def test_pbr_toggle_off(self, mock_orchestrator_class, cli_runner, sample_image_dir, tmp_path):
        """Test PBR toggle set to OFF."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.enhance_image.return_value = {}
        mock_orchestrator_class.return_value = mock_orchestrator

        result = cli_runner.invoke(app, [
            "process",
            "--input-dir", str(sample_image_dir),
            "--output-dir", str(tmp_path / "output"),
            "--pbr", "off",
            "--limit", "1",
        ])

        assert result.exit_code == 0
        config = mock_orchestrator_class.call_args[0][0]
        assert config.generate_pbr is False

    @patch('transformation_portal.lux_depth_v3.cli.EnhanceOrchestrator')
    def test_overwrite_flag(self, mock_orchestrator_class, cli_runner, sample_image_dir, tmp_path):
        """Test overwrite flag sets force flags."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.enhance_image.return_value = {}
        mock_orchestrator_class.return_value = mock_orchestrator

        result = cli_runner.invoke(app, [
            "process",
            "--input-dir", str(sample_image_dir),
            "--output-dir", str(tmp_path / "output"),
            "--overwrite",
            "--limit", "1",
        ])

        assert result.exit_code == 0
        config = mock_orchestrator_class.call_args[0][0]
        assert config.force_depth is True
        assert config.force_v2 is True

    @patch('transformation_portal.lux_depth_v3.cli.EnhanceOrchestrator')
    def test_limit_flag(self, mock_orchestrator_class, cli_runner, sample_image_dir, tmp_path):
        """Test limit flag restricts number of images processed."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.enhance_image.return_value = {}
        mock_orchestrator_class.return_value = mock_orchestrator

        result = cli_runner.invoke(app, [
            "process",
            "--input-dir", str(sample_image_dir),
            "--output-dir", str(tmp_path / "output"),
            "--limit", "2",
        ])

        assert result.exit_code == 0
        # Should only process 2 images out of 3
        assert mock_orchestrator.enhance_image.call_count == 2

    @patch('transformation_portal.lux_depth_v3.cli.EnhanceOrchestrator')
    def test_fail_fast(self, mock_orchestrator_class, cli_runner, sample_image_dir, tmp_path):
        """Test fail-fast flag stops on first error."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.enhance_image.side_effect = Exception("Processing failed")
        mock_orchestrator_class.return_value = mock_orchestrator

        result = cli_runner.invoke(app, [
            "process",
            "--input-dir", str(sample_image_dir),
            "--output-dir", str(tmp_path / "output"),
            "--fail-fast",
        ])

        assert result.exit_code == 1
        # Should stop after first error, not process all 3 images
        assert mock_orchestrator.enhance_image.call_count == 1
        assert "FAIL FAST" in result.stdout

    @patch('transformation_portal.lux_depth_v3.cli.EnhanceOrchestrator')
    def test_json_output(self, mock_orchestrator_class, cli_runner, sample_image_dir, tmp_path):
        """Test JSON output format."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.enhance_image.return_value = {}
        mock_orchestrator_class.return_value = mock_orchestrator

        result = cli_runner.invoke(app, [
            "process",
            "--input-dir", str(sample_image_dir),
            "--output-dir", str(tmp_path / "output"),
            "--limit", "1",
            "--json",
        ])

        assert result.exit_code == 0

        # Parse JSON output
        output = json.loads(result.stdout)

        assert output["status"] == "success"
        assert output["success_count"] == 1
        assert output["error_count"] == 0
        assert "quality_tier" in output
        assert "preset" in output

    @patch('transformation_portal.lux_depth_v3.cli.EnhanceOrchestrator')
    def test_non_commercial_flag(self, mock_orchestrator_class, cli_runner, sample_image_dir, tmp_path):
        """Test non-commercial flag is passed to config."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.enhance_image.return_value = {}
        mock_orchestrator_class.return_value = mock_orchestrator

        result = cli_runner.invoke(app, [
            "process",
            "--input-dir", str(sample_image_dir),
            "--output-dir", str(tmp_path / "output"),
            "--non-commercial-ok", "true",
            "--limit", "1",
        ])

        assert result.exit_code == 0
        config = mock_orchestrator_class.call_args[0][0]
        assert config.non_commercial_ok is True


class TestCLIErrorHandling:
    """Test CLI error handling."""

    @patch('transformation_portal.lux_depth_v3.cli.EnhanceOrchestrator')
    def test_orchestrator_init_failure(self, mock_orchestrator_class, cli_runner, sample_image_dir, tmp_path):
        """Test handling of orchestrator initialization failure."""
        mock_orchestrator_class.side_effect = Exception("Failed to initialize")

        result = cli_runner.invoke(app, [
            "process",
            "--input-dir", str(sample_image_dir),
            "--output-dir", str(tmp_path / "output"),
        ])

        assert result.exit_code == 1
        assert "Failed to initialize orchestrator" in result.stdout

    @patch('transformation_portal.lux_depth_v3.cli.EnhanceOrchestrator')
    def test_partial_success(self, mock_orchestrator_class, cli_runner, sample_image_dir, tmp_path):
        """Test handling of partial processing success."""
        mock_orchestrator = MagicMock()

        # First image succeeds, second and third fail
        mock_orchestrator.enhance_image.side_effect = [
            {},
            Exception("Processing error"),
            Exception("Another error"),
        ]
        mock_orchestrator_class.return_value = mock_orchestrator

        result = cli_runner.invoke(app, [
            "process",
            "--input-dir", str(sample_image_dir),
            "--output-dir", str(tmp_path / "output"),
        ])

        assert result.exit_code == 1  # Exit with error due to failures
        assert "Successful: 1" in result.stdout
        assert "Failed:     2" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
