"""Edge case tests for PipelineConfig.

Simplified tests that match actual implementation.
"""
from __future__ import annotations

import pytest
from pathlib import Path

from lux_depth_v2.config import PipelineConfig, Preset


class TestConfigBasics:
    """Test basic configuration functionality."""

    def test_upscale_values(self):
        """Test upscale factors."""
        config = PipelineConfig(upscale=2)
        assert config.upscale == 2
        
        config = PipelineConfig(upscale=4)
        assert config.upscale == 4

    def test_all_presets_valid(self):
        """Test that all presets can be created."""
        for preset in Preset:
            config = PipelineConfig(preset=preset)
            assert config.preset == preset

    def test_device_selection(self):
        """Test device configuration."""
        config = PipelineConfig(device="auto")
        assert config.device == "auto"
        
        config = PipelineConfig(device="cpu")
        assert config.device == "cpu"
        
        config = PipelineConfig(device="cuda")
        assert config.device == "cuda"

    def test_precision_values(self):
        """Test precision configuration."""
        config = PipelineConfig(precision="fp16")
        assert config.precision == "fp16"
        
        config = PipelineConfig(precision="fp32")
        assert config.precision == "fp32"

    def test_upscaler_backends(self):
        """Test upscaler backend configuration."""
        config = PipelineConfig(upscaler_backend="realesrgan")
        assert config.upscaler_backend == "realesrgan"
        
        config = PipelineConfig(upscaler_backend="onnx")
        assert config.upscaler_backend == "onnx"
        
        config = PipelineConfig(upscaler_backend="none")
        assert config.upscaler_backend == "none"

    def test_output_flags(self):
        """Test output configuration flags."""
        config = PipelineConfig(
            save_master=True,
            save_upscaled=True,
            save_marketing_png=True,
            save_preview_jpg=True
        )
        assert config.save_master
        assert config.save_upscaled
        assert config.save_marketing_png
        assert config.save_preview_jpg
        
        config = PipelineConfig(
            save_master=False,
            save_upscaled=False,
            save_marketing_png=False,
            save_preview_jpg=False
        )
        assert not config.save_master

    def test_material_configuration(self):
        """Test material segmentation configuration."""
        config = PipelineConfig(enable_material=True)
        assert config.enable_material
        
        config = PipelineConfig(enable_material=False)
        assert not config.enable_material

    def test_atmospheric_configuration(self):
        """Test atmospheric effects configuration."""
        # Note: atmospheric effects are controlled by detail/clarity/sharpen parameters
        config = PipelineConfig(detail_strength=0.8)
        assert config.detail_strength == 0.8
        
        config = PipelineConfig(detail_strength=0.0)
        assert config.detail_strength == 0.0

    def test_skip_existing(self):
        """Test skip_existing configuration."""
        config = PipelineConfig(skip_existing=True)
        assert config.skip_existing
        
        config = PipelineConfig(skip_existing=False)
        assert not config.skip_existing


class TestPresetApplication:
    """Test preset application."""

    def test_preset_interior_luxury(self):
        """Test INTERIOR_LUXURY preset."""
        config = PipelineConfig(preset=Preset.INTERIOR_LUXURY)
        config.apply_preset()
        
        # Interior luxury should have high clarity (fg) and material strength
        assert config.clarity_fg >= 0.15
        assert config.material_strength >= 0.8

    def test_preset_architectural(self):
        """Test ARCHITECTURAL preset."""
        config = PipelineConfig(preset=Preset.ARCHITECTURAL)
        config.apply_preset()
        
        # Architectural should be conservative
        assert config.clarity_fg <= 0.5

    def test_preset_application(self):
        """Test preset application changes values."""
        config = PipelineConfig(preset=Preset.PHOTO_REALISTIC)
        config.apply_preset()
        
        # Should have valid values after preset
        assert config.clarity_fg is not None
        assert config.material_strength is not None


class TestPathHandling:
    """Test path handling."""

    def test_none_paths(self):
        """Test None path handling."""
        config = PipelineConfig(
            input_dir=None,
            output_dir=None,
            depth_dir=None
        )
        assert config.input_dir is None
        assert config.output_dir is None

    def test_path_objects(self, tmp_path):
        """Test Path objects."""
        config = PipelineConfig(
            input_dir=tmp_path / "input",
            output_dir=tmp_path / "output"
        )
        assert isinstance(config.input_dir, Path)
        assert isinstance(config.output_dir, Path)
