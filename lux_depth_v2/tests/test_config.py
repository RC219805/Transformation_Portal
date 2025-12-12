"""Unit tests for config module."""
from __future__ import annotations

import pytest
from lux_depth_v2.config import (
    Preset,
    PipelineConfig,
    SegmentationConfig,
    ServiceConfig,
)


class TestPreset:
    """Test Preset enum."""

    def test_preset_values(self):
        """Test all preset enum values exist."""
        assert Preset.PHOTO_REALISTIC.value == "photo_realistic"
        assert Preset.INTERIOR_LUXURY.value == "interior_luxury"
        assert Preset.EXTERIOR_SHOWCASE.value == "exterior_showcase"
        assert Preset.ARCHITECTURAL.value == "architectural"
        assert Preset.ARCHIVAL_QUALITY.value == "archival_quality"

    def test_preset_from_string(self):
        """Test creating preset from string."""
        assert Preset("photo_realistic") == Preset.PHOTO_REALISTIC


class TestSegmentationConfig:
    """Test SegmentationConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        cfg = SegmentationConfig()
        assert cfg.backend == "auto"
        assert cfg.onnx_model_path is None
        assert cfg.onnx_labels_path is None
        # Production default: SegFormer-B5 (highest quality)
        assert cfg.segformer_model == "nvidia/segformer-b5-finetuned-ade-640-640"
        assert cfg.sam_checkpoint is None
        assert cfg.input_long_side == 768
        assert cfg.soften_sigma_px == 2.0
        assert cfg.min_confidence == 0.25
        # Production default: allow downloads for SegFormer-B5
        assert cfg.allow_downloads is True

    def test_custom_values(self):
        """Test custom configuration."""
        cfg = SegmentationConfig(
            backend="onnx",
            input_long_side=1024,
            soften_sigma_px=3.0,
            min_confidence=0.3,
        )
        assert cfg.backend == "onnx"
        assert cfg.input_long_side == 1024
        assert cfg.soften_sigma_px == 3.0
        assert cfg.min_confidence == 0.3


class TestServiceConfig:
    """Test ServiceConfig dataclass."""

    def test_default_values(self):
        """Test default service configuration."""
        cfg = ServiceConfig()
        assert cfg.enabled is False
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 8088
        assert cfg.workers == 1
        assert cfg.max_concurrency == 1

    def test_custom_values(self):
        """Test custom service configuration."""
        cfg = ServiceConfig(
            enabled=True,
            host="127.0.0.1",
            port=9000,
            workers=4,
        )
        assert cfg.enabled is True
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 9000
        assert cfg.workers == 4


class TestPipelineConfig:
    """Test PipelineConfig dataclass."""

    def test_default_values(self):
        """Test default pipeline configuration."""
        cfg = PipelineConfig()
        assert cfg.preset == Preset.PHOTO_REALISTIC
        assert cfg.upscale == 4
        assert cfg.upscaler_backend == "realesrgan"
        assert cfg.device == "auto"
        assert cfg.precision == "fp16"
        assert cfg.save_master is True
        assert cfg.save_upscaled is True
        assert cfg.enable_material is True
        assert cfg.material_strength == 0.75

    def test_preset_application_photo_realistic(self):
        """Test applying photo_realistic preset."""
        cfg = PipelineConfig(preset=Preset.PHOTO_REALISTIC)
        cfg.apply_preset()
        assert cfg.material_strength == 0.70
        assert cfg.temp_fg == 0.010
        assert cfg.sat_fg == 1.030
        assert cfg.detail_strength == 0.65

    def test_preset_application_interior_luxury(self):
        """Test applying interior_luxury preset."""
        cfg = PipelineConfig(preset=Preset.INTERIOR_LUXURY)
        cfg.apply_preset()
        assert cfg.material_strength == 0.90
        assert cfg.temp_fg == 0.013
        assert cfg.sat_fg == 1.045
        assert cfg.detail_strength == 0.70

    def test_preset_application_exterior_showcase(self):
        """Test applying exterior_showcase preset."""
        cfg = PipelineConfig(preset=Preset.EXTERIOR_SHOWCASE)
        cfg.apply_preset()
        assert cfg.material_strength == 0.80
        assert cfg.sat_fg == 1.055
        assert cfg.con_fg == 1.040

    def test_preset_application_architectural(self):
        """Test applying architectural preset."""
        cfg = PipelineConfig(preset=Preset.ARCHITECTURAL)
        cfg.apply_preset()
        assert cfg.material_strength == 0.75
        assert cfg.sat_fg == 1.020
        assert cfg.sharpen_fg == 0.10

    def test_preset_application_archival(self):
        """Test applying archival_quality preset."""
        cfg = PipelineConfig(preset=Preset.ARCHIVAL_QUALITY)
        cfg.apply_preset()
        assert cfg.material_strength == 0.60
        assert cfg.detail_strength == 0.55
        assert cfg.clarity_fg == 0.14

    def test_upscale_clamping(self):
        """Test upscale value is clamped to 2 or 4."""
        cfg = PipelineConfig(upscale=3)
        cfg.apply_preset()
        assert cfg.upscale == 4  # Invalid value clamped to 4

        cfg = PipelineConfig(upscale=2)
        cfg.apply_preset()
        assert cfg.upscale == 2

    def test_material_strength_clamping(self):
        """Test material_strength is clamped to [0.0, 1.25]."""
        cfg = PipelineConfig(material_strength=2.0)
        cfg.apply_preset()
        assert cfg.material_strength <= 1.25

        cfg = PipelineConfig(material_strength=-0.5)
        cfg.apply_preset()
        assert cfg.material_strength >= 0.0

    def test_nested_configs(self):
        """Test nested configuration objects."""
        cfg = PipelineConfig()
        assert isinstance(cfg.segmentation, SegmentationConfig)
        assert isinstance(cfg.service, ServiceConfig)
        assert cfg.segmentation.backend == "auto"
        assert cfg.service.enabled is False

    def test_depth_weight_parameters(self):
        """Test depth weight synthesis parameters."""
        cfg = PipelineConfig()
        assert 0 < cfg.fg_q < 1
        assert 0 < cfg.bg_q < 1
        assert cfg.fg_q < cfg.bg_q
        assert cfg.transition > 0

    def test_ai_validation_parameters(self):
        """Test AI validation thresholds."""
        cfg = PipelineConfig()
        assert cfg.validate_ai is True
        assert cfg.ai_color_warn < cfg.ai_color_fail
        assert cfg.ai_luma_warn < cfg.ai_luma_fail
        assert cfg.ai_color_warn > 0
        assert cfg.ai_luma_warn > 0

    def test_output_parameters(self):
        """Test output format parameters."""
        cfg = PipelineConfig()
        assert cfg.save_master is True
        assert cfg.save_upscaled is True
        assert 0 < cfg.preview_scale < 1
        assert cfg.skip_existing is True
        assert cfg.overwrite is False

    def test_processing_parameters(self):
        """Test processing pipeline parameters."""
        cfg = PipelineConfig()
        assert cfg.detail_sigma > 0
        assert cfg.clarity_sigma > 0
        assert cfg.sharpen_sigma > 0
        assert 0 <= cfg.detail_strength <= 1
        assert 0 < cfg.soft_clip_knee < 1

    def test_surface_tuple(self):
        """Test surfaces configuration."""
        cfg = PipelineConfig()
        assert isinstance(cfg.surfaces, tuple)
        assert "wood" in cfg.surfaces
        assert "metal" in cfg.surfaces
        assert "glass" in cfg.surfaces
