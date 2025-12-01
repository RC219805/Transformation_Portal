"""
Tests for the End-to-End 4K Rendering Enhancement Pipeline.

Tests cover:
- Pipeline initialization and configuration
- Individual processing stages
- Quality assessment metrics
- Preset loading
- Batch processing
- Output generation
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Import pipeline components
from transformation_portal.pipelines.rendering_4k_pipeline import (
    ColorGradingConfig,
    DepthConfig,
    DeviceType,
    MaterialResponseConfig,
    OutputConfig,
    PipelineConfig,
    ProcessingResult,
    QualityAssessor,
    QualityFeedbackConfig,
    QualityLevel,
    QualityMetrics,
    Rendering4KPipeline,
    ToneMappingConfig,
    ToneMappingMethod,
    UpscalingConfig,
    apply_color_grading,
    apply_material_response,
    apply_tone_mapping,
    apply_upscaling,
    estimate_depth_simple,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_image_np():
    """Create a sample RGB image as numpy array."""
    # Create a gradient image with some color variation
    h, w = 256, 384
    r = np.linspace(0.2, 0.8, w)
    g = np.linspace(0.3, 0.7, h)[:, np.newaxis]
    b = np.ones((h, w)) * 0.5

    image = np.stack([
        np.broadcast_to(r, (h, w)),
        np.broadcast_to(g, (h, w)),
        b,
    ], axis=2).astype(np.float32)

    return image


@pytest.fixture
def sample_image_pil(sample_image_np):
    """Create a sample PIL Image."""
    img_uint8 = (sample_image_np * 255).astype(np.uint8)
    return Image.fromarray(img_uint8, mode='RGB')


@pytest.fixture
def sample_depth_map():
    """Create a sample depth map."""
    h, w = 256, 384
    # Simple gradient depth map (near at top, far at bottom)
    depth = np.linspace(0, 1, h)[:, np.newaxis]
    depth = np.broadcast_to(depth, (h, w)).astype(np.float32)
    return depth


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_image_file(sample_image_pil):
    """Create a temporary image file."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        sample_image_pil.save(f.name)
        yield Path(f.name)


# =============================================================================
# Configuration Tests
# =============================================================================

class TestPipelineConfig:
    """Tests for pipeline configuration classes."""

    def test_default_pipeline_config(self):
        """Test default pipeline configuration values."""
        config = PipelineConfig()

        assert config.name == "default"
        assert config.quality_level == QualityLevel.HIGH
        assert config.depth.enabled is True
        assert config.tone_mapping.enabled is True
        assert config.material_response.enabled is True
        assert config.color_grading.enabled is True

    def test_depth_config_defaults(self):
        """Test DepthConfig default values."""
        config = DepthConfig()

        assert config.enabled is True
        assert config.model_variant == "small"
        assert config.backend == "auto"
        assert config.num_zones == 3
        assert config.cache_enabled is True

    def test_tone_mapping_config(self):
        """Test ToneMappingConfig values."""
        config = ToneMappingConfig(
            method=ToneMappingMethod.FILMIC,
            exposure=0.5,
            contrast=1.2,
        )

        assert config.method == ToneMappingMethod.FILMIC
        assert config.exposure == 0.5
        assert config.contrast == 1.2

    def test_material_response_config(self):
        """Test MaterialResponseConfig values."""
        config = MaterialResponseConfig(
            strength=0.8,
            texture_boost=0.35,
        )

        assert config.strength == 0.8
        assert config.texture_boost == 0.35
        assert "wood" in config.surface_types
        assert "metal" in config.surface_types

    def test_output_config(self):
        """Test OutputConfig values."""
        config = OutputConfig(
            master_tiff_16bit=True,
            delivery_jpeg=True,
            jpeg_quality=95,
        )

        assert config.master_tiff_16bit is True
        assert config.delivery_jpeg is True
        assert config.jpeg_quality == 95


# =============================================================================
# Quality Assessment Tests
# =============================================================================

class TestQualityAssessor:
    """Tests for the quality assessment module."""

    def test_quality_assessor_initialization(self):
        """Test QualityAssessor initialization."""
        config = QualityFeedbackConfig()
        assessor = QualityAssessor(config)

        assert assessor.config == config
        assert "sharpness" in assessor._metric_weights

    def test_assess_returns_metrics(self, sample_image_np):
        """Test that assess() returns QualityMetrics."""
        config = QualityFeedbackConfig()
        assessor = QualityAssessor(config)

        metrics = assessor.assess(sample_image_np)

        assert isinstance(metrics, QualityMetrics)
        assert 0 <= metrics.sharpness <= 1
        assert 0 <= metrics.contrast <= 1
        assert 0 <= metrics.colorfulness <= 1
        assert 0 <= metrics.exposure_balance <= 1
        assert 0 <= metrics.overall_score <= 1

    def test_assess_reasonable_scores(self, sample_image_np):
        """Test that assess() returns reasonable score values."""
        config = QualityFeedbackConfig()
        assessor = QualityAssessor(config)

        metrics = assessor.assess(sample_image_np)

        # The sample image should have decent quality metrics
        assert metrics.overall_score > 0.1  # Not completely black/white
        assert metrics.contrast > 0  # Should have some contrast

    def test_suggest_adjustments_low_quality(self):
        """Test adjustment suggestions for low-quality metrics."""
        config = QualityFeedbackConfig()
        assessor = QualityAssessor(config)

        # Create low-quality metrics
        metrics = QualityMetrics(
            sharpness=0.2,
            contrast=0.2,
            colorfulness=0.2,
            exposure_balance=0.2,
            noise_level=0.5,
        )

        adjustments = assessor.suggest_adjustments(metrics)

        assert "clarity_boost" in adjustments
        assert "contrast_increase" in adjustments
        assert "saturation_boost" in adjustments

    def test_quality_metrics_to_dict(self):
        """Test QualityMetrics.to_dict()."""
        metrics = QualityMetrics(
            sharpness=0.8,
            contrast=0.7,
            colorfulness=0.6,
            exposure_balance=0.9,
            overall_score=0.75,
        )

        d = metrics.to_dict()

        assert d["sharpness"] == 0.8
        assert d["contrast"] == 0.7
        assert d["overall_score"] == 0.75


# =============================================================================
# Image Processing Tests
# =============================================================================

class TestToneMapping:
    """Tests for tone mapping functions."""

    def test_tone_mapping_reinhard(self, sample_image_np):
        """Test Reinhard tone mapping."""
        config = ToneMappingConfig(method=ToneMappingMethod.REINHARD)

        result = apply_tone_mapping(sample_image_np, config)

        assert result.shape == sample_image_np.shape
        assert result.dtype == np.float32
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_tone_mapping_filmic(self, sample_image_np):
        """Test Filmic (Hable) tone mapping."""
        config = ToneMappingConfig(method=ToneMappingMethod.FILMIC)

        result = apply_tone_mapping(sample_image_np, config)

        assert result.shape == sample_image_np.shape
        assert result.dtype == np.float32
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_tone_mapping_aces(self, sample_image_np):
        """Test ACES tone mapping."""
        config = ToneMappingConfig(method=ToneMappingMethod.ACES)

        result = apply_tone_mapping(sample_image_np, config)

        assert result.shape == sample_image_np.shape
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_tone_mapping_agx(self, sample_image_np):
        """Test AgX tone mapping."""
        config = ToneMappingConfig(method=ToneMappingMethod.AGX)

        result = apply_tone_mapping(sample_image_np, config)

        assert result.shape == sample_image_np.shape
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_tone_mapping_exposure_adjustment(self, sample_image_np):
        """Test exposure adjustment in tone mapping."""
        config_bright = ToneMappingConfig(exposure=1.0)
        config_dark = ToneMappingConfig(exposure=-1.0)

        result_bright = apply_tone_mapping(sample_image_np, config_bright)
        result_dark = apply_tone_mapping(sample_image_np, config_dark)

        # Brighter image should have higher mean
        assert np.mean(result_bright) > np.mean(result_dark)

    def test_tone_mapping_disabled(self, sample_image_np):
        """Test disabled tone mapping returns clipped input."""
        config = ToneMappingConfig(enabled=False)

        result = apply_tone_mapping(sample_image_np, config)

        # Should be essentially unchanged (just clipped)
        np.testing.assert_allclose(result, np.clip(sample_image_np, 0, 1), rtol=1e-5)


class TestMaterialResponse:
    """Tests for Material Response functions."""

    def test_material_response_basic(self, sample_image_np, sample_depth_map):
        """Test basic Material Response enhancement."""
        config = MaterialResponseConfig(strength=0.7)

        result = apply_material_response(sample_image_np, sample_depth_map, config)

        assert result.shape == sample_image_np.shape
        assert result.dtype == np.float32
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_material_response_without_depth(self, sample_image_np):
        """Test Material Response without depth map."""
        config = MaterialResponseConfig(strength=0.7)

        result = apply_material_response(sample_image_np, None, config)

        assert result.shape == sample_image_np.shape
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_material_response_disabled(self, sample_image_np, sample_depth_map):
        """Test disabled Material Response returns input unchanged."""
        config = MaterialResponseConfig(enabled=False)

        result = apply_material_response(sample_image_np, sample_depth_map, config)

        np.testing.assert_array_equal(result, sample_image_np)

    def test_material_response_strength_effect(self, sample_image_np, sample_depth_map):
        """Test that higher strength produces more change."""
        config_low = MaterialResponseConfig(strength=0.2)
        config_high = MaterialResponseConfig(strength=0.9)

        result_low = apply_material_response(sample_image_np, sample_depth_map, config_low)
        result_high = apply_material_response(sample_image_np, sample_depth_map, config_high)

        # Higher strength should produce larger difference from original
        diff_low = np.abs(result_low - sample_image_np).mean()
        diff_high = np.abs(result_high - sample_image_np).mean()

        assert diff_high > diff_low


class TestColorGrading:
    """Tests for color grading functions."""

    def test_color_grading_basic(self, sample_image_np):
        """Test basic color grading."""
        config = ColorGradingConfig(saturation=1.1, vibrance=1.1)

        result = apply_color_grading(sample_image_np, config)

        assert result.shape == sample_image_np.shape
        assert result.dtype == np.float32
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_color_grading_temperature_shift(self, sample_image_np):
        """Test temperature shift in color grading."""
        config_warm = ColorGradingConfig(temperature_shift=(1.1, 1.0, 0.9))
        config_cool = ColorGradingConfig(temperature_shift=(0.9, 1.0, 1.1))

        result_warm = apply_color_grading(sample_image_np, config_warm)
        result_cool = apply_color_grading(sample_image_np, config_cool)

        # Warm should have higher red, cool should have higher blue
        assert np.mean(result_warm[..., 0]) > np.mean(result_cool[..., 0])
        assert np.mean(result_cool[..., 2]) > np.mean(result_warm[..., 2])

    def test_color_grading_disabled(self, sample_image_np):
        """Test disabled color grading returns input unchanged."""
        config = ColorGradingConfig(enabled=False)

        result = apply_color_grading(sample_image_np, config)

        np.testing.assert_array_equal(result, sample_image_np)


class TestUpscaling:
    """Tests for upscaling functions."""

    def test_upscaling_basic(self, sample_image_pil):
        """Test basic upscaling."""
        config = UpscalingConfig(target_resolution=(1920, 1080))

        result = apply_upscaling(sample_image_pil, config)

        assert isinstance(result, Image.Image)
        # Should be upscaled (but maintain aspect ratio)
        assert max(result.size) >= max(sample_image_pil.size)

    def test_upscaling_disabled(self, sample_image_pil):
        """Test disabled upscaling returns input unchanged."""
        config = UpscalingConfig(enabled=False)

        result = apply_upscaling(sample_image_pil, config)

        assert result.size == sample_image_pil.size

    def test_upscaling_already_large(self, sample_image_pil):
        """Test upscaling when image is already at target size."""
        # Target smaller than image
        config = UpscalingConfig(target_resolution=(128, 128))

        result = apply_upscaling(sample_image_pil, config)

        # Should remain at original size since already larger
        assert result.size == sample_image_pil.size


class TestDepthEstimation:
    """Tests for depth estimation functions."""

    def test_simple_depth_estimation(self, sample_image_np):
        """Test simple depth estimation fallback."""
        depth = estimate_depth_simple(sample_image_np)

        assert depth.shape == sample_image_np.shape[:2]
        assert depth.dtype == np.float32
        assert np.all(depth >= 0)
        assert np.all(depth <= 1)


# =============================================================================
# Pipeline Tests
# =============================================================================

class TestRendering4KPipeline:
    """Tests for the main Rendering4KPipeline class."""

    def test_pipeline_from_preset(self):
        """Test creating pipeline from preset."""
        pipeline = Rendering4KPipeline.from_preset("default")

        assert pipeline.config.name == "default"
        assert isinstance(pipeline.device, DeviceType)

    def test_pipeline_from_preset_luxury_estate(self):
        """Test luxury_estate preset."""
        pipeline = Rendering4KPipeline.from_preset("luxury_estate")

        assert pipeline.config.name == "luxury_estate"
        assert pipeline.config.material_response.strength == 0.75
        assert pipeline.config.color_grading.saturation == 1.08

    def test_pipeline_from_preset_preview(self):
        """Test preview preset (fast mode)."""
        pipeline = Rendering4KPipeline.from_preset("preview")

        assert pipeline.config.name == "preview"
        assert pipeline.config.quality_level == QualityLevel.PREVIEW
        assert pipeline.config.depth.enabled is False
        assert pipeline.config.upscaling.enabled is False

    def test_pipeline_from_preset_invalid(self):
        """Test invalid preset raises error."""
        with pytest.raises(ValueError, match="Unknown preset"):
            Rendering4KPipeline.from_preset("nonexistent")

    def test_pipeline_available_presets(self):
        """Test that expected presets are available."""
        presets = Rendering4KPipeline.PRESETS

        assert "default" in presets
        assert "luxury_estate" in presets
        assert "aerial_exterior" in presets
        assert "editorial" in presets
        assert "preview" in presets

    def test_pipeline_process_single_image(self, temp_image_file, temp_output_dir):
        """Test processing a single image."""
        pipeline = Rendering4KPipeline.from_preset("preview")  # Fast preset

        result = pipeline.process(temp_image_file, temp_output_dir)

        assert isinstance(result, ProcessingResult)
        assert isinstance(result.image, Image.Image)
        assert result.total_duration_ms > 0
        assert len(result.stage_metrics) > 0

    def test_pipeline_process_generates_outputs(self, temp_image_file, temp_output_dir):
        """Test that processing generates expected output files."""
        pipeline = Rendering4KPipeline.from_preset("preview")
        pipeline.config.output.delivery_jpeg = True
        pipeline.config.output.master_tiff_16bit = False  # Skip TIFF for speed

        result = pipeline.process(temp_image_file, temp_output_dir)

        assert "delivery_jpeg" in result.output_paths
        assert result.output_paths["delivery_jpeg"].exists()

    def test_pipeline_stage_metrics(self, temp_image_file, temp_output_dir):
        """Test that stage metrics are recorded."""
        pipeline = Rendering4KPipeline.from_preset("preview")

        result = pipeline.process(temp_image_file, temp_output_dir)

        stage_names = [m.name for m in result.stage_metrics]
        assert "input_validation" in stage_names
        assert "tone_mapping" in stage_names
        assert "output_generation" in stage_names

    def test_pipeline_quality_assessment(self, temp_image_file, temp_output_dir):
        """Test quality assessment in pipeline."""
        pipeline = Rendering4KPipeline.from_preset("default")
        pipeline.config.quality_feedback.enabled = True

        result = pipeline.process(temp_image_file, temp_output_dir)

        assert result.quality_metrics is not None
        assert 0 <= result.quality_score <= 1

    def test_pipeline_cache_clearing(self):
        """Test depth cache clearing."""
        pipeline = Rendering4KPipeline.from_preset("default")

        # Add some items to cache
        pipeline._depth_cache["test"] = np.zeros((10, 10))

        pipeline.clear_cache()

        assert len(pipeline._depth_cache) == 0


class TestProcessingResult:
    """Tests for ProcessingResult class."""

    def test_processing_result_quality_score(self, sample_image_pil):
        """Test quality_score property."""
        metrics = QualityMetrics(overall_score=0.85)
        result = ProcessingResult(
            image=sample_image_pil,
            quality_metrics=metrics,
        )

        assert result.quality_score == 0.85

    def test_processing_result_no_metrics(self, sample_image_pil):
        """Test quality_score with no metrics."""
        result = ProcessingResult(image=sample_image_pil)

        assert result.quality_score == 0.0


# =============================================================================
# Integration Tests
# =============================================================================

class TestPipelineIntegration:
    """Integration tests for complete pipeline workflows."""

    def test_full_pipeline_workflow(self, temp_image_file, temp_output_dir):
        """Test complete pipeline workflow."""
        # Use preview preset for speed
        pipeline = Rendering4KPipeline.from_preset("preview")

        # Process image
        result = pipeline.process(temp_image_file, temp_output_dir)

        # Verify result
        assert result.image is not None
        assert result.total_duration_ms > 0

        # Check outputs were created
        jpeg_files = list(temp_output_dir.glob("*.jpg"))
        assert len(jpeg_files) >= 1

    def test_pipeline_with_custom_config(self, temp_image_file, temp_output_dir):
        """Test pipeline with custom configuration."""
        config = PipelineConfig(
            name="custom_test",
            depth=DepthConfig(enabled=True, num_zones=2),
            tone_mapping=ToneMappingConfig(method=ToneMappingMethod.REINHARD),
            material_response=MaterialResponseConfig(strength=0.5),
            upscaling=UpscalingConfig(enabled=False),
            output=OutputConfig(master_tiff_16bit=False),
        )

        pipeline = Rendering4KPipeline(config)
        result = pipeline.process(temp_image_file, temp_output_dir)

        assert result.config_used.name == "custom_test"
        assert result.config_used.tone_mapping.method == ToneMappingMethod.REINHARD

    def test_pipeline_preserves_aspect_ratio(self, temp_image_file, temp_output_dir):
        """Test that upscaling preserves aspect ratio."""
        pipeline = Rendering4KPipeline.from_preset("default")
        pipeline.config.upscaling.enabled = True
        pipeline.config.upscaling.target_resolution = (3840, 2160)

        result = pipeline.process(temp_image_file, temp_output_dir)

        # Get original aspect ratio
        from PIL import Image as PILImage
        original = PILImage.open(temp_image_file)
        original_ratio = original.width / original.height

        # Check result maintains aspect ratio (approximately)
        result_ratio = result.image.width / result.image.height
        assert abs(original_ratio - result_ratio) < 0.01


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_small_image(self, temp_output_dir):
        """Test processing very small image."""
        # Create tiny image
        tiny_image = Image.new('RGB', (32, 32), color='red')
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            tiny_image.save(f.name)
            tiny_path = Path(f.name)

        pipeline = Rendering4KPipeline.from_preset("preview")
        result = pipeline.process(tiny_path, temp_output_dir)

        assert result.image is not None

    def test_grayscale_handling(self, temp_output_dir):
        """Test handling of grayscale input (converted to RGB)."""
        # Create grayscale image
        gray_image = Image.new('L', (100, 100), color=128)
        rgb_image = gray_image.convert('RGB')  # Convert to RGB first

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            rgb_image.save(f.name)
            gray_path = Path(f.name)

        pipeline = Rendering4KPipeline.from_preset("preview")
        result = pipeline.process(gray_path, temp_output_dir)

        assert result.image is not None
        assert result.image.mode == 'RGB'

    def test_invalid_input_path(self, temp_output_dir):
        """Test handling of invalid input path."""
        pipeline = Rendering4KPipeline.from_preset("preview")

        with pytest.raises(FileNotFoundError):
            pipeline.process(Path("/nonexistent/path.jpg"), temp_output_dir)


# =============================================================================
# Batch Processing Tests
# =============================================================================

class TestBatchProcessing:
    """Tests for batch processing functionality."""

    def test_batch_process_multiple_images(self, sample_image_pil, temp_output_dir):
        """Test batch processing of multiple images."""
        # Create multiple test images
        input_paths = []
        for i in range(3):
            path = temp_output_dir / f"input_{i}.png"
            sample_image_pil.save(path)
            input_paths.append(path)

        # Create output directory
        output_dir = temp_output_dir / "output"
        output_dir.mkdir()

        pipeline = Rendering4KPipeline.from_preset("preview")
        results = pipeline.batch_process(input_paths, output_dir, show_progress=False)

        assert len(results) == 3
        for result in results:
            assert isinstance(result, ProcessingResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
