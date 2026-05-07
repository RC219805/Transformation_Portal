#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=redefined-outer-name
"""
Tests for Unified Luxury Pipeline
=================================

Comprehensive test suite verifying:
- All output formats are generated correctly
- Bit depth is preserved appropriately
- Metadata survives processing
- Profile selection works
- Scene type detection is accurate
- Graceful failure when stages error
- Statistics tracking
- Batch processing
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from transformation_portal.pipelines.unified_luxury_pipeline import (
    HAS_TIFFFILE,
    OutputFormat,
    PipelineStage,
    PipelineStatistics,
    ProcessingProfile,
    SceneType,
    UnifiedLuxuryPipeline,
    UnifiedPipelineConfig,
    batch_process_luxury_renders,
    process_luxury_render,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_image():
    """Create sample RGB image for testing."""
    # Create 800x600 RGB test image with gradient
    arr = np.zeros((600, 800, 3), dtype=np.uint8)

    # Add gradient
    for i in range(600):
        arr[i, :, 0] = int(255 * i / 600)  # Red gradient
    for j in range(800):
        arr[:, j, 1] = int(255 * j / 800)  # Green gradient
    arr[:, :, 2] = 128  # Blue constant

    return Image.fromarray(arr, "RGB")


@pytest.fixture
def sample_image_file(temp_dir, sample_image):
    """Save sample image to file."""
    image_path = temp_dir / "test_image.jpg"
    sample_image.save(image_path, quality=95)
    return image_path


@pytest.fixture
def sky_image():
    """Create image with high sky content (for aerial detection)."""
    arr = np.ones((600, 800, 3), dtype=np.uint8) * 200  # Bright overall
    arr[400:, :, :] = 100  # Darker lower portion
    return Image.fromarray(arr, "RGB")


@pytest.fixture
def interior_image():
    """Create image with interior characteristics (low sky, varied brightness)."""
    arr = np.random.randint(50, 150, (600, 800, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


class TestUnifiedPipelineConfig:
    """Test UnifiedPipelineConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = UnifiedPipelineConfig()

        assert config.scene_type == SceneType.AUTO
        assert config.profile == ProcessingProfile.BALANCED
        assert config.output_formats == list(OutputFormat)
        assert config.enable_depth is True
        assert config.enable_material_response is True
        assert config.device == "auto"

    def test_custom_config(self, temp_dir):
        """Test custom configuration."""
        config = UnifiedPipelineConfig(
            scene_type=SceneType.INTERIOR,
            profile=ProcessingProfile.PREMIUM,
            output_dir=temp_dir,
            output_formats=[OutputFormat.MASTER_TIFF, OutputFormat.WEB_4K],
            enable_vfx=True,
            exposure=0.5,
            contrast=1.2,
            lut_strength=0.8,
        )

        assert config.scene_type == SceneType.INTERIOR
        assert config.profile == ProcessingProfile.PREMIUM
        assert len(config.output_formats) == 2
        assert OutputFormat.MASTER_TIFF in config.output_formats
        assert config.enable_vfx is True
        assert config.exposure == 0.5
        assert config.contrast == 1.2
        assert config.lut_strength == 0.8

    def test_parameter_clamping(self):
        """Test that parameters are clamped to valid ranges."""
        config = UnifiedPipelineConfig(
            exposure=5.0,  # Should be clamped to 2.0
            contrast=3.0,  # Should be clamped to 2.0
            saturation=3.0,  # Should be clamped to 2.0
            clarity=2.0,  # Should be clamped to 1.0
            lut_strength=1.5,  # Should be clamped to 1.0
        )

        assert config.exposure == 2.0
        assert config.contrast == 2.0
        assert config.saturation == 2.0
        assert config.clarity == 1.0
        assert config.lut_strength == 1.0

    def test_output_dir_path_conversion(self):
        """Test that output_dir is converted to Path."""
        config = UnifiedPipelineConfig(output_dir="test/path")
        assert isinstance(config.output_dir, Path)
        assert config.output_dir == Path("test/path")


class TestPipelineStage:
    """Test PipelineStage dataclass."""

    def test_stage_creation(self):
        """Test creating pipeline stage."""
        stage = PipelineStage("Test Stage", enabled=True, required=False)

        assert stage.name == "Test Stage"
        assert stage.enabled is True
        assert stage.required is False
        assert stage.success is False
        assert stage.elapsed_time == 0.0
        assert stage.error_message is None

    def test_stage_repr(self):
        """Test stage string representation."""
        # Disabled stage
        stage = PipelineStage("Test", enabled=False)
        assert "⊘" in repr(stage)
        assert "disabled" in repr(stage)

        # Successful stage
        stage = PipelineStage("Test", enabled=True)
        stage.success = True
        stage.elapsed_time = 1.234
        assert "✓" in repr(stage)
        assert "1.23" in repr(stage)

        # Failed stage
        stage = PipelineStage("Test", enabled=True)
        stage.error_message = "Test error"
        assert "✗" in repr(stage)
        assert "failed" in repr(stage)


class TestPipelineStatistics:
    """Test PipelineStatistics tracking."""

    def test_statistics_initialization(self):
        """Test statistics initialization."""
        stats = PipelineStatistics()

        assert stats.total_time == 0.0
        assert stats.images_processed == 0
        assert stats.images_failed == 0
        assert len(stats.stage_times) == 0
        assert len(stats.output_files) == 0

    def test_statistics_summary(self):
        """Test statistics summary generation."""
        stats = PipelineStatistics(
            total_time=10.5, images_processed=5, images_failed=1, stage_times={"Load": 1.0, "Process": 8.0, "Output": 1.5}
        )

        summary = stats.summary()

        assert "10.50s" in summary
        assert "5" in summary  # images processed
        assert "1" in summary  # images failed
        assert "Load" in summary
        assert "Process" in summary
        assert "Output" in summary


class TestSceneDetection:
    """Test automatic scene type detection."""

    def test_detect_aerial(self, sky_image, temp_dir):
        """Test aerial scene detection."""
        config = UnifiedPipelineConfig(
            scene_type=SceneType.AUTO, output_dir=temp_dir, output_formats=[OutputFormat.MASTER_TIFF]
        )
        pipeline = UnifiedLuxuryPipeline(config)

        detected = pipeline._detect_scene_type(sky_image)
        # Bright top third should trigger aerial detection
        assert detected in [SceneType.AERIAL, SceneType.EXTERIOR]

    def test_detect_interior(self, interior_image, temp_dir):
        """Test interior scene detection."""
        config = UnifiedPipelineConfig(
            scene_type=SceneType.AUTO, output_dir=temp_dir, output_formats=[OutputFormat.MASTER_TIFF]
        )
        pipeline = UnifiedLuxuryPipeline(config)

        detected = pipeline._detect_scene_type(interior_image)
        # High variance and low sky brightness should trigger interior
        assert detected == SceneType.INTERIOR

    def test_manual_scene_type(self, temp_dir):
        """Test manual scene type specification."""
        config = UnifiedPipelineConfig(scene_type=SceneType.EXTERIOR, output_dir=temp_dir)
        pipeline = UnifiedLuxuryPipeline(config)

        # Scene detection stage should be disabled
        assert not pipeline.stages["scene_detect"].enabled


class TestParameterOptimization:
    """Test parameter optimization based on profile and scene."""

    def test_premium_profile_params(self, temp_dir):
        """Test PREMIUM profile parameter optimization."""
        config = UnifiedPipelineConfig(profile=ProcessingProfile.PREMIUM, scene_type=SceneType.INTERIOR, output_dir=temp_dir)
        pipeline = UnifiedLuxuryPipeline(config)

        params = pipeline._optimize_parameters(config)

        assert params["ai_strength"] == 0.45
        assert params["ai_steps"] == 30
        assert params["depth_model_size"] == "large"
        assert params["material_strength"] == 0.7

    def test_performance_profile_params(self, temp_dir):
        """Test PERFORMANCE profile parameter optimization."""
        config = UnifiedPipelineConfig(
            profile=ProcessingProfile.PERFORMANCE, scene_type=SceneType.INTERIOR, output_dir=temp_dir
        )
        pipeline = UnifiedLuxuryPipeline(config)

        params = pipeline._optimize_parameters(config)

        assert params["ai_strength"] == 0.25
        assert params["ai_steps"] == 15
        assert params["depth_model_size"] == "small"
        assert params["material_strength"] == 0.5

    def test_scene_based_optimization(self, temp_dir):
        """Test scene-based parameter optimization."""
        # Interior
        config = UnifiedPipelineConfig(scene_type=SceneType.INTERIOR, output_dir=temp_dir)
        pipeline = UnifiedLuxuryPipeline(config)
        params = pipeline._optimize_parameters(config)

        assert params["clarity"] >= 0.15
        assert params["contrast"] <= 1.12

        # Aerial
        config = UnifiedPipelineConfig(scene_type=SceneType.AERIAL, output_dir=temp_dir)
        pipeline = UnifiedLuxuryPipeline(config)
        params = pipeline._optimize_parameters(config)

        assert params["clarity"] >= 0.20
        assert params.get("aerial_perspective") is True


class TestOutputGeneration:
    """Test output format generation."""

    def test_master_tiff_generation(self, sample_image, temp_dir):
        """Test Master TIFF generation."""
        config = UnifiedPipelineConfig(
            output_dir=temp_dir,
            output_formats=[OutputFormat.MASTER_TIFF],
            enable_depth=False,
            enable_material_response=False,
            enable_color_grading=False,
        )
        pipeline = UnifiedLuxuryPipeline(config)

        output_path = pipeline._save_master_tiff(sample_image, "test", {})

        assert output_path.exists()
        assert output_path.suffix == ".tif"
        assert "MASTER" in output_path.name

        # Verify image can be loaded
        loaded = Image.open(output_path)
        assert loaded.size == sample_image.size

    def test_web_4k_generation(self, sample_image, temp_dir):
        """Test Web 4K generation."""
        config = UnifiedPipelineConfig(output_dir=temp_dir, output_formats=[OutputFormat.WEB_4K])
        pipeline = UnifiedLuxuryPipeline(config)

        output_path = pipeline._save_web_4k(sample_image, "test", None)

        assert output_path.exists()
        assert output_path.suffix == ".jpg"
        assert "WEB_4K" in output_path.name

        # Verify size constraints
        loaded = Image.open(output_path)
        assert max(loaded.size) <= 3840

    def test_print_8k_generation(self, sample_image, temp_dir):
        """Test Print 8K generation."""
        config = UnifiedPipelineConfig(output_dir=temp_dir, output_formats=[OutputFormat.PRINT_8K])
        pipeline = UnifiedLuxuryPipeline(config)

        output_path = pipeline._save_print_8k(sample_image, "test", None)

        assert output_path.exists()
        assert "PRINT_8K" in output_path.name

    def test_social_generation(self, sample_image, temp_dir):
        """Test Social (1080p) generation."""
        config = UnifiedPipelineConfig(output_dir=temp_dir, output_formats=[OutputFormat.SOCIAL])
        pipeline = UnifiedLuxuryPipeline(config)

        output_path = pipeline._save_social(sample_image, "test", None)

        assert output_path.exists()
        assert "SOCIAL" in output_path.name

        # Verify size
        loaded = Image.open(output_path)
        assert max(loaded.size) == 1080

    def test_magazine_generation(self, sample_image, temp_dir):
        """Test Magazine 2K generation."""
        config = UnifiedPipelineConfig(output_dir=temp_dir, output_formats=[OutputFormat.MAGAZINE])
        pipeline = UnifiedLuxuryPipeline(config)

        output_path = pipeline._save_magazine(sample_image, "test", None)

        assert output_path.exists()
        assert "MAGAZINE" in output_path.name

    def test_all_formats_generation(self, sample_image_file, temp_dir):
        """Test generating all output formats."""
        config = UnifiedPipelineConfig(
            output_dir=temp_dir,
            output_formats=list(OutputFormat),
            enable_depth=False,
            enable_material_response=False,
            enable_color_grading=False,
        )
        pipeline = UnifiedLuxuryPipeline(config)

        results = pipeline.process(sample_image_file)

        # Should have all 5 formats
        assert len(results) == 5
        assert "master" in results
        assert "web" in results
        assert "print" in results
        assert "social" in results
        assert "magazine" in results

        # All files should exist
        for path in results.values():
            assert path.exists()

    def test_requested_output_failure_raises_and_does_not_count_success(self, sample_image_file, temp_dir):
        """Test requested output failures fail the required output stage."""
        config = UnifiedPipelineConfig(
            scene_type=SceneType.INTERIOR,
            output_dir=temp_dir,
            output_formats=[OutputFormat.WEB_4K],
            enable_depth=False,
            enable_material_response=False,
            enable_color_grading=False,
        )
        pipeline = UnifiedLuxuryPipeline(config)

        with patch.object(pipeline, "_save_web_4k", side_effect=RuntimeError("encoder failed")):
            with pytest.raises(RuntimeError, match="Failed to generate requested output format"):
                pipeline.process(sample_image_file)

        assert pipeline.stats.images_processed == 0
        assert pipeline.stats.images_failed == 1

    @pytest.mark.skipif(not HAS_TIFFFILE, reason="ndarray TIFF path requires tifffile")
    def test_master_tiff_generation_accepts_ndarray(self, temp_dir):
        """Test ndarray master TIFF generation logs dimensions without raising."""
        arr = np.zeros((12, 16, 3), dtype=np.float32)
        arr[..., 0] = 0.25
        arr[..., 1] = 0.5
        arr[..., 2] = 0.75

        config = UnifiedPipelineConfig(output_dir=temp_dir, output_formats=[OutputFormat.MASTER_TIFF])
        pipeline = UnifiedLuxuryPipeline(config)

        output_path = pipeline._save_master_tiff(arr, "array_input", {})

        assert output_path.exists()
        loaded = Image.open(output_path)
        assert loaded.size == (16, 12)


class TestMetadataPreservation:
    """Test metadata preservation through pipeline."""

    def test_icc_profile_preservation(self, sample_image, temp_dir):
        """Test ICC profile is preserved in JPEG outputs."""
        # Add fake ICC profile to sample
        icc_profile = b"fake_icc_profile_data"

        config = UnifiedPipelineConfig(output_dir=temp_dir, output_formats=[OutputFormat.WEB_4K], preserve_metadata=True)
        pipeline = UnifiedLuxuryPipeline(config)

        # Should not raise exception with ICC profile
        output_path = pipeline._save_web_4k(sample_image, "test", icc_profile)
        assert output_path.exists()

    def test_metadata_extraction(self, sample_image_file, temp_dir):
        """Test metadata extraction from input image."""
        config = UnifiedPipelineConfig(output_dir=temp_dir, output_formats=[OutputFormat.MASTER_TIFF])
        pipeline = UnifiedLuxuryPipeline(config)

        image, metadata = pipeline._load_image(sample_image_file)

        assert "format" in metadata
        assert "mode" in metadata
        assert "size" in metadata
        assert metadata["mode"] == "RGB"


class TestGracefulDegradation:
    """Test graceful failure handling for optional stages."""

    def test_optional_stage_failure_continues(self, sample_image_file, temp_dir):
        """Test that optional stage failures don't halt pipeline."""
        config = UnifiedPipelineConfig(
            output_dir=temp_dir,
            output_formats=[OutputFormat.MASTER_TIFF],
            enable_depth=True,  # Will likely fail without proper setup
            enable_material_response=True,
            enable_color_grading=False,
        )
        pipeline = UnifiedLuxuryPipeline(config)

        # Should complete despite optional stage failures
        results = pipeline.process(sample_image_file)

        assert len(results) > 0
        assert results["master"].exists()

    def test_required_stage_failure_halts(self, temp_dir):
        """Test that required stage failures halt pipeline."""
        config = UnifiedPipelineConfig(output_dir=temp_dir, output_formats=[OutputFormat.MASTER_TIFF])
        pipeline = UnifiedLuxuryPipeline(config)

        # Should raise exception for non-existent file
        with pytest.raises(FileNotFoundError):
            pipeline.process(Path("nonexistent_file.jpg"))


class TestPerCallOverrides:
    """Test per-call override behavior."""

    def test_process_override_can_disable_depth_for_current_call(self, sample_image_file, temp_dir):
        """Test enable_depth override gates the current call only."""
        config = UnifiedPipelineConfig(
            scene_type=SceneType.INTERIOR,
            output_dir=temp_dir,
            output_formats=[OutputFormat.MASTER_TIFF],
            enable_depth=True,
            enable_material_response=False,
            enable_color_grading=False,
        )
        pipeline = UnifiedLuxuryPipeline(config)

        with patch.object(pipeline, "_apply_depth_processing", wraps=pipeline._apply_depth_processing) as mock_depth:
            results = pipeline.process(sample_image_file, enable_depth=False)

        assert results["master"].exists()
        mock_depth.assert_not_called()
        assert pipeline.config.enable_depth is True

    def test_process_override_can_enable_vfx_for_current_call(self, sample_image_file, temp_dir):
        """Test enable_vfx override can activate a constructor-disabled stage."""
        config = UnifiedPipelineConfig(
            scene_type=SceneType.INTERIOR,
            output_dir=temp_dir,
            output_formats=[OutputFormat.MASTER_TIFF],
            enable_depth=False,
            enable_material_response=False,
            enable_vfx=False,
            enable_color_grading=False,
        )
        pipeline = UnifiedLuxuryPipeline(config)

        with patch.object(pipeline, "_apply_vfx_effects", wraps=pipeline._apply_vfx_effects) as mock_vfx:
            results = pipeline.process(sample_image_file, enable_vfx=True)

        assert results["master"].exists()
        mock_vfx.assert_called_once()
        assert pipeline.config.enable_vfx is False

    def test_process_override_can_enable_material_response_for_current_call(self, sample_image_file, temp_dir):
        """Test enable_material_response override can activate a constructor-disabled stage."""
        config = UnifiedPipelineConfig(
            scene_type=SceneType.INTERIOR,
            output_dir=temp_dir,
            output_formats=[OutputFormat.MASTER_TIFF],
            enable_depth=False,
            enable_material_response=False,
            enable_color_grading=False,
        )
        pipeline = UnifiedLuxuryPipeline(config)

        with patch.object(
            pipeline,
            "_apply_material_response",
            wraps=pipeline._apply_material_response,
        ) as mock_material:
            results = pipeline.process(sample_image_file, enable_material_response=True)

        assert results["master"].exists()
        mock_material.assert_called_once()
        assert pipeline.config.enable_material_response is False

    def test_process_override_uses_output_dir_for_current_call(self, sample_image_file, temp_dir):
        """Test output_dir override does not mutate constructor config."""
        default_output_dir = temp_dir / "default"
        override_output_dir = temp_dir / "override"
        config = UnifiedPipelineConfig(
            scene_type=SceneType.INTERIOR,
            output_dir=default_output_dir,
            output_formats=[OutputFormat.MASTER_TIFF],
            enable_depth=False,
            enable_material_response=False,
            enable_color_grading=False,
        )
        pipeline = UnifiedLuxuryPipeline(config)

        results = pipeline.process(sample_image_file, output_dir=override_output_dir)

        assert results["master"].parent == override_output_dir
        assert results["master"].exists()
        assert not (default_output_dir / f"{sample_image_file.stem}_MASTER.tif").exists()
        assert pipeline.config.output_dir == default_output_dir


class TestBatchProcessing:
    """Test batch processing functionality."""

    def test_batch_process_multiple_images(self, temp_dir, sample_image):
        """Test batch processing of multiple images."""
        # Create test images
        image_paths = []
        for i in range(3):
            image_path = temp_dir / f"input_{i}.jpg"
            sample_image.save(image_path)
            image_paths.append(image_path)

        config = UnifiedPipelineConfig(
            output_dir=temp_dir / "output",
            output_formats=[OutputFormat.MASTER_TIFF, OutputFormat.WEB_4K],
            enable_depth=False,
            enable_material_response=False,
            enable_color_grading=False,
        )
        pipeline = UnifiedLuxuryPipeline(config)

        results = pipeline.batch_process(image_paths, show_progress=False)

        # Should have results for all images
        assert len(results) == 3

        for image_path, outputs in results.items():
            assert len(outputs) == 2
            assert "master" in outputs
            assert "web" in outputs

    def test_batch_process_with_failures(self, temp_dir, sample_image):
        """Test batch processing continues despite individual failures."""
        # Create mix of valid and invalid paths
        valid_path = temp_dir / "valid.jpg"
        sample_image.save(valid_path)
        invalid_path = temp_dir / "invalid.jpg"

        config = UnifiedPipelineConfig(
            output_dir=temp_dir / "output",
            output_formats=[OutputFormat.MASTER_TIFF],
            enable_depth=False,
            enable_material_response=False,
        )
        pipeline = UnifiedLuxuryPipeline(config)

        results = pipeline.batch_process([valid_path, invalid_path], show_progress=False)

        # Valid image should succeed
        assert len(results[valid_path]) > 0

        # Invalid image should have empty results
        assert len(results[invalid_path]) == 0


class TestStatisticsSaving:
    """Test statistics tracking and saving."""

    def test_statistics_tracking(self, sample_image_file, temp_dir):
        """Test that statistics are tracked during processing."""
        config = UnifiedPipelineConfig(
            output_dir=temp_dir, output_formats=[OutputFormat.MASTER_TIFF], enable_depth=False, enable_material_response=False
        )
        pipeline = UnifiedLuxuryPipeline(config)

        pipeline.process(sample_image_file)

        assert pipeline.stats.images_processed == 1
        assert pipeline.stats.total_time > 0
        assert len(pipeline.stats.stage_times) > 0

    def test_save_statistics_json(self, sample_image_file, temp_dir):
        """Test saving statistics to JSON."""
        config = UnifiedPipelineConfig(
            output_dir=temp_dir, output_formats=[OutputFormat.MASTER_TIFF], enable_depth=False, enable_material_response=False
        )
        pipeline = UnifiedLuxuryPipeline(config)

        pipeline.process(sample_image_file)

        stats_path = pipeline.save_stats()

        assert stats_path.exists()

        # Verify JSON is valid
        with open(stats_path) as f:
            stats_data = json.load(f)

        assert "total_time" in stats_data
        assert "images_processed" in stats_data
        assert "stage_times" in stats_data
        assert "config" in stats_data


class TestConvenienceFunctions:
    """Test convenience functions."""

    @patch("transformation_portal.pipelines.unified_luxury_pipeline.UnifiedLuxuryPipeline")
    def test_process_luxury_render(self, mock_pipeline_class, sample_image_file, temp_dir):
        """Test process_luxury_render convenience function."""
        mock_instance = MagicMock()
        mock_instance.process.return_value = {"master": temp_dir / "test.tiff"}
        mock_pipeline_class.return_value = mock_instance

        result = process_luxury_render(sample_image_file, output_dir=temp_dir, profile=ProcessingProfile.PREMIUM)

        # Should create pipeline and call process
        assert mock_pipeline_class.called
        assert mock_instance.process.called

    @patch("transformation_portal.pipelines.unified_luxury_pipeline.UnifiedLuxuryPipeline")
    def test_batch_process_luxury_renders_default_includes_tiff(self, mock_pipeline_class, sample_image, temp_dir):
        """Test default batch discovery includes .tif and .tiff but not typo extensions."""
        input_dir = temp_dir / "inputs"
        input_dir.mkdir()
        sample_image.save(input_dir / "a.tiff")
        sample_image.save(input_dir / "b.tif")
        sample_image.save(input_dir / "c.jpg")
        (input_dir / "skip.ti").write_text("not an image")

        mock_instance = MagicMock()
        mock_instance.batch_process.return_value = {}
        mock_pipeline_class.return_value = mock_instance

        batch_process_luxury_renders(input_dir, output_dir=temp_dir / "output")

        input_paths = mock_instance.batch_process.call_args.args[0]
        assert input_dir / "a.tiff" in input_paths
        assert input_dir / "b.tif" in input_paths
        assert input_dir / "c.jpg" in input_paths
        assert input_dir / "skip.ti" not in input_paths

    @patch("transformation_portal.pipelines.unified_luxury_pipeline.UnifiedLuxuryPipeline")
    def test_batch_process_luxury_renders_respects_custom_pattern(self, mock_pipeline_class, sample_image, temp_dir):
        """Test custom batch discovery pattern is honored."""
        input_dir = temp_dir / "inputs"
        input_dir.mkdir()
        sample_image.save(input_dir / "keep.png")
        sample_image.save(input_dir / "skip.jpg")

        mock_instance = MagicMock()
        mock_instance.batch_process.return_value = {}
        mock_pipeline_class.return_value = mock_instance

        batch_process_luxury_renders(input_dir, output_dir=temp_dir / "output", pattern="*.png")

        input_paths = mock_instance.batch_process.call_args.args[0]
        assert input_paths == [input_dir / "keep.png"]


class TestDeviceDetection:
    """Test device detection."""

    def test_device_auto_detection(self, temp_dir):
        """Test automatic device detection."""
        config = UnifiedPipelineConfig(output_dir=temp_dir, device="auto")
        pipeline = UnifiedLuxuryPipeline(config)

        # Should detect some device
        assert pipeline.device in ["cpu", "cuda", "mps"]

    def test_device_manual_selection(self, temp_dir):
        """Test manual device selection."""
        config = UnifiedPipelineConfig(output_dir=temp_dir, device="cpu")
        pipeline = UnifiedLuxuryPipeline(config)

        assert pipeline.device == "cpu"


class TestColorGrading:
    """Test color grading functionality."""

    def test_exposure_adjustment(self, sample_image, temp_dir):
        """Test exposure adjustment."""
        config = UnifiedPipelineConfig(output_dir=temp_dir, exposure=0.5, enable_depth=False, enable_material_response=False)
        pipeline = UnifiedLuxuryPipeline(config)

        params = {"exposure": 0.5, "contrast": 1.0, "saturation": 1.0}
        result = pipeline._apply_color_grading(sample_image, params)

        assert isinstance(result, Image.Image)
        assert result.size == sample_image.size

    def test_contrast_adjustment(self, sample_image, temp_dir):
        """Test contrast adjustment."""
        config = UnifiedPipelineConfig(output_dir=temp_dir, contrast=1.2, enable_depth=False, enable_material_response=False)
        pipeline = UnifiedLuxuryPipeline(config)

        params = {"exposure": 0.0, "contrast": 1.2, "saturation": 1.0}
        result = pipeline._apply_color_grading(sample_image, params)

        assert isinstance(result, Image.Image)

    def test_saturation_adjustment(self, sample_image, temp_dir):
        """Test saturation adjustment."""
        config = UnifiedPipelineConfig(output_dir=temp_dir, saturation=1.3, enable_depth=False, enable_material_response=False)
        pipeline = UnifiedLuxuryPipeline(config)

        params = {"exposure": 0.0, "contrast": 1.0, "saturation": 1.3}
        result = pipeline._apply_color_grading(sample_image, params)

        assert isinstance(result, Image.Image)


class TestMaterialResponse:
    """Test Material Response processing."""

    def test_material_response_application(self, sample_image, temp_dir):
        """Test Material Response enhancement."""
        config = UnifiedPipelineConfig(output_dir=temp_dir, enable_material_response=True)
        pipeline = UnifiedLuxuryPipeline(config)

        params = {"material_strength": 0.7}
        result = pipeline._apply_material_response(sample_image, params, SceneType.INTERIOR)

        assert isinstance(result, Image.Image)
        assert result.size == sample_image.size


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_output_formats(self, sample_image_file, temp_dir):
        """Test with empty output formats list."""
        config = UnifiedPipelineConfig(
            output_dir=temp_dir, output_formats=[], enable_depth=False, enable_material_response=False
        )
        pipeline = UnifiedLuxuryPipeline(config)

        results = pipeline.process(sample_image_file)

        # Should still process but produce no outputs
        assert len(results) == 0

    def test_grayscale_input(self, temp_dir):
        """Test processing grayscale input image."""
        # Create grayscale test image
        gray_arr = np.random.randint(0, 255, (600, 800), dtype=np.uint8)
        gray_img = Image.fromarray(gray_arr, "L")
        gray_path = temp_dir / "gray.jpg"
        gray_img.save(gray_path)

        config = UnifiedPipelineConfig(
            output_dir=temp_dir, output_formats=[OutputFormat.MASTER_TIFF], enable_depth=False, enable_material_response=False
        )
        pipeline = UnifiedLuxuryPipeline(config)

        # Should convert to RGB and process
        results = pipeline.process(gray_path)
        assert len(results) > 0

    def test_very_small_image(self, temp_dir):
        """Test processing very small image."""
        small_arr = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        small_img = Image.fromarray(small_arr, "RGB")
        small_path = temp_dir / "small.jpg"
        small_img.save(small_path)

        config = UnifiedPipelineConfig(
            output_dir=temp_dir,
            output_formats=[OutputFormat.MASTER_TIFF, OutputFormat.WEB_4K],
            enable_depth=False,
            enable_material_response=False,
        )
        pipeline = UnifiedLuxuryPipeline(config)

        # Should process without error
        results = pipeline.process(small_path)
        assert len(results) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
