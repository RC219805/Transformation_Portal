#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the Fully-Integrated Professional Pipeline.

Tests cover:
- Configuration and preset loading
- Individual pipeline stages
- Batch processing
- CLI interface
- Error handling and graceful degradation
"""

# pylint: disable=redefined-outer-name  # pytest fixtures use other fixtures as params

from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image

from transformation_portal.pipelines.pro_pipeline import (
    PipelinePreset,
    PipelineStage,
    ProPipeline,
    ProPipelineConfig,
)

pytestmark = [pytest.mark.unit]


@pytest.fixture
def sample_image():
    """Create a sample RGB image for testing."""
    img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


@pytest.fixture
def temp_image_file(sample_image, tmp_path):
    """Create a temporary image file."""
    image_path = tmp_path / "test_image.jpg"
    sample_image.save(image_path, quality=95)
    return image_path


@pytest.fixture
def pipeline_config(tmp_path):
    """Create a basic pipeline configuration."""
    return ProPipelineConfig(
        input_path=tmp_path / "input.jpg",
        output_dir=tmp_path / "output",
        preset=PipelinePreset.CUSTOM,
        device="cpu",
        quality="standard",
    )


class TestPipelineStage:
    """Tests for PipelineStage dataclass."""

    def test_stage_creation(self):
        """Test creating a pipeline stage."""
        stage = PipelineStage("Test Stage", enabled=True, config={"param": 1.0})
        assert stage.name == "Test Stage"
        assert stage.enabled is True
        assert stage.config["param"] == 1.0

    def test_stage_repr(self):
        """Test string representation."""
        stage_enabled = PipelineStage("Enabled", enabled=True)
        stage_disabled = PipelineStage("Disabled", enabled=False)

        assert "✓" in repr(stage_enabled)
        assert "✗" in repr(stage_disabled)


class TestProPipelineConfig:
    """Tests for ProPipelineConfig."""

    def test_config_creation(self, tmp_path):
        """Test creating a pipeline configuration."""
        config = ProPipelineConfig(
            input_path=tmp_path / "input.jpg",
            output_dir=tmp_path / "output",
        )

        assert config.input_path.name == "input.jpg"
        assert config.output_dir.name == "output"
        assert config.preset == PipelinePreset.CUSTOM
        assert config.device == "auto"

    def test_preset_application(self, tmp_path):
        """Test that presets correctly configure stages."""
        config = ProPipelineConfig(
            input_path=tmp_path / "input.jpg",
            output_dir=tmp_path / "output",
            preset=PipelinePreset.ARCHITECTURAL_HERO,
        )

        # Check that stages are configured according to preset
        assert config.depth_stage.enabled is True
        assert config.ai_stage.enabled is True
        assert config.material_stage.enabled is True
        assert config.grading_stage.enabled is True
        assert config.finishing_stage.enabled is True

    def test_interior_dramatic_preset(self, tmp_path):
        """Test interior dramatic preset configuration."""
        config = ProPipelineConfig(
            input_path=tmp_path / "input.jpg",
            output_dir=tmp_path / "output",
            preset=PipelinePreset.INTERIOR_DRAMATIC,
        )

        # Interior dramatic disables AI enhancement
        assert config.ai_stage.enabled is False
        assert config.depth_stage.enabled is True
        assert config.material_stage.enabled is True

    def test_custom_preset_no_changes(self, tmp_path):
        """Test that custom preset doesn't apply automatic configuration."""
        config = ProPipelineConfig(
            input_path=tmp_path / "input.jpg",
            output_dir=tmp_path / "output",
            preset=PipelinePreset.CUSTOM,
        )

        # Custom preset should have defaults
        assert config.depth_stage.enabled is True
        assert config.depth_stage.config == {}


class TestProPipelineConfigFromYaml:
    """Tests for the YAML loader added by the `--config` CLI flag."""

    def _write(self, tmp_path, body: str):
        p = tmp_path / "config.yaml"
        p.write_text(body)
        return p

    def test_from_yaml_global_section_maps_to_dataclass_fields(self, tmp_path):
        yaml_path = self._write(
            tmp_path,
            """
global:
  device: cpu
  quality: standard
  output_format: png
  bit_depth: 8
  preserve_metadata: false
  use_cache: false
  num_workers: 2
""",
        )
        config = ProPipelineConfig.from_yaml(
            yaml_path,
            input_path=tmp_path / "input.jpg",
            output_dir=tmp_path / "output",
        )
        assert config.device == "cpu"
        assert config.quality == "standard"
        assert config.output_format == "png"
        assert config.bit_depth == 8
        assert config.preserve_metadata is False
        assert config.use_cache is False
        assert config.num_workers == 2

    def test_from_yaml_stages_merge_into_pipeline_stage_config(self, tmp_path):
        yaml_path = self._write(
            tmp_path,
            """
stages:
  depth:
    clarity:
      enabled: true
      amount: 0.25
    model: depth-anything-v2-large
  finishing:
    sharpen:
      enabled: true
      amount: 0.2
  ai:
    enabled: false
    strength: 0.5
""",
        )
        config = ProPipelineConfig.from_yaml(
            yaml_path,
            input_path=tmp_path / "input.jpg",
            output_dir=tmp_path / "output",
        )
        # depth stage: enabled key absent → preserves default; other keys merge
        assert config.depth_stage.enabled is True
        assert config.depth_stage.config["model"] == "depth-anything-v2-large"
        assert config.depth_stage.config["clarity"] == pytest.approx(0.25)
        assert config.finishing_stage.config["sharpen"] == pytest.approx(0.2)
        # ai stage: explicit `enabled: false` overrides default toggle
        assert config.ai_stage.enabled is False
        assert config.ai_stage.config["strength"] == 0.5

    def test_from_yaml_boolean_strings_are_parsed_explicitly(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TP_TEST_AI_ENABLED", "false")
        monkeypatch.setenv("TP_TEST_LINEAR_OUTPUT", "false")
        monkeypatch.setenv("TP_TEST_USE_CACHE", "0")
        yaml_path = self._write(
            tmp_path,
            """
global:
  linear_output: ${TP_TEST_LINEAR_OUTPUT}
  use_cache: ${TP_TEST_USE_CACHE}
stages:
  ai:
    enabled: ${TP_TEST_AI_ENABLED}
""",
        )
        config = ProPipelineConfig.from_yaml(
            yaml_path,
            input_path=tmp_path / "input.jpg",
            output_dir=tmp_path / "output",
        )
        assert config.linear_output is False
        assert config.use_cache is False
        assert config.ai_stage.enabled is False

    def test_from_yaml_invalid_boolean_string_raises(self, tmp_path):
        yaml_path = self._write(
            tmp_path,
            """
stages:
  ai:
    enabled: sometimes
""",
        )
        with pytest.raises(ValueError, match="stages.ai.enabled"):
            ProPipelineConfig.from_yaml(
                yaml_path,
                input_path=tmp_path / "input.jpg",
                output_dir=tmp_path / "output",
            )

    def test_from_yaml_invalid_device_raises(self, tmp_path):
        yaml_path = self._write(tmp_path, "global:\n  device: gpu-9000\n")
        with pytest.raises(ValueError, match="Invalid device"):
            ProPipelineConfig.from_yaml(
                yaml_path,
                input_path=tmp_path / "input.jpg",
                output_dir=tmp_path / "output",
            )

    def test_from_yaml_invalid_bit_depth_raises(self, tmp_path):
        yaml_path = self._write(tmp_path, "global:\n  bit_depth: 12\n")
        with pytest.raises(ValueError, match="Invalid bit_depth"):
            ProPipelineConfig.from_yaml(
                yaml_path,
                input_path=tmp_path / "input.jpg",
                output_dir=tmp_path / "output",
            )

    def test_from_yaml_invalid_preset_raises(self, tmp_path):
        yaml_path = self._write(tmp_path, "preset: not-a-real-preset\n")
        with pytest.raises(ValueError, match="Invalid preset"):
            ProPipelineConfig.from_yaml(
                yaml_path,
                input_path=tmp_path / "input.jpg",
                output_dir=tmp_path / "output",
            )

    def test_from_yaml_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ProPipelineConfig.from_yaml(
                tmp_path / "does-not-exist.yaml",
                input_path=tmp_path / "input.jpg",
                output_dir=tmp_path / "output",
            )

    def test_from_yaml_loads_repo_pro_pipeline_config(self, tmp_path):
        """The shipped config/pro_pipeline_config.yaml should round-trip."""
        from pathlib import Path

        repo_yaml = Path(__file__).resolve().parent.parent / "config" / "pro_pipeline_config.yaml"
        config = ProPipelineConfig.from_yaml(
            repo_yaml,
            input_path=tmp_path / "input.jpg",
            output_dir=tmp_path / "output",
        )
        assert config.device == "auto"
        # 'tiff' is the YAML's spelling; the runtime uses the canonical .tif suffix.
        assert config.output_format == "tif"
        assert config.depth_stage.config.get("model") == "depth-anything-v2-base"
        assert isinstance(config.depth_stage.config["clarity"], float)
        assert isinstance(config.finishing_stage.config["sharpen"], float)


class TestCliConfigOverrides:
    """Tests for the merge precedence between --config YAML and CLI flags."""

    def _yaml(self, tmp_path, body: str):
        p = tmp_path / "config.yaml"
        p.write_text(body)
        return p

    def _build(self, tmp_path, **overrides):
        """Invoke the helper with CLI defaults, optionally overriding flags."""
        from transformation_portal.pipelines.pro_pipeline import _build_config_with_yaml_overrides

        defaults = dict(
            config_path=overrides.pop("config_path", None),
            input_path=tmp_path / "input.jpg",
            output_dir=tmp_path / "output",
            preset=PipelinePreset.ARCHITECTURAL_HERO,
            device="auto",
            quality="high",
            output_format="tif",
            bit_depth=16,
            linear_output=True,
            keep_intermediates=False,
            dry_run=False,
            num_workers=4,
            depth_aware=True,
            ai_enhance=True,
            material_response=True,
            color_grading=True,
            finishing=True,
        )
        defaults.update(overrides)
        return _build_config_with_yaml_overrides(**defaults)

    def test_no_config_uses_cli_only(self, tmp_path):
        config = self._build(tmp_path, device="cpu", quality="ultra")
        assert config.device == "cpu"
        assert config.quality == "ultra"

    def test_yaml_supplies_values_when_cli_at_default(self, tmp_path):
        yaml_path = self._yaml(
            tmp_path,
            """
global:
  device: mps
  quality: draft
""",
        )
        config = self._build(tmp_path, config_path=yaml_path)
        # CLI flags left at default → YAML wins
        assert config.device == "mps"
        assert config.quality == "draft"

    def test_explicit_cli_flag_overrides_yaml(self, tmp_path):
        yaml_path = self._yaml(
            tmp_path,
            """
global:
  device: mps
  quality: draft
""",
        )
        # --device cuda differs from default 'auto' → CLI wins; --quality
        # left at default → YAML's 'draft' wins.
        config = self._build(tmp_path, config_path=yaml_path, device="cuda")
        assert config.device == "cuda"
        assert config.quality == "draft"

    def test_stage_toggles_always_reflect_cli(self, tmp_path):
        """`--no-ai` flips the stage even when YAML enables it."""
        yaml_path = self._yaml(
            tmp_path,
            """
stages:
  ai:
    enabled: true
    strength: 0.9
""",
        )
        config = self._build(tmp_path, config_path=yaml_path, ai_enhance=False)
        assert config.ai_stage.enabled is False
        # Stage config payload from YAML still merged in even when toggled off.
        assert config.ai_stage.config["strength"] == 0.9


class TestProPipeline:
    """Tests for ProPipeline orchestrator."""

    def test_pipeline_creation(self, pipeline_config):
        """Test creating a pipeline instance."""
        pipeline = ProPipeline(pipeline_config)

        assert pipeline.config == pipeline_config
        assert pipeline.device in ["cpu", "cuda", "mps"]
        assert pipeline.stats["images_processed"] == 0

    def test_device_detection(self, pipeline_config):
        """Test automatic device detection."""
        pipeline = ProPipeline(pipeline_config)

        # Device should be detected (cpu, cuda, or mps)
        assert pipeline.device in ["cpu", "cuda", "mps"]

    def test_manual_device_selection(self, tmp_path):
        """Test manual device selection."""
        config = ProPipelineConfig(
            input_path=tmp_path / "input.jpg",
            output_dir=tmp_path / "output",
            device="cpu",
        )
        pipeline = ProPipeline(config)

        assert pipeline.device == "cpu"

    def test_process_image_success(self, pipeline_config, temp_image_file):
        """Test successful image processing."""
        pipeline = ProPipeline(pipeline_config)

        result = pipeline.process_image(temp_image_file)

        assert result is not None
        assert result.exists()
        assert pipeline.stats["images_processed"] == 1
        assert pipeline.stats["images_failed"] == 0

    def test_process_image_creates_output_dir(self, tmp_path, temp_image_file):
        """Test that output directory is created if it doesn't exist."""
        output_dir = tmp_path / "new_output_dir"
        assert not output_dir.exists()

        config = ProPipelineConfig(
            input_path=temp_image_file,
            output_dir=output_dir,
        )
        pipeline = ProPipeline(config)

        result = pipeline.process_image(temp_image_file)

        assert output_dir.exists()
        assert result.parent == output_dir

    def test_process_nonexistent_image(self, pipeline_config, tmp_path):
        """Test handling of nonexistent image."""
        nonexistent = tmp_path / "nonexistent.jpg"
        pipeline = ProPipeline(pipeline_config)

        result = pipeline.process_image(nonexistent)

        assert result is None
        assert pipeline.stats["images_failed"] == 1

    def test_depth_stage_execution(self, pipeline_config, temp_image_file, sample_image):
        """Test depth-aware processing stage."""
        pipeline = ProPipeline(pipeline_config)

        # Enable only depth stage
        pipeline.config.depth_stage.enabled = True
        pipeline.config.ai_stage.enabled = False
        pipeline.config.material_stage.enabled = False
        pipeline.config.grading_stage.enabled = False
        pipeline.config.finishing_stage.enabled = False

        result = pipeline._apply_depth_stage(sample_image, temp_image_file)

        assert result is not None
        assert isinstance(result, Image.Image)
        assert result.size == sample_image.size

    def test_ai_stage_execution(self, pipeline_config, temp_image_file, sample_image):
        """Test AI enhancement stage."""
        pipeline = ProPipeline(pipeline_config)

        result = pipeline._apply_ai_stage(sample_image, temp_image_file)

        assert result is not None
        assert isinstance(result, Image.Image)
        assert result.size == sample_image.size

    def test_material_stage_execution(self, pipeline_config, temp_image_file, sample_image):
        """Test Material Response stage."""
        pipeline = ProPipeline(pipeline_config)

        result = pipeline._apply_material_stage(sample_image, temp_image_file)

        assert result is not None
        assert isinstance(result, Image.Image)
        assert result.size == sample_image.size

    def test_grading_stage_execution(self, pipeline_config, temp_image_file, sample_image):
        """Test color grading stage."""
        pipeline = ProPipeline(pipeline_config)

        result = pipeline._apply_grading_stage(sample_image, temp_image_file)

        assert result is not None
        assert isinstance(result, Image.Image)
        assert result.size == sample_image.size

    def test_finishing_stage_execution(self, pipeline_config, temp_image_file, sample_image):
        """Test finishing stage."""
        pipeline = ProPipeline(pipeline_config)

        result = pipeline._apply_finishing_stage(sample_image, temp_image_file)

        assert result is not None
        assert isinstance(result, Image.Image)
        assert result.size == sample_image.size

    def test_stage_graceful_failure(self, pipeline_config, temp_image_file):
        """Test that pipeline continues when a stage fails."""
        pipeline = ProPipeline(pipeline_config)

        # Create a mock image that will cause processing to fail
        mock_image = Mock(spec=Image.Image)
        mock_image.size = (512, 512)

        # Depth stage should catch exception and return original
        with patch.object(pipeline, "_apply_depth_stage", side_effect=Exception("Test error")):
            result = pipeline.process_image(temp_image_file)
            # Should still complete despite error in one stage
            assert result is not None or pipeline.stats["images_failed"] == 1

    def test_batch_processing(self, pipeline_config, tmp_path):
        """Test batch processing of multiple images."""
        # Create multiple test images
        images = []
        for i in range(3):
            img_path = tmp_path / f"test_{i}.jpg"
            img = Image.new("RGB", (256, 256), color=(i * 50, i * 50, i * 50))
            img.save(img_path)
            images.append(img_path)

        pipeline = ProPipeline(pipeline_config)
        stats = pipeline.batch_process(images)

        assert stats["processed"] == 3
        assert stats["failed"] == 0
        assert len(stats["results"]) == 3
        assert stats["total_time"] > 0

    def test_batch_processing_with_failures(self, pipeline_config, tmp_path):
        """Test batch processing handles failures gracefully."""
        # Mix of valid and invalid images
        img_path = tmp_path / "valid.jpg"
        img = Image.new("RGB", (256, 256), color=(100, 100, 100))
        img.save(img_path)

        nonexistent = tmp_path / "nonexistent.jpg"

        images = [img_path, nonexistent]

        pipeline = ProPipeline(pipeline_config)
        stats = pipeline.batch_process(images)

        assert stats["processed"] == 1
        assert stats["failed"] == 1

    def test_output_filename_generation(self, pipeline_config, temp_image_file):
        """Test output filename generation with presets."""
        # Test with architectural hero preset
        config_hero = ProPipelineConfig(
            input_path=temp_image_file,
            output_dir=pipeline_config.output_dir,
            preset=PipelinePreset.ARCHITECTURAL_HERO,
        )
        pipeline = ProPipeline(config_hero)

        output = pipeline._save_output(Image.new("RGB", (100, 100)), temp_image_file)

        assert "architectural-hero" in output.name
        assert output.suffix == ".tif"  # Default format

    def test_different_output_formats(self, pipeline_config, temp_image_file, sample_image):
        """Test saving in different output formats."""
        formats = {
            "jpg": ".jpg",
            "png": ".png",
            "tif": ".tif",
            "tiff": ".tif",
        }

        for fmt, expected_suffix in formats.items():
            pipeline_config.output_format = fmt
            pipeline = ProPipeline(pipeline_config)

            output = pipeline._save_output(sample_image, temp_image_file)

            assert output.suffix == expected_suffix
            assert output.exists()

    def test_statistics_tracking(self, pipeline_config, tmp_path):
        """Test that statistics are properly tracked."""
        # Create test images
        images = []
        for i in range(2):
            img_path = tmp_path / f"test_{i}.jpg"
            img = Image.new("RGB", (256, 256))
            img.save(img_path)
            images.append(img_path)

        pipeline = ProPipeline(pipeline_config)
        stats = pipeline.batch_process(images)

        # Check statistics structure
        assert "processed" in stats
        assert "failed" in stats
        assert "total_time" in stats
        assert "avg_time" in stats
        assert "stage_times" in stats

        # Check values
        assert stats["processed"] == 2
        assert stats["avg_time"] > 0


class TestPresets:
    """Tests for pipeline presets."""

    def test_all_presets_valid(self, tmp_path):
        """Test that all presets can be instantiated."""
        for preset in PipelinePreset:
            if preset == PipelinePreset.CUSTOM:
                continue

            config = ProPipelineConfig(
                input_path=tmp_path / "input.jpg",
                output_dir=tmp_path / "output",
                preset=preset,
            )

            pipeline = ProPipeline(config)

            # Should be created without errors
            assert pipeline.config.preset == preset

    def test_exterior_golden_hour_preset(self, tmp_path):
        """Test exterior golden hour preset specifics."""
        config = ProPipelineConfig(
            input_path=tmp_path / "input.jpg",
            output_dir=tmp_path / "output",
            preset=PipelinePreset.EXTERIOR_GOLDEN_HOUR,
        )

        # Should have specific configurations
        assert config.depth_stage.enabled is True
        assert config.ai_stage.enabled is True
        assert config.material_stage.enabled is True

        # Verify depth stage has config (atmospheric_haze may or may not be set)
        assert isinstance(config.depth_stage.config, dict)

    def test_aerial_estate_preset(self, tmp_path):
        """Test aerial estate preset specifics."""
        config = ProPipelineConfig(
            input_path=tmp_path / "input.jpg",
            output_dir=tmp_path / "output",
            preset=PipelinePreset.AERIAL_ESTATE,
        )

        # Aerial preset disables AI to preserve natural look
        assert config.ai_stage.enabled is False
        assert config.depth_stage.enabled is True


class TestErrorHandling:
    """Tests for error handling and graceful degradation."""

    def test_missing_dependencies_graceful(self, pipeline_config):
        """Test graceful handling when optional dependencies are missing."""
        pipeline = ProPipeline(pipeline_config)

        # Should create pipeline even if some modules can't be loaded
        assert pipeline is not None
        assert pipeline._depth_pipeline is None  # Lazy loaded

    def test_invalid_input_path(self, tmp_path):
        """Test handling of invalid input path."""
        config = ProPipelineConfig(
            input_path=tmp_path / "nonexistent.jpg",
            output_dir=tmp_path / "output",
        )
        pipeline = ProPipeline(config)

        result = pipeline.process_image(tmp_path / "nonexistent.jpg")

        assert result is None
        assert pipeline.stats["images_failed"] == 1

    def test_corrupted_image_handling(self, pipeline_config, tmp_path):
        """Test handling of corrupted image files."""
        # Create a corrupted image file
        corrupted = tmp_path / "corrupted.jpg"
        corrupted.write_text("This is not an image")

        pipeline = ProPipeline(pipeline_config)
        result = pipeline.process_image(corrupted)

        assert result is None
        assert pipeline.stats["images_failed"] == 1


class TestCLI:
    """Tests for CLI interface (typer commands)."""

    def test_cli_imports(self):
        """Test that CLI can be imported without errors."""
        from transformation_portal.pipelines.pro_pipeline import app, batch, list_presets, process, version

        assert app is not None
        assert process is not None
        assert batch is not None
        assert list_presets is not None
        assert version is not None

    def test_preset_enum_values(self):
        """Test that preset enum has expected values."""
        assert PipelinePreset.ARCHITECTURAL_HERO.value == "architectural-hero"
        assert PipelinePreset.INTERIOR_DRAMATIC.value == "interior-dramatic"
        assert PipelinePreset.EXTERIOR_GOLDEN_HOUR.value == "exterior-golden-hour"


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_end_to_end_processing(self, tmp_path):
        """Test complete end-to-end processing workflow."""
        # Create test image
        input_image = tmp_path / "input" / "test.jpg"
        input_image.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (512, 512), color=(100, 100, 100))
        img.save(input_image)

        output_dir = tmp_path / "output"

        # Create config with all stages enabled
        config = ProPipelineConfig(
            input_path=input_image,
            output_dir=output_dir,
            preset=PipelinePreset.ARCHITECTURAL_HERO,
            device="cpu",
            quality="standard",
        )

        pipeline = ProPipeline(config)
        result = pipeline.process_image(input_image)

        # Verify output
        assert result is not None
        assert result.exists()
        assert result.parent == output_dir

        # Verify output is valid image
        output_image = Image.open(result)
        assert output_image.size == img.size

    def test_metadata_preservation(self, tmp_path):
        """Test that image metadata is preserved when possible."""
        # Create image with metadata
        input_image = tmp_path / "input.jpg"
        img = Image.new("RGB", (512, 512))
        img.save(input_image, exif=b"test_exif_data")

        config = ProPipelineConfig(
            input_path=input_image,
            output_dir=tmp_path / "output",
            preserve_metadata=True,
        )

        pipeline = ProPipeline(config)
        result = pipeline.process_image(input_image)

        # Output should exist
        assert result is not None
        assert result.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
