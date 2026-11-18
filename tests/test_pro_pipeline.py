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

    def test_batch_same_stem_inputs_keep_separate_outputs(self, pipeline_config, tmp_path):
        """M-02: kitchen.png and kitchen.tif must not overwrite each other's output."""
        bright = tmp_path / "kitchen.png"
        dark = tmp_path / "kitchen.tif"
        Image.new("RGB", (64, 64), color=(250, 250, 250)).save(bright)
        Image.new("RGB", (64, 64), color=(10, 10, 10)).save(dark)

        pipeline = ProPipeline(pipeline_config)
        stats = pipeline.batch_process([bright, dark])

        first, second = stats["results"]
        assert stats["processed"] == 2
        assert first != second
        assert first.name.startswith("kitchen_") and not first.name.startswith("kitchen_tif")
        assert second.name.startswith("kitchen_tif_")
        assert first.exists() and second.exists()
        assert np.asarray(Image.open(first)).mean() > np.asarray(Image.open(second)).mean() + 100

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


def test_process_image_scales_16bit_greyscale_input(tmp_path):
    """Regression for H-05: a 16-bit grey PNG (mode I;16) was clipped at 255 and came out white."""
    source = tmp_path / "grey16.png"
    Image.fromarray(np.full((8, 8), 32896, dtype=np.uint16)).save(source)
    config = ProPipelineConfig(
        input_path=source, output_dir=tmp_path / "out", preset=PipelinePreset.CUSTOM, device="cpu", output_format="png"
    )
    for stage in (config.depth_stage, config.ai_stage, config.material_stage, config.grading_stage, config.finishing_stage):
        stage.enabled = False

    result = ProPipeline(config).process_image(source)

    assert result is not None
    with Image.open(result) as out:
        assert out.mode == "RGB"
        assert np.array_equal(np.asarray(out), np.full((8, 8, 3), 128, dtype=np.uint8))


def test_five_stage_run_does_not_darken_mid_grey_noise(tmp_path):
    """Regression for L-03: every stage truncated ``arr * 255`` to uint8, so a 5-stage run
    on symmetric mid-grey noise lost about two levels of mean brightness (127.57 -> 125.56)."""
    rng = np.random.default_rng(0)
    noise = np.clip(rng.normal(127.5, 30, (128, 128, 1)).repeat(3, 2), 0, 255).round().astype(np.uint8)
    source = tmp_path / "noise.png"
    Image.fromarray(noise).save(source)
    config = ProPipelineConfig(
        input_path=source, output_dir=tmp_path / "out", preset=PipelinePreset.CUSTOM, device="cpu", output_format="png"
    )

    result = ProPipeline(config).process_image(source)

    assert result is not None
    with Image.open(result) as out:
        shift = np.asarray(out).astype(np.float64).mean() - noise.mean()
    assert abs(shift) < 0.25


class TestUnsharpStagesStaySpatial:
    """The clarity / micro-contrast / sharpen stages are unsharp masks over the spatial axes only.

    Regression for H-04: ``gaussian_filter(img, sigma=s)`` on an HxWx3 array also blurred the
    colour-channel axis, so the "blurred" base was nearly grey and the stages pushed flat colour
    patches towards saturation (RGB(204,102,51) became RGB(215,99,42) in the depth stage).
    """

    STAGES = ("_apply_depth_stage", "_apply_material_stage", "_apply_finishing_stage")

    @pytest.mark.parametrize("stage_name", STAGES)
    @pytest.mark.parametrize("preset", [PipelinePreset.CUSTOM, PipelinePreset.ARCHITECTURAL_HERO])
    def test_flat_colour_patch_is_left_untouched(self, tmp_path, stage_name, preset):
        config = ProPipelineConfig(input_path=tmp_path / "in.png", output_dir=tmp_path / "out", preset=preset, device="cpu")
        pipeline = ProPipeline(config)
        flat = Image.fromarray(np.tile(np.array([[[204, 102, 51]]], dtype=np.uint8), (32, 32, 1)))

        out = np.asarray(getattr(pipeline, stage_name)(flat, tmp_path / "in.png")).astype(np.int16)

        # One level of slack for the float -> uint8 conversion; channel mixing moved it by 2-11 levels.
        assert np.abs(out - np.array([204, 102, 51], dtype=np.int16)).max() <= 1

    @pytest.mark.parametrize("stage_name", STAGES)
    def test_stage_still_sharpens_a_spatial_edge(self, pipeline_config, tmp_path, stage_name):
        pipeline = ProPipeline(pipeline_config)
        arr = np.full((32, 32, 3), 64, dtype=np.uint8)
        arr[:, 16:] = 192
        out = np.asarray(getattr(pipeline, stage_name)(Image.fromarray(arr), tmp_path / "in.png")).astype(np.int16)

        # Unsharp masking overshoots on both sides of the edge and leaves channels equal on grey input.
        assert out[:, 16:18].max() > 192
        assert out[:, 14:16].min() < 64
        assert np.array_equal(out[..., 0], out[..., 1]) and np.array_equal(out[..., 1], out[..., 2])


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


class TestSourceMetadataPreservation:
    """M-07: ``preserve_metadata=True`` must carry the source ICC profile, DPI and EXIF."""

    @staticmethod
    def _tagged_jpeg(path):
        from PIL import ImageCms

        icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        img = Image.new("RGB", (32, 32), color=(120, 90, 60))
        exif = img.getexif()
        exif[0x0110] = "Camera"
        img.save(path, icc_profile=icc, dpi=(300, 300), exif=exif, quality=95)
        return icc

    @pytest.mark.parametrize("fmt", ["jpg", "png"])
    def test_gamma_encoded_output_keeps_icc_dpi_and_exif(self, tmp_path, fmt):
        source = tmp_path / "tagged.jpg"
        icc = self._tagged_jpeg(source)
        config = ProPipelineConfig(input_path=source, output_dir=tmp_path / "out", output_format=fmt, device="cpu")

        result = ProPipeline(config).process_image(source)

        assert result is not None
        with Image.open(result) as output:
            assert output.info.get("icc_profile") == icc
            assert tuple(round(v) for v in output.info["dpi"]) == (300, 300)
            assert output.getexif().get(0x0110) == "Camera"

    def test_linear_tiff_keeps_dpi_but_not_the_gamma_icc_profile(self, tmp_path):
        pytest.importorskip("tifffile")
        source = tmp_path / "tagged.jpg"
        self._tagged_jpeg(source)
        config = ProPipelineConfig(
            input_path=source, output_dir=tmp_path / "out", output_format="tif", linear_output=True, device="cpu"
        )

        result = ProPipeline(config).process_image(source)

        assert result is not None
        with Image.open(result) as output:
            assert tuple(round(v) for v in output.info["dpi"]) == (300, 300)
            assert not output.info.get("icc_profile")

    def test_preserve_metadata_false_drops_source_metadata(self, tmp_path):
        source = tmp_path / "tagged.jpg"
        self._tagged_jpeg(source)
        config = ProPipelineConfig(
            input_path=source, output_dir=tmp_path / "out", output_format="jpg", preserve_metadata=False, device="cpu"
        )

        result = ProPipeline(config).process_image(source)

        assert result is not None
        with Image.open(result) as output:
            assert not output.info.get("icc_profile")
            assert output.getexif().get(0x0110) is None


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

    def test_batch_exits_nonzero_when_every_image_fails(self, tmp_path):
        """M-08: a batch in which every image failed must not exit 0."""
        from typer.testing import CliRunner

        from transformation_portal.pipelines.pro_pipeline import app

        bad_dir = tmp_path / "bad"
        bad_dir.mkdir()
        (bad_dir / "a.jpg").write_bytes(b"not an image")
        (bad_dir / "b.png").write_bytes(b"also not an image")

        result = CliRunner().invoke(app, ["batch", str(bad_dir), "--out", str(tmp_path / "out"), "--format", "png"])

        assert result.exit_code == 1, result.output
        assert "all 2 images failed" in (result.stdout or "") + (result.stderr or "")

    @pytest.mark.parametrize("command", ["process", "batch"])
    def test_format_short_flag_is_dash_f(self, tmp_path, command):
        """Regression for L-04: ``--format`` had the bare ``-`` as its short flag, so ``-f png`` exited 2."""
        from typer.testing import CliRunner

        from transformation_portal.pipelines.pro_pipeline import app

        source = tmp_path / "in"
        source.mkdir()
        Image.new("RGB", (8, 8), color=(10, 20, 30)).save(source / "a.png")
        target = source / "a.png" if command == "process" else source
        out = tmp_path / "out"

        result = CliRunner().invoke(
            app, [command, str(target), "--out", str(out), "-f", "png", "--preset", "custom", "--no-ai", "--no-depth"]
        )

        assert result.exit_code == 0, result.output
        assert sorted(p.suffix for p in out.iterdir()) == [".png"]

    def test_batch_honours_pattern(self, tmp_path, monkeypatch):
        """Regression for L-04: ``batch --pattern`` was accepted but ignored (every image was processed)."""
        from typer.testing import CliRunner

        from transformation_portal.pipelines import pro_pipeline

        source = tmp_path / "in"
        source.mkdir()
        for name in ("a.png", "b.tif", "c.TIFF", "d.jpg"):
            Image.new("RGB", (4, 4)).save(source / name)
        seen = []
        monkeypatch.setattr(
            pro_pipeline.ProPipeline,
            "batch_process",
            lambda self, paths: seen.extend(sorted(p.name for p in paths)) or {"processed": len(paths), "failed": 0},
        )

        result = CliRunner().invoke(
            pro_pipeline.app, ["batch", str(source), "--out", str(tmp_path / "out"), "--pattern", "*.{tif,TIFF}"]
        )

        assert result.exit_code == 0, result.output
        assert seen == ["b.tif", "c.TIFF"]

    def test_batch_default_pattern_keeps_historical_discovery(self, tmp_path):
        from transformation_portal.pipelines.pro_pipeline import _DEFAULT_BATCH_PATTERN, _discover_batch_inputs

        for name in ("a.png", "b.JPG", "c.tiff", "d.webp", "e.txt"):
            (tmp_path / name).write_bytes(b"x")
        (tmp_path / "dir.png").mkdir()

        assert sorted(p.name for p in _discover_batch_inputs(tmp_path, _DEFAULT_BATCH_PATTERN)) == ["a.png", "b.JPG", "c.tiff"]

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
