"""Integration tests for pipeline module."""
from __future__ import annotations

import json
import pytest
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
pytest.mark.integration

from lux_depth_v2.pipeline import LuxPipelineV2
from lux_depth_v2.config import PipelineConfig, Preset


class TestLuxPipelineV2Init:
    """Test pipeline initialization."""

    def test_pipeline_init_basic(self, mock_config):
        """Test basic pipeline initialization."""
        pipeline = LuxPipelineV2(mock_config)
        assert pipeline.cfg is not None
        assert pipeline.device is not None
        assert pipeline.upscaler is not None

    def test_pipeline_applies_preset(self, mock_config):
        """Test pipeline applies preset on init."""
        cfg = PipelineConfig(preset=Preset.INTERIOR_LUXURY, device="cpu")
        pipeline = LuxPipelineV2(cfg)
        # Preset should be applied
        assert pipeline.cfg.material_strength == 0.90

    def test_pipeline_device_selection(self):
        """Test device selection logic."""
        cfg = PipelineConfig(device="cpu", upscaler_backend="none")
        pipeline = LuxPipelineV2(cfg)
        assert pipeline.device.type == "cpu"

    def test_pipeline_autocast_cuda_only(self):
        """Test autocast only enabled for CUDA + fp16."""
        cfg_cpu = PipelineConfig(device="cpu", precision="fp16", upscaler_backend="none")
        pipe_cpu = LuxPipelineV2(cfg_cpu)
        assert pipe_cpu.autocast is False  # CPU doesn't use fp16 autocast
        
        cfg_fp32 = PipelineConfig(device="cpu", precision="fp32", upscaler_backend="none")
        pipe_fp32 = LuxPipelineV2(cfg_fp32)
        assert pipe_fp32.autocast is False


@pytest.mark.slow
@pytest.mark.integration
class TestPipelineProcessOne:
    """Test pipeline single image processing."""

    def test_process_one_basic(self, temp_dir, sample_image_file, mock_config):
        """Test processing a single image."""
        mock_config.input_dir = sample_image_file.parent
        mock_config.output_dir = temp_dir
        mock_config.depth_dir = None
        mock_config.strict_depth = False
        mock_config.save_preview_jpg = False
        mock_config.enable_material = False
        
        pipeline = LuxPipelineV2(mock_config)
        result = pipeline.process_one(sample_image_file)
        
        assert result["status"] == "ok"
        assert "timing_s" in result
        assert result["zone_weights"] == "uniform_no_depth"
        
        # Check outputs exist
        stem = sample_image_file.stem
        assert (temp_dir / f"{stem}_master16.tif").exists()
        assert (temp_dir / f"{stem}_upscaled16.tif").exists()
        assert (temp_dir / f"{stem}_marketing.png").exists()
        assert (temp_dir / f"{stem}_report.json").exists()

    def test_process_one_with_depth(self, temp_dir, sample_image_file, sample_depth_file, mock_config):
        """Test processing with depth map."""
        mock_config.input_dir = sample_image_file.parent
        mock_config.output_dir = temp_dir
        mock_config.depth_dir = sample_depth_file.parent
        mock_config.save_preview_jpg = False
        
        pipeline = LuxPipelineV2(mock_config)
        result = pipeline.process_one(sample_image_file, depth_path=sample_depth_file)
        
        assert result["status"] == "ok"
        assert result["zone_weights"] == "depth_percentiles"
        assert result["depth"] == str(sample_depth_file)

    def test_process_one_skip_existing(self, temp_dir, sample_image_file, mock_config):
        """Test skip_existing functionality."""
        mock_config.output_dir = temp_dir
        mock_config.skip_existing = True
        mock_config.save_preview_jpg = False
        
        pipeline = LuxPipelineV2(mock_config)
        
        # First run
        result1 = pipeline.process_one(sample_image_file)
        assert result1["status"] == "ok"
        
        # Second run should skip
        result2 = pipeline.process_one(sample_image_file)
        assert result2["status"] == "skipped"

    def test_process_one_report_content(self, temp_dir, sample_image_file, mock_config):
        """Test report file contains expected information."""
        mock_config.output_dir = temp_dir
        mock_config.save_preview_jpg = False
        
        pipeline = LuxPipelineV2(mock_config)
        result = pipeline.process_one(sample_image_file)
        
        stem = sample_image_file.stem
        report_path = temp_dir / f"{stem}_report.json"
        assert report_path.exists()
        
        with open(report_path) as f:
            report = json.load(f)
        
        assert report["status"] == "ok"
        assert "config" in report
        assert "timing_s" in report
        assert isinstance(report["timing_s"], (int, float))

    def test_process_one_missing_output_dir(self, sample_image_file, mock_config):
        """Test error when output_dir not set."""
        mock_config.output_dir = None
        
        pipeline = LuxPipelineV2(mock_config)
        with pytest.raises(ValueError, match="output_dir is required"):
            pipeline.process_one(sample_image_file)


@pytest.mark.slow
@pytest.mark.integration
class TestPipelineProcessDirectory:
    """Test pipeline directory batch processing."""

    def test_process_directory_basic(self, temp_dir, sample_image_file, mock_config):
        """Test processing entire directory."""
        mock_config.input_dir = sample_image_file.parent
        mock_config.output_dir = temp_dir
        mock_config.save_preview_jpg = False
        
        pipeline = LuxPipelineV2(mock_config)
        results = pipeline.process_directory()
        
        assert len(results) >= 1
        assert results[0]["status"] in ("ok", "skipped", "error")

    def test_process_directory_missing_dirs(self, mock_config):
        """Test error when input/output dirs not set."""
        mock_config.input_dir = None
        
        pipeline = LuxPipelineV2(mock_config)
        with pytest.raises(ValueError, match="input_dir and output_dir"):
            pipeline.process_directory()

    def test_process_directory_empty(self, temp_dir, mock_config):
        """Test processing empty directory."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()
        
        mock_config.input_dir = empty_dir
        mock_config.output_dir = temp_dir / "output"
        
        pipeline = LuxPipelineV2(mock_config)
        results = pipeline.process_directory()
        
        assert results == []


class TestPipelinePresets:
    """Test pipeline with different presets."""

    @pytest.mark.parametrize("preset", [
        Preset.PHOTO_REALISTIC,
        Preset.INTERIOR_LUXURY,
        Preset.EXTERIOR_SHOWCASE,
        Preset.ARCHITECTURAL,
        Preset.ARCHIVAL_QUALITY,
    ])
    def test_preset_initialization(self, preset):
        """Test pipeline initializes with each preset."""
        cfg = PipelineConfig(
            preset=preset,
            device="cpu",
            upscaler_backend="none",
            enable_material=False,
        )
        pipeline = LuxPipelineV2(cfg)
        assert pipeline.cfg.preset == preset


class TestPipelineTiler:
    """Test pipeline tiling functionality."""

    def test_tiler_disabled_by_default(self, mock_config):
        """Test tiler is disabled when post_tile=0."""
        mock_config.post_tile = 0
        pipeline = LuxPipelineV2(mock_config)
        assert pipeline.tiler is None

    def test_tiler_enabled(self, mock_config):
        """Test tiler is created when post_tile > 0."""
        mock_config.post_tile = 512
        mock_config.post_overlap = 32
        pipeline = LuxPipelineV2(mock_config)
        assert pipeline.tiler is not None
        assert pipeline.tiler.tile == 512
        assert pipeline.tiler.overlap == 32
