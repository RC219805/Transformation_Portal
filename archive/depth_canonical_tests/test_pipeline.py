"""Tests for DepthPipeline orchestrator."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from transformation_portal.depth_canonical import (
    DepthPipeline,
    DepthPipelineResult,
    PBRConfig,
    ProcessingConfig,
    UnifiedDepthConfig,
)


def test_depth_pipeline_result_initialization():
    """Test DepthPipelineResult initializes with None values."""
    result = DepthPipelineResult()

    assert result.depth_map is None
    assert result.depth_path is None
    assert result.pbr_maps is None
    assert result.pbr_paths is None


def test_depth_pipeline_initialization():
    """Test DepthPipeline initializes with config."""
    config = UnifiedDepthConfig()
    pipeline = DepthPipeline(config)

    assert pipeline.config is config
    assert pipeline.model_registry is not None


def test_depth_pipeline_stores_depth_map():
    """Test pipeline stores depth map in result."""
    config = UnifiedDepthConfig()
    pipeline = DepthPipeline(config)

    depth = np.random.rand(256, 256).astype(np.float32)
    result = pipeline.process(depth_map=depth)

    assert result.depth_map is not None
    np.testing.assert_array_equal(result.depth_map, depth)


def test_depth_pipeline_creates_output_directory():
    """Test pipeline creates output directory if it doesn't exist."""
    config = UnifiedDepthConfig()
    pipeline = DepthPipeline(config)

    depth = np.random.rand(128, 128).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "subdir" / "nested"

        result = pipeline.process(depth_map=depth, output_dir=output_dir, basename="test")

        # Directory should be created
        assert output_dir.exists()
        assert output_dir.is_dir()


def test_depth_pipeline_uses_image_stem_for_basename():
    """Test pipeline uses image filename stem as default basename."""
    config = UnifiedDepthConfig(processing=ProcessingConfig(pbr=PBRConfig(enabled=True)))
    pipeline = DepthPipeline(config)

    depth = np.random.rand(128, 128).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        result = pipeline.process(image_path=Path("/path/to/my_render.jpg"), depth_map=depth, output_dir=output_dir)

        # Basename should be "my_render"
        assert result.pbr_paths["normal"].name == "my_render_normal.png"


def test_depth_pipeline_uses_default_basename_if_no_image_path():
    """Test pipeline uses 'depth_output' if no image_path or basename provided."""
    config = UnifiedDepthConfig(processing=ProcessingConfig(pbr=PBRConfig(enabled=True)))
    pipeline = DepthPipeline(config)

    depth = np.random.rand(128, 128).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        result = pipeline.process(depth_map=depth, output_dir=output_dir)

        # Basename should be "depth_output"
        assert result.pbr_paths["normal"].name == "depth_output_normal.png"


def test_depth_pipeline_custom_pbr_config():
    """Test pipeline uses custom PBR configuration."""
    config = UnifiedDepthConfig(
        processing=ProcessingConfig(pbr=PBRConfig(enabled=True, normal_strength=2.0, roughness_blur_radius=7, ao_bias=0.8))
    )
    pipeline = DepthPipeline(config)

    depth = np.random.rand(256, 256).astype(np.float32)
    result = pipeline.process(depth_map=depth)

    # PBR maps should be generated with custom config
    assert result.pbr_maps is not None
    assert "normal" in result.pbr_maps
    assert "roughness" in result.pbr_maps
    assert "ao" in result.pbr_maps


def test_depth_pipeline_no_output_when_output_dir_none():
    """Test pipeline doesn't save files when output_dir is None."""
    config = UnifiedDepthConfig(processing=ProcessingConfig(pbr=PBRConfig(enabled=True)))
    pipeline = DepthPipeline(config)

    depth = np.random.rand(128, 128).astype(np.float32)

    result = pipeline.process(depth_map=depth)

    # Maps should be generated
    assert result.pbr_maps is not None

    # But not saved
    assert result.pbr_paths is None
