"""
Tests for concrete stage implementations.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from src.transformation_portal.stage_graph.stage import StageContext, StageStatus
from src.transformation_portal.stage_graph.stages import (
    DepthEstimationStage,
    MaterialSegmentationStage,
    EnhancementStage,
    UpscalingStage,
)


@pytest.fixture
def sample_image():
    """Create sample test image."""
    # 100x100 RGB image
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_depth():
    """Create sample depth map."""
    # 100x100 depth map
    return np.random.rand(100, 100).astype(np.float32)


def test_depth_estimation_stage(sample_image):
    """Test depth estimation stage."""
    stage = DepthEstimationStage(model_size="small")
    
    context = StageContext(
        artifacts={"image": sample_image},
        device="cpu",
        cache_enabled=False,
    )
    
    result = stage.execute(context)
    
    assert result.is_success()
    assert "depth_map" in result.artifacts
    
    depth_map = result.get_artifact("depth_map")
    assert depth_map is not None
    assert depth_map.shape == (100, 100)
    assert depth_map.dtype == np.float32
    
    metadata = result.get_artifact("depth_metadata")
    assert metadata["model_size"] == "small"


def test_depth_estimation_stage_missing_image():
    """Test depth stage handles missing image."""
    stage = DepthEstimationStage()
    
    context = StageContext(cache_enabled=False)
    
    result = stage.execute(context)
    
    assert not result.is_success()
    assert result.status == StageStatus.FAILED
    assert "Missing" in result.error


def test_depth_estimation_cache_key(sample_image):
    """Test depth stage generates stable cache keys."""
    stage = DepthEstimationStage(model_size="small")
    
    context = StageContext(artifacts={"image": sample_image})
    
    # Same image should generate same key
    key1 = stage.get_cache_key(context)
    key2 = stage.get_cache_key(context)
    
    assert key1 == key2
    assert "depth" in key1


def test_material_segmentation_stage(sample_image):
    """Test material segmentation stage."""
    stage = MaterialSegmentationStage(backend="heuristic")
    
    context = StageContext(
        artifacts={"image": sample_image},
        device="cpu",
        cache_enabled=False,
    )
    
    result = stage.execute(context)
    
    assert result.is_success()
    assert "material_masks" in result.artifacts
    
    masks = result.get_artifact("material_masks")
    assert isinstance(masks, dict)
    
    # Should detect at least some materials
    if len(masks) > 0:
        for material_name, mask in masks.items():
            assert mask.shape == (100, 100)
            assert mask.dtype == np.float32


def test_material_segmentation_with_depth(sample_image, sample_depth):
    """Test material segmentation uses depth when available."""
    stage = MaterialSegmentationStage(backend="heuristic")
    
    context = StageContext(
        artifacts={
            "image": sample_image,
            "depth_map": sample_depth,
        },
        device="cpu",
        cache_enabled=False,
    )
    
    result = stage.execute(context)
    
    assert result.is_success()
    
    # Cache key should include depth
    cache_key = stage.get_cache_key(context)
    assert len(cache_key) > 0


def test_enhancement_stage(sample_image, sample_depth):
    """Test enhancement stage."""
    stage = EnhancementStage(
        enhancement_strength=0.7,
        clarity_strength=0.5,
    )
    
    context = StageContext(
        artifacts={
            "image": sample_image,
            "depth_map": sample_depth,
            "material_masks": {},
        },
        device="cpu",
        cache_enabled=False,
    )
    
    result = stage.execute(context)
    
    assert result.is_success()
    assert "enhanced_image" in result.artifacts
    
    enhanced = result.get_artifact("enhanced_image")
    assert enhanced.shape == sample_image.shape
    assert enhanced.dtype == np.uint8


def test_enhancement_stage_dependencies():
    """Test enhancement stage declares dependencies."""
    stage = EnhancementStage()
    
    deps = stage.get_dependencies()
    
    assert "depth_estimation" in deps
    assert "material_segmentation" in deps


def test_enhancement_stage_with_materials(sample_image, sample_depth):
    """Test enhancement with material masks."""
    # Create simple material masks
    h, w = sample_image.shape[:2]
    material_masks = {
        "wood": np.ones((h, w), dtype=np.float32) * 0.5,
        "metal": np.ones((h, w), dtype=np.float32) * 0.3,
    }
    
    stage = EnhancementStage(material_strength=0.8)
    
    context = StageContext(
        artifacts={
            "image": sample_image,
            "depth_map": sample_depth,
            "material_masks": material_masks,
        },
        cache_enabled=False,
    )
    
    result = stage.execute(context)
    
    assert result.is_success()
    
    metadata = result.get_artifact("enhancement_metadata")
    assert "materials_applied" in metadata
    assert "wood" in metadata["materials_applied"]


def test_upscaling_stage(sample_image):
    """Test upscaling stage."""
    stage = UpscalingStage(
        scale_factor=2.0,
        backend="bicubic",
    )
    
    context = StageContext(
        artifacts={"enhanced_image": sample_image},
        device="cpu",
        cache_enabled=False,
    )
    
    result = stage.execute(context)
    
    assert result.is_success()
    assert "upscaled_image" in result.artifacts
    
    upscaled = result.get_artifact("upscaled_image")
    assert upscaled.shape[0] == sample_image.shape[0] * 2
    assert upscaled.shape[1] == sample_image.shape[1] * 2


def test_upscaling_stage_skip_factor_1(sample_image):
    """Test upscaling skips when factor is 1.0."""
    stage = UpscalingStage(scale_factor=1.0)
    
    context = StageContext(
        artifacts={"enhanced_image": sample_image},
        cache_enabled=False,
    )
    
    result = stage.execute(context)
    
    assert result.status == StageStatus.SKIPPED
    
    # Should return input image unchanged
    upscaled = result.get_artifact("upscaled_image")
    assert np.array_equal(upscaled, sample_image)


def test_upscaling_stage_dependencies():
    """Test upscaling stage declares dependencies."""
    stage = UpscalingStage()
    
    deps = stage.get_dependencies()
    
    assert "enhancement" in deps


def test_upscaling_fallback_to_image(sample_image):
    """Test upscaling falls back to 'image' artifact."""
    stage = UpscalingStage(scale_factor=2.0, backend="bicubic")
    
    # No enhanced_image, only image
    context = StageContext(
        artifacts={"image": sample_image},
        cache_enabled=False,
    )
    
    result = stage.execute(context)
    
    assert result.is_success()


def test_stage_cache_keys_deterministic(sample_image):
    """Test all stages generate deterministic cache keys."""
    stages = [
        DepthEstimationStage(),
        MaterialSegmentationStage(),
        EnhancementStage(),
        UpscalingStage(),
    ]
    
    for stage in stages:
        context = StageContext(artifacts={"image": sample_image})
        
        # Generate multiple times
        keys = [stage.get_cache_key(context) for _ in range(5)]
        
        # All should be identical
        assert len(set(keys)) == 1


def test_depth_stage_caching(sample_image):
    """Test depth stage caching works correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        
        stage = DepthEstimationStage()
        
        context = StageContext(
            artifacts={"image": sample_image},
            device="cpu",
            cache_enabled=True,
            cache_dir=cache_dir,
        )
        
        # First execution
        result1 = stage.execute(context)
        assert not result1.cache_hit
        
        # Second execution
        result2 = stage.execute(context)
        assert result2.cache_hit
        
        # Results should match
        assert np.array_equal(
            result1.get_artifact("depth_map"),
            result2.get_artifact("depth_map"),
        )


def test_enhancement_stage_cache_invalidation(sample_image, sample_depth):
    """Test enhancement cache invalidates on input change."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        
        stage = EnhancementStage()
        
        # First execution
        context1 = StageContext(
            artifacts={
                "image": sample_image,
                "depth_map": sample_depth,
            },
            cache_enabled=True,
            cache_dir=cache_dir,
        )
        result1 = stage.execute(context1)
        assert not result1.cache_hit
        
        # Second execution with different depth
        different_depth = sample_depth * 2.0
        context2 = StageContext(
            artifacts={
                "image": sample_image,
                "depth_map": different_depth,
            },
            cache_enabled=True,
            cache_dir=cache_dir,
        )
        result2 = stage.execute(context2)
        assert not result2.cache_hit  # Different depth = different cache key
