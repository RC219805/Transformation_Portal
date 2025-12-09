"""
Integration tests for complete stage graph pipelines.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from src.transformation_portal.stage_graph import (
    StageContext,
    GraphBuilder,
    PolicyEngine,
    QualityPreset,
    SceneType,
    DepthEstimationStage,
    MaterialSegmentationStage,
    EnhancementStage,
    UpscalingStage,
)


@pytest.fixture
def sample_image():
    """Create sample test image."""
    # 200x200 RGB image with some structure
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    
    # Add some features
    img[50:150, 50:150] = [200, 150, 100]  # Center region
    img[0:50, :] = [100, 100, 150]         # Top region
    
    return img


def test_full_pipeline_execution(sample_image):
    """Test complete pipeline execution."""
    # Build graph
    graph = (
        GraphBuilder("lux_pipeline")
        .add(DepthEstimationStage(model_size="small"))
        .add(MaterialSegmentationStage(backend="heuristic"))
        .add(EnhancementStage(enhancement_strength=0.7))
        .add(UpscalingStage(scale_factor=2.0, backend="bicubic"))
        .build()
    )
    
    # Create context
    context = StageContext(
        artifacts={"image": sample_image},
        device="cpu",
        cache_enabled=False,
    )
    
    # Execute
    execution = graph.execute(context, parallel=False)
    
    assert execution.success
    assert len(execution.stage_results) == 4
    
    # Check final output
    final_image = context.get_artifact("upscaled_image")
    assert final_image is not None
    assert final_image.shape[0] == sample_image.shape[0] * 2
    assert final_image.shape[1] == sample_image.shape[1] * 2


def test_pipeline_with_caching(sample_image):
    """Test pipeline with caching enabled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        
        # Build graph
        graph = (
            GraphBuilder("lux_pipeline")
            .add(DepthEstimationStage())
            .add(MaterialSegmentationStage())
            .add(EnhancementStage())
            .build()
        )
        
        # First execution
        context1 = StageContext(
            artifacts={"image": sample_image},
            device="cpu",
            cache_enabled=True,
            cache_dir=cache_dir,
        )
        execution1 = graph.execute(context1)
        
        assert execution1.success
        assert execution1.cache_miss_count == 3
        assert execution1.cache_hit_count == 0
        
        # Second execution - should hit cache
        context2 = StageContext(
            artifacts={"image": sample_image},
            device="cpu",
            cache_enabled=True,
            cache_dir=cache_dir,
        )
        execution2 = graph.execute(context2)
        
        assert execution2.success
        assert execution2.cache_hit_count == 3
        assert execution2.cache_miss_count == 0
        
        # Should be faster
        assert execution2.total_duration_ms < execution1.total_duration_ms


def test_pipeline_with_policy_engine(sample_image):
    """Test pipeline with policy engine."""
    # Create policy
    engine = PolicyEngine()
    policy = engine.create_policy(
        quality_preset=QualityPreset.HIGH,
        scene_type=SceneType.INTERIOR,
    )
    
    # Build graph with policy settings
    graph = (
        GraphBuilder("lux_pipeline")
        .add(DepthEstimationStage())
        .add(MaterialSegmentationStage())
        .add(EnhancementStage(
            enhancement_strength=policy.quality.enhancement_strength,
            clarity_strength=policy.quality.clarity_strength,
        ))
        .add(UpscalingStage(
            scale_factor=policy.quality.upscale_factor,
        ))
        .build()
    )
    
    # Create context with policy
    context = StageContext(
        artifacts={"image": sample_image},
        device=policy.device.select_device("depth_estimation"),
        cache_enabled=policy.caching.enabled,
    )
    
    execution = graph.execute(
        context,
        parallel=policy.enable_parallel,
        max_workers=policy.max_workers,
    )
    
    assert execution.success


def test_pipeline_parallel_execution(sample_image):
    """Test pipeline executes independent stages in parallel."""
    # Build graph with no dependencies between material and depth
    graph = (
        GraphBuilder("lux_pipeline")
        .add(DepthEstimationStage())
        .add(MaterialSegmentationStage())  # No explicit depth dependency
        .build()
    )
    
    context = StageContext(
        artifacts={"image": sample_image},
        cache_enabled=False,
    )
    
    # Should execute in parallel
    execution = graph.execute(context, parallel=True, max_workers=2)
    
    assert execution.success
    assert len(execution.stage_results) == 2


def test_pipeline_cache_speedup(sample_image):
    """Test pipeline achieves significant speedup with caching."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        
        graph = (
            GraphBuilder("lux_pipeline")
            .add(DepthEstimationStage())
            .add(MaterialSegmentationStage())
            .add(EnhancementStage())
            .build()
        )
        
        # First run - populate cache
        context1 = StageContext(
            artifacts={"image": sample_image},
            cache_enabled=True,
            cache_dir=cache_dir,
        )
        execution1 = graph.execute(context1)
        time1 = execution1.total_duration_ms
        
        # Second run - from cache
        context2 = StageContext(
            artifacts={"image": sample_image},
            cache_enabled=True,
            cache_dir=cache_dir,
        )
        execution2 = graph.execute(context2)
        time2 = execution2.total_duration_ms
        
        # Cache stats
        stats = execution2.get_cache_stats()
        
        assert stats["hit_rate"] == 1.0
        assert time2 < time1  # Should be faster
        
        # Estimate speedup
        speedup = time1 / time2 if time2 > 0 else 1.0
        assert speedup > 1.0


def test_pipeline_error_recovery(sample_image):
    """Test pipeline handles stage failures gracefully."""
    
    class FailingStage(EnhancementStage):
        """Stage that fails."""
        def __init__(self):
            super().__init__()
            self.name = "failing_enhancement"  # Different name
        
        def get_dependencies(self) -> list:
            return []  # No dependencies
        
        def compute(self, context):
            raise ValueError("Intentional failure")
    
    graph = (
        GraphBuilder("lux_pipeline")
        .add(DepthEstimationStage())
        .add(FailingStage())  # This will fail
        .build()
    )
    
    context = StageContext(
        artifacts={"image": sample_image},
        cache_enabled=False,
    )
    
    execution = graph.execute(context, parallel=False)
    
    assert not execution.success
    assert execution.error is not None
    
    # First stage should have completed
    assert execution.get_result("depth_estimation").is_success()


def test_pipeline_artifact_propagation(sample_image):
    """Test artifacts propagate correctly through pipeline."""
    graph = (
        GraphBuilder("lux_pipeline")
        .add(DepthEstimationStage())
        .add(MaterialSegmentationStage())
        .add(EnhancementStage())
        .build()
    )
    
    context = StageContext(
        artifacts={"image": sample_image},
        cache_enabled=False,
    )
    
    execution = graph.execute(context)
    
    # Check each stage's artifacts are in context
    assert context.get_artifact("depth_map") is not None
    assert context.get_artifact("material_masks") is not None
    assert context.get_artifact("enhanced_image") is not None


def test_pipeline_different_quality_presets(sample_image):
    """Test pipeline with different quality presets."""
    presets = [
        QualityPreset.DRAFT,
        QualityPreset.STANDARD,
        QualityPreset.PRODUCTION,
    ]
    
    for preset in presets:
        engine = PolicyEngine()
        policy = engine.create_policy(quality_preset=preset)
        
        graph = (
            GraphBuilder(f"lux_pipeline_{preset.value}")
            .add(DepthEstimationStage())
            .add(MaterialSegmentationStage())
            .add(EnhancementStage(
                enhancement_strength=policy.quality.enhancement_strength,
            ))
            .build()
        )
        
        context = StageContext(
            artifacts={"image": sample_image},
            cache_enabled=False,
        )
        
        execution = graph.execute(context)
        assert execution.success


def test_pipeline_metrics_collection(sample_image):
    """Test pipeline collects comprehensive metrics."""
    graph = (
        GraphBuilder("lux_pipeline")
        .add(DepthEstimationStage())
        .add(MaterialSegmentationStage())
        .add(EnhancementStage())
        .build()
    )
    
    context = StageContext(
        artifacts={"image": sample_image},
        cache_enabled=False,
    )
    
    execution = graph.execute(context, run_id="test-metrics-123")
    
    # Check execution metadata
    assert execution.run_id == "test-metrics-123"
    assert execution.total_duration_ms > 0
    assert len(execution.execution_order) == 3
    
    # Check stage-level metrics
    for stage_name in execution.execution_order:
        result = execution.get_result(stage_name)
        assert result.duration_ms > 0
        assert result.timestamp > 0


def test_pipeline_scene_type_routing(sample_image):
    """Test pipeline routes differently for scene types."""
    engine = PolicyEngine()
    
    scenes = [
        SceneType.INTERIOR,
        SceneType.EXTERIOR,
        SceneType.AERIAL,
    ]
    
    for scene_type in scenes:
        policy = engine.create_policy(
            quality_preset=QualityPreset.STANDARD,
            scene_type=scene_type,
        )
        
        # Policy should adjust parameters
        assert policy.scene_type == scene_type
        
        # Build and execute pipeline
        graph = (
            GraphBuilder(f"lux_pipeline_{scene_type.value}")
            .add(DepthEstimationStage())
            .add(MaterialSegmentationStage())
            .add(EnhancementStage(
                enhancement_strength=policy.quality.enhancement_strength,
                clarity_strength=policy.quality.clarity_strength,
            ))
            .build()
        )
        
        context = StageContext(
            artifacts={"image": sample_image},
            cache_enabled=False,
        )
        
        execution = graph.execute(context)
        assert execution.success
