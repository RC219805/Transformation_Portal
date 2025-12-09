"""
Tests for policy engine and routing decisions.
"""

import pytest
from pathlib import Path

from src.transformation_portal.stage_graph.policy import (
    DevicePolicy,
    QualityPolicy,
    CachingPolicy,
    ProcessingPolicy,
    PolicyEngine,
    SceneType,
    QualityPreset,
)


def test_device_policy_cuda_preference():
    """Test device selection with CUDA available."""
    policy = DevicePolicy(
        has_cuda=True,
        has_mps=True,
        prefer_gpu=True,
    )
    
    # General stages prefer CUDA
    assert policy.select_device("enhancement") == "cuda"
    assert policy.select_device("upscaling") == "cuda"


def test_device_policy_mps_fallback():
    """Test device selection with MPS but no CUDA."""
    policy = DevicePolicy(
        has_cuda=False,
        has_mps=True,
        prefer_gpu=True,
    )
    
    assert policy.select_device("enhancement") == "mps"


def test_device_policy_coreml_depth():
    """Test CoreML preference for depth stages."""
    policy = DevicePolicy(
        has_cuda=True,
        has_coreml=True,
        prefer_coreml_depth=True,
    )
    
    # Depth prefers CoreML
    assert policy.select_device("depth_estimation") == "coreml"
    
    # Other stages use CUDA
    assert policy.select_device("enhancement") == "cuda"


def test_device_policy_cpu_fallback():
    """Test CPU fallback when no GPU."""
    policy = DevicePolicy(
        has_cuda=False,
        has_mps=False,
        prefer_gpu=True,
    )
    
    assert policy.select_device("any_stage") == "cpu"


def test_device_policy_batch_memory():
    """Test batch size memory estimation."""
    policy = DevicePolicy(available_memory_gb=8.0)
    
    # Small batch, small image - OK
    assert policy.can_use_batch(batch_size=4, image_size_mp=2.0)
    
    # Large batch, large image - too much memory
    assert not policy.can_use_batch(batch_size=32, image_size_mp=12.0)


def test_quality_policy_draft_preset():
    """Test draft quality preset."""
    policy = QualityPolicy()
    policy.apply_preset(QualityPreset.DRAFT)
    
    assert policy.upscale_factor == 1.0
    assert policy.enhancement_strength < 0.5
    assert not policy.enable_materials


def test_quality_policy_standard_preset():
    """Test standard quality preset."""
    policy = QualityPolicy()
    policy.apply_preset(QualityPreset.STANDARD)
    
    assert policy.upscale_factor == 1.0
    assert policy.enable_materials
    assert 0.4 < policy.enhancement_strength < 0.6


def test_quality_policy_production_preset():
    """Test production quality preset."""
    policy = QualityPolicy()
    policy.apply_preset(QualityPreset.PRODUCTION)
    
    assert policy.upscale_factor == 2.0
    assert policy.enhancement_strength > 0.7
    assert policy.enable_materials
    assert policy.material_strength > 0.7


def test_caching_policy_selective():
    """Test selective caching by stage."""
    policy = CachingPolicy(
        enabled=True,
        cache_depth_maps=True,
        cache_material_masks=True,
        cache_enhanced=False,
    )
    
    assert policy.should_cache_stage("depth_estimation")
    assert policy.should_cache_stage("material_segmentation")
    assert not policy.should_cache_stage("enhancement")


def test_caching_policy_disabled():
    """Test caching completely disabled."""
    policy = CachingPolicy(enabled=False)
    
    assert not policy.should_cache_stage("depth_estimation")
    assert not policy.should_cache_stage("enhancement")


def test_processing_policy_defaults():
    """Test processing policy initialization."""
    policy = ProcessingPolicy()
    
    assert policy.device is not None
    assert policy.quality is not None
    assert policy.caching is not None
    assert policy.scene_type == SceneType.UNKNOWN


def test_policy_engine_create_basic():
    """Test policy engine creates valid policy."""
    engine = PolicyEngine()
    
    policy = engine.create_policy(
        quality_preset=QualityPreset.STANDARD,
    )
    
    assert policy is not None
    assert policy.quality.preset == QualityPreset.STANDARD


def test_policy_engine_scene_adjustments():
    """Test policy engine adjusts for scene type."""
    engine = PolicyEngine()
    
    # Aerial scene
    policy_aerial = engine.create_policy(
        quality_preset=QualityPreset.STANDARD,
        scene_type=SceneType.AERIAL,
    )
    
    # Interior scene
    policy_interior = engine.create_policy(
        quality_preset=QualityPreset.STANDARD,
        scene_type=SceneType.INTERIOR,
    )
    
    # Aerial should have different settings
    assert policy_aerial.quality.clarity_strength != policy_interior.quality.clarity_strength


def test_policy_engine_config_overrides():
    """Test policy engine applies config overrides."""
    engine = PolicyEngine()
    
    config = {
        "upscale_factor": 4.0,
        "enhancement_strength": 0.9,
        "cache_enabled": False,
        "device": "cpu",
    }
    
    policy = engine.create_policy(config=config)
    
    assert policy.quality.upscale_factor == 4.0
    assert policy.quality.enhancement_strength == 0.9
    assert not policy.caching.enabled
    assert not policy.device.prefer_gpu


def test_policy_engine_device_detection():
    """Test policy engine detects devices."""
    engine = PolicyEngine()
    
    policy = engine.create_policy()
    
    # Should detect something (actual values depend on hardware)
    assert isinstance(policy.device.has_cuda, bool)
    assert isinstance(policy.device.has_mps, bool)
    assert policy.device.available_memory_gb > 0


def test_quality_preset_progression():
    """Test quality presets form a progression."""
    policy = QualityPolicy()
    
    # Get values for each preset
    presets = [
        QualityPreset.DRAFT,
        QualityPreset.STANDARD,
        QualityPreset.HIGH,
        QualityPreset.PRODUCTION,
    ]
    
    enhancement_strengths = []
    for preset in presets:
        policy.apply_preset(preset)
        enhancement_strengths.append(policy.enhancement_strength)
    
    # Should be monotonically increasing
    for i in range(len(enhancement_strengths) - 1):
        assert enhancement_strengths[i] <= enhancement_strengths[i + 1]


def test_processing_policy_parallel_settings():
    """Test parallel execution settings."""
    policy = ProcessingPolicy()
    
    assert policy.enable_parallel is True
    assert policy.max_workers > 0


def test_caching_policy_size_limits():
    """Test caching policy size limits."""
    policy = CachingPolicy(
        max_size_gb=5.0,
        max_age_hours=24.0,
    )
    
    assert policy.max_size_gb == 5.0
    assert policy.max_age_hours == 24.0


def test_device_policy_no_gpu_preference():
    """Test device policy with GPU disabled."""
    policy = DevicePolicy(
        has_cuda=True,
        has_mps=True,
        prefer_gpu=False,
    )
    
    # Should use CPU even with GPU available
    assert policy.select_device("enhancement") == "cpu"
