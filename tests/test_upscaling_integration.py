#!/usr/bin/env python3
"""Test upscaler integration with stage graph."""

import tempfile
from pathlib import Path

import numpy as np

from transformation_portal.stage_graph.stage import StageContext

# Test UpscalingStage integration
from transformation_portal.stage_graph.stages import UpscalingStage


def test_upscaling_stage_bicubic():
    """Test UpscalingStage with bicubic backend."""
    print("\n=== Testing UpscalingStage with bicubic backend ===")

    # Create stage
    stage = UpscalingStage(
        scale_factor=2.0,
        backend="bicubic",
        version="1.0.0",
    )

    # Create test image
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    # Create context
    context = StageContext(
        device="cpu",
        artifacts={"enhanced_image": test_image},
    )

    # Run stage
    result = stage.compute(context)

    print(f"Status: {result.status}")
    print(f"Duration: {result.duration_ms:.2f}ms")
    print(f"Input shape: {test_image.shape}")
    print(f"Output shape: {result.artifacts['upscaled_image'].shape}")

    assert result.status.value == "completed"
    assert result.artifacts["upscaled_image"].shape == (200, 200, 3)
    print("✅ Bicubic upscaling test passed!")


def test_upscaling_stage_with_default():
    """Test UpscalingStage with 'default' backend (should use bicubic)."""
    print("\n=== Testing UpscalingStage with 'default' backend ===")

    stage = UpscalingStage(
        scale_factor=2.0,
        backend="default",
        version="1.0.0",
    )

    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    context = StageContext(
        device="cpu",
        artifacts={"enhanced_image": test_image},
    )

    result = stage.compute(context)

    print(f"Status: {result.status}")
    assert result.status.value == "completed"
    print("✅ Default backend test passed!")


def test_upscaling_stage_realesrgan_fallback():
    """Test UpscalingStage with realesrgan backend (should fallback to bicubic if ML deps missing)."""
    print("\n=== Testing UpscalingStage with realesrgan backend (fallback test) ===")

    stage = UpscalingStage(
        scale_factor=2.0,
        backend="realesrgan",
        version="1.0.0",
    )

    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    context = StageContext(
        device="cpu",
        artifacts={"enhanced_image": test_image},
    )

    # Should fallback to bicubic gracefully
    result = stage.compute(context)

    print(f"Status: {result.status}")
    assert result.status.value == "completed"
    print("✅ Real-ESRGAN fallback test passed!")


def test_upscaling_stage_skip():
    """Test UpscalingStage with scale_factor=1.0 (should skip)."""
    print("\n=== Testing UpscalingStage with scale_factor=1.0 (skip test) ===")

    stage = UpscalingStage(
        scale_factor=1.0,
        backend="bicubic",
        version="1.0.0",
    )

    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    context = StageContext(
        device="cpu",
        artifacts={"enhanced_image": test_image},
    )

    result = stage.compute(context)

    print(f"Status: {result.status}")
    assert result.status.value == "skipped"
    print("✅ Skip test passed!")


if __name__ == "__main__":
    test_upscaling_stage_bicubic()
    test_upscaling_stage_with_default()
    test_upscaling_stage_realesrgan_fallback()
    test_upscaling_stage_skip()
    print("\n✅ All integration tests passed!")
