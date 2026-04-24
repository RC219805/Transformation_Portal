from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from transformation_portal.lux_depth_v3.pixel_ops_decider import decide_pixel_ops
from transformation_portal.lux_depth_v3.pixel_ops_executor import _compute_delta_stats, apply_pixel_ops
from transformation_portal.lux_depth_v3.pixel_ops_registry import (
    OP_REGISTRY,
    foliage_vibrance_boost,
    stone_microcontrast,
    water_reflection_enhance,
)

pytestmark = pytest.mark.unit


@dataclass
class DummyConfig:
    enabled: bool = True
    apply_pixel_ops: bool = True
    glass_response_enabled: bool = True
    min_coverage_px: int = 500
    min_mean_conf: float = 0.2
    refinement_strategy: str = "canary"


def _make_inputs():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[10:50, 10:50] = 1.0
    segmentation_result = {"materials": {"glass": mask}}
    response_plan = {
        "per_class": {
            "glass": {
                "coverage_px": int(mask.sum()),
                "mean_conf": 0.6,
                "edge_conf": 0.4,
            }
        }
    }
    return image, segmentation_result, response_plan


def test_decider_will_apply_when_enabled():
    config = DummyConfig()
    _, _, response_plan = _make_inputs()
    decision = decide_pixel_ops("glass", response_plan["per_class"]["glass"], config, registry=OP_REGISTRY)
    assert decision["will_apply"] is True
    assert "brightness_boost" in decision["recommended_ops"]


def test_decider_blocks_low_confidence_material_masks():
    config = DummyConfig()
    stats = {
        "coverage_px": 2000,
        "mean_conf": 0.1,
        "edge_conf": 0.6,
    }

    decision = decide_pixel_ops("foliage", stats, config, registry=OP_REGISTRY)

    assert decision["will_apply"] is False
    assert decision["eligible"] is False
    assert decision["reason"] == "below_confidence_threshold"
    assert "below_confidence_threshold" in decision["blocked_by"]


def test_apply_pixel_ops_emits_telemetry():
    config = DummyConfig()
    image, segmentation_result, response_plan = _make_inputs()
    _, telemetry = apply_pixel_ops(image, segmentation_result, response_plan, config, registry=OP_REGISTRY)

    assert telemetry["enabled"] is True
    assert telemetry["applied"]
    assert telemetry["timing_ms"]["total"] >= 0.0


def test_apply_pixel_ops_blocks_low_confidence_material_masks():
    config = DummyConfig()
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[8:56, 8:56] = 1.0
    segmentation_result = {"materials": {"foliage": mask}}
    response_plan = {
        "per_class": {
            "foliage": {
                "coverage_px": int(mask.sum()),
                "mean_conf": 0.1,
                "edge_conf": 0.6,
            }
        }
    }

    output, telemetry = apply_pixel_ops(image, segmentation_result, response_plan, config, registry=OP_REGISTRY)

    assert np.array_equal(output, image)
    assert telemetry["applied"] == []
    assert telemetry["blocked"][0]["material"] == "foliage"
    assert telemetry["blocked"][0]["reason"] == "below_confidence_threshold"


def test_apply_pixel_ops_rechecks_stale_plan_decisions():
    config = DummyConfig()
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[8:56, 8:56] = 1.0
    segmentation_result = {"materials": {"foliage": mask}}
    response_plan = {
        "per_class": {
            "foliage": {
                "coverage_px": int(mask.sum()),
                "mean_conf": 0.1,
                "edge_conf": 0.6,
                "pixel_ops": {
                    "eligible": True,
                    "enabled": True,
                    "implemented": True,
                    "recommended_ops": ["vibrance_boost"],
                    "should_apply": True,
                    "will_apply": True,
                    "blocked_by": [],
                    "reason": "stale_cached_plan",
                },
            }
        }
    }

    output, telemetry = apply_pixel_ops(image, segmentation_result, response_plan, config, registry=OP_REGISTRY)

    assert np.array_equal(output, image)
    assert telemetry["applied"] == []
    assert telemetry["blocked"][0]["reason"] == "below_confidence_threshold"


def test_apply_pixel_ops_disabled_still_emits_object():
    config = DummyConfig(apply_pixel_ops=False)
    image, segmentation_result, response_plan = _make_inputs()
    _, telemetry = apply_pixel_ops(image, segmentation_result, response_plan, config, registry=OP_REGISTRY)

    assert telemetry["enabled"] is False
    assert telemetry["applied"] == []
    assert telemetry["blocked"] == []


def test_compute_delta_stats_handles_mask_shapes():
    before = np.zeros((4, 4, 3), dtype=np.uint8)
    after = before.copy()
    after[1:3, 1:3] = 10
    mask = np.zeros((4, 4), dtype=np.float32)
    mask[1:3, 1:3] = 1.0

    stats_2d = _compute_delta_stats(before, after, mask)
    stats_3d = _compute_delta_stats(before, after, mask[..., None])

    assert isinstance(stats_2d["inside_mask_mean_abs"], float)
    assert isinstance(stats_2d["outside_mask_mean_abs"], float)
    assert isinstance(stats_3d["inside_mask_mean_abs"], float)
    assert isinstance(stats_3d["outside_mask_mean_abs"], float)


def test_stone_microcontrast_implementation():
    """Test that stone microcontrast is implemented and works."""
    # Create test image with variation (not flat gray)
    image = np.ones((32, 32, 3), dtype=np.uint8) * 100
    # Add some variation in the center
    image[8:24, 8:24] = 150

    mask = np.zeros((32, 32), dtype=np.float32)
    mask[8:24, 8:24] = 1.0

    # Apply stone microcontrast
    params = {"strength": 0.12}
    result = stone_microcontrast(image, mask, params)

    # Check result shape and dtype
    assert result.shape == image.shape
    assert result.dtype == image.dtype

    # Check that operation was applied (result differs from input within mask)
    # The contrast operation should enhance the difference from midpoint
    assert not np.array_equal(result[8:24, 8:24], image[8:24, 8:24])

    # Check that areas outside mask are unchanged
    assert np.array_equal(result[0:8, 0:8], image[0:8, 0:8])


def test_stone_ops_in_registry():
    """Test that stone ops are properly registered and marked as implemented."""
    assert "stone" in OP_REGISTRY
    assert "microcontrast" in OP_REGISTRY["stone"]

    stone_op = OP_REGISTRY["stone"]["microcontrast"]
    assert stone_op.implemented is True
    assert stone_op.op == stone_microcontrast
    assert "texture" in stone_op.description.lower() or "stone" in stone_op.description.lower()


def test_water_reflection_implementation():
    """Test that water reflection enhancement is implemented and works."""
    # Create test image with variation
    image = np.ones((32, 32, 3), dtype=np.uint8) * 100
    # Add brighter area in center (simulating water surface)
    image[8:24, 8:24] = 120

    mask = np.zeros((32, 32), dtype=np.float32)
    mask[8:24, 8:24] = 1.0

    # Apply water reflection enhance
    params = {"strength": 0.10}
    result = water_reflection_enhance(image, mask, params)

    # Check result shape and dtype
    assert result.shape == image.shape
    assert result.dtype == image.dtype

    # Check that operation was applied (result differs from input within mask)
    assert not np.array_equal(result[8:24, 8:24], image[8:24, 8:24])

    # Check that areas outside mask are unchanged
    assert np.array_equal(result[0:8, 0:8], image[0:8, 0:8])


def test_water_ops_in_registry():
    """Test that water ops are properly registered and marked as implemented."""
    assert "water" in OP_REGISTRY
    assert "reflection_enhance" in OP_REGISTRY["water"]

    water_op = OP_REGISTRY["water"]["reflection_enhance"]
    assert water_op.implemented is True
    assert water_op.op == water_reflection_enhance
    assert "reflection" in water_op.description.lower() or "water" in water_op.description.lower()


def test_foliage_vibrance_implementation():
    """Test that foliage vibrance boost is implemented and works."""
    # Create test image with green-ish tones
    image = np.ones((32, 32, 3), dtype=np.uint8)
    image[..., 0] = 50  # Red
    image[..., 1] = 120  # Green
    image[..., 2] = 60  # Blue
    # Add variation in center
    image[8:24, 8:24, 1] = 140

    mask = np.zeros((32, 32), dtype=np.float32)
    mask[8:24, 8:24] = 1.0

    # Apply foliage vibrance boost
    params = {"strength": 0.08}
    result = foliage_vibrance_boost(image, mask, params)

    # Check result shape and dtype
    assert result.shape == image.shape
    assert result.dtype == image.dtype

    # Check that operation was applied (result differs from input within mask)
    # Green channel should be enhanced
    assert not np.array_equal(result[8:24, 8:24, 1], image[8:24, 8:24, 1])

    # Check that areas outside mask are unchanged
    assert np.array_equal(result[0:8, 0:8], image[0:8, 0:8])


def test_foliage_ops_in_registry():
    """Test that foliage ops are properly registered and marked as implemented."""
    assert "foliage" in OP_REGISTRY
    assert "vibrance_boost" in OP_REGISTRY["foliage"]

    foliage_op = OP_REGISTRY["foliage"]["vibrance_boost"]
    assert foliage_op.implemented is True
    assert foliage_op.op == foliage_vibrance_boost
    assert "vibrance" in foliage_op.description.lower() or "foliage" in foliage_op.description.lower()
