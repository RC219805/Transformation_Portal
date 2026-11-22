from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from transformation_portal.lux_depth_v3.materials_v3 import MaterialsV3Engine
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
    enable_materials_v3: bool = True
    enabled: bool = True
    apply_pixel_ops: bool = True
    glass_response_enabled: bool = True
    min_coverage_px: int = 500
    min_mean_conf: float = 0.2
    refinement_strategy: str = "canary"
    quality_tier: str = "standard"


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
        "mean_conf": 0.6,
        "material_confidence": 0.1,
        "edge_conf": 0.6,
    }

    decision = decide_pixel_ops("foliage", stats, config, registry=OP_REGISTRY)

    assert decision["will_apply"] is False
    assert decision["eligible"] is False
    assert decision["reason"] == "below_confidence_threshold"
    assert "below_confidence_threshold" in decision["blocked_by"]


def test_apex_decider_rejects_raw_clip_similarity_as_pixel_op_authority():
    config = DummyConfig(quality_tier="apex")
    stats = {
        "coverage_px": 2000,
        "mean_conf": 0.6,
        "material_confidence": 0.9,
        "confidence_score_type": "raw_clip_similarity",
        "edge_conf": 0.6,
    }

    decision = decide_pixel_ops("water", stats, config, registry=OP_REGISTRY)

    assert decision["will_apply"] is False
    assert decision["reason"] == "unsupported_confidence_score_type"
    assert "unsupported_confidence_score_type" in decision["blocked_by"]


def test_apex_decider_accepts_calibrated_clip_softmax_margin_confidence():
    config = DummyConfig(quality_tier="apex")
    stats = {
        "coverage_px": 2000,
        "mean_conf": 0.6,
        "material_confidence": 0.82,
        "confidence_score_type": "clip_softmax_margin_v1",
        "clip_softmax_probability": 0.82,
        "clip_top2_margin": 0.24,
        "calibration_version": "materials_v3_calibration_v1",
        "edge_conf": 0.6,
    }

    decision = decide_pixel_ops("water", stats, config, registry=OP_REGISTRY)

    assert decision["will_apply"] is True
    assert decision["authority_score_type_accepted"] is True
    assert decision["confidence_score_type"] == "clip_softmax_margin_v1"


def test_decider_does_not_apply_taxonomy_threshold_to_mask_coverage_mean():
    config = DummyConfig()
    stats = {
        "coverage_px": 2000,
        "mean_conf": 0.25,
        "edge_conf": 0.6,
    }

    decision = decide_pixel_ops("foliage", stats, config, registry=OP_REGISTRY)

    assert decision["will_apply"] is True
    assert decision["eligible"] is True
    assert decision["reason"] == "material_present_with_coverage"


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
                "mean_conf": 0.6,
                "material_confidence": 0.1,
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
                "mean_conf": 0.6,
                "material_confidence": 0.1,
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


def test_apply_pixel_ops_falls_back_when_serialized_recommended_ops_is_not_a_sequence():
    config = DummyConfig()
    image = np.ones((64, 64, 3), dtype=np.uint8) * 48
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[8:56, 8:56] = 1.0
    segmentation_result = {"materials": {"glass": mask}}
    response_plan = {
        "per_class": {
            "glass": {
                "coverage_px": int(mask.sum()),
                "mean_conf": 0.6,
                "edge_conf": 0.4,
                "pixel_ops": {"recommended_ops": "brightness_boost"},
            }
        }
    }

    _, telemetry = apply_pixel_ops(image, segmentation_result, response_plan, config, registry=OP_REGISTRY)

    assert telemetry["blocked"] == []
    assert telemetry["applied"][0]["ops"] == list(OP_REGISTRY["glass"].keys())


def test_apply_pixel_ops_filters_serialized_recommended_ops_against_registry():
    config = DummyConfig()
    image = np.ones((64, 64, 3), dtype=np.uint8) * 48
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[8:56, 8:56] = 1.0
    segmentation_result = {"materials": {"glass": mask}}
    response_plan = {
        "per_class": {
            "glass": {
                "coverage_px": int(mask.sum()),
                "mean_conf": 0.6,
                "edge_conf": 0.4,
                "pixel_ops": {"recommended_ops": ["stale_op", "brightness_boost", 17]},
            }
        }
    }

    _, telemetry = apply_pixel_ops(image, segmentation_result, response_plan, config, registry=OP_REGISTRY)

    assert telemetry["blocked"] == []
    assert telemetry["applied"][0]["ops"] == ["brightness_boost"]


def test_materials_v3_engine_uses_segmentation_material_confidence_for_pixel_ops():
    config = DummyConfig()
    image = np.ones((64, 64, 3), dtype=np.uint8) * 48
    mask = np.ones((64, 64), dtype=np.float32)
    engine = MaterialsV3Engine(config)

    result = engine.process(
        image,
        {
            "materials": {"foliage": mask},
            "segmentation_metadata": {"material_confidences": {"foliage": 0.1}},
        },
    )

    plan_entry = result["materials_v3_response_plan"]["per_class"]["foliage"]
    assert plan_entry["material_confidence"] == pytest.approx(0.1)
    assert plan_entry["pixel_ops"]["reason"] == "below_confidence_threshold"
    assert result["materials_v3_pixel_ops"]["blocked"][0]["reason"] == "below_confidence_threshold"


def test_materials_v3_engine_propagates_confidence_contract_fields():
    config = DummyConfig(quality_tier="apex")
    image = np.ones((64, 64, 3), dtype=np.uint8) * 48
    mask = np.ones((64, 64), dtype=np.float32)
    engine = MaterialsV3Engine(config)

    result = engine.process(
        image,
        {
            "materials": {"water": mask},
            "segmentation_metadata": {
                "material_confidences": {"water": 0.78},
                "material_confidence_evidence": {
                    "water": {
                        "material_confidence": 0.78,
                        "confidence_score_type": "clip_softmax_margin_v1",
                        "raw_clip_similarity": 0.24,
                        "clip_softmax_probability": 0.78,
                        "clip_top2_margin": 0.11,
                        "calibration_version": "materials_v3_calibration_v1",
                    }
                },
            },
        },
    )

    plan_entry = result["materials_v3_response_plan"]["per_class"]["water"]
    assert plan_entry["material_confidence"] == pytest.approx(0.78)
    assert plan_entry["confidence_score_type"] == "clip_softmax_margin_v1"
    assert plan_entry["raw_clip_similarity"] == pytest.approx(0.24)
    assert result["materials_v3_pixel_ops"]["applied"][0]["confidence_score_type"] == "clip_softmax_margin_v1"


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


def _glass_and_water_plan(mask):
    entry = {"coverage_px": int(mask.sum()), "mean_conf": 0.6, "material_confidence": 0.9, "edge_conf": 0.4}
    return {"per_class": {"glass": dict(entry), "water": dict(entry)}}


@pytest.mark.parametrize(
    "bad_mask_shape",
    [(64, 64, 3), (32, 32), (64,)],
    ids=["multi_channel", "dimension_mismatch", "one_dimensional"],
)
def test_apply_pixel_ops_blocks_only_the_material_with_an_invalid_mask(bad_mask_shape):
    """LC-17: one malformed mask must block its own material, not abort every material."""
    config = DummyConfig()
    image = np.full((64, 64, 3), 128, dtype=np.uint8)
    good = np.zeros((64, 64), dtype=np.float32)
    good[10:50, 10:50] = 1.0
    bad = np.ones(bad_mask_shape, dtype=np.float32)

    output, telemetry = apply_pixel_ops(
        image,
        {"materials": {"water": good, "glass": bad}},
        _glass_and_water_plan(good),
        config,
        registry=OP_REGISTRY,
    )

    blocked = {entry["material"]: entry for entry in telemetry["blocked"]}
    assert blocked["glass"]["reason"] == "invalid_mask_shape"
    assert [entry["material"] for entry in telemetry["applied"]] == ["water"]
    assert not np.array_equal(output[20:40, 20:40], image[20:40, 20:40])


def test_apply_pixel_ops_valid_masks_unchanged_by_per_material_validation():
    """LC-17: validating masks up front must not alter output for valid (H, W) / (H, W, 1) masks."""
    config = DummyConfig()
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[10:50, 10:50] = 1.0
    plan = _glass_and_water_plan(mask)

    out_2d, tel_2d = apply_pixel_ops(image, {"materials": {"glass": mask}}, plan, config, registry=OP_REGISTRY)
    out_3d, tel_3d = apply_pixel_ops(image, {"materials": {"glass": mask[..., None]}}, plan, config, registry=OP_REGISTRY)

    assert np.array_equal(out_2d, out_3d)
    assert tel_2d["blocked"] == tel_3d["blocked"] == []
    assert [entry["material"] for entry in tel_2d["applied"]] == ["glass"]


def _registered_op(op_name):
    return (OP_REGISTRY["glass"].get(op_name) or OP_REGISTRY["sky"][op_name]).op


@pytest.mark.parametrize("shape", [(40, 56), (48, 48)], ids=["rectangular", "square"])
@pytest.mark.parametrize("op_name", ["brightness_boost", "edge_contrast", "dehaze", "gradient_smooth"])
def test_pixel_ops_support_grayscale_images(shape, op_name):
    """LC-17: 2-D grayscale input keeps its shape (no (N, N, N) broadcast, no ValueError)."""
    gray = np.full(shape, 0.5, dtype=np.float32)
    gray[::2] = 0.3
    mask = np.zeros(shape, dtype=np.float32)
    mask[: shape[0] // 2] = 1.0

    result = _registered_op(op_name)(gray, mask, {"normalized": gray, "scale": 1.0})

    assert result.shape == shape
    assert np.array_equal(result[shape[0] // 2 :], gray[shape[0] // 2 :])
    assert not np.array_equal(result[: shape[0] // 2], gray[: shape[0] // 2])


@pytest.mark.parametrize("op_name", ["brightness_boost", "dehaze", "gradient_smooth", "temperature_shift"])
def test_pixel_ops_leave_alpha_channel_untouched(op_name):
    """LC-17: on RGBA input only the colour channels are edited; alpha passes through."""
    rng = np.random.default_rng(3)
    rgb = rng.random((32, 32, 3), dtype=np.float32)
    rgba = np.dstack([rgb, np.full((32, 32), 0.6, dtype=np.float32)])
    mask = np.zeros((32, 32), dtype=np.float32)
    mask[4:28, 4:28] = 1.0
    op = _registered_op(op_name)

    result = op(rgba, mask, {"normalized": rgba, "scale": 1.0})
    rgb_only = op(rgb, mask, {"normalized": rgb, "scale": 1.0})

    assert np.array_equal(result[..., 3], rgba[..., 3])
    assert np.array_equal(result[..., :3], rgb_only)
