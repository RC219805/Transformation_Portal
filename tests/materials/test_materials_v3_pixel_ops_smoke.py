from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from transformation_portal.lux_depth_v3.pixel_ops_decider import decide_pixel_ops
from transformation_portal.lux_depth_v3.pixel_ops_executor import apply_pixel_ops, _compute_delta_stats
from transformation_portal.lux_depth_v3.pixel_ops_registry import OP_REGISTRY


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


def test_apply_pixel_ops_emits_telemetry():
    config = DummyConfig()
    image, segmentation_result, response_plan = _make_inputs()
    _, telemetry = apply_pixel_ops(image, segmentation_result, response_plan, config, registry=OP_REGISTRY)

    assert telemetry["enabled"] is True
    assert telemetry["applied"]
    assert telemetry["timing_ms"]["total"] >= 0.0


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
