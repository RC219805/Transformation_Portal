from __future__ import annotations

import numpy as np
import pytest

from lux_depth_v3.enhance.depth_zones import DepthZoneConfig, DepthZoneGenerator


def _make_depth(h: int = 64, w: int = 96) -> np.ndarray:
    # Smooth near→far gradient in [0,1]
    x = np.linspace(0.0, 1.0, w, dtype=np.float32)
    depth = np.tile(x[None, :], (h, 1))
    return depth


def test_config_validate_defaults() -> None:
    cfg = DepthZoneConfig()
    cfg.validate()


def test_generate_zones_shape_and_sum_to_one() -> None:
    cfg = DepthZoneConfig()
    gen = DepthZoneGenerator(cfg)

    depth = _make_depth()
    zones, stats = gen.generate_zones(depth=depth, image=None)

    assert zones.shape == (depth.shape[0], depth.shape[1], 4)
    assert zones.dtype == np.float32

    zone_sum = zones.sum(axis=2)
    assert np.allclose(zone_sum, 1.0, atol=1e-3)


def test_saturated_far_field_not_invalid_by_default() -> None:
    # Construct depth with a large saturated far region at 1.0
    cfg = DepthZoneConfig(apply_sky_heuristic=False)
    gen = DepthZoneGenerator(cfg)

    depth = _make_depth()
    depth[:, depth.shape[1] // 2 :] = 1.0  # 50% saturated far

    zones, stats = gen.generate_zones(depth=depth, image=None)

    # After fix: far saturation should remain valid unless sky heuristic enabled
    dc = stats["depth_convention"]
    assert dc["valid_coverage_pct"] >= 99.0


def test_inverted_depth_convention_is_detected_but_not_rejected_when_monotonic() -> None:
    cfg = DepthZoneConfig(depth_convention="auto")
    gen = DepthZoneGenerator(cfg)

    depth = _make_depth()
    depth = 1.0 - depth  # inverted but monotonic

    zones, stats = gen.generate_zones(depth=depth, image=None)

    dc = stats["depth_convention"]
    assert dc["override"] is False
    assert dc["valid_coverage_pct"] > 95.0
    assert dc["detected"] in ("near_to_far_increasing", "near_to_far_decreasing")


def test_sky_heuristic_is_noop_when_disabled() -> None:
    cfg = DepthZoneConfig(apply_sky_heuristic=False)
    gen = DepthZoneGenerator(cfg)

    depth = _make_depth()
    image = np.ones((depth.shape[0], depth.shape[1], 3), dtype=np.float32)  # bright everywhere

    zones_a, _ = gen.generate_zones(depth=depth, image=None)
    zones_b, _ = gen.generate_zones(depth=depth, image=image)

    # Since apply_sky_heuristic=False, passing an image should not change zones
    assert np.allclose(zones_a, zones_b, atol=1e-4)
