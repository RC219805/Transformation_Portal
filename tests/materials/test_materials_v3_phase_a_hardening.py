"""Tests for Materials V3 Phase A: Harden Pixel Ops Executor.

Test coverage for all 5 Phase A items:
- A1: 3D mask bug fix (_canonical_mask)
- A2: Feathering edge clipping fix (bbox padding)
- A3: Configurable feathering (per-material sigma)
- A4: Normalized input contract (uint16 support)
- A5: Overlap resolution (priority-based)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pytest

from transformation_portal.lux_depth_v3.pixel_ops_executor import (
    _canonical_mask,
    _expand_bbox_with_padding,
    _feather_mask,
    _resolve_overlaps,
    apply_pixel_ops,
)

pytestmark = pytest.mark.unit


@dataclass
class PhaseATestConfig:
    """Test configuration with feathering settings."""

    apply_pixel_ops: bool = True
    glass_response_enabled: bool = True
    water_response_enabled: bool = True
    min_coverage_px: int = 500
    min_mean_conf: float = 0.2
    refinement_strategy: str = "canary"

    # Feathering config (A3)
    mask_feather_sigma_default: float = 3.0
    mask_feather_sigma_overrides: Dict[str, float] = field(default_factory=dict)
    mask_feather_disabled_materials: list = field(default_factory=list)

    # Phase A tests verify baseline feather-sigma plumbing with synthetic flat
    # fixtures, so they must opt out of the low-texture seam-safe guard
    # (which would otherwise widen the feather on flat-plus-large fixtures).
    # Setting the bbox fraction gate above 1.0 disables the guard for any ROI.
    pixel_ops_low_grad_threshold: float = 0.01
    pixel_ops_low_tex_min_bbox_frac: float = 2.0
    pixel_ops_low_tex_feather_multiplier: float = 8.0
    pixel_ops_low_tex_delta_ceiling: float = 0.04


# =============================================================================
# A1: Fix 3D Mask Bug - _canonical_mask() tests
# =============================================================================


def test_canonical_mask_handles_2d():
    """Test A1: _canonical_mask handles (H, W) correctly."""
    mask = np.ones((32, 32), dtype=np.float32)
    result = _canonical_mask(mask)

    assert result.shape == (32, 32)
    assert result.dtype == np.float32
    assert np.allclose(result, mask)


def test_canonical_mask_handles_hwc1():
    """Test A1: _canonical_mask handles (H, W, 1) correctly."""
    mask = np.ones((32, 32, 1), dtype=np.uint8)
    result = _canonical_mask(mask)

    assert result.shape == (32, 32)
    assert result.dtype == np.float32
    assert np.allclose(result, 1.0)


def test_canonical_mask_handles_1hw():
    """Test A1: _canonical_mask handles (1, H, W) correctly."""
    mask = np.ones((1, 32, 32), dtype=np.float64)
    result = _canonical_mask(mask)

    assert result.shape == (32, 32)
    assert result.dtype == np.float32
    assert np.allclose(result, 1.0)


def test_canonical_mask_rejects_invalid_3d():
    """Test A1: _canonical_mask raises on invalid 3D shapes."""
    mask = np.ones((32, 32, 3), dtype=np.float32)  # RGB-like, not squeezable

    with pytest.raises(ValueError, match="Cannot canonicalize 3D mask"):
        _canonical_mask(mask)


def test_canonical_mask_rejects_4d():
    """Test A1: _canonical_mask raises on 4D arrays."""
    mask = np.ones((1, 32, 32, 1), dtype=np.float32)

    with pytest.raises(ValueError, match="Cannot canonicalize mask with 4 dimensions"):
        _canonical_mask(mask)


# =============================================================================
# A2: Fix Feathering Edge Clipping - bbox padding tests
# =============================================================================


def test_expand_bbox_with_padding_interior():
    """Test A2: Bbox expansion for interior regions (no edge clipping)."""
    bbox = (10, 10, 20, 20)
    pad = 5

    expanded = _expand_bbox_with_padding(bbox, pad, img_height=100, img_width=100)

    assert expanded == (5, 5, 25, 25)


def test_expand_bbox_at_image_edge():
    """Test A2: Bbox expansion clips at image boundaries."""
    bbox = (0, 0, 10, 10)
    pad = 5

    expanded = _expand_bbox_with_padding(bbox, pad, img_height=50, img_width=50)

    # Should clip at 0, but expand to right/bottom
    assert expanded == (0, 0, 15, 15)


def test_expand_bbox_at_right_bottom_edge():
    """Test A2: Bbox expansion clips at right/bottom edges."""
    bbox = (90, 90, 100, 100)
    pad = 8

    expanded = _expand_bbox_with_padding(bbox, pad, img_height=100, img_width=100)

    # Should clip at image boundaries
    assert expanded == (82, 82, 100, 100)


def test_feather_mask_no_blur():
    """Test A2: Feathering with sigma=0 returns unchanged mask."""
    mask = np.zeros((32, 32), dtype=np.float32)
    mask[8:24, 8:24] = 1.0

    result = _feather_mask(mask, sigma=0.0)

    assert np.allclose(result, mask)


def test_feather_mask_applies_blur():
    """Test A2: Feathering with sigma>0 smooths edges."""
    mask = np.zeros((32, 32), dtype=np.float32)
    mask[8:24, 8:24] = 1.0

    result = _feather_mask(mask, sigma=2.0)

    # Should smooth edges - check that edge pixels are between 0 and 1
    assert 0.0 < result[7, 7] < 1.0  # Just outside the box
    assert result[16, 16] > 0.9  # Center should remain high


# =============================================================================
# A3: Configurable Feathering - material-specific sigma tests
# =============================================================================


def test_feathering_uses_default_sigma():
    """Test A3: Default feathering sigma is applied."""
    config = PhaseATestConfig(mask_feather_sigma_default=5.0)
    image = np.ones((64, 64, 3), dtype=np.uint8) * 100

    mask = np.zeros((64, 64), dtype=np.float32)
    mask[16:48, 16:48] = 1.0

    segmentation_result = {"materials": {"glass": mask}}
    response_plan = {"per_class": {"glass": {"coverage_px": 1024, "mean_conf": 0.7}}}

    _, telemetry = apply_pixel_ops(image, segmentation_result, response_plan, config)

    # Check that feathering was applied with correct sigma
    if telemetry["applied"]:
        assert telemetry["applied"][0]["feather_sigma"] == 5.0


def test_feathering_uses_material_override():
    """Test A3: Material-specific sigma overrides default."""
    config = PhaseATestConfig(mask_feather_sigma_default=3.0, mask_feather_sigma_overrides={"glass": 7.0})
    image = np.ones((64, 64, 3), dtype=np.uint8) * 100

    mask = np.zeros((64, 64), dtype=np.float32)
    mask[16:48, 16:48] = 1.0

    segmentation_result = {"materials": {"glass": mask}}
    response_plan = {"per_class": {"glass": {"coverage_px": 1024, "mean_conf": 0.7}}}

    _, telemetry = apply_pixel_ops(image, segmentation_result, response_plan, config)

    # Check that override was used
    if telemetry["applied"]:
        assert telemetry["applied"][0]["feather_sigma"] == 7.0


def test_feathering_disabled_for_material():
    """Test A3: Feathering can be disabled for specific materials."""
    config = PhaseATestConfig(mask_feather_sigma_default=3.0, mask_feather_disabled_materials=["glass"])
    image = np.ones((64, 64, 3), dtype=np.uint8) * 100

    mask = np.zeros((64, 64), dtype=np.float32)
    mask[16:48, 16:48] = 1.0

    segmentation_result = {"materials": {"glass": mask}}
    response_plan = {"per_class": {"glass": {"coverage_px": 1024, "mean_conf": 0.7}}}

    _, telemetry = apply_pixel_ops(image, segmentation_result, response_plan, config)

    # Check that feathering was disabled (sigma=0)
    if telemetry["applied"]:
        assert telemetry["applied"][0]["feather_sigma"] == 0.0


# =============================================================================
# A4: Normalized Input Contract - uint16 support test
# =============================================================================


def test_pixel_ops_supports_uint16():
    """Test A4: Pixel ops correctly handle uint16 images."""
    config = PhaseATestConfig()

    # Create uint16 image
    image = np.ones((64, 64, 3), dtype=np.uint16) * 30000

    mask = np.zeros((64, 64), dtype=np.float32)
    mask[16:48, 16:48] = 1.0

    segmentation_result = {"materials": {"glass": mask}}
    response_plan = {"per_class": {"glass": {"coverage_px": 1024, "mean_conf": 0.7}}}

    output, telemetry = apply_pixel_ops(image, segmentation_result, response_plan, config)

    # Verify output is uint16
    assert output.dtype == np.uint16
    assert output.shape == image.shape

    # Verify processing occurred
    assert len(telemetry["applied"]) > 0


# =============================================================================
# A5: Overlap Resolution - priority-based tests
# =============================================================================


def test_resolve_overlaps_sky_and_water():
    """Test A5: Overlapping sky+water resolved by priority."""
    from transformation_portal.lux_depth_v3.materials_v3_taxonomy import DEFAULT_MATERIAL_METADATA

    # Create overlapping masks
    sky_mask = np.zeros((64, 64), dtype=np.float32)
    sky_mask[0:32, :] = 1.0  # Top half

    water_mask = np.zeros((64, 64), dtype=np.float32)
    water_mask[24:40, :] = 1.0  # Overlaps with sky

    materials = {"sky": sky_mask, "water": water_mask}

    # Add sky to metadata if missing
    metadata = DEFAULT_MATERIAL_METADATA.copy()
    if "sky" not in metadata:
        metadata["sky"] = {"priority": 8, "threshold": 0.40}

    resolved, telemetry = _resolve_overlaps(materials, metadata, (64, 64))

    # Check that overlap was detected
    assert telemetry["overlap_percent"] > 0.0
    assert telemetry["overlapping_pixels"] > 0

    # Check that higher priority material kept overlap region
    # Water has priority 9, sky has priority 8 (if added)
    water_priority = metadata["water"]["priority"]
    sky_priority = metadata.get("sky", {}).get("priority", 0)

    if water_priority > sky_priority:
        # Water should have won the overlap region
        assert telemetry["reassignments"].get("sky", 0) > 0
    else:
        # Sky should have won
        assert telemetry["reassignments"].get("water", 0) > 0


def test_resolve_overlaps_priority_ordering():
    """Test A5: Materials are processed in priority order."""
    from transformation_portal.lux_depth_v3.materials_v3_taxonomy import DEFAULT_MATERIAL_METADATA

    # Create three overlapping masks
    glass_mask = np.zeros((64, 64), dtype=np.float32)
    glass_mask[16:48, 16:48] = 1.0

    water_mask = np.zeros((64, 64), dtype=np.float32)
    water_mask[20:44, 20:44] = 1.0

    foliage_mask = np.zeros((64, 64), dtype=np.float32)
    foliage_mask[24:40, 24:40] = 1.0

    materials = {"glass": glass_mask, "water": water_mask, "foliage": foliage_mask}

    resolved, telemetry = _resolve_overlaps(materials, DEFAULT_MATERIAL_METADATA, (64, 64))

    # Verify overlap detected
    assert telemetry["overlap_percent"] > 0.0

    # Verify priorities: glass(10) > water(9) > foliage(5)
    # Glass should have won all overlaps
    assert "glass" not in telemetry["reassignments"]  # Glass loses nothing
    assert telemetry["reassignments"].get("water", 0) > 0
    assert telemetry["reassignments"].get("foliage", 0) > 0


def test_resolve_overlaps_telemetry():
    """Test A5: Overlap resolution emits correct telemetry."""
    metadata = {"mat_a": {"priority": 10}, "mat_b": {"priority": 5}}

    mask_a = np.zeros((32, 32), dtype=np.float32)
    mask_a[8:24, 8:24] = 1.0

    mask_b = np.zeros((32, 32), dtype=np.float32)
    mask_b[16:28, 16:28] = 1.0  # Overlaps with mat_a

    materials = {"mat_a": mask_a, "mat_b": mask_b}

    resolved, telemetry = _resolve_overlaps(materials, metadata, (32, 32))

    # Check telemetry fields
    assert "overlap_percent" in telemetry
    assert "reassignments" in telemetry
    assert "total_pixels" in telemetry
    assert "overlapping_pixels" in telemetry

    # mat_a (priority 10) should win over mat_b (priority 5)
    assert "mat_b" in telemetry["reassignments"]
    assert telemetry["reassignments"]["mat_b"] > 0


def test_resolve_overlaps_no_overlap():
    """Test A5: Non-overlapping materials have zero reassignments."""
    metadata = {"mat_a": {"priority": 10}, "mat_b": {"priority": 5}}

    mask_a = np.zeros((32, 32), dtype=np.float32)
    mask_a[0:16, 0:16] = 1.0

    mask_b = np.zeros((32, 32), dtype=np.float32)
    mask_b[16:32, 16:32] = 1.0  # No overlap

    materials = {"mat_a": mask_a, "mat_b": mask_b}

    resolved, telemetry = _resolve_overlaps(materials, metadata, (32, 32))

    # Should have no overlap
    assert telemetry["overlap_percent"] == 0.0
    assert telemetry["overlapping_pixels"] == 0
    assert len(telemetry["reassignments"]) == 0


def test_apply_pixel_ops_emits_overlap_telemetry():
    """Test A5: apply_pixel_ops includes overlap telemetry."""
    config = PhaseATestConfig()
    image = np.ones((64, 64, 3), dtype=np.uint8) * 100

    # Create overlapping masks
    glass_mask = np.zeros((64, 64), dtype=np.float32)
    glass_mask[16:48, 16:48] = 1.0

    water_mask = np.zeros((64, 64), dtype=np.float32)
    water_mask[32:56, 32:56] = 1.0

    segmentation_result = {"materials": {"glass": glass_mask, "water": water_mask}}
    response_plan = {
        "per_class": {"glass": {"coverage_px": 1024, "mean_conf": 0.7}, "water": {"coverage_px": 576, "mean_conf": 0.6}}
    }

    _, telemetry = apply_pixel_ops(image, segmentation_result, response_plan, config)

    # Check overlap telemetry is present
    assert "overlap_resolution" in telemetry
    assert "overlap_percent" in telemetry["overlap_resolution"]
    assert "reassignments" in telemetry["overlap_resolution"]


# =============================================================================
# Integration test: All Phase A features together
# =============================================================================


def test_phase_a_integration():
    """Integration test: All Phase A features work together."""
    config = PhaseATestConfig(
        mask_feather_sigma_default=2.0, mask_feather_sigma_overrides={"water": 4.0}, mask_feather_disabled_materials=[]
    )

    # uint16 image (A4)
    image = np.ones((128, 128, 3), dtype=np.uint16) * 40000

    # 3D masks (A1)
    glass_mask = np.zeros((128, 128, 1), dtype=np.float32)
    glass_mask[20:60, 20:60, 0] = 1.0

    # Edge-touching mask (A2)
    water_mask = np.zeros((128, 128), dtype=np.float32)
    water_mask[50:100, 0:128] = 1.0  # Touches left and right edges

    segmentation_result = {"materials": {"glass": glass_mask, "water": water_mask}}

    response_plan = {
        "per_class": {
            "glass": {"coverage_px": 1600, "mean_conf": 0.8},
            "water": {"coverage_px": 6400, "mean_conf": 0.7},
        }
    }

    output, telemetry = apply_pixel_ops(image, segmentation_result, response_plan, config)

    # A4: Check uint16 preserved
    assert output.dtype == np.uint16

    # A5: Check overlap resolution occurred
    assert "overlap_resolution" in telemetry

    # A3: Check feathering config applied
    applied_materials = {item["material"]: item for item in telemetry["applied"]}
    if "water" in applied_materials:
        assert applied_materials["water"]["feather_sigma"] == 4.0
    if "glass" in applied_materials:
        assert applied_materials["glass"]["feather_sigma"] == 2.0

    # A2: Check bbox padding was computed
    for item in telemetry["applied"]:
        assert "bbox_padding" in item
        assert item["bbox_padding"] >= 0
