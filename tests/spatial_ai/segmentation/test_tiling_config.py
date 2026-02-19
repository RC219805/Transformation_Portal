"""Tests for segmentation tiling configuration parsing/validation."""

import pytest

from transformation_portal.spatial_ai.segmentation.tiling.config import SegmentationTilingConfig


def test_from_dict_defaults_disabled_when_missing():
    cfg = SegmentationTilingConfig.from_dict(None)
    assert cfg.enabled is False


def test_from_dict_parses_nested_values():
    cfg = SegmentationTilingConfig.from_dict(
        {
            "enabled": True,
            "tile_size_px": 1024,
            "overlap_px": 128,
            "apply_to_modes": ["auto"],
            "merge": {"mode": "binary_union", "instance_merge": {"iou_threshold": 0.4}},
            "validation": {"enabled": False},
        }
    )
    assert cfg.enabled is True
    assert cfg.tile_size_px == 1024
    assert cfg.overlap_px == 128
    assert cfg.apply_to_modes == ("auto",)
    assert cfg.merge.mode == "binary_union"
    assert cfg.merge.instance_merge.iou_threshold == pytest.approx(0.4)
    assert cfg.validation.enabled is False
