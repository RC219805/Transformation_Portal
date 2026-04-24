"""Tests for segmentation tiling configuration parsing/validation."""

import pytest

from transformation_portal.spatial_ai.segmentation.tiling.config import SegmentationTilingConfig

pytestmark = pytest.mark.unit


def test_from_dict_defaults_disabled_when_missing():
    cfg = SegmentationTilingConfig.from_dict(None)
    assert cfg.enabled is False
    assert cfg.apply_to_modes == ("auto",)


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


def test_from_dict_requires_explicit_enable_opt_in():
    cfg = SegmentationTilingConfig.from_dict({"tile_size_px": 512, "overlap_px": 64})
    assert cfg.enabled is False


def test_from_dict_rejects_prompt_tiling_until_roi_tiling_exists():
    with pytest.raises(ValueError, match="supports only auto mode"):
        SegmentationTilingConfig.from_dict({"enabled": True, "apply_to_modes": ["auto", "points"]})


def test_from_dict_rejects_content_adaptive_policy_until_implemented():
    with pytest.raises(ValueError, match="policy='content_adaptive'.*use 'uniform'"):
        SegmentationTilingConfig.from_dict({"enabled": True, "policy": "content_adaptive"})


def test_from_dict_rejects_parallel_tiling_until_implemented():
    with pytest.raises(ValueError, match="max_concurrency must be 1"):
        SegmentationTilingConfig.from_dict({"enabled": True, "max_concurrency": 2})


def test_from_dict_rejects_unsupported_soft_merge_options():
    with pytest.raises(ValueError, match="merge.mode='weighted_soft'.*use 'binary_union'"):
        SegmentationTilingConfig.from_dict({"enabled": True, "merge": {"mode": "weighted_soft"}})
    with pytest.raises(ValueError, match="embedding_cosine_threshold is not supported"):
        SegmentationTilingConfig.from_dict({"enabled": True, "merge": {"instance_merge": {"embedding_cosine_threshold": 0.8}}})
