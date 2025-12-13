# lux_depth_v2/tests/test_fusion_integration.py
"""
Integration tests for Material Segmentation V3 fusion.

Tests the complete fusion pipeline:
- Base segmenter + refinement provider + fusion logic
- Graceful fallback on provider failure
- IoU gating behavior
- Stats emission
"""

import numpy as np
import pytest

pytest.importorskip("torch")

from lux_depth_v2 import torch_ops
from lux_depth_v2.material_segmentation import FusedMaterialSegmenter, MaterialSegmenter
from lux_depth_v2.backends.refinement_provider import MockRefinementProvider
from lux_depth_v2.segmentation_fusion import FusionMode
from dataclasses import dataclass
from typing import Dict


@dataclass
class MockConfig:
    """Mock segmentation config for testing."""
    fusion_mode: FusionMode = FusionMode.CONFIDENCE_WEIGHTED
    fusion_min_iou: float = 0.30
    fusion_core_thresh: float = 0.70
    fusion_edge_low: float = 0.20
    fusion_edge_high: float = 0.70
    fusion_alpha_edge: float = 0.70
    fusion_alpha_core: float = 0.30


class SimpleMockSegmenter(MaterialSegmenter):
    """Simple mock base segmenter that returns a square mask."""

    def predict(self, rgb: torch_ops.torch.Tensor) -> Dict[str, torch_ops.torch.Tensor]:
        torch_ops.require_torch()
        b, c, h, w = rgb.shape
        mask = torch_ops.torch.zeros(1, 1, h, w, dtype=torch_ops.torch.float32, device=rgb.device)
        # Create a centered square mask for "glass"
        h_start, h_end = h // 4, 3 * h // 4
        w_start, w_end = w // 4, 3 * w // 4
        mask[0, 0, h_start:h_end, w_start:w_end] = 0.9
        # Add edge band
        mask[0, 0, h_start - 2 : h_end + 2, w_start - 2 : w_end + 2] = torch_ops.torch.maximum(
            mask[0, 0, h_start - 2 : h_end + 2, w_start - 2 : w_end + 2],
            torch_ops.torch.tensor(0.4, dtype=torch_ops.torch.float32),
        )
        return {
            "glass": mask.clone(),
            "wood": torch_ops.torch.zeros_like(mask),  # Not in refinement set
        }


def test_fusion_integration_with_mock_provider():
    """Test that fusion runs end-to-end with mock provider."""
    torch_ops.require_torch()

    device = torch_ops.torch.device("cpu")
    cfg = MockConfig(fusion_mode=FusionMode.CONFIDENCE_WEIGHTED)
    base_segmenter = SimpleMockSegmenter()
    provider = MockRefinementProvider(mode="dilate")

    fused_segmenter = FusedMaterialSegmenter(base_segmenter, cfg, device, provider)

    # Create dummy RGB input
    rgb = torch_ops.torch.rand(1, 3, 64, 64, device=device)

    # Run prediction
    masks = fused_segmenter.predict(rgb)

    # Should have both classes
    assert "glass" in masks
    assert "wood" in masks

    # Glass should be refined (in EDGE_REFINEMENT_CLASSES)
    # Wood should be unchanged (not in refinement set)
    assert "glass" in fused_segmenter.fusion_stats
    assert fused_segmenter.fusion_stats["glass"]["fusion_applied"] == 1.0

    # Wood should not have fusion stats (not refined)
    assert "wood" not in fused_segmenter.fusion_stats


def test_fusion_fallback_when_provider_returns_none():
    """Test graceful fallback when refinement provider returns None."""
    torch_ops.require_torch()

    device = torch_ops.torch.device("cpu")
    cfg = MockConfig(fusion_mode=FusionMode.CONFIDENCE_WEIGHTED)
    base_segmenter = SimpleMockSegmenter()
    provider = MockRefinementProvider(mode="none")  # Always returns None

    fused_segmenter = FusedMaterialSegmenter(base_segmenter, cfg, device, provider)

    rgb = torch_ops.torch.rand(1, 3, 64, 64, device=device)
    masks = fused_segmenter.predict(rgb)

    # Should still get masks (fallback to base)
    assert "glass" in masks

    # Fusion should not be applied
    assert "glass" in fused_segmenter.fusion_stats
    assert fused_segmenter.fusion_stats["glass"]["fusion_applied"] == 0.0


def test_fusion_respects_iou_gating():
    """Test that fusion is skipped when IoU is too low."""
    torch_ops.require_torch()

    device = torch_ops.torch.device("cpu")

    class DisjointProvider(MockRefinementProvider):
        """Returns a mask that doesn't overlap with the base."""

        def get_refined_mask(self, rgb, base_mask, material_class):
            torch_ops.require_torch()
            # Return a mask in a completely different region
            h, w = base_mask.shape[2], base_mask.shape[3]
            refined = torch_ops.torch.zeros_like(base_mask)
            refined[0, 0, 0:h // 8, 0:w // 8] = 1.0  # Top-left corner only
            return refined

    cfg = MockConfig(
        fusion_mode=FusionMode.CONFIDENCE_WEIGHTED,
        fusion_min_iou=0.5,  # High threshold
    )
    base_segmenter = SimpleMockSegmenter()
    provider = DisjointProvider()

    fused_segmenter = FusedMaterialSegmenter(base_segmenter, cfg, device, provider)

    rgb = torch_ops.torch.rand(1, 3, 64, 64, device=device)
    masks = fused_segmenter.predict(rgb)

    # Fusion should be rejected due to low IoU
    assert "glass" in fused_segmenter.fusion_stats
    assert fused_segmenter.fusion_stats["glass"]["iou_base_vs_refined"] < 0.5
    assert fused_segmenter.fusion_stats["glass"]["fusion_applied"] == 0.0


def test_fusion_disabled_when_mode_is_none():
    """Test that fusion is skipped when fusion_mode is NONE."""
    torch_ops.require_torch()

    device = torch_ops.torch.device("cpu")
    cfg = MockConfig(fusion_mode=FusionMode.NONE)
    base_segmenter = SimpleMockSegmenter()
    provider = MockRefinementProvider(mode="dilate")

    fused_segmenter = FusedMaterialSegmenter(base_segmenter, cfg, device, provider)

    rgb = torch_ops.torch.rand(1, 3, 64, 64, device=device)
    masks = fused_segmenter.predict(rgb)

    # Should get masks but no fusion stats
    assert "glass" in masks
    assert len(fused_segmenter.fusion_stats) == 0


def test_fusion_disabled_when_provider_is_none():
    """Test that fusion is skipped when no provider is given."""
    torch_ops.require_torch()

    device = torch_ops.torch.device("cpu")
    cfg = MockConfig(fusion_mode=FusionMode.CONFIDENCE_WEIGHTED)
    base_segmenter = SimpleMockSegmenter()

    fused_segmenter = FusedMaterialSegmenter(
        base_segmenter, cfg, device, refinement_provider=None
    )

    rgb = torch_ops.torch.rand(1, 3, 64, 64, device=device)
    masks = fused_segmenter.predict(rgb)

    # Should get masks but no fusion
    assert "glass" in masks
    assert len(fused_segmenter.fusion_stats) == 0


def test_only_edge_classes_are_refined():
    """Test that only classes in EDGE_REFINEMENT_CLASSES are refined."""
    torch_ops.require_torch()

    from lux_depth_v2.material_segmentation import EDGE_REFINEMENT_CLASSES

    device = torch_ops.torch.device("cpu")
    cfg = MockConfig(fusion_mode=FusionMode.CONFIDENCE_WEIGHTED)

    class MultiClassSegmenter(MaterialSegmenter):
        def predict(self, rgb):
            torch_ops.require_torch()
            h, w = rgb.shape[2], rgb.shape[3]
            base = torch_ops.torch.ones(1, 1, h, w, dtype=torch_ops.torch.float32) * 0.8
            return {
                "glass": base.clone(),  # In EDGE_REFINEMENT_CLASSES
                "water": base.clone(),  # In EDGE_REFINEMENT_CLASSES
                "wood": base.clone(),  # NOT in EDGE_REFINEMENT_CLASSES
                "metal": base.clone(),  # NOT in EDGE_REFINEMENT_CLASSES
            }

    base_segmenter = MultiClassSegmenter()
    provider = MockRefinementProvider(mode="dilate")

    fused_segmenter = FusedMaterialSegmenter(base_segmenter, cfg, device, provider)

    rgb = torch_ops.torch.rand(1, 3, 32, 32, device=device)
    masks = fused_segmenter.predict(rgb)

    # Only edge classes should have fusion stats
    for cls in EDGE_REFINEMENT_CLASSES:
        if cls in masks:
            assert cls in fused_segmenter.fusion_stats

    # Non-edge classes should not have fusion stats
    assert "wood" not in fused_segmenter.fusion_stats
    assert "metal" not in fused_segmenter.fusion_stats
