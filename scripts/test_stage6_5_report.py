#!/usr/bin/env python3
"""Quick test for Stage 6.5 segmentation_v3 report generation."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_fusion_report_method():
    """Test that FusedMaterialSegmenter has get_segmentation_v3_report."""
    from lux_depth_v2.material_segmentation import FusedMaterialSegmenter, EDGE_REFINEMENT_CLASSES
    from lux_depth_v2.config import SegmentationConfig, SegmentationBackend, FusionMode
    from lux_depth_v2 import torch_ops
    
    # Mock config
    class MockBaseSegmenter:
        def predict(self, rgb):
            return {}
    
    cfg = SegmentationConfig()
    cfg.backend_v3 = SegmentationBackend.FUSED
    cfg.fusion_mode = FusionMode.CONFIDENCE_WEIGHTED
    cfg.efficientSAM_model = "efficientsam_s"
    
    device = torch_ops.pick_device("cpu")
    
    segmenter = FusedMaterialSegmenter(
        base_segmenter=MockBaseSegmenter(),
        cfg=cfg,
        device=device,
        refinement_provider=None,
    )
    
    # Simulate some fusion stats
    segmenter.fusion_stats = {
        "glass": {"iou_base_vs_refined": 0.65, "fusion_applied": 1.0},
        "water": {"iou_base_vs_refined": 0.72, "fusion_applied": 1.0},
        "foliage": {"iou_base_vs_refined": 0.18, "fusion_applied": 0.0},
    }
    
    # Get report
    report = segmenter.get_segmentation_v3_report()
    
    # Validate structure
    assert "backend_v3" in report, "Missing backend_v3"
    assert "fusion_mode" in report, "Missing fusion_mode"
    assert "model" in report, "Missing model"
    assert "refined_classes" in report, "Missing refined_classes"
    assert "per_class" in report, "Missing per_class"
    
    # Validate content
    assert report["backend_v3"] == "SegmentationBackend.FUSED"
    assert report["fusion_mode"] == "FusionMode.CONFIDENCE_WEIGHTED"
    assert report["model"] == "efficientsam_s"
    assert set(report["refined_classes"]) == EDGE_REFINEMENT_CLASSES
    assert report["per_class"] == segmenter.fusion_stats
    
    print("✓ get_segmentation_v3_report() works correctly")
    print(f"  Report structure: {list(report.keys())}")
    print(f"  Refined classes: {report['refined_classes']}")
    print(f"  Per-class stats: {len(report['per_class'])} classes")
    
    return True


if __name__ == "__main__":
    try:
        test_fusion_report_method()
        print("\n✓ Stage 6.5 unit test PASSED")
        raise SystemExit(0)
    except Exception as e:
        print(f"\n✗ Stage 6.5 unit test FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise SystemExit(1)
