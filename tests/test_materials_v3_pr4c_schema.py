"""Unit tests for PR-4C schema separation (refinement vs pixel ops)."""

import numpy as np
import pytest

from lux_depth_v2.materials_v3_response import (
    ResponsePlanConfig,
    compute_edge_signals,
    decide_refinement,
    decide_pixel_ops,
    generate_response_plan,
)


def test_response_plan_schema_v31_structure():
    """Verify v3.1 schema has refinement + pixel_ops + edge_signals blocks."""
    # Create synthetic glass mask
    mask = np.zeros((256, 256), dtype=np.float32)
    mask[64:192, 64:192] = 0.85  # Center region
    
    # Create synthetic RGB image
    rgb = np.ones((256, 256, 3), dtype=np.float32) * 0.5
    
    config = ResponsePlanConfig()
    plan = generate_response_plan(
        canonical_materials={"glass": mask},
        config=config,
        strategy="canary",
        rgb_image=rgb,
    )
    
    # Schema checks
    assert plan["version"] == "v3.1"
    assert "summary" in plan
    assert "per_class" in plan
    
    glass = plan["per_class"]["glass"]
    
    # PR-4C: Three independent blocks
    assert "refinement" in glass
    assert "pixel_ops" in glass
    assert "edge_signals" in glass
    
    # Refinement block
    assert "eligible" in glass["refinement"]
    assert "should_refine_edges" in glass["refinement"]
    assert "reason" in glass["refinement"]
    assert "strategy" in glass["refinement"]
    
    # Pixel ops block
    assert "eligible" in glass["pixel_ops"]
    assert "should_apply" in glass["pixel_ops"]
    assert "reason" in glass["pixel_ops"]
    assert "recommended_ops" in glass["pixel_ops"]
    
    # Edge signals block
    assert "boundary_pixels" in glass["edge_signals"]
    assert "edge_alignment" in glass["edge_signals"]
    
    # Backward compat (deprecated)
    assert "should_refine" in glass
    assert glass["should_refine"] == glass["refinement"]["should_refine_edges"]


def test_glass_pixel_ops_should_apply_flips_on_thresholds():
    """Pixel ops decision should flip based on mean_conf/edge_conf."""
    config = ResponsePlanConfig()
    
    # High confidence → skip
    stats_high = {
        "coverage_px": 10000,
        "mean_conf": 0.85,
        "edge_conf": 0.75,
    }
    eligible, should_apply, reason, ops = decide_pixel_ops("glass", stats_high, config)
    assert eligible is True
    assert should_apply is False
    assert reason == "confidence_already_high"
    
    # Low mean conf → apply
    stats_low_mean = {
        "coverage_px": 10000,
        "mean_conf": 0.75,
        "edge_conf": 0.75,
    }
    eligible, should_apply, reason, ops = decide_pixel_ops("glass", stats_low_mean, config)
    assert eligible is True
    assert should_apply is True
    assert reason == "low_mean_confidence"
    
    # Low edge conf → apply
    stats_low_edge = {
        "coverage_px": 10000,
        "mean_conf": 0.85,
        "edge_conf": 0.50,
    }
    eligible, should_apply, reason, ops = decide_pixel_ops("glass", stats_low_edge, config)
    assert eligible is True
    assert should_apply is True
    assert reason == "low_edge_confidence"
    
    # Below coverage → not eligible
    stats_low_cov = {
        "coverage_px": 500,
        "mean_conf": 0.70,
        "edge_conf": 0.50,
    }
    eligible, should_apply, reason, ops = decide_pixel_ops("glass", stats_low_cov, config)
    assert eligible is False
    assert should_apply is False
    assert reason == "below_coverage_threshold"


def test_edge_signals_boundary_pixels_guard():
    """Edge signals should guard against degenerate boundaries."""
    # Small boundary → edge_alignment = 0.0
    mask_small = np.zeros((256, 256), dtype=np.float32)
    mask_small[120:136, 120:136] = 0.9  # Tiny region
    
    edge_band_small = np.zeros((256, 256), dtype=bool)
    edge_band_small[119:137, 119:137] = True
    edge_band_small[120:136, 120:136] = False  # ~100 boundary pixels
    
    rgb = np.ones((256, 256, 3), dtype=np.float32) * 0.5
    
    signals = compute_edge_signals(mask_small, rgb, edge_band_small)
    
    assert signals["boundary_pixels"] < 250
    assert signals["edge_alignment"] == 0.0
    assert "boundary_too_small" in signals["notes"]
    
    # Larger boundary → edge_alignment computed
    mask_large = np.zeros((256, 256), dtype=np.float32)
    mask_large[64:192, 64:192] = 0.85
    
    edge_band_large = np.zeros((256, 256), dtype=bool)
    edge_band_large[59:197, 59:197] = True
    edge_band_large[64:192, 64:192] = False  # ~1000 boundary pixels
    
    signals_large = compute_edge_signals(mask_large, rgb, edge_band_large)
    
    assert signals_large["boundary_pixels"] >= 250
    assert signals_large["edge_alignment"] >= 0.0
    assert len(signals_large["notes"]) == 0


def test_refinement_decision_requires_edge_signals():
    """Refinement should require edge_alignment >= 0.10 and boundary >= 250."""
    config = ResponsePlanConfig()
    
    # Use mean_conf below ambiguity threshold (default 0.5)
    stats = {
        "coverage_px": 10000,
        "mean_conf": 0.45,  # Ambiguous (below 0.5 threshold)
        "edge_conf": 0.40,
    }
    
    # Weak edge alignment → skip
    edge_signals_weak = {
        "boundary_pixels": 300,
        "edge_alignment": 0.05,  # Below 0.10
        "notes": [],
    }
    eligible, should_refine, reason = decide_refinement(
        "glass", stats, edge_signals_weak, config, "canary"
    )
    assert eligible is True
    assert should_refine is False
    assert reason == "weak_edge_alignment"
    
    # Small boundary → skip
    edge_signals_small = {
        "boundary_pixels": 200,  # Below 250
        "edge_alignment": 0.25,
        "notes": ["boundary_too_small"],
    }
    eligible, should_refine, reason = decide_refinement(
        "glass", stats, edge_signals_small, config, "canary"
    )
    assert eligible is True
    assert should_refine is False
    assert reason == "boundary_too_small"
    
    # Strong edge signals → refine
    edge_signals_strong = {
        "boundary_pixels": 500,
        "edge_alignment": 0.30,
        "notes": [],
    }
    eligible, should_refine, reason = decide_refinement(
        "glass", stats, edge_signals_strong, config, "canary"
    )
    assert eligible is True
    assert should_refine is True
    assert reason == "canary_eligible"


def test_other_materials_get_recommended_ops():
    """Non-glass materials should get recommended_ops but no implementation."""
    config = ResponsePlanConfig()
    
    stats = {"coverage_px": 10000, "mean_conf": 0.75, "edge_conf": 0.65}
    
    # Wood
    eligible, should_apply, reason, ops = decide_pixel_ops("wood", stats, config)
    assert eligible is False
    assert should_apply is False
    assert reason == "no_implementation"
    assert "microcontrast" in ops
    
    # Metal
    eligible, should_apply, reason, ops = decide_pixel_ops("metal", stats, config)
    assert eligible is False
    assert "microcontrast" in ops or "highlight_boost" in ops
    
    # Foliage
    eligible, should_apply, reason, ops = decide_pixel_ops("foliage", stats, config)
    assert "color_pop" in ops or "edge_clarity" in ops


def test_summary_tracks_eligible_classes():
    """Summary should track eligible_for_pixel_ops and eligible_for_refinement."""
    # Create masks for glass (eligible) and wood (not eligible)
    glass_mask = np.zeros((256, 256), dtype=np.float32)
    glass_mask[64:192, 64:192] = 0.75  # Low conf → should apply
    
    wood_mask = np.zeros((256, 256), dtype=np.float32)
    wood_mask[50:200, 50:200] = 0.85
    
    rgb = np.ones((256, 256, 3), dtype=np.float32) * 0.5
    
    config = ResponsePlanConfig()
    plan = generate_response_plan(
        canonical_materials={"glass": glass_mask, "wood": wood_mask},
        config=config,
        strategy="canary",
        rgb_image=rgb,
    )
    
    assert "glass" in plan["summary"]["present_classes"]
    assert "wood" in plan["summary"]["present_classes"]
    
    # Glass eligible for pixel ops (has implementation)
    assert "glass" in plan["summary"]["eligible_for_pixel_ops"]
    assert "wood" not in plan["summary"]["eligible_for_pixel_ops"]
    
    # Glass eligible for refinement (in canary set)
    assert "glass" in plan["summary"]["eligible_for_refinement"]
    # Wood not in canary set
    assert "wood" not in plan["summary"]["eligible_for_refinement"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
