"""Tests for Materials V3 Response Planning (PR-4A)."""

import numpy as np
import pytest

from lux_depth_v2.materials_v3_response import (
    ResponsePlanConfig,
    compute_class_stats,
    compute_response_strengths,
    decide_should_refine,
    extract_edge_band,
    generate_response_plan,
)


def test_extract_edge_band_simple():
    """Test edge band extraction on a simple square mask."""
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:24, 8:24] = True
    
    core, edge = extract_edge_band(mask, edge_width_px=3)
    
    # Core should be smaller than original
    assert core.sum() < mask.sum()
    # Edge should be non-empty
    assert edge.sum() > 0
    # Core + edge should equal original
    assert (core | edge).sum() == mask.sum()


def test_extract_edge_band_float_mask():
    """Test edge band extraction on float confidence mask."""
    mask = np.zeros((32, 32), dtype=np.float32)
    mask[8:24, 8:24] = 0.8
    
    core, edge = extract_edge_band(mask, edge_width_px=2)
    
    assert isinstance(core, np.ndarray)
    assert core.dtype == bool
    assert edge.dtype == bool


def test_compute_class_stats_boolean():
    """Test stats computation on boolean mask."""
    mask = np.zeros((64, 64), dtype=bool)
    mask[16:48, 16:48] = True
    
    stats = compute_class_stats(mask, edge_width_px=3)
    
    assert stats["coverage_px"] == 32 * 32
    assert 0 < stats["coverage"] < 1
    assert stats["mean_conf"] == 1.0
    assert stats["edge_pixels"] > 0
    assert stats["core_pixels"] > 0


def test_compute_class_stats_float():
    """Test stats computation on float confidence mask."""
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[16:48, 16:48] = 0.75
    
    stats = compute_class_stats(mask, edge_width_px=5)
    
    assert stats["coverage_px"] == 32 * 32
    assert 0.74 < stats["mean_conf"] < 0.76
    assert stats["edge_conf"] >= 0
    assert stats["core_conf"] >= 0


def test_compute_class_stats_empty_mask():
    """Test stats computation on empty mask."""
    mask = np.zeros((32, 32), dtype=np.float32)
    
    stats = compute_class_stats(mask)
    
    assert stats["coverage"] == 0.0
    assert stats["coverage_px"] == 0
    assert stats["mean_conf"] == 0.0
    assert stats["edge_pixels"] == 0
    assert stats["core_pixels"] == 0


def test_compute_response_strengths_glass():
    """Test response strength computation for glass (conservative)."""
    config = ResponsePlanConfig()
    stats = {
        "coverage_px": 2000,
        "edge_conf": 0.40,
    }
    
    core_str, edge_str = compute_response_strengths("glass", stats, config)
    
    # Glass should have conservative strengths
    assert core_str <= config.default_core_strength
    assert edge_str < core_str  # Edge more conservative than core


def test_compute_response_strengths_low_coverage():
    """Test attenuation for low coverage."""
    config = ResponsePlanConfig(min_coverage_px=1000)
    stats = {
        "coverage_px": 300,  # Below threshold
        "edge_conf": 0.50,
    }
    
    core_str, edge_str = compute_response_strengths("wood", stats, config)
    
    # Both should be attenuated
    assert core_str < config.default_core_strength
    assert edge_str < config.default_edge_strength


def test_compute_response_strengths_low_edge_conf():
    """Test extra attenuation for very low edge confidence."""
    config = ResponsePlanConfig()
    stats = {
        "coverage_px": 2000,
        "edge_conf": 0.15,  # Very low
    }
    
    core_str, edge_str = compute_response_strengths("wood", stats, config)
    
    # Edge should be extra attenuated
    assert edge_str < 0.8 * config.default_edge_strength


def test_decide_should_refine_strategy_off():
    """Test refinement decision with strategy=off."""
    config = ResponsePlanConfig()
    stats = {"coverage_px": 2000, "mean_conf": 0.40}
    
    should_refine, reason = decide_should_refine("glass", stats, config, strategy="off")
    
    assert should_refine is False
    assert reason == "strategy_off"


def test_decide_should_refine_canary_eligible():
    """Test refinement decision for canary-eligible class."""
    config = ResponsePlanConfig(refine_conf_ambiguity_threshold=0.50)
    stats = {
        "coverage_px": 2000,
        "mean_conf": 0.40,  # Ambiguous
    }
    
    should_refine, reason = decide_should_refine("glass", stats, config, strategy="canary")
    
    assert should_refine is True
    assert reason == "canary_eligible"


def test_decide_should_refine_canary_high_conf():
    """Test refinement decision for canary class with high confidence."""
    config = ResponsePlanConfig(refine_conf_ambiguity_threshold=0.50)
    stats = {
        "coverage_px": 2000,
        "mean_conf": 0.65,  # Already high
    }
    
    should_refine, reason = decide_should_refine("glass", stats, config, strategy="canary")
    
    assert should_refine is False
    assert reason == "confidence_already_high"


def test_decide_should_refine_not_canary_class():
    """Test refinement decision for non-canary class."""
    config = ResponsePlanConfig()
    stats = {
        "coverage_px": 2000,
        "mean_conf": 0.40,
    }
    
    should_refine, reason = decide_should_refine("wood", stats, config, strategy="canary")
    
    assert should_refine is False
    assert reason == "not_in_canary_set"


def test_decide_should_refine_below_coverage():
    """Test refinement decision when coverage too low."""
    config = ResponsePlanConfig(min_coverage_px=1000)
    stats = {
        "coverage_px": 300,
        "mean_conf": 0.50,
    }
    
    should_refine, reason = decide_should_refine("glass", stats, config, strategy="canary")
    
    assert should_refine is False
    assert reason == "below_coverage_threshold"


def test_decide_should_refine_selective_ambiguous():
    """Test selective strategy with ambiguous confidence."""
    config = ResponsePlanConfig(refine_conf_ambiguity_threshold=0.50)
    stats = {
        "coverage_px": 2000,
        "mean_conf": 0.35,  # Ambiguous
    }
    
    should_refine, reason = decide_should_refine("wood", stats, config, strategy="selective")
    
    assert should_refine is True
    assert reason == "selective_ambiguous_confidence"


def test_decide_should_refine_aggressive():
    """Test aggressive strategy."""
    config = ResponsePlanConfig()
    stats = {
        "coverage_px": 2000,
        "mean_conf": 0.80,  # High
    }
    
    should_refine, reason = decide_should_refine("wood", stats, config, strategy="aggressive")
    
    assert should_refine is True
    assert reason == "aggressive_all_classes"


def test_generate_response_plan_simple():
    """Test response plan generation with simple masks."""
    glass_mask = np.zeros((64, 64), dtype=np.float32)
    # Use 0.55 for glass (above 0.5 threshold, below 0.70 for ambiguity, large enough for coverage)
    glass_mask[8:56, 8:56] = 0.55  # 48x48 = 2304 pixels > 500 min_coverage_px
    
    wood_mask = np.zeros((64, 64), dtype=np.float32)
    wood_mask[20:44, 20:44] = 0.70
    
    canonical_materials = {
        "glass": glass_mask,
        "wood": wood_mask,
    }
    
    config = ResponsePlanConfig(
        min_coverage_px=500,
        refine_conf_ambiguity_threshold=0.60,  # Glass at 0.55 is below this
    )
    plan = generate_response_plan(
        canonical_materials,
        config,
        strategy="canary",
        intent="client",
        quality_tier="max",
    )
    
    # Validate structure
    assert plan["enabled"] is True
    assert plan["strategy"] == "canary"
    assert plan["scene"]["intent"] == "client"
    assert plan["scene"]["quality_tier"] == "max"
    
    # Per-class plans
    assert "glass" in plan["per_class"]
    assert "wood" in plan["per_class"]
    
    glass_plan = plan["per_class"]["glass"]
    assert glass_plan["present"] is True
    assert glass_plan["coverage"] > 0  # Should have coverage
    assert glass_plan["mean_conf"] > 0.5  # Should have meaningful confidence
    assert "core_strength" in glass_plan
    assert "edge_strength" in glass_plan
    assert "should_refine" in glass_plan
    assert "refine_reason" in glass_plan
    
    # Glass should be canary-eligible with ambiguous confidence
    assert glass_plan["should_refine"] is True
    assert glass_plan["refine_reason"] == "canary_eligible"
    
    # Wood should not refine (not in canary set)
    wood_plan = plan["per_class"]["wood"]
    assert wood_plan["should_refine"] is False
    assert wood_plan["skip_reason"] == "not_in_canary_set"


def test_generate_response_plan_empty_dict():
    """Test response plan generation with no materials."""
    plan = generate_response_plan(
        {},
        ResponsePlanConfig(),
        strategy="off",
    )
    
    assert plan["enabled"] is True
    assert plan["per_class"] == {}


def test_response_plan_schema_stable():
    """Test that response plan schema matches expected structure."""
    mask = np.ones((32, 32), dtype=np.float32) * 0.5
    
    plan = generate_response_plan(
        {"glass": mask},
        ResponsePlanConfig(),
        strategy="canary",
    )
    
    # Required top-level keys
    required_keys = {"enabled", "taxonomy", "strategy", "scene", "per_class", "notes"}
    assert all(k in plan for k in required_keys)
    
    # Scene keys
    assert "intent" in plan["scene"]
    assert "quality_tier" in plan["scene"]
    
    # Per-class keys
    glass = plan["per_class"]["glass"]
    class_required = {
        "present", "coverage", "coverage_px", "mean_conf",
        "edge_conf", "core_conf", "edge_pixels", "core_pixels",
        "core_strength", "edge_strength", "should_refine",
        "refine_reason", "skip_reason",
    }
    assert all(k in glass for k in class_required)
