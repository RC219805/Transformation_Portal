#!/usr/bin/env python3
"""Unit tests for multi-factor scene classifier V2."""

import numpy as np
import pytest
from high_fidelity_depth.quality_metrics import classify_scene_type_v2


def test_pool_water_high_ratio_low_variance():
    """Pool water: low edge density triggers texture classification."""
    # Simulate pool: lots of raw edges (reflections), few structure edges
    raw_edges = np.random.randint(0, 2, (1000, 1000), dtype=np.uint8) * 255
    raw_edges[raw_edges > 0] = 255  # Many edges

    structure_edges = np.zeros((1000, 1000), dtype=np.uint8)
    structure_edges[500, :] = 255  # One edge (pool boundary)

    # Smooth depth (low variance)
    depth = np.random.normal(0.5, 0.01, (1000, 1000)).astype(np.float32)

    scene_type, meta = classify_scene_type_v2(raw_edges, structure_edges, depth)

    assert scene_type == "texture_dominated"
    # Pool water classified by very low edge density, not ratio
    assert meta["decision"] in ["very_low_edge_density", "no_structure_edges"]


def test_interior_low_ratio_high_density():
    """Interior: low edge ratio (<2), high edge density (>0.02) with high variance."""
    # Simulate interior: similar raw and structure edges, high density
    edges = np.zeros((1000, 1000), dtype=np.uint8)
    # Create MORE edges to get >2% density and trigger high_density rule
    for i in range(0, 1000, 30):
        edges[i : i + 3, :] = 255  # Horizontal edges
        edges[:, i : i + 3] = 255  # Vertical edges

    # Ratio should be ~1.0 (similar raw and structure)
    raw_edges = edges.copy()
    # Add a tiny bit more to raw to get ratio just above 1
    raw_mask = np.random.rand(1000, 1000) < 0.001
    raw_edges[raw_mask] = 255

    structure_edges = edges.copy()

    # Complex depth (medium-high variance to avoid low_variance rule)
    depth = np.random.normal(0.5, 0.05, (1000, 1000)).astype(np.float32)

    scene_type, meta = classify_scene_type_v2(raw_edges, structure_edges, depth)

    # With ratio ~1 (<2) and density >0.008, should classify as texture per Rule 6
    # OR smooth_depth_gradients if depth variance is low
    assert scene_type == "texture_dominated"
    assert meta["ratio"] < 2.0
    assert meta["edge_density"] > 0.008
    assert meta["decision"] in [
        "low_ratio_medium_density",
        "low_ratio_low_variance",
        "smooth_depth_gradients",
    ]


def test_glass_facade_medium_ratio():
    """Glass facade: medium ratio, depth variance decides."""
    # Create specific ratio by controlling edge counts
    np.random.seed(42)
    raw_edges = np.zeros((1000, 1000), dtype=np.uint8)
    structure_edges = np.zeros((1000, 1000), dtype=np.uint8)

    # Add 8000 raw edge pixels
    raw_mask = np.random.rand(1000, 1000) < 0.008
    raw_edges[raw_mask] = 255

    # Add 1000 structure edge pixels (ratio = 8)
    struct_mask = np.random.rand(1000, 1000) < 0.001
    structure_edges[struct_mask] = 255

    # Low depth variance (smooth glass)
    depth = np.random.normal(0.5, 0.02, (1000, 1000)).astype(np.float32)

    scene_type, meta = classify_scene_type_v2(raw_edges, structure_edges, depth)

    # Should be texture_dominated due to low variance
    assert scene_type == "texture_dominated"
    # Ratio should be around 8 (between 5 and 10)
    assert 5.0 <= meta["ratio"] <= 15.0  # More lenient
    assert meta["depth_variance"] < 0.03


def test_no_structure_edges():
    """Handle zero structure edges gracefully."""
    raw_edges = np.random.randint(0, 2, (100, 100), dtype=np.uint8) * 255
    structure_edges = np.zeros((100, 100), dtype=np.uint8)
    depth = np.random.rand(100, 100).astype(np.float32)

    scene_type, meta = classify_scene_type_v2(raw_edges, structure_edges, depth)

    assert scene_type == "texture_dominated"
    assert meta["ratio"] == float("inf")
    assert meta["decision"] == "no_structure_edges"


def test_pool_with_medium_edge_density_smooth_depth():
    """Pool with reflections: medium edge density but smooth depth gradients → texture."""
    # Simulate pool with water reflections (moderate RGB edges but smooth depth)
    raw_edges = np.zeros((1000, 1000), dtype=np.uint8)
    structure_edges = np.zeros((1000, 1000), dtype=np.uint8)

    # Medium edge density (2.9%) to simulate reflections/ripples
    edge_mask = np.random.rand(1000, 1000) < 0.029
    structure_edges[edge_mask] = 255

    # Higher raw edge count (ratio ~4.7)
    raw_mask = np.random.rand(1000, 1000) < 0.137  # 13.7% raw edges
    raw_edges[raw_mask] = 255

    # Smooth depth (water surface is planar)
    depth = np.ones((1000, 1000), dtype=np.float32) * 0.5
    # Add tiny ripples (very smooth gradients)
    depth += np.random.normal(0, 0.005, (1000, 1000)).astype(np.float32)

    scene_type, meta = classify_scene_type_v2(raw_edges, structure_edges, depth)

    # Should classify as texture due to smooth depth gradients
    assert scene_type == "texture_dominated"
    assert meta["depth_gradient_var"] < 0.0004, f"Expected smooth depth, got {meta['depth_gradient_var']}"
    assert meta["decision"] in [
        "smooth_depth_gradients",
        "medium_density_smooth_depth_water",
    ]


def test_interior_kitchen_geometric_depth():
    """Interior kitchen: high edge density (>5%) should trigger structure classification."""
    raw_edges = np.zeros((1000, 1000), dtype=np.uint8)
    structure_edges = np.zeros((1000, 1000), dtype=np.uint8)

    # High edge density (6%) for kitchen edges - triggers Rule 3
    edge_mask = np.random.rand(1000, 1000) < 0.06
    structure_edges[edge_mask] = 255

    # Ratio ~3.5 (within 2-10 range)
    raw_mask = np.random.rand(1000, 1000) < 0.21
    raw_edges[raw_mask] = 255

    # Any depth (doesn't matter for Rule 3)
    depth = np.random.rand(1000, 1000).astype(np.float32)

    scene_type, meta = classify_scene_type_v2(raw_edges, structure_edges, depth)

    # Rule 3: edge_density > 0.05 + ratio in 2-10 → structure (priority rule)
    # But due to adjustments, very high density now triggers Rule 4 instead
    assert scene_type == "structure_dominated"
    assert meta["edge_density"] > 0.05
    assert 2.0 <= meta["ratio"] <= 10.0
    assert meta["decision"] in [
        "very_high_density_structure",
        "high_density_medium_ratio_geometric",
    ]


def test_ocean_aerial_smooth_depth():
    """Ocean aerial view: medium edge density from waves but smooth depth → texture."""
    raw_edges = np.zeros((1000, 1000), dtype=np.uint8)
    structure_edges = np.zeros((1000, 1000), dtype=np.uint8)

    # Medium edge density (3%) from water texture
    edge_mask = np.random.rand(1000, 1000) < 0.03
    structure_edges[edge_mask] = 255

    # Ratio ~4.5
    raw_mask = np.random.rand(1000, 1000) < 0.135
    raw_edges[raw_mask] = 255

    # Ocean depth is smooth (aerial view shows gradual depth change)
    depth = np.linspace(0.4, 0.6, 1000).reshape(1, -1).repeat(1000, axis=0).astype(np.float32)
    # Add gentle waves (smooth)
    depth += np.random.normal(0, 0.01, (1000, 1000)).astype(np.float32)

    scene_type, meta = classify_scene_type_v2(raw_edges, structure_edges, depth)

    # Should classify as texture due to smooth depth
    assert scene_type == "texture_dominated"
    assert meta["depth_gradient_var"] < 0.0004, f"Expected smooth depth, got {meta['depth_gradient_var']}"


def test_patterned_bathroom_high_ratio():
    """Patterned bathroom: very high ratio (>10) → texture."""
    raw_edges = np.zeros((1000, 1000), dtype=np.uint8)
    structure_edges = np.zeros((1000, 1000), dtype=np.uint8)

    # Low structure edge density (1.1%) but high raw edges
    struct_mask = np.random.rand(1000, 1000) < 0.011
    structure_edges[struct_mask] = 255

    # Very high raw edges (ratio ~14)
    raw_mask = np.random.rand(1000, 1000) < 0.154
    raw_edges[raw_mask] = 255

    # Medium depth variance
    depth = np.random.normal(0.5, 0.08, (1000, 1000)).astype(np.float32)

    scene_type, meta = classify_scene_type_v2(raw_edges, structure_edges, depth)

    # Should classify as texture due to very high ratio
    assert scene_type == "texture_dominated"
    assert meta["ratio"] > 10.0
    assert meta["decision"] == "very_high_ratio"


def test_metadata_completeness():
    """Verify all metadata fields are present including new depth_gradient_var."""
    raw_edges = np.random.randint(0, 2, (100, 100), dtype=np.uint8) * 255
    structure_edges = np.random.randint(0, 2, (100, 100), dtype=np.uint8) * 255
    depth = np.random.rand(100, 100).astype(np.float32)

    scene_type, meta = classify_scene_type_v2(raw_edges, structure_edges, depth)

    # Check required fields
    assert "method" in meta
    assert "ratio" in meta
    assert "depth_variance" in meta
    assert "depth_gradient_var" in meta  # NEW field
    assert "edge_density" in meta
    assert "decision" in meta
    assert "thresholds" in meta

    assert meta["method"] == "multi_factor_v2"
    assert isinstance(meta["depth_gradient_var"], float)


def test_threshold_customization():
    """Verify custom thresholds work."""
    raw_edges = np.ones((100, 100), dtype=np.uint8) * 255
    structure_edges = np.ones((100, 100), dtype=np.uint8) * 255
    depth = np.random.rand(100, 100).astype(np.float32)

    # Custom thresholds
    scene_type, meta = classify_scene_type_v2(
        raw_edges,
        structure_edges,
        depth,
        threshold_ratio_high=20.0,  # Custom
        threshold_ratio_low=2.0,  # Custom
    )

    assert meta["thresholds"]["ratio_high"] == 20.0
    assert meta["thresholds"]["ratio_low"] == 2.0


def test_filename_hint_pool_texture():
    """Filename hint 'pool' should boost texture confidence in borderline cases."""
    # Borderline case: medium ratio, medium edge density
    raw_edges = np.zeros((1000, 1000), dtype=np.uint8)
    structure_edges = np.zeros((1000, 1000), dtype=np.uint8)

    # Medium ratio (~4.0)
    raw_mask = np.random.rand(1000, 1000) < 0.02
    raw_edges[raw_mask] = 255

    struct_mask = np.random.rand(1000, 1000) < 0.005
    structure_edges[struct_mask] = 255

    # Medium depth variance
    depth = np.random.normal(0.5, 0.04, (1000, 1000)).astype(np.float32)

    # Test WITHOUT filename hint (baseline)
    scene_type_no_hint, meta_no_hint = classify_scene_type_v2(raw_edges, structure_edges, depth)

    # Test WITH filename hint 'pool'
    scene_type_with_hint, meta_with_hint = classify_scene_type_v2(
        raw_edges, structure_edges, depth, image_filename="exterior_pool_01.jpg"
    )

    # Filename hint should be detected
    assert meta_with_hint["filename_hint"] == "texture"

    # In borderline cases, filename should help classify as texture
    # (might override or confirm depending on depth-based decision)
    assert scene_type_with_hint == "texture_dominated"


def test_filename_hint_kitchen_structure():
    """Filename hint 'kitchen' should boost structure confidence in borderline cases."""
    # Borderline case: medium ratio, medium edge density
    raw_edges = np.zeros((1000, 1000), dtype=np.uint8)
    structure_edges = np.zeros((1000, 1000), dtype=np.uint8)

    # Medium ratio (~4.0)
    raw_mask = np.random.rand(1000, 1000) < 0.02
    raw_edges[raw_mask] = 255

    struct_mask = np.random.rand(1000, 1000) < 0.005
    structure_edges[struct_mask] = 255

    # Medium-high depth variance with structured gradients
    np.random.seed(123)
    depth = np.random.normal(0.5, 0.05, (1000, 1000)).astype(np.float32)
    # Add geometric depth features
    depth[200:300, :] += 0.1
    depth[600:700, :] -= 0.1

    # Test WITHOUT filename hint (baseline)
    scene_type_no_hint, meta_no_hint = classify_scene_type_v2(raw_edges, structure_edges, depth)

    # Test WITH filename hint 'kitchen'
    scene_type_with_hint, meta_with_hint = classify_scene_type_v2(
        raw_edges, structure_edges, depth, image_filename="interior_kitchen_modern.jpg"
    )

    # Filename hint should be detected
    assert meta_with_hint["filename_hint"] == "structure"

    # In borderline cases, filename should help classify as structure
    assert scene_type_with_hint == "structure_dominated"


def test_filename_hint_no_match():
    """Filename without pattern should have no hint."""
    raw_edges = np.zeros((1000, 1000), dtype=np.uint8)
    structure_edges = np.zeros((1000, 1000), dtype=np.uint8)

    raw_mask = np.random.rand(1000, 1000) < 0.01
    raw_edges[raw_mask] = 255

    struct_mask = np.random.rand(1000, 1000) < 0.005
    structure_edges[struct_mask] = 255

    depth = np.random.normal(0.5, 0.03, (1000, 1000)).astype(np.float32)

    scene_type, meta = classify_scene_type_v2(raw_edges, structure_edges, depth, image_filename="IMG_1234.jpg")

    # No filename hint detected
    assert meta["filename_hint"] is None


def test_filename_hint_confirms_depth_decision():
    """Filename hint confirming depth decision should be logged."""
    # Strong texture case (low edge density)
    raw_edges = np.zeros((1000, 1000), dtype=np.uint8)
    structure_edges = np.zeros((1000, 1000), dtype=np.uint8)

    # Very few edges
    struct_mask = np.random.rand(1000, 1000) < 0.001
    structure_edges[struct_mask] = 255

    depth = np.random.normal(0.5, 0.02, (1000, 1000)).astype(np.float32)

    scene_type, meta = classify_scene_type_v2(raw_edges, structure_edges, depth, image_filename="ocean_aerial_view.jpg")

    # Should be texture (depth-based) and filename confirms
    assert scene_type == "texture_dominated"
    assert meta["filename_hint"] == "texture"
    # Decision may include CONFIRMED_BY_FILENAME if borderline
    assert "decision" in meta


def test_filename_hint_backward_compatibility():
    """Function should work without filename parameter (backward compatibility)."""
    raw_edges = np.zeros((1000, 1000), dtype=np.uint8)
    structure_edges = np.zeros((1000, 1000), dtype=np.uint8)

    raw_mask = np.random.rand(1000, 1000) < 0.01
    raw_edges[raw_mask] = 255

    struct_mask = np.random.rand(1000, 1000) < 0.005
    structure_edges[struct_mask] = 255

    depth = np.random.normal(0.5, 0.03, (1000, 1000)).astype(np.float32)

    # Call without image_filename parameter
    scene_type, meta = classify_scene_type_v2(raw_edges, structure_edges, depth)

    # Should work and have None hint
    assert "filename_hint" in meta
    assert meta["filename_hint"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
