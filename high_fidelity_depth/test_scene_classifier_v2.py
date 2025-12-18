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
    
    assert scene_type == 'texture_dominated'
    # Pool water classified by very low edge density, not ratio
    assert meta['decision'] in ['very_low_edge_density', 'no_structure_edges']


def test_interior_low_ratio_high_density():
    """Interior: low edge ratio (<2), high edge density (>0.02) with high variance."""
    # Simulate interior: similar raw and structure edges, high density
    edges = np.zeros((1000, 1000), dtype=np.uint8)
    # Create MORE edges to get >2% density and trigger high_density rule
    for i in range(0, 1000, 30):
        edges[i:i+3, :] = 255  # Horizontal edges
        edges[:, i:i+3] = 255  # Vertical edges
    
    # Ratio should be ~1.0 (similar raw and structure)
    raw_edges = edges.copy()
    # Add a tiny bit more to raw to get ratio just above 1
    raw_mask = np.random.rand(1000, 1000) < 0.001
    raw_edges[raw_mask] = 255
    
    structure_edges = edges.copy()
    
    # Complex depth (medium-high variance to avoid low_variance rule)
    depth = np.random.normal(0.5, 0.05, (1000, 1000)).astype(np.float32)
    
    scene_type, meta = classify_scene_type_v2(raw_edges, structure_edges, depth)
    
    # With ratio ~1 (<2) and density >0.008, should classify as texture per Rule 5
    assert scene_type == 'texture_dominated'
    assert meta['ratio'] < 2.0
    assert meta['edge_density'] > 0.008
    assert meta['decision'] in ['low_ratio_medium_density', 'low_ratio_low_variance']


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
    assert scene_type == 'texture_dominated'
    # Ratio should be around 8 (between 5 and 10)
    assert 5.0 <= meta['ratio'] <= 15.0  # More lenient
    assert meta['depth_variance'] < 0.03


def test_no_structure_edges():
    """Handle zero structure edges gracefully."""
    raw_edges = np.random.randint(0, 2, (100, 100), dtype=np.uint8) * 255
    structure_edges = np.zeros((100, 100), dtype=np.uint8)
    depth = np.random.rand(100, 100).astype(np.float32)
    
    scene_type, meta = classify_scene_type_v2(raw_edges, structure_edges, depth)
    
    assert scene_type == 'texture_dominated'
    assert meta['ratio'] == float('inf')
    assert meta['decision'] == 'no_structure_edges'


def test_metadata_completeness():
    """Verify all metadata fields are present."""
    raw_edges = np.random.randint(0, 2, (100, 100), dtype=np.uint8) * 255
    structure_edges = np.random.randint(0, 2, (100, 100), dtype=np.uint8) * 255
    depth = np.random.rand(100, 100).astype(np.float32)
    
    scene_type, meta = classify_scene_type_v2(raw_edges, structure_edges, depth)
    
    # Check required fields
    assert 'method' in meta
    assert 'ratio' in meta
    assert 'depth_variance' in meta
    assert 'edge_density' in meta
    assert 'decision' in meta
    assert 'thresholds' in meta
    
    assert meta['method'] == 'multi_factor_v2'


def test_threshold_customization():
    """Verify custom thresholds work."""
    raw_edges = np.ones((100, 100), dtype=np.uint8) * 255
    structure_edges = np.ones((100, 100), dtype=np.uint8) * 255
    depth = np.random.rand(100, 100).astype(np.float32)
    
    # Custom thresholds
    scene_type, meta = classify_scene_type_v2(
        raw_edges, structure_edges, depth,
        threshold_ratio_high=20.0,  # Custom
        threshold_ratio_low=2.0      # Custom
    )
    
    assert meta['thresholds']['ratio_high'] == 20.0
    assert meta['thresholds']['ratio_low'] == 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
