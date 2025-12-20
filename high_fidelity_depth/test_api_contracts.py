#!/usr/bin/env python3
"""API contract tests to catch signature mismatches."""

import inspect
import pytest
import numpy as np

from high_fidelity_depth.quality_metrics import validate_depth_quality


def test_validate_depth_quality_signature():
    """Ensure validate_depth_quality signature matches expected contract."""
    sig = inspect.signature(validate_depth_quality)
    params = list(sig.parameters.keys())
    
    # Required parameters
    assert 'rgb' in params, "Missing rgb parameter"
    assert 'depth' in params, "Missing depth parameter"
    
    # Should NOT have return_dict (we don't support it)
    assert 'return_dict' not in params, \
        "return_dict parameter found - this causes validation runner to fail"
    
    print(f"✓ validate_depth_quality signature: {params}")


def test_validate_depth_quality_returns_object():
    """Ensure function returns structured object, not dict."""
    # Create dummy inputs
    depth = np.random.rand(100, 100).astype(np.float32)
    rgb = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    
    result = validate_depth_quality(rgb, depth)
    
    # Should return object with attributes, not dict
    assert hasattr(result, 'edge_f1'), "Missing edge_f1 attribute"
    assert hasattr(result, 'chamfer_distance'), "Missing chamfer_distance attribute"
    assert hasattr(result, 'edge_overlap'), "Missing edge_overlap attribute"
    assert hasattr(result, 'edge_count_ratio'), "Missing edge_count_ratio attribute"
    assert hasattr(result, 'halo_score'), "Missing halo_score attribute"
    assert hasattr(result, 'overshoot_penalty'), "Missing overshoot_penalty attribute"
    
    # Should have quality_score method
    assert hasattr(result, 'quality_score'), "Missing quality_score method"
    assert callable(result.quality_score), "quality_score should be callable"
    
    # Should have to_dict method
    assert hasattr(result, 'to_dict'), "Missing to_dict method"
    assert callable(result.to_dict), "to_dict should be callable"
    
    # Test to_dict conversion
    metrics_dict = result.to_dict()
    assert isinstance(metrics_dict, dict), "to_dict should return dict"
    assert 'edge_f1' in metrics_dict, "to_dict should include edge_f1"
    
    print(f"✓ Returns object with attributes: {type(result).__name__}")


def test_edge_metrics_dataclass_consistency():
    """Ensure EdgeMetrics dataclass has all expected fields."""
    from high_fidelity_depth.quality_metrics import EdgeMetrics
    
    # Check if it's a dataclass
    assert hasattr(EdgeMetrics, '__dataclass_fields__'), "EdgeMetrics should be a dataclass"
    
    # Required fields
    required_fields = [
        'edge_f1', 'edge_overlap', 'edge_alignment_corr', 'chamfer_distance',
        'edge_width', 'edge_sharpness_p95', 'edge_count_ratio', 'halo_score',
        'overshoot_penalty', 'rgb_edge_count', 'depth_edge_count'
    ]
    
    fields = EdgeMetrics.__dataclass_fields__.keys()
    for field in required_fields:
        assert field in fields, f"Missing required field: {field}"
    
    print(f"✓ EdgeMetrics has all required fields: {list(fields)}")


def test_validate_depth_quality_parameter_order():
    """Ensure validate_depth_quality has correct parameter order (rgb, depth)."""
    sig = inspect.signature(validate_depth_quality)
    params = list(sig.parameters.keys())
    
    # First two parameters should be rgb and depth (in that order)
    assert params[0] == 'rgb', f"First parameter should be 'rgb', got '{params[0]}'"
    assert params[1] == 'depth', f"Second parameter should be 'depth', got '{params[1]}'"
    
    print(f"✓ Parameter order correct: {params[:2]}")


if __name__ == "__main__":
    print("Running API contract tests...\n")
    
    try:
        test_validate_depth_quality_signature()
        test_validate_depth_quality_returns_object()
        test_edge_metrics_dataclass_consistency()
        test_validate_depth_quality_parameter_order()
        
        print("\n✅ All API contract tests passed")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import sys
        sys.exit(1)
