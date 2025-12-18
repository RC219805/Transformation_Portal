#!/usr/bin/env python3
"""
Regression Test: Atomic JSON Write + Metric Coherence
======================================================

Validates that:
1. JSON writes are atomic and parseable
2. All test paths use the same metric implementation
3. Metrics are internally consistent
"""

import json
import logging
import tempfile
from pathlib import Path

import numpy as np

from quality_metrics import (
    EdgeMetrics,
    validate_depth_quality,
    save_metrics_atomic
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_atomic_write():
    """Test atomic JSON write with validation."""
    logger.info("Testing atomic JSON write...")
    
    # Create test metrics
    metrics = {
        "test": {
            "edge_f1": np.float32(0.42),
            "edge_overlap": np.float64(0.75),
            "edge_count_ratio": 1.5,
            "passed": True
        }
    }
    
    # Write to temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_metrics.json"
        
        try:
            save_metrics_atomic(metrics, output_path)
            
            # Validate by reading back
            with open(output_path, 'r') as f:
                loaded = json.load(f)
            
            assert "test" in loaded
            assert abs(loaded["test"]["edge_f1"] - 0.42) < 1e-6
            assert loaded["test"]["passed"] is True
            
            logger.info("✅ Atomic write test PASSED")
            
        except Exception as e:
            logger.error(f"❌ Atomic write test FAILED: {e}")
            raise


def test_metric_consistency():
    """Test that metrics are internally consistent."""
    logger.info("Testing metric consistency...")
    
    # Create synthetic depth and RGB
    h, w = 512, 512
    rgb = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    depth = np.random.rand(h, w).astype(np.float32)
    
    # Compute metrics twice
    metrics1 = validate_depth_quality(rgb, depth)
    metrics2 = validate_depth_quality(rgb, depth)
    
    # Should be identical
    assert abs(metrics1.edge_f1 - metrics2.edge_f1) < 1e-6
    assert abs(metrics1.edge_overlap - metrics2.edge_overlap) < 1e-6
    assert abs(metrics1.chamfer_distance - metrics2.chamfer_distance) < 1e-6
    
    logger.info("✅ Metric consistency test PASSED")


def test_metric_dict_serialization():
    """Test that EdgeMetrics.to_dict() is JSON-serializable."""
    logger.info("Testing EdgeMetrics serialization...")
    
    metrics = EdgeMetrics(
        edge_f1=np.float32(0.42),
        edge_overlap=np.float64(0.75),
        edge_alignment_corr=0.05,
        chamfer_distance=np.float32(3.5),
        edge_width=2.1,
        edge_sharpness_p95=85.0,
        edge_count_ratio=1.5,
        halo_score=0.8,
        overshoot_penalty=0.2,
        rgb_edge_count=1000,
        depth_edge_count=1500
    )
    
    # Convert to dict
    metrics_dict = metrics.to_dict()
    
    # Should be JSON-serializable
    try:
        json_str = json.dumps(metrics_dict)
        loaded = json.loads(json_str)
        
        assert abs(loaded["edge_f1"] - 0.42) < 1e-5
        assert abs(loaded["edge_overlap"] - 0.75) < 1e-5
        assert loaded["rgb_edge_count"] == 1000
        
        logger.info("✅ EdgeMetrics serialization test PASSED")
        
    except Exception as e:
        logger.error(f"❌ EdgeMetrics serialization test FAILED: {e}")
        raise


def test_quality_score_bounds():
    """Test that quality_score() is always in [0, 1]."""
    logger.info("Testing quality score bounds...")
    
    # Extreme values
    extreme_metrics = EdgeMetrics(
        edge_f1=1.0,
        edge_overlap=1.0,
        edge_alignment_corr=1.0,
        chamfer_distance=0.0,
        edge_width=0.5,
        edge_sharpness_p95=300.0,  # Very high
        edge_count_ratio=10.0,  # Very high
        halo_score=1.0,
        overshoot_penalty=0.0,
        rgb_edge_count=1000,
        depth_edge_count=10000
    )
    
    score = extreme_metrics.quality_score()
    
    assert 0.0 <= score <= 1.0, f"Quality score {score} out of bounds"
    
    logger.info(f"✅ Quality score bounds test PASSED (score={score:.3f})")


def run_all_tests():
    """Run all regression tests."""
    logger.info("="*60)
    logger.info("METRIC SYSTEM REGRESSION TESTS")
    logger.info("="*60)
    
    test_atomic_write()
    test_metric_consistency()
    test_metric_dict_serialization()
    test_quality_score_bounds()
    
    logger.info("="*60)
    logger.info("✅ ALL TESTS PASSED")
    logger.info("="*60)


if __name__ == "__main__":
    run_all_tests()
