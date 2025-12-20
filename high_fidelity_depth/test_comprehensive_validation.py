#!/usr/bin/env python3
"""
Test Comprehensive Validation System
=====================================

Smoke test to verify:
1. Metric system works end-to-end
2. JSON is atomic and parseable
3. Edge overlay generation works
4. Seam detection works
"""

import json
import logging
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from comprehensive_validation import run_comprehensive_validation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_synthetic_test_case(tmpdir: Path) -> tuple[Path, Path]:
    """
    Create synthetic RGB + depth for testing.
    
    Creates:
    - RGB with some edges (checkerboard pattern)
    - Depth with aligned edges
    
    Returns:
        (rgb_path, depth_path)
    """
    h, w = 512, 512
    
    # Create checkerboard RGB
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    tile_size = 64
    for i in range(0, h, tile_size):
        for j in range(0, w, tile_size):
            if ((i // tile_size) + (j // tile_size)) % 2 == 0:
                rgb[i:i+tile_size, j:j+tile_size] = [200, 200, 200]
            else:
                rgb[i:i+tile_size, j:j+tile_size] = [50, 50, 50]
    
    # Create depth map with aligned edges
    depth = np.zeros((h, w), dtype=np.float32)
    for i in range(0, h, tile_size):
        for j in range(0, w, tile_size):
            if ((i // tile_size) + (j // tile_size)) % 2 == 0:
                depth[i:i+tile_size, j:j+tile_size] = 0.8
            else:
                depth[i:i+tile_size, j:j+tile_size] = 0.2
    
    # Smooth depth slightly to make it realistic
    depth = cv2.GaussianBlur(depth, (5, 5), 1.0)
    
    # Save
    rgb_path = tmpdir / "test_rgb.png"
    depth_path = tmpdir / "test_depth.png"
    
    Image.fromarray(rgb).save(rgb_path)
    
    # Save depth as 16-bit
    depth_uint16 = (depth * 65535).astype(np.uint16)
    Image.fromarray(depth_uint16, mode='I;16').save(depth_path)
    
    logger.info(f"Created synthetic test case: {tmpdir}")
    
    return rgb_path, depth_path


def test_comprehensive_validation():
    """Test the comprehensive validation end-to-end."""
    logger.info("="*60)
    logger.info("TESTING COMPREHENSIVE VALIDATION SYSTEM")
    logger.info("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create synthetic test case
        rgb_path, depth_path = create_synthetic_test_case(tmpdir)
        
        # Run validation
        output_dir = tmpdir / "validation_output"
        report = run_comprehensive_validation(
            rgb_path,
            depth_path,
            output_dir,
            tile_size=1024,
            overlap=128
        )
        
        # Verify report structure
        assert "metrics" in report
        assert "quality_score" in report
        assert "seam_validation" in report
        assert "passed_lenient" in report
        assert "passed_strict" in report
        
        # Verify JSON was written and is parseable
        report_path = output_dir / "validation_report.json"
        assert report_path.exists(), "Report JSON not created"
        
        with open(report_path, 'r') as f:
            loaded = json.load(f)
        
        assert "quality_score" in loaded
        assert isinstance(loaded["quality_score"], (int, float))
        
        # Verify overlay was created
        overlay_path = output_dir / "edge_overlay.png"
        assert overlay_path.exists(), "Edge overlay not created"
        
        # Verify overlay is valid image
        overlay = Image.open(overlay_path)
        assert overlay.size == (512, 512)
        
        logger.info("="*60)
        logger.info("✅ ALL VALIDATION SYSTEM TESTS PASSED")
        logger.info("="*60)
        logger.info(f"Quality score: {report['quality_score']:.3f}")
        logger.info(f"Edge F1: {report['metrics']['edge_f1']:.3f}")
        logger.info(f"Passed (lenient): {report['passed_lenient']}")


if __name__ == "__main__":
    test_comprehensive_validation()
