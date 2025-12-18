#!/usr/bin/env python3
"""
Run High-Fidelity Depth Pipeline Isolation Tests
================================================

Tests the fixes for:
1. Internal resize bug (PRIORITY 1)
2. Edge detection for float depth (PRIORITY 2)
3. Robust scale reconciliation (PRIORITY 3)
4. Global anchor fusion (PRIORITY 4)
5. Edge snapping refinement (PRIORITY 5)
"""

import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add high_fidelity_depth to path
sys.path.insert(0, str(Path(__file__).parent))

from high_fidelity_depth.isolation_tests import run_isolation_tests


def main():
    """Run isolation tests on sample image."""
    
    # Load test image
    test_image_path = Path("/Users/rc/Transformation_Portal/input_images/750_Picacho/Pool.tif")
    
    if not test_image_path.exists():
        logger.error(f"Test image not found: {test_image_path}")
        sys.exit(1)
    
    logger.info(f"Loading test image: {test_image_path}")
    rgb_pil = Image.open(test_image_path)
    
    # Resize to manageable size for testing (if needed)
    max_size = 2048
    if max(rgb_pil.size) > max_size:
        scale = max_size / max(rgb_pil.size)
        new_size = (int(rgb_pil.width * scale), int(rgb_pil.height * scale))
        rgb_pil = rgb_pil.resize(new_size, Image.LANCZOS)
        logger.info(f"Resized to {new_size} for testing")
    
    rgb = np.array(rgb_pil)
    logger.info(f"Image shape: {rgb.shape}, dtype: {rgb.dtype}")
    
    # Create output directory
    output_dir = Path("outputs/isolation_tests")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run isolation tests
    logger.info("="*60)
    logger.info("STARTING ISOLATION TESTS")
    logger.info("="*60)
    
    results = run_isolation_tests(rgb, output_dir=output_dir)
    
    # Report results
    logger.info("\n" + "="*60)
    logger.info("FINAL RESULTS")
    logger.info("="*60)
    
    all_passed = True
    for name, result in results.items():
        status = "✅ PASS" if result.passed else "❌ FAIL"
        logger.info(f"{status} {name}: Edge F1={result.metrics.edge_f1:.3f}, Quality={result.metrics.quality_score():.3f}")
        
        if name != "baseline" and not result.passed:
            all_passed = False
    
    logger.info("="*60)
    
    if all_passed:
        logger.info("✅ ALL TESTS PASSED")
        logger.info("Fixes validated successfully!")
        return 0
    else:
        logger.error("❌ SOME TESTS FAILED")
        logger.error("Review logs for details")
        return 1


if __name__ == "__main__":
    sys.exit(main())
