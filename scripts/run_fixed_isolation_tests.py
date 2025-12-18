#!/usr/bin/env python3
"""
Re-run isolation tests with fixed quality metrics.

This script:
1. Uses canonical quality_metrics module (atomic JSON, F1 score, proper thresholds)
2. Tests on 750 Picacho interior images
3. Validates that JSON is not truncated
4. Produces coherent, calibrated metrics

Reference: Fix for truncated JSON and mismatched metric definitions.
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


def main():
    """Run comprehensive isolation tests."""
    from high_fidelity_depth.isolation_tests import run_isolation_tests
    
    # Find test images
    input_dir = Path("input_images/750_Picacho/Source_TIFFs")
    
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return 1
    
    # Get first image for testing
    image_files = list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))
    
    if not image_files:
        logger.error(f"No TIFF images found in {input_dir}")
        return 1
    
    test_image = image_files[0]
    logger.info(f"Testing with: {test_image.name}")
    
    # Load image
    rgb = np.array(Image.open(test_image))
    logger.info(f"Image shape: {rgb.shape}, dtype: {rgb.dtype}")
    
    # Convert to float32 if needed
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32) / 255.0
    
    # Run isolation tests
    output_dir = Path("outputs/high_fidelity_validation_fixed")
    
    try:
        results = run_isolation_tests(rgb, output_dir=output_dir)
        
        # Validate JSON was saved correctly
        json_path = output_dir / "isolation_test_results.json"
        
        if not json_path.exists():
            logger.error("❌ JSON file not created")
            return 1
        
        # Verify JSON is valid
        import json
        with open(json_path) as f:
            data = json.load(f)
        
        logger.info(f"✅ JSON validated: {len(data)} test results")
        
        # Print summary
        print("\n" + "="*60)
        print("VALIDATION SUMMARY (Fixed Metrics)")
        print("="*60)
        
        for name, result in results.items():
            print(f"\n{result}")
        
        # Check if any tests failed
        failed = [name for name, result in results.items() if not result.passed]
        
        if failed:
            print(f"\n⚠️  Tests failed: {', '.join(failed)}")
            print("Review metrics and thresholds.")
        else:
            print("\n✅ All tests passed!")
        
        return 0
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
