#!/usr/bin/env python3
"""
Quick validation: Confirm bypass_image_processor prevents 518px resize
"""

import logging
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_bypass():
    """Validate that bypass mode preserves tile resolution."""
    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        import torch
    except ImportError:
        logger.error("PyTorch and transformers required")
        return
    
    logger.info("=" * 60)
    logger.info("VALIDATION: Bypass Image Processor")
    logger.info("=" * 60)
    
    # Load model
    model_name = "depth-anything/Depth-Anything-V2-Large-hf"
    logger.info(f"Loading: {model_name}")
    
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name)
    model.eval()
    
    # Test tile sizes
    for tile_size in [512, 1024]:
        logger.info(f"\n--- Testing {tile_size}×{tile_size} ---")
        
        # Create test tile
        test_img = np.random.randint(0, 255, (tile_size, tile_size, 3), dtype=np.uint8)
        test_pil = Image.fromarray(test_img)
        
        logger.info(f"Input PIL size: {test_pil.size}")
        
        # Process WITH bypass (do_resize=False)
        inputs_bypass = processor(
            images=test_pil,
            return_tensors="pt",
            do_resize=False
        )
        
        tensor_shape = tuple(inputs_bypass['pixel_values'].shape)
        logger.info(f"Bypass mode tensor: {tensor_shape}")
        
        # Check
        batch, channels, h, w = tensor_shape
        if h == tile_size and w == tile_size:
            logger.info(f"✓ PASS: No resize ({tile_size}×{tile_size} preserved)")
        else:
            logger.warning(f"✗ FAIL: Resize detected ({tile_size}×{tile_size} → {h}×{w})")
        
        # Compare to default mode
        inputs_default = processor(
            images=test_pil,
            return_tensors="pt"
        )
        default_shape = tuple(inputs_default['pixel_values'].shape)
        logger.info(f"Default mode tensor: {default_shape} (should be 518×518)")
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ Validation complete")


if __name__ == "__main__":
    validate_bypass()
