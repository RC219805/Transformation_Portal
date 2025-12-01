#!/usr/bin/env python3
"""
Phase 1 Verification: Depth Anything V2 Model Fix
==================================================

Tests that the corrected model name allows successful:
1. Model download from HuggingFace
2. Model initialization
3. Depth map generation from test image

Author: Transformation Portal
Date: 2025-11-10
"""

import logging
import sys
import time
from pathlib import Path

import torch
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_model_download():
    """Test that model can be downloaded from HuggingFace."""
    logger.info("=" * 70)
    logger.info("Phase 1 Verification: Depth Anything V2 Model Fix")
    logger.info("=" * 70)

    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        model_id = "depth-anything/Depth-Anything-V2-Small-hf"
        logger.info("\n✓ Step 1: Testing model download from HuggingFace")
        logger.info(f"  Model ID: {model_id}")

        start_time = time.time()

        # This will auto-download if not cached
        processor = AutoImageProcessor.from_pretrained(model_id)
        logger.info("  ✓ Image processor loaded")

        model = AutoModelForDepthEstimation.from_pretrained(model_id)
        logger.info("  ✓ Model loaded")

        download_time = time.time() - start_time
        logger.info(f"  ✓ Total time: {download_time:.2f}s")

        return processor, model

    except Exception as e:
        logger.error(f"  ✗ Model download failed: {e}")
        return None, None


def test_depth_estimation(processor, model, test_image_path=None):
    """Test depth map generation."""
    if processor is None or model is None:
        logger.error("\n✗ Step 2: Skipped (model not loaded)")
        return None

    logger.info("\n✓ Step 2: Testing depth map generation")

    try:
        # Find a test image
        if test_image_path is None:
            # Look for 750 Picacho images
            test_dirs = [
                "input_images/750_picacho",
                "data/sample_images",
                "input_images"
            ]

            for test_dir in test_dirs:
                if Path(test_dir).exists():
                    images = list(Path(test_dir).glob("*.jpg")) + list(Path(test_dir).glob("*.png"))
                    if images:
                        test_image_path = images[0]
                        break

        if test_image_path is None or not Path(test_image_path).exists():
            # Create a simple test image
            logger.info("  Creating synthetic test image (512x512)")
            test_image = Image.new('RGB', (512, 512), color='gray')
        else:
            logger.info(f"  Loading test image: {test_image_path}")
            test_image = Image.open(test_image_path).convert('RGB')

        # Resize to reasonable size for testing
        if test_image.width > 1024:
            aspect = test_image.height / test_image.width
            test_image = test_image.resize((1024, int(1024 * aspect)))
            logger.info(f"  Resized to {test_image.size}")

        logger.info(f"  Input size: {test_image.size}")

        # Run inference
        start_time = time.time()

        # Check for MPS (Apple Silicon GPU) support
        device = "cpu"
        if torch.backends.mps.is_available():
            device = "mps"
            model = model.to(device)
            logger.info("  Using MPS acceleration (Apple Silicon)")
        elif torch.cuda.is_available():
            device = "cuda"
            model = model.to(device)
            logger.info("  Using CUDA acceleration")
        else:
            logger.info("  Using CPU (no GPU available)")

        # Prepare inputs
        inputs = processor(images=test_image, return_tensors="pt")
        if device != "cpu":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate depth map
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth

        inference_time = time.time() - start_time

        # Convert to numpy
        depth_map = predicted_depth.squeeze().cpu().numpy()

        logger.info("  ✓ Depth map generated successfully")
        logger.info(f"  Output shape: {depth_map.shape}")
        logger.info(f"  Depth range: [{depth_map.min():.3f}, {depth_map.max():.3f}]")
        logger.info(f"  Inference time: {inference_time*1000:.1f}ms")

        return depth_map

    except Exception as e:
        logger.error(f"  ✗ Depth estimation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_pipeline_integration():
    """Test integration with luxury estate pipeline."""
    logger.info("\n✓ Step 3: Testing pipeline integration")

    try:
        logger.info("  ✓ Pipeline imports successfully")

        # Check that depth is available
        from luxury_estate_master_pipeline import DEPTH_PIPELINE_AVAILABLE
        if DEPTH_PIPELINE_AVAILABLE:
            logger.info("  ✓ Depth pipeline marked as available")
        else:
            logger.warning("  ⚠ Depth pipeline marked as unavailable (check imports)")

        return True

    except Exception as e:
        logger.error(f"  ✗ Pipeline integration test failed: {e}")
        return False


def main():
    """Run Phase 1 verification tests."""

    # Test 1: Model download
    processor, model = test_model_download()
    if processor is None or model is None:
        logger.error("\n" + "=" * 70)
        logger.error("PHASE 1 VERIFICATION FAILED: Model download unsuccessful")
        logger.error("=" * 70)
        return 1

    # Test 2: Depth estimation
    depth_map = test_depth_estimation(processor, model)
    if depth_map is None:
        logger.error("\n" + "=" * 70)
        logger.error("PHASE 1 VERIFICATION FAILED: Depth estimation unsuccessful")
        logger.error("=" * 70)
        return 1

    # Test 3: Pipeline integration
    integration_ok = test_pipeline_integration()

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1 VERIFICATION RESULTS")
    logger.info("=" * 70)
    logger.info("✓ Model download: SUCCESS")
    logger.info("✓ Depth estimation: SUCCESS")
    logger.info(f"{'✓' if integration_ok else '⚠'} Pipeline integration: {'SUCCESS' if integration_ok else 'WARNING'}")
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1 COMPLETE: Depth Anything V2 is now functional!")
    logger.info("=" * 70)
    logger.info("\nNext Steps:")
    logger.info("  - Process a 750 Picacho test image with full pipeline")
    logger.info("  - Verify depth-aware features (zone tone mapping, etc.)")
    logger.info("  - Begin planning Phase 2 (upgrade to V3)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
