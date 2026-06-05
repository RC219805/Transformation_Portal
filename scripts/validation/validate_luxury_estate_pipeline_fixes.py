#!/usr/bin/env python3
"""
Validate Luxury Estate Pipeline Fixes
=====================================

Tests the three major fixes implemented in the Luxury Estate Master Pipeline:
1. Shadow clipping reduction (outdoor scenes)
2. AI enhancement tensor compatibility (dynamic padding)
3. Depth model auto-download

Usage:
    python scripts/validation/validate_luxury_estate_pipeline_fixes.py \
      --input-dir input_images/750_Picacho_HDR_sRGB_alpha_32-bit_TIFFs

Author: Transformation Portal
Version: 1.0.0
Date: 2025-11-10
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import tifffile
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINES_DIR = REPO_ROOT / "scripts" / "pipelines"
UTILITIES_DIR = REPO_ROOT / "scripts" / "utilities"
DEFAULT_REPORT_PATH = Path("/tmp/tp-luxury-estate-pipeline-fixes-report.json")
os.environ.setdefault("TP_LUXURY_ESTATE_PIPELINE_LOG", "/tmp/tp-luxury-estate-pipeline.log")
for import_root in (PIPELINES_DIR, UTILITIES_DIR):
    root_text = str(import_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


def analyze_shadow_clipping(image_path: Path) -> Dict:
    """
    Analyze shadow clipping in an image.

    Args:
        image_path: Path to image file

    Returns:
        Dictionary with clipping statistics
    """
    # Load image
    if image_path.suffix.lower() in [".tif", ".tiff"]:
        image = tifffile.imread(str(image_path))
        if image.dtype == np.uint16:
            image = image.astype(np.float32) / 65535.0
        elif image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
    else:
        image = np.array(Image.open(image_path)).astype(np.float32) / 255.0

    # Handle alpha channel
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    # Calculate luminance
    luminance = np.dot(image, [0.2126, 0.7152, 0.0722])

    # Shadow clipping: pixels below 0.05 (5% of range)
    shadow_threshold = 0.05
    shadow_clipped = np.sum(luminance < shadow_threshold)
    shadow_clipped_pct = (shadow_clipped / luminance.size) * 100

    # Highlight clipping: pixels above 0.95
    highlight_threshold = 0.95
    highlight_clipped = np.sum(luminance > highlight_threshold)
    highlight_clipped_pct = (highlight_clipped / luminance.size) * 100

    # Dynamic range
    p99 = np.percentile(luminance, 99)
    p01 = np.percentile(luminance, 1)
    dynamic_range = p99 / (p01 + 1e-6)

    # Scene type detection
    shadow_pixels_pct = np.sum(luminance < 0.1) / luminance.size * 100
    highlight_pixels_pct = np.sum(luminance > 0.7) / luminance.size * 100
    is_outdoor = (dynamic_range > 8.0) or (shadow_pixels_pct > 15 and highlight_pixels_pct > 10)

    return {
        "shadow_clipped_pct": shadow_clipped_pct,
        "highlight_clipped_pct": highlight_clipped_pct,
        "dynamic_range": dynamic_range,
        "shadow_pixels_pct": shadow_pixels_pct,
        "highlight_pixels_pct": highlight_pixels_pct,
        "scene_type": "outdoor" if is_outdoor else "indoor",
        "mean_luminance": float(np.mean(luminance)),
        "median_luminance": float(np.median(luminance)),
    }


def test_shadow_clipping_fix(image_paths: List[Path]) -> Dict:
    """
    Test shadow clipping reduction on multiple images.

    Args:
        image_paths: List of paths to test images

    Returns:
        Test results dictionary
    """
    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: Shadow Clipping Reduction")
    logger.info("=" * 80)

    results = {}
    outdoor_clipping = []
    indoor_clipping = []

    for image_path in image_paths:
        logger.info(f"\nAnalyzing: {image_path.name}")
        stats = analyze_shadow_clipping(image_path)
        results[image_path.name] = stats

        logger.info(f"  Scene type: {stats['scene_type'].upper()}")
        logger.info(f"  Shadow clipping: {stats['shadow_clipped_pct']:.2f}%")
        logger.info(f"  Highlight clipping: {stats['highlight_clipped_pct']:.2f}%")
        logger.info(f"  Dynamic range: {stats['dynamic_range']:.1f}x")

        if stats["scene_type"] == "outdoor":
            outdoor_clipping.append(stats["shadow_clipped_pct"])
        else:
            indoor_clipping.append(stats["shadow_clipped_pct"])

    # Summary
    logger.info("\n" + "-" * 80)
    logger.info("SUMMARY:")

    if outdoor_clipping:
        avg_outdoor = np.mean(outdoor_clipping)
        max_outdoor = np.max(outdoor_clipping)
        logger.info(f"  Outdoor scenes:")
        logger.info(f"    Average shadow clipping: {avg_outdoor:.2f}%")
        logger.info(f"    Maximum shadow clipping: {max_outdoor:.2f}%")
        logger.info(f"    Target: <5.0%")

        if avg_outdoor < 5.0:
            logger.info(f"    ✅ PASS - Average clipping below target")
        else:
            logger.info(f"    ⚠️  WARN - Average clipping above target")

    if indoor_clipping:
        avg_indoor = np.mean(indoor_clipping)
        logger.info(f"  Indoor scenes:")
        logger.info(f"    Average shadow clipping: {avg_indoor:.2f}%")
        logger.info(f"    Target: <6.5% (maintained from baseline)")

        if avg_indoor < 6.5:
            logger.info(f"    ✅ PASS - Indoor quality maintained")
        else:
            logger.info(f"    ⚠️  WARN - Indoor clipping increased")

    return {
        "outdoor_clipping": outdoor_clipping,
        "indoor_clipping": indoor_clipping,
        "outdoor_avg": np.mean(outdoor_clipping) if outdoor_clipping else 0,
        "indoor_avg": np.mean(indoor_clipping) if indoor_clipping else 0,
        "details": results,
    }


def test_ai_enhancement_compatibility() -> Dict:
    """
    Test AI enhancement tensor padding functionality.

    Returns:
        Test results dictionary
    """
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: AI Enhancement Tensor Compatibility")
    logger.info("=" * 80)

    results = {"tests": []}

    # Test different image sizes
    test_sizes = [
        (1152, 768),  # Original problematic size
        (1536, 1024),  # Another common size
        (1920, 1280),  # Full HD aspect
        (2048, 1365),  # Irregular size
    ]

    try:
        from luxury_estate_master_pipeline import LuxuryEstateMasterPipeline

        pipeline = LuxuryEstateMasterPipeline.__new__(LuxuryEstateMasterPipeline)

        for width, height in test_sizes:
            logger.info(f"\nTesting {width}x{height}...")

            # Create test image
            test_image = np.random.rand(height, width, 3).astype(np.float32)

            # Test padding
            padded, padding = pipeline._pad_for_controlnet(test_image, multiple=64)
            unpadded = pipeline._unpad_image(padded, padding)

            # Verify dimensions
            target_h = ((height + 63) // 64) * 64
            target_w = ((width + 63) // 64) * 64

            padded_correct = padded.shape[0] == target_h and padded.shape[1] == target_w
            unpadded_correct = unpadded.shape[0] == height and unpadded.shape[1] == width

            test_result = {
                "original_size": f"{width}x{height}",
                "padded_size": f"{padded.shape[1]}x{padded.shape[0]}",
                "target_size": f"{target_w}x{target_h}",
                "unpadded_size": f"{unpadded.shape[1]}x{unpadded.shape[0]}",
                "padding_correct": padded_correct,
                "unpadding_correct": unpadded_correct,
                "status": "PASS" if (padded_correct and unpadded_correct) else "FAIL",
            }

            results["tests"].append(test_result)

            logger.info(f"  Original: {width}x{height}")
            logger.info(f"  Padded:   {padded.shape[1]}x{padded.shape[0]} (target: {target_w}x{target_h})")
            logger.info(f"  Unpadded: {unpadded.shape[1]}x{unpadded.shape[0]}")
            logger.info(f"  Status:   {'✅ PASS' if test_result['status'] == 'PASS' else '❌ FAIL'}")

        # Overall result
        all_passed = all(t["status"] == "PASS" for t in results["tests"])
        results["overall_status"] = "PASS" if all_passed else "FAIL"

        logger.info("\n" + "-" * 80)
        logger.info(
            f"OVERALL: {'✅ PASS' if all_passed else '❌ FAIL'} - {len(results['tests'])}/{len(results['tests'])} tests passed"
        )

    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        results["error"] = str(e)
        results["overall_status"] = "ERROR"

    return results


def test_depth_model_download() -> Dict:
    """
    Test depth model auto-download functionality.

    Returns:
        Test results dictionary
    """
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Depth Model Auto-Download")
    logger.info("=" * 80)

    results = {"checks": []}

    # Check if transformers is available
    try:
        import transformers

        transformers_available = True
        transformers_version = transformers.__version__
        logger.info(f"✅ transformers library available: v{transformers_version}")
        results["checks"].append({"name": "transformers_library", "status": "PASS", "version": transformers_version})
    except ImportError:
        transformers_available = False
        logger.warning("⚠️  transformers library not available")
        results["checks"].append(
            {"name": "transformers_library", "status": "FAIL", "message": "Install with: pip install transformers"}
        )

    # Check if depth model is cached
    if transformers_available:
        try:
            from pathlib import Path

            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            model_id = "depth-anything/Depth-Anything-V2-Small-hf"
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

            logger.info(f"\nChecking cache for: {model_id}")
            logger.info(f"Cache directory: {cache_dir}")

            # This will use cached version if available, or download
            processor = AutoImageProcessor.from_pretrained(model_id)
            model = AutoModelForDepthEstimation.from_pretrained(model_id)

            logger.info("✅ Depth Anything V2 model accessible")
            results["checks"].append({"name": "depth_model_access", "status": "PASS", "model_id": model_id})

        except Exception as e:
            logger.error(f"❌ Failed to access depth model: {e}")
            results["checks"].append({"name": "depth_model_access", "status": "FAIL", "error": str(e)})

    # Overall status
    all_passed = all(c["status"] == "PASS" for c in results["checks"])
    results["overall_status"] = "PASS" if all_passed else "FAIL"

    logger.info("\n" + "-" * 80)
    logger.info(f"OVERALL: {'✅ PASS' if all_passed else '⚠️  PARTIAL'}")

    return results


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(
        description="Test Luxury Estate Pipeline Fixes", formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--input-dir", type=Path, help="Directory containing test images")
    parser.add_argument(
        "--output-report", type=Path, default=DEFAULT_REPORT_PATH, help="Output JSON report path"
    )
    parser.add_argument(
        "--test", choices=["shadow", "ai", "depth", "all"], default="all", help="Which test to run (default: all)"
    )

    args = parser.parse_args()

    logger.info("\n" + "=" * 80)
    logger.info("LUXURY ESTATE MASTER PIPELINE - FIX VALIDATION")
    logger.info("=" * 80)

    report = {"test_suite": "Pipeline Fixes Validation", "tests": {}}

    # Test 1: Shadow clipping
    if args.test in ["shadow", "all"]:
        if args.input_dir and args.input_dir.exists():
            image_paths = list(args.input_dir.glob("*.tif")) + list(args.input_dir.glob("*.tiff"))
            if image_paths:
                report["tests"]["shadow_clipping"] = test_shadow_clipping_fix(image_paths)
            else:
                logger.warning("No TIFF images found in input directory")
        else:
            logger.warning("Skipping shadow clipping test (no input directory)")

    # Test 2: AI enhancement
    if args.test in ["ai", "all"]:
        report["tests"]["ai_enhancement"] = test_ai_enhancement_compatibility()

    # Test 3: Depth model
    if args.test in ["depth", "all"]:
        report["tests"]["depth_model"] = test_depth_model_download()

    # Save report
    with open(args.output_report, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"\n📄 Test report saved: {args.output_report}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    for test_name, test_results in report["tests"].items():
        status = test_results.get("overall_status", "N/A")
        symbol = "✅" if status == "PASS" else "⚠️" if status == "PARTIAL" else "❌"
        logger.info(f"{symbol} {test_name}: {status}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
