#!/usr/bin/env python3
"""
Phase 2 Visual Comparison: V2-Small vs V2-Large
================================================

Generates side-by-side visual comparisons of depth maps from both models.

Author: Transformation Portal Specialist
Date: November 10, 2025
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_depth_comparison(
    image_path: Path,
    output_dir: Path,
    model_variants: list = ['small', 'large']
):
    """Create side-by-side comparison of depth maps from different models."""

    from depth_anything_v2 import DepthAnythingV2Model, ModelVariant, ModelBackend

    logger.info(f"\n{'='*70}")
    logger.info(f"Processing: {image_path.name}")
    logger.info(f"{'='*70}")

    # Load input image
    img = Image.open(image_path).convert('RGB')
    _img_array = np.array(img)  # noqa: F841
    logger.info(f"✓ Loaded image: {img.size[0]}x{img.size[1]}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Map variant names
    variant_map = {
        'small': ModelVariant.SMALL,
        'base': ModelVariant.BASE,
        'large': ModelVariant.LARGE,
    }

    results = {}
    depth_maps = {}

    # Process with each model
    for variant_name in model_variants:
        logger.info(f"\nProcessing with V2-{variant_name.upper()}...")

        variant = variant_map[variant_name]

        # Initialize model
        model = DepthAnythingV2Model(
            variant=variant,
            backend=ModelBackend.PYTORCH_MPS,
            device='mps',
            precision='fp16'
        )

        # Estimate depth
        start = time.time()
        result = model.estimate_depth(img)
        elapsed = (time.time() - start) * 1000

        depth_map = result['depth']
        depth_maps[variant_name] = depth_map

        logger.info(f"  ✓ Inference: {elapsed:.1f}ms")
        logger.info(f"  ✓ Depth range: [{depth_map.min():.4f}, {depth_map.max():.4f}]")

        results[variant_name] = {
            'inference_ms': elapsed,
            'depth_shape': depth_map.shape,
            'depth_range': [float(depth_map.min()), float(depth_map.max())]
        }

        # Save individual depth map
        depth_vis = (depth_map * 255).astype(np.uint8)
        depth_img = Image.fromarray(depth_vis, mode='L')

        output_path = output_dir / f"{image_path.stem}_{variant_name}_depth.png"
        depth_img.save(output_path)
        logger.info(f"  ✓ Saved: {output_path.name}")

    # Create side-by-side comparison
    if len(depth_maps) >= 2:
        create_comparison_grid(
            image_path,
            img,
            depth_maps,
            results,
            output_dir
        )

    return results


def create_comparison_grid(
    image_path: Path,
    original_img: Image.Image,
    depth_maps: dict,
    results: dict,
    output_dir: Path
):
    """Create a grid comparing original image with depth maps."""

    logger.info("\nCreating comparison grid...")

    # Get dimensions
    w, h = original_img.size

    # Create grid: Original | V2-Small | V2-Large
    grid_w = w * 3
    grid_h = h + 100  # Extra space for labels

    grid = Image.new('RGB', (grid_w, grid_h), color='white')
    draw = ImageDraw.Draw(grid)

    # Try to load a font, fallback to default
    try:
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        font_info = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except BaseException:
        font_title = ImageFont.load_default()
        font_info = ImageFont.load_default()

    # Paste original image
    grid.paste(original_img, (0, 80))
    draw.text((w // 2 - 50, 20), "Original", fill='black', font=font_title)

    # Paste depth maps
    x_offset = w
    for variant_name in ['small', 'large']:
        if variant_name in depth_maps:
            depth_map = depth_maps[variant_name]
            depth_vis = (depth_map * 255).astype(np.uint8)
            depth_img = Image.fromarray(depth_vis, mode='L').convert('RGB')

            grid.paste(depth_img, (x_offset, 80))

            # Add title
            title = f"V2-{variant_name.title()}"
            draw.text((x_offset + w // 2 - 50, 20), title, fill='black', font=font_title)

            # Add performance info
            if variant_name in results:
                info_text = f"{results[variant_name]['inference_ms']:.1f}ms"
                draw.text((x_offset + 10, 50), info_text, fill='green', font=font_info)

            x_offset += w

    # Save comparison
    output_path = output_dir / f"{image_path.stem}_comparison.jpg"
    grid.save(output_path, quality=95)
    logger.info(f"✓ Saved comparison: {output_path.name}")


def main():
    """Run Phase 2 visual comparison."""

    logger.info("=" * 70)
    logger.info("PHASE 2 VISUAL COMPARISON: V2-SMALL vs V2-LARGE")
    logger.info("=" * 70)

    # Find test image
    input_dir = Path("input_images/Temporary_Holding_Files")
    test_images = list(input_dir.glob("*.png"))[:1]  # Just test with 1 image

    if not test_images:
        logger.error("No test images found in input_images/Temporary_Holding_Files")
        return 1

    # Create output directory
    output_dir = Path("output_phase2_comparison")

    # Process images
    all_results = {}
    for img_path in test_images:
        results = create_depth_comparison(img_path, output_dir)
        all_results[img_path.name] = results

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2 VISUAL COMPARISON COMPLETE")
    logger.info("=" * 70)
    logger.info(f"\nProcessed {len(test_images)} image(s)")
    logger.info(f"Output saved to: {output_dir}")
    logger.info("\nNext: Review visual comparisons to assess quality improvement")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
