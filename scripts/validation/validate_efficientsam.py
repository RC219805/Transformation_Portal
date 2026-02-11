#!/usr/bin/env python3
"""Manual validation script for EfficientSAM segmentation backend.

This script demonstrates the EfficientSAM backend with real images
and validates the integration with Materials V3.

Usage:
    python validate_efficientsam.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from transformation_portal.lux_depth_v3.config import EnhanceConfig
from transformation_portal.lux_depth_v3.segmentation_backend import segment_materials

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_test_image() -> np.ndarray:
    """Create a synthetic test image with clear material signatures."""
    # Create 512×512 RGB image
    img = np.zeros((512, 512, 3), dtype=np.uint8)

    # Region 1: Blue water (top-left)
    img[0:256, 0:256] = [50, 100, 200]

    # Region 2: Green foliage (top-right)
    img[0:256, 256:512] = [60, 180, 90]

    # Region 3: Gray stone (bottom-left)
    img[256:512, 0:256] = [120, 125, 120]

    # Region 4: Bright glass (bottom-right)
    img[256:512, 256:512] = [190, 200, 220]

    return img


def test_stub_backend():
    """Test stub backend (should return empty masks)."""
    logger.info("=" * 60)
    logger.info("Testing Stub Backend")
    logger.info("=" * 60)

    config = EnhanceConfig(
        enable_material_segmentation=True,
        material_segmentation_backend="stub",
    )

    image = create_test_image()
    masks = segment_materials(image, config)

    logger.info(f"✅ Stub backend returned {len(masks)} masks (expected 0)")
    assert len(masks) == 0, "Stub backend should return empty dict"


def test_efficientsam_backend():
    """Test EfficientSAM backend (should detect materials)."""
    logger.info("=" * 60)
    logger.info("Testing EfficientSAM Backend")
    logger.info("=" * 60)

    config = EnhanceConfig(
        enable_material_segmentation=True,
        material_segmentation_backend="efficientsam",
        depth_device="auto",
    )

    image = create_test_image()
    masks = segment_materials(image, config)

    logger.info(f"✅ EfficientSAM backend returned {len(masks)} masks")
    logger.info(f"Detected materials: {list(masks.keys())}")

    for material, mask in masks.items():
        coverage_px = mask.sum()
        coverage_pct = (coverage_px / (512 * 512)) * 100
        mean_conf = mask.mean()

        logger.info(f"  - {material}: {coverage_px:.0f} px ({coverage_pct:.1f}%), " f"mean_conf={mean_conf:.2f}")

        # Validate mask properties
        assert mask.shape == (512, 512), f"Mask shape mismatch for {material}"
        assert mask.dtype == np.float32, f"Mask dtype mismatch for {material}"
        assert 0.0 <= mask.min() <= mask.max() <= 1.0, f"Mask values out of range for {material}"

    # Should detect multiple materials
    assert len(masks) > 0, "EfficientSAM should detect at least one material"


def test_device_selection():
    """Test device selection (MPS/CUDA/CPU)."""
    logger.info("=" * 60)
    logger.info("Testing Device Selection")
    logger.info("=" * 60)

    import torch

    # Test auto-detection
    config_auto = EnhanceConfig(
        enable_material_segmentation=True,
        material_segmentation_backend="efficientsam",
        depth_device="auto",
    )

    image = create_test_image()
    masks_auto = segment_materials(image, config_auto)

    logger.info(f"✅ Auto device detected and processed {len(masks_auto)} materials")

    # Test explicit CPU
    config_cpu = EnhanceConfig(
        enable_material_segmentation=True,
        material_segmentation_backend="efficientsam",
        depth_device="cpu",
    )

    masks_cpu = segment_materials(image, config_cpu)
    logger.info(f"✅ CPU device processed {len(masks_cpu)} materials")

    # Test MPS if available
    if torch.backends.mps.is_available():
        config_mps = EnhanceConfig(
            enable_material_segmentation=True,
            material_segmentation_backend="efficientsam",
            depth_device="mps",
        )

        masks_mps = segment_materials(image, config_mps)
        logger.info(f"✅ MPS device processed {len(masks_mps)} materials")


def test_fallback_behavior():
    """Test fallback to stub when backend fails."""
    logger.info("=" * 60)
    logger.info("Testing Fallback Behavior")
    logger.info("=" * 60)

    # Test with unknown backend (should fall back to stub)
    config = EnhanceConfig(
        enable_material_segmentation=True,
        material_segmentation_backend="unknown_backend",
        strict_backend=False,
    )

    image = create_test_image()
    masks = segment_materials(image, config)

    logger.info(f"✅ Unknown backend fell back to stub: {len(masks)} masks (expected 0)")
    assert len(masks) == 0, "Should fall back to stub for unknown backend"


def test_strict_mode():
    """Test strict mode behavior."""
    logger.info("=" * 60)
    logger.info("Testing Strict Mode")
    logger.info("=" * 60)

    # Test with valid backend (should work)
    config_valid = EnhanceConfig(
        enable_material_segmentation=True,
        material_segmentation_backend="efficientsam",
        strict_backend=True,
    )

    image = create_test_image()
    masks = segment_materials(image, config_valid)

    logger.info(f"✅ Strict mode with valid backend: {len(masks)} materials detected")


def save_visualization(image: np.ndarray, masks: dict[str, np.ndarray], output_path: Path):
    """Save visualization of detected materials."""
    from PIL import ImageDraw, ImageFont

    # Create output image (original + mask overlays)
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil, "RGBA")

    # Color map for materials
    colors = {
        "glass": (100, 150, 255, 100),  # Light blue
        "water": (0, 100, 255, 100),  # Blue
        "foliage": (50, 200, 50, 100),  # Green
        "stone": (150, 150, 150, 100),  # Gray
    }

    # Overlay masks
    for material, mask in masks.items():
        color = colors.get(material, (255, 255, 255, 100))

        # Create mask overlay
        mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        mask_rgba[..., :3] = color[:3]
        mask_rgba[..., 3] = (mask * color[3]).astype(np.uint8)

        overlay = Image.fromarray(mask_rgba, "RGBA")
        img_pil.paste(overlay, (0, 0), overlay)

        # Add label
        coverage = mask.sum()
        label = f"{material}: {coverage:.0f} px"
        # Note: Using default font since we don't have a specific font file
        draw.text((10, 10 + len(masks) * 20), label, fill=color[:3] + (255,))

    img_pil.save(output_path)
    logger.info(f"✅ Saved visualization to {output_path}")


def main():
    """Run all validation tests."""
    logger.info("=" * 60)
    logger.info("EfficientSAM Segmentation Backend Validation")
    logger.info("=" * 60)

    try:
        # Test 1: Stub backend
        test_stub_backend()

        # Test 2: EfficientSAM backend
        test_efficientsam_backend()

        # Test 3: Device selection
        test_device_selection()

        # Test 4: Fallback behavior
        test_fallback_behavior()

        # Test 5: Strict mode
        test_strict_mode()

        # Bonus: Save visualization
        output_dir = Path("output_segmentation_validation")
        output_dir.mkdir(exist_ok=True)

        config = EnhanceConfig(
            enable_material_segmentation=True,
            material_segmentation_backend="efficientsam",
        )
        image = create_test_image()
        masks = segment_materials(image, config)

        save_visualization(image, masks, output_dir / "test_segmentation.png")

        logger.info("=" * 60)
        logger.info("✅ ALL VALIDATION TESTS PASSED")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"❌ Validation failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
