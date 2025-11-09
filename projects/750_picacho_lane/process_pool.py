#!/usr/bin/env python3
"""
750 Picacho Pool - Luxury Real Estate Rendering Pipeline
Context-aware processing with Material Response Technology
"""

import sys
from pathlib import Path
import json
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transformation_portal.utils.image_utils import load_image, save_image
from transformation_portal.utils.error_handling import safe_execute
import numpy as np
from PIL import Image

def load_exr_to_array(exr_path: Path) -> np.ndarray:
    """Load 16-bit EXR and convert to working format."""
    print(f"📂 Loading EXR: {exr_path.name}")

    # Try OpenEXR first (more reliable for EXR files)
    try:
        import OpenEXR
        import Imath

        exr_file = OpenEXR.InputFile(str(exr_path))
        header = exr_file.header()
        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1

        print(f"  • Resolution: {width} x {height}")

        # Read RGB channels
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        channels = ['R', 'G', 'B']
        arrays = []

        for channel in channels:
            channel_str = exr_file.channel(channel, FLOAT)
            channel_array = np.frombuffer(channel_str, dtype=np.float32)
            channel_array = channel_array.reshape(height, width)
            arrays.append(channel_array)

        img_linear = np.stack(arrays, axis=-1)

        print(f"  • Shape: {img_linear.shape}")
        print(f"  • Dtype: {img_linear.dtype}")
        print(f"  • Range: [{img_linear.min():.3f}, {img_linear.max():.3f}]")

        return img_linear

    except ImportError:
        print("  ⚠️  OpenEXR not available, trying imageio")
        try:
            import imageio.v3 as iio
            # Load EXR (linear color space)
            img_linear = iio.imread(exr_path)

            print(f"  • Shape: {img_linear.shape}")
            print(f"  • Dtype: {img_linear.dtype}")
            print(f"  • Range: [{img_linear.min():.3f}, {img_linear.max():.3f}]")

            # Convert to float32 for processing
            if img_linear.dtype != np.float32:
                img_linear = img_linear.astype(np.float32)

            return img_linear

        except Exception as e:
            raise ImportError(f"Could not load EXR file: {e}")


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Convert linear RGB to sRGB color space."""
    print("  🎨 Converting linear → sRGB")

    # Clip to valid range
    linear = np.clip(linear, 0, None)

    # Apply sRGB gamma curve
    srgb = np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055
    )

    return srgb


def apply_pool_enhancement(img: np.ndarray) -> np.ndarray:
    """Apply pool-specific enhancements."""
    print("💧 Applying pool water enhancement")

    # Detect water regions (typically blue-dominant)
    # This is a simple heuristic - could be enhanced with segmentation
    blue_channel = img[:, :, 2]
    red_channel = img[:, :, 0]
    green_channel = img[:, :, 1]

    # Water mask: blue > (red + green) / 2
    water_mask = blue_channel > (red_channel + green_channel) / 2.0

    # Enhance water clarity and color depth
    enhanced = img.copy()
    if water_mask.any():
        # Boost blue saturation in water regions
        water_mult = 1.25
        enhanced[:, :, 2] = np.where(water_mask, enhanced[:, :, 2] * water_mult, enhanced[:, :, 2])

        # Slight reduction in red for cooler water tone
        enhanced[:, :, 0] = np.where(water_mask, enhanced[:, :, 0] * 0.92, enhanced[:, :, 0])

        print(f"  • Water regions enhanced: ~{water_mask.sum() / water_mask.size * 100:.1f}% of image")

    return enhanced


def apply_color_adjustments(img: np.ndarray, config: dict) -> np.ndarray:
    """Apply color grading adjustments."""
    print("🎨 Applying color adjustments")

    adj = config.get('adjustments', {})

    # Contrast
    contrast = adj.get('contrast', 1.0)
    if contrast != 1.0:
        img = np.clip((img - 0.5) * contrast + 0.5, 0, None)
        print(f"  • Contrast: {contrast}")

    # Saturation
    saturation = adj.get('saturation', 1.0)
    if saturation != 1.0:
        gray = np.mean(img, axis=2, keepdims=True)
        img = np.clip(gray + (img - gray) * saturation, 0, None)
        print(f"  • Saturation: {saturation}")

    # Temperature shift (warm/cool)
    temp = adj.get('temperature', 0)
    if temp != 0:
        temp_factor = temp / 100.0  # -100 to +100 scale
        img[:, :, 0] = np.clip(img[:, :, 0] * (1 + temp_factor * 0.1), 0, None)  # Red
        img[:, :, 2] = np.clip(img[:, :, 2] * (1 - temp_factor * 0.1), 0, None)  # Blue
        print(f"  • Temperature: {temp:+d}")

    return img


def save_outputs(img: np.ndarray, output_dir: Path, base_name: str):
    """Save processed image in multiple formats."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Saving outputs to: {output_dir}")

    # Clip and convert to uint16 for TIFF
    img_clipped = np.clip(img, 0, 1)
    img_uint16 = (img_clipped * 65535).astype(np.uint16)

    # Convert to PIL Image
    img_pil = Image.fromarray(img_uint16, mode='RGB')

    # Save 16-bit TIFF
    tiff_path = output_dir / f"{base_name}_Master.tif"
    img_pil.save(tiff_path, compression='lzw')
    print(f"  ✅ Master TIFF (16-bit): {tiff_path.name} ({tiff_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Save high-quality JPEG for web/preview
    img_uint8 = (img_clipped * 255).astype(np.uint8)
    img_pil_8bit = Image.fromarray(img_uint8, mode='RGB')

    jpg_path = output_dir / f"{base_name}_Web.jpg"
    img_pil_8bit.save(jpg_path, quality=95, optimize=True)
    print(f"  ✅ Web JPEG: {jpg_path.name} ({jpg_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Save thumbnail
    thumb_size = (1200, int(1200 * img_uint8.shape[0] / img_uint8.shape[1]))
    img_thumb = img_pil_8bit.resize(thumb_size, Image.Resampling.LANCZOS)
    thumb_path = output_dir / f"{base_name}_Thumbnail.jpg"
    img_thumb.save(thumb_path, quality=90, optimize=True)
    print(f"  ✅ Thumbnail: {thumb_path.name} ({thumb_path.stat().st_size / 1024:.0f} KB)")


def main():
    print("\n" + "="*70)
    print("🏊 750 PICACHO POOL - LUXURY RENDERING PIPELINE")
    print("="*70 + "\n")

    start_time = time.time()

    # Load configuration
    config_path = Path(__file__).parent / "presets" / "pool_preset.json"
    print(f"📋 Loading configuration: {config_path.name}")

    with open(config_path) as f:
        config = json.load(f)

    # Input/output paths
    input_path = Path(config['input']['file'])
    output_dir = Path(config['output']['directory'])
    base_name = config['output']['base_name']

    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_path}")
        return 1

    print(f"  • Scene: {config['metadata']['scene']}")
    print(f"  • Location: {config['metadata']['location']}")
    print(f"  • Input: {input_path.name} ({input_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Stage 1: Load EXR
    print(f"\n{'─'*70}")
    print("STAGE 1: Load & Convert EXR → Working Format")
    print(f"{'─'*70}")

    img_linear = load_exr_to_array(input_path)

    # Convert to sRGB
    img_srgb = linear_to_srgb(img_linear)

    # Stage 2: Material Response (simplified - pool water enhancement)
    print(f"\n{'─'*70}")
    print("STAGE 2: Material Response - Water Enhancement")
    print(f"{'─'*70}")

    if config['processing_stages']['3_material_response']['enabled']:
        img_enhanced = apply_pool_enhancement(img_srgb)
    else:
        img_enhanced = img_srgb

    # Stage 3: Color Grading
    print(f"\n{'─'*70}")
    print("STAGE 3: Color Grading - Santa Barbara Aesthetic")
    print(f"{'─'*70}")

    if config['processing_stages']['4_color_grading']['enabled']:
        img_graded = apply_color_adjustments(
            img_enhanced,
            config['processing_stages']['4_color_grading']
        )
    else:
        img_graded = img_enhanced

    # Stage 4: Save outputs
    print(f"\n{'─'*70}")
    print("STAGE 4: Save Deliverables")
    print(f"{'─'*70}")

    save_outputs(img_graded, output_dir, base_name)

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"✅ PROCESSING COMPLETE - {elapsed:.1f} seconds")
    print(f"{'='*70}")
    print(f"\n📦 Deliverables ready in: {output_dir}")
    print("\n🎯 Next steps:")
    print("  1. Review Master TIFF in photo editor")
    print("  2. Apply additional refinements if needed")
    print("  3. Run AI enhancement stage (optional)")
    print("  4. Deliver to client\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
