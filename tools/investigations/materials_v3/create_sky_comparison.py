#!/usr/bin/env python3
"""Create side-by-side comparison of before/after sky fix."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def create_comparison(before_path, after_path, output_path, crop_sky_region=True):
    """Create side-by-side comparison focusing on sky region."""
    print(f"Loading images...")
    before = Image.open(before_path)
    after = Image.open(after_path)

    print(f"  Before: {before.size} {before.mode}")
    print(f"  After: {after.size} {after.mode}")

    # Convert to RGB if needed
    if before.mode != "RGB":
        before = before.convert("RGB")
    if after.mode != "RGB":
        after = after.convert("RGB")

    # Crop to sky region if requested (top 40% of image)
    if crop_sky_region:
        w, h = before.size
        sky_height = int(h * 0.4)
        before = before.crop((0, 0, w, sky_height))
        after = after.crop((0, 0, w, sky_height))
        print(f"  Cropped to sky region: {before.size}")

    # Downscale for easier viewing (4K -> 2K)
    scale = 0.5
    new_size = (int(before.width * scale), int(before.height * scale))
    before = before.resize(new_size, Image.Resampling.LANCZOS)
    after = after.resize(new_size, Image.Resampling.LANCZOS)
    print(f"  Downscaled to: {before.size}")

    # Create side-by-side comparison
    comparison = Image.new("RGB", (before.width * 2 + 20, before.height + 60))

    # Fill background
    comparison.paste((50, 50, 50), (0, 0, comparison.width, comparison.height))

    # Paste images with labels
    comparison.paste(before, (10, 50))
    comparison.paste(after, (before.width + 10, 50))

    # Add labels (using PIL's default font - basic but functional)
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(comparison)

    # Try to use a better font if available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except Exception:
        font = ImageFont.load_default()

    # Labels
    draw.text((10, 10), "BEFORE FIX (Sky Degraded)", fill=(255, 100, 100), font=font)
    draw.text((before.width + 10, 10), "AFTER FIX (Sky Corrected)", fill=(100, 255, 100), font=font)

    # Save
    comparison.save(output_path, quality=95)
    print(f"\n✓ Comparison saved to: {output_path}")
    return comparison


def analyze_sky_brightness(image_path, region_name):
    """Analyze sky region brightness."""
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Crop to sky (top 30%)
    w, h = img.size
    sky = img.crop((0, 0, w, int(h * 0.3)))

    # Convert to numpy
    sky_arr = np.array(sky).astype(np.float32) / 255.0

    print(f"\n{region_name} Sky Statistics:")
    print(f"  RGB Mean: [{sky_arr[:,:,0].mean():.4f}, {sky_arr[:,:,1].mean():.4f}, {sky_arr[:,:,2].mean():.4f}]")
    print(f"  Overall brightness: {sky_arr.mean():.4f}")
    print(f"  Std dev: {sky_arr.std():.4f}")

    return sky_arr.mean()


def main():
    """Run comparison analysis."""
    parser = argparse.ArgumentParser(
        description="Generate side-by-side visual comparisons for sky processing validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default hardcoded paths (original investigation)
  python create_sky_comparison.py

  # Compare custom before/after images
  python create_sky_comparison.py \\
    --input input_images/test_sky.jpg \\
    --output comparison_sky.png \\
    --amplify 10

  # Process specific before/after pair
  python create_sky_comparison.py \\
    --before output_old/aerial.tiff \\
    --after output_new/aerial.tiff \\
    --output comparison.jpg
        """,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Single input image (will be used as 'before', requires processing logic)",
    )
    parser.add_argument(
        "--before",
        type=Path,
        default=Path("output_apex_v2_luxury/V2_750Picacho_Aerial.tiff"),
        help="Before image path (default: original investigation image)",
    )
    parser.add_argument(
        "--after",
        type=Path,
        default=Path("test_sky_fix/V2_750Picacho_Aerial_FIXED.tiff"),
        help="After image path (default: original investigation fixed image)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test_sky_fix/sky_fix_comparison.jpg"),
        help="Output comparison image path (default: test_sky_fix/sky_fix_comparison.jpg)",
    )
    parser.add_argument(
        "--amplify",
        type=int,
        default=10,
        help="Amplification factor for difference visualization (default: 10)",
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Disable cropping to sky region (show full image)",
    )

    args = parser.parse_args()

    # Handle --input mode (single image, not yet implemented - would need processing)
    if args.input:
        print("⚠️  --input mode requires processing logic (not yet implemented)")
        print("    Use --before and --after to compare existing images")
        return

    before = args.before
    after = args.after
    comparison_out = args.output
    crop_sky_region = not args.no_crop

    print("=" * 60)
    print("Sky Degradation Fix - Visual Comparison")
    print("=" * 60)

    # Analyze brightness
    brightness_before = analyze_sky_brightness(before, "BEFORE FIX")
    brightness_after = analyze_sky_brightness(after, "AFTER FIX")

    change_pct = ((brightness_after - brightness_before) / brightness_before) * 100
    print(f"\nBrightness change: {change_pct:+.1f}%")

    if change_pct < -2:
        print("✓ Sky is now DARKER (compressed) - Fix working as expected!")
    elif change_pct > 2:
        print("⚠ Sky is now BRIGHTER - Issue may persist")
    else:
        print("≈ Sky brightness similar - Check visual quality")

    # Create comparison
    print(f"\n{'='*60}")
    print("Creating visual comparison...")
    print(f"{'='*60}")

    comparison_out.parent.mkdir(parents=True, exist_ok=True)
    create_comparison(before, after, comparison_out, crop_sky_region=crop_sky_region)

    print(f"\n{'='*60}")
    print("Review the comparison image to verify:")
    print("  1. Sky should appear more subtle/compressed")
    print("  2. Sky color should be more neutral")
    print("  3. No unnatural gradients or banding")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
