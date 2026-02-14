#!/usr/bin/env python3
"""
Analyze Great Room sky coloration comparing input vs output.
"""
import json

import numpy as np
from PIL import Image


def analyze_sky_region(image_path, name):
    """Analyze sky region in an image."""
    img = Image.open(image_path)
    arr = np.array(img)

    # Great Room likely has sky in upper portion
    # Sample top 20% of image as sky region
    height, width = arr.shape[:2]
    sky_region = arr[: int(height * 0.2), :, :]

    # Convert to float for analysis
    if arr.dtype == np.uint16:
        sky_float = sky_region.astype(np.float32) / 65535.0
    elif arr.dtype == np.uint8:
        sky_float = sky_region.astype(np.float32) / 255.0
    else:
        sky_float = sky_region.astype(np.float32)

    # Calculate statistics
    r, g, b = sky_float[:, :, 0], sky_float[:, :, 1], sky_float[:, :, 2]

    stats = {
        "name": name,
        "dtype": str(arr.dtype),
        "shape": arr.shape,
        "sky_region_shape": sky_region.shape,
        "sky_region_pixels": sky_region.shape[0] * sky_region.shape[1],
        # RGB channels
        "red_mean": float(r.mean()),
        "green_mean": float(g.mean()),
        "blue_mean": float(b.mean()),
        "red_std": float(r.std()),
        "green_std": float(g.std()),
        "blue_std": float(b.std()),
        "red_min": float(r.min()),
        "green_min": float(g.min()),
        "blue_min": float(b.min()),
        "red_max": float(r.max()),
        "green_max": float(g.max()),
        "blue_max": float(b.max()),
        # Color metrics
        "brightness": float((r.mean() + g.mean() + b.mean()) / 3),
        "saturation": float(np.std([r.mean(), g.mean(), b.mean()])),
    }

    return stats, sky_float


def compare_sky(input_stats, output_stats):
    """Compare input vs output sky statistics."""
    comparison = {
        "brightness_change": output_stats["brightness"] - input_stats["brightness"],
        "brightness_change_pct": ((output_stats["brightness"] - input_stats["brightness"]) / input_stats["brightness"]) * 100,
        "saturation_change": output_stats["saturation"] - input_stats["saturation"],
        "saturation_change_pct": ((output_stats["saturation"] - input_stats["saturation"]) / input_stats["saturation"]) * 100,
        "red_change": output_stats["red_mean"] - input_stats["red_mean"],
        "red_change_pct": ((output_stats["red_mean"] - input_stats["red_mean"]) / input_stats["red_mean"]) * 100,
        "green_change": output_stats["green_mean"] - input_stats["green_mean"],
        "green_change_pct": ((output_stats["green_mean"] - input_stats["green_mean"]) / input_stats["green_mean"]) * 100,
        "blue_change": output_stats["blue_mean"] - input_stats["blue_mean"],
        "blue_change_pct": ((output_stats["blue_mean"] - input_stats["blue_mean"]) / input_stats["blue_mean"]) * 100,
    }

    return comparison


def main():
    input_path = (
        "/Users/rc/Projects/Transformation_Portal/input_images/750Picacho_16-bit_TIFFs/750Picacho_GreatRoom_master16.tif"
    )
    output_path = "/Users/rc/Projects/Transformation_Portal/output_bugfix_validation_final/v2/750Picacho_GreatRoom_master16_tif_d73fda6e_materials_v3_enhanced.tif"

    print("=" * 80)
    print("GREAT ROOM SKY COLORATION ANALYSIS")
    print("=" * 80)

    # Analyze input
    print("\n📥 Analyzing INPUT sky region...")
    input_stats, input_sky = analyze_sky_region(input_path, "INPUT")

    print(f"   Sky region: {input_stats['sky_region_shape']} ({input_stats['sky_region_pixels']:,} pixels)")
    print(f"   Dtype: {input_stats['dtype']}")
    print(
        f"   RGB means: R={input_stats['red_mean']:.4f}, G={input_stats['green_mean']:.4f}, B={input_stats['blue_mean']:.4f}"
    )
    print(f"   Brightness: {input_stats['brightness']:.4f}")
    print(f"   Saturation: {input_stats['saturation']:.4f}")

    # Analyze output
    print("\n📤 Analyzing OUTPUT sky region...")
    output_stats, output_sky = analyze_sky_region(output_path, "OUTPUT")

    print(f"   Sky region: {output_stats['sky_region_shape']} ({output_stats['sky_region_pixels']:,} pixels)")
    print(f"   Dtype: {output_stats['dtype']}")
    print(
        f"   RGB means: R={output_stats['red_mean']:.4f}, G={output_stats['green_mean']:.4f}, B={output_stats['blue_mean']:.4f}"
    )
    print(f"   Brightness: {output_stats['brightness']:.4f}")
    print(f"   Saturation: {output_stats['saturation']:.4f}")

    # Compare
    print("\n🔍 COMPARISON (Output vs Input):")
    comp = compare_sky(input_stats, output_stats)

    print(f"\n   Brightness:")
    print(f"      Change: {comp['brightness_change']:+.6f} ({comp['brightness_change_pct']:+.2f}%)")

    print(f"\n   Saturation:")
    print(f"      Change: {comp['saturation_change']:+.6f} ({comp['saturation_change_pct']:+.2f}%)")

    print(f"\n   RGB Channel Changes:")
    print(f"      Red:   {comp['red_change']:+.6f} ({comp['red_change_pct']:+.2f}%)")
    print(f"      Green: {comp['green_change']:+.6f} ({comp['green_change_pct']:+.2f}%)")
    print(f"      Blue:  {comp['blue_change']:+.6f} ({comp['blue_change_pct']:+.2f}%)")

    # Check for specific sky degradation patterns
    print("\n⚠️  SKY DEGRADATION INDICATORS:")

    warnings = []

    # Check for desaturation (common degradation)
    if comp["saturation_change_pct"] < -5:
        warnings.append(f"DESATURATION: {comp['saturation_change_pct']:.1f}% saturation loss")

    # Check for blue channel loss (sky-specific)
    if comp["blue_change_pct"] < -5:
        warnings.append(f"BLUE LOSS: {comp['blue_change_pct']:.1f}% blue channel degradation")

    # Check for brightness loss
    if comp["brightness_change_pct"] < -5:
        warnings.append(f"DARKENING: {comp['brightness_change_pct']:.1f}% brightness loss")

    # Check for color cast (red/green shift in sky)
    if abs(comp["red_change_pct"]) > 10 or abs(comp["green_change_pct"]) > 10:
        warnings.append(
            f"COLOR CAST: Unusual red/green shift (R: {comp['red_change_pct']:.1f}%, G: {comp['green_change_pct']:.1f}%)"
        )

    if warnings:
        for w in warnings:
            print(f"   🔴 {w}")
    else:
        print("   ✅ No significant degradation detected")
        print(f"   • Brightness change: {comp['brightness_change_pct']:+.2f}% (< 5% threshold)")
        print(f"   • Saturation change: {comp['saturation_change_pct']:+.2f}% (< 5% threshold)")
        print(f"   • Blue channel change: {comp['blue_change_pct']:+.2f}% (< 5% threshold)")

    # Save full analysis
    analysis = {"input": input_stats, "output": output_stats, "comparison": comp, "warnings": warnings}

    with open("greatroom_sky_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    print(f"\n💾 Full analysis saved to: greatroom_sky_analysis.json")
    print("=" * 80)

    # Create visual comparison
    print("\n🎨 Creating visual comparison...")

    # Stack input and output sky regions side by side
    combined = np.concatenate([input_sky, output_sky], axis=1)

    # Convert to uint8 for display
    combined_uint8 = (np.clip(combined, 0, 1) * 255).astype(np.uint8)

    img = Image.fromarray(combined_uint8)
    output_img_path = "greatroom_sky_comparison.jpg"
    img.save(output_img_path, quality=95)

    print(f"   Saved comparison image to: {output_img_path}")
    print(f"   (Left half = INPUT, Right half = OUTPUT)")


if __name__ == "__main__":
    main()
