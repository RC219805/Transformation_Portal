#!/usr/bin/env python3
"""
Quick image info display for processed luxury pool image.
Generates a side-by-side comparison summary.
"""

from pathlib import Path
from PIL import Image
import numpy as np

def analyze_image(path):
    """Analyze image properties."""
    img = Image.open(path)
    arr = np.array(img)

    return {
        'path': path.name,
        'size': f"{img.width}x{img.height}",
        'mode': img.mode,
        'mean_r': np.mean(arr[:,:,0]),
        'mean_g': np.mean(arr[:,:,1]),
        'mean_b': np.mean(arr[:,:,2]),
        'mean_overall': np.mean(arr),
        'std': np.std(arr),
        'min': np.min(arr),
        'max': np.max(arr),
    }

if __name__ == "__main__":
    repo = Path(__file__).parent

    original = repo / "input_images" / "V2_750Picacho_Pool.tiff"
    enhanced = repo / "output_images" / "V2_750Picacho_Pool_Luxury_Enhanced.jpg"

    if not enhanced.exists():
        enhanced = repo / "output_images" / "V2_750Picacho_Pool_Luxury_Enhanced.tiff"

    print("=" * 80)
    print("750 PICACHO POOL - LUXURY ENHANCEMENT ANALYSIS")
    print("=" * 80)
    print()

    if original.exists():
        orig_stats = analyze_image(original)
        print(f"📷 ORIGINAL IMAGE: {orig_stats['path']}")
        print(f"   Dimensions: {orig_stats['size']}")
        print(f"   Average Brightness: {orig_stats['mean_overall']:.1f}/255")
        print(f"   RGB Averages: R={orig_stats['mean_r']:.1f}, G={orig_stats['mean_g']:.1f}, B={orig_stats['mean_b']:.1f}")
        print(f"   Dynamic Range: {orig_stats['min']} - {orig_stats['max']}")
        print(f"   Contrast (StdDev): {orig_stats['std']:.2f}")
        print()

    if enhanced.exists():
        enh_stats = analyze_image(enhanced)
        print(f"✨ ENHANCED IMAGE: {enh_stats['path']}")
        print(f"   Dimensions: {enh_stats['size']}")
        print(f"   Average Brightness: {enh_stats['mean_overall']:.1f}/255")
        print(f"   RGB Averages: R={enh_stats['mean_r']:.1f}, G={enh_stats['mean_g']:.1f}, B={enh_stats['mean_b']:.1f}")
        print(f"   Dynamic Range: {enh_stats['min']} - {enh_stats['max']}")
        print(f"   Contrast (StdDev): {enh_stats['std']:.2f}")
        print()

        if original.exists():
            brightness_change = enh_stats['mean_overall'] - orig_stats['mean_overall']
            contrast_change = enh_stats['std'] - orig_stats['std']

            print("📊 ENHANCEMENT IMPACT:")
            print(f"   Brightness Shift: {brightness_change:+.1f} ({brightness_change/orig_stats['mean_overall']*100:+.1f}%)")
            print(f"   Contrast Change: {contrast_change:+.2f} ({contrast_change/orig_stats['std']*100:+.1f}%)")
            print(f"   Red Channel: {enh_stats['mean_r'] - orig_stats['mean_r']:+.1f}")
            print(f"   Green Channel: {enh_stats['mean_g'] - orig_stats['mean_g']:+.1f}")
            print(f"   Blue Channel: {enh_stats['mean_b'] - orig_stats['mean_b']:+.1f}")

            # Color balance shift
            orig_temp = orig_stats['mean_r'] / orig_stats['mean_b']
            enh_temp = enh_stats['mean_r'] / enh_stats['mean_b']
            temp_direction = "(cooler)" if enh_temp < orig_temp else "(warmer)"
            print(f"   Color Temperature Shift: {orig_temp:.3f} → {enh_temp:.3f} {temp_direction}")
            print()

    print("=" * 80)
    print("🎨 APPLIED ENHANCEMENTS:")
    print("=" * 80)
    print("• Exposure: +0.15 stops (brighter, more inviting)")
    print("• Clarity: +0.35 (enhanced architectural details)")
    print("• Vibrance: +0.25 (vibrant pool blues and greens)")
    print("• Saturation: +0.15 (luxury color richness)")
    print("• Shadow Lift: 0.08 (revealed poolside detail)")
    print("• Highlight Recovery: 0.12 (preserved sky/reflections)")
    print("• Midtone Contrast: +0.18 (dimensional depth)")
    print("• White Balance: 5800K, -2 tint (cool, refreshing water)")
    print("• Luxury Glow: 0.20 (premium editorial finish)")
    print("• Chroma Denoise: 0.10 (clean, artifact-free)")
    print("=" * 80)
