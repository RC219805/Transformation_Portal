#!/usr/bin/env python3
"""
Visual comparison tool for pool enhancement outputs.
Creates side-by-side comparisons with quality metrics.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Try to use tifffile for better TIFF support
try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False
    print("Note: tifffile not available, using PIL for TIFF files")

def load_image(path):
    """Load image with proper TIFF support."""
    path = Path(path)
    if HAS_TIFFFILE and path.suffix.lower() in ['.tif', '.tiff']:
        try:
            array = tifffile.imread(str(path))

            # Handle different data types
            if array.dtype == np.float32 or array.dtype == np.float64:
                # Clip and normalize float data to 0-255 range
                array = np.clip(array, 0, 1)
                array = (array * 255).astype(np.uint8)
            elif array.dtype == np.uint16:
                # Normalize 16-bit to 8-bit for visualization
                array = (array / 256).astype(np.uint8)

            # Handle alpha channel
            if array.ndim == 3 and array.shape[2] == 4:
                # Convert RGBA to RGB (discard alpha)
                array = array[:, :, :3]

            return Image.fromarray(array)
        except Exception as e:
            print(f"Warning: tifffile failed ({e}), falling back to PIL")
            return Image.open(path)
    else:
        return Image.open(path)

def calculate_metrics(original, enhanced):
    """Calculate quality metrics between images."""
    orig_array = np.array(original.convert('RGB'), dtype=np.float32)
    enh_array = np.array(enhanced.convert('RGB'), dtype=np.float32)

    # Ensure same size
    if orig_array.shape != enh_array.shape:
        enhanced_resized = enhanced.resize(original.size, Image.LANCZOS)
        enh_array = np.array(enhanced_resized.convert('RGB'), dtype=np.float32)

    # Calculate metrics
    mse = np.mean((orig_array - enh_array) ** 2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')

    # Brightness comparison
    orig_brightness = np.mean(orig_array)
    enh_brightness = np.mean(enh_array)
    brightness_delta = ((enh_brightness - orig_brightness) / orig_brightness) * 100

    # Saturation comparison (simplified)
    orig_sat = np.std(orig_array)
    enh_sat = np.std(enh_array)
    saturation_delta = ((enh_sat - orig_sat) / orig_sat) * 100

    return {
        'mse': mse,
        'psnr': psnr,
        'brightness_delta': brightness_delta,
        'saturation_delta': saturation_delta,
        'orig_brightness': orig_brightness,
        'enh_brightness': enh_brightness
    }

def create_comparison(original_path, enhanced_paths, output_dir):
    """Create side-by-side comparisons."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"\n{'='*80}")
    print("POOL ENHANCEMENT QUALITY COMPARISON")
    print(f"{'='*80}\n")

    original = load_image(original_path)
    print(f"Original: {original_path}")
    print(f"  Size: {original.size}")
    print(f"  Mode: {original.mode}")
    print(f"  Brightness: {np.mean(np.array(original.convert('RGB'))):.2f}/255")

    results = []

    for i, enh_path in enumerate(enhanced_paths, 1):
        enhanced = load_image(enh_path)

        print(f"\n{'-'*80}")
        print(f"Version {i}: {Path(enh_path).name}")
        print(f"  Size: {enhanced.size}")
        print(f"  Mode: {enhanced.mode}")

        # Calculate metrics
        metrics = calculate_metrics(original, enhanced)

        print(f"\n  Quality Metrics:")
        print(f"    PSNR: {metrics['psnr']:.2f} dB")
        print(f"    MSE: {metrics['mse']:.2f}")
        print(f"    Brightness: {metrics['enh_brightness']:.2f}/255 ({metrics['brightness_delta']:+.1f}%)")
        print(f"    Saturation: {metrics['saturation_delta']:+.1f}%")

        # Visual assessment
        if metrics['brightness_delta'] < -10:
            print(f"  ⚠️  WARNING: Image is significantly darker ({metrics['brightness_delta']:.1f}%)")
        elif metrics['brightness_delta'] > 10:
            print(f"  ⚠️  WARNING: Image is significantly brighter ({metrics['brightness_delta']:.1f}%)")
        else:
            print(f"  ✓ Brightness change is reasonable ({metrics['brightness_delta']:+.1f}%)")

        if metrics['psnr'] < 25:
            print(f"  ⚠️  WARNING: Low PSNR indicates significant distortion")
        elif metrics['psnr'] > 40:
            print(f"  ✓ Excellent quality preservation (PSNR: {metrics['psnr']:.2f} dB)")
        else:
            print(f"  ✓ Good quality (PSNR: {metrics['psnr']:.2f} dB)")

        # Create side-by-side comparison
        # Resize for comparison (max 2000px wide per image)
        max_width = 2000
        if original.width > max_width:
            scale = max_width / original.width
            new_size = (int(original.width * scale), int(original.height * scale))
            orig_resized = original.resize(new_size, Image.LANCZOS)
            enh_resized = enhanced.resize(new_size, Image.LANCZOS)
        else:
            orig_resized = original
            enh_resized = enhanced.resize(original.size, Image.LANCZOS)

        # Create comparison image
        comparison = Image.new('RGB', (orig_resized.width * 2 + 20, orig_resized.height + 60))
        comparison.paste((50, 50, 50), (0, 0, comparison.width, comparison.height))
        comparison.paste(orig_resized, (10, 50))
        comparison.paste(enh_resized, (orig_resized.width + 10, 50))

        # Save comparison
        version_name = Path(enh_path).stem
        comparison_path = output_dir / f"comparison_{version_name}.jpg"
        comparison.save(comparison_path, 'JPEG', quality=95)
        print(f"\n  Comparison saved: {comparison_path}")

        results.append({
            'name': Path(enh_path).name,
            'metrics': metrics,
            'comparison_path': comparison_path
        })

    # Summary and recommendations
    print(f"\n{'='*80}")
    print("SUMMARY & RECOMMENDATIONS")
    print(f"{'='*80}\n")

    if len(results) > 1:
        # Find best version
        best_psnr = max(results, key=lambda x: x['metrics']['psnr'])
        best_brightness = min(results, key=lambda x: abs(x['metrics']['brightness_delta']))

        print(f"Best PSNR (quality): {best_psnr['name']} ({best_psnr['metrics']['psnr']:.2f} dB)")
        print(f"Best brightness match: {best_brightness['name']} ({best_brightness['metrics']['brightness_delta']:+.1f}%)")

    # Overall assessment
    avg_brightness_delta = np.mean([r['metrics']['brightness_delta'] for r in results])

    if avg_brightness_delta < -15:
        print(f"\n⚠️  CRITICAL: All versions are too dark (avg: {avg_brightness_delta:.1f}%)")
        print("   Recommendation: Increase exposure significantly (try +0.5 to +1.0)")
    elif avg_brightness_delta < -5:
        print(f"\n⚠️  All versions are slightly dark (avg: {avg_brightness_delta:.1f}%)")
        print("   Recommendation: Increase exposure moderately (try +0.2 to +0.4)")
    elif avg_brightness_delta > 15:
        print(f"\n⚠️  All versions are too bright (avg: {avg_brightness_delta:.1f}%)")
        print("   Recommendation: Decrease exposure (try -0.3 to -0.5)")
    else:
        print(f"\n✓ Brightness levels are acceptable (avg: {avg_brightness_delta:+.1f}%)")

    print(f"\nAll comparisons saved to: {output_dir}/")
    print(f"Open comparison images to visually inspect quality.\n")

    return results

if __name__ == "__main__":
    original = "input_images/750Picacho_Pool.tiff"
    enhanced = [
        "processed_images/Conservative/750Picacho_Pool_Enhanced.tif",
        "processed_images/Conservative/750Picacho_Pool_Enhanced_v2.tif",
        "processed_images/Conservative/750Picacho_Pool_Enhanced_v3.tif",
        "processed_images/Conservative/750Picacho_Pool_Enhanced_v4.tif",
    ]

    # Filter to only existing files
    enhanced = [e for e in enhanced if Path(e).exists()]

    if not Path(original).exists():
        print(f"Error: Original not found: {original}")
        sys.exit(1)

    if not enhanced:
        print("Error: No enhanced versions found")
        sys.exit(1)

    results = create_comparison(original, enhanced, "processed_images/Pool_Comparisons")
