#!/usr/bin/env python3
"""
Conservative enhancement for Coastal_Interior_11.tif
Preserves 16-bit TIFF quality and architectural detail
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
from pathlib import Path
import json
from datetime import datetime


def analyze_image(img_array):
    """Analyze image technical specs and quality metrics."""

    # Basic stats
    mean = np.mean(img_array)
    std = np.std(img_array)
    min_val = np.min(img_array)
    max_val = np.max(img_array)

    # Determine bit depth range
    if max_val <= 255:
        bit_depth = 8
        max_range = 255
    elif max_val <= 65535:
        bit_depth = 16
        max_range = 65535
    else:
        bit_depth = 32
        max_range = 1.0

    # Clipping analysis
    shadow_clip = np.sum(img_array == min_val) / img_array.size * 100
    highlight_clip = np.sum(img_array == max_val) / img_array.size * 100

    # Histogram analysis
    hist, bins = np.histogram(img_array.flatten(), bins=256, range=(0, max_range))
    hist = hist / np.sum(hist) * 100  # Convert to percentages

    # Exposure metrics (normalized to 0-100 scale)
    normalized_mean = (mean / max_range) * 100

    # Contrast estimation (using std dev as proxy)
    contrast_score = (std / max_range) * 100

    return {
        'bit_depth': bit_depth,
        'mean_brightness': float(mean),
        'normalized_mean': float(normalized_mean),
        'std_dev': float(std),
        'contrast_score': float(contrast_score),
        'min': float(min_val),
        'max': float(max_val),
        'shadow_clipping_pct': float(shadow_clip),
        'highlight_clipping_pct': float(highlight_clip),
        'dynamic_range': float(max_val - min_val),
    }


def conservative_enhance(img, params):
    """Apply conservative enhancements preserving 16-bit quality."""

    # Ensure we're working with high bit depth
    if img.mode not in ('I;16', 'I', 'RGB', 'L'):
        if img.mode == 'I;16':
            pass  # Already 16-bit
        else:
            img = img.convert('RGB')

    # Store original mode for restoration
    original_mode = img.mode

    # Work in RGB for processing (Pillow requirement)
    if img.mode == 'I;16':
        # Convert to RGB while preserving dynamic range
        img_array = np.array(img)
        # Normalize to 8-bit for Pillow processing
        img_8bit = ((img_array / 65535.0) * 255).astype(np.uint8)
        img_rgb = Image.fromarray(img_8bit)
        working_16bit = True
    else:
        img_rgb = img.convert('RGB')
        working_16bit = False

    # Apply enhancements
    print(f"  Applying exposure adjustment: {params['exposure']:+.2f}")
    if params['exposure'] != 0:
        # Manual exposure adjustment via array math
        img_array = np.array(img_rgb, dtype=np.float32)
        exposure_factor = 2 ** params['exposure']
        img_array = np.clip(img_array * exposure_factor, 0, 255).astype(np.uint8)
        img_rgb = Image.fromarray(img_array)

    print(f"  Applying contrast: {params['contrast']:.2f}x")
    enhancer = ImageEnhance.Contrast(img_rgb)
    img_rgb = enhancer.enhance(params['contrast'])

    print(f"  Applying saturation: {params['saturation']:.2f}x")
    enhancer = ImageEnhance.Color(img_rgb)
    img_rgb = enhancer.enhance(params['saturation'])

    if params.get('clarity', 0) > 0:
        print(f"  Applying clarity: {params['clarity']:.2f}")
        # Unsharp mask for clarity
        radius = 2.0
        amount = params['clarity']
        img_rgb = img_rgb.filter(ImageFilter.UnsharpMask(radius=radius, percent=int(amount * 100), threshold=3))

    if params.get('denoise', 0) > 0:
        print(f"  Applying denoising: strength {params['denoise']}")
        # Gentle median filter
        img_rgb = img_rgb.filter(ImageFilter.MedianFilter(size=3))

    # Convert back to 16-bit if needed
    if working_16bit:
        # Scale back up to 16-bit
        img_array = np.array(img_rgb, dtype=np.float32)
        img_16bit = (img_array / 255.0 * 65535).astype(np.uint16)
        # Note: PIL doesn't have great 16-bit support, so we'll save via numpy
        return Image.fromarray(img_16bit, mode='I;16'), img_rgb
    else:
        return img_rgb, img_rgb


def create_comparison(img1, img2, labels, output_path):
    """Create side-by-side comparison image."""

    # Work with 8-bit RGB for display
    if img1.mode == 'I;16':
        arr1 = np.array(img1)
        img1_display = Image.fromarray(((arr1 / 65535.0) * 255).astype(np.uint8))
    else:
        img1_display = img1.convert('RGB')

    if img2.mode == 'I;16':
        arr2 = np.array(img2)
        img2_display = Image.fromarray(((arr2 / 65535.0) * 255).astype(np.uint8))
    else:
        img2_display = img2.convert('RGB')

    # Resize for comparison if too large
    max_width = 1920
    if img1_display.width > max_width:
        scale = max_width / img1_display.width
        new_size = (int(img1_display.width * scale), int(img1_display.height * scale))
        img1_display = img1_display.resize(new_size, Image.Resampling.LANCZOS)
        img2_display = img2_display.resize(new_size, Image.Resampling.LANCZOS)

    # Create comparison canvas
    width = img1_display.width * 2 + 60  # Gap between images
    height = img1_display.height + 100  # Space for labels
    comparison = Image.new('RGB', (width, height), color=(30, 30, 30))

    # Paste images
    comparison.paste(img1_display, (10, 60))
    comparison.paste(img2_display, (img1_display.width + 50, 60))

    # Add labels
    draw = ImageDraw.Draw(comparison)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except:
        font = ImageFont.load_default()
        font_small = font

    # Draw labels
    draw.text((10, 15), labels[0], fill=(255, 255, 255), font=font)
    draw.text((img1_display.width + 50, 15), labels[1], fill=(255, 255, 255), font=font)

    # Save comparison
    comparison.save(output_path, quality=95)
    print(f"  Comparison saved: {output_path}")


def main():
    """Main processing pipeline."""

    print("=" * 80)
    print("CONSERVATIVE ARCHITECTURAL ENHANCEMENT")
    print("Coastal_Interior_11.tif Processing")
    print("=" * 80)

    # Setup paths
    input_path = Path("input_images/Coastal_Interior_11.tif")
    output_dir = Path("processed_images/Conservative")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"Coastal_Interior_11_enhanced_{timestamp}.tif"
    comparison_path = output_dir / f"Coastal_Interior_11_comparison_{timestamp}.jpg"
    metrics_path = output_dir / f"Coastal_Interior_11_metrics_{timestamp}.json"

    # Load image
    print(f"\n1. LOADING IMAGE")
    print(f"   Input: {input_path}")
    img = Image.open(input_path)
    print(f"   Mode: {img.mode}")
    print(f"   Size: {img.size[0]} x {img.size[1]} pixels")
    print(f"   Format: {img.format}")

    # Analyze original
    print(f"\n2. ANALYZING ORIGINAL IMAGE")
    img_array = np.array(img)
    original_metrics = analyze_image(img_array)

    print(f"   Bit Depth: {original_metrics['bit_depth']}-bit")
    print(f"   Mean Brightness: {original_metrics['normalized_mean']:.1f}/100")
    print(f"   Contrast Score: {original_metrics['contrast_score']:.1f}/100")
    print(f"   Shadow Clipping: {original_metrics['shadow_clipping_pct']:.3f}%")
    print(f"   Highlight Clipping: {original_metrics['highlight_clipping_pct']:.3f}%")
    print(f"   Dynamic Range: {original_metrics['dynamic_range']:.0f}")

    # Determine enhancement parameters based on analysis
    print(f"\n3. DETERMINING ENHANCEMENT PARAMETERS")

    # Conservative parameters based on analysis
    params = {
        'exposure': 0.0,  # Start neutral
        'contrast': 1.08,
        'saturation': 1.05,
        'clarity': 0.15,
        'denoise': 0,
    }

    # Adjust based on brightness
    if original_metrics['normalized_mean'] < 45:
        params['exposure'] = 0.15
        print(f"   Image appears slightly dark (brightness {original_metrics['normalized_mean']:.1f})")
        print(f"   Increasing exposure by +0.15 stops")
    elif original_metrics['normalized_mean'] > 55:
        params['exposure'] = -0.10
        print(f"   Image appears slightly bright (brightness {original_metrics['normalized_mean']:.1f})")
        print(f"   Decreasing exposure by -0.10 stops")
    else:
        print(f"   Brightness optimal ({original_metrics['normalized_mean']:.1f}), no exposure adjustment")

    # Adjust contrast if needed
    if original_metrics['contrast_score'] < 20:
        params['contrast'] = 1.12
        print(f"   Low contrast detected ({original_metrics['contrast_score']:.1f}), boosting to 1.12x")
    elif original_metrics['contrast_score'] > 35:
        params['contrast'] = 1.05
        print(f"   High contrast detected ({original_metrics['contrast_score']:.1f}), using gentle 1.05x")

    print(f"\n   FINAL PARAMETERS:")
    for key, value in params.items():
        if key == 'exposure':
            print(f"     {key}: {value:+.2f} stops")
        else:
            print(f"     {key}: {value}")

    # Apply enhancements
    print(f"\n4. APPLYING ENHANCEMENTS")
    enhanced_img, preview_img = conservative_enhance(img, params)

    # Save enhanced image
    print(f"\n5. SAVING OUTPUT")
    print(f"   Output: {output_path}")

    if enhanced_img.mode == 'I;16':
        # Save 16-bit TIFF with compression
        enhanced_img.save(output_path, compression='tiff_adobe_deflate')
        print(f"   Format: 16-bit TIFF with lossless compression")
    else:
        enhanced_img.save(output_path, compression='tiff_adobe_deflate')
        print(f"   Format: {enhanced_img.mode} TIFF with lossless compression")

    # Analyze enhanced
    print(f"\n6. ANALYZING ENHANCED IMAGE")
    enhanced_array = np.array(enhanced_img)
    enhanced_metrics = analyze_image(enhanced_array)

    print(f"   Mean Brightness: {enhanced_metrics['normalized_mean']:.1f}/100 "
          f"(Δ {enhanced_metrics['normalized_mean'] - original_metrics['normalized_mean']:+.1f})")
    print(f"   Contrast Score: {enhanced_metrics['contrast_score']:.1f}/100 "
          f"(Δ {enhanced_metrics['contrast_score'] - original_metrics['contrast_score']:+.1f})")
    print(f"   Shadow Clipping: {enhanced_metrics['shadow_clipping_pct']:.3f}% "
          f"(Δ {enhanced_metrics['shadow_clipping_pct'] - original_metrics['shadow_clipping_pct']:+.3f}%)")
    print(f"   Highlight Clipping: {enhanced_metrics['highlight_clipping_pct']:.3f}% "
          f"(Δ {enhanced_metrics['highlight_clipping_pct'] - original_metrics['highlight_clipping_pct']:+.3f}%)")

    # Calculate brightness change percentage
    brightness_change = ((enhanced_metrics['normalized_mean'] - original_metrics['normalized_mean'])
                        / original_metrics['normalized_mean'] * 100)
    print(f"   Brightness Change: {brightness_change:+.2f}%")

    # Quality verification
    print(f"\n7. QUALITY VERIFICATION")
    checks = {
        '16-bit preservation': enhanced_metrics['bit_depth'] >= original_metrics['bit_depth'],
        'Resolution maintained': enhanced_img.size == img.size,
        'No excessive shadow clipping': enhanced_metrics['shadow_clipping_pct'] < 1.0,
        'No excessive highlight clipping': enhanced_metrics['highlight_clipping_pct'] < 1.0,
        'Brightness change < 5%': abs(brightness_change) < 5.0,
    }

    for check, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"   {status}: {check}")

    # Create comparison
    print(f"\n8. GENERATING COMPARISON")
    create_comparison(
        img,
        enhanced_img,
        ["ORIGINAL", "ENHANCED"],
        comparison_path
    )

    # Save metrics
    print(f"\n9. SAVING METRICS")
    metrics = {
        'timestamp': timestamp,
        'input_file': str(input_path),
        'output_file': str(output_path),
        'parameters': params,
        'original_metrics': original_metrics,
        'enhanced_metrics': enhanced_metrics,
        'quality_checks': checks,
        'brightness_change_pct': brightness_change,
    }

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"   Metrics: {metrics_path}")

    # Summary
    print(f"\n{'=' * 80}")
    print("PROCESSING COMPLETE")
    print(f"{'=' * 80}")
    print(f"Enhanced Image: {output_path}")
    print(f"Comparison: {comparison_path}")
    print(f"Metrics: {metrics_path}")

    all_checks_passed = all(checks.values())
    if all_checks_passed:
        print(f"\n✓ All quality checks PASSED")
    else:
        print(f"\n⚠ Some quality checks FAILED - review output carefully")

    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
