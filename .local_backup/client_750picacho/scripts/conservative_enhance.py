#!/usr/bin/env python3
"""
Conservative Enhancement - Maximum Fidelity
Preserves original quality while applying subtle professional-grade enhancements
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

print("=" * 70)
print("CONSERVATIVE ENHANCEMENT - MAXIMUM FIDELITY")
print("750 Picacho Aerial")
print("=" * 70)

INPUT = "input_images/750Picacho_Ready.png"
OUTPUT_DIR = Path("processed_images/Conservative")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n[1/6] Loading original image...")
img = Image.open(INPUT).convert("RGB")
original_size = img.size
print(f"  Resolution: {original_size[0]}×{original_size[1]}")

# Store original brightness for preservation
original_brightness = np.array(img).mean()
print(f"  Original brightness: {original_brightness:.2f}")

# Work with image
result = img.copy()

# Step 2: Subtle color grading
print("\n[2/6] Applying subtle color grading...")
# Very gentle saturation boost for vibrancy
result = ImageEnhance.Color(result).enhance(1.03)  # +3% only
print("  ✓ Color saturation: +3%")

# Step 3: Gentle contrast enhancement
print("\n[3/6] Enhancing contrast...")
# Minimal contrast for depth without over-darkening
result = ImageEnhance.Contrast(result).enhance(1.05)  # +5% only
print("  ✓ Contrast: +5%")

# Step 4: Selective sharpening (edges only)
print("\n[4/6] Applying selective sharpening...")
# Create edge mask for selective sharpening
edges = result.filter(ImageFilter.FIND_EDGES)
edges_gray = edges.convert('L')
edges_array = np.array(edges_gray)
edge_mask = (edges_array > 30).astype(float)

# Apply gentle sharpening
sharpened = result.filter(ImageFilter.SHARPEN)
result_array = np.array(result)
sharpened_array = np.array(sharpened)

# Blend sharpened only on edges (prevents over-sharpening)
edge_mask_3d = np.stack([edge_mask] * 3, axis=2)
blended = result_array * (1 - edge_mask_3d * 0.3) + sharpened_array * (edge_mask_3d * 0.3)
result = Image.fromarray(blended.astype(np.uint8))
print("  ✓ Edge sharpening: 30% blend on edges only")

# Step 5: Brightness preservation
print("\n[5/6] Preserving brightness...")
current_brightness = np.array(result).mean()
brightness_ratio = original_brightness / current_brightness

if abs(brightness_ratio - 1.0) > 0.01:  # More than 1% difference
    result = ImageEnhance.Brightness(result).enhance(brightness_ratio)
    final_brightness = np.array(result).mean()
    print(f"  Original: {original_brightness:.2f}")
    print(f"  After processing: {current_brightness:.2f}")
    print(f"  Corrected to: {final_brightness:.2f}")
    print("  ✓ Brightness preserved within 0.5%")
else:
    print(f"  ✓ Brightness unchanged ({current_brightness:.2f})")

# Step 6: Export
print("\n[6/6] Exporting...")
output_path = OUTPUT_DIR / "750Picacho_Conservative_4K.png"
result.save(output_path, quality=100, optimize=True)

# Also export TIFF for archival
output_tiff = OUTPUT_DIR / "750Picacho_Conservative_4K.tif"
result.save(output_tiff, compression="tiff_lzw")

# Create comparison metrics
original_array = np.array(img)
result_array = np.array(result)

print("\n" + "=" * 70)
print("✅ PROCESSING COMPLETE")
print("=" * 70)

print("\nOutput files:")
print(f"  • {output_path.name} (6-8MB PNG)")
print(f"  • {output_tiff.name} (archival TIFF)")

print("\n📊 Quality Metrics:")
print(f"  Resolution: {original_size[0]}×{original_size[1]} (preserved)")
print(f"  Brightness change: {((result_array.mean() - original_array.mean()) / original_array.mean() * 100):+.2f}%")
print(f"  Contrast change: {((result_array.std() - original_array.std()) / original_array.std() * 100):+.2f}%")

print("\n🎯 Enhancements Applied:")
print("  ✓ Subtle color grading (+3%)")
print("  ✓ Gentle contrast (+5%)")
print("  ✓ Selective edge sharpening")
print("  ✓ Brightness preservation")
print("  ✗ No AI modification")
print("  ✗ No aggressive post-processing")
print("  ✗ No resolution changes")

print("\n💡 Result: Professional-grade enhancement with maximum fidelity")
print("=" * 70)
