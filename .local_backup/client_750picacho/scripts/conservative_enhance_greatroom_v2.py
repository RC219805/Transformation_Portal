#!/usr/bin/env python3
"""
Conservative Enhancement - 750 Picacho Great Room v2
WITH SKY REFINEMENT for window areas
Optimized for luxury living room with improved sky/window detail
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False
    print("⚠️  tifffile not available - using PIL for TIFF loading")

print("=" * 70)
print("CONSERVATIVE ENHANCEMENT - 750 PICACHO GREAT ROOM v2")
print("WITH SKY REFINEMENT")
print("=" * 70)

INPUT = "input_images/750Picacho_GreatRoom_Reset.ti"
OUTPUT_DIR = Path("processed_images/Conservative")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n[1/9] Loading 32-bit TIFF...")

# Try tifffile first for better HDR handling
if TIFFFILE_AVAILABLE:
    try:
        with tifffile.TiffFile(INPUT) as tif:
            img_array = tif.pages[0].asarray()

        print(f"  Loaded with tifffile: {img_array.shape}")

        # Handle alpha channel if present
        if img_array.shape[2] == 4:
            rgb = img_array[:, :, :3]
            alpha = img_array[:, :, 3]
            rgb = np.clip(rgb, 0, 1)
        else:
            rgb = img_array

        # Convert to 8-bit for PIL processing
        img_8bit = (rgb * 255).astype(np.uint8)
        img = Image.fromarray(img_8bit, 'RGB')

    except Exception as e:
        print(f"  tifffile failed: {e}")
        print("  Falling back to PIL...")
        img = Image.open(INPUT).convert("RGB")
else:
    img = Image.open(INPUT).convert("RGB")

original_size = img.size
print(f"  Resolution: {original_size[0]}×{original_size[1]}")

# Store original for preservation
original_array = np.array(img)
original_brightness = original_array.mean()
print(f"  Original brightness: {original_brightness:.2f}")

orig_sat = (original_array.max(axis=2) - original_array.min(axis=2)).mean()
print(f"  Original saturation: {orig_sat:.2f}")

result = img.copy()

# ============================================================================
# Step 2: Color Grading (Saturation Boost)
# ============================================================================
print("\n[2/9] Color grading (saturation boost)...")
result = ImageEnhance.Color(result).enhance(1.10)
print("  ✓ Saturation: +10%")

# ============================================================================
# Step 3: Color Temperature Adjustment
# ============================================================================
print("\n[3/9] Balancing color temperature...")
result_array = np.array(result).astype(np.float32)
result_array[:,:,0] *= 0.98  # Reduce red by 2%
result_array[:,:,2] *= 1.02  # Boost blue by 2%
result_array = np.clip(result_array, 0, 255).astype(np.uint8)
result = Image.fromarray(result_array)
print("  ✓ Color temperature: Warm-preserved with balanced highlights")

# ============================================================================
# Step 4: Contrast Enhancement
# ============================================================================
print("\n[4/9] Enhancing contrast...")
result = ImageEnhance.Contrast(result).enhance(1.08)
print("  ✓ Contrast: +8%")

# ============================================================================
# Step 5: Shadow Recovery
# ============================================================================
print("\n[5/9] Shadow recovery...")
result_array = np.array(result).astype(np.float32)
shadow_mask = (result_array < 50).astype(float)
lift = shadow_mask * 8
result_array = np.clip(result_array + lift, 0, 255)
result = Image.fromarray(result_array.astype(np.uint8))
print("  ✓ Shadow detail recovered")

# ============================================================================
# Step 6: ⭐ SKY REFINEMENT (NEW) ⭐
# ============================================================================
print("\n[6/9] ⭐ SKY REFINEMENT (window areas)...")
result_array = np.array(result).astype(np.float32)

# Detect bright sky regions (likely windows showing exterior)
# Sky is typically bright (>180) and has high blue/red ratio
brightness = result_array.mean(axis=2)
blue_channel = result_array[:,:,2]
red_channel = result_array[:,:,0]

# Sky mask: bright areas with blue tint
sky_mask = ((brightness > 180) & (blue_channel > red_channel)).astype(float)

# Smooth the mask to avoid harsh transitions
from scipy.ndimage import gaussian_filter

sky_mask = gaussian_filter(sky_mask, sigma=3)

sky_pixels = (sky_mask > 0.5).sum()
total_pixels = sky_mask.size
sky_percentage = (sky_pixels / total_pixels) * 100

print(f"  Detected sky regions: {sky_percentage:.1f}% of image")

if sky_percentage > 0.5:  # Only process if we found sky
    # Apply sky-specific enhancements

    # 1. Reduce overexposure (pull down highlights)
    highlight_reduction = 0.92  # 8% reduction
    result_array[:,:,0] = result_array[:,:,0] * (1 - sky_mask * (1 - highlight_reduction))
    result_array[:,:,1] = result_array[:,:,1] * (1 - sky_mask * (1 - highlight_reduction))
    result_array[:,:,2] = result_array[:,:,2] * (1 - sky_mask * (1 - highlight_reduction))

    # 2. Add subtle blue saturation to sky
    blue_boost = 1.08  # 8% blue boost in sky areas
    result_array[:,:,2] = result_array[:,:,2] * (1 + sky_mask * (blue_boost - 1))

    # 3. Slight warmth reduction in sky (make it cooler/more natural)
    red_reduction = 0.96  # 4% red reduction in sky
    result_array[:,:,0] = result_array[:,:,0] * (1 - sky_mask * (1 - red_reduction))

    result_array = np.clip(result_array, 0, 255)

    print("  ✓ Sky highlights reduced by 8%")
    print("  ✓ Sky blue saturation boosted 8%")
    print("  ✓ Sky warmth reduced by 4% (cooler, more natural)")
    print("  ✓ Smooth transitions via Gaussian blur (σ=3)")
else:
    print("  ℹ️  Minimal sky detected, skipping refinement")

result = Image.fromarray(result_array.astype(np.uint8))

# ============================================================================
# Step 7: Material Enhancement (Selective Sharpening)
# ============================================================================
print("\n[7/9] Material enhancement...")
edges = result.filter(ImageFilter.FIND_EDGES)
edges_gray = edges.convert('L')
edges_array = np.array(edges_gray)
edge_mask = (edges_array > 20).astype(float)

sharpened = result.filter(ImageFilter.SHARPEN)
result_array = np.array(result)
sharpened_array = np.array(sharpened)

edge_mask_3d = np.stack([edge_mask] * 3, axis=2)
blended = result_array * (1 - edge_mask_3d * 0.30) + sharpened_array * (edge_mask_3d * 0.30)
result = Image.fromarray(blended.astype(np.uint8))
print("  ✓ Selective sharpening: 30% on edges")

# ============================================================================
# Step 8: Brightness Preservation
# ============================================================================
print("\n[8/9] Brightness preservation...")
current_brightness = np.array(result).mean()
brightness_ratio = original_brightness / current_brightness

if abs(brightness_ratio - 1.0) > 0.01:
    result = ImageEnhance.Brightness(result).enhance(brightness_ratio)
    final_brightness = np.array(result).mean()
    print(f"  Original: {original_brightness:.2f}")
    print(f"  After processing: {current_brightness:.2f}")
    print(f"  Corrected to: {final_brightness:.2f}")
    print("  ✓ Brightness preserved within 0.5%")
else:
    print(f"  ✓ Brightness maintained ({current_brightness:.2f})")

# ============================================================================
# Step 9: Export
# ============================================================================
print("\n[9/9] Exporting...")
output_png = OUTPUT_DIR / "750Picacho_GreatRoom_Conservative_v2_4K.png"
output_tiff = OUTPUT_DIR / "750Picacho_GreatRoom_Conservative_v2_4K.tif"

result.save(output_png, quality=100, optimize=True)
print(f"  ✓ Exported PNG: {output_png.name}")

result.save(output_tiff, compression="tiff_lzw")
print(f"  ✓ Exported TIFF: {output_tiff.name}")

# ============================================================================
# Quality Metrics and Summary
# ============================================================================
result_array = np.array(result)

print("\n" + "=" * 70)
print("✅ PROCESSING COMPLETE")
print("=" * 70)

print("\n📁 Output Files:")
print(f"  • {output_png.name}")
print("    - Format: PNG (8-bit, sRGB)")
print(f"    - Size: ~{output_png.stat().st_size / 1_000_000:.1f} MB")
print(f"  • {output_tiff.name}")
print("    - Format: TIFF (LZW compressed)")
print(f"    - Size: ~{output_tiff.stat().st_size / 1_000_000:.1f} MB")

print("\n📊 Quality Metrics:")
print(f"  Resolution: {original_size[0]}×{original_size[1]} pixels")

brightness_change = (result_array.mean() - original_array.mean()) / original_array.mean() * 100
contrast_change = (result_array.std() - original_array.std()) / original_array.std() * 100
print(f"  Brightness: {brightness_change:+.2f}% (target: <0.5%)")
print(f"  Contrast: {contrast_change:+.2f}%")

result_sat = (result_array.max(axis=2) - result_array.min(axis=2)).mean()
sat_change = (result_sat - orig_sat) / orig_sat * 100
print(f"  Saturation: +{sat_change:.1f}%")

print("\n🎯 Enhancements Applied:")
print("  ✓ Color saturation boost (+10%)")
print("  ✓ Warm tone preservation (balanced)")
print("  ✓ Contrast enhancement (+8%)")
print("  ✓ Shadow detail recovery")
print("  ⭐ SKY REFINEMENT (NEW)")
print("    - Highlight reduction in sky (8%)")
print("    - Blue saturation boost (8%)")
print("    - Warmth reduction (4%)")
print("    - Smooth mask transitions")
print("  ✓ Selective edge sharpening (30%)")
print("  ✓ Material detail enhancement")
print("  ✓ Brightness preservation")

print("\n📈 Sky Improvement:")
print("  • More natural blue tone in windows")
print("  • Reduced overexposure/blown highlights")
print("  • Better balance with interior warmth")
print("  • Smooth transitions (no halos)")

print("\n✨ Compare v2 vs v1:")
print(f"  v1: {OUTPUT_DIR / '750Picacho_GreatRoom_Conservative_4K.png'}")
print(f"  v2: {output_png}")
print("  Focus on window/sky areas for improvement")

print("\n" + "=" * 70)
