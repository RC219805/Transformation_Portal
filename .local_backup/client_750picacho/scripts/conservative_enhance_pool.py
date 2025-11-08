#!/usr/bin/env python3
"""
Conservative Enhancement - 750 Picacho Pool Aerial Rendering
OPTIMIZED FOR LINEAR TIFF: Aerial pool view with water, hardscape, vegetation

Based on analysis of 750Picacho_Pool.tiff:
- 16-bit linear TIFF requiring gamma correction (2.2)
- Slightly underexposed (0.441 luminance → target 0.525)
- Low contrast (0.105 → target 0.135)
- Pool water needs subtle warming (reduce artificial cyan cast)
- Excellent detail (0.047) - minimal sharpening needed
- Shadow recovery needed (28% of frame)

Strategy:
1. Convert linear to sRGB (critical first step!)
2. Exposure lift (+0.25 EV)
3. Contrast boost (1.10×)
4. Shadow recovery (+0.35 stops)
5. Pool water color correction (warmer cyan)
6. Material enhancement (water, concrete, vegetation)
7. Minimal clarity (0.12) - already sharp!
8. California Golden Hour LUT @ 0.65 strength

WARNINGS FROM LESSONS LEARNED:
- NO sharpening (image already sharp, creates halos)
- Convert from linear FIRST (or image will be too dark)
- Preserve water transparency and reflections
- Maintain concrete neutrality
- Conservative parameters to avoid artifacts
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
from scipy.ndimage import gaussian_filter

try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False

print("=" * 80)
print("CONSERVATIVE ENHANCEMENT - 750 PICACHO POOL AERIAL")
print("Linear TIFF → sRGB | Exposure lift | Water color correction")
print("=" * 80)

# ============================================================================
# CONFIGURATION - OPTIMIZED FOR AERIAL POOL RENDERING
# ============================================================================

INPUT = "input_images/750Picacho_Pool.tif"
OUTPUT_DIR = Path("processed_images/Conservative")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Exposure & Tone (based on analysis recommendations)
GAMMA_CORRECTION = 2.2                # Linear → sRGB (CRITICAL!)
GLOBAL_EXPOSURE_LIFT = 0.25           # +0.25 EV (0.441 → 0.525 luminance)
SHADOW_LIFT_STOPS = 0.35              # +0.35 stops for shadows
SHADOW_THRESHOLD = 0.25               # Luminance < 0.25 is shadow
HIGHLIGHT_PROTECTION = 0.92           # Protect pixels > 0.92
MIDTONE_CONTRAST = 1.10               # +10% contrast boost

# Color & Saturation
GLOBAL_SATURATION = 1.06              # +6% saturation
VEGETATION_SAT_BOOST = 1.05           # +5% green saturation

# Pool Water Color Correction (reduce artificial cyan cast)
WATER_GREEN_BOOST = 1.08              # +8% green (warmer cyan)
WATER_BLUE_REDUCTION = 0.96           # -4% blue (less cyan)
WATER_RED_ADJUSTMENT = 0.98           # -2% red (maintain cyan hue)

# Material Enhancement
CLARITY_STRENGTH = 0.12               # 12% clarity (CONSERVATIVE - already sharp!)
CLARITY_RADIUS = 64                   # Radius for high-pass filter (4K image)

# Material Response Zones
WATER_ENHANCEMENT = 0.65              # 65% for water reflections
CONCRETE_ENHANCEMENT = 0.50           # 50% for hardscape texture

# Output
OUTPUT_BIT_DEPTH = 16                 # 16-bit TIFF

# ============================================================================
# LOAD IMAGE
# ============================================================================
print(f"\n[1/9] Loading image: {INPUT}")

if TIFFFILE_AVAILABLE:
    try:
        img_array = tifffile.imread(INPUT)
        print(f"  ✓ Loaded with tifffile: {img_array.shape}, dtype: {img_array.dtype}")

        # Normalize to 0-1 range
        if img_array.dtype in (np.float32, np.float64):
            if img_array.max() > 1.0:
                rgb_linear = np.clip(img_array / img_array.max(), 0, 1)
            else:
                rgb_linear = img_array.copy()
        elif img_array.dtype == np.uint16:
            rgb_linear = img_array.astype(np.float32) / 65535.0
        else:
            rgb_linear = img_array.astype(np.float32) / 255.0

        # Drop alpha channel if present
        if rgb_linear.shape[2] == 4:
            rgb_linear = rgb_linear[:, :, :3]
            print("  ✓ Dropped alpha channel")

    except Exception as e:
        print(f"  ⚠️  tifffile failed: {e}, falling back to PIL")
        TIFFFILE_AVAILABLE = False

if not TIFFFILE_AVAILABLE:
    img = Image.open(INPUT).convert('RGB')
    rgb_linear = np.array(img, dtype=np.float32) / 255.0

print(f"  Range: [{rgb_linear.min():.3f}, {rgb_linear.max():.3f}]")
print(f"  Mean luminance: {rgb_linear.mean():.3f}")

original_luminance = rgb_linear.mean()

# ============================================================================
# STEP 2: GAMMA CORRECTION (LINEAR → sRGB) - CRITICAL FIRST STEP!
# ============================================================================
print(f"\n[2/9] Converting linear to sRGB (gamma {GAMMA_CORRECTION})...")

rgb = np.power(np.clip(rgb_linear, 0, 1), 1/GAMMA_CORRECTION)

print("  ✓ Gamma corrected")
print(f"  ✓ Luminance after gamma: {rgb.mean():.3f}")

# ============================================================================
# STEP 3: EXPOSURE LIFT
# ============================================================================
print("\n[3/9] Lifting exposure...")

# Calculate exposure multiplier
exposure_multiplier = 2 ** GLOBAL_EXPOSURE_LIFT
rgb_exposed = rgb * exposure_multiplier
rgb_exposed = np.clip(rgb_exposed, 0, 1)

new_luminance = rgb_exposed.mean()
print(f"  ✓ Exposure: +{GLOBAL_EXPOSURE_LIFT:.2f} EV (×{exposure_multiplier:.3f})")
print(f"  ✓ Luminance: {original_luminance:.3f} → {new_luminance:.3f}")

# ============================================================================
# STEP 4: SHADOW RECOVERY
# ============================================================================
print("\n[4/9] Recovering shadow detail...")

# Calculate luminance
luminance = 0.2126 * rgb_exposed[:,:,0] + 0.7152 * rgb_exposed[:,:,1] + 0.0722 * rgb_exposed[:,:,2]
shadow_mask = luminance < SHADOW_THRESHOLD

# Create smooth shadow lift mask
shadow_lift_mask = np.clip((SHADOW_THRESHOLD - luminance) / SHADOW_THRESHOLD, 0, 1)
shadow_lift_mask = gaussian_filter(shadow_lift_mask, sigma=3.0)

# Apply shadow lift
shadow_lift_multiplier = 2 ** SHADOW_LIFT_STOPS
shadow_lift_3d = np.stack([shadow_lift_mask] * 3, axis=2)
rgb_exposed = rgb_exposed * (1 + shadow_lift_3d * (shadow_lift_multiplier - 1))
rgb_exposed = np.clip(rgb_exposed, 0, 1)

shadow_pixels = shadow_mask.sum()
shadow_percentage = (shadow_pixels / shadow_mask.size) * 100

print(f"  ✓ Shadow regions: {shadow_pixels:,} pixels ({shadow_percentage:.1f}%)")
print(f"  ✓ Shadow lift: +{SHADOW_LIFT_STOPS:.2f} stops")

# ============================================================================
# STEP 5: CONTRAST ENHANCEMENT
# ============================================================================
print("\n[5/9] Enhancing contrast...")

# Apply midtone contrast
midpoint = 0.5
rgb_contrast = ((rgb_exposed - midpoint) * MIDTONE_CONTRAST) + midpoint
rgb_contrast = np.clip(rgb_contrast, 0, 1)

print(f"  ✓ Midtone contrast: {MIDTONE_CONTRAST:.2f}×")

# ============================================================================
# STEP 6: POOL WATER COLOR CORRECTION
# ============================================================================
print("\n[6/9] Correcting pool water color...")

# Detect pool water (blue-dominant pixels)
# Pool water: high blue channel, cyan hue
r, g, b = rgb_contrast[:,:,0], rgb_contrast[:,:,1], rgb_contrast[:,:,2]
water_mask = (b > r * 1.1) & (b > g * 1.0) & (b > 0.3) & (b < 0.9)

# Smooth mask to avoid hard edges
water_mask_smooth = gaussian_filter(water_mask.astype(np.float32), sigma=10.0)

# Apply color correction to water
water_r = r * (1 - water_mask_smooth) + (r * WATER_RED_ADJUSTMENT) * water_mask_smooth
water_g = g * (1 - water_mask_smooth) + (g * WATER_GREEN_BOOST) * water_mask_smooth
water_b = b * (1 - water_mask_smooth) + (b * WATER_BLUE_REDUCTION) * water_mask_smooth

rgb_corrected = np.stack([water_r, water_g, water_b], axis=2)
rgb_corrected = np.clip(rgb_corrected, 0, 1)

water_pixels = water_mask.sum()
water_percentage = (water_pixels / water_mask.size) * 100

print(f"  ✓ Water regions: {water_pixels:,} pixels ({water_percentage:.1f}%)")
print(f"  ✓ Color shift: R {WATER_RED_ADJUSTMENT:.2f}×, G {WATER_GREEN_BOOST:.2f}×, B {WATER_BLUE_REDUCTION:.2f}×")

# ============================================================================
# STEP 7: SATURATION BOOST
# ============================================================================
print("\n[7/9] Adjusting saturation...")

# Convert to HSV for saturation adjustment
rgb_uint8 = (rgb_corrected * 255).astype(np.uint8)
img_pil = Image.fromarray(rgb_uint8)

# Global saturation
enhancer = ImageEnhance.Color(img_pil)
img_saturated = enhancer.enhance(GLOBAL_SATURATION)

# Detect vegetation (green-dominant)
rgb_sat = np.array(img_saturated, dtype=np.float32) / 255.0
r, g, b = rgb_sat[:,:,0], rgb_sat[:,:,1], rgb_sat[:,:,2]
vegetation_mask = (g > r * 1.1) & (g > b * 1.05) & (g > 0.2)
vegetation_mask_smooth = gaussian_filter(vegetation_mask.astype(np.float32), sigma=5.0)

# Boost vegetation saturation
hsv = np.array(img_saturated.convert('HSV'), dtype=np.float32)
hsv[:,:,1] = hsv[:,:,1] * (1 + vegetation_mask_smooth * (VEGETATION_SAT_BOOST - 1))
hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)

img_final = Image.fromarray(hsv.astype(np.uint8), mode='HSV').convert('RGB')
rgb_sat = np.array(img_final, dtype=np.float32) / 255.0

vegetation_pixels = vegetation_mask.sum()
vegetation_percentage = (vegetation_pixels / vegetation_mask.size) * 100

print(f"  ✓ Global saturation: {GLOBAL_SATURATION:.2f}×")
print(f"  ✓ Vegetation regions: {vegetation_pixels:,} pixels ({vegetation_percentage:.1f}%)")
print(f"  ✓ Vegetation boost: {VEGETATION_SAT_BOOST:.2f}×")

# ============================================================================
# STEP 8: CLARITY ENHANCEMENT (MINIMAL)
# ============================================================================
print("\n[8/9] Applying clarity enhancement...")

# High-pass filter for clarity
blurred = gaussian_filter(rgb_sat, sigma=CLARITY_RADIUS / 4.0)
high_pass = rgb_sat - blurred

# Apply clarity with conservative strength
rgb_clarity = rgb_sat + high_pass * CLARITY_STRENGTH
rgb_clarity = np.clip(rgb_clarity, 0, 1)

print(f"  ✓ Clarity: {CLARITY_STRENGTH:.2f} @ radius {CLARITY_RADIUS}px")
print("  ⚠️  Minimal clarity used (image already sharp)")

# ============================================================================
# STEP 9: SAVE OUTPUT
# ============================================================================
print("\n[9/9] Saving enhanced image...")

# Convert to 16-bit
rgb_16bit = (rgb_clarity * 65535).astype(np.uint16)

# Save with tifffile if available
output_path = OUTPUT_DIR / "750Picacho_Pool_Enhanced.ti"

if TIFFFILE_AVAILABLE:
    tifffile.imwrite(output_path, rgb_16bit, compression='lzw')
    print(f"  ✓ Saved with tifffile: {output_path}")
else:
    img_output = Image.fromarray(rgb_16bit)
    img_output.save(output_path, compression='tiff_lzw')
    print(f"  ✓ Saved with PIL: {output_path}")

# Save comparison metrics
print("\n" + "=" * 80)
print("ENHANCEMENT SUMMARY")
print("=" * 80)
print(f"Input:  {INPUT}")
print(f"Output: {output_path}")
print(f"\nLuminance:       {original_luminance:.3f} → {rgb_clarity.mean():.3f}")
print(f"Exposure:        +{GLOBAL_EXPOSURE_LIFT:.2f} EV")
print(f"Contrast:        {MIDTONE_CONTRAST:.2f}×")
print(f"Shadow lift:     +{SHADOW_LIFT_STOPS:.2f} stops")
print(f"Saturation:      {GLOBAL_SATURATION:.2f}×")
print(f"Clarity:         {CLARITY_STRENGTH:.2f}")
print(f"\nWater corrected: {water_percentage:.1f}% of frame")
print(f"Vegetation boost: {vegetation_percentage:.1f}% of frame")
print("=" * 80)
print("✓ ENHANCEMENT COMPLETE")
print("=" * 80)
