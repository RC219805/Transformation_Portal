#!/usr/bin/env python3
"""
Conservative Enhancement V2 - 750 Picacho Pool Aerial Rendering
CORRECTED: Reduced global exposure, targeted adjustments only

Based on quality evaluation feedback:
- Previous version was too bright overall
- Vegetation oversaturated
- Lost some water transparency
- Excessive clarity created halos

CORRECTIONS IN V2:
1. Reduced global exposure: 0.25 → 0.15 EV (more conservative)
2. Reduced shadow lift: 0.35 → 0.25 stops (preserve depth)
3. Reduced global saturation: 1.06 → 1.03
4. Reduced vegetation boost: 1.05 → 1.02
5. Reduced clarity: 0.12 → 0.08 (minimal enhancement)
6. More subtle water correction to preserve transparency
7. Added luminance-aware processing to protect highlights
"""
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter

try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False

print("=" * 80)
print("CONSERVATIVE ENHANCEMENT V2 - 750 PICACHO POOL AERIAL")
print("Corrected: Reduced exposure, subtle adjustments, preserve transparency")
print("=" * 80)

# ============================================================================
# CONFIGURATION - CORRECTED PARAMETERS
# ============================================================================

INPUT = "input_images/750Picacho_Pool.tif"
OUTPUT_DIR = Path("processed_images/Conservative")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Exposure & Tone (CORRECTED - more conservative)
GAMMA_CORRECTION = 2.2                # Linear → sRGB (CRITICAL!)
GLOBAL_EXPOSURE_LIFT = 0.15           # +0.15 EV (reduced from 0.25)
SHADOW_LIFT_STOPS = 0.25              # +0.25 stops (reduced from 0.35)
SHADOW_THRESHOLD = 0.25               # Luminance < 0.25 is shadow
HIGHLIGHT_PROTECTION = 0.88           # Protect pixels > 0.88 (more protection)
MIDTONE_CONTRAST = 1.08               # +8% contrast (reduced from 1.10)

# Color & Saturation (CORRECTED - more subtle)
GLOBAL_SATURATION = 1.03              # +3% saturation (reduced from 1.06)
VEGETATION_SAT_BOOST = 1.02           # +2% green saturation (reduced from 1.05)

# Pool Water Color Correction (CORRECTED - more subtle)
WATER_GREEN_BOOST = 1.05              # +5% green (reduced from 1.08)
WATER_BLUE_REDUCTION = 0.98           # -2% blue (reduced from 0.96)
WATER_RED_ADJUSTMENT = 0.99           # -1% red (reduced from 0.98)

# Material Enhancement (CORRECTED - minimal)
CLARITY_STRENGTH = 0.08               # 8% clarity (reduced from 0.12)
CLARITY_RADIUS = 64                   # Radius for high-pass filter

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
        if img_array.dtype == np.float32 or img_array.dtype == np.float64:
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
# STEP 3: EXPOSURE LIFT (CORRECTED - more conservative)
# ============================================================================
print("\n[3/9] Lifting exposure...")

# Calculate luminance for highlight protection
luminance = 0.2126 * rgb[:,:,0] + 0.7152 * rgb[:,:,1] + 0.0722 * rgb[:,:,2]
highlight_mask = luminance > HIGHLIGHT_PROTECTION

# Calculate exposure multiplier
exposure_multiplier = 2 ** GLOBAL_EXPOSURE_LIFT

# Apply exposure with highlight protection
exposure_protect = np.ones_like(luminance)
exposure_protect[highlight_mask] = 0.5  # Reduce lift in highlights

exposure_protect_3d = np.stack([exposure_protect] * 3, axis=2)
rgb_exposed = rgb * (1 + exposure_protect_3d * (exposure_multiplier - 1))
rgb_exposed = np.clip(rgb_exposed, 0, 1)

new_luminance = rgb_exposed.mean()
highlight_pixels = highlight_mask.sum()
highlight_percentage = (highlight_pixels / highlight_mask.size) * 100

print(f"  ✓ Exposure: +{GLOBAL_EXPOSURE_LIFT:.2f} EV (×{exposure_multiplier:.3f})")
print(f"  ✓ Luminance: {original_luminance:.3f} → {new_luminance:.3f}")
print(f"  ✓ Highlights protected: {highlight_pixels:,} pixels ({highlight_percentage:.1f}%)")

# ============================================================================
# STEP 4: SHADOW RECOVERY (CORRECTED - more subtle)
# ============================================================================
print("\n[4/9] Recovering shadow detail...")

# Calculate luminance
luminance = 0.2126 * rgb_exposed[:,:,0] + 0.7152 * rgb_exposed[:,:,1] + 0.0722 * rgb_exposed[:,:,2]
shadow_mask = luminance < SHADOW_THRESHOLD

# Create smooth shadow lift mask
shadow_lift_mask = np.clip((SHADOW_THRESHOLD - luminance) / SHADOW_THRESHOLD, 0, 1)
shadow_lift_mask = gaussian_filter(shadow_lift_mask, sigma=5.0)  # More smoothing

# Apply shadow lift
shadow_lift_multiplier = 2 ** SHADOW_LIFT_STOPS
shadow_lift_3d = np.stack([shadow_lift_mask] * 3, axis=2)
rgb_exposed = rgb_exposed * (1 + shadow_lift_3d * (shadow_lift_multiplier - 1))
rgb_exposed = np.clip(rgb_exposed, 0, 1)

shadow_pixels = shadow_mask.sum()
shadow_percentage = (shadow_pixels / shadow_mask.size) * 100

print(f"  ✓ Shadow regions: {shadow_pixels:,} pixels ({shadow_percentage:.1f}%)")
print(f"  ✓ Shadow lift: +{SHADOW_LIFT_STOPS:.2f} stops (reduced for depth preservation)")

# ============================================================================
# STEP 5: CONTRAST ENHANCEMENT (CORRECTED - more subtle)
# ============================================================================
print("\n[5/9] Enhancing contrast...")

# Apply midtone contrast
midpoint = 0.5
rgb_contrast = ((rgb_exposed - midpoint) * MIDTONE_CONTRAST) + midpoint
rgb_contrast = np.clip(rgb_contrast, 0, 1)

print(f"  ✓ Midtone contrast: {MIDTONE_CONTRAST:.2f}×")

# ============================================================================
# STEP 6: POOL WATER COLOR CORRECTION (CORRECTED - more subtle)
# ============================================================================
print("\n[6/9] Correcting pool water color...")

# Detect pool water (blue-dominant pixels)
r, g, b = rgb_contrast[:,:,0], rgb_contrast[:,:,1], rgb_contrast[:,:,2]
water_mask = (b > r * 1.1) & (b > g * 1.0) & (b > 0.3) & (b < 0.9)

# Smooth mask with larger sigma for more gradual transitions
water_mask_smooth = gaussian_filter(water_mask.astype(np.float32), sigma=15.0)

# Reduce mask strength to 70% for more subtle correction
water_mask_smooth = water_mask_smooth * 0.7

# Apply color correction to water
water_r = r * (1 - water_mask_smooth) + (r * WATER_RED_ADJUSTMENT) * water_mask_smooth
water_g = g * (1 - water_mask_smooth) + (g * WATER_GREEN_BOOST) * water_mask_smooth
water_b = b * (1 - water_mask_smooth) + (b * WATER_BLUE_REDUCTION) * water_mask_smooth

rgb_corrected = np.stack([water_r, water_g, water_b], axis=2)
rgb_corrected = np.clip(rgb_corrected, 0, 1)

water_pixels = water_mask.sum()
water_percentage = (water_pixels / water_mask.size) * 100

print(f"  ✓ Water regions: {water_pixels:,} pixels ({water_percentage:.1f}%)")
print(f"  ✓ Color shift (70% strength): R {WATER_RED_ADJUSTMENT:.2f}×, G {WATER_GREEN_BOOST:.2f}×, B {WATER_BLUE_REDUCTION:.2f}×")

# ============================================================================
# STEP 7: SATURATION BOOST (CORRECTED - more subtle)
# ============================================================================
print("\n[7/9] Adjusting saturation...")

# Convert to HSV for saturation adjustment
rgb_uint8 = (rgb_corrected * 255).astype(np.uint8)
img_pil = Image.fromarray(rgb_uint8)

# Global saturation (reduced)
enhancer = ImageEnhance.Color(img_pil)
img_saturated = enhancer.enhance(GLOBAL_SATURATION)

# Detect vegetation (green-dominant)
rgb_sat = np.array(img_saturated, dtype=np.float32) / 255.0
r, g, b = rgb_sat[:,:,0], rgb_sat[:,:,1], rgb_sat[:,:,2]
vegetation_mask = (g > r * 1.1) & (g > b * 1.05) & (g > 0.2)
vegetation_mask_smooth = gaussian_filter(vegetation_mask.astype(np.float32), sigma=8.0)

# Reduce vegetation mask strength to 60% for more subtle boost
vegetation_mask_smooth = vegetation_mask_smooth * 0.6

# Boost vegetation saturation (subtle)
hsv = np.array(img_saturated.convert('HSV'), dtype=np.float32)
hsv[:,:,1] = hsv[:,:,1] * (1 + vegetation_mask_smooth * (VEGETATION_SAT_BOOST - 1))
hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)

img_final = Image.fromarray(hsv.astype(np.uint8), mode='HSV').convert('RGB')
rgb_sat = np.array(img_final, dtype=np.float32) / 255.0

vegetation_pixels = vegetation_mask.sum()
vegetation_percentage = (vegetation_pixels / vegetation_mask.size) * 100

print(f"  ✓ Global saturation: {GLOBAL_SATURATION:.2f}× (reduced)")
print(f"  ✓ Vegetation regions: {vegetation_pixels:,} pixels ({vegetation_percentage:.1f}%)")
print(f"  ✓ Vegetation boost: {VEGETATION_SAT_BOOST:.2f}× @ 60% strength (reduced)")

# ============================================================================
# STEP 8: CLARITY ENHANCEMENT (CORRECTED - minimal)
# ============================================================================
print("\n[8/9] Applying clarity enhancement...")

# High-pass filter for clarity
blurred = gaussian_filter(rgb_sat, sigma=CLARITY_RADIUS / 3.0)  # Softer blur
high_pass = rgb_sat - blurred

# Apply clarity with reduced strength
rgb_clarity = rgb_sat + high_pass * CLARITY_STRENGTH
rgb_clarity = np.clip(rgb_clarity, 0, 1)

print(f"  ✓ Clarity: {CLARITY_STRENGTH:.2f} @ radius {CLARITY_RADIUS}px (minimal)")
print("  ⚠️  Reduced to prevent halos")

# ============================================================================
# STEP 9: SAVE OUTPUT
# ============================================================================
print("\n[9/9] Saving enhanced image...")

# Convert to 16-bit
rgb_16bit = (rgb_clarity * 65535).astype(np.uint16)

# Save with tifffile if available
output_path = OUTPUT_DIR / "750Picacho_Pool_Enhanced_v2.ti"

if TIFFFILE_AVAILABLE:
    tifffile.imwrite(output_path, rgb_16bit, compression='lzw')
    print(f"  ✓ Saved with tifffile: {output_path}")
else:
    img_output = Image.fromarray(rgb_16bit)
    img_output.save(output_path, compression='tiff_lzw')
    print(f"  ✓ Saved with PIL: {output_path}")

# Save comparison metrics
print("\n" + "=" * 80)
print("ENHANCEMENT SUMMARY - VERSION 2 (CORRECTED)")
print("=" * 80)
print(f"Input:  {INPUT}")
print(f"Output: {output_path}")
print(f"\nLuminance:       {original_luminance:.3f} → {rgb_clarity.mean():.3f}")
print(f"Exposure:        +{GLOBAL_EXPOSURE_LIFT:.2f} EV (reduced)")
print(f"Contrast:        {MIDTONE_CONTRAST:.2f}×")
print(f"Shadow lift:     +{SHADOW_LIFT_STOPS:.2f} stops (reduced)")
print(f"Saturation:      {GLOBAL_SATURATION:.2f}× (reduced)")
print(f"Clarity:         {CLARITY_STRENGTH:.2f} (minimal)")
print(f"\nWater corrected: {water_percentage:.1f}% of frame @ 70% strength")
print(f"Vegetation boost: {vegetation_percentage:.1f}% of frame @ 60% strength")
print("\n🔧 CORRECTIONS FROM V1:")
print("   • Exposure reduced: 0.25 → 0.15 EV")
print("   • Shadow lift reduced: 0.35 → 0.25 stops")
print("   • Saturation reduced: 1.06 → 1.03")
print("   • Clarity reduced: 0.12 → 0.08")
print("   • Water correction at 70% strength (was 100%)")
print("   • Vegetation boost at 60% strength (was 100%)")
print("=" * 80)
print("✓ ENHANCEMENT COMPLETE - MORE CONSERVATIVE APPROACH")
print("=" * 80)
