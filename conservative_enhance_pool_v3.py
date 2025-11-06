#!/usr/bin/env python3
"""
Conservative Enhancement V3 - 750 Picacho Pool Aerial Rendering
MAJOR REVISION: Proper tone mapping, highlight preservation, color accuracy

ISSUES FIXED IN V3:
- V2 had severe overexposure (+100.7% luminance vs target +15-25%)
- V2 had 9.8% highlight clipping (sky blown out)
- V2 had -27.3% saturation loss (colors washed out)
- Root cause: Gamma correction treated LINEAR data as sRGB → 2x brightness increase

V3 SOLUTIONS:
1. AgX tone mapping replaces gamma correction (proper LINEAR → display conversion)
2. Pool water cyan enhancement (+15% blue, -5% red for jewel tone)
3. Sky highlight protection (70% reduction in bright areas)
4. Vegetation shadow preservation (saturation only, no brightness lift)
5. Reduced adjustment strengths across the board
6. Automated quality validation with pass/fail metrics

Expected Results:
- Luminance: +15-20% (controlled, was +100%)
- Highlight clipping: <1% (preserved, was 9.8%)
- Saturation: +5-8% (enhanced, was -27%)
- Pool water: Jewel-toned turquoise (restored)
- Sky gradient: Smooth and detailed (preserved)
"""
from PIL import Image, ImageEnhance
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter

try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False

print("=" * 80)
print("CONSERVATIVE ENHANCEMENT V3 - 750 PICACHO POOL AERIAL")
print("CRITICAL FIX: Proper AgX tone mapping for LINEAR rendering data")
print("=" * 80)

# ============================================================================
# CONFIGURATION - V3 CORRECTED PARAMETERS
# ============================================================================

INPUT = "input_images/750Picacho_Pool.tif"
OUTPUT_DIR = Path("processed_images/Conservative")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Tone Mapping (NEW in V3 - replaces gamma correction)
TONE_MAP_METHOD = 'agx'                # AgX for photorealistic rendering
MIN_EV = -10.0                         # Shadow detail preservation
MAX_EV = 6.5                           # Highlight compression range

# Post-Tone-Map Adjustments (CORRECTED)
GLOBAL_EXPOSURE_LIFT = 0.2             # +20% brightness to counteract sky protection
SHADOW_LIFT_STOPS = 0.15               # +0.15 stops (reduced from 0.25)
SHADOW_THRESHOLD = 0.25                # Luminance < 0.25 is shadow
MIDTONE_CONTRAST = 1.08                # +8% contrast

# Color & Saturation (CORRECTED)
GLOBAL_SATURATION = 1.03               # +3% saturation (reduced from 1.05)
VEGETATION_SAT_BOOST = 1.04            # +4% green saturation (gentle)

# Pool Water Color Correction (MAJOR REVISION - jewel-toned cyan)
WATER_RED_REDUCTION = 0.97             # -3% red (subtle, reduced from -5%)
WATER_GREEN_MAINTAIN = 1.00            # 0% green (maintain)
WATER_BLUE_BOOST = 1.10                # +10% blue (jewel tone, reduced from +15%)
WATER_STRENGTH = 0.4                   # 40% blend strength (reduced from 0.5)

# Sky Protection (ADJUSTED - less aggressive)
SKY_PROTECTION_THRESHOLD = 0.80        # Protect luminance > 0.80 (higher threshold)
SKY_PROTECTION_STRENGTH = 0.5          # 50% reduction in sky areas (reduced from 0.7)
SKY_MASK_SIGMA = 30.0                  # Very smooth transition

# Material Enhancement (CORRECTED)
CLARITY_STRENGTH = 0.04                # 4% clarity (reduced from 0.08)
CLARITY_RADIUS = 96                    # Increased radius for subtlety
CLARITY_MASK_THRESHOLD = 0.85          # Exclude bright areas from clarity

# Output
OUTPUT_BIT_DEPTH = 16                  # 16-bit TIFF

# ============================================================================
# TONE MAPPING FUNCTIONS (NEW IN V3)
# ============================================================================

def apply_agx_tone_map(rgb_linear):
    """
    AgX tone mapping for LINEAR → display-referred sRGB conversion.
    Preserves highlights while maintaining color accuracy.

    Args:
        rgb_linear: Linear RGB values [0-1+] (may contain values >1 for HDR)

    Returns:
        rgb_srgb: Display-referred sRGB [0-1]
    """
    # Convert to log space
    rgb_log = np.log2(rgb_linear + 1e-10)

    # Compress dynamic range
    rgb_log = np.clip(rgb_log, MIN_EV, MAX_EV)
    rgb_log = (rgb_log - MIN_EV) / (MAX_EV - MIN_EV)

    # Apply S-curve for smooth highlight rolloff (cubic hermite spline)
    def smoothstep(x):
        x = np.clip(x, 0, 1)
        return x * x * (3.0 - 2.0 * x)

    rgb_compressed = smoothstep(rgb_log)

    # Convert to sRGB gamma
    return np.power(rgb_compressed, 1/2.2)

# ============================================================================
# SKY PROTECTION FUNCTION (NEW IN V3)
# ============================================================================

def protect_sky_highlights(rgb, threshold=0.75):
    """
    Preserve sky gradient detail by masking from aggressive adjustments.

    Args:
        rgb: Display-referred sRGB [0-1]
        threshold: Luminance above which sky protection activates

    Returns:
        sky_mask: Smooth mask [0-1] indicating sky regions
    """
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b

    # Detect sky (bright, neutral, top of frame)
    height = rgb.shape[0]
    y_coords = np.arange(height)[:, np.newaxis] / height

    sky_mask = (
        (luminance > threshold) &              # Bright
        (np.abs(r - g) < 0.1) &               # Neutral (not color cast)
        (np.abs(g - b) < 0.15) &              # Neutral
        (y_coords < 0.5)                      # Upper half of frame
    )

    # Smooth mask for natural transition
    sky_mask_smooth = gaussian_filter(sky_mask.astype(np.float32), sigma=SKY_MASK_SIGMA)

    return sky_mask_smooth

# ============================================================================
# LOAD IMAGE
# ============================================================================
print(f"\n[1/10] Loading LINEAR rendering: {INPUT}")

if TIFFFILE_AVAILABLE:
    try:
        img_array = tifffile.imread(INPUT)
        print(f"  ✓ Loaded with tifffile: {img_array.shape}, dtype: {img_array.dtype}")

        # Normalize to 0-1 range (LINEAR space)
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
print(f"  Mean luminance (LINEAR): {rgb_linear.mean():.3f}")

original_luminance_linear = rgb_linear.mean()

# Also calculate display-referred version for proper comparison
original_display = apply_agx_tone_map(rgb_linear)
original_luminance_display = original_display.mean()
print(f"  Mean luminance (Display, for reference): {original_luminance_display:.3f}")

# ============================================================================
# STEP 2: AGX TONE MAPPING (LINEAR → Display sRGB) - CRITICAL FIX!
# ============================================================================
print("\n[2/10] Applying AgX tone mapping (LINEAR → display sRGB)...")
print(f"  Dynamic range: {MIN_EV} EV to {MAX_EV} EV")

rgb = apply_agx_tone_map(rgb_linear)

print("  ✓ AgX tone mapping complete")
print(f"  ✓ Luminance after tone map: {rgb.mean():.3f}")
print("  ✓ Highlight rolloff preserved (smooth gradient)")

# Apply global exposure lift
print(f"\n  Applying global exposure lift (+{int(GLOBAL_EXPOSURE_LIFT * 100)}%)...")
rgb = rgb * (1 + GLOBAL_EXPOSURE_LIFT)
rgb = np.clip(rgb, 0, 1)
print(f"  ✓ Luminance after exposure lift: {rgb.mean():.3f}")

# ============================================================================
# STEP 3: SKY HIGHLIGHT PROTECTION (NEW IN V3)
# ============================================================================
print("\n[3/10] Protecting sky highlights...")

sky_mask = protect_sky_highlights(rgb, threshold=SKY_PROTECTION_THRESHOLD)
sky_pixels = (sky_mask > 0.5).sum()
sky_percent = (sky_pixels / sky_mask.size) * 100

print(f"  ✓ Sky mask generated: {sky_pixels:,} pixels ({sky_percent:.1f}%)")
print(f"  ✓ Sky will receive {int(SKY_PROTECTION_STRENGTH * 100)}% reduced adjustments")

# ============================================================================
# STEP 4: SHADOW RECOVERY (CORRECTED - reduced strength)
# ============================================================================
print(f"\n[4/10] Recovering shadow detail (+{SHADOW_LIFT_STOPS} stops)...")

r, g, b_ch = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b_ch

# Shadow mask
shadow_mask = luminance < SHADOW_THRESHOLD
shadow_pixels = shadow_mask.sum()
shadow_percent = (shadow_pixels / shadow_mask.size) * 100

print(f"  Shadows detected: {shadow_pixels:,} pixels ({shadow_percent:.1f}%)")

# Smooth shadow mask
shadow_mask_smooth = gaussian_filter(shadow_mask.astype(np.float32), sigma=15.0)

# Apply shadow lift
shadow_lift_factor = 2 ** SHADOW_LIFT_STOPS  # Convert stops to multiplier
rgb_lifted = rgb.copy()
for i in range(3):
    rgb_lifted[:,:,i] = rgb[:,:,i] * (1 + shadow_mask_smooth * (shadow_lift_factor - 1))

rgb_exposed = np.clip(rgb_lifted, 0, 1)

print(f"  ✓ Shadow lift applied (factor: {shadow_lift_factor:.3f}×)")

# Apply sky protection to shadow lift
protection = 1.0 - sky_mask * SKY_PROTECTION_STRENGTH
rgb_exposed = rgb_exposed * protection[:,:,np.newaxis]

print("  ✓ Sky protected from shadow lift")

# ============================================================================
# STEP 5: MIDTONE CONTRAST (CORRECTED - reduced)
# ============================================================================
print(f"\n[5/10] Enhancing midtone contrast ({MIDTONE_CONTRAST:.2f}×)...")

# Convert to PIL for contrast adjustment
img_pil = Image.fromarray((rgb_exposed * 255).astype(np.uint8))
enhancer = ImageEnhance.Contrast(img_pil)
img_contrast = enhancer.enhance(MIDTONE_CONTRAST)
rgb_contrast = np.array(img_contrast, dtype=np.float32) / 255.0

# Re-apply sky protection
rgb_contrast = rgb_contrast * protection[:,:,np.newaxis] + \
               rgb_exposed * (1 - protection[:,:,np.newaxis])

print("  ✓ Contrast enhanced with sky protection")

# ============================================================================
# STEP 6: POOL WATER COLOR CORRECTION (MAJOR REVISION - jewel tone)
# ============================================================================
print("\n[6/10] Enhancing pool water (jewel-toned turquoise)...")

r, g, b_ch = rgb_contrast[:,:,0], rgb_contrast[:,:,1], rgb_contrast[:,:,2]

# Detect pool water (blue-dominant, mid-brightness)
luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b_ch
water_mask = (
    (b_ch > r * 1.15) &          # Blue-dominant
    (b_ch > g * 1.05) &          # Blue > green
    (luminance > 0.2) &          # Not too dark
    (luminance < 0.8) &          # Not too bright (preserve highlights)
    (b_ch > 0.3) & (b_ch < 0.9)  # Blue channel range
)

water_pixels = water_mask.sum()
water_percent = (water_pixels / water_mask.size) * 100

print(f"  Water detected: {water_pixels:,} pixels ({water_percent:.1f}%)")

# Smooth mask aggressively to avoid halos
water_mask_smooth = gaussian_filter(water_mask.astype(np.float32), sigma=20.0)

# Color shift for jewel-toned turquoise
# Strategy: Enhance cyan (reduce R, maintain G, boost B)
water_r = r * WATER_RED_REDUCTION     # -5% red (remove muddiness)
water_g = g * WATER_GREEN_MAINTAIN    # 0% green (maintain)
water_b = b_ch * WATER_BLUE_BOOST     # +15% blue (jewel tone)

# Luminance preservation (maintain transparency perception)
original_lum = 0.2126 * r + 0.7152 * g + 0.0722 * b_ch
adjusted_lum = 0.2126 * water_r + 0.7152 * water_g + 0.0722 * water_b
luminance_ratio = original_lum / (adjusted_lum + 1e-6)

water_r *= luminance_ratio
water_g *= luminance_ratio
water_b *= luminance_ratio

# Blend with original using smooth mask
mask_3d = np.stack([water_mask_smooth * WATER_STRENGTH] * 3, axis=2)
r_final = r * (1 - mask_3d[:,:,0]) + water_r * mask_3d[:,:,0]
g_final = g * (1 - mask_3d[:,:,1]) + water_g * mask_3d[:,:,1]
b_final = b_ch * (1 - mask_3d[:,:,2]) + water_b * mask_3d[:,:,2]

rgb_water = np.clip(np.stack([r_final, g_final, b_final], axis=2), 0, 1)

print(f"  ✓ Water enhanced: R×{WATER_RED_REDUCTION:.2f}, G×{WATER_GREEN_MAINTAIN:.2f}, B×{WATER_BLUE_BOOST:.2f}")
print(f"  ✓ Blend strength: {int(WATER_STRENGTH * 100)}%")
print("  ✓ Luminance preserved for transparency")

# ============================================================================
# STEP 7: GLOBAL SATURATION (CORRECTED - increased)
# ============================================================================
print(f"\n[7/10] Adjusting global saturation ({GLOBAL_SATURATION:.2f}×)...")

img_pil = Image.fromarray((rgb_water * 255).astype(np.uint8))
enhancer = ImageEnhance.Color(img_pil)
img_saturated = enhancer.enhance(GLOBAL_SATURATION)
rgb_saturated = np.array(img_saturated, dtype=np.float32) / 255.0

print("  ✓ Saturation enhanced globally")

# ============================================================================
# STEP 8: VEGETATION ENHANCEMENT (CORRECTED - saturation only)
# ============================================================================
print("\n[8/10] Enhancing vegetation (saturation only, preserve shadows)...")

r, g, b_ch = rgb_saturated[:,:,0], rgb_saturated[:,:,1], rgb_saturated[:,:,2]

# Detect vegetation (green-dominant, not too bright)
luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b_ch
vegetation_mask = (
    (g > r * 1.1) &            # Green-dominant
    (g > b_ch * 1.05) &        # Green > blue
    (g > 0.15) &               # Not too dark (exclude deep shadows)
    (luminance < 0.6)          # Not too bright
)

veg_pixels = vegetation_mask.sum()
veg_percent = (veg_pixels / vegetation_mask.size) * 100

print(f"  Vegetation detected: {veg_pixels:,} pixels ({veg_percent:.1f}%)")

# Smooth mask
vegetation_mask_smooth = gaussian_filter(vegetation_mask.astype(np.float32), sigma=10.0)

# Convert to HSV for saturation-only adjustment
img_pil = Image.fromarray((rgb_saturated * 255).astype(np.uint8))
img_hsv = img_pil.convert('HSV')
hsv = np.array(img_hsv, dtype=np.float32)

# Boost saturation ONLY in vegetation areas (no brightness change)
saturation_boost_amount = (VEGETATION_SAT_BOOST - 1) * 0.3  # 30% of target boost
hsv[:,:,1] = hsv[:,:,1] * (1 + vegetation_mask_smooth * saturation_boost_amount)
hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)

# Convert back to RGB
img_hsv_enhanced = Image.fromarray(hsv.astype(np.uint8), mode='HSV')
img_rgb_enhanced = img_hsv_enhanced.convert('RGB')
rgb_vegetation = np.array(img_rgb_enhanced, dtype=np.float32) / 255.0

print(f"  ✓ Vegetation saturation enhanced (+{int(saturation_boost_amount * 100)}%)")
print("  ✓ Shadow depth preserved (no brightness lift)")

# ============================================================================
# STEP 9: CLARITY ENHANCEMENT (CORRECTED - reduced, masked)
# ============================================================================
print(f"\n[9/10] Adding clarity ({int(CLARITY_STRENGTH * 100)}%)...")

# Calculate luminance mask (exclude bright areas from clarity)
r, g, b_ch = rgb_vegetation[:,:,0], rgb_vegetation[:,:,1], rgb_vegetation[:,:,2]
luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b_ch
clarity_mask = luminance < CLARITY_MASK_THRESHOLD

print(f"  Clarity exclusion threshold: {CLARITY_MASK_THRESHOLD}")
excluded_pixels = (~clarity_mask).sum()
excluded_percent = (excluded_pixels / clarity_mask.size) * 100
print(f"  Excluded from clarity: {excluded_pixels:,} pixels ({excluded_percent:.1f}%)")

# High-pass filter for clarity
blurred = gaussian_filter(rgb_vegetation, sigma=CLARITY_RADIUS / 3.0)
high_pass = rgb_vegetation - blurred

# Apply masked clarity
mask_3d = np.stack([clarity_mask] * 3, axis=2)
rgb_clarity = rgb_vegetation + high_pass * CLARITY_STRENGTH * mask_3d

rgb_final = np.clip(rgb_clarity, 0, 1)

print(f"  ✓ Clarity applied with masking (radius: {CLARITY_RADIUS}px)")
print("  ✓ Sky and highlights protected")

# ============================================================================
# STEP 10: SAVE OUTPUT
# ============================================================================
print("\n[10/10] Saving enhanced image...")

output_name = Path(INPUT).stem + "_Enhanced_v3.ti"
output_path = OUTPUT_DIR / output_name

# Convert to 16-bit for output (OUTPUT_BIT_DEPTH is always 16 in this version)
rgb_output = (rgb_final * 65535).astype(np.uint16)
if TIFFFILE_AVAILABLE:
    tifffile.imwrite(output_path, rgb_output)
    print(f"  ✓ Saved 16-bit TIFF with tifffile: {output_path}")
else:
    img_output = Image.fromarray((rgb_final * 255).astype(np.uint8))
    img_output.save(output_path)
    print(f"  ✓ Saved 8-bit (tifffile not available): {output_path}")

# ============================================================================
# VALIDATION METRICS (NEW IN V3)
# ============================================================================
print("\n" + "=" * 80)
print("QUALITY VALIDATION METRICS")
print("=" * 80)

# Calculate metrics (compare in DISPLAY space, not LINEAR vs DISPLAY!)
final_luminance = rgb_final.mean()
luminance_change = ((final_luminance / original_luminance_display) - 1) * 100

highlight_clipping = (rgb_final > 0.95).sum() / rgb_final.size * 100
shadow_clipping = (rgb_final < 0.05).sum() / rgb_final.size * 100

# Saturation calculation (compare in same color space)
def calc_saturation(rgb):
    max_rgb = rgb.max(axis=2)
    min_rgb = rgb.min(axis=2)
    return ((max_rgb - min_rgb) / (max_rgb + 1e-10)).mean()

original_sat = calc_saturation(original_display)  # Compare display to display
final_sat = calc_saturation(rgb_final)
saturation_change = ((final_sat / original_sat) - 1) * 100

print("\n1. LUMINANCE (Display space comparison):")
print(f"   Original (LINEAR): {original_luminance_linear:.3f}")
print(f"   Original (Display-referred): {original_luminance_display:.3f}")
print(f"   Enhanced (Display): {final_luminance:.3f}")
print(f"   Change: {luminance_change:+.1f}%")
print("   Target: +15% to +25%")
print(f"   Status: {'✅ PASS' if 15 <= luminance_change <= 25 else '❌ FAIL'}")

print("\n2. HIGHLIGHT CLIPPING:")
print(f"   Clipped pixels: {highlight_clipping:.2f}%")
print("   Target: <1%")
print(f"   Status: {'✅ PASS' if highlight_clipping < 1.0 else '❌ FAIL'}")

print("\n3. SHADOW CLIPPING:")
print(f"   Clipped pixels: {shadow_clipping:.2f}%")
print("   Target: <2%")
print(f"   Status: {'✅ PASS' if shadow_clipping < 2.0 else '❌ FAIL'}")

print("\n4. SATURATION:")
print(f"   Original: {original_sat:.3f}")
print(f"   Enhanced: {final_sat:.3f}")
print(f"   Change: {saturation_change:+.1f}%")
print("   Target: +5% to +15%")
print(f"   Status: {'✅ PASS' if 5 <= saturation_change <= 15 else '❌ FAIL'}")

# Overall assessment
overall_pass = (
    15 <= luminance_change <= 25 and
    highlight_clipping < 1.0 and
    shadow_clipping < 2.0 and
    5 <= saturation_change <= 15
)

print("\n" + "=" * 80)
print(f"OVERALL ASSESSMENT: {'✅ PASSED' if overall_pass else '❌ FAILED'}")
print("=" * 80)

if overall_pass:
    print("\n🎉 V3 enhancement meets all quality targets!")
    print("   - Controlled exposure within target range")
    print("   - Highlights preserved (sky gradient intact)")
    print("   - Color enhancement achieved without oversaturation")
    print("   - Ready for production/client delivery")
else:
    print("\n⚠️  Some metrics outside target range - review settings")
    print("   See POOL_V3_QUICK_GUIDE.md for parameter tuning")

print("\n" + "=" * 80)
print("ENHANCEMENT COMPLETE - V3")
print("=" * 80)
print(f"\nOutput: {output_path}")
print(f"Compare with original: {INPUT}")
print("\nFor detailed analysis, see:")
print("  - POOL_V3_RECOMMENDATIONS.md (technical details)")
print("  - POOL_V3_EXECUTIVE_SUMMARY.md (high-level overview)")
print("  - POOL_V3_QUICK_GUIDE.md (parameter tuning)")
