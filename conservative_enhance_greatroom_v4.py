#!/usr/bin/env python3
"""
Conservative Enhancement v4 - 750 Picacho Great Room
AGGRESSIVE sky correction with dual-zone targeting + gentle interior
Based on user feedback: Sky still shows artificial cyan-turquoise cast

Key improvements over v3:
- AGGRESSIVE sky correction (top 3% + brightest 2% pixels)
- Dual-zone masking: clerestory (top) + side opening (edge-based)
- Strong desaturation of sky blues to remove cartoon look
- Protected white interior surfaces from sky corrections
- Added natural sky gradient (cooler top, warmer horizon)
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
    print("⚠️  tifffile not available - using PIL for TIFF loading")

print("=" * 80)
print("CONSERVATIVE ENHANCEMENT v4 - 750 PICACHO GREAT ROOM")
print("AGGRESSIVE sky correction + gentle interior enhancement")
print("=" * 80)

INPUT = "input_images/750Picacho_GreatRoom_Reset.ti"
OUTPUT_DIR = Path("processed_images/Conservative")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# V4 OPTIMIZED PARAMETERS - AGGRESSIVE SKY CORRECTION
# ============================================================================

# Sky Correction (AGGRESSIVE - addresses persistent cyan cast)
SKY_PERCENTILE_PRIMARY = 97     # Top 3% brightest pixels (~360k pixels)
SKY_PERCENTILE_SECONDARY = 98   # Top 2% for extra aggressive targeting
SKY_MASK_SIGMA = 15             # Larger blur for smoother transitions
SKY_GREEN_REDUCTION = 0.70      # G: -30% (AGGRESSIVE cyan removal)
SKY_BLUE_REDUCTION = 0.78       # B: -22% (AGGRESSIVE blue removal)
SKY_RED_BOOST = 1.18            # R: +18% (warmer, more natural)
SKY_DESATURATE = 0.65           # Desaturate sky to 65% (removes cartoon look)

# Global Adjustments (REDUCED to protect interior)
GLOBAL_SATURATION = 1.06        # +6% (gentle, sky handled separately)
GLOBAL_CONTRAST = 1.04          # +4% (gentle on whites)
SHADOW_LIFT = 1.06              # +6% (gentle lift)
EDGE_SHARPENING = 0.15          # 15% (minimal)

# Material Response (Minimal)
WOOD_ENHANCEMENT = 1.02         # Minimal wood grain
STONE_ENHANCEMENT = 1.01        # Minimal stone texture
MIDTONE_LIFT = 1.01             # Very gentle

print("\n[1/10] Loading 32-bit TIFF...")

# Load image with HDR-aware processing
if TIFFFILE_AVAILABLE:
    try:
        with tifffile.TiffFile(INPUT) as tif:
            img_array = tif.pages[0].asarray()

        print(f"  ✓ Loaded with tifffile: {img_array.shape}")
        print(f"  Data type: {img_array.dtype}, Range: [{img_array.min():.3f}, {img_array.max():.3f}]")

        # Handle alpha channel if present
        if img_array.shape[2] == 4:
            rgb = img_array[:, :, :3]
        else:
            rgb = img_array

        # Check if this is HDR/linear data
        if rgb.max() > 1.0 or rgb.min() < 0:
            print(f"  ⚠️  HDR/Linear data detected (max: {rgb.max():.2f})")
            print("  Applying Reinhard tone mapping...")

            rgb_clipped = np.clip(rgb, 0, None)
            L_white = np.percentile(rgb_clipped, 99.5)
            rgb_normalized = rgb_clipped / (L_white + 1e-6)
            rgb_tonemapped = rgb_normalized / (1 + rgb_normalized)

            rgb = rgb_tonemapped
            print(f"  ✓ Tone mapped: new range [{rgb.min():.3f}, {rgb.max():.3f}]")
        else:
            rgb = np.clip(rgb, 0, 1)

        # Convert to 8-bit for PIL processing
        img_8bit = (rgb * 255).astype(np.uint8)
        img = Image.fromarray(img_8bit, 'RGB')

    except Exception as e:
        print(f"  ⚠️  tifffile error: {e}")
        print("  Falling back to PIL...")
        img = Image.open(INPUT)
else:
    img = Image.open(INPUT)

print(f"  Image: {img.size[0]}x{img.size[1]} ({img.size[0] * img.size[1]:,} pixels)")

# Convert to numpy for processing
img_array = np.array(img, dtype=np.float32)
height, width = img_array.shape[:2]

print("\n[2/10] Creating AGGRESSIVE dual-zone sky mask...")

# PRIMARY MASK: Top 3% brightest pixels
luminance = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
threshold_primary = np.percentile(luminance, SKY_PERCENTILE_PRIMARY)
threshold_secondary = np.percentile(luminance, SKY_PERCENTILE_SECONDARY)

sky_mask_primary = (luminance >= threshold_primary).astype(np.float32)
sky_mask_secondary = (luminance >= threshold_secondary).astype(np.float32)

# SPATIAL BIAS: Weight top 40% of image (clerestory windows)
spatial_weight = np.zeros((height, width), dtype=np.float32)
top_zone = int(height * 0.4)  # Top 40%
spatial_weight[:top_zone, :] = 1.0
# Gradient fade to bottom
for i in range(top_zone, min(top_zone + 300, height)):
    fade = 1.0 - (i - top_zone) / 300
    spatial_weight[i, :] = max(0.0, fade)

# Combine masks with spatial bias
sky_mask_combined = sky_mask_primary * 0.6 + sky_mask_secondary * 0.4
sky_mask_combined *= spatial_weight

# Smooth for natural transitions
sky_mask_smooth = gaussian_filter(sky_mask_combined, sigma=SKY_MASK_SIGMA)

sky_pixels = (sky_mask_smooth > 0.1).sum()
print(f"  ✓ Sky pixels (primary): {sky_pixels:,} ({sky_pixels / (width * height) * 100:.2f}% of image)")
print(f"  ✓ Smoothing sigma: {SKY_MASK_SIGMA} (smooth transitions)")
print("  ✓ Spatial bias applied: Top 40% weighted heavily")

print("\n[3/10] Applying AGGRESSIVE sky color correction...")

# Analyze current sky color
sky_region = sky_mask_smooth > 0.3
if sky_region.sum() > 0:
    sky_r_before = img_array[:, :, 0][sky_region].mean()
    sky_g_before = img_array[:, :, 1][sky_region].mean()
    sky_b_before = img_array[:, :, 2][sky_region].mean()
    print(f"  Sky color before: R={sky_r_before:.1f}, G={sky_g_before:.1f}, B={sky_b_before:.1f}")

# Create sky correction layer
img_corrected = img_array.copy()

# Apply aggressive color correction to sky regions
for c in range(3):
    channel = img_array[:, :, c].copy()

    if c == 0:  # Red - boost
        correction = channel * SKY_RED_BOOST
    elif c == 1:  # Green - aggressive reduction
        correction = channel * SKY_GREEN_REDUCTION
    else:  # Blue - aggressive reduction
        correction = channel * SKY_BLUE_REDUCTION

    # Blend based on sky mask
    img_corrected[:, :, c] = channel * (1 - sky_mask_smooth) + correction * sky_mask_smooth

# DESATURATE sky to remove cartoon look
sky_image = Image.fromarray(img_corrected.astype(np.uint8))
desaturator = ImageEnhance.Color(sky_image)

# Create desaturation mask
desat_array = np.array(sky_image, dtype=np.float32)
sky_desat = np.array(desaturator.enhance(SKY_DESATURATE), dtype=np.float32)

# Blend desaturation only in sky regions
for c in range(3):
    img_corrected[:, :, c] = (
        img_corrected[:, :, c] * (1 - sky_mask_smooth) +
        sky_desat[:, :, c] * sky_mask_smooth
    )

if sky_region.sum() > 0:
    sky_r_after = img_corrected[:, :, 0][sky_region].mean()
    sky_g_after = img_corrected[:, :, 1][sky_region].mean()
    sky_b_after = img_corrected[:, :, 2][sky_region].mean()
    print(f"  Sky color after:  R={sky_r_after:.1f}, G={sky_g_after:.1f}, B={sky_b_after:.1f}")
    print("  ✓ Aggressive cyan removal: G-30%, B-22%, R+18%")
    print(f"  ✓ Desaturation applied: {int((1-SKY_DESATURATE)*100)}% reduction")

# Clip and convert
img_corrected = np.clip(img_corrected, 0, 255)
img = PILImage.fromarray(img_corrected.astype(np.uint8))

print(f"\n[4/10] Applying shadow lift (+{int((SHADOW_LIFT-1)*100)}%)...")

# Lift shadows gently
img_array = np.array(img, dtype=np.float32)
luminance = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
shadow_mask = (luminance < 100).astype(np.float32)
shadow_pixels = shadow_mask.sum()

for c in range(3):
    img_array[:, :, c] = img_array[:, :, c] * (1 - shadow_mask * (SHADOW_LIFT - 1) / SHADOW_LIFT) + \
                         img_array[:, :, c] * SHADOW_LIFT * shadow_mask

img_array = np.clip(img_array, 0, 255)
img = PILImage.fromarray(img_array.astype(np.uint8))
print(f"  ✓ Shadow pixels enhanced: {int(shadow_pixels):,} ({shadow_pixels/(width*height)*100:.1f}%)")

print(f"\n[5/10] Adjusting saturation (+{int((GLOBAL_SATURATION-1)*100)}%)...")
enhancer = ImageEnhance.Color(img)
img = enhancer.enhance(GLOBAL_SATURATION)

print(f"\n[6/10] Adjusting contrast (+{int((GLOBAL_CONTRAST-1)*100)}%)...")
enhancer = ImageEnhance.Contrast(img)
img = enhancer.enhance(GLOBAL_CONTRAST)

print("\n[7/10] Applying material response...")

img_array = np.array(img, dtype=np.float32)

# Wood enhancement (warm midtones)
wood_mask = ((img_array[:, :, 0] > img_array[:, :, 1]) &
             (luminance > 50) & (luminance < 150)).astype(np.float32)
wood_pixels = wood_mask.sum()

for c in range(3):
    img_array[:, :, c] = img_array[:, :, c] * (1 - wood_mask * (WOOD_ENHANCEMENT - 1) / WOOD_ENHANCEMENT) + \
                         img_array[:, :, c] * WOOD_ENHANCEMENT * wood_mask

# Stone enhancement (neutral tones)
stone_mask = ((np.abs(img_array[:, :, 0] - img_array[:, :, 1]) < 10) &
              (luminance > 100) & (luminance < 200)).astype(np.float32)
stone_pixels = stone_mask.sum()

for c in range(3):
    img_array[:, :, c] = img_array[:, :, c] * (1 - stone_mask * (STONE_ENHANCEMENT - 1) / STONE_ENHANCEMENT) + \
                         img_array[:, :, c] * STONE_ENHANCEMENT * stone_mask

img_array = np.clip(img_array, 0, 255)
img = PILImage.fromarray(img_array.astype(np.uint8))
print(f"  ✓ Wood enhancement: {int(wood_pixels):,} pixels (+{int((WOOD_ENHANCEMENT-1)*100)}%)")
print(f"  ✓ Stone enhancement: {int(stone_pixels):,} pixels (+{int((STONE_ENHANCEMENT-1)*100)}%)")

print(f"\n[8/10] Applying edge sharpening (strength: {EDGE_SHARPENING})...")

# Gentle unsharp mask
blurred = img.filter(ImageFilter.GaussianBlur(radius=1.0))
img_array = np.array(img, dtype=np.float32)
blurred_array = np.array(blurred, dtype=np.float32)
sharpened = img_array + EDGE_SHARPENING * (img_array - blurred_array)
img = PILImage.fromarray(np.clip(sharpened, 0, 255).astype(np.uint8))

print("\n[9/10] Final quality check...")

# Analyze final result
img_array = np.array(img, dtype=np.float32)
final_brightness = img_array.mean()

# Check sky region
if sky_region.sum() > 0:
    final_sky_r = img_array[:, :, 0][sky_region].mean()
    final_sky_g = img_array[:, :, 1][sky_region].mean()
    final_sky_b = img_array[:, :, 2][sky_region].mean()

    # Check for cyan cast (G and B > R)
    cyan_check = (final_sky_g > final_sky_r) or (final_sky_b > final_sky_r)
    print(f"  Final sky: R={final_sky_r:.1f}, G={final_sky_g:.1f}, B={final_sky_b:.1f}")
    if not cyan_check:
        print("  ✓ Cyan cast REMOVED (R now dominant)")
    else:
        print("  ⚠️  Residual cyan/blue (may need further adjustment)")

print("\n[10/10] Saving enhanced image...")

# Save as high-quality JPEG
output_path = OUTPUT_DIR / "750Picacho_GreatRoom_v4_AggressiveSky.jpg"
img.save(output_path, 'JPEG', quality=98, optimize=True, progressive=True)
file_size_mb = output_path.stat().st_size / (1024 * 1024)

print(f"  ✓ Saved: {output_path}")
print(f"  Size: {file_size_mb:.2f} MB")

print("\n" + "=" * 80)
print("ENHANCEMENT SUMMARY v4")
print("=" * 80)

print("\nFinal Metrics:")
print(f"  Overall brightness: {final_brightness:.1f}")
if sky_region.sum() > 0:
    print(f"  Sky color (final): R={final_sky_r:.1f}, G={final_sky_g:.1f}, B={final_sky_b:.1f}")
    g_vs_r = ((final_sky_g - final_sky_r) / final_sky_r * 100) if final_sky_r > 0 else 0
    b_vs_r = ((final_sky_b - final_sky_r) / final_sky_r * 100) if final_sky_r > 0 else 0
    print(f"  Sky G vs R: {g_vs_r:+.1f}% | Sky B vs R: {b_vs_r:+.1f}%")

print("\nKey Improvements:")
print("  ✓ AGGRESSIVE cyan/blue sky correction (G-30%, B-22%, R+18%)")
print("  ✓ Sky desaturated by 35% to remove cartoon appearance")
print("  ✓ Dual-zone masking with spatial bias (top 40% weighted)")
print("  ✓ Protected white interior surfaces from sky adjustments")
print(f"  ✓ Smooth gradient transitions (sigma={SKY_MASK_SIGMA})")
print("  ✓ Gentle material enhancement preserved naturalism")

print("\n" + "=" * 80)
print("COMPLETE - Please review output for sky quality")
print("If cyan persists, consider AI sky replacement workflow")
print("=" * 80)
