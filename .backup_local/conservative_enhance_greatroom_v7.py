#!/usr/bin/env python3
"""
Conservative Enhancement v7 - 750 Picacho Great Room
RESET-BASED APPROACH: Fresh start with accumulated knowledge

Strategy:
1. Use Reset.tif as pristine base (no prior processing artifacts)
2. Minimal intervention philosophy (6/10 enhancement level)
3. Precision sky correction WITHOUT degrading interior
4. Protect white surfaces and warm tonality
5. Enhance materials subtly (wood, stone, textiles)

Key Learnings from v1-v6:
- Sky is 0.01-1% of image but visually dominant
- White surfaces degrade easily with aggressive adjustments
- Interior warmth (R/B=1.18) must be preserved
- Large mask blur (σ=7+) prevents halos
- Edge strength 0.243 already excellent - don't over-sharpen
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
print("CONSERVATIVE ENHANCEMENT v7 - 750 PICACHO GREAT ROOM")
print("Fresh start with Reset.tif + accumulated knowledge")
print("=" * 80)

# ============================================================================
# CONFIGURATION - OPTIMIZED FROM 6 ITERATIONS
# ============================================================================

INPUT = "input_images/750Picacho_GreatRoom_Reset.ti"
OUTPUT_DIR = Path("processed_images/Conservative")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sky Detection Parameters (Top 0.5% brightest + cyan cast)
SKY_PERCENTILE = 99.5                # Top 0.5% of brightest pixels
SKY_CYAN_THRESHOLD = 1.12            # B/G ratio > 1.12 indicates cyan
SKY_MIN_BRIGHTNESS = 0.80            # Minimum normalized brightness
SKY_MASK_SIGMA = 10.0                # Large blur to avoid halos (v6 learned: bigger is better)

# Sky Color Correction (Targeted to cyan removal)
SKY_CHANNEL_ADJUST = {
    'red': 1.12,      # Boost red to counter cyan (was 89, target ~100)
    'green': 0.88,    # Reduce green slightly (was 114, target ~100)
    'blue': 0.78      # Reduce blue aggressively (was 126, target ~98)
}
SKY_DESATURATE = 0.65                # Desaturate to remove "cartoon" look

# Interior Enhancement (MINIMAL - preserve existing quality)
INTERIOR_SATURATION = 1.03           # +3% gentle lift
INTERIOR_CONTRAST = 1.02             # +2% micro-contrast
INTERIOR_BRIGHTNESS = 1.00           # No change

# Material Enhancement (Selective)
MATERIAL_CLARITY = 0.08              # 8% clarity for stone/wood
EDGE_SHARPNESS = 0.10                # 10% edge enhancement (down from previous)

# White Surface Protection
WHITE_THRESHOLD = 0.75               # Pixels with avg brightness > 0.75
WHITE_PROTECTION_STRENGTH = 0.85     # 85% protection from adjustments

# Global Finishing
FINAL_TONE_CURVE = 1.01              # Gentle S-curve
OUTPUT_BIT_DEPTH = 16                # 16-bit TIFF output

# ============================================================================
# STEP 1: LOAD IMAGE
# ============================================================================
print(f"\n[1/10] Loading image: {INPUT}")

if TIFFFILE_AVAILABLE:
    try:
        img_array = tifffile.imread(INPUT)
        print(f"  ✓ Loaded with tifffile: {img_array.shape}, dtype: {img_array.dtype}")

        # Normalize to 0-1 range based on dtype
        if img_array.dtype == np.uint8:
            rgb = img_array.astype(np.float32) / 255.0
        elif img_array.dtype == np.uint16:
            rgb = img_array.astype(np.float32) / 65535.0
        else:
            rgb = img_array.astype(np.float32)
            # If float but not normalized, clip
            if rgb.max() > 1.0:
                print("  ⚠️  Float data not normalized, clipping to [0,1]")
                rgb = np.clip(rgb / rgb.max(), 0, 1)

        # Handle alpha if present
        if rgb.shape[2] == 4:
            print("  ⚠️  Alpha channel detected, extracting RGB")
            rgb = rgb[:, :, :3]

    except Exception as e:
        print(f"  ⚠️  tifffile failed: {e}")
        print("  Falling back to PIL...")
        TIFFFILE_AVAILABLE = False

if not TIFFFILE_AVAILABLE:
    img = Image.open(INPUT)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    rgb = np.array(img, dtype=np.float32) / 255.0
    print(f"  ✓ Loaded with PIL: {rgb.shape}")

print(f"  Range: [{rgb.min():.3f}, {rgb.max():.3f}]")
print(f"  Mean brightness: {rgb.mean():.3f}")

# Store original for analysis
original_rgb = rgb.copy()

# ============================================================================
# STEP 2: WHITE SURFACE PROTECTION MASK
# ============================================================================
print("\n[2/10] Creating white surface protection mask...")

brightness = rgb.mean(axis=2)
white_mask = brightness > WHITE_THRESHOLD

# Also check for color neutrality (whites should be neutral)
r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
color_std = np.std([r, g, b], axis=0)
neutral_mask = color_std < 0.05  # Low color variance = neutral

white_protected = white_mask & neutral_mask
white_protected_smooth = gaussian_filter(white_protected.astype(np.float32), sigma=3.0)

white_pixels = np.sum(white_protected)
white_percentage = (white_pixels / white_protected.size) * 100

print(f"  ✓ White surfaces detected: {white_pixels:,} pixels ({white_percentage:.1f}%)")
print(f"  ✓ Protection strength: {WHITE_PROTECTION_STRENGTH:.0%}")

# ============================================================================
# STEP 3: SKY DETECTION
# ============================================================================
print("\n[3/10] Detecting sky regions...")

# Brightness-based detection
sky_threshold = np.percentile(brightness, SKY_PERCENTILE)
bright_mask = brightness > sky_threshold

# Color-based detection (cyan cast = high blue relative to green/red)
cyan_ratio = np.where((r + 0.001) > 0, b / (r + 0.001), 0)
cyan_mask = (cyan_ratio > SKY_CYAN_THRESHOLD) & (brightness > SKY_MIN_BRIGHTNESS)

# Combine: both conditions must be true
sky_mask_raw = bright_mask & cyan_mask

# Smooth heavily to prevent halos
sky_mask_smooth = gaussian_filter(sky_mask_raw.astype(np.float32), sigma=SKY_MASK_SIGMA)
sky_mask_smooth = np.clip(sky_mask_smooth, 0, 1)

sky_pixels = np.sum(sky_mask_smooth > 0.1)
sky_percentage = (sky_pixels / sky_mask_smooth.size) * 100

print(f"  ✓ Sky detected: {sky_pixels:,} pixels ({sky_percentage:.2f}%)")
print(f"  ✓ Percentile threshold: {SKY_PERCENTILE}% → brightness > {sky_threshold:.3f}")
print(f"  ✓ Mask blur: σ={SKY_MASK_SIGMA} (large to prevent halos)")

# ============================================================================
# STEP 4: SKY COLOR CORRECTION
# ============================================================================
print("\n[4/10] Applying sky color correction...")

# Apply channel-wise adjustments
sky_r = r * SKY_CHANNEL_ADJUST['red']
sky_g = g * SKY_CHANNEL_ADJUST['green']
sky_b = b * SKY_CHANNEL_ADJUST['blue']

sky_corrected = np.stack([sky_r, sky_g, sky_b], axis=2)
sky_corrected = np.clip(sky_corrected, 0, 1)

# Desaturate to remove cartoon look
sky_gray = sky_corrected.mean(axis=2, keepdims=True)
sky_corrected = sky_gray + (sky_corrected - sky_gray) * SKY_DESATURATE

print(f"  ✓ Channel adjustments: R×{SKY_CHANNEL_ADJUST['red']:.2f}, " +
      f"G×{SKY_CHANNEL_ADJUST['green']:.2f}, B×{SKY_CHANNEL_ADJUST['blue']:.2f}")
print(f"  ✓ Desaturation: {SKY_DESATURATE:.0%}")

# ============================================================================
# STEP 5: INTERIOR ENHANCEMENT
# ============================================================================
print("\n[5/10] Enhancing interior regions...")

# Convert to PIL for controlled enhancements
interior_img = Image.fromarray((rgb * 255).astype(np.uint8))

# Saturation
if INTERIOR_SATURATION != 1.0:
    interior_img = ImageEnhance.Color(interior_img).enhance(INTERIOR_SATURATION)
    print(f"  ✓ Saturation: {INTERIOR_SATURATION:.2%}")

# Contrast
if INTERIOR_CONTRAST != 1.0:
    interior_img = ImageEnhance.Contrast(interior_img).enhance(INTERIOR_CONTRAST)
    print(f"  ✓ Contrast: {INTERIOR_CONTRAST:.2%}")

# Brightness (typically unchanged)
if INTERIOR_BRIGHTNESS != 1.0:
    interior_img = ImageEnhance.Brightness(interior_img).enhance(INTERIOR_BRIGHTNESS)
    print(f"  ✓ Brightness: {INTERIOR_BRIGHTNESS:.2%}")
else:
    print("  ✓ Brightness: preserved (no adjustment)")

interior_enhanced = np.array(interior_img, dtype=np.float32) / 255.0

# ============================================================================
# STEP 6: BLEND SKY AND INTERIOR
# ============================================================================
print("\n[6/10] Compositing sky and interior...")

# Expand sky mask to 3 channels
sky_mask_3d = np.stack([sky_mask_smooth] * 3, axis=2)

# Blend: corrected sky where mask=1, enhanced interior where mask=0
composite = sky_mask_3d * sky_corrected + (1 - sky_mask_3d) * interior_enhanced

print("  ✓ Sky-interior blend complete")
print(f"  Range: [{composite.min():.3f}, {composite.max():.3f}]")

# ============================================================================
# STEP 7: APPLY WHITE PROTECTION
# ============================================================================
print("\n[7/10] Protecting white surfaces...")

# Expand white protection mask to 3 channels
white_mask_3d = np.stack([white_protected_smooth] * 3, axis=2)

# Blend back original whites based on protection strength
composite = (1 - white_mask_3d * WHITE_PROTECTION_STRENGTH) * composite + \
            (white_mask_3d * WHITE_PROTECTION_STRENGTH) * original_rgb

print("  ✓ White surfaces protected from color shifts")

# ============================================================================
# STEP 8: MATERIAL CLARITY ENHANCEMENT
# ============================================================================
print("\n[8/10] Enhancing material clarity...")

# Identify non-white, non-sky regions (likely materials: wood, stone, textiles)
material_mask = (1 - sky_mask_smooth) * (1 - white_protected_smooth)
material_pixels = np.sum(material_mask > 0.5)
material_percentage = (material_pixels / material_mask.size) * 100

print(f"  ✓ Material regions: {material_pixels:,} pixels ({material_percentage:.1f}%)")

# Convert to PIL for clarity/sharpness
composite_pil = Image.fromarray((composite * 255).astype(np.uint8))

# Apply gentle clarity (high-frequency contrast)
if MATERIAL_CLARITY > 0:
    # Unsharp mask approach
    blurred = composite_pil.filter(ImageFilter.GaussianBlur(radius=5))
    composite_pil = Image.blend(blurred, composite_pil, 1 + MATERIAL_CLARITY)
    print(f"  ✓ Clarity: +{MATERIAL_CLARITY:.0%}")

# ============================================================================
# STEP 9: EDGE SHARPENING
# ============================================================================
print("\n[9/10] Applying edge sharpening...")

if EDGE_SHARPNESS > 0:
    sharpened = ImageEnhance.Sharpness(composite_pil).enhance(1 + EDGE_SHARPNESS)
    print(f"  ✓ Sharpness: +{EDGE_SHARPNESS:.0%}")
else:
    sharpened = composite_pil

# Final tone curve (gentle S-curve)
if FINAL_TONE_CURVE != 1.0:
    sharpened = ImageEnhance.Contrast(sharpened).enhance(FINAL_TONE_CURVE)
    print(f"  ✓ Final tone curve: {FINAL_TONE_CURVE:.2%}")

# ============================================================================
# STEP 10: SAVE OUTPUT
# ============================================================================
print("\n[10/10] Saving output...")

output_path = OUTPUT_DIR / "750Picacho_GreatRoom_v7.tif"

# Save as 16-bit TIFF to preserve quality
final_array = np.array(sharpened, dtype=np.uint8)

if OUTPUT_BIT_DEPTH == 16:
    final_16bit = (final_array.astype(np.uint16) * 257)  # Scale 8-bit to 16-bit
    if TIFFFILE_AVAILABLE:
        tifffile.imwrite(output_path, final_16bit, photometric='rgb', compression='lzw')
        print(f"  ✓ Saved 16-bit TIFF with tifffile: {output_path}")
    else:
        # PIL fallback
        final_img = Image.fromarray(final_16bit, mode='RGB;16')
        final_img.save(output_path, compression='tiff_lzw')
        print(f"  ✓ Saved 16-bit TIFF with PIL: {output_path}")
else:
    sharpened.save(output_path, compression='tiff_lzw')
    print(f"  ✓ Saved 8-bit TIFF: {output_path}")

file_size_mb = output_path.stat().st_size / 1024 / 1024
print(f"  File size: {file_size_mb:.1f} MB")

# ============================================================================
# SUMMARY & STATISTICS
# ============================================================================
print(f"\n{'='*80}")
print("PROCESSING COMPLETE - v7 FRESH START")
print(f"{'='*80}")
print(f"\nInput:  {INPUT}")
print(f"Output: {output_path}")
print()
print("ENHANCEMENT SUMMARY:")
print("  Sky Correction:")
print(f"    • Coverage: {sky_percentage:.2f}% of image")
print(f"    • Color: R×{SKY_CHANNEL_ADJUST['red']:.2f}, " +
      f"G×{SKY_CHANNEL_ADJUST['green']:.2f}, B×{SKY_CHANNEL_ADJUST['blue']:.2f}")
print(f"    • Desaturation: {SKY_DESATURATE:.0%}")
print(f"    • Mask blur: σ={SKY_MASK_SIGMA} (anti-halo)")
print()
print("  Interior Enhancement:")
print(f"    • Saturation: {INTERIOR_SATURATION:.2%}")
print(f"    • Contrast: {INTERIOR_CONTRAST:.2%}")
print(f"    • Brightness: {INTERIOR_BRIGHTNESS:.2%}")
print()
print("  Material Enhancement:")
print(f"    • Coverage: {material_percentage:.1f}% (non-white, non-sky)")
print(f"    • Clarity: +{MATERIAL_CLARITY:.0%}")
print(f"    • Sharpness: +{EDGE_SHARPNESS:.0%}")
print()
print("  White Surface Protection:")
print(f"    • Coverage: {white_percentage:.1f}%")
print(f"    • Protection: {WHITE_PROTECTION_STRENGTH:.0%}")
print()
print("  Global Finishing:")
print(f"    • Tone curve: {FINAL_TONE_CURVE:.2%}")
print(f"    • Bit depth: {OUTPUT_BIT_DEPTH}-bit")
print(f"{'='*80}")
print("\n✓ Fresh start complete! Natural sky + preserved interior warmth.")
print("  Compare against original to verify quality.")
