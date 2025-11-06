#!/usr/bin/env python3
"""
Conservative Enhancement v6 - 750 Picacho Great Room
PRECISION SKY CORRECTION: Surgical sky fix + protected interior preservation

Based on accumulated knowledge:
- Sky: Overly saturated cyan/turquoise (visible through clerestory + large opening)
- Interior: Warm lighting, white surfaces must be preserved
- Approach: Precision masking to isolate sky from interior

Key improvements:
- Dual-threshold sky detection (brightness + color profile)
- Separate sky and interior processing pipelines
- Edge-aware blending to prevent artifacts
- Preserve interior color temperature and brightness
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
print("CONSERVATIVE ENHANCEMENT v6 - 750 PICACHO GREAT ROOM")
print("Precision sky correction + protected interior preservation")
print("=" * 80)

INPUT = "input_images/750Picacho_GreatRoom.tiff"  # Using original TIFF
OUTPUT_DIR = Path("processed_images/Conservative")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# V6 PRECISION PARAMETERS - SURGICAL SKY CORRECTION
# ============================================================================

# Sky Detection (Dual-threshold approach)
SKY_BRIGHTNESS_THRESHOLD = 0.85  # Top 15% brightness (likely sky)
SKY_CYAN_THRESHOLD = 1.15        # B/R ratio > 1.15 indicates cyan cast
SKY_BLUE_MIN = 0.75              # Minimum blue value (0-1) to be considered sky
SKY_MASK_BLUR = 20               # Edge-aware blur for smooth transitions
SKY_EROSION_ITERATIONS = 2       # Erode mask to protect edges

# Sky Correction (Applied only to detected sky regions)
SKY_BLUE_REDUCTION = 0.70        # B: -30% (remove cyan/turquoise)
SKY_GREEN_REDUCTION = 0.75       # G: -25% (reduce cyan component)
SKY_RED_BOOST = 1.20             # R: +20% (warmer, natural blue)
SKY_SATURATION = 0.60            # Desaturate sky to 60% (remove cartoon look)

# Interior Preservation (NO sky corrections applied)
INTERIOR_SATURATION = 1.04       # +4% gentle enhancement
INTERIOR_CONTRAST = 1.02         # +2% minimal contrast
INTERIOR_BRIGHTNESS = 1.00       # No change - preserve as-is

# Global Finishing (Applied to composite)
GLOBAL_SHARPNESS = 0.12          # 12% edge enhancement
MIDTONE_CONTRAST = 1.02          # Gentle S-curve

print(f"\n[1/9] Loading TIFF: {INPUT}")

# Load image
if TIFFFILE_AVAILABLE:
    try:
        img_array = tifffile.imread(INPUT)
        print(f"  ✓ Loaded with tifffile: {img_array.shape}, dtype: {img_array.dtype}")
        
        # Normalize to 0-1 range
        if img_array.dtype == np.uint8:
            rgb = img_array.astype(np.float32) / 255.0
        elif img_array.dtype == np.uint16:
            rgb = img_array.astype(np.float32) / 65535.0
        else:
            rgb = img_array.astype(np.float32)
            if rgb.max() > 1.0:
                rgb = np.clip(rgb / rgb.max(), 0, 1)
        
        # Handle alpha if present
        if rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]
            
    except Exception as e:
        print(f"  ⚠️  tifffile failed: {e}, falling back to PIL")
        TIFFFILE_AVAILABLE = False

if not TIFFFILE_AVAILABLE:
    img = Image.open(INPUT)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    rgb = np.array(img, dtype=np.float32) / 255.0
    print(f"  ✓ Loaded with PIL: {rgb.shape}")

print(f"  Range: [{rgb.min():.3f}, {rgb.max():.3f}]")

# ============================================================================
# [2/9] SKY DETECTION - DUAL THRESHOLD APPROACH
# ============================================================================
print(f"\n[2/9] Detecting sky regions...")

r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

# Brightness-based detection
brightness = (r + g + b) / 3.0
bright_mask = brightness > SKY_BRIGHTNESS_THRESHOLD

# Color-based detection (cyan/blue cast)
# Avoid division by zero
cyan_ratio = np.where(r > 0.01, b / (r + 0.001), 0)
cyan_mask = (cyan_ratio > SKY_CYAN_THRESHOLD) & (b > SKY_BLUE_MIN)

# Combine masks
sky_mask_raw = bright_mask & cyan_mask

# Morphological operations to clean mask
from scipy.ndimage import binary_erosion, binary_dilation
sky_mask_clean = binary_erosion(sky_mask_raw, iterations=SKY_EROSION_ITERATIONS)
sky_mask_clean = binary_dilation(sky_mask_clean, iterations=SKY_EROSION_ITERATIONS)

# Smooth edges
sky_mask_smooth = gaussian_filter(sky_mask_clean.astype(np.float32), sigma=SKY_MASK_BLUR)
sky_mask_smooth = np.clip(sky_mask_smooth, 0, 1)

sky_pixels = np.sum(sky_mask_smooth > 0.5)
sky_percentage = (sky_pixels / sky_mask_smooth.size) * 100

print(f"  ✓ Sky detected: {sky_pixels:,} pixels ({sky_percentage:.1f}% of image)")
print(f"  ✓ Mask smoothing: σ={SKY_MASK_BLUR}")

# ============================================================================
# [3/9] SKY CORRECTION - SURGICAL COLOR ADJUSTMENT
# ============================================================================
print(f"\n[3/9] Applying surgical sky correction...")

# Create corrected sky
sky_r = r * SKY_RED_BOOST
sky_g = g * SKY_GREEN_REDUCTION
sky_b = b * SKY_BLUE_REDUCTION

# Stack corrected channels
sky_corrected = np.stack([sky_r, sky_g, sky_b], axis=2)

# Desaturate sky
sky_gray = np.mean(sky_corrected, axis=2, keepdims=True)
sky_corrected = sky_gray + (sky_corrected - sky_gray) * SKY_SATURATION

print(f"  ✓ Color correction: R×{SKY_RED_BOOST:.2f}, G×{SKY_GREEN_REDUCTION:.2f}, B×{SKY_BLUE_REDUCTION:.2f}")
print(f"  ✓ Desaturation: {SKY_SATURATION:.0%}")

# ============================================================================
# [4/9] INTERIOR ENHANCEMENT - GENTLE ADJUSTMENTS
# ============================================================================
print(f"\n[4/9] Enhancing interior regions...")

# Convert to PIL for controlled adjustments
interior_img = Image.fromarray((rgb * 255).astype(np.uint8))

# Saturation
interior_img = ImageEnhance.Color(interior_img).enhance(INTERIOR_SATURATION)

# Contrast
interior_img = ImageEnhance.Contrast(interior_img).enhance(INTERIOR_CONTRAST)

interior_enhanced = np.array(interior_img, dtype=np.float32) / 255.0

print(f"  ✓ Saturation: {INTERIOR_SATURATION:.2%}")
print(f"  ✓ Contrast: {INTERIOR_CONTRAST:.2%}")
print(f"  ✓ Brightness: preserved (no adjustment)")

# ============================================================================
# [5/9] COMPOSITE - BLEND SKY AND INTERIOR
# ============================================================================
print(f"\n[5/9] Compositing sky and interior...")

# Expand mask to 3 channels
sky_mask_3d = np.stack([sky_mask_smooth] * 3, axis=2)

# Blend: sky_corrected where mask=1, interior_enhanced where mask=0
composite = sky_mask_3d * sky_corrected + (1 - sky_mask_3d) * interior_enhanced

print(f"  ✓ Edge-aware blending complete")
print(f"  Range: [{composite.min():.3f}, {composite.max():.3f}]")

# ============================================================================
# [6/9] MIDTONE CONTRAST - GENTLE S-CURVE
# ============================================================================
print(f"\n[6/9] Applying midtone contrast...")

# Simple S-curve
composite_curve = composite * MIDTONE_CONTRAST
composite_curve = np.clip(composite_curve, 0, 1)

# Blend back to preserve highlights and shadows
alpha = 0.5  # 50% blend
composite = alpha * composite_curve + (1 - alpha) * composite

print(f"  ✓ Midtone enhancement: {MIDTONE_CONTRAST:.2%}")

# ============================================================================
# [7/9] SHARPENING - EDGE ENHANCEMENT
# ============================================================================
print(f"\n[7/9] Applying edge sharpening...")

composite_pil = Image.fromarray((composite * 255).astype(np.uint8))
sharpened = ImageEnhance.Sharpness(composite_pil).enhance(1 + GLOBAL_SHARPNESS)

print(f"  ✓ Sharpness: +{GLOBAL_SHARPNESS:.0%}")

# ============================================================================
# [8/9] SAVE OUTPUT
# ============================================================================
print(f"\n[8/9] Saving output...")

output_path = OUTPUT_DIR / "750Picacho_GreatRoom_v6.tiff"

# Save as 16-bit TIFF to preserve quality
final_array = np.array(sharpened, dtype=np.uint8)
final_16bit = (final_array.astype(np.uint16) * 257)  # Scale 8-bit to 16-bit

if TIFFFILE_AVAILABLE:
    tifffile.imwrite(output_path, final_16bit, photometric='rgb')
    print(f"  ✓ Saved 16-bit TIFF: {output_path}")
else:
    sharpened.save(output_path, compression='tiff_deflate')
    print(f"  ✓ Saved TIFF: {output_path}")

print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

# ============================================================================
# [9/9] SUMMARY
# ============================================================================
print(f"\n[9/9] PROCESSING COMPLETE")
print("=" * 80)
print(f"Input:  {INPUT}")
print(f"Output: {output_path}")
print()
print("ADJUSTMENTS APPLIED:")
print(f"  Sky Correction:")
print(f"    - Color: R×{SKY_RED_BOOST:.2f}, G×{SKY_GREEN_REDUCTION:.2f}, B×{SKY_BLUE_REDUCTION:.2f}")
print(f"    - Saturation: {SKY_SATURATION:.0%}")
print(f"    - Coverage: {sky_percentage:.1f}% of image")
print(f"  Interior Enhancement:")
print(f"    - Saturation: {INTERIOR_SATURATION:.2%}")
print(f"    - Contrast: {INTERIOR_CONTRAST:.2%}")
print(f"    - Brightness: preserved")
print(f"  Global Finishing:")
print(f"    - Midtone contrast: {MIDTONE_CONTRAST:.2%}")
print(f"    - Sharpness: +{GLOBAL_SHARPNESS:.0%}")
print("=" * 80)
print("\n✓ Enhancement complete! Sky corrected while preserving interior warmth.")
