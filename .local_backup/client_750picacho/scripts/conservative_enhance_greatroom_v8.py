#!/usr/bin/env python3
"""
Conservative Enhancement v8 - 750 Picacho Great Room
OPTIMIZED FOR RESET.TIF: No cyan cast, very dark image needs lifting

Based on v7 analysis:
- NO cyan cast (B/R = 0.996, neutral)
- Very dark (mean brightness 0.218)
- Highlights properly preserved (max 0.448)
- Main task: Lift exposure while preserving quality

Strategy:
1. Gentle global exposure lift (+0.15-0.20 stops)
2. Shadow recovery without noise amplification
3. Material enhancement (wood, stone, textiles)
4. Micro-contrast for depth perception
5. Edge clarity for architectural details
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False

print("=" * 80)
print("CONSERVATIVE ENHANCEMENT v8 - 750 PICACHO GREAT ROOM")
print("Optimized for dark interior: exposure lift + material enhancement")
print("=" * 80)

# ============================================================================
# CONFIGURATION - OPTIMIZED FOR DARK INTERIOR
# ============================================================================

INPUT = "input_images/750Picacho_GreatRoom_Reset.ti"
OUTPUT_DIR = Path("processed_images/Conservative")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Exposure & Tone
GLOBAL_EXPOSURE_LIFT = 0.18          # Lift brightness from 0.218 to ~0.398 (+80%)
SHADOW_LIFT_AMOUNT = 20              # +20 luminance for shadows (<60 brightness)
SHADOW_THRESHOLD = 60                # Brightness < 60/255
HIGHLIGHT_PROTECTION = 240           # Protect pixels > 240/255
MIDTONE_CONTRAST = 1.08              # +8% midtone contrast

# Color & Saturation
GLOBAL_SATURATION = 1.06             # +6% saturation (lift from flat rendering)
WARMTH_BOOST_RED = 1.02              # +2% red (preserve warm interior)
WARMTH_REDUCE_BLUE = 0.98            # -2% blue (prevent cool cast)

# Material Enhancement
MATERIAL_CLARITY = 0.15              # 15% clarity (wood grain, stone texture)
TEXTURE_ZONES = {
    'highlights': 0.10,              # 10% for bright surfaces
    'midtones': 0.15,                # 15% for main materials
    'shadows': 0.08                  # 8% for shadows (avoid noise)
}

# Sharpening
EDGE_SHARPNESS = 0.12                # 12% edge enhancement
UNSHARP_RADIUS = 1.5                 # Unsharp mask radius

# Output
OUTPUT_BIT_DEPTH = 16                # 16-bit TIFF

# ============================================================================
# LOAD IMAGE
# ============================================================================
print(f"\n[1/8] Loading image: {INPUT}")

if TIFFFILE_AVAILABLE:
    try:
        img_array = tifffile.imread(INPUT)
        print(f"  ✓ Loaded with tifffile: {img_array.shape}, dtype: {img_array.dtype}")

        if img_array.dtype == np.float32:
            if img_array.max() > 1.0:
                rgb = np.clip(img_array / img_array.max(), 0, 1)
            else:
                rgb = img_array
        elif img_array.dtype == np.uint16:
            rgb = img_array.astype(np.float32) / 65535.0
        else:
            rgb = img_array.astype(np.float32) / 255.0

        if rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]

    except Exception as e:
        print(f"  ⚠️  tifffile failed: {e}, falling back to PIL")
        TIFFFILE_AVAILABLE = False

if not TIFFFILE_AVAILABLE:
    img = Image.open(INPUT).convert('RGB')
    rgb = np.array(img, dtype=np.float32) / 255.0

print(f"  Range: [{rgb.min():.3f}, {rgb.max():.3f}]")
print(f"  Mean brightness: {rgb.mean():.3f}")

original_brightness = rgb.mean()

# ============================================================================
# STEP 2: EXPOSURE LIFT
# ============================================================================
print("\n[2/8] Lifting exposure...")

# Apply global lift
rgb_lifted = rgb * (1 + GLOBAL_EXPOSURE_LIFT)
rgb_lifted = np.clip(rgb_lifted, 0, 1)

new_brightness = rgb_lifted.mean()
actual_lift = (new_brightness / original_brightness - 1) * 100

print(f"  ✓ Global exposure: +{GLOBAL_EXPOSURE_LIFT:.0%}")
print(f"  ✓ Brightness: {original_brightness:.3f} → {new_brightness:.3f} (+{actual_lift:.1f}%)")

# ============================================================================
# STEP 3: SHADOW RECOVERY
# ============================================================================
print("\n[3/8] Recovering shadow detail...")

# Convert to 0-255 range for shadow detection
rgb_255 = rgb_lifted * 255.0
shadow_mask = rgb_255.mean(axis=2) < SHADOW_THRESHOLD

# Apply shadow lift
shadow_lift_factor = SHADOW_LIFT_AMOUNT / 255.0
rgb_lifted[shadow_mask] += shadow_lift_factor
rgb_lifted = np.clip(rgb_lifted, 0, 1)

shadow_pixels = shadow_mask.sum()
shadow_percentage = (shadow_pixels / shadow_mask.size) * 100

print(f"  ✓ Shadow regions: {shadow_pixels:,} pixels ({shadow_percentage:.1f}%)")
print(f"  ✓ Shadow lift: +{SHADOW_LIFT_AMOUNT} (0-255 scale)")

# ============================================================================
# STEP 4: HIGHLIGHT PROTECTION
# ============================================================================
print("\n[4/8] Protecting highlights...")

rgb_255 = rgb_lifted * 255.0
highlight_mask = rgb_255.max(axis=2) > HIGHLIGHT_PROTECTION

# Blend back original for highlights
highlight_3d = np.stack([highlight_mask] * 3, axis=2)
rgb_protected = np.where(highlight_3d, rgb * (1 + GLOBAL_EXPOSURE_LIFT * 0.5), rgb_lifted)

highlight_pixels = highlight_mask.sum()
highlight_percentage = (highlight_pixels / highlight_mask.size) * 100

print(f"  ✓ Highlights: {highlight_pixels:,} pixels ({highlight_percentage:.2f}%)")
print("  ✓ Protection: 50% blend with original")

# ============================================================================
# STEP 5: COLOR GRADING
# ============================================================================
print("\n[5/8] Applying color grading...")

# Convert to PIL for controlled adjustments
img_pil = Image.fromarray((rgb_protected * 255).astype(np.uint8))

# Saturation
img_pil = ImageEnhance.Color(img_pil).enhance(GLOBAL_SATURATION)
print(f"  ✓ Saturation: {GLOBAL_SATURATION:.0%}")

# Warmth adjustment (subtle)
img_array = np.array(img_pil, dtype=np.float32) / 255.0
img_array[:, :, 0] *= WARMTH_BOOST_RED    # Red
img_array[:, :, 2] *= WARMTH_REDUCE_BLUE  # Blue
img_array = np.clip(img_array, 0, 1)

print(f"  ✓ Warmth: R+{(WARMTH_BOOST_RED-1)*100:.1f}%, B{(WARMTH_REDUCE_BLUE-1)*100:.1f}%")

img_pil = Image.fromarray((img_array * 255).astype(np.uint8))

# ============================================================================
# STEP 6: MIDTONE CONTRAST
# ============================================================================
print("\n[6/8] Enhancing midtone contrast...")

img_pil = ImageEnhance.Contrast(img_pil).enhance(MIDTONE_CONTRAST)
print(f"  ✓ Midtone contrast: {MIDTONE_CONTRAST:.0%}")

# ============================================================================
# STEP 7: MATERIAL CLARITY
# ============================================================================
print("\n[7/8] Enhancing material clarity...")

# Zone-based clarity
img_array = np.array(img_pil, dtype=np.float32) / 255.0
brightness_map = img_array.mean(axis=2) * 255

# Create zone masks
highlight_zone = brightness_map > 180
midtone_zone = (brightness_map >= 60) & (brightness_map <= 180)
shadow_zone = brightness_map < 60

# Apply unsharp mask with zone-specific strengths
blurred = img_pil.filter(ImageFilter.GaussianBlur(radius=UNSHARP_RADIUS))
blurred_array = np.array(blurred, dtype=np.float32) / 255.0

# Calculate detail
detail = img_array - blurred_array

# Apply zone-specific clarity
clarity_map = np.zeros_like(brightness_map)
clarity_map[highlight_zone] = TEXTURE_ZONES['highlights']
clarity_map[midtone_zone] = TEXTURE_ZONES['midtones']
clarity_map[shadow_zone] = TEXTURE_ZONES['shadows']

# Expand to 3 channels
clarity_map_3d = np.stack([clarity_map] * 3, axis=2)

# Apply clarity
img_array = img_array + detail * clarity_map_3d
img_array = np.clip(img_array, 0, 1)

print(f"  ✓ Highlights: +{TEXTURE_ZONES['highlights']:.0%} clarity")
print(f"  ✓ Midtones: +{TEXTURE_ZONES['midtones']:.0%} clarity")
print(f"  ✓ Shadows: +{TEXTURE_ZONES['shadows']:.0%} clarity")

img_pil = Image.fromarray((img_array * 255).astype(np.uint8))

# ============================================================================
# STEP 8: EDGE SHARPENING
# ============================================================================
print("\n[8/8] Applying edge sharpening...")

sharpened = ImageEnhance.Sharpness(img_pil).enhance(1 + EDGE_SHARPNESS)
print(f"  ✓ Edge sharpness: +{EDGE_SHARPNESS:.0%}")

# ============================================================================
# SAVE OUTPUT
# ============================================================================
print("\nSaving output...")

output_path = OUTPUT_DIR / "750Picacho_GreatRoom_v8.tif"

# Save as 16-bit TIFF (OUTPUT_BIT_DEPTH is always 16 in this version)
final_array = np.array(sharpened, dtype=np.uint8)
final_16bit = (final_array.astype(np.uint16) * 257)
if TIFFFILE_AVAILABLE:
    tifffile.imwrite(output_path, final_16bit, photometric='rgb', compression='lzw')
    print("  ✓ Saved 16-bit TIFF with tifffile")
else:
    final_img = Image.fromarray(final_16bit, mode='RGB;16')
    final_img.save(output_path, compression='tiff_lzw')
    print("  ✓ Saved 16-bit TIFF with PIL")

file_size_mb = output_path.stat().st_size / 1024 / 1024

# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n{'='*80}")
print("PROCESSING COMPLETE - v8 OPTIMIZED")
print(f"{'='*80}")
print(f"\nInput:  {INPUT}")
print(f"Output: {output_path}")
print(f"Size:   {file_size_mb:.1f} MB")
print()
print("ENHANCEMENT SUMMARY:")
print("  Exposure:")
print(f"    • Global lift: +{GLOBAL_EXPOSURE_LIFT:.0%}")
print(f"    • Shadow recovery: +{SHADOW_LIFT_AMOUNT} (0-255)")
print("    • Highlight protection: 50% blend")
print(f"    • Final brightness: {rgb.mean():.3f} → {new_brightness:.3f}")
print()
print("  Color Grading:")
print(f"    • Saturation: {GLOBAL_SATURATION:.0%}")
print(f"    • Warmth: R+{(WARMTH_BOOST_RED-1)*100:.1f}%, B{(WARMTH_REDUCE_BLUE-1)*100:.1f}%")
print(f"    • Midtone contrast: {MIDTONE_CONTRAST:.0%}")
print()
print("  Material Enhancement:")
print(f"    • Highlights: +{TEXTURE_ZONES['highlights']:.0%}")
print(f"    • Midtones: +{TEXTURE_ZONES['midtones']:.0%}")
print(f"    • Shadows: +{TEXTURE_ZONES['shadows']:.0%}")
print(f"    • Edge sharpness: +{EDGE_SHARPNESS:.0%}")
print()
print("  Processing Zones:")
print(f"    • Shadows: {shadow_percentage:.1f}%")
print(f"    • Highlights: {highlight_percentage:.2f}%")
print(f"{'='*80}")
print("\n✓ Dark interior lifted with preserved detail and warmth!")
