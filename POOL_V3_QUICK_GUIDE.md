# Pool Enhancement V3 - Quick Implementation Guide

**Purpose:** Fast reference for implementing V3 fixes  
**Time Required:** 2-3 hours  
**Files to Create:** `conservative_enhance_pool_v3.py`

---

## Critical Changes from V2 → V3

### 1. Remove Line 113 (Gamma Correction)
```python
# ❌ DELETE THIS (V2 - line 113):
rgb = np.power(np.clip(rgb_linear, 0, 1), 1/GAMMA_CORRECTION)

# ✅ REPLACE WITH (V3):
rgb = apply_agx_tone_map(rgb_linear)
```

### 2. Add AgX Tone Mapping Function
```python
def apply_agx_tone_map(rgb_linear):
    """AgX tone mapping for LINEAR → display sRGB conversion."""
    # Constants
    MIN_EV = -10.0
    MAX_EV = 6.5
    
    # Convert to log space
    rgb_log = np.log2(rgb_linear + 1e-10)
    rgb_log = np.clip(rgb_log, MIN_EV, MAX_EV)
    rgb_log = (rgb_log - MIN_EV) / (MAX_EV - MIN_EV)
    
    # S-curve for smooth highlights
    rgb_compressed = rgb_log * rgb_log * (3.0 - 2.0 * rgb_log)
    
    # sRGB gamma
    return np.power(rgb_compressed, 1/2.2)
```

### 3. Update Parameters
```python
# V2 → V3 Parameter Changes
GLOBAL_EXPOSURE_LIFT = 0.0        # Was 0.15 (now in tone map)
SHADOW_LIFT_STOPS = 0.15          # Was 0.25 (reduced)
MIDTONE_CONTRAST = 1.05           # Was 1.08 (reduced)
GLOBAL_SATURATION = 1.05          # Was 1.03 (increased)
CLARITY_STRENGTH = 0.04           # Was 0.08 (reduced)
```

### 4. Fix Water Color
```python
# V2 (line 196-202) - WRONG:
WATER_GREEN_BOOST = 1.05
WATER_BLUE_REDUCTION = 0.98
WATER_RED_ADJUSTMENT = 0.99

# V3 - CORRECT (jewel-toned cyan):
WATER_RED_REDUCTION = 0.95     # -5% red (remove muddy)
WATER_GREEN_MAINTAIN = 1.00    # 0% green
WATER_BLUE_BOOST = 1.15        # +15% blue (jewel tone)
```

### 5. Add Sky Protection
```python
# ADD BEFORE LINE 174 (contrast enhancement):
def protect_sky_highlights(rgb, threshold=0.75):
    """Mask sky from aggressive adjustments."""
    luminance = 0.2126 * rgb[:,:,0] + 0.7152 * rgb[:,:,1] + 0.0722 * rgb[:,:,2]
    height = rgb.shape[0]
    y_coords = np.arange(height)[:, np.newaxis] / height
    
    sky_mask = (
        (luminance > threshold) &
        (np.abs(rgb[:,:,0] - rgb[:,:,1]) < 0.1) &
        (y_coords < 0.5)
    )
    
    return gaussian_filter(sky_mask.astype(np.float32), sigma=30.0)

# USE IT:
sky_mask = protect_sky_highlights(rgb_exposed, threshold=0.75)
protection = 1.0 - sky_mask * 0.7
rgb_exposed *= protection[:,:,np.newaxis]
```

### 6. Fix Vegetation Processing
```python
# V2 (line 236) - WRONG (boosts brightness):
hsv[:,:,1] = hsv[:,:,1] * (1 + vegetation_mask_smooth * (VEGETATION_SAT_BOOST - 1))

# V3 - CORRECT (saturation only, no brightness):
VEGETATION_SAT_BOOST = 1.06  # +6% saturation
# Keep same code but ensure no V (brightness) channel modification
# Only modify S (saturation) channel
```

---

## Complete V3 Script Structure

```python
#!/usr/bin/env python3
"""
Conservative Enhancement V3 - 750 Picacho Pool
FIXES: Proper tone mapping, highlight preservation, color accuracy
"""

# [Lines 1-71: Same as V2 - imports, config, load image]

# [NEW] Add AgX tone mapping function here
def apply_agx_tone_map(rgb_linear):
    # [See section 2 above]
    pass

# [NEW] Add sky protection function here
def protect_sky_highlights(rgb, threshold=0.75):
    # [See section 5 above]
    pass

# [Line 72-107: Same - load image]

# [Line 108-110: DELETE gamma correction section]
# ❌ REMOVE: rgb = np.power(rgb_linear, 1/GAMMA_CORRECTION)

# [NEW LINE 108] Replace with tone mapping
print(f"\n[2/9] Applying AgX tone mapping (LINEAR → display sRGB)...")
rgb = apply_agx_tone_map(rgb_linear)
print(f"  ✓ Tone mapped with highlight preservation")

# [Lines 119-145: Modify exposure lift]
# Change GLOBAL_EXPOSURE_LIFT from 0.15 to 0.0 (no additional exposure)
# Or remove this section entirely (tone map handles it)

# [Lines 146-170: Modify shadow recovery]
# Change SHADOW_LIFT_STOPS from 0.25 to 0.15
# Add sky protection BEFORE shadow lift:
sky_mask = protect_sky_highlights(rgb_exposed, threshold=0.75)
protection = 1.0 - sky_mask * 0.7
rgb_exposed *= protection[:,:,np.newaxis]

# [Lines 171-182: Modify contrast]
# Change MIDTONE_CONTRAST from 1.08 to 1.05

# [Lines 183-210: Fix water color correction]
# Update color adjustments (see section 4 above)

# [Lines 211-247: Modify saturation]
# Change GLOBAL_SATURATION from 1.03 to 1.05
# Ensure vegetation only modifies saturation, not brightness

# [Lines 248-263: Modify clarity]
# Change CLARITY_STRENGTH from 0.08 to 0.04

# [Lines 264-308: Same - save output with updated summary]
```

---

## Testing Checklist

After running V3, verify:

### Automated Metrics
```bash
python3 << 'EOF'
import numpy as np
from PIL import Image
import tifffile

# Load images
orig = tifffile.imread("input_images/750Picacho_Pool.tiff").astype(np.float32) / 65535.0
enh = tifffile.imread("processed_images/Conservative/750Picacho_Pool_Enhanced_v3.tif").astype(np.float32) / 65535.0

# Check metrics
lum_change = (enh.mean() / orig.mean() - 1) * 100
highlight_clip = (enh > 0.95).sum() / enh.size * 100
sat_orig = ((enh.max(axis=2) - enh.min(axis=2)) / (enh.max(axis=2) + 1e-10)).mean()
sat_enh = ((orig.max(axis=2) - orig.min(axis=2)) / (orig.max(axis=2) + 1e-10)).mean()
sat_change = (sat_enh / sat_orig - 1) * 100

print(f"Luminance change: {lum_change:.1f}% (target: 15-25%)")
print(f"Highlight clip: {highlight_clip:.2f}% (target: <1%)")
print(f"Saturation change: {sat_change:.1f}% (target: +5-15%)")

# Pass/fail
pass_fail = (
    15 <= lum_change <= 25 and
    highlight_clip < 1.0 and
    5 <= sat_change <= 15
)
print(f"\n{'✅ PASSED' if pass_fail else '❌ FAILED'}")
EOF
```

### Visual Inspection
- [ ] Sky gradient smooth (no white blowout)
- [ ] Pool water jewel-toned turquoise (not washed out)
- [ ] Water reflections visible
- [ ] Vegetation shadows natural (not floating)
- [ ] Overall balanced exposure
- [ ] No halos around edges
- [ ] No yellow/green color cast

---

## If V3 Still Has Issues

### Too Bright?
```python
# Reduce exposure compensation in tone map
# Or add subtle darkening post-tone-map:
rgb = rgb * 0.95  # -5% brightness
```

### Too Dark?
```python
# Add subtle brightening post-tone-map:
rgb = rgb * 1.05  # +5% brightness
```

### Water Not Cyan Enough?
```python
# Increase blue boost:
WATER_BLUE_BOOST = 1.20  # +20% blue (was 1.15)
```

### Sky Still Clipped?
```python
# Stronger sky protection:
protection = 1.0 - sky_mask * 0.85  # 85% reduction (was 70%)
```

---

## Quick Reference: V3 vs V2

| Parameter | V2 Value | V3 Value | Reason |
|-----------|----------|----------|--------|
| Tone Mapping | Gamma 2.2 | AgX | Proper LINEAR conversion |
| Exposure | +0.15 EV | 0.0 EV | Tone map handles it |
| Shadow Lift | +0.25 stops | +0.15 stops | Less aggressive |
| Contrast | 1.08× | 1.05× | More subtle |
| Saturation | 1.03× | 1.05× | Compensate for desaturation |
| Clarity | 0.08 | 0.04 | Prevent halos |
| Water Red | 0.99× | 0.95× | Remove muddiness |
| Water Blue | 0.98× | 1.15× | Jewel tone boost |
| Sky Protection | None | 70% reduction | Preserve gradient |

---

## Files Generated

After V3 completes:
- `processed_images/Conservative/750Picacho_Pool_Enhanced_v3.tif` - Main output
- Console output with metrics
- Optional: Comparison images

---

## Next Steps After V3

1. **Run automated validation** (see Testing Checklist above)
2. **Visual inspection** against original
3. **If passed:** Document final parameters, ready for production
4. **If failed:** Adjust parameters using "If V3 Still Has Issues" guide
5. **Compare V2 vs V3:** Side-by-side to confirm improvement

---

## Estimated Time

- Copy V2 script → V3: 5 minutes
- Implement AgX function: 15 minutes
- Update parameters: 10 minutes
- Add sky protection: 15 minutes
- Fix water/vegetation: 15 minutes
- Testing & validation: 30 minutes
- Parameter tuning (if needed): 30-60 minutes

**Total:** 2-3 hours for production-ready result

---

## Support

See full documentation:
- `POOL_V3_RECOMMENDATIONS.md` - Complete technical details (26KB)
- `POOL_V3_EXECUTIVE_SUMMARY.md` - High-level overview (6KB)
- `ANALYSIS_750Picacho_Pool.md` - Original image analysis

---

**Status:** ✅ Ready for implementation  
**Priority:** HIGH - Required for client delivery  
**Difficulty:** MEDIUM - Requires understanding of tone mapping concepts
