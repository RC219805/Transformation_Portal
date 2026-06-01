# LUT & Pipeline Analysis Report
## 750 Picacho Lane - Color Cast Investigation

> **Historical 750 Picacho project record**
>
> This November 2025 analysis is retained as point-in-time evidence. Paths under
> `projects/750_picacho_lane/` are historical references only; current operator
> guidance starts at [Documentation Map](../governance/DOCUMENTATION_MAP.md).

**Date**: November 10, 2025
**Analysis**: Transformation Portal Specialist
**Status**: ROOT CAUSE IDENTIFIED

---

## Executive Summary

✅ **LUTs are NOT the problem** - They have a warm bias that actually helps
❌ **Real-ESRGAN upscaling is ADDING blue cast**
✅ **Auto white balance is working correctly**

---

## LUT Analysis Results

### 1. Montecito Golden Hour HDR
**File**: `assets/luts/location_aesthetic/California/Montecito_Golden_Hour_HDR.cube`

- **Size**: 33x33x33 (35,937 entries)
- **File size**: 948KB
- **Color Bias**:
  - Red: +7.17% (warmer)
  - Green: +2.83%
  - Blue: **-9.99%** (LESS blue)
- **Midtone Analysis**:
  - Blue/Red ratio: 0.84x
  - **WARM BIAS**: Red is 19% higher than blue

✅ **VERDICT**: This LUT REDUCES blue, doesn't add it!

### 2. Kodak 2393 D55 HDR
**File**: `assets/luts/film_emulation/Kodak/Kodak_2393_D55_HDR.cube`

- **Size**: 33x33x33 (35,937 entries)
- **File size**: 948KB
- **Color Bias**:
  - Red: 0.00%
  - Green: 0.00%
  - Blue: 0.00%
- **Midtone Analysis**:
  - Blue/Red ratio: 1.01x
  - **NEUTRAL**: Perfectly balanced

✅ **VERDICT**: This LUT is color-neutral

### Combined Effect (Montecito 70% + Kodak 50%)
- **Estimated Blue/Red**: 0.91x
- **Net effect**: WARM (less blue, more red)

✅ **LUTs are HELPING, not hurting!**

---

## Pipeline Stage Color Tracking

| Stage | Red | Green | Blue | Blue/Red | Status |
|-------|-----|-------|------|----------|--------|
| **Source** | 0.366 | 0.334 | 0.573 | **1.57x** | ❌ BLUE CAST |
| **After White Balance** | 0.424 | 0.424 | 0.424 | **1.00x** | ✅ PERFECT |
| **After Tone Mapping** | 89.1 | 77.8 | 95.7 | **1.07x** | ✅ GOOD |
| **After Color Grading (LUTs)** | - | - | - | - | (Disabled in test) |
| **After Real-ESRGAN** | 64.5 | 69.6 | 97.3 | **1.51x** | ❌ BLUE CAST! |

---

## Root Cause: Real-ESRGAN Upscaling

### Problem Identified
**Real-ESRGAN is adding blue cast during upscaling**:
- Input (tonemapped): Blue/Red = 1.07x ✅
- Output (upscaled): Blue/Red = 1.51x ❌
- **Change**: +0.44x blue shift

### Why This Happens
Real-ESRGAN was trained on specific datasets that may have color biases. When upscaling, the model:
1. Converts RGB → BGR (for OpenCV)
2. Runs inference (adds blue bias here)
3. Converts BGR → RGB

The blue cast is being introduced during the AI inference step.

---

## Solutions

### Option 1: Disable Upscaling ⭐ RECOMMENDED SHORT-TERM
```python
upscaling=UpscalingConfig(
    enabled=False,  # Disable Real-ESRGAN
    method="esrgan",
    scale_factor=4.0,
),
```

**Pros**:
- Immediate fix for blue cast
- Faster processing
- Still get good quality at 4K x 2.25K

**Cons**:
- No 4x upscaling
- Lower final resolution

### Option 2: Use Lanczos Instead
```python
upscaling=UpscalingConfig(
    enabled=True,
    method="lanczos",  # Traditional algorithm, no AI bias
    scale_factor=4.0,
),
```

**Pros**:
- No color bias
- Still get 4x upscaling
- Deterministic results

**Cons**:
- Less AI "enhancement"
- Won't add detail like ESRGAN

### Option 3: Color-Correct After Upscaling
Add a post-upscaling white balance step:

```python
# After upscaling, re-apply white balance
if upscaling_enabled:
    image_upscaled = self._auto_white_balance(image_upscaled, strength=0.5)
```

**Pros**:
- Keep ESRGAN quality
- Fix color cast

**Cons**:
- Two white balance steps
- More complex pipeline

### Option 4: Replace Real-ESRGAN Model ⭐ RECOMMENDED LONG-TERM
Find or train a color-neutral Real-ESRGAN model:
- Try different ESRGAN variants
- Look for models trained on architectural images
- Consider BSRGAN or other alternatives

---

## LUT Upgrade Recommendations

### Current LUTs: ✅ GOOD QUALITY

The existing LUTs are well-made:
- Proper 33x33x33 resolution
- HDR-optimized domains
- Good color characteristics

### No Immediate Upgrade Needed

The LUTs are NOT causing the blue cast problem. However, for future enhancement:

### Potential Additions:
1. **More Location Styles**:
   - Malibu (cooler, oceanfront)
   - Palm Springs (desert warmth)
   - San Francisco (fog/cool tones)

2. **Time of Day Variations**:
   - Blue hour variants
   - Midday variants
   - Sunset variants

3. **Material-Specific LUTs**:
   - Stone-optimized
   - Wood-optimized
   - Glass-optimized

### LUT Best Practices (Already Followed):
✅ 33x33x33 size (good balance of quality/performance)
✅ HDR domain (0.0-1.0 range)
✅ Descriptive naming
✅ Organized directory structure

---

## Immediate Action Plan

### Step 1: Fix Blue Cast Now
```bash
# Edit luxury_estate_master_pipeline.py
# Change upscaling method from "esrgan" to "lanczos"
# OR disable upscaling entirely
```

### Step 2: Reprocess Pool Image
```bash
python luxury_estate_master_pipeline.py \
  --preset 750_picacho \
  --room-type pool \
  --output-dir output_750_picacho_v2large_FIXED \
  "projects/750_picacho_lane/Final_Production_UltraQuality/750Picacho_Pool_UltraQuality.tif"
```

### Step 3: Verify Results
Check that Blue/Red ratio < 1.15x in final output

### Step 4: Process Remaining Images
Once pool looks good, process all 6 images with corrected settings

---

## Long-Term Recommendations

1. **Find Better Upscaling Model**:
   - Research color-neutral ESRGAN variants
   - Test on architectural imagery
   - Validate color accuracy

2. **Add Post-Processing White Balance**:
   - Optional final color correction
   - Safety net for any color drift

3. **Expand LUT Library** (Optional):
   - More location styles
   - More film stocks
   - Client-specific looks

4. **Document Color Pipeline**:
   - Track color space conversions
   - Validate at each stage
   - Add color accuracy tests

---

## Conclusion

### What We Learned:
✅ LUTs are high-quality and color-appropriate
✅ Auto white balance works perfectly
✅ Tone mapping preserves color well
❌ Real-ESRGAN adds unwanted blue cast

### The Fix:
**Switch from Real-ESRGAN to Lanczos upscaling** or disable upscaling for now.

### LUT Status:
**No upgrade needed** - current LUTs are excellent and NOT causing issues.

---

**Report Complete**: November 10, 2025 10:00 AM
**Next Step**: Implement upscaling fix and reprocess
**Status**: Ready for production with corrected settings
