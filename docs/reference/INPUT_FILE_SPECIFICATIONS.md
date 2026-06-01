# Transformation Portal - Input File Quality Specifications

> **Historical 750 Picacho project record**
>
> This November 2025 analysis is retained as point-in-time evidence. Paths under
> `projects/750_picacho_lane/` are historical references only; current operator
> guidance starts at [Documentation Map](../governance/DOCUMENTATION_MAP.md).

## Executive Summary

Based on comprehensive testing of 6 different source files for the 750 Picacho Lane project, we have determined the optimal input file specifications for the Transformation Portal pipeline.

---

## 🏆 OPTIMAL INPUT FILE SPECIFICATIONS

### **Winner: 16-bit LDR with Neutral Color Balance**

**Example:** `/Users/rc/Desktop/20251108-750Picacho_Pool_MASTER.tif`

### Technical Requirements

| Specification | Requirement | Why |
|--------------|-------------|-----|
| **Bit Depth** | **16-bit unsigned integer** | Best balance of quality and compatibility |
| **Color Space** | **sRGB or Adobe RGB** | Standard for architectural rendering |
| **Color Profile** | **Embedded ICC profile** | Ensures proper color management |
| **Compression** | **None or LZW** | Preserves maximum quality |
| **File Format** | **TIFF (.tif)** | Industry standard, lossless |
| **Software Origin** | **Adobe Lightroom HDR merge** | Professional quality processing |
| **Color Balance** | **Blue/Red ratio: 0.85x - 1.15x** | Neutral (minimal color cast) |
| **Dynamic Range** | **Full 16-bit range utilized** | Max values near 65535 |
| **File Size** | **30-60 MB for 4000x2250** | Indicates uncompressed/quality |

---

## Test Results Summary

We tested 6 different source files. Here are the results:

### Source File Comparison

| Source | Type | Bit Depth | Blue/Red | Output Contrast | Score | Rank |
|--------|------|-----------|----------|-----------------|-------|------|
| **Desktop Master** | **16-bit LDR** | **16-bit** | **0.91x ✅** | **50.5** 🏆 | **10/10** | **🥇 1st** |
| HDR_Balanced.tif | 32-bit HDR | 32-bit float | 1.09x | 33.3 | 8/10 | 🥈 2nd |
| Pool_TEST.tif | 32-bit HDR | 32-bit float | 1.86x | 30.7 | 6/10 | 🥉 3rd |
| TEST_HDR.tif | 32-bit HDR | 32-bit float | 3.96x | N/A | 5/10 | 4th |
| Pool.tiff | 8-bit LDR | 8-bit | 1.23x | 24.9 | 4/10 | 5th |
| projects/ files | 16-bit degraded | 16-bit | 1.57x | 37.2 | 3/10 | 6th |

### Detailed Test Results

#### 🥇 **Desktop Master (WINNER)**
```
File: /Users/rc/Desktop/20251108-750Picacho_Pool_MASTER.tif
Type: 16-bit LDR
Size: 52 MB (uncompressed)
Software: Adobe Lightroom 11.0 (iOS)
Color Profile: Embedded ✅

Input Metrics:
  Blue/Red ratio: 0.91x ✅ (nearly perfect neutral)
  Max value: 65535 (full 16-bit range)
  Mean: 0.277 (proper exposure)

Output Quality:
  Color balance: 1.02x ✅ EXCELLENT
  Brightness: 70.6 ✅ GOOD
  Contrast: 50.5 ✅ PERFECT
  Highlights: max 221 ✅ GOOD

✅ Best overall quality - image has depth, punch, and perfect color
```

#### 🥈 **HDR_Balanced.tif (2nd Place)**
```
File: input_images/750Picacho_Pool_HDR_Balanced.tif
Type: 32-bit float HDR
Size: 103 MB
Software: Adobe Lightroom 9.0 (Macintosh)
Color Profile: Embedded ✅

Input Metrics:
  Blue/Red ratio: 1.09x ✅ (neutral)
  HDR max: 15.04 (3.9 stops above white)
  Dynamic range: 13.9 stops
  12.1% pixels above 1.0

Output Quality:
  Color balance: 0.98x ✅ EXCELLENT
  Brightness: 100.2 ✅ PERFECT
  Contrast: 33.3 ⚠️ LOW (tone mapping compressed range)
  Highlights: max 255 ✅ EXCELLENT

⚠️ HDR data is amazing but tone mapping reduces contrast
```

#### 🥉 **Pool_TEST.tif (3rd Place)**
```
File: input_images/750Picacho_Pool_TEST.tif
Type: 32-bit float HDR
Size: 103 MB
Software: Adobe Lightroom 9.0 (Macintosh)
Color Profile: Embedded ✅

Input Metrics:
  Blue/Red ratio: 1.86x ❌ (86% excess blue)
  HDR max: 4.92 (2.3 stops above white)
  Dynamic range: 12.3 stops
  8.8% pixels above 1.0

Output Quality:
  Color balance: 1.00x ✅ PERFECT (corrected by white balance)
  Brightness: 95.4 ✅ GOOD
  Contrast: 30.7 ⚠️ LOW
  Highlights: max 255 ✅ EXCELLENT

⚠️ Blue cast corrected but still has low contrast from tone mapping
```

#### ❌ **Pool.tiff (Poor)**
```
File: input_images/750Picacho_Pool.tiff
Type: 8-bit LDR
Size: 26 MB
Software: Adobe Photoshop 25.0 (Windows)
Color Profile: Embedded ✅

Input Metrics:
  Blue/Red ratio: 1.23x ⚠️ (23% excess blue)
  Bit depth: 8-bit ❌ (limited precision)
  Max value: 255

Output Quality:
  Color balance: 1.00x ✅ PERFECT
  Brightness: 106.4 ✅ GOOD
  Contrast: 24.9 ❌ VERY LOW (8-bit banding)
  Highlights: max 183 ⚠️ LIMITED

❌ 8-bit source creates flat, posterized output
```

#### ❌ **projects/ files (Worst)**
```
File: projects/750_picacho_lane/.../750Picacho_Pool_UltraQuality.tif
Type: 16-bit LDR (degraded/re-exported)
Size: 25 MB (compressed)
Software: tifffile.py (re-exported)
Color Profile: None ❌

Input Metrics:
  Blue/Red ratio: 1.57x ❌ (57% excess blue)
  Compression: ZIP (degraded)
  No color profile

Output Quality:
  Color balance: 1.08x ⚠️ FAIR
  Brightness: 82.8 ✅ GOOD
  Contrast: 37.2 ⚠️ LOW
  Highlights: max 232 ✅ GOOD

❌ Degraded re-export with severe blue cast
```

---

## Why 16-bit LDR Beats 32-bit HDR?

### The HDR Tone Mapping Problem

Our testing revealed a **critical issue** with 32-bit HDR sources:

1. **HDR sources have 12-14 stops of dynamic range**
2. **Tone mapping compresses this to display range (8 stops)**
3. **Compression reduces contrast** (33.3 vs ideal 45-60)
4. **Result: Flat, low-contrast output** despite amazing source data

### 16-bit LDR Advantages

1. **Already tone-mapped** in Lightroom with professional controls
2. **Preserves contrast** through the pipeline (50.5 = perfect)
3. **No compression artifacts** from tone mapping
4. **Better color balance** (0.91x vs 1.09x-3.96x for HDR)
5. **Simpler pipeline** (fewer processing steps = fewer errors)

---

## Critical Quality Metrics

### Input File (Before Processing)

**Must Have:**
- ✅ Blue/Red ratio: **0.85 - 1.15** (within 15% of neutral)
- ✅ Full bit depth utilized (16-bit: values 0-65535)
- ✅ Embedded color profile (ICC)
- ✅ Professional software (Lightroom, Capture One, etc.)
- ✅ Uncompressed or lossless compression only

**Avoid:**
- ❌ 8-bit files (creates banding/posterization)
- ❌ Blue cast > 1.2x (correction has limits)
- ❌ Re-exported/degraded files
- ❌ Missing color profiles
- ❌ Lossy compression (JPEG artifacts)

### Output File (After Processing)

**Target Metrics:**
- 🎯 Blue/Red ratio: **0.95 - 1.05** (perfect neutral)
- 🎯 Mean luminance: **70 - 120** (proper brightness)
- 🎯 Contrast (std dev): **45 - 60** (depth and punch)
- 🎯 Highlights: **max 220-255** (full range)
- 🎯 Dynamic range: **>130** (1st to 99th percentile)

---

## Recommended Workflow

### For Architectural Renderings

1. **Render in 16-bit** from rendering software
2. **HDR merge in Lightroom** (if multiple exposures)
   - Export as **16-bit TIFF**
   - Apply tone curve/exposure in Lightroom
   - Do NOT export as 32-bit float
3. **Color correct** to neutral in Lightroom
   - Target Blue/Red ratio near 1.0
   - Adjust temperature/tint
4. **Export settings:**
   - Format: TIFF
   - Bit depth: 16-bit
   - Color space: sRGB or Adobe RGB
   - Compression: None
   - Include color profile: Yes

### For HDR Sources (Advanced)

If you must use 32-bit HDR:
- ⚠️ Expect lower contrast (33-35 vs 50)
- ⚠️ Requires tone mapping adjustment in pipeline
- ✅ Better for extreme dynamic range scenes
- ✅ More flexibility in post-processing

---

## File Organization

### Recommended Structure

```
project_name/
├── source_masters/          # Original 16-bit TIFF files
│   ├── Scene_01_Master.tif
│   ├── Scene_02_Master.tif
│   └── ...
├── pipeline_output/         # Processed files
│   ├── Scene_01_master.tif      (16-bit master)
│   ├── Scene_01_delivery.jpg    (8-bit delivery)
│   └── Scene_01_tonemapped.jpg  (preview)
└── quality_reports/         # Analysis reports
    └── processing_log.json
```

### Naming Convention

- **Source files:** `{Project}_{Scene}_{Type}_Master.tif`
  - Example: `750Picacho_Pool_Exterior_Master.tif`
- **Output files:** Auto-generated by pipeline
  - Master: `{filename}_master.tif` (16-bit)
  - Delivery: `{filename}_delivery.jpg` (8-bit, 95% quality)
  - Preview: `{filename}_tonemapped.jpg` (web preview)

---

## Quality Checklist

### Before Processing

- [ ] File is 16-bit TIFF
- [ ] File size > 30 MB (indicates uncompressed)
- [ ] Color profile is embedded
- [ ] Exported from professional software (Lightroom, etc.)
- [ ] Blue/Red ratio is 0.85 - 1.15
- [ ] Full 16-bit range utilized (max near 65535)
- [ ] No visible artifacts or banding
- [ ] Proper exposure (not too dark/bright)

### After Processing

- [ ] Color balance 0.95 - 1.05
- [ ] Brightness 70 - 120
- [ ] Contrast 45 - 60
- [ ] Highlights max > 220
- [ ] No clipping in shadows or highlights
- [ ] Smooth gradients (no banding)
- [ ] Material response visible (depth/surfaces)
- [ ] Color grading applied correctly

---

## Conclusion

**OPTIMAL INPUT:** 16-bit LDR TIFF from Lightroom with neutral color balance

**Key Takeaway:** More data (32-bit HDR) doesn't always mean better output. The 16-bit LDR files produce superior results because they're already professionally tone-mapped and maintain better contrast through the pipeline.

**Production Standard:** Desktop master files at `/Users/rc/Desktop/20251108-750Picacho_*_MASTER.tif`

---

## Examples

### ✅ PERFECT Input File
```
File: 20251108-750Picacho_Pool_MASTER.tif
Software: Adobe Lightroom 11.0 (iOS)
Bit Depth: 16-bit unsigned
Color Profile: Embedded (520 bytes)
Blue/Red: 0.91x
Size: 52 MB (uncompressed)
Dimensions: 4000 x 2250

Result:
  Output Color: 1.02x ✅
  Output Contrast: 50.5 ✅
  Quality: EXCELLENT
```

### ⚠️ ACCEPTABLE Input File
```
File: 750Picacho_Pool_HDR_Balanced.tif
Software: Adobe Lightroom 9.0 (Macintosh)
Bit Depth: 32-bit float
Color Profile: Embedded (520 bytes)
Blue/Red: 1.09x
HDR Max: 15.04
Size: 103 MB

Result:
  Output Color: 0.98x ✅
  Output Contrast: 33.3 ⚠️ (low)
  Quality: GOOD (but needs tone mapping tuning)
```

### ❌ POOR Input File
```
File: 750Picacho_Pool_UltraQuality.tif
Software: tifffile.py (re-export)
Bit Depth: 16-bit unsigned
Color Profile: None ❌
Blue/Red: 1.57x ❌
Size: 25 MB (ZIP compressed)

Result:
  Output Color: 1.08x ⚠️
  Output Contrast: 37.2 ⚠️
  Quality: POOR (degraded source)
```

---

**Document Version:** 1.0
**Date:** November 10, 2025
**Project:** 750 Picacho Lane Pipeline Testing
**Author:** Transformation Portal Specialist
