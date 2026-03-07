# Runtime Skew Investigation - 750 Picacho GreatRoom

**Date:** 2026-02-09
**Context:** APEX production run revealed one image (GreatRoom) taking 8.41s vs ~1.3-1.7s median
**Ratio:** 5× slower than median

---

## Investigation Summary

### Image Properties (GreatRoom)
```
File: V2_750Picacho_GreatRoom.tiff
Size: 4000×3000 (1.333 aspect ratio)
Mode: RGB (8-bit)
File size: 68.7 MB
Compression: raw (uncompressed)
DPI: 1200×1200
XMP metadata: 7077 bytes
ICC profile: 3048 bytes (sRGB IEC61966-2.1)
```

### Comparison with Other Images
| Image              | Resolution  | Aspect | Size    |
|--------------------|-------------|--------|---------|
| Aerial             | 6000×3600   | 1.667  | 395.9MB |
| **GreatRoom**      | **4000×3000** | **1.333** | **68.7MB** |
| Kitchen            | 6000×3375   | 1.778  | 115.9MB |
| Pool               | 6000×3375   | 1.778  | 115.9MB |
| PrimaryBathroom    | 8000×6000   | 1.333  | 274.7MB |
| PrimaryBedroom     | 6000×4000   | 1.500  | 137.4MB |

### Key Findings

**✅ NOT the cause:**
1. **Resolution** - GreatRoom is actually SMALLER (4000×3000 vs 6000× for others)
2. **Compression** - Uses raw (uncompressed), same as typical TIFFs
3. **Bit depth** - Standard 8-bit RGB (not 16-bit)
4. **Color space** - Standard sRGB IEC61966-2.1
5. **Metadata overhead** - XMP (7KB) and ICC (3KB) are typical

**🔍 Possible Causes (Requires Further Investigation):**
1. **Processing History** - XMP reveals complex Photoshop + Topaz Gigapixel history
   - Multiple saves/conversions from PSD → JPEG → current TIFF
   - Could trigger unusual PIL/Pillow codepaths

2. **Dynamic Range / Content Complexity**
   - Mean pixel value: 169.80 (fairly bright)
   - Need to check if depth estimation is slower on high-key images

3. **Memory/Cache Effects**
   - Processing order may matter (if GreatRoom was processed when VRAM was fragmented)
   - Batch position could affect inference time

4. **Model Warm-up**
   - If GreatRoom was processed early, model may have been warming up
   - First few images often take longer

### Recommendations

1. **Add Per-Image Timing Breakdown** ✅ IMPLEMENTED
   - Now logs warnings for >5× median runtime
   - Captured in batch manifest stats

2. **Re-test GreatRoom in Isolation**
   - Process GreatRoom alone to see if it's consistently slow
   - Compare with processing it in different batch positions

3. **Profile Depth Estimation Substages**
   - Break down timing into: load → preprocess → inference → postprocess
   - Identify which stage is slow

4. **Check for PIL/Pillow TIFF Decoder Issues**
   - The JPEG compression metadata in XMP (from Photoshop conversion) might confuse PIL
   - Try converting to clean TIFF and re-testing

### Status

- [x] Runtime outlier detection implemented
- [x] Batch manifest now records outlier metadata
- [ ] Isolated re-test of GreatRoom image
- [ ] Substage timing profiling
- [ ] Clean TIFF conversion test

---

## Code Changes Made

See:
- `src/transformation_portal/lux_depth_v3/batch_stats.py` - Added `detect_runtime_outliers()`
- `src/transformation_portal/lux_depth_v3/orchestrator.py` - Integrated outlier detection
- `tests/test_apex_artifact_assertions.py` - Added outlier detection tests

**Next Steps:** If runtime skew persists in future runs, add per-image timing breakdown to capture load/inference/postprocess substages separately.
