# 750 Picacho Lane - Enhancement Roadmap & Action Plan

**Date:** November 8, 2025
**Project:** 750 Picacho Lane Luxury Estate
**Current Status:** Quality optimization phase required
**Priority:** HIGH - Client delivery dependent

---

## Quick Status Summary

| Item | Status | Notes |
|------|--------|-------|
| Source EXR files | ✅ Available | 5 views located (2 missing from stated 7) |
| Current TIFFs | ⚠️ 8-bit | Need true 16-bit conversion |
| Material Response | ✅ Complete | Good luxury metrics (0.59-0.73) |
| Unified Pipeline | ✅ Ready | Fixed Nov 8, 2025 |
| CoreML Depth | ✅ Available | M4 Max optimized |
| Delivery Package | ⏳ Pending | Awaiting re-processing |

---

## Critical Issue: 8-Bit TIFF Degradation

**Problem:** All current TIFF outputs are 8-bit (uint8) despite TIFF container, limiting quality to 256 values per channel instead of 65,536.

**Visual Impact:**
- Gradient banding in skies and smooth surfaces
- Limited shadow detail recovery
- Highlight clipping in bright areas
- Cannot withstand further post-processing
- Not suitable for large-format luxury printing

**Resolution:** Re-process through fixed unified_luxury_pipeline.py with tifffile backend (confirmed working Nov 8, 2025)

---

## Immediate Action Plan

### ✅ Phase 1: Environment Verification (5 minutes)

```bash
# Verify dependencies
pip install tifffile imagecodecs  # For true 16-bit TIFF
pip install torch torchvision     # For depth processing
pip install transformers          # For Depth Anything V2

# Verify tools
python -c "import tifffile; print('✓ tifffile')"
python -c "import torch; print('✓ PyTorch')"
which oiiotool  # For EXR conversion (optional)

# Check disk space (need ~100 GB)
df -h ~
```

### ⏳ Phase 2: EXR to 16-Bit TIFF Conversion (15 minutes)

**Option A: Using OpenImageIO (Recommended)**
```bash
# Convert all EXRs to true 16-bit TIFFs
cd ~/input_renderings_750
mkdir -p ~/750_Picacho_16bit_Sources

for exr in *.exr; do
    base=$(basename "$exr" .exr)
    oiiotool "$exr" \
        --colorconvert linear sRGB \
        -o:type=uint16 \
        ~/750_Picacho_16bit_Sources/"${base}_16bit.tif"
done
```

**Option B: Using Existing Python Script**
```bash
# If oiiotool not available, use existing script
cd /Users/rc/Transformation_Portal
python process_750_picacho.py --phase convert
```

**Verification:**
```bash
# Check first conversion
cd /Users/rc/Transformation_Portal
python << 'EOF'
import numpy as np
from PIL import Image

img = Image.open('/Users/rc/750_Picacho_16bit_Sources/750Picacho_Kitchen_16bit.tif')
arr = np.array(img)

print(f"Dtype: {arr.dtype}")  # Should be uint16
print(f"Max value: {arr.max()}")  # Should be > 255
print(f"Status: {'✅ OK' if arr.dtype == np.uint16 and arr.max() > 255 else '❌ FAILED'}")
EOF
```

### ⏳ Phase 3: Unified Luxury Pipeline (30 minutes)

**Batch Process All Views:**
```bash
cd /Users/rc/Transformation_Portal

python -m transformation_portal.pipelines.unified_luxury_pipeline \
    --input ~/750_Picacho_16bit_Sources/ \
    --output ~/750_Picacho_Final_Delivery/ \
    --profile PREMIUM \
    --scene-type AUTO \
    --enable-depth \
    --depth-model depth-anything-v2-small-coreml \
    --material-response \
    --lut assets/luts/location_aesthetic/California_Coastal_Luxury.cube \
    --formats MASTER_TIFF WEB_4K PRINT_8K MAGAZINE_2K SOCIAL \
    --save-statistics \
    --verbose

# Expected runtime: ~5-6 minutes for all 5 views
```

**Alternative: Use Wrapper Script**
```bash
python process_750_picacho.py --phase process
```

### ⏳ Phase 4: Scene-Specific Refinement (1 hour)

**Pool View (Critical - Too Dark)**
```bash
# Re-process Pool with enhanced exposure
python -m transformation_portal.pipelines.unified_luxury_pipeline \
    --input ~/750_Picacho_16bit_Sources/750Picacho_Pool_16bit.tif \
    --output ~/750_Picacho_Final_Delivery_Refined/ \
    --profile PREMIUM \
    --scene-type EXTERIOR \
    --exposure 0.25 \
    --saturation 1.15 \
    --clarity 0.22 \
    --material-response \
    --material-boost water=1.3 stone=1.1
```

**Aerial View (Enhance Clarity)**
```bash
# Re-process Aerial with atmospheric effects
python -m transformation_portal.pipelines.unified_luxury_pipeline \
    --input ~/750_Picacho_16bit_Sources/750Picacho_Aerial_16bit.tif \
    --output ~/750_Picacho_Final_Delivery_Refined/ \
    --profile PREMIUM \
    --scene-type AERIAL \
    --exposure 0.15 \
    --clarity 0.25 \
    --depth-strength 0.40
```

### ⏳ Phase 5: Quality Verification (15 minutes)

**Verify All Master TIFFs:**
```bash
cd /Users/rc/Transformation_Portal

# Quick verification script
python << 'EOF'
import numpy as np
from PIL import Image
from pathlib import Path

master_dir = Path.home() / '750_Picacho_Final_Delivery/01_Master_TIFFs_16bit'

if not master_dir.exists():
    print(f"❌ Directory not found: {master_dir}")
    exit(1)

tiffs = sorted(master_dir.glob('*.tiff')) + sorted(master_dir.glob('*.tif'))
print(f"Verifying {len(tiffs)} master TIFFs:\n")

all_ok = True
for tiff_path in tiffs:
    img = Image.open(tiff_path)
    arr = np.array(img)

    is_ok = arr.dtype == np.uint16 and arr.max() > 255
    status = "✅ OK" if is_ok else "❌ FAILED"

    print(f"{tiff_path.name}:")
    print(f"  Dtype: {arr.dtype}")
    print(f"  Max: {arr.max()}")
    print(f"  Status: {status}\n")

    if not is_ok:
        all_ok = False

print("\n" + ("✅ All verified" if all_ok else "❌ Issues found"))
EOF
```

**Check File Sizes:**
```bash
# 16-bit TIFFs should be ~800 MB each
ls -lh ~/750_Picacho_Final_Delivery/01_Master_TIFFs_16bit/

# Compare to 8-bit (should be ~2x larger)
# 8-bit: ~400 MB
# 16-bit: ~800 MB ← Expected
```

**Visual Inspection Checklist:**
1. Open in Photoshop/Affinity Photo
2. Check sky gradients for banding (should be smooth)
3. Inspect shadow areas for detail (should preserve)
4. Check highlights in windows (should not clip)
5. Compare to JPEG outputs (TIFF should be equal/better)

### ⏳ Phase 6: Delivery Package (30 minutes)

**Organize Final Deliverables:**
```bash
cd ~/750_Picacho_Final_Delivery

# Expected structure:
# 01_Master_TIFFs_16bit/     (5 files, ~800 MB each)
# 02_Web_4K/                 (5 JPEGs, ~15 MB each)
# 03_Print_8K/               (5 JPEGs, ~45 MB each)
# 04_Magazine_2K/            (5 JPEGs, ~5 MB each)
# 05_Social_Media/           (5 JPEGs, ~3 MB each)
# 06_Quality_Reports/        (JSON stats)

# Create delivery README
cat > README.md << 'EOF'
# 750 Picacho Lane - Final Delivery Package

## Contents

### 01_Master_TIFFs_16bit/
True 16-bit TIFF masters for archival and future editing
- Format: TIFF, 16-bit per channel
- Color Space: ProPhoto RGB
- Compression: LZW
- Size: ~800 MB per file
- Use: Professional editing, printing, archival

### 02_Web_4K/
High-resolution web images
- Format: JPEG
- Resolution: 3840x2160
- Color Space: sRGB
- Quality: 95%
- Size: ~15 MB per file
- Use: Website hero images, online galleries

### 03_Print_8K/
Ultra-high resolution for large format printing
- Format: JPEG
- Resolution: 7680x4320
- Color Space: Adobe RGB
- Quality: 98%
- Size: ~45 MB per file
- Use: Large format prints, trade show displays

### 04_Magazine_2K/
Editorial publication ready
- Format: JPEG
- Resolution: 2048x1152
- Color Space: CMYK
- Quality: 95%
- Size: ~5 MB per file
- Use: Magazine spreads, editorial features

### 05_Social_Media/
Optimized for social platforms
- Format: JPEG
- Resolution: 1920x1080
- Color Space: sRGB
- Quality: 90%
- Size: ~3 MB per file
- Use: Instagram, Facebook, LinkedIn

## Processing Details

- Pipeline: Unified Luxury Pipeline (PREMIUM profile)
- Depth Processing: Depth Anything V2 with CoreML acceleration
- Material Response: Scene-specific surface enhancement
- Color Grading: Santa Barbara Coastal Luxury aesthetic
- Processed: November 8, 2025

## Quality Metrics

All images verified as true 16-bit masters with:
- Tonal range: 65,536 values per channel
- Smooth gradients throughout
- Preserved shadow and highlight detail
- Professional print-ready quality

## Contact
For questions or additional formats, contact Transformation Portal team.
EOF

# Create tarball for delivery
tar -czf 750_Picacho_Final_$(date +%Y%m%d).tar.gz \
    01_Master_TIFFs_16bit/ \
    02_Web_4K/ \
    03_Print_8K/ \
    04_Magazine_2K/ \
    05_Social_Media/ \
    README.md

echo "✓ Delivery package created: 750_Picacho_Final_$(date +%Y%m%d).tar.gz"
```

---

## Scene-Specific Recommendations

### 🏠 Aerial View
**Current Status:** Moderate luxury index (0.593)
**Primary Issue:** Low luminance, moderate awe factor
**Enhancements:**
- Exposure: +0.15 EV
- Clarity: +0.25 (enhance property details)
- Atmospheric depth: +0.40 (create aerial perspective)
- Suggested LUT: California_Coastal_Aerial.cube

**Expected Improvement:**
- Luxury index: 0.593 → 0.72
- Better property prominence and hierarchy

### 🍳 Kitchen
**Current Status:** Excellent luxury index (0.730), High awe (0.98)
**Primary Issue:** None - already strong
**Fine-Tuning:**
- Maintain current exposure/contrast
- Enhance countertop specular highlights +10%
- Subtle warmth increase (+100K)
- Preserve excellent wood grain clarity

**Expected Improvement:**
- Luxury index: 0.730 → 0.78 (optimize already-strong base)

### 🛋️ Great Room
**Current Status:** Good luxury index (0.636), High awe (0.76)
**Primary Issue:** Moderate focus (0.58)
**Enhancements:**
- Vignette: +0.15 (draw eye to focal points)
- Midtone contrast: +8%
- Enhance textile materials
- Brighten window views +20%

**Expected Improvement:**
- Luxury index: 0.636 → 0.74
- Focus: 0.58 → 0.72

### 🏊 Pool (CRITICAL)
**Current Status:** Moderate luxury index (0.600), LOW luminance (0.220)
**Primary Issue:** Too dark - significantly impacts perceived quality
**Enhancements:**
- Exposure: +0.25 EV (CRITICAL)
- Water saturation: +15%
- Pool-specific Material Response
- Enhance reflections and highlights

**Expected Improvement:**
- Luxury index: 0.600 → 0.75
- Mean luminance: 0.220 → 0.280
- Water appeal significantly enhanced

### 🛏️ Primary Bedroom
**Current Status:** Excellent luxury index (0.710), Outstanding comfort (0.98)
**Primary Issue:** None - already very strong
**Fine-Tuning:**
- Maintain high comfort aesthetic
- Subtle clarity to bedding textures
- Enhance window views
- Warm color temperature (+150K)

**Expected Improvement:**
- Luxury index: 0.710 → 0.76 (maintain cohesion)

---

## Expected Timeline

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Environment setup | 5 min | 5 min |
| EXR conversion | 15 min | 20 min |
| Batch processing | 30 min | 50 min |
| Scene refinement | 60 min | 1h 50min |
| Quality verification | 15 min | 2h 5min |
| Delivery packaging | 30 min | **2h 35min** |

**Total Estimated Time:** 2.5-3 hours (including refinement iterations)

---

## Success Criteria

### ✅ Technical Quality
- [ ] All master TIFFs verified as true 16-bit (uint16)
- [ ] No gradient banding in smooth areas
- [ ] Shadow detail preserved (verified in dark areas)
- [ ] Highlight recovery in bright windows
- [ ] Material Response properly applied
- [ ] Metadata (EXIF, ICC) preserved
- [ ] File sizes ~800 MB per master TIFF

### ✅ Aesthetic Quality
- [ ] Luxury index average ≥ 0.75
- [ ] Consistent Santa Barbara coastal aesthetic
- [ ] Proper white balance and color temperature
- [ ] Enhanced but natural material rendering
- [ ] Balanced luminance across all views
- [ ] Professional finishing (sharpening, clarity)

### ✅ Deliverable Completeness
- [ ] 5 master TIFFs (16-bit, ~800 MB each)
- [ ] 5 web JPEGs (4K, ~15 MB each)
- [ ] 5 print JPEGs (8K, ~45 MB each)
- [ ] 5 magazine JPEGs (2K, ~5 MB each)
- [ ] 5 social JPEGs (1080p, ~3 MB each)
- [ ] Quality reports and statistics
- [ ] Delivery README with usage guide
- [ ] Compressed delivery package

---

## Risk Mitigation

### Missing Source Views
**Risk:** Only 5 of 7 stated views located
**Mitigation:**
- Document actual inventory (5 views)
- Confirm with client scope expectations
- Flag missing views in delivery notes

### Processing Time
**Risk:** Premium processing may take longer than expected
**Mitigation:**
- Estimated 5-6 minutes for batch is reasonable
- Can use BALANCED profile (30 sec/view) for previews
- M4 Max CoreML acceleration should meet timeline

### Disk Space
**Risk:** ~25 GB required for full delivery package
**Mitigation:**
- Verified 100+ GB available recommended
- Archive old 8-bit outputs after verification
- Clean up intermediate files after successful delivery

### Color Accuracy
**Risk:** LUT application might alter material colors
**Mitigation:**
- Use LUTs at 70-75% strength (not 100%)
- Preserve neutral reference areas
- Generate comparison samples for review
- Keep non-graded masters as backup

---

## Post-Delivery Recommendations

### For Future Projects
1. **Always start with 16-bit:** Convert EXRs to true 16-bit TIFFs immediately
2. **Use unified pipeline:** Leverage fixed pipeline for consistent quality
3. **Enable CoreML:** 3-5x speedup on M-series chips
4. **Scene-specific configs:** Create presets for common scene types
5. **Early verification:** Check first output bit depth before batch processing

### Archive Strategy
- Keep master TIFFs indefinitely (archival quality)
- Archive EXR sources after verification
- Document processing settings for future reference
- Maintain delivery package for 2+ years

### Client Communication
- Explain 16-bit quality improvement (256x tonal range)
- Highlight smooth gradients and print quality
- Provide multi-format usage guide
- Offer scene-specific optimization notes

---

## Quick Command Reference

```bash
# Phase 1: Convert
python process_750_picacho.py --phase convert

# Phase 2: Process
python process_750_picacho.py --phase process

# Phase 3: Refine (manual, per view)
# See scene-specific commands above

# Phase 4: Verify
python process_750_picacho.py --phase verify

# All phases
python process_750_picacho.py --phase all
```

---

## Conclusion

The 750 Picacho Lane project has strong foundational processing (Material Response metrics 0.59-0.73) but requires critical 16-bit TIFF re-processing to achieve professional luxury real estate quality.

**Key Actions:**
1. Convert 5 EXR sources to true 16-bit TIFFs
2. Process through fixed unified luxury pipeline
3. Apply scene-specific optimizations (especially Pool +0.25 EV)
4. Verify all outputs are true 16-bit
5. Package multi-format delivery

**Expected Outcome:**
- 256x tonal range improvement
- Smooth gradients throughout
- Professional print-ready quality
- Luxury index improvement to 0.72-0.78 range
- Complete client-ready delivery package

**Ready to proceed:** All tools verified, pipeline tested, configuration optimized.

---

**Document Version:** 1.0
**Created:** November 8, 2025
**Last Updated:** November 8, 2025
**Author:** Transformation Portal Team
**Related Files:**
- `/Users/rc/Transformation_Portal/process_750_picacho.py` (automation script)
- `docs/projects/750_PICACHO_LANE_ANALYSIS.md` (detailed analysis)
- `docs/sessions/nov-8-2025/TIFF_FIX_SUMMARY_NOV8.md` (technical fix details)
- `docs/UNIFIED_LUXURY_PIPELINE.md` (pipeline documentation)
