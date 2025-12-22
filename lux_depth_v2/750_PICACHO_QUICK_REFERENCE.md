# 750 Picacho TIFF Testing - Quick Reference Card

**Pipeline**: Lux Depth V2  
**Files**: 6 × 16-bit TIFF (23-43 MB each, 180 MB total)  
**Status**: ✅ READY FOR TESTING

---

## Quick Start (2 Options)

### Option A: Using Test Script (Recommended)

```bash
# 1. Install dependencies (one-time)
pip install numpy opencv-python tifffile torch tqdm

# 2. Run test script
cd /home/runner/work/Transformation_Portal/Transformation_Portal
python lux_depth_v2/test_750_picacho.py

# Or use bash script:
bash lux_depth_v2/test_750_picacho.sh
```

### Option B: Manual CLI Invocation

```bash
# 1. Install dependencies (one-time)
pip install numpy opencv-python tifffile torch tqdm

# 2. Run pipeline
cd /home/runner/work/Transformation_Portal/Transformation_Portal
python -m lux_depth_v2.cli \
  --input-dir projects/750_picacho_lane/Final_Production_UltraQuality/ \
  --output-dir lux_depth_v2/test_outputs/750_picacho/ \
  --preset interior_luxury \
  --device cpu \
  --upscaler-backend torch
```

---

## Test Script Options

```bash
# Dry run (check dependencies only)
python lux_depth_v2/test_750_picacho.py --dry-run

# Use different preset
python lux_depth_v2/test_750_picacho.py --preset balanced

# Enable edge refinement (experimental)
python lux_depth_v2/test_750_picacho.py --edge-refinement

# Use GPU (if available)
python lux_depth_v2/test_750_picacho.py --device cuda

# Show help
python lux_depth_v2/test_750_picacho.py --help
```

---

## Expected Results

### Processing Time
- **CPU (4 cores)**: 24-48 minutes total (4-8 min per file)
- **CPU (8+ cores)**: 12-30 minutes total (2-5 min per file)
- **GPU (CUDA/MPS)**: 3-6 minutes total (30-60 sec per file)

### Output Files (per input)
```
750Picacho_Pool_UltraQuality_master16.tif    # 30-50 MB, 16-bit TIFF
750Picacho_Pool_UltraQuality_upscaled16.tif  # 80-120 MB, 16-bit TIFF
750Picacho_Pool_UltraQuality_marketing.png   # 5-10 MB, 8-bit PNG
750Picacho_Pool_UltraQuality_preview.jpg     # 500KB-1MB, JPEG
750Picacho_Pool_UltraQuality_report.json     # <10KB, metadata
```

### Total Outputs
- **6 master TIFFs** (16-bit, pre-upscale)
- **6 upscaled TIFFs** (16-bit, final)
- **6 marketing PNGs** (8-bit, fast review)
- **6 preview JPEGs** (thumbnails)
- **6 JSON reports** (processing metadata)
- **2 summary files** (TEST_SUMMARY.json, TEST_SUMMARY.txt)

**Total**: 32 files

---

## Validation Checklist

After processing, verify:

- [ ] All 6 files processed without errors
- [ ] 32 output files generated (6 × 5 + 2 summaries)
- [ ] Master TIFFs are 16-bit (check TEST_SUMMARY.txt)
- [ ] Processing time within expected range
- [ ] No visible artifacts in outputs
- [ ] Material detection > 60% accuracy (check JSON reports)

---

## Presets Available

| Preset | Speed | Quality | Use For |
|--------|-------|---------|---------|
| **interior_luxury** | Medium | High | Luxury interiors (RECOMMENDED) |
| **exterior_showcase** | Medium | High | Exterior/aerial views |
| **balanced** | Fast | Good | General purpose, quick tests |
| **premium** | Slow | Maximum | Absolute best quality |

---

## Troubleshooting

### Import Errors
```bash
# Fix: Install dependencies
pip install numpy opencv-python tifffile torch tqdm
```

### Out of Memory
```bash
# Fix: Use balanced preset or enable tiling
python lux_depth_v2/test_750_picacho.py --preset balanced
```

### Slow Processing
```bash
# Fix: Use GPU or balanced preset
python lux_depth_v2/test_750_picacho.py --device cuda
# OR
python lux_depth_v2/test_750_picacho.py --preset balanced
```

---

## Files & Documentation

### Test Files
- **Test Script (Python)**: `lux_depth_v2/test_750_picacho.py`
- **Test Script (Bash)**: `lux_depth_v2/test_750_picacho.sh`
- **Readiness Checklist**: `lux_depth_v2/750_PICACHO_READINESS_CHECKLIST.md`
- **This Quick Reference**: `lux_depth_v2/750_PICACHO_QUICK_REFERENCE.md`

### Source Files
```
projects/750_picacho_lane/Final_Production_UltraQuality/
├── 750Picacho_Aerial_UltraQuality.tif (29 MB)
├── 750Picacho_GreatRoom_UltraQuality.tif (24 MB)
├── 750Picacho_Kitchen_UltraQuality.tif (23 MB)
├── 750Picacho_Pool_UltraQuality.tif (26 MB)
├── 750Picacho_PrimaryBathroom_UltraQuality.tif (43 MB)
└── 750Picacho_PrimaryBedroom_UltraQuality.tif (35 MB)
```

### Output Directory
```
lux_depth_v2/test_outputs/750_picacho/
├── [30 output files from processing]
├── TEST_SUMMARY.json
└── TEST_SUMMARY.txt
```

---

## Quick Decision Guide

**Ready to test?** → Run dry run first:
```bash
python lux_depth_v2/test_750_picacho.py --dry-run
```

**Dependencies not installed?** → Install:
```bash
pip install numpy opencv-python tifffile torch tqdm
```

**Need fastest test?** → Use balanced preset:
```bash
python lux_depth_v2/test_750_picacho.py --preset balanced
```

**Want best quality?** → Use default (interior_luxury):
```bash
python lux_depth_v2/test_750_picacho.py
```

**Have GPU?** → Enable it:
```bash
python lux_depth_v2/test_750_picacho.py --device cuda
```

---

## Success Criteria

### ✅ PASS
- All 6 files process without errors
- 32 output files generated
- Master TIFFs are 16-bit
- Processing time < 10 min/file (CPU)
- No visible artifacts

### ❌ FAIL
- Any file fails to process
- Output bit depth < 16-bit
- Processing time > 10 min/file (CPU)
- Visible artifacts in outputs

---

## Contact & Support

**Documentation**: See `lux_depth_v2/750_PICACHO_READINESS_CHECKLIST.md` for detailed guide

**Issues**: Create GitHub issue during feature freeze period (use template)

**Questions**: Refer to main README and Phase 2 documentation

---

**Last Updated**: December 22, 2025  
**Status**: Production Ready - Test Execution Pending
