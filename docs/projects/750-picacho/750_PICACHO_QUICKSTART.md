# 750 Picacho Lane - Quick Start Guide

**GOAL:** Absolute maximum quality processing for luxury real estate delivery

---

## ✅ System Status

- **Hardware:** Apple M4 Max (40-core GPU) ✅
- **Pipeline:** Unified Luxury Pipeline (16-bit TIFF fix applied) ✅
- **Dependencies:** All installed ✅
- **Tests:** 510/511 passing (99.8%) ✅

**YOU ARE READY TO PROCESS**

---

## Option 1: Process Single Test Image (RECOMMENDED FIRST STEP)

```bash
cd /Users/rc/Transformation_Portal

# Test with Pool view
python -c "
from transformation_portal.pipelines import process_luxury_render, ProcessingProfile

outputs = process_luxury_render(
    '/Users/rc/Desktop/Cache/750_LightFiction_Final_Views/16-Bit_EXRs/750Picacho_Pool.exr',
    profile=ProcessingProfile.PREMIUM,
    output_dir='/Users/rc/Desktop/Cache/TEST_OUTPUT/',
    scene_type='auto'
)

print('\n✅ Processing complete! Outputs:')
for fmt, path in outputs.items():
    print(f'  {fmt}: {path.name}')
"
```

**Expected output:** 5 files in `/Users/rc/Desktop/Cache/TEST_OUTPUT/`
- `750Picacho_Pool_MASTER.tiff` (16-bit, ~80-120 MB)
- `750Picacho_Pool_print_8K.jpg` (Q98, ~15-20 MB)
- `750Picacho_Pool_web_4K.jpg` (Q96, ~5-8 MB)
- `750Picacho_Pool_magazine_2K.jpg` (Q95, ~2-3 MB)
- `750Picacho_Pool_social_1080p.jpg` (Q92, ~0.8-1.2 MB)

**Time:** 2-5 minutes

---

## Option 2: Batch Process All 7 Views

```bash
cd /Users/rc/Transformation_Portal

python -c "
from transformation_portal.pipelines import batch_process_luxury_renders, ProcessingProfile

stats = batch_process_luxury_renders(
    input_dir='/Users/rc/Desktop/Cache/750_LightFiction_Final_Views/16-Bit_EXRs/',
    output_dir='/Users/rc/Desktop/Cache/750_LightFiction_Final_Views/FINALS_16BIT/',
    profile=ProcessingProfile.PREMIUM,
    scene_type='auto',
    save_statistics=True
)

print(f'\n✅ Complete!')
print(f'Processed: {stats.images_processed} images')
print(f'Total time: {stats.total_time/60:.1f} minutes')
print(f'Avg time: {stats.total_time/stats.images_processed:.1f} seconds/image')
"
```

**Time:** 14-35 minutes (7 images × 2-5 min each)

---

## Option 3: Use Run Script

```bash
cd /Users/rc/Transformation_Portal

python run_unified_pipeline.py \
    --input-dir /Users/rc/Desktop/Cache/750_LightFiction_Final_Views/16-Bit_EXRs/ \
    --output-dir /Users/rc/Desktop/Cache/750_LightFiction_Final_Views/FINALS/ \
    --profile premium \
    --formats all \
    --scene-type auto
```

---

## Verify Quality

After processing, verify TIFFs are true 16-bit:

```bash
python diagnose_tiff_quality.py /path/to/output/*_MASTER.tiff
```

**Expected:**
- ✅ dtype: uint16
- ✅ Bits per sample: 16
- ✅ Status: OK

---

## Optional: Speed Up with CoreML (3-4x faster)

```bash
# Download Neural Engine optimized models
python download_depth_models.py --coreml

# This downloads:
# - vits: 100 MB (fastest, good quality)
# - vitb: 350 MB (balanced)
# - vitl: 1.3 GB (best quality, slower)
```

**Recommended:** Download `vits` and `vitb` for 3x speedup

---

## Processing Profiles

| Profile | Time/Image | Quality | Use For |
|---------|-----------|---------|---------|
| **PREMIUM** | 2-5 min | Maximum | Final deliverables (RECOMMENDED) |
| **BALANCED** | 30-90 sec | High | Client review |
| **PERFORMANCE** | 10-30 sec | Good | Quick iteration |

---

## Troubleshooting

**Issue:** Import error
```bash
# Fix: Install package in development mode
pip install -e .
```

**Issue:** Out of memory
```bash
# Fix: Use BALANCED profile instead of PREMIUM
```

**Issue:** TIFF looks degraded
```bash
# Verify it's true 16-bit
python diagnose_tiff_quality.py output/*_MASTER.tiff

# If dtype shows uint8, the fix wasn't applied - report this!
```

---

## Quick Quality Check

Visual inspection checklist:
- [ ] No banding in skies
- [ ] Smooth gradients on walls
- [ ] Shadow detail visible (not crushed)
- [ ] Highlights recoverable (not blown)
- [ ] TIFF file ~2x larger than JPEG (confirms 16-bit)

---

## Next Steps After Processing

1. **Archive masters** - Keep `*_MASTER.tiff` files (16-bit)
2. **Client delivery:**
   - Portfolio/web → `*_web_4K.jpg`
   - Print marketing → `*_print_8K.jpg`
   - Social media → `*_social_1080p.jpg`
   - Magazine → `*_magazine_2K.jpg`

---

**Ready to process?** Run Option 1 (test single image) first to verify quality!
