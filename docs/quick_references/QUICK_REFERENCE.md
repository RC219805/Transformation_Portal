# Quick Reference - Premium Pipeline Fixed

## ⚡ Quick Start

```bash
# Process single image (recommended settings)
python3 premium_pipeline_fixed.py \
  input_images/YOUR_IMAGE.tiff \
  --preset kitchen-bright \
  --output output_premium_fixed \
  --enable-4k
```

**Outputs:** 5 files (Master TIFF, Print 8K, Web 4K, Magazine 2K, Social)

---

## 🎯 Common Commands

### Kitchen Rendering
```bash
python3 premium_pipeline_fixed.py \
  input_images/Giga-V2_750Picacho_Kitchen_compatible_kitchen-bright.tiff \
  --preset kitchen-bright --output output_750picacho_final --enable-4k
```

### Pool Rendering
```bash
python3 premium_pipeline_fixed.py \
  input_images/750Picacho_Pool_compatible.tiff \
  --preset pool-luxury --output output_750picacho_final --enable-4k
```

### Great Room Rendering
```bash
python3 premium_pipeline_fixed.py \
  input_images/750Picacho_GreatRoom_Reset_compatible.tiff \
  --preset interior-dramatic --output output_750picacho_final --enable-4k
```

### Batch Process All
```bash
for img in input_images/*.tiff; do
  python3 premium_pipeline_fixed.py "$img" \
    --output output_750picacho_final --enable-4k
done
```

---

## 📊 Output Specifications

| Format | Size | Quality | File Size | Use Case |
|--------|------|---------|-----------|----------|
| **Master TIFF** | 16000×9000 | Lossless | ~400 MB | Archival |
| **Print 8K** | 8000×4500 | Q98 | ~13 MB | Large prints, brochures |
| **Web 4K** | 4000×2250 | Q96 | ~4 MB | Website heroes |
| **Magazine 2K** | 2000×1125 | Q95 | ~1 MB | Editorial, magazines |
| **Social** | 1200×675 | Q92 | ~250 KB | Instagram, Facebook |

---

## 🔧 Options

### Skip 4K Upscaling (faster, standard resolution)
```bash
python3 premium_pipeline_fixed.py input.tiff --no-4k
```

### Enable AI Enhancement (conservative, optional)
```bash
python3 premium_pipeline_fixed.py input.tiff --enable-ai --enable-4k
```

### Quiet Mode (no progress output)
```bash
python3 premium_pipeline_fixed.py input.tiff --quiet
```

### Custom Output Directory
```bash
python3 premium_pipeline_fixed.py input.tiff --output custom_dir/
```

---

## ✅ Quality Validation

### Before Delivery Checklist
- [ ] Open Print 8K at 100% zoom → No artifacts
- [ ] Check Web 4K load time → Fast on web
- [ ] Verify Magazine 2K print size → 6.7" × 3.75" at 300 DPI
- [ ] Test Social on mobile → Looks sharp
- [ ] Validate Master TIFF → Archival quality

---

## 🚫 Common Mistakes to Avoid

❌ **Don't use old premium pipeline** (quality issues)  
✅ **Use premium_pipeline_fixed.py instead**

❌ **Don't use quality < 95 for print**  
✅ **Always use Q96-98 for client deliverables**

❌ **Don't resize with BILINEAR/BICUBIC**  
✅ **Always use LANCZOS resampling**

❌ **Don't enable AI enhancement by default**  
✅ **Skip AI unless specifically needed (safer)**

---

## 📁 File Locations

### Input Images
```
input_images/
├── Giga-V2_750Picacho_Kitchen_compatible_kitchen-bright.tiff
├── 750Picacho_Pool_compatible.tiff
└── 750Picacho_GreatRoom_Reset_compatible.tiff
```

### Outputs (Fixed Pipeline)
```
output_premium_fixed/  or  output_750picacho_final/
├── [basename]_PREMIUM_MASTER.tiff      (~400 MB, 16K)
├── [basename]_PRINT_8K_FIXED.jpg       (~13 MB, 8K)
├── [basename]_WEB_4K_FIXED.jpg         (~4 MB, 4K)
├── [basename]_MAGAZINE_2K_FIXED.jpg    (~1 MB, 2K)
└── [basename]_SOCIAL_FIXED.jpg         (~250 KB)
```

---

## 📖 Documentation

- **SESSION_SUMMARY.md** - What we accomplished today
- **QUALITY_FIX_SUMMARY.md** - Technical analysis
- **PIPELINE_FIX_COMPLETE.md** - Complete documentation
- **NEXT_STEPS.md** - Action plan

---

## 🆘 Troubleshooting

### "Real-ESRGAN unavailable"
✅ Normal - will use high-quality Lanczos instead  
💡 Optional: Install with `pip install realesrgan`

### Large file warnings
✅ Expected - we process 16K images (144 megapixels)  
💡 Warnings are safe to ignore

### Out of memory
✅ Try `--no-4k` to process at standard resolution  
💡 Or close other applications

### Color looks different
✅ Ensure viewing in color-managed application  
💡 ICC profiles are preserved in exports

---

## 💡 Pro Tips

1. **Always keep Master TIFF** - Future-proof for re-editing
2. **Test on target display** - Different screens show different colors
3. **Print proof before bulk printing** - Verify quality at scale
4. **Use 4K upscale** - It works perfectly (proven)
5. **Skip AI enhancement** - Safer, faster, artifact-free

---

## ⏱️ Expected Processing Times

| Configuration | Time per Image |
|---------------|----------------|
| **Standard (no 4K)** | ~30 seconds |
| **Premium (4K upscale)** | ~2-3 minutes |
| **With AI (not recommended)** | ~5-8 minutes |

*Times on Apple M4 Max. May vary by hardware.*

---

**Last Updated:** November 7, 2025  
**Version:** 1.0 (Production Ready)  
**Status:** ✅ All systems operational
