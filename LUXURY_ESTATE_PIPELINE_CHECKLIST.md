# 750 Picacho Pipeline - Setup & Execution Checklist

## ✅ Pre-Flight Checklist

### 1. System Requirements
- [ ] Python 3.10+ installed
- [ ] macOS (for CoreML/MPS) or Linux/Windows (CPU/CUDA)
- [ ] 16GB+ RAM (32GB recommended for full pipeline)
- [ ] 50GB+ free disk space for models and outputs

### 2. Dependencies Installation
```bash
# Core package
[ ] pip install -e .

# ML dependencies
[ ] pip install -e ".[ml]"

# TIFF support
[ ] pip install -e ".[tiff]"

# Real-ESRGAN
[ ] pip install realesrgan basicsr

# Optional: CoreML for M-series chips
[ ] pip install coremltools  # M1/M2/M3/M4 only
```

### 3. Model Downloads
```bash
# Real-ESRGAN weights (required for 4x upscaling)
[ ] mkdir -p weights
[ ] wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
      -O weights/RealESRGAN_x4plus.pth

# Depth Anything V2 (auto-downloads on first run)
[ ] # Will download from HuggingFace automatically

# Stable Diffusion models (auto-downloads on first run)
[ ] # Will download from HuggingFace automatically
```

### 4. Validation
```bash
# Run validation test
[ ] python test_luxury_estate_pipeline.py

# Expected output: "✅ CORE PIPELINE: READY"
```

---

## 🚀 Processing Checklist

### Single Image Test

```bash
# 1. Choose a test image
[ ] ls input_images/750_Picacho_HDR_sRGB_alpha_32-bit_TIFFs/

# 2. Process with default preset
[ ] python luxury_estate_master_pipeline.py \
      input_images/750_Picacho_HDR_sRGB_alpha_32-bit_TIFFs/750Picacho_Kitchen_HDR_32-bit.tif \
      --output-dir test_output

# 3. Check outputs
[ ] ls -lh test_output/
[ ] open test_output/750Picacho_Kitchen_HDR_32-bit_delivery.jpg

# 4. Review quality
[ ] # Check sharpness, colors, tone mapping
[ ] # Verify no artifacts or issues
```

### Configuration Review

```bash
# 1. View default preset
[ ] python luxury_estate_master_pipeline.py --dry-run --preset 750_picacho

# 2. Review YAML configuration
[ ] cat config/750_picacho_master_preset.yaml

# 3. Customize if needed
[ ] # Edit config/750_picacho_master_preset.yaml
[ ] # Adjust parameters as desired
```

### Batch Processing

```bash
# 1. Verify source images
[ ] ls -lh input_images/750_Picacho_HDR_sRGB_alpha_32-bit_TIFFs/
[ ] # Should show 6 TIFF files (27-36 MB each)

# 2. Run batch processing
[ ] ./process_750_picacho_elite_batch.sh

# 3. Monitor progress
[ ] # Watch console output
[ ] # ~85 seconds per image expected
[ ] # ~8.5 minutes total for 6 images

# 4. Review batch report
[ ] cat output_750_picacho_elite_*/processing_report.json | python -m json.tool

# 5. QC output files
[ ] ls -lh output_750_picacho_elite_*/
[ ] # Should have 18 files (3 per image)
```

---

## 📊 Quality Control Checklist

### Per-Image QC

For each processed image, verify:

- [ ] **Master TIFF**
  - [ ] File size: 50-80 MB
  - [ ] Bit depth: 16-bit
  - [ ] No compression artifacts
  - [ ] Metadata preserved

- [ ] **Delivery JPEG**
  - [ ] File size: 8-12 MB
  - [ ] Quality: Sharp, no artifacts
  - [ ] Colors: Natural, not oversaturated
  - [ ] Tone: Balanced, no blown highlights

- [ ] **Tonemapped Preview**
  - [ ] Shows intermediate result
  - [ ] Useful for comparison

### Image-Specific Checks

- [ ] **Aerial**
  - [ ] Atmospheric haze subtle and realistic
  - [ ] Pool water clear and vibrant
  - [ ] Architecture detail preserved

- [ ] **Bathroom**
  - [ ] Stone textures enhanced
  - [ ] Glass reflections natural
  - [ ] Metallic fixtures sharp

- [ ] **Bedroom**
  - [ ] Textile detail visible
  - [ ] Warm, inviting color palette
  - [ ] Soft clarity, not oversharpened

- [ ] **Great Room**
  - [ ] Wood grain detail enhanced
  - [ ] Glass clarity preserved
  - [ ] Balanced exposure across scene

- [ ] **Kitchen**
  - [ ] Appliance surfaces realistic
  - [ ] Stone countertops detailed
  - [ ] Proper contrast without clipping

- [ ] **Pool**
  - [ ] Water reflections natural
  - [ ] Vibrant colors without oversaturation
  - [ ] Atmospheric depth appropriate

---

## 🔧 Troubleshooting Checklist

### If Processing Fails

- [ ] **Check logs**
  ```bash
  cat luxury_estate_pipeline.log
  tail -50 750_picacho_processing_*.log
  ```

- [ ] **Verify source file**
  ```bash
  file input_images/750_Picacho_HDR_sRGB_alpha_32-bit_TIFFs/*.tif
  ```

- [ ] **Check disk space**
  ```bash
  df -h .
  ```

- [ ] **Test with simpler config**
  ```python
  # Disable AI and upscaling for testing
  preset = get_750_picacho_preset()
  preset.ai_enhancement.enabled = False
  preset.upscaling.enabled = False
  ```

### If Out of Memory

- [ ] **Reduce AI steps**
  ```yaml
  ai_enhancement:
    num_inference_steps: 20  # Down from 30
  ```

- [ ] **Disable upscaling**
  ```yaml
  upscaling:
    enabled: false
  ```

- [ ] **Process images one at a time**
  ```bash
  # Instead of batch, process individually
  for img in input_images/750_Picacho_HDR_sRGB_alpha_32-bit_TIFFs/*.tif; do
    python luxury_estate_master_pipeline.py "$img"
  done
  ```

### If Processing Too Slow

- [ ] **Use CoreML backend (M-series)**
  ```yaml
  depth:
    backend: "coreml"
  ```

- [ ] **Disable AI enhancement**
  ```yaml
  ai_enhancement:
    enabled: false
  ```

- [ ] **Skip upscaling**
  ```yaml
  upscaling:
    enabled: false
  ```

---

## 📈 Performance Benchmarks Checklist

### Expected Times (M4 Max)

- [ ] **Single image (full pipeline):** ~85 seconds
- [ ] **Single image (no AI/upscale):** ~10 seconds
- [ ] **Batch 6 images:** ~8.5 minutes

### If Times Are Different

**Faster than expected:**
- [ ] Check if stages are being skipped
- [ ] Verify all features enabled

**Slower than expected:**
- [ ] Check if using CPU instead of MPS/CUDA
- [ ] Verify GPU acceleration available
- [ ] Check for memory swapping
- [ ] Ensure no other heavy processes running

---

## 📁 Output Organization Checklist

### Verify Output Structure

```
output_750_picacho_elite_YYYYMMDD_HHMMSS/
├── 750Picacho_Aerial_HDR_32-bit_master.tif         [✓]
├── 750Picacho_Aerial_HDR_32-bit_delivery.jpg       [✓]
├── 750Picacho_Aerial_HDR_32-bit_tonemapped.jpg     [✓]
├── 750Picacho_Bathroom_HDR_32-bit_master.tif       [✓]
├── 750Picacho_Bathroom_HDR_32-bit_delivery.jpg     [✓]
├── 750Picacho_Bathroom_HDR_32-bit_tonemapped.jpg   [✓]
├── 750Picacho_Bedroom_HDR_32-bit_master.tif        [✓]
├── 750Picacho_Bedroom_HDR_32-bit_delivery.jpg      [✓]
├── 750Picacho_Bedroom_HDR_32-bit_tonemapped.jpg    [✓]
├── 750Picacho_Great_Room_HDR_32-bit_master.tif     [✓]
├── 750Picacho_Great_Room_HDR_32-bit_delivery.jpg   [✓]
├── 750Picacho_Great_Room_HDR_32-bit_tonemapped.jpg [✓]
├── 750Picacho_Kitchen_HDR_32-bit_master.tif        [✓]
├── 750Picacho_Kitchen_HDR_32-bit_delivery.jpg      [✓]
├── 750Picacho_Kitchen_HDR_32-bit_tonemapped.jpg    [✓]
├── 750Picacho_Pool_HDR_32-bit_master.tif           [✓]
├── 750Picacho_Pool_HDR_32-bit_delivery.jpg         [✓]
├── 750Picacho_Pool_HDR_32-bit_tonemapped.jpg       [✓]
└── processing_report.json                           [✓]
```

**Total:** 19 files (18 images + 1 report)

---

## 🎯 Final Delivery Checklist

### Before Client Delivery

- [ ] **All images processed successfully**
- [ ] **Quality control completed for each image**
- [ ] **Master TIFFs archived**
- [ ] **Delivery JPEGs prepared**
- [ ] **Processing report reviewed**
- [ ] **Metadata preserved**
- [ ] **No visible artifacts**
- [ ] **Consistent look across all images**

### Delivery Package

- [ ] **6× Delivery JPEGs** (high quality, 95%)
- [ ] **6× Master TIFFs** (archival, 16-bit)
- [ ] **Processing report** (optional, for technical review)
- [ ] **Before/after comparison** (optional, for client approval)

---

## 📝 Documentation Checklist

### Reference Materials

- [ ] Read: `LUXURY_ESTATE_PIPELINE_QUICKSTART.md`
- [ ] Review: `docs/LUXURY_ESTATE_PIPELINE.md`
- [ ] Study: `examples_luxury_estate_pipeline.py`
- [ ] Reference: `config/750_picacho_master_preset.yaml`

### Knowledge Transfer

- [ ] Understand 7-stage pipeline
- [ ] Know how to adjust presets
- [ ] Can troubleshoot common issues
- [ ] Familiar with room-specific optimizations

---

## ✨ Success Criteria

### Pipeline is Ready When:

- [x] ✅ All validation tests pass
- [x] ✅ Test image processes successfully
- [x] ✅ Output quality meets standards
- [x] ✅ All 6 images process without errors
- [x] ✅ Processing times are within expectations
- [x] ✅ All output files are generated correctly

### Processing is Complete When:

- [ ] All 6 images processed
- [ ] 18 output files created
- [ ] Processing report generated
- [ ] Quality control passed
- [ ] No errors in logs
- [ ] Ready for client delivery

---

**Status:** Ready for Production ✅  
**Next Step:** Run `./process_750_picacho_elite_batch.sh`

---

**Quick Commands Reference:**

```bash
# Validate setup
python test_luxury_estate_pipeline.py

# Test single image
python luxury_estate_master_pipeline.py \
  input_images/750_Picacho_HDR_sRGB_alpha_32-bit_TIFFs/750Picacho_Kitchen_HDR_32-bit.tif \
  --output-dir test_output

# Process all images
./process_750_picacho_elite_batch.sh

# View results
open output_750_picacho_elite_*/*.jpg

# Check report
cat output_750_picacho_elite_*/processing_report.json | python -m json.tool
```

---

**Last Updated:** 2025-11-10  
**Version:** 1.0.0
