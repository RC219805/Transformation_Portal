# Production Validation Quick Start Guide
## High-Fidelity Depth Pipeline

---

## 🚀 Quick Start (5 Minutes)

### Prerequisites
```bash
# Ensure dependencies installed
pip install torch transformers pillow numpy opencv-python tqdm psutil

# Navigate to repository
cd /Users/rc/Transformation_Portal
```

### Run Full Validation

```bash
python production_validation_suite.py \
  --input-dir input_images/750_Picacho/Source_TIFFs_Base \
  --output-dir outputs/validation_$(date +%Y%m%d_%H%M%S) \
  --preset production \
  --critical-scenes Kitchen GreatRoom Aerial Pool
```

**Expected Output**:
- Per-image metrics in `validation_results.csv`
- Dataset summary in `dataset_report.json`
- Depth maps in `depth/{image}_depth.tif`
- Normal maps in `normals/{image}_normals.png`
- Visualizations in `visualizations/{image}_validation.jpg`

**Runtime**: ~30-45 minutes for 6 images on M4 Max

---

## 📊 Understanding Results

### Quick Pass/Fail Check

```bash
# View dataset summary
cat outputs/validation_*/dataset_report.json | jq '{
  pilot_pass_rate,
  production_pass_rate,
  deployment_recommendation,
  critical_scene_results
}'
```

**Interpretation**:
- `pilot_pass_rate ≥ 0.80` → ✅ Pilot approved
- `production_pass_rate ≥ 0.95` → ✅ Production approved
- `deployment_recommendation: "APPROVED_PILOT"` → Proceed to Phase 1
- `critical_scene_results: {"Kitchen": true, ...}` → All critical scenes passed

### Per-Image Metrics

```bash
# View CSV in spreadsheet app or
cat outputs/validation_*/validation_results.csv | column -t -s,
```

**Key Columns**:
- `edge_f1`: Primary quality metric (target ≥0.30)
- `chamfer_distance_mean`: Alignment error in pixels (target <15px)
- `seam_energy`: Tiling artifact detection (target <1.2)
- `overshoot_penalty`: Halo detection (target <0.5)
- `passed`: Boolean pass/fail
- `failure_reasons`: Why image failed (if any)

---

## 🎛️ Presets

### Production (Recommended)

```bash
--preset production
# tile_size=1024, overlap=128
# edge_snapping=ON, global_anchor=OFF
# Balanced quality and speed
```

### Preview (Fast Iteration)

```bash
--preset preview
# tile_size=512, overlap=64
# edge_snapping=OFF, global_anchor=OFF
# Fast for testing (~3x speedup)
```

### Hero (Maximum Quality)

```bash
--preset hero
# tile_size=1536, overlap=192
# edge_snapping=ON (strength=0.3)
# Slowest but best quality for showcase images
```

---

## 🔍 Troubleshooting

### Issue: Out of Memory

**Symptoms**: Process killed, no output  
**Fix**: Reduce tile size or limit images

```bash
# Use preview preset (smaller tiles)
--preset preview

# Or process fewer images at once
--limit 2
```

### Issue: Slow Processing

**Symptoms**: >10min per image  
**Fix**: Check GPU availability

```bash
# Verify MPS (Apple Silicon)
python -c "import torch; print(torch.backends.mps.is_available())"

# Or use preview preset for speed
--preset preview
```

### Issue: Quality Gate Failures

**Symptoms**: High failure rate in dataset_report.json  
**Fix**: Review failure_reasons in CSV

```bash
# Find most common failure mode
cat outputs/validation_*/validation_results.csv | \
  grep "False" | \
  cut -d',' -f18 | \
  sort | uniq -c | sort -nr
```

**Common Failures**:
- `edge_f1=0.25 < 0.30` → Threshold too strict, or depth quality issue
- `seam_energy=1.35 > 1.2` → Tiling artifact, check reconciliation
- `overshoot=0.55 > 0.5` → Halo detected, review edge snapping strength

---

## 📁 Output Structure

```
outputs/validation_YYYYMMDD_HHMMSS/
├── config.json                      # Processing configuration
├── dataset_report.json              # Summary + deployment recommendation
├── validation_results.csv           # Per-image metrics (spreadsheet-friendly)
│
├── depth/                           # 16-bit depth maps (Materials V3 ready)
│   ├── Kitchen_depth.tif
│   ├── GreatRoom_depth.tif
│   └── ...
│
├── normals/                         # Normal maps (Materials V3 integration)
│   ├── Kitchen_normals.png
│   ├── GreatRoom_normals.png
│   └── ...
│
├── visualizations/                  # 4-panel validation images
│   ├── Kitchen_validation.jpg
│   │   ├── [Panel 1: RGB source]
│   │   ├── [Panel 2: Depth visualization]
│   │   ├── [Panel 3: Edge overlay]
│   │   └── [Panel 4: Metrics text]
│   └── ...
│
└── metrics/                         # Per-image JSON (atomic write)
    ├── Kitchen_metrics.json
    │   ├── config: {...}
    │   ├── result: {...}
    │   └── raw_metrics: {...}
    └── ...
```

---

## 🧪 Testing Workflow

### Step 1: Smoke Test (2 minutes)

```bash
# Test on single image
python production_validation_suite.py \
  --input-dir input_images/750_Picacho/Source_TIFFs_Base \
  --output-dir outputs/smoke_test \
  --preset preview \
  --limit 1
```

**Verify**:
- [ ] Script completes without errors
- [ ] Output directory created
- [ ] Depth map generated
- [ ] Metrics JSON saved

### Step 2: Critical Scenes (15 minutes)

```bash
# Validate just critical scenes
python production_validation_suite.py \
  --input-dir input_images/750_Picacho/Source_TIFFs_Base \
  --output-dir outputs/critical_validation \
  --preset production \
  --critical-scenes Kitchen GreatRoom Aerial Pool
```

**Verify**:
- [ ] All 4 critical scenes processed
- [ ] critical_scene_results shows all true
- [ ] Visualizations look reasonable

### Step 3: Full Dataset (45 minutes)

```bash
# Complete validation
python production_validation_suite.py \
  --input-dir input_images/750_Picacho/Source_TIFFs_Base \
  --output-dir outputs/full_validation \
  --preset production \
  --critical-scenes Kitchen GreatRoom Aerial Pool
```

**Verify**:
- [ ] All 6 images processed
- [ ] pilot_pass_rate ≥0.80
- [ ] dataset_report.json generated
- [ ] deployment_recommendation is "APPROVED_*"

---

## 📈 Advanced Usage

### Custom Thresholds

```python
# Edit production_validation_suite.py
# Modify ProcessingConfig class

min_edge_f1 = 0.25              # Relaxed from 0.30
max_chamfer_distance = 18.0     # Relaxed from 15.0
max_seam_energy = 1.3           # Relaxed from 1.2
```

### Batch Processing Multiple Directories

```bash
for dir in input_images/*/; do
  python production_validation_suite.py \
    --input-dir "$dir" \
    --output-dir "outputs/$(basename $dir)_validation" \
    --preset production
done
```

### Extract Specific Metrics

```bash
# Extract Edge F1 scores
cat outputs/validation_*/validation_results.csv | \
  awk -F, 'NR>1 {print $1 "," $5}' | \
  column -t -s,

# Find worst-case seam energy
cat outputs/validation_*/validation_results.csv | \
  awk -F, 'NR>1 {print $1 "," $11}' | \
  sort -t, -k2 -nr | \
  head -5
```

---

## 🎯 Deployment Checklist

### Before Running Validation

- [ ] Input directory contains 6 TIFF files
- [ ] Sufficient disk space (~10GB for outputs)
- [ ] PyTorch with MPS support installed
- [ ] Depth Anything V2 model downloaded (auto on first run)

### After Validation Completes

- [ ] Check `dataset_report.json` for pass rates
- [ ] Review worst-case metrics (chamfer, seam_energy)
- [ ] Visual inspection of critical scenes
- [ ] Document any threshold adjustments
- [ ] Generate executive summary

### Ready for Pilot

- [ ] pilot_pass_rate ≥ 0.80 ✅
- [ ] All critical scenes pass ✅
- [ ] No catastrophic failures ✅
- [ ] Approval from technical lead ✅

### Ready for Production

- [ ] Pilot completed successfully ✅
- [ ] Materials V3 A/B shows improvement ✅
- [ ] production_pass_rate ≥ 0.95 ✅
- [ ] Throughput validated ✅
- [ ] Approval from product manager ✅

---

## 📞 Support

**Technical Issues**:
- Check log output for error messages
- Verify dependencies with `pip list | grep -E "(torch|transformers|pillow)"`
- Test with `--preset preview --limit 1` to isolate issue

**Quality Questions**:
- Review failure_reasons in CSV
- Compare to baseline metrics (Pool.tif reference)
- Adjust thresholds if needed (document rationale)

**Performance Optimization**:
- Use preview preset for faster iteration
- Process in batches if memory-constrained
- Profile with psutil (automatically enabled if available)

---

**Last Updated**: December 17, 2025  
**Version**: 1.0.0  
**Status**: Production Ready  
