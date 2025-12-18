# Production Validation Run Guide

**CRITICAL FIXES IMPLEMENTED - READY FOR COMPREHENSIVE VALIDATION**

---

## 🚀 Quick Start

Run comprehensive validation on full dataset (6 images):

```bash
python production_depth_validation.py \
  --input-dir input_images/750_Picacho/Source_TIFFs_Base \
  --output-dir outputs/production_validation_comprehensive_20251218 \
  --tile-size 1024 \
  --overlap 192
```

**Estimated runtime:** 30-45 minutes (6 large TIFF images)

---

## 📊 Expected Output

### Terminal Output (PRIORITY 1 FIX - Separated Reporting)

```
VALIDATION COMPLETE
======================================================================
Total: 6
Execution: 6/6 succeeded, 0/6 failed
Seam validation: 5/6 passed
Quality (lenient): 4/6 passed
Quality (strict): 2/6 passed ⚠️ KEY METRIC

--- Per-Category Results ---
INTERIOR: 2/4 strict pass, avg_edge_f1=0.650, avg_seam_ratio=1.12
EXTERIOR: 1/2 strict pass, avg_edge_f1=0.705, avg_seam_ratio=1.14
```

### Output Files (Per Image)

```
outputs/production_validation_comprehensive_20251218/
├── 750Picacho_Aerial_Ultimate_depth.tiff         # 16-bit depth map
├── 750Picacho_Aerial_Ultimate_edges.png          # PRIORITY 4: Readable overlay
├── 750Picacho_Aerial_Ultimate_overshoot.png      # PRIORITY 3: Overshoot heatmap
├── 750Picacho_Aerial_Ultimate_metrics.json       # PRIORITY 1: Separated outcomes
├── 750Picacho_GreatRoom_Ultimate_depth.tiff
├── 750Picacho_GreatRoom_Ultimate_edges.png
├── 750Picacho_GreatRoom_Ultimate_overshoot.png
├── 750Picacho_GreatRoom_Ultimate_metrics.json
├── ... (4 more images)
└── validation_report.json                        # Aggregate report with categories
```

---

## 🔍 What's New (All 5 Priorities Implemented)

### ✅ PRIORITY 1: Reporting Integrity

**Before:**
```
Succeeded: 2/2  ← Conflated execution with quality
```

**After:**
```
Execution: 2/2 succeeded
Seam validation: 2/2 passed
Quality (lenient): 1/2 passed
Quality (strict): 0/2 passed ⚠️ KEY METRIC  ← True quality metric
```

### ✅ PRIORITY 2: Seam Stabilization

**Spatial calibration smoothing** (sigma=1.0) prevents grid artifacts.

**Expected improvement:**
- Aerial: seam_ratio 1.17 → <1.15

**How to verify:**
- Check `seam_validation.boundary_ratio` in metrics JSON
- Lower is better (threshold: <1.2 to pass)

### ✅ PRIORITY 3: Overshoot Diagnosis

**Overshoot heatmap** (`*_overshoot.png`) visualizes hallucinated edges in RED.

**Metrics JSON now includes:**
```json
{
  "overshoot_ratio": 0.0234,
  "overshoot_pixel_count": 45123,
  "halo_score": 0.823,
  "overshoot_penalty": 0.234,
  "depth_edge_threshold": 0.0234,
  "rgb_smooth_threshold": 12.34,
  "mean_depth_gradient_at_overshoot": 0.0456,
  "mean_rgb_detail_at_overshoot": 3.21
}
```

**How to use:**
- Open `*_overshoot.png` to see red overshoot regions
- Compare with edge overlay to diagnose hallucinations
- Check `overshoot_ratio` in metrics (lower is better)

### ✅ PRIORITY 4: Readable Edge Overlay

**Old overlay:** Green wash (unusable)

**New overlay:**
- **RED:** RGB edges depth is missing
- **BLUE:** Depth edges that are hallucinated
- **GREEN:** Correctly aligned edges
- **Legend:** Color key + alignment percentage

**How to use:**
- Open `*_edges.png`
- RED areas → depth needs more detail
- BLUE areas → depth is hallucinating edges
- GREEN areas → correct alignment

### ✅ PRIORITY 5: Full Dataset + Categories

**Before:** 2 images (Aerial, GreatRoom)

**After:** 6 images with category reporting
- **INTERIOR:** GreatRoom, Kitchen, PrimaryBedroom, PrimaryBathroom
- **EXTERIOR:** Aerial, Pool

**Category metrics:**
```
INTERIOR: X/4 strict pass, avg_edge_f1=0.XXX, avg_seam_ratio=X.XX
EXTERIOR: X/2 strict pass, avg_edge_f1=0.XXX, avg_seam_ratio=X.XX
```

---

## 📋 Validation Checklist

### Before Running

- [ ] Confirm input directory exists: `input_images/750_Picacho/Source_TIFFs_Base`
- [ ] Create output directory will auto-create
- [ ] Ensure 16GB+ RAM available (for 4K images)
- [ ] Ensure GPU/MPS available (check with `torch.cuda.is_available()` or `torch.backends.mps.is_available()`)

### During Execution

- [ ] Monitor memory usage (should stay <16GB)
- [ ] Check progress (6 images, ~5-7 min each)
- [ ] Watch for seam validation warnings
- [ ] Look for calibration smoothing logs: `"✓ Smoothed N tile calibrations"`

### After Completion

- [ ] Check execution rate: Should be 6/6
- [ ] Review seam pass rate: Target >80% (5/6)
- [ ] Review strict quality pass rate: Expect 30-50% (2-3/6)
- [ ] Inspect overshoot heatmaps for GreatRoom
- [ ] Verify Aerial seam_ratio <1.15 (spatial smoothing working)
- [ ] Review category breakdown (interior vs exterior)

---

## 🎯 Success Criteria

### Execution (Must Pass)
✅ All 6 images process without exceptions  
✅ All artifacts generated (depth, edges, heatmap, metrics)

### Seam Validation (Target)
🎯 >80% pass rate (5/6 images)  
🎯 Aerial seam_ratio <1.15 (spatial smoothing working)

### Quality Gates (Expected)
🎯 Lenient: 50-70% pass rate (3-4/6)  
🎯 Strict: 30-50% pass rate (2-3/6 initially)

### Reporting Integrity (Must Pass)
✅ Terminal shows separated execution/seam/quality rates  
✅ Category breakdown visible  
✅ Overshoot components logged

---

## 🔧 Troubleshooting

### Out of Memory

**Symptom:** Crash during depth estimation

**Fix:**
- Reduce tile size: `--tile-size 768`
- Or process images one at a time with `--force`

### Slow Performance

**Symptom:** >10 min per image

**Fix:**
- Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`
- If CPU-only, expect 2-3x slower

### Calibration Smoothing Errors

**Symptom:** `"scipy not available, skipping calibration smoothing"`

**Fix:**
- Install scipy: `pip install scipy`
- Or continue without smoothing (non-critical)

### High Seam Ratios

**Symptom:** seam_ratio >1.2 after spatial smoothing

**Next steps:**
- Increase overlap: `--overlap 256`
- Or review heatmap to identify problematic tiles

---

## 📊 Interpreting Results

### Metrics JSON Structure (PRIORITY 1)

```json
{
  "success": true,               ← Execution succeeded
  "seam_validation": {
    "passed": true,              ← Seam check passed
    "boundary_ratio": 1.14       ← <1.2 is good
  },
  "quality_passed_lenient": true,   ← Lenient gate
  "quality_passed_strict": false,   ← Strict gate (KEY METRIC)
  "quality_score": 0.567,
  "metrics": {
    "edge_f1": 0.692,            ← Primary alignment metric
    "edge_overlap": 0.734,
    "chamfer_distance": 1.6,     ← Lower is better
    "edge_count_ratio": 1.17,    ← Should be <2.5
    "halo_score": 0.823,         ← Higher is better
    "overshoot_penalty": 0.234   ← Lower is better
  }
}
```

### Category Report Structure (PRIORITY 5)

```json
{
  "category_stats": {
    "interior": {
      "total": 4,
      "seam_passed": 3,
      "quality_passed_strict": 2,
      "avg_edge_f1": 0.650,
      "avg_seam_ratio": 1.12
    },
    "exterior": {
      "total": 2,
      "seam_passed": 2,
      "quality_passed_strict": 1,
      "avg_edge_f1": 0.705,
      "avg_seam_ratio": 1.14
    }
  }
}
```

---

## 🚦 Next Steps After Validation

### If All Criteria Met (Production Ready)

1. Archive validation run:
   ```bash
   tar -czf validation_20251218.tar.gz outputs/production_validation_comprehensive_20251218/
   ```

2. Deploy to production pipeline

3. Monitor first production batch

### If Improvements Needed

1. **High seam ratios:** Increase overlap or tune smoothing sigma
2. **Low strict pass rate:** Review overshoot heatmaps, adjust edge snapping
3. **Category-specific issues:** Tune parameters per category

### If Critical Failures

1. Review failed image tracebacks in `production_validation.log`
2. Check memory usage during failure
3. Isolate failing image for debugging

---

## 📝 Log Files

**Main log:** `production_validation.log`

**What to check:**
- `[MEMORY]` lines: Track memory usage
- `"✓ Smoothed N tile calibrations"`: Spatial smoothing working
- `"Overshoot analysis:"`: Overshoot detection working
- `"Edge overlay saved (PRIORITY 4 readable format)"`: New overlay format
- Category breakdown at end

---

## ✅ Command Summary

```bash
# Full validation (recommended)
python production_depth_validation.py \
  --input-dir input_images/750_Picacho/Source_TIFFs_Base \
  --output-dir outputs/production_validation_comprehensive_20251218 \
  --tile-size 1024 \
  --overlap 192

# Force reprocess (skip resumability)
python production_depth_validation.py \
  --input-dir input_images/750_Picacho/Source_TIFFs_Base \
  --output-dir outputs/production_validation_comprehensive_20251218 \
  --tile-size 1024 \
  --overlap 192 \
  --force

# Smaller tiles (if OOM)
python production_depth_validation.py \
  --input-dir input_images/750_Picacho/Source_TIFFs_Base \
  --output-dir outputs/production_validation_comprehensive_20251218 \
  --tile-size 768 \
  --overlap 192
```

---

**Status:** READY FOR EXECUTION  
**Blocking Issues:** NONE  
**Estimated Runtime:** 30-45 minutes  
**Expected Success Rate:** 6/6 execution, 2-3/6 strict quality

🎯 **GO/NO-GO:** ✅ GO - All fixes implemented and tested
