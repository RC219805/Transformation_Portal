# Stage 6 Golden Baseline A/B - Quick Reference

## Prerequisites

### 1. Verify EfficientSAM Model
```bash
ls -lh weights/efficientsam/efficientsam_s.onnx
# Should show ~101 MB file
```

If missing:
```bash
python -m lux_depth_v2.cli --download-efficientsam --efficientsam-model efficientsam_s
```

### 2. Verify Benchmark Images
```bash
ls -1 assets/phase2_bench/
# Should contain:
#   interior_kitchen_750.tiff
#   exterior_pool_750.tiff
```

---

## Running Stage 6 A/B

### Quick Test (Single Image)
```bash
# Edit scripts/stage6_ab_golden_baseline_v2.py
# Comment out all benchmarks except interior_kitchen_750

python scripts/stage6_ab_golden_baseline_v2.py
```

### Full A/B Matrix
```bash
python scripts/stage6_ab_golden_baseline_v2.py
```

**Expected outputs**:
```
outputs/stage6_ab/
  interior_kitchen_750_A_baseline/
    interior_kitchen_750_master.tif
    interior_kitchen_750_report.json  # ← Check this
  interior_kitchen_750_B_efficientsam/
    interior_kitchen_750_master.tif
    interior_kitchen_750_report.json  # ← Check for segmentation_v3
  exterior_pool_750_A_baseline/
    ...
  exterior_pool_750_B_efficientsam/
    ...
  stage6_ab_summary.json
```

---

## Analyzing Results

### 1. Check Console Output

Look for:
```
📊 EfficientSAM V3 Stats:
   Backend: SegmentationBackend.FUSED
   Fusion Mode: FusionMode.CONFIDENCE_WEIGHTED
   Model: efficientsam_s
   glass: IoU=0.650, Applied=1.0
   water: IoU=0.720, Applied=1.0
   foliage: IoU=0.180, Applied=0.0
```

**Good signs**:
- At least one class shows `Applied=1.0`
- IoU values > 0.30 for applied classes

**Bad signs**:
- All classes show `Applied=0.0`
- IoU values consistently < 0.20
- Missing `segmentation_v3` block

### 2. Inspect Report JSONs

```bash
# Baseline (should NOT have segmentation_v3)
jq '.segmentation_v3' outputs/stage6_ab/interior_kitchen_750_A_baseline/interior_kitchen_750_report.json

# Canary (MUST have segmentation_v3)
jq '.segmentation_v3' outputs/stage6_ab/interior_kitchen_750_B_efficientsam/interior_kitchen_750_report.json
```

Expected canary output:
```json
{
  "backend_v3": "SegmentationBackend.FUSED",
  "fusion_mode": "FusionMode.CONFIDENCE_WEIGHTED",
  "model": "efficientsam_s",
  "refined_classes": ["foliage", "glass", "water"],
  "per_class": {
    "glass": {
      "iou_base_vs_refined": 0.65,
      "fusion_applied": 1.0
    },
    ...
  }
}
```

### 3. Compare Runtimes

```bash
jq '.timing_s' outputs/stage6_ab/*/interior_kitchen_750_report.json
```

**Acceptable delta**: Canary < +40% slower than baseline for APEX

### 4. Visual Inspection

Compare edge quality:
- Glass windows (sharp edges, no halos)
- Pool water (smooth surface, no spill into adjacent materials)
- Foliage (clean silhouettes, no neon green)

---

## Decision: Promote FUSED to Default APEX?

### ✅ Promote if ALL true:

1. **Fusion applies**: At least one class shows `fusion_applied=1.0` on most scenes
2. **IoU meaningful**: Average IoU > 0.30 for applied classes
3. **No visual regressions**: Edge crops look better or same (no halos/spill)
4. **Runtime acceptable**: APEX delta < +40%

### ⚠️ Keep Canary-Only if ANY true:

- Fusion rarely applies (most classes `Applied=0.0`)
- IoU consistently low (< 0.30)
- Visual artifacts (halos, banding, color spill)
- Runtime delta unacceptable for hero frames

### 🔧 Tune & Re-run if:

- IoU gating too strict (increase `fusion_min_iou`)
- Fusion weights wrong (adjust `alpha_edge` / `alpha_core`)
- Prompt generation failing (check logs for provider errors)

---

## Troubleshooting

### "Missing segmentation_v3 in canary report"

**Cause**: Canary preset not configured correctly or fusion provider unavailable

**Fix**:
```bash
# Verify preset config
python -c "
from lux_depth_v2.config import PipelineConfig, Preset
cfg = PipelineConfig()
cfg.preset = Preset.INTERIOR_LUXURY_APEX_QUALITY_EFFICIENTSAM
cfg.apply_preset()
print('backend_v3:', cfg.segmentation.backend_v3)
print('fusion_mode:', cfg.segmentation.fusion_mode)
print('model:', cfg.segmentation.efficientSAM_model)
"
```

Should print:
```
backend_v3: SegmentationBackend.FUSED
fusion_mode: FusionMode.CONFIDENCE_WEIGHTED
model: efficientsam_s
```

### "All classes show fusion_applied=0.0"

**Cause**: IoU gate rejecting all masks (SegFormer and EfficientSAM disagree)

**Options**:
1. Lower `fusion_min_iou` threshold (currently 0.30)
2. Check if EfficientSAM model loaded correctly
3. Verify prompts are being generated from SegFormer masks

**Debug**:
```bash
# Run with debug logging
LOGLEVEL=DEBUG python scripts/stage6_ab_golden_baseline_v2.py 2>&1 | grep "V3 refine"
```

Look for lines like:
```
V3 refine glass: refined=True iou=0.180 applied=False
```

If `iou` consistently low → prompts or model issue

### "EfficientSAM model not found"

**Fix**:
```bash
python -m lux_depth_v2.cli --download-efficientsam \
  --efficientsam-model efficientsam_s \
  --efficientsam-url "https://huggingface.co/yunyangx/EfficientSAM/resolve/main/efficientsam_s.onnx" \
  --efficientsam-sha256 "b257787eeecdfd0db0626f83a8241874c35c74eb4c25c4d12ff0a478f90f30f9"
```

---

## Next Steps After A/B

### If Promoting FUSED to Default

1. Update presets in `lux_depth_v2/config.py`:
   ```python
   # Make FUSED the default for APEX
   self.segmentation.backend_v3 = SegmentationBackend.FUSED
   ```

2. Document in `docs/QUALITY_TIERS.md`:
   - Note that APEX uses EfficientSAM fusion by default
   - Document typical IoU ranges and fusion success rates

3. Update `README.md` and user guides

4. Merge to `main` with detailed commit message

### If Keeping Canary-Only

1. Document why in `docs/sessions/efficientsam-v3/STAGE6_AB_RESULTS.md`
2. List tuning actions needed
3. Keep canary presets available for experimental use
4. Plan next iteration with adjusted params

---

## Files to Archive

After A/B completion:

```
outputs/stage6_ab/                              # Keep for reference
docs/sessions/efficientsam-v3/
  2025-12-13_STAGE6_AB_RESULTS.md               # Create this
  visual_diffs/                                 # Screenshots
    kitchen_glass_edges_comparison.png
    pool_water_surface_comparison.png
```

---

**Last Updated**: 2025-12-13  
**Stage**: 6.5 Complete → Ready for 6 A/B Execution
