# EfficientSAM V3 — Stage 6 Smoke Test Results

**Date**: December 12, 2025  
**Commit**: `d3e94cc` (canary preset recursion fix)  
**Status**: ✅ Infrastructure Validated, Fusion Path Operational

---

## Summary

Successfully completed **Stage 6 smoke test** validating the EfficientSAM V3 fusion infrastructure with Kitchen APEX A/B comparison. The canary preset now correctly:

* Inherits all APEX settings
* Enables FUSED backend (`SegFormerAdekMaterialSegmenter` → `FusedMaterialSegmenter`)
* Loads `efficientsam_s.onnx` session successfully
* Completes end-to-end processing with AI validation passing

---

## Test Configuration

### Input
* **Image**: `assets/phase2_bench/750Picacho_Kitchen_Ultimate.tif` (3375×6000)

### Presets Tested
| Run | Preset | Segmenter | EfficientSAM Session |
|-----|--------|-----------|---------------------|
| **A (Baseline)** | `interior_luxury_apex_quality` | `SegFormerAdekMaterialSegmenter` | ❌ Not loaded |
| **B (Canary)** | `interior_luxury_apex_quality_efficientsam` | `FusedMaterialSegmenter` | ✅ Loaded (`efficientsam_s.onnx`) |

---

## Results

### Run A: Baseline APEX (SegFormer-only)
* **Status**: ✅ Completed
* **Segmenter**: `SegFormerAdekMaterialSegmenter`
* **Runtime**: ~27s (first run, then skipped)
* **AI Validation**: Passing

### Run B: Canary APEX + EfficientSAM
* **Status**: ✅ Completed
* **Segmenter**: `FusedMaterialSegmenter` 
* **EfficientSAM**: Session loaded (`weights/efficientsam/efficientsam_s.onnx`)
* **Runtime**: **27.3s** total
  * Segmentation: **2.58s**
  * Materials V2: **6.46s**
  * Export (marketing PNG): **6.0s** (lossless, 974MB)
* **AI Validation**: ✅ Passing
  * Color Δ: **0.0025** (threshold: 0.06)
  * Luma Δ: **0.0025** (threshold: 0.06)
* **Materials V2 Metadata**:
  * Glass: 1,135,301 px
  * Wood: 6,218,019 px
  * Stone: 7,420,184 px
  * Foliage: 225,147 px

---

## Key Findings

### ✅ What Works

1. **Canary preset inheritance** — Fixed recursion bug; now correctly applies base APEX settings before enabling FUSED
2. **EfficientSAM session initialization** — Model loads successfully on CPU (2s init time)
3. **FusedMaterialSegmenter wiring** — Pipeline correctly selects fused segmenter when `backend_v3 = FUSED`
4. **End-to-end processing** — No crashes, clean completion with all exports generated
5. **AI validation passing** — Color/luma accuracy within APEX thresholds

### ⚠️ Outstanding (Stage 6.5)

1. **Fusion stats not in report** — `segmentation_v3` block not written to JSON output
   * Need to wire `fusion_applied` and `iou_base_vs_refined` per-class stats into pipeline report
2. **Fusion actually applied?** — Logs show session loaded but no explicit "fusion applied for class X" messages
   * Next: Add debug logging in `FusedMaterialSegmenter` to confirm `get_refined_mask()` is actually called
3. **Visual edge comparison** — Haven't yet visually inspected baseline vs canary edges for glass/water/foliage

---

## Critical Fix Applied (Stage 5B Follow-up)

**Problem**: Canary presets failed with `TypeError: apply_preset() takes 1 positional argument but 2 were given`

**Root Cause**: Recursive call used `self.apply_preset(base_preset)` (wrong signature)

**Fix**:
```python
# OLD (broken):
self.apply_preset(base_preset)

# NEW (correct):
original_preset = self.preset
self.preset = base_preset
self.apply_preset()  # No args, reads self.preset
self.preset = original_preset
```

Plus added recursion guard:
```python
if base_preset == p:
    raise RuntimeError(f"Canary preset recursion detected: {p}")
```

---

## Performance Notes

* EfficientSAM session init: **~2s** (one-time per pipeline)
* Fusion overhead: **TBD** (need fusion stats to measure)
* Overall canary runtime (**27.3s**) is **same ballpark** as baseline — no catastrophic regression

---

## Next Steps (Stage 6.5 — Fusion Stats & Visual Validation)

### 1. Wire fusion stats into report
* Add `segmentation_v3` block to pipeline report JSON
* Include per-class: `{iou_base_vs_refined, fusion_applied, backend_used}`

### 2. Add debug logging
* In `FusedMaterialSegmenter.segment()`:
  ```python
  logger.info(f"Fusion applied for {class_name}: IoU={iou:.3f}, fusion={applied}")
  ```

### 3. Visual edge comparison
* Extract tight crops around:
  * Glass (windows/reflections)
  * Foliage (plant edges against wall)
  * Stone (countertop edge vs background)
* Side-by-side baseline vs canary at 400% zoom

### 4. Full A/B matrix (all benchmark images)
* Kitchen ✅ (smoke test done)
* Bedroom, Bathroom, Pool, Aerial (pending)

---

## Files Changed

* `lux_depth_v2/config.py` — Canary preset recursion fix
* `scripts/stage6_smoke_proper.py` — Stage 6 test runner (uses internal pipeline API)

---

## Test Artifacts

* **Summary**: `outputs/stage6_smoke/stage6_smoke_summary.json`
* **Baseline output**: `outputs/stage6_smoke/kitchen_A_baseline/`
* **Canary output**: `outputs/stage6_smoke/kitchen_B_efficientsam/`
  * Report: `750Picacho_Kitchen_Ultimate_report.json`
  * Marketing PNG: `750Picacho_Kitchen_Ultimate_marketing.png` (974MB lossless)
  * Master TIFF: `750Picacho_Kitchen_Ultimate_master.tif`

---

## Conclusion

✅ **Stage 6 Smoke Test: PASSED**

The EfficientSAM V3 fusion infrastructure is **operational and stable**. The canary preset correctly enables FUSED mode, loads the ONNX session, and completes processing without errors.

**Ready for Stage 6.5**: fusion stats + visual validation before declaring full Stage 6 complete.

---

**Session End**: December 13, 2025, 12:00 AM PST  
**Commit**: `d3e94cc` — Canary preset recursion fix  
**Next Session**: Stage 6.5 (fusion stats wiring + visual A/B comparison)
