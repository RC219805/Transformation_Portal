# PR-3C Critical Fixes Applied

**Date**: 2025-12-13  
**Session**: Stage 6 A/B Test Preparation  
**Status**: ✅ Ready for execution

---

## Executive Summary

The original `stage6_ab_with_boundary_metrics.py` had **4 critical bugs** that would have caused failures or misleading promotion decisions. All have been corrected in `stage6_ab_with_boundary_metrics_FIXED.py`.

---

## Critical Bugs Fixed

### 1. ✅ Pipeline API signature mismatch (CRITICAL)

**Problem**:
```python
pipeline.process_one(input_path=input_path, output_dir=output_dir)
```

`LuxPipelineV2.process_one()` signature is:
```python
def process_one(self, img_path: Path, depth_path: Optional[Path] = None)
```

It does **not** accept `output_dir` as a parameter. The pipeline uses `cfg.output_dir` set during initialization.

**Fix**:
```python
cfg = PipelineConfig(output_dir=output_dir, preset=preset)
pipe = LuxPipelineV2(cfg)
result = pipe.process_one(input_path)
```

**Impact if unfixed**: Script would crash immediately on first pipeline call.

---

### 2. ✅ Mask extraction (BLOCKER)

**Problem**:  
Original script tried to load masks from disk:
```python
load_mask_from_output(output_dir, class_name)
```

But `pipeline.py` **does not write per-class mask PNGs** anywhere—it only writes final images and reports. The function would always return `None`, making all boundary metrics meaningless.

**Fix**:  
Extract masks **in-memory** directly from the segmenter:

```python
def run_segmentation_only(input_path, preset, target_classes):
    # Build config from preset
    cfg = PipelineConfig()
    cfg.apply_preset(preset)
    
    # Load image
    rgb01, _ = io_utils.read_rgb_any(input_path)
    rgb_t = torch_ops.to_torch_rgb(rgb01, device)
    
    # Create segmenter (respects backend_v3/fusion from preset)
    seg = create_material_segmenter(cfg.segmentation, device)
    
    # Run segmentation
    masks_dict_torch = seg.predict(rgb_t)  # dict[str, torch.Tensor] (1,1,H,W)
    
    # Extract to numpy
    masks = {}
    for cls in target_classes:
        if cls in masks_dict_torch:
            mask_np = masks_dict_torch[cls][0, 0].cpu().numpy()
            masks[cls] = mask_np
    
    return masks, rgb01, runtime_sec
```

**Impact if unfixed**: All boundary metrics would be computed on `None` or missing masks → invalid results.

---

### 3. ✅ A/B comparability (INVALID TEST)

**Problem**:  
Aerial scene used:
```python
baseline_preset = Preset.INTERIOR_LUXURY_MAX_QUALITY
canary_preset   = Preset.INTERIOR_LUXURY_APEX_QUALITY_EFFICIENTSAM
```

This compares **two different quality tiers** (Max vs APEX), not "same tier with/without EfficientSAM."

**Fix**:
```python
baseline_preset = Preset.INTERIOR_LUXURY_APEX_QUALITY
canary_preset   = Preset.INTERIOR_LUXURY_APEX_QUALITY_EFFICIENTSAM
```

**Impact if unfixed**: Aerial comparison would be invalid (tier change confounds EfficientSAM effect).

---

### 4. ✅ Promotion logic (WRONG METRIC)

**Problem**:  
Original logic treated BF1 (boundary F1 between canary and baseline) as an "improvement" signal:
```python
if BF1 >= 0.95:
    improvements.append(cls)
```

But BF1 computed as `pred=canary, ref=baseline` measures **similarity to baseline**, not "better edges." High BF1 means "nothing changed."

**Fix**:  
Use **edge alignment vs image gradients** as the improvement signal:

```python
# Baseline vs gradients
base_edge_align = compute_full_boundary_metrics(
    pred_mask=baseline_mask,
    image_gradients=image_gradients
)["edge_alignment"]

# Canary vs gradients
canary_edge_align = compute_full_boundary_metrics(
    pred_mask=canary_mask,
    image_gradients=image_gradients
)["edge_alignment"]

# Improvement = delta > threshold
delta = canary_edge_align - base_edge_align
if delta > 0.02 and bf1 >= 0.85:  # bf1 is regression guard
    improvements.append(cls)
```

**Impact if unfixed**: Promotion decision would be based on wrong metric.

---

## Additional Hardening

### 5. Gradient computation at correct resolution

**Fix**: Compute image gradients **after** loading the image at the resolution masks will use:

```python
rgb01, _ = io_utils.read_rgb_any(input_path)  # HxWx3
image_gradients = compute_image_gradients(rgb01)  # same HxW as masks
```

Original approach risked resolution mismatch.

---

### 6. Materials V3 duplicate config removed

**Issue**: `materials_v3.py` previously had a duplicate `PromptGenerationConfig` class that conflicted with the real PR-2 implementation in `backends/prompt_generation.py`.

**Status**: ✅ Already resolved in commit `0b4f2fc`.

---

## Corrected Promotion Gate

### New decision logic:

✅ **Promote FUSED to default APEX** only if:

1. **Edge alignment improves** on ≥3/5 scenes (`edge_align_delta > +0.02`)
2. **Regression guard passes** (`bf1_canary_vs_baseline ≥ 0.85` and `boundary_pixels > 0`)
3. Visual diffs show no artifacts (manual inspection)
4. Runtime delta acceptable for APEX hero frames

Otherwise: **keep canary-only** and proceed with Materials V3 PR-3B/PR-4.

---

## Files Modified

* ✅ **NEW**: `scripts/stage6_ab_with_boundary_metrics_FIXED.py` (corrected version)
* ⚠️ **ORIGINAL**: `scripts/stage6_ab_with_boundary_metrics.py` (has bugs, do not use)

---

## Recommendations

### Immediate

1. Run the **FIXED** script:
   ```bash
   python scripts/stage6_ab_with_boundary_metrics_FIXED.py
   ```

2. Review output:
   ```bash
   cat outputs/stage6_ab_boundary_metrics/stage6_ab_summary.json
   ```

3. If promotion gate passes (≥3/5 scenes improved):
   - Run visual diff validation
   - If no artifacts → promote FUSED to default APEX
   
4. If promotion gate fails:
   - Keep canary-only
   - Proceed with Materials V3 PR-3B (edge-aware gating + taxonomy)

### After A/B completion

* Archive original buggy script to `scripts/archive/`
* Rename FIXED version to canonical name
* Document decision in session summary

---

## Testing Checklist

Before running the corrected script, verify:

- [ ] Benchmark images exist in `assets/phase2_bench/`
- [ ] EfficientSAM model downloaded (`efficientsam_s.onnx`)
- [ ] Phase 2 CLIP dependencies available (or tests will skip gracefully)
- [ ] Output directory writable: `outputs/stage6_ab_boundary_metrics/`

---

## Expected Runtime

* **Per scene**: ~30–90s (2 segmentation runs + metrics)
* **Total (5 scenes)**: ~3–7 minutes

---

## Exit Codes

* `0` - Promotion gate passed (≥3/5 scenes improved)
* `1` - Promotion gate failed (keep canary-only)

---

**Session Status**: ✅ Ready to proceed with Stage 6 A/B execution
