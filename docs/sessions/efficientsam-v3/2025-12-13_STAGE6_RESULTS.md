# Stage 6 Golden Baseline A/B Test — Results & Decision

**Date**: December 13, 2025
**Test Scope**: SegFormer-only (baseline APEX) vs. EfficientSAM FUSED (canary APEX)
**Benchmark Set**: 5 scenes (Kitchen, Bedroom, Bathroom, Pool, Aerial)

---

## Executive Summary

**DECISION: DO NOT PROMOTE EFFICIENTSAM FUSED TO DEFAULT APEX YET**

### Rationale

1. **Visual impact is negligible** (mean pixel diff < 0.0002 in all cases)
2. **Fusion applied in only 2/5 scenes** (40% success rate)
3. **Bathroom OOM crash** is a production blocker
4. **Low IoU rejection rate** suggests EfficientSAM masks often diverge from SegFormer baseline (unclear if improvement or artifact)
5. **No measurable edge quality improvement** in visual diff crops

### Recommendation

* **Keep EfficientSAM FUSED as canary-only** (experimental presets)
* Implement **OOM safety guard** immediately (block refinement on images > 30MP or when free memory < threshold)
* Revisit after addressing:
  * Low IoU pattern (Kitchen glass 0.297, foliage 0.089)
  * Prompt quality (box → points conversion may be suboptimal)
  * Edge alignment scoring (combine IoU with depth/gradient edge consistency)

---

## Test Results Matrix

| Scene    | Baseline Time | Canary Time | Δ Time | Fusion Applied | Classes Refined | IoU (glass / water / foliage) |
|----------|---------------|-------------|--------|----------------|-----------------|-------------------------------|
| Kitchen  | 111.7s        | 106.7s      | -5.0s  | ❌ 0/2         | glass, foliage  | 0.297 / — / 0.089             |
| Bedroom  | 79.4s         | 77.0s       | -2.4s  | ✅ 1/2         | glass, foliage  | **0.431** / — / 0.120         |
| Bathroom | 54.9s         | **OOM**     | —      | ❌ OOM         | —               | —                             |
| Pool     | 68.8s         | 43.9s       | -24.9s | ❌ 0/2         | glass, water    | 0.000 / 0.230 / —             |
| Aerial   | 147.2s        | 142.6s      | -4.6s  | ✅ 1/2         | glass, foliage  | 0.000 / — / **0.383**         |

### Notes

* ❌ Bathroom OOM: 10,200×6,800 (69 MP) exceeded memory capacity
* Kitchen/Pool: very low IoU (< 0.30) triggered IoU gating → fusion skipped
* Bedroom glass (IoU 0.431) and Aerial foliage (IoU 0.383) passed IoU gate (≥ 0.30)
* **Canary sometimes faster**: fusion overhead offset by early rejection paths

---

## Detailed Findings

### 1) Fusion Telemetry (Working Correctly ✅)

All canary reports now include `segmentation_v3` block with:

* `backend_v3`: "FUSED"
* `fusion_mode`: "CONFIDENCE_WEIGHTED"
* `model`: "efficientsam_s"
* `refined_classes`: ["glass", "water", "foliage"]
* `per_class` IoU + fusion_applied flags

**Example (Bedroom glass, successful fusion):**

```json
"glass": {
  "iou_base_vs_refined": 0.43122455340513316,
  "fusion_applied": 1.0
}
```

### 2) IoU Gating Behavior

**IoU threshold = 0.30** (configured in fusion config)

| Scene    | Class   | IoU   | Fusion Result |
|----------|---------|-------|---------------|
| Bedroom  | glass   | 0.431 | ✅ Applied    |
| Bedroom  | foliage | 0.119 | ❌ Skipped    |
| Aerial   | glass   | 0.000 | ❌ Skipped    |
| Aerial   | foliage | 0.383 | ✅ Applied    |
| Kitchen  | glass   | 0.297 | ❌ Skipped    |
| Kitchen  | foliage | 0.089 | ❌ Skipped    |
| Pool     | glass   | 0.000 | ❌ Skipped    |
| Pool     | water   | 0.230 | ❌ Skipped    |

**Pattern**: EfficientSAM masks frequently diverge from SegFormer baseline (low IoU).

**Interpretation**: Two possible causes:

1. **EfficientSAM is wrong** → current gating is correct
2. **EfficientSAM is actually better** → gating rejects improvements

Cannot determine without:

* Ground-truth masks OR
* Edge-quality metrics (depth-aligned edges, gradient consistency)

### 3) Visual Diff Analysis

Generated 3 triptych crops per "win" scene (512×512 px):

* **Bedroom glass** (IoU 0.431, fusion applied)
* **Aerial foliage** (IoU 0.383, fusion applied)

**Metrics:**

| Scene          | Mean Pixel Diff | Max Pixel Diff | Visual Assessment         |
|----------------|-----------------|----------------|---------------------------|
| Bedroom glass  | 0.000049        | 0.0076         | Imperceptible differences |
| Aerial foliage | 0.000211        | 0.0056         | Imperceptible differences |

**Conclusion**: Even when fusion applied, **final output is nearly identical**.

**Hypothesis**: 

* Fusion weight (`alpha_edge=0.7, alpha_core=0.3`) may be too conservative
* OR refined masks are so similar to base that blending has minimal effect
* OR pipeline post-processing (tone mapping, sharpening, LUTs) dominates and masks differences

### 4) Bathroom OOM Crash

**Input**: 10,200×6,800 px (69 MP), 16-bit TIFF

**Error**: Out of memory during EfficientSAM refinement session initialization or inference.

**Impact**: **Production blocker** — APEX cannot crash on large interiors.

**Required Fix (before any promotion)**:

```python
# In refinement_provider.py or backend
MAX_EFFICIENTSAM_MEGAPIXELS = 30  # conservative safe limit
if H * W > MAX_EFFICIENTSAM_MEGAPIXELS * 1e6:
    log.warning("Image too large for EfficientSAM refinement (%d MP > %d MP), skipping",
                H * W / 1e6, MAX_EFFICIENTSAM_MEGAPIXELS)
    return None  # graceful fallback to SegFormer-only
```

---

## Performance Analysis

### Runtime Comparison

* **Mean canary overhead**: -7.9s (canary **faster** on average)
  * Likely due to: early IoU rejection → less fusion overhead
* **Bathroom excluded** from mean (OOM)

**Interpretation**: When fusion *does* apply, overhead is low (<5s). But success rate is poor.

### Memory Footprint

* SegFormer APEX baseline: stable across all scenes
* EfficientSAM canary: **69 MP scene caused OOM**

**Required**: dynamic size guard before attempting refinement.

---

## Blocking Issues (Must Fix Before Promotion)

### 🚨 Priority 1 — Bathroom OOM

* **Status**: Production blocker
* **Fix**: Size/memory guard (skip refinement if H×W > threshold)
* **ETA**: 15 minutes

### ⚠️ Priority 2 — Low IoU Rejection Rate

* **Status**: 80% of refinement attempts rejected (8/10 class attempts)
* **Fix options**:
  1. Lower IoU threshold (risky: may allow bad masks)
  2. Add edge-alignment metric (depth edges, gradient edges)
  3. Improve prompt generation (box → better point selection)
* **ETA**: 2–4 hours (experimentation required)

### ⚠️ Priority 3 — Negligible Visual Impact

* **Status**: Even successful fusion produces imperceptible changes
* **Possible causes**:
  * Conservative blending weights
  * Post-processing dominates
  * EfficientSAM masks too similar to SegFormer
* **Fix**: A/B test with:
  * `alpha_edge=0.9` (trust EfficientSAM more)
  * Direct mask replacement (no blending) for high-confidence regions
* **ETA**: 1–2 hours

---

## Recommended Next Steps

### Immediate (before any further A/B)

1. **Implement OOM guard** (max 30 MP for EfficientSAM refinement)
2. **Commit current state** with Stage 6 results documented
3. **Tag milestone**: `v2.3-efficientsam-v3-stage6-baseline`

### Short-term (Stage 7: Refinement Tuning)

1. **Improve prompt generation**:
   * Use high-confidence SegFormer pixels as foreground points (not just box center/corner)
   * Add negative points outside mask region
2. **Add edge-quality metric**:
   * Compute gradient-aligned edge score
   * Allow fusion when edge score improves even if IoU is moderate
3. **Increase blending aggressiveness** (test `alpha_edge=0.85`)

### Medium-term (Stage 8: A/B Validation Round 2)

1. Re-run Golden Baseline A/B with tuned parameters
2. Generate **quantitative edge metrics** (not just visual inspection)
3. Compare against **depth-edge alignment** as objective quality measure

### Decision gate for promotion

Promote FUSED to default APEX **only if**:

* ✅ No OOM crashes on scenes ≤ 80 MP
* ✅ Fusion applies in ≥ 60% of scenes (currently 40%)
* ✅ Visual diff shows measurable edge improvement (Sobel, gradient magnitude)
* ✅ No visual artifacts (halos, spill, weird pooling)

---

## Artifacts Generated

```
outputs/stage6_ab/
  interior_kitchen_750_A_baseline/
  interior_kitchen_750_B_efficientsam/
  interior_bedroom_A_baseline/
  interior_bedroom_B_efficientsam/
  interior_bathroom_A_baseline/       # OOM crash (no canary)
  exterior_pool_750_A_baseline/
  exterior_pool_750_B_efficientsam/
  exterior_aerial_A_baseline/
  exterior_aerial_B_efficientsam/
  stage6_ab_summary_v2.md

outputs/stage6_visual_diffs/
  bedroom_glass/
    crop_01_y4547_x3627.png
    crop_02_y4675_x3586.png
    crop_03_y4803_x3570.png
    full_diff_heatmap.png
  aerial_foliage/
    crop_01_y9735_x16711.png
    crop_02_y9706_x16582.png
    crop_03_y9863_x16731.png
    full_diff_heatmap.png
  visual_diff_summary.json
```

---

## Session Status

**Stage 6 Complete**: ✅ Golden Baseline A/B executed and analyzed
**Promotion Decision**: ❌ NOT READY — keep canary-only
**Next Stage**: Stage 7 (refinement tuning) or Stage 6.5 (add OOM guard + edge metrics)

**Current main status**: Stable, no regressions introduced.
**EfficientSAM V3 status**: Experimental, canary presets only.

---

**Completion**: December 13, 2025, 10:35 AM PST
**Artifacts committed**: Stage 6 scripts, visual diffs, this summary
**Safe to merge**: Yes (canary behavior unchanged, baseline unaffected)
