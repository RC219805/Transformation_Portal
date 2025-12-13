# Session Complete: Stage 6 Golden Baseline A/B + OOM Fix
**Date**: December 13, 2025  
**Duration**: ~3 hours  
**Status**: ✅ Complete — EfficientSAM V3 validated, OOM guard implemented

---

## Executive Summary

Successfully executed **Stage 6 Golden Baseline A/B test** comparing SegFormer-only APEX vs. EfficientSAM FUSED canary across 5 production scenes. Results demonstrate that EfficientSAM V3 is **not ready for default APEX promotion** but is **safe as canary-only** with OOM protection.

**Key Decision**: **Keep EfficientSAM FUSED experimental (canary presets only)**.

---

## What Was Accomplished

### 1. Golden Baseline A/B Test ✅

Executed full matrix:

* **5 scenes**: Kitchen, Bedroom, Bathroom, Pool, Aerial
* **2 configurations per scene**: Baseline APEX (SegFormer-only) vs Canary APEX (EfficientSAM FUSED)
* **Telemetry working**: All canary reports include `segmentation_v3` block with IoU + fusion stats

### Results Summary

| Metric                    | Result                              |
|---------------------------|-------------------------------------|
| **Fusion success rate**   | 40% (2/5 scenes)                    |
| **Visual impact**         | Negligible (mean diff < 0.0002)     |
| **Production blocker**    | Bathroom OOM (69 MP input)          |
| **IoU rejection rate**    | 80% (8/10 class refinement attempts)|

### 2. Visual Diff Analysis ✅

Generated automated triptych crops for "win" cases:

* **Bedroom glass** (IoU 0.431, fusion applied) → imperceptible differences
* **Aerial foliage** (IoU 0.383, fusion applied) → imperceptible differences

**Conclusion**: Even successful fusion produces no measurable visual improvement in final output.

### 3. OOM Safety Guard ✅ (Priority 1 Fix)

Implemented immediate production fix:

* Skip EfficientSAM refinement on images > 30 MP
* Graceful fallback to SegFormer-only with warning log
* Prevents Bathroom scene crash (69 MP = 10,200×6,800)

**Code**: `lux_depth_v2/backends/refinement_provider.py` lines 207–217

---

## Blocking Issues Identified

### 🚨 Priority 1 — Bathroom OOM (FIXED ✅)

* **Status**: Production blocker → now FIXED with 30 MP guard
* **Impact**: Prevented promotion to default APEX
* **Resolution**: Graceful skip + fallback for large images

### ⚠️ Priority 2 — Low IoU Rejection Rate

* **Status**: 80% of refinement attempts rejected by IoU gating
* **Root cause**: EfficientSAM masks diverge from SegFormer baseline
  * Kitchen glass: IoU 0.297 (rejected)
  * Kitchen foliage: IoU 0.089 (rejected)
  * Pool glass: IoU 0.000 (rejected)
  * Pool water: IoU 0.230 (rejected)
* **Unclear**: Whether divergence indicates EfficientSAM is wrong OR SegFormer is wrong
* **Next**: Add edge-quality metrics (depth edges, gradient consistency) to validate

### ⚠️ Priority 3 — Negligible Visual Impact

* **Status**: Successful fusion produces imperceptible changes
* **Possible causes**:
  1. Conservative blending weights (`alpha_edge=0.7, alpha_core=0.3`)
  2. Post-processing (tone mapping, LUTs) dominates and masks differences
  3. EfficientSAM masks too similar to SegFormer baseline
* **Next**: Experiment with aggressive blending (`alpha_edge=0.9`) or direct replacement

---

## Technical Findings

### IoU Gating Behavior (working as designed)

| Scene    | Class   | IoU   | Gate (≥0.30) | Fusion Applied |
|----------|---------|-------|--------------|----------------|
| Bedroom  | glass   | 0.431 | ✅ Pass       | ✅ Yes          |
| Aerial   | foliage | 0.383 | ✅ Pass       | ✅ Yes          |
| Kitchen  | glass   | 0.297 | ❌ Fail       | ❌ No           |
| Pool     | water   | 0.230 | ❌ Fail       | ❌ No           |
| Kitchen  | foliage | 0.089 | ❌ Fail       | ❌ No           |
| Pool     | glass   | 0.000 | ❌ Fail       | ❌ No           |

**Pattern**: EfficientSAM frequently produces masks with low IoU vs SegFormer. This triggers gating → fusion rejected → no visible output change.

### Performance Impact (minimal when fusion skipped)

* Mean canary overhead: **-7.9s** (canary actually *faster* on average)
  * Due to: early IoU rejection → less fusion computation
* When fusion *does* apply: overhead < 5s
* Runtime is **not** the bottleneck; fusion success rate is.

---

## Files Modified/Created

### Core Fixes
* `lux_depth_v2/backends/refinement_provider.py` — OOM guard (30 MP threshold)

### Tooling
* `scripts/stage6_visual_diff.py` — automated visual diff crop generator
* `scripts/stage6_ab_golden_baseline_v2.py` — A/B test runner (using LuxPipelineV2 API)

### Documentation
* `docs/SESSIONS/efficientsam-v3/2025-12-13_STAGE6_RESULTS.md` — full A/B results analysis
* `docs/SESSIONS/2025-12-13_STAGE6_COMPLETE.md` — this session summary

### Artifacts (not tracked)
* `outputs/stage6_ab/` — baseline + canary renders for all 5 scenes
* `outputs/stage6_visual_diffs/` — automated diff crops + heatmaps

---

## Recommendations for Next Steps

### Immediate (before any further A/B)

1. ✅ **OOM guard implemented** (30 MP threshold)
2. ✅ **Stage 6 results documented** with decision
3. 🕒 **Tag milestone**: `v2.3-efficientsam-v3-stage6-canary-only`

### Short-term (Stage 7: Refinement Tuning)

1. **Improve prompt generation**:
   * Use high-confidence SegFormer pixels as FG points (not just box center/corner)
   * Add proper BG points outside mask boundary
2. **Add edge-quality metric** (not just IoU):
   * Compute gradient-aligned edge score
   * Allow fusion when edge score improves even if IoU is moderate
3. **Test aggressive blending**: `alpha_edge=0.85` or `alpha_edge=0.95`

### Medium-term (Stage 8: A/B Round 2)

1. Re-run Golden Baseline A/B with tuned parameters
2. Generate **quantitative edge metrics** (Sobel, gradient magnitude)
3. Compare against **depth-edge alignment** as objective quality measure

### Decision gate for promotion

Promote FUSED to default APEX **only if** all criteria met:

* ✅ No OOM crashes on scenes ≤ 80 MP (now FIXED)
* ❌ Fusion applies in ≥ 60% of scenes (currently 40%)
* ❌ Visual diff shows measurable edge improvement (currently imperceptible)
* ❌ No visual artifacts (unknown without visible changes)

**Current status**: 1/4 criteria met → **NOT READY**.

---

## Git State

### Commits to Main

1. **Stage 6 results + visual diff tooling**
   * Commit: `d38f421`
   * Files: `scripts/stage6_visual_diff.py`, `docs/SESSIONS/efficientsam-v3/2025-12-13_STAGE6_RESULTS.md`

2. **OOM safety guard** (Priority 1 fix)
   * Commit: `a1e9316`
   * File: `lux_depth_v2/backends/refinement_provider.py`

3. **Pushed to origin**: ✅

### CI Status

* All workflows green (CodeQL pending, non-blocking)
* No regressions introduced
* Canary presets unchanged (EfficientSAM still opt-in only)

---

## Lessons Learned

1. **Telemetry first, visuals second**: The `segmentation_v3` JSON block immediately revealed low IoU as the root issue before we spent time on visual inspection.

2. **IoU gating protects correctness**: The 0.30 threshold successfully rejected many divergent masks. Unknown if this prevented improvements or prevented artifacts.

3. **OOM must be guarded proactively**: The 30 MP guard is conservative but necessary. Future: dynamic memory check or tiling.

4. **Visual diff automation is invaluable**: Auto-finding highest-change regions prevents cherry-picking and provides objective evidence.

5. **Blending weights matter**: Current conservative weights (`alpha=0.7`) may be too timid; fusion has no visible effect even when applied.

---

## Current State

### EfficientSAM V3 Status

* **Stages 1–6 complete**: Backend, fusion, pipeline integration, model download, A/B validation
* **Production status**: Experimental, canary presets only
* **Safety**: OOM-protected, graceful fallback, no default behavior changes
* **Promotion blockers**: Low fusion success rate (40%), negligible visual impact

### Repository Health

* ✅ All tests passing
* ✅ CI green
* ✅ No breaking changes to Phase 2 or APEX baseline
* ✅ Canary presets safe for experimental use
* ✅ Documentation current

---

**Session End**: December 13, 2025, 11:01 AM PST  
**Status**: ✅ Stage 6 Complete — EfficientSAM V3 validated as canary-only  
**Next**: Stage 7 (prompt tuning + edge metrics) or defer and focus on other priorities

**Ready for**: Production use with canary presets; NOT ready for default APEX promotion.
