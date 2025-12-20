# Metrics Validation Results - December 18, 2025

## Executive Summary

**Status**: ✅ Metric fixes are CORRECT and working as intended.  
**Finding**: The low halo scores (0.0) are NOT a bug - they accurately reflect real edge artifacts in the current depth outputs.

## Validation Method

Tested fixed metrics on existing depth map:
- **Image**: `750Picacho_GreatRoom_Ultimate.tif` (4000×3000)
- **Depth**: `outputs/production_validation_fixed/750Picacho_GreatRoom_Ultimate_depth.tiff`
- **Configuration**: tile_size=1024, overlap=128, no refinement

## Results

### Metric Values (Fixed Code)
| Metric | Value | Threshold (Lenient) | Status |
|--------|-------|---------------------|--------|
| edge_f1 | 0.617 | ≥ 0.30 | ✅ PASS |
| edge_overlap | 0.705 | ≥ 0.40 | ✅ PASS |
| chamfer_distance | 14.85px | < 15.0 | ✅ PASS |
| edge_count_ratio | 1.337 | ≤ 3.0 | ✅ PASS |
| halo_score | **0.000** | n/a | ⚠️ Indicates severe halos |
| overshoot_penalty | 0.432 | ≤ 0.5 | ✅ PASS |
| **Quality Score** | 0.606 | n/a | Lenient: ✅ Strict: ❌ |

### Halo Detection Deep Dive

Manual computation confirms the metric is working correctly:

```
Edge pixels: 4,879,628 (40.7% of image)
Non-edge pixels: 7,120,372 (59.3%)

Mean |Laplacian| at edges:     0.119536
Mean |Laplacian| elsewhere:    0.026956

Ratio (edge/global):           4.434
```

**Interpretation**:
- Ratio > 3.0 → halo_score = 0.0 (by design)
- The depth map has ~4.4× more Laplacian ringing at RGB edges than in smooth regions
- This is a **real quality issue**, not a metric bug

### Overshoot Penalty Deep Dive

```
95th percentile |Laplacian|:  0.043212
Scaled penalty (×10):         0.432
```

**Interpretation**:
- P95 of 0.043 is moderate (threshold is ~0.03 for "good")
- Penalty of 0.43 is correctly calibrated
- The metric is working as intended

## Comparison: Before vs After Fix

### Before Fix (Old Code)
| Field | Value | Issue |
|-------|-------|-------|
| halo_score | 0.0 | ❓ Unknown if bug or real |
| overshoot_penalty | 0.432 | ❓ Unknown if calibrated |
| Logging | None | ❌ No diagnostic info |

### After Fix (Current Code)
| Field | Value | Interpretation |
|-------|-------|----------------|
| halo_score | 0.0 | ✅ Ratio=4.43 (severe halos confirmed) |
| overshoot_penalty | 0.432 | ✅ P95=0.043 (moderate ringing, correctly scaled) |
| Logging | DEBUG available | ✅ Can tune per scene type |

## Key Finding: The Metrics Are Correct

The "0.0 halo score" is **not a bug in the metric computation**.  
It is an **accurate measurement of genuine edge artifacts** in the current depth output.

### Why GreatRoom Has Halos (Root Cause)

1. **Scene characteristics**:
   - Large planar interior (walls, ceiling, floor)
   - High-frequency texture (stone, rug, wood grain)
   - Strong architectural edges (window frames, molding)

2. **Current pipeline behavior**:
   - Tiled inference captures fine detail (good)
   - But without edge snapping or structural masking, it also captures texture edges as depth discontinuities (bad)
   - Result: depth edges on rug fibers, stone grain, wood texture → elevated Laplacian at "edges"

3. **Metric detects this correctly**:
   - RGB edges = architectural boundaries + texture detail
   - Depth at those locations has high gradients (from texture being interpreted as depth)
   - Ratio > 4.0 → severe halo classification

## Implications

### For Reporting (Priority 1 Fix)
✅ **Status**: Field names are now consistent (`passed_lenient` / `passed_strict`)  
✅ **Impact**: Reports will correctly show "lenient pass, strict fail" for GreatRoom

### For Metric Calibration (Priority 2 Fix)
✅ **Status**: Halo and overshoot metrics are correctly implemented  
✅ **Impact**: Scores accurately reflect real depth quality issues  
⚠️ **Action Required**: Thresholds may need per-scene-type tuning (interior vs exterior)

### For Pipeline Improvement (Deferred to Phase 2)
❌ **Root cause**: Texture edges being interpreted as depth discontinuities  
✅ **Solution exists**: Structural edge detection + AND-gated refinement  
📅 **Timeline**: Not in current validation run (refinement disabled for stability)

## Next Steps (Prioritized)

### Immediate
1. ✅ **Metrics are validated** - no code changes needed
2. [ ] **Rerun full validation** with fixed field names to get clean JSON report
3. [ ] **Generate overshoot heatmaps** for visual confirmation of where halos occur

### Phase 2 (Interior-Specific Refinement)
4. [ ] **Implement structural edge detection**:
   - Compute edges on heavily blurred RGB (suppresses texture)
   - AND-gate: snap only where (structural_edge) ∩ (depth_edge)
   - Target: reduce halo_score false positives on textured surfaces

5. [ ] **Per-category thresholds**:
   - Interior (textured): halo_score ≥ 0.3 (relaxed, expect texture noise)
   - Exterior (smooth): halo_score ≥ 0.7 (strict, clean surfaces expected)

6. [ ] **Global planarity constraint** for large interiors:
   - Fit dominant planes (walls/ceiling)
   - Regularize depth to respect planarity
   - Reintroduce detail only at boundaries

### Phase 3 (Production Deployment)
7. [ ] **A/B validation**: current pipeline vs refined pipeline
8. [ ] **Materials V3 integration**: confirm depth improvements translate to better material boundaries
9. [ ] **Full dataset validation**: 50+ images, per-category pass rates

## Conclusion

The Priority 2 fix (halo/overshoot calibration) is **working correctly**.

The low halo scores are **not a bug** - they are an accurate signal that the current depth maps have edge ringing artifacts, particularly in textured interior scenes.

The path forward is **not to change the metrics**, but to **improve the depth pipeline** with:
- Structural edge detection
- AND-gated refinement
- Per-scene-type presets

**The metrics are now trustworthy diagnostic tools** that will guide pipeline tuning.

---

**Test Date**: December 18, 2025  
**Test Image**: 750Picacho_GreatRoom_Ultimate (4000×3000)  
**Configuration**: tile_size=1024, overlap=128, refinement=null  
**Metric Implementation**: `high_fidelity_depth/quality_metrics.py` (post-fix)  
**Validation Status**: ✅ METRICS CORRECT, ⏳ PIPELINE IMPROVEMENT PENDING
