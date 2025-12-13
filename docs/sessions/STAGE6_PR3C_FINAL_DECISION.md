# Stage 6 PR-3C Results: EfficientSAM FUSED - Final Decision

**Date**: December 13, 2025  
**Test**: Stage 6 A/B with Boundary Metrics (PR-3C Final)  
**Device**: CPU (stable, reproducible)

---

## Executive Summary

✅ **Test completed successfully** across all 5 benchmark scenes  
❌ **Promotion NOT recommended** - keep EfficientSAM FUSED as **canary-only**

### Key Findings

- **0/5 scenes improved** with EfficientSAM refinement
- **2 regressions** detected (pool foliage, aerial foliage)
- **Glass refinement ineffective**: all interior glass classes either skipped (low coverage) or unchanged
- **Water class missing**: canonicalization did not resolve pool water detection
- **Foliage showed boundary degradation**: BF1 scores of 0.138 (pool) and 0.156 (aerial) indicate strong divergence from baseline

---

## Detailed Results by Scene

### 1. Kitchen
- **Target**: glass
- **Result**: ✗ Skipped (low coverage - 0 boundary pixels)
- **Status**: Glass not detected in meaningful quantities

### 2. Bedroom
- **Target**: glass
- **Result**: = Unchanged
- **Metrics**:
  - Boundary F1: 1.0 (perfect agreement with baseline)
  - Edge alignment: 0.367 (both baseline and canary)
  - Boundary pixels: 583 (above minimum threshold)
- **Analysis**: EfficientSAM produced identical output to SegFormer

### 3. Bathroom
- **Target**: glass
- **Result**: ✗ Skipped (low coverage - 0 boundary pixels)
- **Status**: Glass not detected in meaningful quantities

### 4. Pool (Exterior)
- **Targets**: water, foliage
- **Results**:
  - **Water**: ✗ Missing mask (canonicalization did not resolve)
  - **Foliage**: ✗ **REGRESSED**
- **Foliage Metrics**:
  - Boundary F1: **0.138** (87% boundary disagreement)
  - Edge alignment delta: **-0.006** (slight degradation)
  - Boundary pixels: 156,519 (massive boundary, highly divergent)
- **Analysis**: EfficientSAM produced dramatically different foliage boundaries that do NOT align better with image edges

### 5. Aerial
- **Target**: foliage
- **Result**: ✗ **REGRESSED**
- **Metrics**:
  - Boundary F1: **0.156** (84% boundary disagreement)
  - Edge alignment delta: **+0.005** (marginal improvement, but...)
  - Boundary pixels: 112,749
- **Analysis**: Despite tiny edge alignment gain, boundary diverged massively (BF1 failed guard threshold 0.85)

---

## Technical Analysis

### Why EfficientSAM Failed to Improve

1. **Prompt strategy still suboptimal**
   - Box→center-point prompts don't capture complex boundaries
   - PR-2 ROI improvements were insufficient
   - Need mask-driven multi-point prompts (not yet implemented)

2. **IoU gating vs boundary quality mismatch**
   - Previous Stage 6 used IoU gating (SegFormer as truth)
   - This test used boundary metrics (image gradients as truth)
   - **Result**: low BF1 scores prove EfficientSAM disagrees with SegFormer boundaries, but edge alignment doesn't justify the divergence

3. **Water class still missing**
   - Canonicalization (`normalize_material_dict`) did not resolve
   - Likely: SegFormer not emitting water at all for pool scenes
   - Materials V3 taxonomy needs deeper investigation

4. **Glass ineffective**
   - 3/3 interior scenes: either no glass detected or identical output
   - EfficientSAM refinement not triggered or not helpful

### Regression Analysis

**Pool foliage** and **Aerial foliage** both showed BF1 < 0.20, meaning:
- Canary boundary differs radically from baseline boundary
- This is a **structural disagreement**, not refinement
- Edge alignment didn't improve enough (+0.005 for aerial, -0.006 for pool) to justify massive boundary changes

---

## Decision Matrix

| Criterion | Threshold | Result | Pass |
|-----------|-----------|--------|------|
| Improved scenes | ≥3/5 | 0/5 | ❌ |
| Pool improvement | At least one class | None (water missing, foliage regressed) | ❌ |
| No regressions | BF1 ≥ 0.85 for all | 2 regressions (BF1 ~0.14-0.16) | ❌ |
| Edge alignment gains | Δ ≥ +0.02 | 0 classes | ❌ |

### Promotion Recommendation

**DO NOT promote EfficientSAM FUSED to default APEX.**

Keep as **canary-only** and:
1. Stop investing EfficientSAM time until Materials V3 gating decisions are production-ready
2. Focus on **auto-preset v2** and **Materials V3 canonical taxonomy** instead
3. Revisit EfficientSAM only if:
   - New prompt strategy (mask-driven multi-point) is implemented
   - Boundary metrics show consistent ≥+0.03 edge alignment gains
   - Water detection issue is resolved

---

## What This Means for Materials V3 Roadmap

### ✅ Validated

- Boundary metrics infrastructure works correctly
- Canonicalization framework exists (but needs debugging for water)
- Per-class gating ("should_refine") decision framework is correct approach

### ❌ Blocked

- EfficientSAM refinement does not improve edge quality with current prompt strategy
- Water class detection/mapping needs Materials V3 taxonomy fixes
- Glass detection too sparse to evaluate

### 🎯 Next Steps

1. **PR-3D: Materials V3 Taxonomy Debugging**
   - Fix water class mapping (pool scene critical)
   - Add debug logging for canonical key resolution
   - Unit tests for normalize_material_dict edge cases

2. **PR-4: Auto-Preset v2**
   - `--quality-tier auto`
   - `--intent {preview,client,hero}`
   - Complexity heuristic (gradient entropy)
   - **Hard gate**: `--allow-canary` required for EfficientSAM presets

3. **Materials V3 PR-4: Plan Mode Implementation**
   - Decision engine emits "should_refine" per class
   - Edge-aware gating uses boundary complexity
   - Emit materials_v3 report block (no pixel changes yet)

4. **EfficientSAM Stage 7 (deferred)**
   - Only after Materials V3 gating is production-validated
   - Requires new prompt strategy (farthest-point sampling from high-confidence regions)
   - Gate on boundary metrics, not IoU

---

## Reproducibility

All results are fully reproducible:

```bash
cd /Users/rc/Transformation_Portal
python scripts/stage6_ab_final_v3.py
```

Output:
- `outputs/stage6_ab_pr3c_final/stage6_ab_summary.json`
- `outputs/stage6_ab_run.log`

Device: CPU (forced for stability)  
Benchmark set: 5 scenes from `projects/750_picacho_lane/Final_Production_UltraQuality/`

---

## Conclusion

EfficientSAM V3 infrastructure is correctly implemented, observable, and safe. The **problem is not the implementation**—it's that **EfficientSAM refinement with current prompts does not improve edge quality** on these scenes.

**Keep canary-only. Move on to Materials V3 taxonomy + auto-preset v2.**

---

**Session Status**: ✅ PR-3C Complete  
**Promotion Decision**: ❌ **Do NOT promote** - canary-only remains correct  
**Next PR**: PR-3D (Materials V3 taxonomy debugging) or PR-4 (auto-preset v2)
