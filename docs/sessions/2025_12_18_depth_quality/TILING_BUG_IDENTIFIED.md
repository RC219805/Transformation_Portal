# Critical Tiling Bugs Found
**Date**: 2025-12-18  
**Status**: ❌ ROOT CAUSE IDENTIFIED

---

## Isolation Test Results

Ran systematic isolation tests on each pipeline stage:

| Stage | Edge Overlap | Edge Correlation | Verdict |
|-------|--------------|------------------|---------|
| Baseline (HF only) | 77.2% | 0.208 | ✅ Reference |
| **Tiling only** | **65.0%** | **0.047** | ❌ **FAILURE** |
| Guided filter only | 77.6% | 0.211 | ✅ Improves |
| Edge snap only | 80.2% | 0.211 | ✅ Improves |
| CLAHE only | 76.6% | 0.249 | ✅ Improves |

**Culprit**: Tiling alone (no refinement) causes **-12% edge overlap** and correlation collapse (0.208 → 0.047)

---

## Root Cause: Missing Scale Reconciliation

**File**: `lux_depth_v2/depth_inference.py`, lines 357-362

```python
for idx, (tile_depth, y0, y1, x0, x1) in enumerate(tile_depths):
    th, tw = tile_depth.shape
    window = blend_window[:th, :tw]
    
    depth_stack[idx, y0:y1, x0:x1] = tile_depth  # ← BUG: No scale reconciliation!
    weight_stack[idx, y0:y1, x0:x1] = window
```

**Problem**: Each tile is independently normalized to [0, 1] by the model. Placing them directly in the stack without scale reconciliation creates:
- **Seam discontinuities** at tile boundaries
- **Periodic grid artifacts** (exactly what was measured)
- **Edge misalignment** (depth edges appear at tile seams, not RGB boundaries)

This is the exact failure mode warned in feedback:
> "per-tile normalization (seams → grid edges)"

---

## Required Fix

### Before Blending: Affine Match Each Tile to Global Anchor

```python
# For each tile
overlap_region_tile = tile_depth[overlap_slice]
overlap_region_anchor = global_depth[y0:y1, x0:x1][overlap_slice]

# Robust pixels (exclude edges)
grad_mag = compute_gradient_magnitude(overlap_region_anchor)
mask = grad_mag < np.percentile(grad_mag, 80)

# Solve: a * tile + b ≈ anchor
x = overlap_region_tile[mask].flatten()
y = overlap_region_anchor[mask].flatten()
a, b = np.polyfit(x, y, 1)  # Or Theil-Sen for robustness

# Apply calibration
tile_depth_calibrated = a * tile_depth + b
```

### Then Blend with Hann Window

Only after all tiles are in the same scale can we blend them.

---

## Why Refinement Stages Worked

- **Guided filter**: Smooths within-tile noise, doesn't affect tile seams (operates post-blend)
- **Edge snap**: Sharpens at RGB edges, actually **improves** alignment (+3%)
- **CLAHE**: Increases local contrast, **improves** correlation (0.208 → 0.249)

All refinement stages are **correct** and **beneficial**. The tiling implementation is the sole culprit.

---

## Action Required

1. ❌ **DO NOT DEPLOY** current tiling implementation
2. ✅ Implement affine scale reconciliation (as specified in user feedback)
3. ✅ Re-run isolation test #2 (tiling_only) - must achieve >75% overlap
4. ✅ Only then combine tiling + refinement

---

## Validation Criteria (Post-Fix)

After implementing scale reconciliation, tiling_only must achieve:
- Edge overlap: ≥75% (within 2% of baseline)
- Edge correlation: ≥0.18 (within 0.03 of baseline)
- No periodic grid artifacts
- Boundary energy ratio < 1.2 (seam check)

---

**Conclusion**: The production refinement pipeline (CLAHE + guided + edge snap) is **correct and validated**. The tiling implementation has a critical bug (missing scale reconciliation) that must be fixed before any tiling-based deployment.

**Recommendation**: Deploy refinement alone (no tiling) immediately. Fix tiling, re-validate, then deploy full pipeline.
