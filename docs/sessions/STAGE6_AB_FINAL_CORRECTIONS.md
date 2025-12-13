# Stage 6 A/B Boundary Metrics: Final Corrections

**Date**: 2025-12-13  
**Status**: Ready to Run  
**Script**: `scripts/stage6_ab_corrected_final.py`

---

## Critical Fixes Applied

### 1. BoundaryMetrics Return Type Handling ✅

**Problem**: `compute_full_boundary_metrics()` returns a `BoundaryMetrics` dataclass, not a dict. Calling `.get()` on it would fail at runtime.

**Fix**: Added defensive `_as_dict()` helper:

```python
def _as_dict(metrics_obj):
    """Convert metrics object to dict defensively."""
    if metrics_obj is None:
        return {}
    if isinstance(metrics_obj, dict):
        return metrics_obj
    if hasattr(metrics_obj, "to_dict"):
        return metrics_obj.to_dict()
    raise TypeError(f"Unexpected metrics type: {type(metrics_obj)}")
```

All metrics objects are now converted via `_as_dict()` before accessing fields.

---

### 2. Material Key Canonicalization ✅

**Problem**: Stage 6 showed "water" frequently missing because mask keys varied (`pool_water`, `water_surface`, etc.) but evaluation looked for exact string match.

**Fix**: Use `normalize_material_dict()` from Materials V3 taxonomy:

```python
from lux_depth_v2.materials_v3_taxonomy import normalize_material_dict

# After extracting masks from segmenter
raw_masks = {k: masks_dict_torch[k][0,0].cpu().numpy() for k in ...}
normalized_masks = normalize_material_dict(raw_masks)  # Canonical keys
```

This maps:
- `pool_water`, `pool`, `ocean`, `sea` → `water`
- `window`, `mirror` → `glass`
- `tree`, `vegetation`, `shrub` → `foliage`

---

### 3. Minimum Boundary Pixels Guard ✅

**Problem**: Small/degenerate masks (few boundary pixels) produce noisy edge alignment scores. Counting these as "improvements" would inflate the promotion decision.

**Fix**: Added `MIN_BOUNDARY_PX = 250` threshold:

```python
is_improvement = (
    edge_align_delta > EDGE_ALIGN_IMPROVEMENT_THRESHOLD
    and bf1 >= BF1_REGRESSION_THRESHOLD
    and boundary_px >= MIN_BOUNDARY_PX  # ← Guard
)
```

Classes with < 250 boundary pixels are excluded from improvement/regression counts.

---

## Additional Hardening

### A) Fixed Device Stability

Set `FORCE_DEVICE = "cpu"` to keep A/B runs comparable (no MPS/CUDA variance).

### B) Fixed Aerial Baseline Preset

Changed from `MAX_QUALITY` to `APEX_QUALITY` to ensure apples-to-apples comparison (same tier, only EfficientSAM differs).

### C) Segmentation-Only Extraction

No longer relies on disk artifacts (`masks/*.png`). Masks are extracted directly from the segmenter in memory via `extract_masks_from_segmenter()`.

### D) Gradient Resolution Matching

Gradients are computed **at mask resolution** (after resizing RGB if needed), preventing shape mismatches in edge alignment computation.

---

## Promotion Gate (Strict)

Promote FUSED to default APEX **only if all are true**:

1. ✅ **Success rate**: ≥ 4/5 scenes complete without error
2. ✅ **Improvement rate**: ≥ 3/5 scenes show improvement on at least one target class
3. ✅ **No regressions**: Zero classes show BF1 < 0.85 with sufficient boundary pixels
4. ✅ **Edge alignment delta**: Median delta > +0.02 for improved classes

Otherwise: **KEEP CANARY**.

---

## Metrics Interpretation

### Primary Decision Metric: Edge Alignment Delta

```
edge_align_delta = edge_align_canary - edge_align_baseline
```

- **Positive delta** → canary aligns better with image gradients (real improvement)
- **Near-zero delta** → no meaningful change
- **Negative delta** → canary introduced noise/artifacts

### Regression Guard: Boundary F1 vs Baseline

```
boundary_f1 = f1(canary_boundary, baseline_boundary)
```

- **BF1 ≥ 0.85** → canary doesn't diverge wildly from baseline
- **BF1 < 0.85** → canary mask is substantially different (regression risk)

### Degenerate Case Guard: Boundary Pixels

```
boundary_pixels >= 250
```

Prevents tiny masks from counting as "improvements."

---

## What's Different from Previous Versions

| Issue | Previous Behavior | Corrected Behavior |
|-------|-------------------|-------------------|
| Metrics extraction | `.get()` on dataclass → runtime error | `.to_dict()` via helper |
| Water detection | `pool_water` ≠ `water` → missing | Canonical normalization |
| Small masks | Counted as improvement | Filtered by MIN_BOUNDARY_PX |
| Aerial baseline | MAX tier (wrong A/B) | APEX tier (correct A/B) |
| Gradient resolution | Full image size | Matched to mask size |

---

## Expected Runtime

- **Per scene**: ~15–30s (segmentation-only, CPU)
- **Total**: ~2–3 minutes for 5 scenes

---

## Next Steps

1. **Run the script**:
   ```bash
   python scripts/stage6_ab_corrected_final.py
   ```

2. **Review output**:
   ```
   outputs/stage6_ab_boundary_metrics/stage6_ab_summary.json
   ```

3. **Decision**:
   - If `promote_to_apex: true` → update default APEX presets to enable FUSED
   - If `promote_to_apex: false` → keep canary-only, proceed to Materials V3 PR-3

---

## Known Limitations

1. **No ground-truth masks**: Edge alignment uses image gradients as proxy (reasonable but not perfect)
2. **CPU-only for stability**: Faster on MPS/CUDA, but introduces variance
3. **Bathroom OOM risk**: If it still fails, that's a blocker for promotion (requires tiling/guard)

---

**Status**: Script validated, imports green, ready to execute.

**Author**: RC + Copilot  
**Session**: 2025-12-13 Stage 6 Final Validation
