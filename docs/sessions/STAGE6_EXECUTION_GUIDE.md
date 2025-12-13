# Stage 6 A/B Boundary Metrics: Ready to Execute

**Date**: 2025-12-13 23:13 UTC  
**Status**: ✅ **VALIDATED AND READY**  
**Script**: `scripts/stage6_ab_corrected_final.py`  
**Sanity Check**: `scripts/stage6_sanity.py`

---

## Summary

All three critical issues from the previous version have been fixed:

1. ✅ **BoundaryMetrics return type** handling via `.to_dict()`
2. ✅ **Material key canonicalization** via `normalize_material_dict()`
3. ✅ **Minimum boundary pixels guard** (250 px threshold)

Additional hardening:
- ✅ Device stability (`FORCE_DEVICE = "cpu"`)
- ✅ Correct baseline presets (APEX for all scenes)
- ✅ Segmentation-only mask extraction (no disk I/O)
- ✅ Gradient resolution matching

---

## How to Run

### Option 1: Sanity Check First (Recommended)

Run kitchen scene only (~20s):

```bash
python scripts/stage6_sanity.py
```

Expected output:
```
Sanity check: kitchen
Input: assets/phase2_bench/750Picacho_Kitchen_Ultimate.tif
Exists: True
Device: cpu

=== KITCHEN ===
  Baseline: INTERIOR_LUXURY_APEX_QUALITY
  Canary: INTERIOR_LUXURY_APEX_QUALITY_EFFICIENTSAM
  glass: ...
  ...

=== RESULT ===
Status: success
Improved: True/False
Improvements: [...]
Regressions: []

✓ Sanity check PASSED
```

### Option 2: Full Benchmark (2–3 minutes)

Run all 5 scenes:

```bash
python scripts/stage6_ab_corrected_final.py
```

Expected output:
```
Stage 6 A/B Test: Boundary Metrics (Corrected Final)
Device: cpu
Min boundary pixels: 250
Edge align threshold: +0.02
BF1 regression threshold: 0.85

=== KITCHEN ===
  ...
=== BEDROOM ===
  ...
=== BATHROOM ===
  ...
=== POOL ===
  ...
=== AERIAL ===
  ...

=== SUMMARY ===
Success: 5/5 (or 4/5 if bathroom OOMs)
Improved: X/5
Improvements: [glass, water, foliage, ...]
Regressions: []

DECISION: PROMOTE / KEEP CANARY

Full results: outputs/stage6_ab_boundary_metrics/stage6_ab_summary.json
```

---

## Promotion Decision Logic

**PROMOTE to default APEX** only if:

```python
success_count >= 4
and improved_count >= 3
and len(all_regressions) == 0
```

Otherwise: **KEEP CANARY**.

Per-class improvement criteria:
```python
edge_align_delta > +0.02        # Real edge improvement
and boundary_f1 >= 0.85         # No catastrophic regression
and boundary_pixels >= 250      # No degenerate mask
```

---

## Expected Outcomes

### Scenario A: Promotion (Optimistic)

```json
{
  "promote_to_apex": true,
  "decision": "PROMOTE: Enable FUSED in default APEX presets",
  "success_count": 5,
  "improved_count": 4,
  "all_improvements": ["glass", "water", "foliage"],
  "all_regressions": []
}
```

**Next action**: Update APEX presets to set `backend_v3=FUSED` by default.

### Scenario B: Keep Canary (Realistic)

```json
{
  "promote_to_apex": false,
  "decision": "KEEP CANARY: Insufficient improvement or regressions present",
  "success_count": 4,
  "improved_count": 2,
  "all_improvements": ["glass", "foliage"],
  "all_regressions": []
}
```

**Next action**: Proceed to Materials V3 PR-3 (edge-aware gating + taxonomy).

### Scenario C: Regressions (Red Flag)

```json
{
  "promote_to_apex": false,
  "decision": "KEEP CANARY: Insufficient improvement or regressions present",
  "success_count": 5,
  "improved_count": 3,
  "all_improvements": ["glass", "water", "foliage"],
  "all_regressions": ["water"]  # ← Problem
}
```

**Next action**: Investigate regression, tune fusion config or prompts.

---

## Files Created

1. **`scripts/stage6_ab_corrected_final.py`** (main script, 400+ lines)
2. **`scripts/stage6_sanity.py`** (quick validation, kitchen only)
3. **`docs/SESSIONS/STAGE6_AB_FINAL_CORRECTIONS.md`** (technical explanation)
4. **This file** (execution guide)

---

## Validation Checklist

- ✅ Imports validated
- ✅ Constants verified (`MIN_BOUNDARY_PX=250`, `TARGET_CLASSES=[glass,water,foliage]`)
- ✅ Input files exist (symlinks to `input_images/750_Picacho/Ultimate_TIFFs_Base/`)
- ✅ Boundary metrics module tested (`BoundaryMetrics.to_dict()`)
- ✅ Taxonomy normalization tested (`normalize_material_dict()`)
- ✅ Device set to CPU for stability

---

## Post-Execution

After running, review:

```bash
# Summary JSON
cat outputs/stage6_ab_boundary_metrics/stage6_ab_summary.json | jq

# Key fields
jq '.promote_to_apex' outputs/stage6_ab_boundary_metrics/stage6_ab_summary.json
jq '.decision' outputs/stage6_ab_boundary_metrics/stage6_ab_summary.json
jq '.all_improvements' outputs/stage6_ab_boundary_metrics/stage6_ab_summary.json
jq '.all_regressions' outputs/stage6_ab_boundary_metrics/stage6_ab_summary.json

# Per-scene details
jq '.scene_results[] | {scene, scene_improved, improvements, regressions}' \
  outputs/stage6_ab_boundary_metrics/stage6_ab_summary.json
```

---

## Known Risks

1. **Bathroom OOM**: If it happens again, that's a blocker for promotion (requires size guard).
2. **Pool water missing**: Canonicalization should fix this, but verify in results.
3. **Low edge alignment deltas**: If all deltas < +0.02, no improvements will be counted.

---

## Next Steps After This Run

### If PROMOTE:

1. Update default APEX presets:
   ```python
   # lux_depth_v2/config.py
   # In interior_luxury_apex_quality and exterior_pool_apex_quality:
   self.segmentation.backend_v3 = SegmentationBackend.FUSED
   ```

2. Run Golden Baseline validation (full pipeline, not just segmentation).

3. Update docs to reflect FUSED as default for APEX.

### If KEEP CANARY:

1. Proceed to **Materials V3 PR-3** (edge-aware gating + taxonomy).

2. Consider prompt tuning or IoU threshold adjustments for future canary improvements.

3. Archive Stage 6 results as baseline for future EfficientSAM iterations.

---

**Author**: RC + GitHub Copilot  
**Session**: 2025-12-13 Stage 6 Final Validation  
**Repository**: Transformation_Portal (`origin/main` @ commit f869f73)

**Ready to execute.**
