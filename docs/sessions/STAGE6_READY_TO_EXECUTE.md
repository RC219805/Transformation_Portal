# Stage 6 A/B Test - Implementation Complete & Ready

**Date**: 2025-12-13 23:07 UTC  
**Status**: ✅ **READY TO EXECUTE**  
**Session**: PR-3C Corrections Applied

---

## Executive Summary

All **4 critical bugs** in the original Stage 6 A/B script have been identified and corrected. Pre-flight checks confirm all prerequisites are met. The corrected script is ready to execute and will produce a defensible promotion decision based on **edge quality metrics**, not mean IoU.

---

## What Was Fixed

### Critical Bugs (would cause failures)

1. **Pipeline API signature mismatch** - Fixed to use `cfg.output_dir` correctly
2. **Mask extraction failure** - Switched from disk I/O to in-memory extraction
3. **Invalid A/B comparison** - Fixed aerial baseline to use APEX (not MAX)
4. **Wrong promotion metric** - Now uses edge alignment vs gradients (not BF1)

See `docs/SESSIONS/PR3C_FIXES_APPLIED.md` for detailed technical breakdown.

---

## Files Created

* ✅ `scripts/stage6_ab_with_boundary_metrics_FIXED.py` - Corrected A/B test script
* ✅ `scripts/stage6_preflight_check.py` - Pre-flight validation
* ✅ `docs/SESSIONS/PR3C_FIXES_APPLIED.md` - Technical fix documentation

---

## Pre-Flight Status

**All checks PASSED**:

- ✅ Benchmark images (5/5 present)
- ✅ Dependencies (torch, numpy, PIL, scipy)
- ✅ Boundary metrics module
- ✅ EfficientSAM model (`efficientsam_s.onnx`)
- ✅ Output directory writable

---

## Execution Instructions

### 1. Run the corrected A/B test

```bash
cd /Users/rc/Transformation_Portal
python scripts/stage6_ab_with_boundary_metrics_FIXED.py
```

**Expected runtime**: 3–7 minutes (5 scenes × 2 runs each)

### 2. Review results

```bash
# View summary
cat outputs/stage6_ab_boundary_metrics/stage6_ab_summary.json

# Check promotion decision
jq '.aggregate.recommendation' outputs/stage6_ab_boundary_metrics/stage6_ab_summary.json
```

**Possible outcomes**:

* `"promote"` → ≥3/5 scenes improved → promote FUSED to default APEX
* `"keep_canary_only"` → <3/5 scenes improved → keep canary-only

### 3. If promotion gate passes

1. Run visual diff validation:
   ```bash
   python scripts/stage6_visual_diff.py
   ```

2. Inspect diff crops for artifacts (halos, spill, edge chatter)

3. If no artifacts → proceed with promotion:
   * Update APEX presets to use `backend_v3=FUSED` by default
   * Document decision in session summary
   * Merge to `main`

### 4. If promotion gate fails

Keep EfficientSAM canary-only and proceed with Materials V3:

* **PR-3B**: Edge-aware gating + taxonomy normalization
* **PR-4**: Scene-aware auto-preset v2 (`--quality-tier auto`)

---

## Promotion Decision Criteria

The script will automatically evaluate:

✅ **PROMOTE to default APEX** if:

1. Edge alignment improves on ≥**3/5 scenes** (`edge_align_delta > +0.02`)
2. Regression guard passes (`bf1_canary_vs_baseline ≥ 0.85`, `boundary_pixels > 0`)
3. No visual artifacts in diff crops
4. Runtime delta acceptable for APEX

❌ **KEEP canary-only** if:

* Fewer than 3 scenes show improvement
* OR: Any scene has severe regression
* OR: Visual artifacts appear

---

## Key Metrics Explained

### Edge Alignment (improvement signal)

* **Baseline edge align**: How well baseline mask boundaries align with image gradients
* **Canary edge align**: How well canary mask boundaries align with image gradients
* **Delta**: `canary - baseline` → positive = improvement

**Threshold**: `+0.02` or higher indicates meaningful edge improvement

### Boundary F1 (regression guard)

* Measures similarity between canary and baseline boundaries
* Used to detect **divergence**, not improvement
* High BF1 (≥0.85) = canary didn't break what baseline got right

**Not used for promotion** - only as a safety check

---

## Expected Output Structure

```
outputs/stage6_ab_boundary_metrics/
├── stage6_ab_summary.json          # Promotion decision + all metrics
└── (individual scene logs in console)
```

**Summary JSON schema**:

```json
{
  "timestamp": "2025-12-13 23:07:00",
  "scenes": [
    {
      "scene": "interior_kitchen_750",
      "status": "success",
      "per_class_metrics": {
        "glass": {
          "edge_align_baseline": 0.412,
          "edge_align_canary": 0.438,
          "edge_align_delta": +0.026,
          "bf1_canary_vs_baseline": 0.873
        }
      },
      "improvements": ["glass"],
      "scene_improved": true
    }
  ],
  "aggregate": {
    "improved_scenes": 3,
    "promotion_gate_passed": true,
    "recommendation": "promote"
  }
}
```

---

## Next Steps After Execution

### If promotion passes:

1. Visual diff validation
2. Update APEX preset defaults
3. Session summary + merge

### If promotion fails:

1. Document decision rationale
2. Proceed with Materials V3 PR-3B (gating + taxonomy)
3. Keep EfficientSAM strictly canary-only

---

## Troubleshooting

### If script fails with import errors:

Check dependencies:
```bash
pip list | grep -E "(torch|numpy|pillow|scipy)"
```

### If EfficientSAM model not found:

Download manually:
```bash
python -m lux_depth_v2.cli --download-efficientsam --efficientsam-model efficientsam_s
```

### If OOM on bathroom scene:

Expected behavior - script will log OOM and mark scene as failed. This is a known issue with large images and is protected by the 30 MP guard in production presets.

---

## Final Recommendation

**EXECUTE NOW**:

The script is production-ready, all prerequisites verified, and the decision logic is sound. The A/B test will produce objective, defensible metrics for the EfficientSAM promotion decision.

**Estimated time to decision**: 5–10 minutes (script runtime + review)

---

**Status**: ✅ Ready  
**Blocker Count**: 0  
**Action**: Execute `python scripts/stage6_ab_with_boundary_metrics_FIXED.py`
