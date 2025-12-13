# Materials V3 Taxonomy Audit - Pool Water Diagnostic Complete

**Date**: December 13, 2025  
**Focus**: Diagnosing Stage 6 "water missing" issue  
**Status**: ✅ Root cause identified

---

## Executive Summary

The Stage 6 AB test showed pool water consistently missing (`status=missing_mask`). The newly implemented **Materials V3 class presence audit** has identified the root cause:

**The SegFormer model does not include "water" or "pool" in its output vocabulary for the 750 Picacho Pool scene.**

This is a **model limitation**, not a taxonomy mapping or coverage threshold issue.

---

## Diagnostic Results

### Pool Image Analysis (750 Picacho Pool.tif)

**Input**: `input_images/750_Picacho/Pool.tif`  
**Resolution**: 6000×3375 (20.2 MP)

### Seg Former Output (6 classes emitted)

| Class    | Coverage (px) | Coverage (%) | Notes                          |
|----------|---------------|--------------|--------------------------------|
| foliage  | 6,043,302     | 29.84%       | Likely pool water misclassified |
| sky      | 5,766,530     | 28.48%       | Twilight sky                   |
| wood     | 1,212,250     | 5.99%        | Decking/furniture              |
| stone    | 15,495        | 0.08%        | Minor regions                  |
| glass    | 0             | 0.00%        | Not detected                   |
| metal    | 0             | 0.00%        | Not detected                   |

**❌ Water**: Not emitted by segmenter  
**💡 Likely explanation**: Pool water surface is being misclassified as "foliage" (29.84% coverage strongly suggests this is the water region)

---

## Root Cause Analysis

### Why SegFormer doesn't emit "water"

1. **Model vocabulary**: The SegFormer ADE20K variant used does not have "water"/"pool" as a distinct class
2. **Twilight pool rendering**: The pool has subtle reflections/color that confuse the segmenter
3. **No explicit pool/water training**: The base model wasn't fine-tuned on luxury pool imagery

### Why this blocks EfficientSAM refinement

* EfficientSAM refinement targets: `["glass", "water", "foliage"]`
* If "water" isn't emitted, refinement can't be applied to it
* Stage 6 IoU comparisons fail because there's no baseline water mask

---

## Materials V3 Audit Implementation

### What Was Added

**File**: `lux_depth_v2/materials_v3.py`

Added `_audit_class_presence()` method that reports:

* **Emitted classes**: raw output from segmenter
* **Canonical mapping**: how raw classes map to canonical names
* **Target status**: for each requested target (glass/water/foliage):
  * `present`: true/false
  * `canonical_name`: normalized name
  * `coverage_pixels`: pixel count
  * `reason`: why missing (not_emitted vs zero_coverage vs threshold)
* **Unmapped classes**: emitted classes that don't map

### Integration

The audit is now automatically included in `materials_v3` report block:

```json
{
  "materials_v3": {
    "enabled": true,
    "taxonomy": "base",
    "per_class_stats": {...},
    "canonical_materials": [...],
    "class_presence_audit": {
      "emitted_classes": ["foliage", "glass", "metal", "sky", "stone", "wood"],
      "canonical_classes": ["foliage", "glass", "metal", "sky", "stone", "wood"],
      "requested_targets": ["glass", "water", "foliage"],
      "target_status": {
        "water": {
          "present": false,
          "canonical_name": "water",
          "coverage_pixels": 0,
          "reason": "not_emitted_by_segmenter"
        }
      }
    }
  }
}
```

---

## Recommended Next Steps

### Immediate (High Priority)

1. **Accept the model limitation** for now:
   * SegFormer without fine-tuning won't reliably detect pool water
   * Keep EfficientSAM canary-only
   * Document this as a known limitation

2. **Add heuristic water detection** (Materials V3 PR-4):
   * If scene is exterior + large "foliage" region in lower-center + blue/teal color → reclassify as "water_candidate"
   * This gets you refinement targets without model changes

### Medium-Term (Next Sprint)

3. **Fine-tune or swap segmenter**:
   * Option A: Fine-tune SegFormer on luxury pool imagery
   * Option B: Use a segmenter with explicit water classes (Mask2Former, OneFormer)
   * Option C: Add a water-specific detector (SAM + CLIP "pool water" prompt)

4. **Auto-preset v2** improvements:
   * Use color histogram + scene CLIP to detect pool scenes
   * Automatically apply water heuristic when pool detected
   * Never auto-select canary presets (already planned)

---

## Files Modified

* ✅ `lux_depth_v2/materials_v3.py`: Added `_audit_class_presence()` + integrated into report
* ✅ `scripts/diagnose_pool_water.py`: Diagnostic tool (reproducible diagnostic)

---

## Validation

Diagnostic run confirms:

* ✅ Audit correctly identifies missing "water"
* ✅ Reason correctly reported as "not_emitted_by_segmenter"
* ✅ All emitted classes logged
* ✅ Canonical mapping working correctly
* ✅ No taxonomy bugs (foliage/glass/wood all map correctly)

---

## Decision

**Keep EfficientSAM FUSED canary-only** (Stage 6 conclusion remains valid).

The missing water issue is a **segmenter vocabulary limitation**, not something EfficientSAM fusion can fix. Proceed with:

1. **Auto-preset v2** (quality-tier auto, complexity heuristic, --allow-canary gate)
2. **Materials V3 PR-4**: heuristic water detection + edge-aware gating improvements
3. Defer segmenter swap/fine-tuning until client demand justifies the effort

---

## Session Artifacts

* `scripts/diagnose_pool_water.py` – reproducible diagnostic
* Diagnostic output: confirmed SegFormer emits no water class
* Materials V3 audit integrated into pipeline report

---

**Session Status**: ✅ Complete  
**Next PR**: Auto-preset v2 (--quality-tier auto + --intent + complexity + canary gate)

