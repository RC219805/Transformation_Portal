# Phase C Quick Reference

**Status:** ✅ **APPROVED** (C1+C2) | 🛑 **DEFERRED** (C3)
**Date:** 2026-02-17
**Authority:** Transformation Portal Architect
**Full Decision:** `docs/architecture/PHASE_C_ARCHITECTURAL_DECISION.md`

---

## TL;DR

**Original concern:** Phase C might introduce cross-pipeline `SAM2Session` contracts and require video architecture.

**Reality:**
- C1 already done (Phase A.6)
- C2 is internal to SAM2 backend (no new contracts)
- C3 needs video architecture (doesn't exist yet)

**Decision:** ✅ Proceed with C2, skip C3 for now.

---

## What You Can Do Now

### ✅ C2: Confidence Semantics (1 week)

**Task:** Make SAM2 return real IoU/stability scores instead of placeholders.

**Changes:**
1. Extract `model_output.iou_predictions` and `model_output.stability_scores`
2. Populate `MaskMetadata.stability_score` (field exists, just has 1.0 placeholder)
3. Use IoU for `SegmentationResult.scores` (field exists, just has 1.0 placeholder)

**Files to modify:**
- `src/transformation_portal/spatial_ai/segmentation/sam2_backend.py` (~50 lines)
- `tests/spatial_ai/test_sam2_confidence_semantics.py` (new, ~100 lines)

**No contract changes required** — fields already exist, just populating them properly.

**Effort:** 4 hours implementation + 3 hours testing = ~1 week with review

**Risk:** 🟢 LOW (internal improvement, backward compatible)

---

## What You Cannot Do Yet

### 🛑 C3: SAM2Long (Deferred)

**Blocker:** No video pipeline architecture exists.

**Why:**
- Materials V3 is image-only
- No video workflow orchestration
- No ADR for video processing
- Research-tier complexity (40+ hours)

**When to revisit:** After creating `ADR-0XX: Video Pipeline Architecture`

---

## Implementation Plan (C2)

### Day 1-2: Code (4 hours)
```python
# sam2_backend.py
def _extract_sam2_predictions(self, model_output):
    """Extract masks, IoU, and stability from SAM2 output."""
    masks = model_output.pred_masks
    iou_scores = model_output.iou_predictions  # Real values, not 1.0
    stability_scores = model_output.stability_scores  # Real values, not 1.0
    return masks, iou_scores, stability_scores
```

Update `_segment_auto()` and `_segment_prompted()` to use this helper.

### Day 3: Test (3 hours)
- `test_sam2_iou_extraction()` — mock SAM2 output
- `test_sam2_stability_extraction()` — mock SAM2 output
- `test_metadata_stability_populated()` — integration test
- `test_stub_backend_unchanged()` — regression test

### Day 4: Docs (1 hour)
- Update `sam2_backend.py` docstrings
- Add score semantics to `SegmentationResult`
- Update roadmap status

### Day 5: Review + Merge

---

## Governance Compliance

| Criterion | C2 Status | Notes |
|-----------|-----------|-------|
| New Dependencies | ✅ NONE | Uses existing SAM2 |
| CI/CD Changes | ✅ NONE | Standard testing |
| Security Impact | ✅ NONE | Internal improvement |
| Contract Changes | ✅ NONE | Populating existing fields |
| ADR Conflicts | ✅ NONE | Aligns with ADR-027 |
| Escalation Required | ✅ NO | Internal backend change |

**Verdict:** No escalation needed — proceed with implementation.

---

## What Changed Since Escalation Flag?

### Original Escalation Concerns (MATERIALS_V3_ROADMAP_COMPREHENSIVE_REVIEW.md:437-447)

1. **"Cross-pipeline contracts (`SAM2Session` API)"**
   → ✅ **Resolved:** C1+C2 don't introduce `SAM2Session`. C3 deferred.

2. **"Long-running state (memory bank across frames)"**
   → ✅ **Resolved:** C1 cleanup already done (Phase A.6). C3 deferred.

3. **"ADR uncertainty (video architecture undefined)"**
   → ✅ **Resolved:** C3 deferred until video architecture exists.

### Why Landscape Changed

1. **Phase A.6 implemented memory cleanup** (commit `d8004b35`)
   - `_cleanup_inference_state()` exists and is production-ready
   - Device-agnostic (CUDA + MPS)
   - Defensive exception handling

2. **C2 doesn't introduce new contracts**
   - `MaskMetadata.stability_score` already exists (just has 1.0 placeholder)
   - `SegmentationResult.scores` already exists (just has 1.0 placeholder)
   - No `SAM2Session` API needed for this

3. **No video pipeline exists yet**
   - lux_depth_v3 is image-only
   - No `video_pipeline/` directory
   - No video ADRs
   - C3 is premature

---

## Success Criteria (C2)

After implementation:

✅ `MaskMetadata.stability_score` has real SAM2 values (not 1.0)
✅ `SegmentationResult.scores` has real SAM2 IoU (not 1.0)
✅ Tests pass with real SAM2 models and stubs
✅ Stub backend unchanged (backward compatible)
✅ No performance regression
✅ Documentation updated

---

## Risks and Mitigations (C2)

### Risk 1: SAM2 Output Schema Mismatch
**Scenario:** `model_output` missing `iou_predictions` or `stability_scores`
**Mitigation:** Defensive attribute checking, fallback to 1.0
**Likelihood:** Low (SAM2 API stable)

### Risk 2: Behavior Change Impact
**Scenario:** Materials V3 thresholds assume scores=1.0
**Mitigation:** Canary testing, document in CHANGELOG
**Likelihood:** Medium (expected behavior change)

### Risk 3: Numerical Instability
**Scenario:** Scores outside [0, 1]
**Mitigation:** Contract validation in `__post_init__()` (already exists)
**Likelihood:** Very Low (SAM2 guarantees valid ranges)

---

## Questions and Answers

**Q: Why defer C3 if video mode exists in contracts?**
A: Video mode is scaffolded for future use, but no video pipeline architecture exists yet. `_segment_video()` is a `NotImplementedError` stub. We need ADR-0XX to define where video processing lives before implementing tracking.

**Q: Does C2 break backward compatibility?**
A: No. API unchanged (same fields, same types). Behavior improves (better scores). Only risk is hardcoded thresholds assuming 1.0 → mitigate with canary testing.

**Q: Can I implement C3 now anyway?**
A: No. Architect decision is binding. C3 is blocked on video architecture ADR. Implementing it now would violate governance (unapproved cross-pipeline contracts).

**Q: What if I find a bug in C1 (memory cleanup)?**
A: C1 is already in production (Phase A.6). File a bug report, fix separately from Phase C.

**Q: Do I need Architect approval for C2 implementation details?**
A: No. Decision is made — proceed with implementation. Architect delegates execution to Specialist.

---

## Recommended Next Actions

### This Week
1. ✅ Implement C2 (Specialist)
2. ✅ Test with real SAM2 models
3. ✅ Update roadmap docs

### Future (When Video is Scoped)
1. 📋 Create ADR-0XX: Video Pipeline Architecture
2. 📋 Validate video use cases
3. 📋 Re-evaluate C3 (SAM2Long)

---

## References

- **Full Decision:** `docs/architecture/PHASE_C_ARCHITECTURAL_DECISION.md`
- **Roadmap Review:** `docs/materials/MATERIALS_V3_ROADMAP_COMPREHENSIVE_REVIEW.md`
- **Phase A Complete:** `docs/materials/PHASE_A_COMPLETE.md` (includes A.6 memory cleanup)
- **Contracts:** `src/transformation_portal/spatial_ai/segmentation/contracts.py`
- **SAM2 Backend:** `src/transformation_portal/spatial_ai/segmentation/sam2_backend.py`
- **Governance:** `docs/architecture/agent_governance.md`

---

**Binding Status:** ✅ **APPROVED** — Proceed with C2 implementation without further escalation.
