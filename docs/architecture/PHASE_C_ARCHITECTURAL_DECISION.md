# Phase C Architectural Decision: SAM2 Tracking Enhancements

**Date:** 2026-02-17
**Decision Authority:** Transformation Portal Architect
**Status:** ✅ **OPTION 2: SIMPLIFIED SCOPE - PROCEED WITH C1+C2, DEFER C3**
**Related:** Materials V3 Roadmap, ADR-030, Phase A Complete, Phase B Complete

---

## Executive Summary

**DECISION:** Phase C items C1 (Memory Protocol) and C2 (Confidence Semantics) are **APPROVED for immediate implementation** as internal SAM2 backend improvements. C3 (SAM2Long) is **DEFERRED** pending video architecture definition.

**Rationale:** The original escalation concern was based on anticipated cross-pipeline contracts (`SAM2Session` API) and video tracking architecture. However, repository analysis reveals:

1. **C1 is already implemented** (Phase A.6 commit `d8004b35`)
2. **C2 is an internal backend enhancement** (no new contracts)
3. **C3 requires video pipeline architecture** (doesn't exist yet)
4. **No cross-pipeline coupling risk** (lux_depth_v3 doesn't consume video APIs)

The landscape has **fundamentally changed** since the escalation flag was raised. What remains of Phase C are localized improvements that don't require architectural approval.

---

## Decision Breakdown

### ✅ C1: Memory Protocol — ALREADY IMPLEMENTED

**Status:** **COMPLETE** (Phase A.6, commit `d8004b35`)

**Evidence:**
```python
# src/transformation_portal/spatial_ai/segmentation/sam2_backend.py:264-301
def _cleanup_inference_state(self, inference_state: object) -> None:
    """Clean up SAM2 inference state to prevent memory leaks."""
    if inference_state is None:
        return

    try:
        import gc
        import torch

        # SAM2 video predictor exposes reset_state()
        if hasattr(inference_state, "reset_state"):
            inference_state.reset_state()

        # Delete reference
        del inference_state

        # Force garbage collection
        gc.collect()

        # Empty device cache (device-specific)
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception as e:
        logger.warning(f"Error during SAM2 inference state cleanup: {e}")
```

**Assessment:**
- ✅ **reset_state() pattern**: Implemented
- ✅ **Device-agnostic sync**: CUDA + MPS supported (fixes roadmap concern)
- ✅ **GC + empty_cache**: Proper cleanup sequence
- ✅ **Defensive exception handling**: Never crashes cleanup
- ✅ **Already documented in Phase A**: TODOs reference this pattern

**Action Required:** **NONE** — Already production-ready.

---

### ✅ C2: Confidence Semantics — APPROVE FOR IMPLEMENTATION

**Status:** **APPROVED** (Internal backend enhancement, no contract changes)

**Scope:**
1. Decouple `mean_conf` (coverage ratio) from IoU (model confidence)
2. Return tuple from SAM2: `(binary_mask, iou_score, stability_score)`
3. Use SAM2's native `predicted_iou` predictions
4. Add `stability_score` to `MaskMetadata` (already exists in contract)

**Current Contract (Already Supports This):**
```python
# src/transformation_portal/spatial_ai/segmentation/contracts.py:89-103
@dataclass
class MaskMetadata:
    """Per-mask metadata.

    Attributes:
        area: Pixel count of mask.
        bbox: Bounding box (x, y, w, h) in image coordinates.
        stability_score: Mask stability score [0, 1] (higher = more stable).  # ← ALREADY EXISTS
        material_label: Optional material classification (e.g., "wood", "marble").
        material_confidence: Optional material classification confidence [0, 1].
    """
    area: int
    bbox: Tuple[int, int, int, int]
    stability_score: float  # ← Field exists, just needs population
    material_label: Optional[str] = None
    material_confidence: Optional[float] = None
```

**Implementation Plan:**

#### C2.1: Extract IoU and Stability from SAM2 Output
```python
# sam2_backend.py (in _segment_auto or _segment_prompted)
# Current (placeholder):
#   masks = model_output.pred_masks
#   scores = np.ones(len(masks))  # Stub

# Proposed:
def _extract_sam2_predictions(self, model_output):
    """Extract masks, IoU, and stability from SAM2 output."""
    masks = model_output.pred_masks  # (N, H, W) bool
    iou_scores = model_output.iou_predictions  # (N,) float32 [0, 1]
    stability_scores = model_output.stability_scores  # (N,) float32 [0, 1]

    return masks, iou_scores, stability_scores
```

#### C2.2: Populate MaskMetadata.stability_score
```python
# Current (Phase A):
metadata = MaskMetadata(
    area=int(mask.sum()),
    bbox=bbox,
    stability_score=1.0,  # Placeholder
)

# Proposed (Phase C2):
metadata = MaskMetadata(
    area=int(mask.sum()),
    bbox=bbox,
    stability_score=float(stability_scores[i]),  # Native SAM2 confidence
)
```

#### C2.3: Use IoU for SegmentationResult.scores
```python
# Current:
return SegmentationResult(
    masks=masks,
    scores=np.ones(len(masks)),  # Placeholder
    metadata=metadata_list,
)

# Proposed:
return SegmentationResult(
    masks=masks,
    scores=iou_scores,  # SAM2 native IoU predictions
    metadata=metadata_list,
)
```

**Contract Impact:** **NONE**
- `SegmentationResult.scores` already exists (just changing values)
- `MaskMetadata.stability_score` already exists (just populating)
- No new fields, no schema changes
- 100% backward compatible

**Coupling Analysis:** **NONE**
- Changes internal to `sam2_backend.py` only
- `lux_depth_v3/materials_v3.py` consumes `SegmentationResult` contract (unchanged)
- No new APIs, no session state

**Risk Assessment:** 🟢 **LOW**
- Isolated to SAM2 backend implementation
- Contract already designed for this data
- No performance regression (removing placeholder logic)

**Effort:** 3-4 hours
- Extract model outputs (1 hour)
- Update metadata population (1 hour)
- Add tests for IoU/stability ranges (1-2 hours)

**Action Required:** **PROCEED** with implementation by Specialist.

---

### 🛑 C3: SAM2Long Integration — DEFER PENDING VIDEO ARCHITECTURE

**Status:** **REJECTED FOR NOW** (No video pipeline architecture exists)

**Why Defer:**

1. **No Video Pipeline Architecture**
   - `src/transformation_portal/video_pipeline/` doesn't exist
   - No ADRs for video processing
   - No video workflow orchestration
   - lux_depth_v3 is **image-only** (no frame sequence handling)

2. **SAM2Long Not Needed for Current Use Cases**
   - Materials V3 is for **single-image enhancement** (luxury real estate photos)
   - Video tracking is **out of scope** for current roadmap
   - Roadmap review notes: "Luxury real estate videos are typically short (<1 min)" → unclear ROI

3. **Research-Tier Complexity**
   - Tree memory structure (40+ hours implementation)
   - Multi-hypothesis propagation
   - Occlusion handling
   - Not in SAM2 core library (external research)

4. **Video Mode Already Scaffolded**
   - `contracts.py` has `mode="video"` and `temporal_ids`
   - `sam2_backend.py` has `_segment_video()` stub with cleanup pattern
   - When video pipeline is designed, implementation path is clear

**Architectural Questions Requiring ADR:**
- Where do video frames live? (new pipeline? lux_depth_v3 extension?)
- How is frame state managed? (stateless per-frame? session object?)
- What are video output contracts? (frame sequences? video files?)
- How does video interact with existing image pipelines?

**Action Required:**
1. **Create ADR-0XX: Video Pipeline Architecture** when video use cases are validated
2. **Re-evaluate C3** after video architecture is defined
3. **Defer indefinitely** if video is out of scope

---

## Governance Trigger Analysis

### Section A: New Dependencies ✅ PASS
- ❓ Does Phase C add new ML dependencies? → **NO**
- ❓ Are there license implications? → **NO**

**Rationale:** C1 complete, C2 uses existing SAM2 integration, C3 deferred.

---

### Section B: CI/CD Changes ✅ PASS
- ❓ Does Phase C require CI/CD changes? → **NO**
- ❓ New model fixtures or GPU testing? → **NO** (uses existing SAM2 fixtures)

**Rationale:** C2 tests use existing SAM2 backend test infrastructure.

---

### Section C: Performance Impact ✅ PASS
- ❓ Does Phase C regress performance beyond tolerance? → **NO**
- ❓ Memory overhead acceptable? → **YES** (C1 improves memory cleanup)

**Rationale:** C1 prevents memory leaks (performance improvement). C2 removes placeholder logic (neutral/slight improvement).

---

### Section D: Cross-Pipeline Contracts ✅ PASS
- ❓ Does Phase C introduce new APIs consumed by other pipelines? → **NO**
- ❓ Are video tracking contracts stable? → **N/A** (C3 deferred)

**Rationale:**
- **Original escalation concern was `SAM2Session` API** → Not introduced in C1/C2
- `SegmentationResult` contract unchanged (just populating existing fields)
- lux_depth_v3 doesn't consume video APIs (image-only)
- C3 (video tracking) deferred pending architecture

---

### Section E: ADR Requirements ✅ PASS
- ❓ Does Phase C require new architectural decisions? → **NO** (for C1+C2)
- ❓ Is video architecture defined in ADRs? → **NO** (C3 deferred for this reason)

**Rationale:** C1/C2 are implementation details within existing SAM2 backend. C3 requires ADR-0XX (video architecture) first.

---

### Section F: Security/Privacy ✅ PASS
- ❓ Any security implications? → **NO**
- ❓ Licensing or compliance issues? → **NO**

**Rationale:** Internal improvements only, no new dependencies or data handling.

---

## Risk Assessment

### C1: Memory Protocol
**Status:** ✅ **COMPLETE** (Phase A.6)
**Risk:** 🟢 **NONE** (already in production)

### C2: Confidence Semantics
**Status:** ✅ **APPROVED**
**Risk:** 🟢 **LOW**

**What Could Go Wrong:**

1. **SAM2 Output Schema Mismatch**
   - **Scenario:** `model_output` doesn't have `iou_predictions` or `stability_scores`
   - **Mitigation:** Defensive attribute checking with fallback to 1.0
   - **Likelihood:** Low (SAM2 API is stable)

2. **Numerical Instability**
   - **Scenario:** IoU/stability scores outside [0, 1] range
   - **Mitigation:** Contract validation in `MaskMetadata.__post_init__()` (already exists)
   - **Likelihood:** Very Low (SAM2 guarantees valid ranges)

3. **Behavior Change Impact**
   - **Scenario:** Materials V3 threshold logic behaves differently with real scores vs placeholders
   - **Mitigation:** Canary testing with actual SAM2 outputs before production rollout
   - **Likelihood:** Medium (Expected - this is the point of the change)

**Mitigation Plan:**
- Add defensive attribute checking
- Test with real SAM2 models (not just stubs)
- Document score semantics in `SegmentationResult` docstring
- Add integration test comparing stub vs real backend behavior

### C3: SAM2Long
**Status:** 🛑 **DEFERRED**
**Risk:** 🔴 **BLOCKED** (no video architecture)

---

## Implementation Plan (C2 Only)

### Timeline: 1 Week

#### Day 1-2: Implementation (4 hours)
1. Add `_extract_sam2_predictions()` helper
2. Update `_segment_auto()` to use real IoU/stability
3. Update `_segment_prompted()` to use real IoU/stability
4. Add defensive fallbacks for missing attributes

#### Day 3: Testing (3 hours)
1. Unit test: `test_sam2_iou_extraction()` (mock SAM2 output)
2. Unit test: `test_sam2_stability_extraction()` (mock SAM2 output)
3. Integration test: `test_metadata_stability_populated()` (real SAM2 if available)
4. Regression test: Ensure stub backend still works (scores=1.0 fallback)

#### Day 4: Documentation (1 hour)
1. Update `sam2_backend.py` docstrings
2. Add `SegmentationResult.scores` semantic documentation (IoU vs coverage)
3. Update Phase C status in roadmap

#### Day 5: Review + Merge
1. Code review
2. CI validation
3. Merge to main

### Success Criteria
- ✅ `MaskMetadata.stability_score` populated with real SAM2 values (not 1.0)
- ✅ `SegmentationResult.scores` uses SAM2 IoU predictions
- ✅ Tests pass with real SAM2 models (if GPU available) and stubs
- ✅ Backward compatibility: stub backend unchanged
- ✅ No performance regression

---

## Scope Definition

### ✅ APPROVED FOR IMMEDIATE IMPLEMENTATION

**C1: Memory Protocol**
- Status: **COMPLETE** (Phase A.6)
- No action required

**C2: Confidence Semantics**
- Extract SAM2 `iou_predictions` and `stability_scores`
- Populate `MaskMetadata.stability_score` (field exists)
- Use IoU for `SegmentationResult.scores` (field exists)
- Add defensive fallbacks for attribute errors
- Test with real SAM2 outputs

### 🛑 DEFERRED PENDING ARCHITECTURE REVIEW

**C3: SAM2Long Integration**
- Tree memory structure
- Multi-hypothesis propagation
- Occlusion handling
- **Blocked on:** ADR-0XX (Video Pipeline Architecture)
- **Re-evaluate when:** Video use cases validated and architecture defined

---

## Dependencies and Blockers

### C2 Dependencies: ✅ ALL CLEAR
- ✅ SAM2 backend exists (`sam2_backend.py`)
- ✅ Contracts support stability_score (`contracts.py:93`)
- ✅ Phase A memory cleanup complete (cleanup pattern ready)
- ✅ Zero new dependencies required

### C3 Blockers: 🛑 ARCHITECTURAL
- ❌ No video pipeline architecture
- ❌ No video use case validation
- ❌ No ADR for video processing
- ❌ lux_depth_v3 is image-only (no frame sequences)

---

## Migration and Backward Compatibility

### C2 Migration: **ZERO-IMPACT**

**Existing Code:**
```python
# Consumers of SegmentationResult
result = backend.segment(image, gamma=1.0, mode="auto")
for mask, score, metadata in zip(result.masks, result.scores, result.metadata):
    if score > threshold:  # Currently always 1.0 from placeholders
        apply_enhancement(mask, metadata.stability_score)  # Currently 1.0
```

**After C2:**
```python
# EXACT SAME INTERFACE - just better values
result = backend.segment(image, gamma=1.0, mode="auto")
for mask, score, metadata in zip(result.masks, result.scores, result.metadata):
    if score > threshold:  # Now real IoU from SAM2 (0.x - 1.0)
        apply_enhancement(mask, metadata.stability_score)  # Now real stability (0.x - 1.0)
```

**Impact:**
- ✅ **API unchanged** (same fields, same types)
- ✅ **Behavior improvement** (better mask filtering with real scores)
- ⚠️ **Threshold tuning may be needed** (if hardcoded to assume 1.0)

**Mitigation:**
- Document in CHANGELOG: "SAM2 now returns real confidence scores (breaking if thresholds hardcoded)"
- Recommend config-driven thresholds (already best practice)
- Canary testing before production

---

## Enforcement and Validation

### C2 Enforcement Mechanisms

**1. Contract Validation (Existing)**
```python
# contracts.py:106-109
def __post_init__(self):
    # Stability score in [0, 1]
    if not 0.0 <= self.stability_score <= 1.0:
        raise ValueError(f"Stability score must be in [0, 1], got {self.stability_score}")
```
**Status:** ✅ Already enforced by contract

**2. CI Testing**
- Add test: `test_sam2_real_scores_within_bounds()` (ensure [0,1] range)
- Add test: `test_sam2_stability_not_placeholder()` (ensure not always 1.0)
- Existing: Contract validation in `__post_init__()` fails CI if violated

**3. Documentation**
- Update `SegmentationResult` docstring to clarify score semantics
- Add example showing IoU-based filtering in Materials V3 docs

---

## Conclusion

### BINDING DECISION: ✅ OPTION 2 - SIMPLIFIED SCOPE

**Phase C Breakdown:**
- ✅ **C1 (Memory Protocol):** COMPLETE (Phase A.6) — No action required
- ✅ **C2 (Confidence Semantics):** APPROVED — Proceed with implementation (1 week)
- 🛑 **C3 (SAM2Long):** DEFERRED — Blocked on video architecture ADR

**Overall Timeline:**
- **C1:** 0 days (already done)
- **C2:** 5 days (1 week)
- **C3:** TBD (pending ADR-0XX: Video Pipeline Architecture)

**Governance Status:** ✅ **COMPLIANT** (no escalation required for C1+C2)

**Risk Level:** 🟢 **LOW** (internal improvements, no new contracts, backward compatible)

---

## Next Steps

### Immediate (This Week)
1. ✅ **Architect:** Approve this decision document
2. 🔨 **Specialist:** Implement C2 (confidence semantics)
3. 📝 **Specialist:** Update roadmap status

### Future (When Video is Scoped)
1. 📋 **Create ADR-0XX:** Video Pipeline Architecture
2. 📋 **Validate use case:** Is video processing needed for luxury real estate?
3. 📋 **Re-evaluate C3:** SAM2Long integration if video architecture approved

---

## Decision Authority

**Architect Signature:** Transformation Portal Architect
**Date:** 2026-02-17
**Binding Status:** ✅ **APPROVED** (C1 complete, C2 approved, C3 deferred)

This decision is **final and binding** under `docs/architecture/agent_governance.md`. The Specialist may proceed with C2 implementation without further escalation. C3 remains blocked pending video architecture definition.
