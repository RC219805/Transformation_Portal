# Materials V3 Roadmap — Production Implementation Plan

**Date:** 2026-02-13
**Status:** Approved for Execution
**Specialist:** Transformation Portal Specialist
**Risk Assessment:** Phased, merge-safe approach with clear escalation criteria

---

## Executive Summary

This document provides a **production-ready, governance-compliant implementation plan** for the Materials V3 enhancement roadmap. It breaks the comprehensive roadmap into merge-safe milestones, identifies quick wins, assesses risks, and ensures compliance with APEX governance and Quality Firewall requirements.

**Key Principle:** Ruthless prioritization for maximum ROI with minimum risk, respecting existing architecture and contracts.

---

## Current State Assessment

### ✅ Existing Production Capabilities

**Core Architecture:**
- **Materials V3 Engine:** Production-ready with 8 materials (glass, water, foliage, wood, stone, metal, fabric, stucco)
- **Pixel Ops:** 5 operations (brightness_boost, edge_contrast, microcontrast, reflection_enhance, vibrance_boost)
- **Segmentation Backends:** 3 backends (stub, efficientsam, sam2)
- **V2 Integration:** Full mask serialization and subprocess wiring (ADR-030)
- **SAM2 Integration:** Adapter pattern with HuggingFace pipeline (February 2026)
- **Testing:** 52 tests passing, zero regressions
- **Performance:** <100ms pixel ops overhead, 1-5s segmentation depending on backend

**Integration Points:**
- `pixel_ops_executor.py` - ROI-based ops with uint16 support
- `pixel_ops_registry.py` - Material→Op registry
- `materials_v3_taxonomy.py` - Priority/threshold metadata
- `segmentation_backend.py` - Backend abstraction layer
- `sam2_adapter.py` - SAM2 integration via spatial_ai

**Contract Stability:**
- ✅ Golden Path contract: stable (ADR-030)
- ✅ V2 subprocess: masks passed via NPZ serialization
- ✅ Manifest integration: materials_v3_metadata included
- ✅ Backward compatibility: 100% maintained

---

## Quick Win Summary

### ⚡ Can Ship This Week
1. **A1: Fix 3D Mask Bug** (2 hours) — Critical crash fix
2. **A4: Eliminate Redundant Normalization** (2 hours) — Performance improvement
3. **A3: Configurable Feathering** (3 hours) — Quality tuning

### 🚀 Can Ship in 1-2 Weeks
4. **A2: Fix Feathering Edge Clipping** (4 hours) — Quality improvement
5. **A5: Overlap Resolution** (4 hours) — Correctness improvement
6. **Phase B: Sky Material** (16 hours total) — High user value, zero new dependencies

### ⏸️ Requires Escalation (2-4 weeks planning)
7. **Phase C: Video Tracking** — Architectural review needed
8. **Phase D: Detector Integration** — Dependency governance
9. **Phase E: Material Synthesis** — Research initiative

**Recommended Focus:** Phases A + B (31 hours, 4 weeks, zero escalation needed)

---

## Phase A: Harden Pixel Ops Executor ⚡

**Objective:** Fix bugs and add critical enhancements
**Risk:** 🟢 LOW — Isolated changes, full test coverage
**Timeline:** 2 weeks (15 hours total)
**Dependencies:** None (uses existing numpy, scipy)

### A1: Fix `_bounding_box()` 3D Mask Bug 🐛

**Problem:** Crash when backends return `(H,W,1)` or `(H,W,3)` masks
**Root Cause:** `np.where()` expects 2D input

**Solution:**
```python
def _canonical_mask(mask: np.ndarray) -> np.ndarray:
    """Ensure mask is (H,W) float32."""
    if mask.ndim == 3:
        mask = mask.squeeze()
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D or 3D, got shape {mask.shape}")
    return mask.astype(np.float32)
```

**Files Modified:** `pixel_ops_executor.py` (~10 lines)
**Tests:** 3 new tests (3D mask handling, backward compat, property test)
**Effort:** 2 hours

---

### A2: Fix Mask Feathering Edge Clipping 🐛

**Problem:** Gaussian blur clips at bbox edges → visible halos
**Impact:** Artifacts in high-contrast scenes (glass against sky)

**Solution:** Expand bbox by 3σ before blurring
```python
def _compute_feathered_roi(mask, bbox, sigma=2.0):
    """Expand bbox, feather, return expanded region."""
    pad = int(np.ceil(3 * sigma))
    # Expand bbox with clipping to image boundaries
    # Apply gaussian_filter to expanded region
    return feathered_roi, expanded_bbox
```

**Files Modified:** `pixel_ops_executor.py` (~30 lines)
**Tests:** 2 new tests (edge clipping, performance)
**Effort:** 4 hours
**Performance:** +<5ms (Gaussian on slightly larger ROI)

---

### A3: Configurable Feathering Per Material ✨

**Rationale:** Sky needs large sigma (5.0), glass needs small (1.5)

**Config Extension:**
```python
# config.py
mask_feather_sigma_default: float = 2.0
mask_feather_sigma_overrides: Dict[str, float] = {
    "sky": 5.0,
    "glass": 1.5,
    "water": 3.0,
}
```

**Files Modified:** `config.py` (~5 lines), `pixel_ops_executor.py` (~10 lines)
**Tests:** 2 new tests
**Effort:** 3 hours
**Risk:** NONE (additive config with safe defaults)

---

### A4: Eliminate Redundant Normalization 🔧

**Problem:** Executor normalizes at line 112-114, but ops have fallback normalization
**Impact:** Confusing code path, potential double-processing

**Solution:** Remove fallback `_normalize_image()` from all ops, enforce contract

**Files Modified:** `pixel_ops_registry.py` (~20 lines)
**Tests:** 1 test (verify params passed correctly)
**Effort:** 2 hours
**Performance:** **-2ms** (eliminates redundant checks)

---

### A5: Overlap Resolution (Priority-Based) 🔧

**Problem:** Overlapping masks cause double-processing (e.g., sky+water both enhanced)

**Solution:**
```python
def _resolve_material_overlap(materials, priority_map):
    """Higher priority material wins at each pixel."""
    # Sort by priority: sky(15) > glass(10) > water(9) > ...
    # Iteratively mask out occupied pixels
    return non_overlapping_masks
```

**Files Modified:** `pixel_ops_executor.py` (~40 lines)
**Tests:** 3 tests (priority order, conservation, edge cases)
**Effort:** 4 hours
**Performance:** +<10ms (single pass over masks)

---

**Phase A Summary:**
- **Total Effort:** 15 hours (~2 days)
- **Risk:** 🟢 LOW
- **New Tests:** 11 tests
- **Performance:** Net ~0ms (A4 saves 2ms, A2+A5 add ~15ms)
- **Backward Compat:** 100% maintained
- **Escalation:** NONE required

---

## Phase B: Sky as First-Class Material 🌤️

**Objective:** Add sky detection + pixel ops (top user request)
**Risk:** 🟡 MEDIUM — New material, new bootstrap module
**Timeline:** 2 weeks (16 hours total)
**Dependencies:** None (uses existing scipy.ndimage)

### B1: Extend Material Taxonomy (1 hour)

```python
# materials_v3_taxonomy.py
"sky": {"priority": 15, "threshold": 0.30, "canary": True}  # Highest priority
```

---

### B2: Sky Bootstrap Heuristic (6 hours)

**Strategy:** Geometric + color priors (no ML dependencies)

**Heuristic Components:**
1. **Top-of-frame prior:** Exponential decay from top 40%
2. **Low gradient:** Sky is smooth (sobel filter)
3. **High brightness:** Sky is typically bright
4. **Depth prior:** Sky is far (if depth available)

**Output:** Confidence mask + bbox/point proposals for SAM2 refinement

**New File:** `sky_bootstrap.py` (~150 lines)
**Tests:** 4 tests (outdoor detection, indoor rejection, depth integration, output format)

---

### B3: Sky Pixel Operations (4 hours)

**Three Operations:**
1. **sky_dehaze:** Reduce atmospheric haze (contrast + brightness)
2. **sky_gradient_smooth:** Smooth color transitions (Gaussian per channel)
3. **sky_temperature_shift:** Warmer/cooler sky (R/B channel adjustment)

**Files Modified:** `pixel_ops_registry.py` (+~80 lines)
**Tests:** 3 tests (one per operation)

---

### B4: Integration (3 hours)

**Heuristic-only mode (Phase 1):**
```python
# segmentation_backend.py
if config.enable_sky_bootstrap:
    from .sky_bootstrap import bootstrap_sky_heuristic
    sky_mask = bootstrap_sky_heuristic(image, depth_map)["confidence_mask"]
    materials["sky"] = (sky_mask > threshold).astype(np.float32)
```

**SAM2 refinement (Future Phase 2):** Use bbox/point proposals

**Files Modified:** `segmentation_backend.py` (~15 lines)
**Tests:** 1 integration test

---

### B5: Configuration + Validation (2 hours)

```python
# config.py
enable_sky_bootstrap: bool = False  # Opt-in
sky_confidence_threshold: float = 0.6
```

**New Preset:** `config/materials_v3_sky.yaml`
**Tests:** 2 tests (opt-in verification, preset loading)

---

**Phase B Summary:**
- **Total Effort:** 16 hours (~2 days)
- **Risk:** 🟡 MEDIUM (new module, but isolated)
- **New Tests:** 8 tests
- **Performance:** +10-20ms (heuristic bootstrap)
- **Backward Compat:** 100% (opt-in via config)
- **Escalation:** NONE (no new dependencies, no contract changes)

---

## Phase C: Video Tracking ⚠️ ESCALATION REQUIRED

**Objective:** SAM2 video tracking for temporal stability
**Risk:** 🔴 HIGH — Cross-pipeline contracts, session state

**Escalation Triggers:**
- New cross-pipeline contract: `SAM2Session` API
- State management across frames (memory bank)
- Potential coupling between `lux_depth_v3` and `spatial_ai`

**Required Actions:**
1. Create escalation packet (see Appendix A)
2. Architect reviews session API design
3. ADR for video tracking architecture
4. Performance budget for video workflows

**Deferred Until:** Architect approval

---

## Phase D: Detector Integration ⚠️ ESCALATION REQUIRED

**Objective:** YOLO/RT-DETR for "things" detection
**Risk:** 🔴 HIGH — New ML dependencies, licensing

**Escalation Triggers:**
- New dependency: YOLOv8 (AGPL-3.0) or RT-DETR (Apache-2.0)
- Dependency governance approval needed
- Model size: 50-100MB

**Required Actions:**
1. Research detector options (YOLO vs RT-DETR vs SAM2-only)
2. License analysis
3. Architect reviews dependency proposal
4. Performance impact assessment

**Deferred Until:** Dependency governance approval

---

## Phase E: Material Synthesis ⚠️ RESEARCH

**Objective:** Advanced synthesis (water ripples, sky replacement)
**Risk:** 🔴 VERY HIGH — Research-heavy, unclear ROI

**Recommendation:** Defer to research roadmap (out of scope for production V3)

---

## Risk Matrix

| Phase | Risk | Technical | Dependencies | Contracts | Performance | Escalation |
|-------|------|-----------|--------------|-----------|-------------|------------|
| **A** | 🟢 LOW | Isolated bugs | None | None | <15ms | ✅ None |
| **B** | 🟡 MED | New bootstrap | None | None | +10-20ms | ✅ None |
| **C** | 🔴 HIGH | Video state | SAM2 | Session API | +100-500ms | ⚠️ Required |
| **D** | 🔴 HIGH | Detector | YOLOv8/RT-DETR | Taxonomy | +50-200ms | ⚠️ Required |
| **E** | 🔴 VERY HIGH | Synthesis | Unclear | Output schema | TBD | ⚠️ Required |

---

## Dependency Assessment

### ✅ Phases A + B: Zero New Dependencies
- Uses: numpy, scipy, pillow (already in requirements.txt)
- License: BSD/MIT (commercial-friendly)
- Offline: 100% supported
- Security: No external downloads

### ⚠️ Phase C: Optional Dependency
| Dependency | License | Size | Risk | Decision |
|-----------|---------|------|------|----------|
| filterpy (Kalman) | MIT | ~2MB | LOW | Optional, defer |

### ⚠️ Phase D: High-Risk Dependencies
| Dependency | License | Size | Risk | Decision |
|-----------|---------|------|------|----------|
| ultralytics (YOLO) | **AGPL-3.0** | 100MB | **HIGH** | **ESCALATE** |
| RT-DETR (Baidu) | Apache-2.0 | 50MB | MEDIUM | **ESCALATE** |

**Recommendation:** Proceed with A+B (zero deps), escalate C/D separately

---

## Contract Impact

### ✅ Backward-Compatible Extensions (A + B)

**No Breaking Changes:**
- Pixel ops executor: internal only
- Taxonomy: additive (sky added to existing 8 materials)
- Config: optional fields with safe defaults
- Segmentation: new bootstrap module, existing backends untouched

**Golden Path Stability:**
- V2 subprocess: NPZ serialization unchanged
- Manifest: materials_v3_metadata schema unchanged
- Output naming: unchanged

### ⚠️ Potential Breaking Changes (C/D/E)

**Phase C:**
- New API: `SAM2Session.start()`, `.propagate()`, `.close()`
- **Impact:** spatial_ai contracts → **ESCALATE**

**Phase D:**
- New taxonomy classes (environmental/built materials)
- **Impact:** materials_v3_taxonomy schema → **ESCALATE**

**Phase E:**
- New output artifacts (synthesized materials)
- **Impact:** Output contracts → **ESCALATE**

---

## Quality Firewall

### Phase A: No Updates Needed
- Performance: <15ms total (<0.2% of 8.5s budget)
- Existing budgets cover regression tolerance

### Phase B: Shadow Mode Monitoring

**New Budget (Canary Tier):**
```yaml
workflow_version: "v2_materials_sky"
bucket_name: "exterior_with_sky_mps"
stability_tier: "canary"
thresholds:
  p50_sec: 8.6  # +100ms safety margin
  p95_sec: 12.1
enforcement:
  mode: "shadow"  # 2 weeks baseline collection
```

**Transition:** Shadow → Enforce after 30-day calibration

### Phases C/D/E: New Budgets Required
- Video tracking: Separate budget (video-specific)
- Detector: Detector-enabled workflows only
- Synthesis: TBD during research phase

---

## Testing Strategy

### Phase A Tests (11 total)
```python
# tests/materials/test_pixel_ops_phase_a.py
- test_canonical_mask_squeezes_3d()
- test_canonical_mask_backward_compat()
- test_feathered_roi_no_edge_clipping()
- test_feathered_roi_performance()
- test_configurable_feathering_per_material()
- test_configurable_feathering_fallback()
- test_no_double_normalization()
- test_overlap_resolution_priority_order()
- test_overlap_resolution_conservation()
- test_overlap_resolution_edge_cases()
- test_phase_a_backward_compatibility()
```

### Phase B Tests (8 total)
```python
# tests/materials/test_sky_material.py
- test_sky_bootstrap_outdoor()
- test_sky_bootstrap_indoor()
- test_sky_bootstrap_with_depth()
- test_sky_bootstrap_output_format()
- test_sky_dehaze_operation()
- test_sky_gradient_smooth_operation()
- test_sky_temperature_shift_operation()
- test_sky_integration_end_to_end()
```

### Regression Tests
- All 52 existing tests must pass
- No modifications to existing tests required

---

## Merge-Safe Milestones

### Milestone 1: Phase A Complete ✅

**Deliverables:**
- [ ] 5 PRs (A1-A5), one per item
- [ ] 11 new tests, all passing
- [ ] Documentation: inline comments + docstrings
- [ ] Performance: <15ms overhead verified
- [ ] Backward compat: 52 existing tests pass

**Success Criteria:**
- Zero regressions
- Linter clean (flake8, pylint)
- Performance budget met (<5% regression)

**Timeline:** 2 weeks

---

### Milestone 2: Phase B Complete ✅

**Deliverables:**
- [ ] 3 PRs: (B1+B2), (B3+B4), (B5)
- [ ] 8 new tests, all passing
- [ ] New file: `sky_bootstrap.py`
- [ ] New preset: `config/materials_v3_sky.yaml`
- [ ] Documentation: `SKY_MATERIAL_GUIDE.md`

**Success Criteria:**
- Sky detected in >80% of outdoor test images
- False positives <10% on indoor images
- Performance: +10-20ms measured
- Opt-in only (zero impact when disabled)
- All 71 tests pass (63 existing + 8 new)

**Timeline:** 2 weeks

---

### Milestone 3+: Escalation Required

**Phases C/D/E deferred** pending Architect review

---

## Architecture Integration

### Dependency Flow
```
orchestrator.py
  └─> materials_v3.py (engine)
      ├─> segmentation_backend.py
      │   ├─> sam2_adapter.py
      │   │   └─> spatial_ai/segmentation/sam2_backend.py
      │   └─> sky_bootstrap.py ◄─ NEW (Phase B)
      ├─> pixel_ops_executor.py ◄─ MODIFIED (Phase A)
      │   └─> pixel_ops_registry.py ◄─ EXTENDED (Phase A, B)
      └─> materials_v3_taxonomy.py ◄─ EXTENDED (Phase B)
```

**Coupling Analysis:**
- 🟢 Phase A: Zero new coupling
- 🟢 Phase B: Minimal (new optional module)
- 🔴 Phase C: High coupling risk → **ESCALATE**

---

## Performance Impact

### Phase A: Negligible
- A1: <1ms (squeeze)
- A2: <5ms (Gaussian on ROI)
- A3: 0ms (config lookup)
- A4: **-2ms** (removes redundant normalization)
- A5: <10ms (overlap resolution)
- **Total: ~12ms net** (<0.2% of 8.5s budget)

### Phase B: Minor
- Bootstrap: 10-20ms (heuristic, no ML)
- Pixel ops: 3-5ms (only if sky detected)
- **Total: 13-25ms** (<0.3% of budget)

### Phases C/D/E: TBD
- Phase C: +100-500ms (SAM2 video tracking)
- Phase D: +50-200ms (detector inference)
- Phase E: +500-2000ms (synthesis)

---

## Execution Sequence

### ✅ Week 1-2: Phase A
1. **Monday W1:** A1 (3D mask bug) — PR #1
2. **Tuesday W1:** A4 (redundant normalization) — PR #2
3. **Wednesday W1:** A3 (configurable feathering) — PR #3
4. **Thursday-Friday W1:** A2 (feathering edge fix) — PR #4
5. **Monday-Tuesday W2:** A5 (overlap resolution) — PR #5
6. **Wednesday W2:** Code review + merge
7. **Thursday-Friday W2:** Integration testing

**Merge Strategy:** Review in parallel, merge sequentially

---

### ✅ Week 3-4: Phase B
1. **Monday-Tuesday W3:** B1+B2 (taxonomy + bootstrap) — PR #6
2. **Wednesday-Thursday W3:** B3+B4 (ops + integration) — PR #7
3. **Friday W3:** B5 (config + tests) — PR #8
4. **Monday-Tuesday W4:** Code review
5. **Wednesday W4:** Performance benchmarking
6. **Thursday-Friday W4:** Documentation + merge

**Merge Strategy:** 3 PRs, sequential review/merge

---

### ⏸️ Week 5+: Planning
- **Phase C:** Escalation packet creation
- **Phase D:** Detector research + licensing analysis
- **Phase E:** Research roadmap (deferred)

---

## Success Metrics

### Phase A
- ✅ Zero crashes on 3D masks
- ✅ No feathering artifacts
- ✅ Configurable sigma working
- ✅ Single normalization path
- ✅ No double-processing
- ✅ Performance <0.5% regression

### Phase B
- ✅ Sky detection >80% outdoors
- ✅ False positives <10% indoors
- ✅ Visual quality improvement (subjective)
- ✅ Performance <25ms overhead
- ✅ Opt-in verified

---

## Governance Compliance

### ✅ Phases A + B: Compliant
- **Dependencies:** None (Section A: ✅)
- **CI/CD:** No changes (Section B: ✅)
- **Security:** No input handling changes (Section C: ✅)
- **Contracts:** Backward-compatible only (Section D: ✅)
- **ADRs:** No conflicts (Section E: ✅)

### ⚠️ Phases C/D/E: Escalation Required
- **C:** Cross-pipeline contracts (governance.md Section D)
- **D:** New dependencies (governance.md Section A)
- **E:** ADR uncertainty (governance.md Section E)

---

## Final Recommendations

### ✅ PROCEED: Phases A + B

**Rationale:**
- High ROI: Fixes critical bugs + adds top user feature
- Low risk: Zero new dependencies, backward compatible
- Well-scoped: 31 hours, 4 weeks, 19 tests
- Governance-compliant: No escalation needed

**Confidence:** 95%
**Timeline:** 4 weeks
**Effort:** ~1 sprint
**Performance Impact:** <40ms (<0.5%)

---

### ⏸️ ESCALATE: Phases C/D/E

**Phase C (Video Tracking):**
- **Trigger:** Cross-pipeline contracts
- **Action:** Create escalation packet (Appendix A)
- **Timeline:** 2-3 weeks planning + approval

**Phase D (Detector Integration):**
- **Trigger:** New ML dependencies (AGPL-3.0 concern)
- **Action:** Dependency governance review
- **Timeline:** 1-2 weeks research + approval

**Phase E (Synthesis):**
- **Trigger:** Research initiative, unclear ROI
- **Action:** Defer to research roadmap
- **Timeline:** 4-6 weeks research phase

---

## Appendix A: Escalation Packet Template

```markdown
# Escalation: SAM2 Video Tracking for Materials V3

## Objective
Improve material detection stability in video by implementing SAM2 temporal tracking with memory management.

## Affected Areas
- `spatial_ai/segmentation/sam2_backend.py` — Session API
- `lux_depth_v3/materials_v3.py` — Video state
- Orchestration — Frame propagation

## Proposed Approach
1. Add `SAM2Session` class: `start()`, `propagate()`, `close()`
2. Kalman gating for centroid drift detection
3. Memory bank pruning (MoSAM-inspired stability)

## Alternatives
1. Frame-independent (status quo) — simple but flicker-prone
2. Optical flow + warping — no ML, less accurate
3. ByteTrack (third-party) — new dependency

## Risks
- **Security:** Memory leaks, GPU OOM
- **Coupling:** lux_depth_v3 ↔ spatial_ai
- **Compatibility:** Image workflows must work
- **Performance:** +100-500ms/frame

## Enforcement
- Video-specific performance budget
- Frame IoU stability >0.9
- CI: memory leak detection (pytest-memray)

## Migration
- Opt-in: `enable_video_tracking=True`
- Image workflows unchanged
- Backward compatible
```

---

## Appendix B: File Modifications

### Phase A
| File | Type | Lines | Risk |
|------|------|-------|------|
| `pixel_ops_executor.py` | Modified | ~50 | LOW |
| `pixel_ops_registry.py` | Modified | ~20 | LOW |
| `config.py` | Modified | ~5 | LOW |

### Phase B
| File | Type | Lines | Risk |
|------|------|-------|------|
| `materials_v3_taxonomy.py` | Modified | ~5 | LOW |
| `sky_bootstrap.py` | **New** | ~150 | MED |
| `pixel_ops_registry.py` | Modified | ~80 | LOW |
| `segmentation_backend.py` | Modified | ~15 | LOW |
| `config.py` | Modified | ~5 | LOW |
| `config/materials_v3_sky.yaml` | **New** | ~100 | LOW |

**Total:** ~330 lines new, ~180 modified

---

## Document Metadata

| Field | Value |
|-------|-------|
| Version | 1.0 |
| Date | 2026-02-13 |
| Author | Transformation Portal Specialist |
| Status | Approved for A+B |
| Next Review | After Milestone 2 (4 weeks) |

---

**Next Action:** Begin Phase A.1 (3D mask bug fix)
**Escalation Status:** C/D/E on hold pending Architect review
**Confidence Level:** ✅ High (Phases A+B) / ⏸️ Pending (C/D/E)
