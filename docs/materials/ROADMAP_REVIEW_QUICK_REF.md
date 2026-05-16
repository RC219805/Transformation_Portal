# Materials V3 Roadmap Review — Quick Reference

**Date:** 2026-02-15
**Full Review:** `MATERIALS_V3_ROADMAP_COMPREHENSIVE_REVIEW.md`
**Status:** ✅ Approved for A+B, ⏸️ Escalate C/D, ⏸️ Defer E

---

## TL;DR — Executive Decision

**✅ START IMMEDIATELY:** Phases A+B (31 hours, 4 weeks, zero new dependencies)
**⏸️ ESCALATE:** Phases C+D (cross-pipeline contracts, ML dependencies)
**⏸️ DEFER:** Phase E (out of scope, research initiative)

**Confidence:** 95% (A+B) | Pending (C/D/E)

---

## Top 3 Recommendations

### 1. ✅ Phase A: Pixel Ops Hardening (Week 1-2)

**Priority:** 🔴 **CRITICAL** — Fixes SAM2 blocker crash

**Why Start Immediately:**
- A1 fixes 3D mask crash (blocks SAM2 production use)
- Low risk: isolated changes, full test coverage
- High ROI: improves quality + performance (-2ms)

**Effort:** 15 hours (~2 days)
**Tests:** 11 new tests
**Risk:** 🟢 LOW

---

### 2. ✅ Phase B: Sky Material (Week 3-4)

**Priority:** 🌟 **HIGH** — Top user request

**Why High Priority:**
- Top user request for exterior rendering
- Zero new dependencies (uses scipy)
- Opt-in (zero risk to existing users)

**Effort:** 16 hours (~2 days)
**Tests:** 8 new tests
**Risk:** 🟡 MEDIUM (new module, needs validation)

---

### 3. ⚠️ NEW: Memory Management (Add to Phase A)

**Priority:** 🔴 **HIGH** — Prevents OOM errors

**Why Add to Phase A:**
- SAM2 inference state leaks GPU memory
- Users report OOM after 50+ images
- Simple fix: try-finally cleanup pattern

**Effort:** 2 hours
**Tests:** 2 tests
**Risk:** 🟢 LOW

---

## Decision Matrix

| Phase | Decision | Effort | Risk | When |
|-------|----------|--------|------|------|
| **A: Pixel Ops** | ✅ **GO** | 15h | 🟢 LOW | Week 1-2 (start Mon) |
| **B: Sky Material** | ✅ **GO** | 16h | 🟡 MED | Week 3-4 |
| **NEW: Memory** | ✅ **ADD** | 2h | 🟢 LOW | Week 1 (add to Phase A) |
| **C: Video Tracking** | ⏸️ **ESCALATE** | TBD | 🔴 HIGH | Week 5+ (planning) |
| **D: Detector** | ⏸️ **ESCALATE** | TBD | 🔴 HIGH | Week 5+ (research) |
| **E: Synthesis** | ⏸️ **DEFER** | 40h+ | 🔴 VERY HIGH | Out of scope |

---

## Phase A: Critical Items (Week 1-2)

### ✅ A1: Fix 3D Mask Bug (2 hours) — START FIRST

**Problem:** SAM2 returns `(H,W,1)` masks → crash in `_bounding_box()`
**Solution:** `_canonical_mask()` helper with squeeze + validation
**Impact:** 🔴 **BLOCKER** — Unblocks SAM2 production use

### ✅ A2: Fix Feathering Edge Clipping (4 hours)

**Problem:** Gaussian blur clips at bbox edges → visible halos
**Solution:** Expand bbox by 3σ before blurring
**Impact:** Quality improvement (glass against sky scenes)

### ✅ A3: Configurable Feathering (3 hours)

**Feature:** Per-material sigma (sky=5.0, glass=1.5, water=3.0)
**Config:** `mask_feather_sigma_overrides: Dict[str, float]`
**Impact:** Better quality tuning per material

### ✅ A4: Remove Redundant Normalization (2 hours)

**Problem:** Executor normalizes, then ops normalize again
**Solution:** Single normalization point (executor only)
**Impact:** -2ms performance + simpler code

### ✅ A5: Overlap Resolution (4 hours)

**Problem:** Overlapping masks cause double-processing
**Solution:** Priority-based resolution (sky > glass > water)
**Impact:** Correctness (no oversaturation)

### ✅ NEW A6: SAM2 Memory Cleanup (2 hours) — RECOMMENDED

**Problem:** SAM2 inference state leaks GPU memory
**Solution:** Try-finally cleanup with `reset_state()` + GPU sync
**Impact:** Prevents OOM errors

**Total Phase A:** 17 hours (15 + 2 for A6)

---

## Phase B: Sky Material (Week 3-4)

### ✅ B1: Extend Taxonomy (1 hour)

**Addition:** `"sky": {"priority": 15, "threshold": 0.30, "canary": True}`
**Impact:** Sky becomes highest priority material

### ✅ B2: Sky Bootstrap Heuristic (6 hours)

**Strategy:** Geometric + color priors (no ML)
- Top-of-frame prior (exponential decay from top 40%)
- Low gradient (sky is smooth)
- High brightness (sky is bright)
- Depth prior (sky is far, if depth available)

**Output:** Confidence mask + bbox/point proposals (SAM2-ready)

### ✅ B3: Sky Pixel Ops (4 hours)

**Three Operations:**
1. `sky_dehaze`: Reduce atmospheric haze
2. `sky_gradient_smooth`: Smooth color transitions
3. `sky_temperature_shift`: Warmer/cooler sky

### ✅ B4: Integration (3 hours)

**Approach:** Opt-in via `enable_sky_bootstrap=False` (default)
**Coexistence:** Works alongside SAM2 backend

### ✅ B5: Config + Tests (2 hours)

**Config:** `enable_sky_bootstrap`, `sky_confidence_threshold`, etc.
**Tests:** 8 tests (outdoor, indoor, depth, ops)

**Total Phase B:** 16 hours

---

## Phase C/D/E: Escalation Required

### ⏸️ Phase C: Video Tracking (Escalate)

**Why Escalate:**
- Cross-pipeline contracts (`SAM2Session` API)
- Memory bank state management
- Governance Section D triggered

**Recommended Refinements:**
1. Simplify scope: Basic tracking only (no SAM2Long)
2. Memory protocol: Try-finally cleanup
3. Confidence semantics: Decouple IoU from coverage
4. Defer SAM2Long: Wait for official release

**Escalation Packet:** See roadmap Appendix A (complete)
**Timeline:** 2-3 weeks planning + approval

---

### ⏸️ Phase D: Detector Integration (Escalate)

**Why Escalate:**
- New ML dependencies (YOLOv8/RT-DETR)
- License risk (AGPL-3.0 for YOLOv8)
- Governance Section A triggered

**Recommended Approach:**
- ✅ Prefer Det-SAM2 (Meta, Apache-2.0)
- ❌ Reject YOLOv8 (AGPL-3.0)
- ⚠️ Validate use case first

**Questions for Architect:**
1. Do we need object detection? (Use case)
2. Which detector? (Det-SAM2 vs RT-DETR)
3. Which tier? (Commercial vs Research)
4. Performance budget? (+50-200ms OK?)

**Timeline:** 1-2 weeks research + approval

---

### ⏸️ Phase E: Material Synthesis (Defer)

**Why Defer:**
- Out of scope (enhancement, not CGI synthesis)
- High complexity, unclear ROI
- Better suited for research roadmap

**Recommendation:** Defer indefinitely

---

## Governance Compliance Summary

### ✅ Phases A+B: No Escalation Needed

| Criterion | Status | Notes |
|-----------|--------|-------|
| Dependencies (Section A) | ✅ NONE | Uses numpy, scipy (existing) |
| CI/CD (Section B) | ✅ NONE | No workflow changes |
| Security (Section C) | ✅ NONE | No untrusted input |
| Contracts (Section D) | ✅ ADDITIVE | Backward-compatible only |
| ADRs (Section E) | ✅ COMPLIANT | No conflicts |

### ⚠️ Phases C/D/E: Escalation Required

| Phase | Trigger | Section |
|-------|---------|---------|
| **C** | Cross-pipeline contracts | Section D |
| **D** | New ML dependencies | Section A |
| **E** | ADR uncertainty | Section E |

---

## Timeline Summary

### 4-Week Approved Plan (A+B Only)

```
Week 1: Phase A (Part 1)
├─ Mon: A1 (3D mask bug) — CRITICAL, START FIRST
├─ Mon-Tue: NEW A6 (memory cleanup) — RECOMMENDED
├─ Tue: A4 (redundant normalization)
├─ Wed: A3 (configurable feathering)
└─ Thu-Fri: A2 (feathering edge fix)

Week 2: Phase A (Part 2)
├─ Mon-Tue: A5 (overlap resolution)
├─ Wed: Code review + merge
└─ Thu-Fri: Integration testing

Week 3: Phase B
├─ Mon-Tue: B1+B2 (taxonomy + bootstrap)
├─ Wed-Thu: B3+B4 (ops + integration)
└─ Fri: B5 (config + tests)

Week 4: Phase B Review
├─ Mon-Tue: Code review
├─ Wed: Performance benchmarking
└─ Thu-Fri: Documentation

Week 5+: Planning
├─ Phase C escalation packet
├─ Phase D detector research
└─ Phase E deferred
```

---

## Performance Budget

| Phase | Overhead | Budget Impact |
|-------|----------|---------------|
| **A** | ~12ms net | <0.2% of 8.5s budget |
| **B** | +13-25ms | <0.3% of 8.5s budget |
| **NEW A6** | +<5ms | Negligible (cleanup) |
| **Total** | **~37ms** | **<0.5% regression** |

**Quality Firewall:** 10% regression tolerance = 850ms
**Actual Impact:** 37ms (<5% of tolerance) ✅

---

## Success Criteria

### Phase A

- [ ] Zero crashes on 3D masks
- [ ] No feathering halos at edges
- [ ] Material-specific sigma working
- [ ] Single normalization path
- [ ] No double-processing
- [ ] No memory leaks (NEW A6)
- [ ] Performance <0.5% regression

### Phase B

- [ ] Sky detection >80% outdoors
- [ ] False positives <10% indoors
- [ ] Visual quality improvement
- [ ] Performance <25ms overhead
- [ ] Opt-in verified

---

## Risk Assessment

| Phase | Risk | Mitigation |
|-------|------|------------|
| **A** | 🟢 LOW | Full test coverage, isolated changes |
| **NEW A6** | 🟢 LOW | Defensive cleanup, no logic changes |
| **B** | 🟡 MED | Opt-in config, indoor rejection tests |
| **C** | 🔴 HIGH | Escalate for memory protocol ADR |
| **D** | 🔴 HIGH | Escalate for dependency approval |
| **E** | 🔴 VERY HIGH | Defer entirely (out of scope) |

---

## Key Technical Findings

### ✅ SAM2 Integration Complete (Feb 2026)

**Status:** Production-ready
- Adapter pattern: `sam2_adapter.py` (270 lines)
- HuggingFace pipeline integration
- 15 tests passing, zero regressions
- Models: sam2-base (1.2GB), sam2-large (2.5GB)

**Limitation:** Auto-mask-generation scaffolding incomplete
**Fix:** Phase A.1 (3D mask bug) enables production use

---

### ✅ Materials V3 Architecture Sound

**52 tests passing, zero regressions**
- Protocol-based backends (stub/efficientsam/sam2)
- V2 integration complete (NPZ serialization)
- Pixel ops: 5 operations, ROI-based
- Taxonomy: 8 materials (glass, water, foliage, etc.)

**Governance Alignment:** Perfect ADR-030 compliance

---

### ⚠️ Memory Management Gap (NEW FINDING)

**Issue:** SAM2 inference state leaks GPU memory
**Evidence:** User reports of OOM after 50+ images
**Solution:** NEW A6 (try-finally cleanup)
**Priority:** 🔴 HIGH

---

## Dependency Analysis

### ✅ Phases A+B: Zero New Dependencies

**Uses:**
- numpy (existing)
- scipy (existing)
- pillow (existing)

**License:** BSD/MIT (commercial-friendly)
**Offline:** 100% supported
**Security:** No external downloads

---

### ⚠️ Phase C: Optional Dependency

**filterpy** (Kalman filtering)
- License: MIT
- Size: ~2MB
- Risk: LOW
- Decision: Optional, defer

---

### ⚠️ Phase D: High-Risk Dependencies

| Dependency | License | Size | Risk | Decision |
|-----------|---------|------|------|----------|
| **YOLOv8** (Ultralytics) | **AGPL-3.0** | 100MB | **🔴 HIGH** | **❌ REJECT** |
| **RT-DETR** (Baidu) | Apache-2.0 | 50MB | 🟡 MED | ⏸️ Escalate |
| **Det-SAM2** (Meta) | Apache-2.0 | 80MB | 🟢 LOW | ✅ Preferred |

**Recommendation:** Det-SAM2 if detector needed

---

## Files Modified Summary

### Phase A (6 files)

- `pixel_ops_executor.py` (~60 lines)
- `pixel_ops_registry.py` (~20 lines)
- `config.py` (~5 lines)
- **NEW:** `sam2_backend.py` (~15 lines for A6)

### Phase B (5 files)

- `materials_v3_taxonomy.py` (+5 lines)
- `bootstrap/sky_seed.py` (**NEW**, ~150 lines)
- `pixel_ops_registry.py` (+80 lines)
- `segmentation_backend.py` (+15 lines)
- `config.py` (+5 lines)

**Total:** ~150 lines new, ~120 modified

---

## Testing Summary

| Phase | New Tests | Regression Tests | Total |
|-------|-----------|------------------|-------|
| **A** | 11 (+2 for A6) | 52 existing | 65 |
| **B** | 8 | 65 from A | 73 |
| **Total** | **21** | **73** | **73** |

**CI Impact:**
- Test suite: <10 minutes (heuristic-only, fast)
- Offline-friendly: No model downloads
- Isolated: Mock heavy dependencies

---

## Next Actions

### Immediate (Monday Morning)

1. ✅ **START A1:** 3D mask bug fix (2 hours) — BLOCKER
2. ✅ **START NEW A6:** SAM2 memory cleanup (2 hours) — HIGH PRIORITY

### Week 1

3. Complete Phase A items (A2-A5)
4. Code review + merge PRs
5. Integration testing

### Week 2

6. Performance benchmarking
7. Documentation updates
8. Prepare for Phase B

### Week 3-4

9. Phase B implementation (sky material)
10. Performance validation
11. Documentation (SKY_MATERIAL_GUIDE.md)

### Week 5+

12. Create Phase C escalation packet
13. Research Phase D detectors
14. Review progress with Architect

---

## Additional Recommendations

### 1. ✅ LRU Cache Tuning (Nice-to-Have)

**Current:** `@lru_cache(maxsize=2)` in `segmentation_backend.py`
**Recommendation:** Make configurable (8-16 for batch processing)
**Effort:** 1 hour
**Priority:** 🟡 MEDIUM

---

### 2. ✅ SAM2 Best Practices Guide (Documentation)

**Create:** `docs/materials/SAM2_BEST_PRACTICES.md`
**Contents:** Memory management, when to use SAM2, troubleshooting
**Effort:** 3 hours
**Priority:** 🟡 MEDIUM

---

### 3. ⚠️ Indoor Rejection Tests (Phase B Enhancement)

**Recommendation:** Add property-based tests for indoor false positives
**Effort:** 2 hours
**Priority:** 🟡 MEDIUM

---

## Key Contacts

**Specialist:** Transformation Portal Specialist (implementation)
**Architect:** Transformation Portal Architect (escalation, approval)
**Governance:** `docs/architecture/agent_governance.md`

---

## Document References

**Full Review:** `MATERIALS_V3_ROADMAP_COMPREHENSIVE_REVIEW.md`
**Implementation Plan:** `MATERIALS_V3_ROADMAP_IMPLEMENTATION_PLAN.md`
**Quick Ref (Original):** `ROADMAP_QUICK_REF.md`
**Governance:** `docs/architecture/agent_governance.md`
**ADR-048:** `docs/architecture/ADR-048-materials-v3-production-integration.md` (renumbered 2026-05-16 from ADR-030 to resolve collision with the canonical ADR-030)

---

## Final Recommendation

### ✅ **APPROVE & START IMMEDIATELY**

**Phases A+B:** 31 hours, 4 weeks, zero new dependencies
**Confidence:** 95%
**Starting Point:** Phase A.1 (3D mask bug) + NEW A6 (memory cleanup)

**Action:** Begin Monday morning — START WITH A1 (CRITICAL BLOCKER)

---

**Status:** ✅ Approved for A+B, ⏸️ Escalate C/D, ⏸️ Defer E
**Last Updated:** 2026-02-15
**Next Review:** After Phase A Complete (Week 2)
