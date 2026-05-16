# Materials V3 Roadmap — Quick Reference

**Last Updated:** 2026-02-13
**Full Plan:** See `MATERIALS_V3_ROADMAP_IMPLEMENTATION_PLAN.md`

---

## TL;DR

**Phases A + B:** ✅ Approved — 31 hours, 4 weeks, zero new dependencies
**Phases C/D/E:** ⏸️ Escalation required — defer until Architect review

---

## Quick Win Summary

| Item | Effort | Risk | Impact | When |
|------|--------|------|--------|------|
| **A1: 3D Mask Bug** | 2h | 🟢 LOW | Critical crash fix | This week |
| **A4: Redundant Normalization** | 2h | 🟢 LOW | -2ms performance | This week |
| **A3: Configurable Feathering** | 3h | 🟢 LOW | Quality tuning | This week |
| **A2: Feathering Edge Fix** | 4h | 🟢 LOW | No more halos | Week 1-2 |
| **A5: Overlap Resolution** | 4h | 🟢 LOW | Correctness | Week 1-2 |
| **B: Sky Material** | 16h | 🟡 MED | Top user request | Week 3-4 |

---

## Phase Summary

### ✅ Phase A: Pixel Ops Hardening (Week 1-2)

**Goal:** Fix bugs + improve quality
**Effort:** 15 hours
**Risk:** 🟢 LOW
**Dependencies:** None

**5 Items:**
1. A1: Fix 3D mask crash (2h)
2. A2: Fix feathering edge clipping (4h)
3. A3: Configurable feathering per material (3h)
4. A4: Remove redundant normalization (2h)
5. A5: Overlap resolution by priority (4h)

**Deliverables:**
- 5 PRs
- 11 new tests
- <15ms performance impact
- 100% backward compatible

---

### ✅ Phase B: Sky Material (Week 3-4)

**Goal:** Add sky detection + ops
**Effort:** 16 hours
**Risk:** 🟡 MEDIUM
**Dependencies:** None

**5 Items:**
1. B1: Extend taxonomy (1h)
2. B2: Bootstrap heuristic (6h)
3. B3: Sky pixel ops (4h)
4. B4: Integration (3h)
5. B5: Config + tests (2h)

**Deliverables:**
- 3 PRs
- 8 new tests
- New file: `bootstrap/sky_seed.py`
- Sky configuration integrated into `EnhanceConfig`
- +10-20ms performance

---

### ⏸️ Phase C: Video Tracking (ESCALATE)

**Goal:** Temporal stability for video
**Risk:** 🔴 HIGH — Cross-pipeline contracts
**Action:** Create escalation packet

**Why Escalate:**
- New SAM2 session API
- Cross-pipeline state management
- Potential coupling issues

---

### ⏸️ Phase D: Detector Integration (ESCALATE)

**Goal:** YOLO/RT-DETR for "things"
**Risk:** 🔴 HIGH — New dependencies (AGPL-3.0)
**Action:** Dependency governance review

**Why Escalate:**
- YOLOv8: AGPL-3.0 license (commercial risk)
- 50-100MB model size
- New taxonomy classes

---

### ⏸️ Phase E: Material Synthesis (DEFER)

**Goal:** Advanced synthesis (ripples, sky replacement)
**Risk:** 🔴 VERY HIGH — Research-heavy
**Action:** Defer to research roadmap

---

## Testing Breakdown

| Phase | New Tests | Regression Tests | Total |
|-------|-----------|------------------|-------|
| A | 11 | 52 existing | 63 |
| B | 8 | 63 from A | 71 |
| **Total** | **19** | **71** | **71** |

---

## Performance Budget

| Phase | Overhead | Budget Impact |
|-------|----------|---------------|
| **A** | ~12ms net | <0.2% of 8.5s |
| **B** | +13-25ms | <0.3% of 8.5s |
| **Total** | **~37ms** | **<0.5% regression** |

**Existing Budget:** 8.5s p50, 12.0s p95
**Tolerance:** 10% (850ms)
**Actual Impact:** ~37ms (<5% of tolerance)

---

## Files Modified

### Phase A (5 files, ~75 lines)
- `pixel_ops_executor.py` (~50 lines)
- `pixel_ops_registry.py` (~20 lines)
- `config.py` (~5 lines)

### Phase B (5 files, ~255 lines)
- `materials_v3_taxonomy.py` (+5 lines)
- `bootstrap/sky_seed.py` (NEW, ~150 lines)
- `pixel_ops_registry.py` (+80 lines)
- `segmentation_backend.py` (+15 lines)
- `config.py` (+5 lines)

**Total:** ~255 lines (150 new, 105 modified)

---

## Governance Checklist

### ✅ Phases A + B: No Escalation Needed

- [x] **Dependencies:** None (uses existing numpy, scipy)
- [x] **CI/CD:** No changes
- [x] **Security:** No untrusted input handling
- [x] **Contracts:** Backward-compatible extensions only
- [x] **ADRs:** No conflicts

### ⚠️ Phases C/D/E: Escalation Required

- [ ] **C:** Cross-pipeline contracts (Session API)
- [ ] **D:** New dependencies (YOLO/RT-DETR)
- [ ] **E:** Research initiative (unclear ROI)

---

## Risk Matrix

| Phase | Risk | Mitigation |
|-------|------|------------|
| **A** | 🟢 LOW | Full test coverage, isolated changes |
| **B** | 🟡 MED | Opt-in config, new module isolated |
| **C** | 🔴 HIGH | **Escalate for architectural review** |
| **D** | 🔴 HIGH | **Escalate for dependency approval** |
| **E** | 🔴 VERY HIGH | **Defer to research roadmap** |

---

## Execution Timeline

### Week 1: Phase A (Part 1)
- **Mon:** A1 (3D mask bug) — PR #1
- **Tue:** A4 (redundant norm) — PR #2
- **Wed:** A3 (configurable feathering) — PR #3
- **Thu-Fri:** A2 (feathering edge fix) — PR #4

### Week 2: Phase A (Part 2)
- **Mon-Tue:** A5 (overlap resolution) — PR #5
- **Wed:** Code review + merge
- **Thu-Fri:** Integration testing

### Week 3: Phase B (Part 1)
- **Mon-Tue:** B1+B2 (taxonomy + bootstrap) — PR #6
- **Wed-Thu:** B3+B4 (ops + integration) — PR #7
- **Fri:** B5 (config + tests) — PR #8

### Week 4: Phase B (Part 2)
- **Mon-Tue:** Code review
- **Wed:** Performance benchmarking
- **Thu-Fri:** Documentation + merge

### Week 5+: Planning
- Phase C escalation packet
- Phase D dependency research
- Phase E deferred

---

## Success Criteria

### Phase A
- [ ] Zero crashes on 3D masks
- [ ] No feathering artifacts at edges
- [ ] Material-specific sigma working
- [ ] Single normalization path
- [ ] No double-processing in overlaps
- [ ] Performance <0.5% regression

### Phase B
- [ ] Sky detection >80% on outdoor images
- [ ] False positives <10% on indoor images
- [ ] Sky ops improve visual quality
- [ ] Performance <25ms overhead
- [ ] Opt-in verified (zero impact when disabled)

---

## Next Actions

### Immediate (This Week)
1. ✅ Start A1 (3D mask bug) — PR #1
2. ✅ Start A4 (redundant normalization) — PR #2
3. ✅ Start A3 (configurable feathering) — PR #3

### Week 1-2
4. Complete Phase A (A2, A5)
5. Integration testing
6. Merge all 5 PRs

### Week 3-4
7. Phase B implementation
8. Performance benchmarking
9. Documentation

### Week 5+
10. Create Phase C escalation packet
11. Research Phase D detectors
12. Review progress with Architect

---

## Quick Reference: Implementation Plan Location

**Full Plan:** `docs/materials/MATERIALS_V3_ROADMAP_IMPLEMENTATION_PLAN.md`
**This Quick Ref:** `docs/materials/ROADMAP_QUICK_REF.md`
**Governance:** `docs/architecture/agent_governance.md`
**ADR-048:** `docs/architecture/ADR-048-materials-v3-production-integration.md` (renumbered 2026-05-16 from ADR-030 to resolve collision with the canonical ADR-030)

---

## Key Contacts

**Specialist:** Transformation Portal Specialist (implementation)
**Architect:** Transformation Portal Architect (escalation, approval)
**Governance:** See `docs/architecture/agent_governance.md`

---

**Status:** ✅ Approved for Phases A + B
**Next Milestone:** Phase A Complete (Week 2)
**Next Review:** After Milestone 1 (2 weeks)
