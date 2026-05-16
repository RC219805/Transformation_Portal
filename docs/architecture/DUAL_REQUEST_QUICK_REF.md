# Dual Request Decision - Quick Reference

**Date:** 2026-02-15
**Decider:** Transformation Portal Architect
**Full Decision:** `docs/architecture/DUAL_REQUEST_ARCHITECT_DECISION.md`

---

## TL;DR

| Request | Decision | Action |
|---------|----------|--------|
| **Materials V3 Phase D** | ❌ **REJECTED** | Phase E (synthesis) out of scope |
| **APEX Phase 2 Branch** | ❌ **CLOSE BRANCH** | 3 months stale, massive drift |
| **Priority** | ⏸️ **HOLD ALL WORK** | Clean up debt first |

---

## Decision 1: Materials V3 Phase D

### ❌ REJECTED - OUT OF SCOPE

**What user requested:**
- Sky Rayleigh Scattering (physics-based rendering)
- Water Displacement Mapping (FFT-based ripples)

**Reality:**
- This is **Phase E** (Material Synthesis), not Phase D
- Phase E was already rejected in roadmap review (Feb 15, 2026)
- Synthesis is CGI/VFX work, not real estate enhancement

**What's actually implemented:**
- ✅ Phase A: Pixel Ops Hardening (complete)
- ✅ Phase B: Sky Enhancement (heuristic-based, complete)
- ✅ Phase C.2: SAM2 Confidence Semantics (merged PR #947)
- ⏸️ Phase C.3: SAM2Long (deferred, needs video architecture)
- ⏸️ Phase D: Detector Integration (escalated, needs use case)
- ❌ Phase E: Material Synthesis (REJECTED)

**No further Materials V3 work approved.**

---

## Decision 2: APEX Phase 2 Branch Status

### ❌ CLOSE BRANCH - MASSIVE DRIFT

**Branch:** `feat/apex-phase2-real-pipeline-integration`

**Status:**
- Last commit: Nov 7, 2024 (3 months old)
- 1,674 commits behind main
- 16 commits ahead of main

**Assessment:**
- Abandoned work from November 2024
- Predates StageGraph, Materials V3, security fixes
- Merge conflicts would be extensive and risky

**Recommendation:**
1. **Close branch** (don't continue)
2. **Document** why it was abandoned
3. **Re-assess** if real APEX integration still needed
4. **If yes:** Create fresh branch from current main

---

## Decision 3: Priority Sequencing

### ⏸️ HOLD ALL WORK - CLEANUP FIRST

**Sequence:**
1. ✅ **Close APEX Phase 2 branch** (immediate)
2. ✅ **Document closure reason** (1 day)
3. ⏸️ **Re-assess APEX needs** (1 week)
4. ⏸️ **Hold Materials V3 work** (nothing approved)

**Rationale:**
- Materials V3 in good state (Phases A+B complete)
- APEX branch is technical debt
- No urgent pressure, good time for planning

---

## Architecture Questions - Answered

### Q1: Does Phase D align with repository mission?

**A:** Phase E (what user described) does **NOT** align. Transformation Portal is **enhancement** (Lightroom), not **synthesis** (Blender).

### Q2: Are sky/water shaders blocked by video pipeline?

**A:** Sky/water **enhancement** works on images (implemented in Phase B). Sky/water **synthesis with temporal consistency** would need video pipeline (not built, out of scope).

### Q3: Do sky/water require SAM2Long?

**A:** Existing SAM2 backend (Phase C.2) sufficient for images. SAM2Long needed for video tracking (deferred, not required for current work).

### Q4: Do new material types require schema changes?

**A:** Sky already in taxonomy (Phase B). Water already in taxonomy. No schema changes needed or approved.

### Q5: Should Phase D proceed before or after APEX Phase 2?

**A:** Neither. Phase E rejected, APEX branch should be closed.

---

## Governance Enforcement

### Mission Alignment

**Approved:**
- ✅ Detect materials in real images (SAM2 backend)
- ✅ Enhance detected regions (pixel ops)
- ✅ Maintain realistic appearance

**Rejected:**
- ❌ Synthesize materials with physics simulation
- ❌ Replace real sky with Rayleigh scattering
- ❌ Generate water ripples with FFT

### Branch Hygiene

**Policy:** Branches >1 month stale with >100 commits drift should be closed.

**Enforcement:**
- `feat/apex-phase2-real-pipeline-integration` violates policy
- Close and re-plan instead of heroic merge

### Video Pipeline Blocker

**Reality:** No video pipeline exists (lux_depth_v3 is image-only).

**Blocked work:**
- SAM2Long (Phase C.3) - deferred
- Temporal material tracking
- Any cross-frame consistency features

**Don't approve video-dependent work until foundation exists.**

---

## Materials V3 Current State

| Phase | Status | Description |
|-------|--------|-------------|
| **A** | ✅ COMPLETE | 3D mask fix, feathering, overlap resolution |
| **B** | ✅ COMPLETE | Sky detection + enhancement (heuristic) |
| **C.1** | ✅ COMPLETE | Memory protocol (via A.6) |
| **C.2** | ✅ MERGED | SAM2 confidence semantics (PR #947) |
| **C.3** | ⏸️ DEFERRED | SAM2Long (needs video architecture) |
| **D** | ⏸️ ESCALATED | Detector integration (needs use case) |
| **E** | ❌ REJECTED | Material synthesis (out of scope) |

**Repository commits:**
- `0582b30d` - Phase C.2 merged (Feb 15, 2026)
- `d8004b35` - Phase A.1 + A.6 (SAM2 stability)
- Previous: Phase A and B implementation

---

## APEX Current State

**Scaffolding:** ✅ Complete (merged)
- SQLite ledger, CI integration, PR comments
- Performance bucketing, zone-aware metrics
- All running in `--dry-run` mode (synthetic data)

**Phase 2 (Real Integration):** ⏸️ Stale branch, needs re-plan
- Connect to real orchestrator
- Remove `--dry-run` flag
- Real timing measurements
- Shadow mode rollout

**Decision:** Close old branch, re-assess if needed, re-plan from fresh main.

---

## What Happens Next

### Immediate (User Actions)

1. **Close APEX Phase 2 branch:**
   ```bash
   git branch -D feat/apex-phase2-real-pipeline-integration
   ```

2. **Document closure:**
   - Create `docs/apex/APEX_PHASE2_CLOSURE.md`
   - Explain 3-month staleness, massive drift
   - Note decision to re-plan if needed

### 1 Week (Assessment)

3. **Re-assess APEX needs:**
   - Is dry-run mode sufficient?
   - What value does real execution provide?
   - Are there blockers today?

4. **If real integration needed:**
   - Create fresh branch from current main
   - Re-scope against StageGraph/Materials V3
   - Estimate: 1-2 weeks (not 3 months of merges)

### Hold Indefinitely

5. **Materials V3:**
   - No work approved
   - Phases A+B complete, C.2 merged
   - Phase E rejected (synthesis out of scope)
   - Phase C.3/D deferred pending dependencies

---

## Key Takeaways

1. **Phase confusion:** User requested Phase E (synthesis), not Phase D (detector)
2. **Already rejected:** Phase E was assessed and rejected in roadmap review
3. **APEX drift:** 3-month-old branch should be closed, not continued
4. **Good state:** Repository is healthy, no urgent work needed
5. **Strategic pause:** Good time for planning, not execution

---

## References

- **Full Decision:** `docs/architecture/DUAL_REQUEST_ARCHITECT_DECISION.md`
- **Roadmap Review:** `docs/materials/MATERIALS_V3_ROADMAP_COMPREHENSIVE_REVIEW.md`
- **APEX Phase 2 Scope:** `docs/guides/APEX_REAL_PIPELINE_INTEGRATION.md`
- **Governance Policy:** `docs/architecture/agent_governance.md`
- **Materials V3 ADR:** `docs/architecture/ADR-048-materials-v3-production-integration.md` (renumbered 2026-05-16 from ADR-030)

---

**Signed:** Transformation Portal Architect
**Date:** 2026-02-15
**Status:** Binding per governance policy
