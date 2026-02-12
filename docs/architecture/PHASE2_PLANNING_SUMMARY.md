# Phase 2 Planning Summary

**Date:** 2026-02-11
**Status:** Awaiting Architect Approval
**Planning Session:** Complete ✅

---

## Documents Created

| Document | Purpose | Status |
|----------|---------|--------|
| **PHASE2_IMPLEMENTATION_PLAN.md** | Comprehensive implementation plan (48KB) | ✅ Complete |
| **PHASE2_QUICKREF.md** | Quick reference guide (8KB) | ✅ Complete |
| **ADR-027-phase2-spatial-ai-extension.md** | Architectural decision record (17KB) | ✅ Complete |
| **PHASE2_PLANNING_SUMMARY.md** | This document | ✅ Complete |

**Total Documentation:** 73KB (comprehensive)

---

## Executive Summary

Phase 2 extends the Spatial AI Foundation (Phase 1) with:

1. **SAM2 Integration** — Temporal-consistent segmentation
2. **MaterialGAN Integration** — Physics-based PBR texture generation
3. **3D Gaussian Splatting** — Depth-guided scene reconstruction

**Timeline:** 5 weeks (23 days)
**Estimated LOC:** 3,700 implementation + 1,850 tests
**Risk Level:** MEDIUM (well-scoped, mitigated)

---

## Key Architectural Decisions

### 1. Namespace Structure

```
src/transformation_portal/
├── spatial_ai/              # Phase 2 code goes here
│   ├── ingest/              # Phase 1 (done)
│   ├── segmentation/        # Phase 2.1 (SAM2)
│   ├── materials/           # Phase 2.2 (MaterialGAN)
│   ├── reconstruction/      # Phase 2.3 (3DGS)
│   └── orchestration/       # Phase 2.4 (pipeline)
└── core/geometry/           # Shared utilities (NEW)
```

### 2. Constraint Compliance

Phase 2 respects **all** Phase 1.1 constraints:

- ✅ **ADR-023 Isolation:** No imports from `lux_depth_v3` (AST-enforced)
- ✅ **HF Revision Pinning:** Experimental → stable promotion path
- ✅ **OpenEXR Preflight:** Fail-fast with clear errors
- ✅ **Lane-Based Strictness:** Dev fast, research strict

### 3. Model Selection

| Component | Primary Model | License | Tier |
|-----------|--------------|---------|------|
| Segmentation | SAM2 Large | Apache 2.0 | All tiers |
| PBR Textures | NVDIFFREC | BSD-3-Clause | All tiers |
| Reconstruction | 3DGS | Inria (research) | Research only |

### 4. Testing Strategy

**Test Pyramid:**
- 40+ unit tests (≥85% coverage)
- 12 component integration tests
- 3 E2E integration tests
- CI enforcement (isolation, HF revisions)

---

## Implementation Phases

| Week | Phase | Deliverable | Days |
|------|-------|-------------|------|
| 1 | 2.1 SAM2 | Segmentation working | 5 |
| 2 | 2.2 MaterialGAN | PBR textures working | 5 |
| 3 | 2.3 3DGS | Reconstruction working | 5 |
| 4 | 2.4 Orchestration | E2E pipeline working | 5 |
| 5 | 2.5 Hardening | Merge-ready | 3 |

**Total:** 23 days (~5 weeks)

---

## Risk Assessment

| Risk | Impact | Mitigation | Status |
|------|--------|------------|--------|
| Dependency hell | HIGH | Tiered deps, fallbacks | Documented |
| GPU OOM | MEDIUM | Model unloading, batching | Documented |
| HF verification overhead | LOW | Automated script | Addressed |
| ADR-023 violations | MEDIUM | Pre-commit hook | Enforced |
| 3DGS license ambiguity | HIGH | Tier restriction | Policy set |
| Performance regression | MEDIUM | Opt-in stages, caching | Planned |

**Overall Risk Posture:** MEDIUM (acceptable with mitigations)

---

## Success Criteria

### Technical Metrics

- Test coverage ≥85%
- CI green (100% checks passing)
- Isolation compliance 100%
- HF revision compliance 100% (stable presets)
- Performance regression <5%

### Functional Metrics

- SAM2: 512x512 in <3s on GPU
- MaterialGAN: 1024x1024 PBR in <10s on GPU
- 3DGS: 3-view scene in <30s, RMSE <2%
- E2E pipeline: <60s (512x512, GPU)

### Governance Metrics

- Zero ADR-023 violations
- All stable presets have commit SHAs
- Security audit passes
- Documentation complete

---

## Compatibility with Phase 1.1

Phase 2 implements **all** items from the compatibility checklist:

1. ✅ **OpenEXR preflight checks** — Added to orchestration
2. ✅ **Lane-based strictness** — Dev vs research configs
3. ✅ **Correct namespace** — All code in `spatial_ai/`
4. ✅ **Experimental presets** — Promotion workflow defined
5. ✅ **Documentation updates** — All invariants documented

**Benefit:** No governance friction. Design for constraints, not against them.

---

## Dependencies Added

**New Packages:**

| Package | Purpose | License | Install Tier |
|---------|---------|---------|--------------|
| sam2 (HF) | Segmentation | Apache 2.0 | spatial-ai |
| nvdiffrast | Diff rendering | BSD-3-Clause | spatial-ai |
| pytorch3d | 3D transforms | BSD-3-Clause | spatial-ai |
| trimesh | Mesh processing | MIT | spatial-ai |
| open3d | Mesh export | MIT | spatial-ai |
| 3dgs (Inria) | Gaussian splatting | Inria (research) | spatial-ai-research |

**Installation:**
```bash
pip install transformation-portal[spatial-ai]           # Base
pip install transformation-portal[spatial-ai-research]  # + 3DGS
```

---

## Next Steps

### Immediate (This Week)

1. **Architect Review (REQUIRED)**
   - Review PHASE2_IMPLEMENTATION_PLAN.md
   - Review ADR-027-phase2-spatial-ai-extension.md
   - Approve or request changes

2. **If Approved:**
   - Begin Phase 2.1 (SAM2 integration)
   - Specialist implements segmentation module
   - Create experimental preset
   - Write tests

3. **If Changes Requested:**
   - Address Architect feedback
   - Update plan and ADR
   - Re-submit for approval

---

## Documents to Review

### Primary Review Documents

1. **`docs/architecture/PHASE2_IMPLEMENTATION_PLAN.md`**
   - Comprehensive 48KB implementation plan
   - Architecture design, component breakdown, testing strategy
   - Implementation phases (5 weeks)
   - Risk assessment and mitigation
   - **Review Priority: HIGH**

2. **`docs/architecture/ADR-027-phase2-spatial-ai-extension.md`**
   - Architectural decision record (17KB)
   - Formalizes namespace, contracts, constraints
   - License compliance decisions
   - **Review Priority: HIGH**

### Supporting Documents

3. **`docs/architecture/PHASE2_QUICKREF.md`**
   - Quick reference guide (8KB)
   - Import cheat sheet, common mistakes, quick commands
   - **Review Priority: MEDIUM**

### Session Files (Context)

4. **`~/.copilot/session-state/.../phase2_compatibility_checklist.md`**
   - Phase 1.1 constraints that Phase 2 must respect
   - **Review Priority: MEDIUM** (already integrated into plan)

5. **`~/.copilot/session-state/.../phase1.1_completion_summary.md`**
   - Phase 1.1 completion status
   - **Review Priority: LOW** (context only)

---

## Approval Workflow

```
┌─────────────────────────────────────────────────────┐
│ 1. Architect Reviews Plan & ADR                    │
│    - PHASE2_IMPLEMENTATION_PLAN.md                  │
│    - ADR-027-phase2-spatial-ai-extension.md         │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ 2. Decision                                         │
│    Option A: APPROVE → Begin Phase 2.1             │
│    Option B: REQUEST CHANGES → Update & Re-submit  │
└─────────────────────────────────────────────────────┘
                        ↓
         ┌──────────────┴──────────────┐
         ↓                             ↓
┌──────────────────┐        ┌──────────────────────┐
│ APPROVED         │        │ CHANGES REQUESTED    │
│ → Begin Week 1   │        │ → Address Feedback   │
│ → SAM2 integration│       │ → Update Documents   │
└──────────────────┘        └──────────────────────┘
```

---

## Key Principles

### Design Philosophy

> **"Respect Phase 1.1 constraints. Design for rigor, not against it."**

- Constraints are epistemic hygiene, not obstacles
- OpenEXR preflight prevents late failures
- Lane-based strictness preserves velocity
- Isolation boundaries prevent contamination
- Experimental presets enable rapid iteration

### Architectural Position

> **"Phase 2 components are first-class members of spatial_ai, not add-ons."**

- SAM2, MaterialGAN, 3DGS extend the linear foundation
- Clear module boundaries (one concern per module)
- Explicit contracts (input/output validation)
- CI enforcement (isolation, revisions, security)

### Testing Posture

> **"Test coverage is governance, not bureaucracy."**

- Unit tests validate contracts and edge cases
- Integration tests validate cross-component behavior
- E2E tests validate end-to-end workflows
- CI tests validate compliance (isolation, revisions)

---

## Questions for Architect

### Critical Decisions

1. **NVDIFFREC vs MaterialGAN?**
   - Recommendation: NVDIFFREC (BSD-3-Clause, better license)
   - Alternative: MaterialGAN (CC BY-NC 4.0, research only)
   - Decision needed?

2. **3DGS License Enforcement?**
   - Recommendation: Tier restriction (`apex_research` only)
   - Alternative: NeuS2 (MIT) for commercial tier
   - CLI enforcement sufficient, or need legal review?

3. **CLIP Integration Timing?**
   - Recommendation: Add in Phase 2.1 (simple, opt-in)
   - Alternative: Defer to Phase 3
   - Decision needed?

### Scope Questions

4. **Multi-View Input for 3DGS?**
   - Support video, image directory, or both?
   - Recommendation: Both (video primary, directory fallback)

5. **Heuristic Fallbacks?**
   - MaterialGAN: Heuristic fallback if GPU unavailable?
   - Recommendation: Yes (CPU fallback to existing PBR code)

---

## Estimated Effort Summary

| Category | Estimate | Confidence |
|----------|----------|------------|
| **Implementation** | 18 days | High |
| **Testing** | 3 days | High |
| **Documentation** | 1.5 days | High |
| **Hardening** | 2.5 days | Medium |
| **Buffer** | 2 days | - |
| **Total** | **23 days** | **High** |

**Actual Range:** 20-28 days (accounting for unknowns)

---

## Status

**Planning Phase:** ✅ COMPLETE
**Architect Review:** 🔄 PENDING
**Implementation:** ⏸️ BLOCKED (awaiting approval)

---

## Contact

**Plan Owner:** Transformation Portal Architect
**Implementation Lead:** Transformation Portal Specialist (after approval)
**Session ID:** 761a306f-125d-46e1-828c-9a10f47cd12f

---

**Last Updated:** 2026-02-11
**Next Update:** After Architect review
