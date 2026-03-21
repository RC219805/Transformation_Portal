# TODO Inventory Quick Reference

**Version:** 2.2.0 | **Date:** 2026-03-21 | **Status:** ACTIVE

**Full Inventory:** [docs/analysis/TODO_INVENTORY.md](../../analysis/TODO_INVENTORY.md)
**Executive Summary:** [TODO_INVENTORY_EXECUTIVE_SUMMARY.md](TODO_INVENTORY_EXECUTIVE_SUMMARY.md)

---

## At A Glance

| Metric | Value |
|--------|-------|
| **Total Items** | 65 |
| **✅ Completed** | 22 (34%) |
| **🟢 Correct (No Action)** | 32 (49%) |
| **📦 Obsolete (Archived)** | 14 (22%) |
| **⏳ Action Required** | 2 (3%) |
| **P0 Blockers** | 0 |
| **Repository Health** | **EXCELLENT** ✅ |

**Recently Completed (2026-03-21):**
- P0: Test Marker Enforcement (ADR-044) ✅ - 95.1% coverage achieved
- P0: GitHub Actions SHA Pinning ✅
- P0: pytest-xdist parallelization ✅
- P0: mypy hard-fail for critical modules ✅
- P1: HuggingFace model revision pinning ✅
- P1: Rollback procedures documented ✅ (docs/operations/ROLLBACK_PROCEDURES.md)
- P2: ICC profile preservation in 16-bit TIFF ✅
- P2: depth_canonical module archived ✅
- P2: context_aware_rendering.py archived ✅
- P2: lux_render_pipeline_plus_v3.py archived ✅

---

## Immediate Actions (v2.4.0 Planning - Q2 2026)

**Total Effort:** 4 hours → **Remaining:** 15min
**Owner:** Architect + DevOps
**Last Updated:** 2026-03-21

| # | Task | Priority | Effort | Owner | Status |
|---|------|----------|--------|-------|--------|
| 1 | Create rollback procedures | P1 | 2h | Architect | ✅ Complete |
| 2 | Verify branch protection | P1 | 15min | Admin | Pending (GitHub Admin) |
| 3 | Update V2_0_0_RELEASE_REVIEW.md | P2 | 30min | Architect | Low priority (historical) |
| 4 | Archive obsolete modules | P2 | 1h | Specialist | ✅ Complete |

---

## Sprint Planning

### Sprint 1: Cleanup (Week 1, 8h) - **✅ Complete**
1. ~~Delete depth_canonical module (1h)~~ ✅ Archived
2. ~~Archive obsolete PR docs (30min)~~ ✅ Completed 2026-03-14
3. ~~Update binary cleanup docs (30min)~~ ✅ Completed
4. Complete CLI e2e tests (4h) - Deferred to v2.4.0
5. ~~Audit ADR-023 manifest (2h)~~ ✅ Verified current

### Sprint 2: Phase 2 Foundation (Week 2-3, 16h)
6. SAM2 auto mask generation (P2, 3-4 days) - Roadmap item
7. ~~Dependency pinning validation (P2, 4h)~~ ✅ HuggingFace revisions pinned (2026-03-14)
8. Create V2_3_0 release checklist (P2, 2h) - As needed

### Sprint 3: Nice-to-Have (v2.3.0+, 12h)
9. Sample data GitHub release (P2, 4h) - Low priority
10. Golden fixture tests (P3, 8h) - Nice to have

---

## Quick Status Lookup

### ✅ COMPLETED (Since v1.0.0)
- ADR-019 Backend Registry Integration (PR #906)
- CI Coverage Enforcement (PR #832)
- Security Scanning (Bandit, pip-audit, gitleaks, CodeQL, Safety)
- Contract Validation (ingest_contract_validation.yml)
- Nightly Regression Suite (nightly.yml)
- HuggingFace model revision pinning (config/model_lock_manifest.yaml)
- Rollback procedures (docs/operations/ROLLBACK_PROCEDURES.md)
- ICC profile preservation in 16-bit TIFF

### 🟢 CORRECT (No Action Required)
- 25 NotImplementedError instances (abstract methods, phase gates, limitations)
- 4 Code TODOs (observational performance tracking)
- 2 Documentation TODOs (status markers)
- 2 Test patterns (dependency-conditional skips)

### 📦 ARCHIVED (Cleanup Complete)
- ~~`depth_canonical/` module~~ ✅ Removed from src/
- ~~`context_aware_rendering.py`~~ ✅ archive/scripts/
- ~~`lux_render_pipeline_plus_v3.py`~~ ✅ archive/scripts/pipelines/
- PR #98 action items → Historical reference
- Binary cleanup TODOs → Historical reference

### ⏳ PENDING (Action Required)
- Branch protection verification (P1, GitHub Admin task)
- CLI e2e test suite (P2, deferred to v2.4.0)
- SAM2 auto mode (P2, roadmap item)
- Sample data upload (P2, low priority)

### 🔴 DEFERRED (No Action for v2.x)
- Staging environment (P3, LOW ROI)
- ADR consistency checks (P3)
- Backend CLI commands (P4)
- ~~Parallax occlusion mapping (P5)~~ ✅ Archived

---

## Priority Reference

| Priority | Count | Definition | Examples |
|----------|-------|------------|----------|
| **P0** | 3 (DONE) | Critical, release blocker | ADR-019, Coverage, Security |
| **P1** | 1 | High, complete before next release | Branch protection verification |
| **P2** | 3 | Medium, schedule for v2.4.0 | SAM2, CLI e2e tests, samples |
| **P3** | 8 | Low, nice-to-have | ADR checks, SLERP, golden fixtures |
| **P4** | 7 | Enhancement, user request driven | Backend CLI, MaterialGAN |
| **P5** | 0 | Research, defer indefinitely | (All archived) |

---

## Critical Discrepancies

### ✅ Finding #1: V2.0.0 Checklist Mismatch (RESOLVED)
- **Issue:** Release shipped without all "required" items
- **Severity:** MEDIUM (process issue)
- **Action:** ✅ Checklists updated to be realistic

### ✅ Finding #2: Inventory Scope Growth (RESOLVED)
- **Issue:** 36 → 65 items in 8 days
- **Severity:** LOW (improved discovery)
- **Action:** ✅ Monthly review cadence established

### ✅ Finding #3: Obsolete Module Accumulation (RESOLVED)
- **Issue:** 3 modules unused (depth_canonical, etc.)
- **Severity:** LOW (cleanup)
- **Action:** ✅ All archived in Sprint 1

### ✅ Finding #4: CI/CD Maturity Acceleration
- **Achievement:** 4 of 6 gaps closed in 8 days
- **Impact:** POSITIVE (repository hardened)

### 🟡 Finding #5: Sample Data URLs Still TODO
- **Issue:** download_samples.py URLs not set
- **Severity:** LOW (UX improvement)
- **Action:** GitHub Release - defer to v2.4.0 (low priority)

---

## Governance Changes

### Definition of Done Updates
- ✅ Categorize NotImplementedError when added
- ✅ Archive superseded modules within 1 release
- ✅ Validate release checklists (not aspirational)

### Maintenance Cadence
- **Frequency:** Monthly or per-release
- **Owner:** Architect + Specialist
- **Duration:** 1 hour

### Stub Classification Standard
All NotImplementedError must include context:
- Feature gate: `"RAW format is Phase II (ADR-027)"`
- Abstract method: `"Subclass must implement"`
- Platform limitation: `"Windows not supported"`
- Technical debt: `"TODO: Implement NVDIFFREC"`

---

## v2.3.0 Success Criteria

### Must Have ✅
- [x] Rollback procedures documented ✅ (docs/operations/ROLLBACK_PROCEDURES.md)
- [ ] Branch protection verified (GitHub Admin task)
- [x] depth_canonical archived ✅
- [x] Dependency pinning enforced ✅ (config/model_lock_manifest.yaml)

### Should Have (v2.4.0)
- [ ] CLI e2e tests passing (deferred)
- [ ] SAM2 auto mode implemented (roadmap)
- [ ] Sample data GitHub Release (low priority)

### Nice to Have
- [ ] Golden fixture baseline (defer ok)
- [ ] ADR consistency checks (defer ok)
- [ ] Staging environment (defer ok)

---

## Common Queries

**Q: How many items are actual blockers?**
A: 0 P0 blockers. 1 P1 item (branch protection - GitHub Admin task).

**Q: How many items are technical debt?**
A: Of the 25 NotImplementedError instances, only 3 are technical debt requiring future work (NVDIFFREC, MaterialGAN, PBRFusion). The rest are intentional stubs (abstract methods, phase gates, platform limitations).

**Q: When should we update the inventory?**
A: Monthly or per-release. This review: 2026-03-15 (v2.3.0 planning).

**Q: Can we release v2.3.0 with pending items?**
A: Yes. All P1 items except branch protection are complete.

**Q: What's the biggest risk?**
A: None identified. Repository health is excellent.

---

## Contact & Escalation

**Inventory Owner:** Transformation Portal Architect
**Updates:** Specialist (execution) + Architect (review)
**Escalation:** Per agent_governance.md (Architect has final authority)
**Review Frequency:** Monthly or per-release

---

**Last Updated:** 2026-03-15
**Next Review:** April 2026 (v2.4.0 planning)
**Authority:** Binding under architectural governance

---

**Quick Links:**
- [Full Inventory](../../analysis/TODO_INVENTORY.md) (1,798 lines, comprehensive)
- [Executive Summary](TODO_INVENTORY_EXECUTIVE_SUMMARY.md) (2 pages, key findings)
- [Agent Governance](agent_governance.md) (escalation protocol)
- [CHANGELOG.md](../../CHANGELOG.md) (what's been completed)
- [CONTRIBUTING.md](../../CONTRIBUTING.md) (process updates pending)
