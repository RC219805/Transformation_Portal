# TODO Inventory Quick Reference

**Version:** 2.0.1 | **Date:** 2026-03-14 | **Status:** ACTIVE

**Full Inventory:** [docs/analysis/TODO_INVENTORY.md](../../analysis/TODO_INVENTORY.md)
**Executive Summary:** [TODO_INVENTORY_EXECUTIVE_SUMMARY.md](TODO_INVENTORY_EXECUTIVE_SUMMARY.md)

---

## At A Glance

| Metric | Value |
|--------|-------|
| **Total Items** | 65 |
| **✅ Completed** | 12 (18%) |
| **🟢 Correct (No Action)** | 32 (49%) |
| **📦 Obsolete** | 17 (26%) |
| **⏳ Action Required** | 21 (32%) |
| **P0 Blockers** | 0 |
| **Repository Health** | **EXCELLENT** ✅ |

**Recently Completed (2026-03-14):**
- P1: HuggingFace model revision pinning
- P2: ICC profile preservation in 16-bit TIFF
- P2: depth_canonical example archived

---

## Immediate Actions (By March 1, 2026)

**Total Effort:** 4 hours → **Remaining:** 45 min
**Owner:** Architect + DevOps
**Last Updated:** 2026-03-14

| # | Task | Priority | Effort | Owner | Status |
|---|------|----------|--------|-------|--------|
| 1 | Create rollback procedures | P1 | 2h | Architect | ✅ Already exists |
| 2 | Verify branch protection | P1 | 15min | Admin | Pending |
| 3 | Update V2_0_0_RELEASE_REVIEW.md | P1 | 30min | Architect | Pending |
| 4 | Archive obsolete modules | P2 | 1h | Specialist | ✅ Already done |

---

## Sprint Planning

### Sprint 1: Cleanup (Week 1, 8h) - **Mostly Completed**
1. ~~Delete depth_canonical module (1h)~~ ✅ Already archived
2. ~~Archive obsolete PR docs (30min)~~ ✅ Completed 2026-03-14
3. Update binary cleanup docs (30min)
4. Complete CLI e2e tests (4h)
5. Audit ADR-023 manifest (2h)

### Sprint 2: Phase 2 Foundation (Week 2-3, 16h)
6. SAM2 auto mask generation (P2, 3-4 days)
7. ~~Dependency pinning validation (P2, 4h)~~ ✅ HuggingFace revisions pinned (2026-03-14)
8. Create V2_3_0 release checklist (P2, 2h)

### Sprint 3: Nice-to-Have (v2.3.0, 12h)
9. Sample data GitHub release (P2, 4h)
10. Golden fixture tests (P3, 8h)

---

## Quick Status Lookup

### ✅ COMPLETED (Since v1.0.0)
- ADR-019 Backend Registry Integration (PR #906)
- CI Coverage Enforcement (PR #832)
- Security Scanning (Bandit, pip-audit, gitleaks, CodeQL, Safety)
- Contract Validation (ingest_contract_validation.yml)
- Nightly Regression Suite (nightly.yml)

### 🟢 CORRECT (No Action Required)
- 25 NotImplementedError instances (abstract methods, phase gates, limitations)
- 4 Code TODOs (observational performance tracking)
- 2 Documentation TODOs (status markers)
- 2 Test patterns (dependency-conditional skips)

### 📦 OBSOLETE (Cleanup Required)
- `depth_canonical/` module → DELETE
- `context_aware_rendering.py` → ARCHIVE
- `lux_render_pipeline_plus_v3.py` → ARCHIVE
- PR #98 action items → ARCHIVE
- Binary cleanup TODOs → REMOVE

### ⏳ PENDING (Action Required)
- Rollback procedures documentation (P1)
- Branch protection verification (P1)
- CLI e2e test suite (P1)
- SAM2 auto mode (P2)
- Dependency pinning validation (P2)
- Sample data upload (P2)

### 🔴 DEFERRED (No Action for v2.x)
- Staging environment (P2, LOW ROI)
- ADR consistency checks (P3)
- Backend CLI commands (P4)
- Parallax occlusion mapping (P5, archive)

---

## Priority Reference

| Priority | Count | Definition | Examples |
|----------|-------|------------|----------|
| **P0** | 3 (DONE) | Critical, release blocker | ADR-019, Coverage, Security |
| **P1** | 3 | High, complete before next release | Rollback docs, CLI tests |
| **P2** | 6 | Medium, schedule for v2.3.0 | SAM2, dep pinning, samples |
| **P3** | 8 | Low, nice-to-have | ADR checks, SLERP, golden fixtures |
| **P4** | 7 | Enhancement, user request driven | Backend CLI, MaterialGAN |
| **P5** | 1 | Research, defer indefinitely | Parallax occlusion mapping |

---

## Critical Discrepancies

### 🔴 Finding #1: V2.0.0 Checklist Mismatch
- **Issue:** Release shipped without all "required" items
- **Severity:** MEDIUM (process issue)
- **Action:** Update checklists to be realistic

### 🟡 Finding #2: Inventory Scope Growth (+80%)
- **Issue:** 36 → 65 items in 8 days
- **Severity:** LOW (improved discovery)
- **Action:** Establish monthly review cadence

### 🟢 Finding #3: Obsolete Module Accumulation
- **Issue:** 3 modules unused (depth_canonical, etc.)
- **Severity:** LOW (cleanup)
- **Action:** Delete/archive in Sprint 1

### ✅ Finding #4: CI/CD Maturity Acceleration
- **Achievement:** 4 of 6 gaps closed in 8 days
- **Impact:** POSITIVE (repository hardened)

### 🟡 Finding #5: Sample Data URLs Still TODO
- **Issue:** download_samples.py URLs not set
- **Severity:** LOW (UX improvement)
- **Action:** GitHub Release by v2.3.0

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
- [x] Rollback procedures documented ✅ (already exists: docs/operations/ROLLBACK_PROCEDURES.md)
- [ ] Branch protection verified
- [x] depth_canonical deleted ✅ (already archived)
- [ ] CLI e2e tests passing
- [x] Dependency pinning enforced ✅ (HuggingFace revisions pinned 2026-03-14)

### Should Have ✅
- [ ] SAM2 auto mode implemented
- [ ] Sample data GitHub Release
- [ ] Golden fixture baseline

### Nice to Have
- [ ] ADR consistency checks (defer ok)
- [ ] Staging environment (defer ok)

---

## Common Queries

**Q: How many items are actual blockers?**
A: 0 P0 blockers. 3 P1 items for v2.3.0 (manageable).

**Q: How many items are technical debt?**
A: Only 3 NotImplementedError instances (10%). Rest are intentional stubs.

**Q: When should we update the inventory?**
A: Monthly or per-release. Next review: v2.3.0 planning (March 2026).

**Q: Can we release v2.3.0 with pending items?**
A: Yes, but complete P1 immediate actions first (4h total).

**Q: What's the biggest risk?**
A: None identified. Repository health is excellent.

---

## Contact & Escalation

**Inventory Owner:** Transformation Portal Architect
**Updates:** Specialist (execution) + Architect (review)
**Escalation:** Per agent_governance.md (Architect has final authority)
**Review Frequency:** Monthly or per-release

---

**Last Updated:** 2026-02-13
**Next Review:** March 2026 (v2.3.0 planning)
**Authority:** Binding under architectural governance

---

**Quick Links:**
- [Full Inventory](../../analysis/TODO_INVENTORY.md) (1,798 lines, comprehensive)
- [Executive Summary](TODO_INVENTORY_EXECUTIVE_SUMMARY.md) (2 pages, key findings)
- [Agent Governance](agent_governance.md) (escalation protocol)
- [CHANGELOG.md](../../CHANGELOG.md) (what's been completed)
- [CONTRIBUTING.md](../../CONTRIBUTING.md) (process updates pending)
