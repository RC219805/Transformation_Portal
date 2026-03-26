# TODO Inventory Quick Reference

**Version:** 2.4.1 | **Date:** 2026-03-26 | **Status:** ACTIVE

**Full Inventory:** [docs/analysis/TODO_INVENTORY.md](../analysis/TODO_INVENTORY.md)
**Action Plan:** [docs/analysis/TODO_ACTION_PLAN.md](../analysis/TODO_ACTION_PLAN.md)

---

## At A Glance

| Metric | Value |
|--------|-------|
| **Source Code TODOs (`src/`)** | 0 ✅ |
| **Test TODOs** | 3 (observational) |
| **NotImplementedError** | 12 (all intentional) |
| **P0 Blockers** | 0 |
| **P1 Pending** | 1 (branch protection verification) |
| **P2 Pending** | 3 (sample uploads) |
| **P3 Deferred** | 3 (ComfyUI, NVDIFFREC, MaterialGAN) |
| **Repository Health** | **EXCELLENT** ✅ |

**Recently Completed (2026-03-26):**
- Source code TODOs: All cleaned up ✅
- Quality Firewall docs: Created ✅
- SLERP interpolation: Implemented ✅
- ICC profile preservation: Implemented ✅
- depth_canonical module: Archived ✅
- P2: context_aware_rendering.py archived ✅
- P2: lux_render_pipeline_plus_v3.py archived ✅
- Curated web-stack compatibility update merged (FastAPI 0.135.1 / Starlette 1.0.0 / Uvicorn 0.42.0) ✅

---

## Immediate Actions (v2.4.0+ Planning)

**Total Effort:** ~4 hours
**Last Updated:** 2026-03-25

| # | Task | Priority | Effort | Owner | Status |
|---|------|----------|--------|-------|--------|
| 1 | Verify branch protection | P1 | 15min | Admin | Pending (GitHub Admin) |
| 2 | Upload sample data to GitHub Release | P2 | 4h | DevOps | Pending |
| 3 | ComfyUI subprocess integration | P3 | 1-2w | ML | Deferred |
| 4 | NVDIFFREC integration | P3 | 3-4w | ML Research | Deferred |
| 5 | MaterialGAN integration | P3 | 2-3w | ML Research | Deferred |

---

## Sprint Planning

### ✅ Completed (v2.1.0 - v2.4.0)
- ~~Delete depth_canonical module~~ ✅ Archived
- ~~Archive obsolete PR docs~~ ✅ Completed 2026-03-14
- ~~Dependency pinning validation~~ ✅ HuggingFace revisions pinned (2026-03-14)
- ~~SLERP interpolation~~ ✅ Completed 2026-03-16
- ~~ICC profile preservation~~ ✅ Completed 2026-03-14
- ~~Source code TODO cleanup~~ ✅ All cleaned up 2026-03-25

### Deferred to v2.5.0+
- SAM2 auto mask generation (P3)
- Sample data GitHub release (P2, 4h)
- Golden fixture tests (P3, 8h)

---

## Quick Status Lookup

### ✅ COMPLETED (Since v1.0.0)
- ADR-019 Backend Registry Integration (PR #906)
- CI Coverage Enforcement (PR #832)
- Security Scanning (Bandit, pip-audit, gitleaks, CodeQL)
- Contract Validation (ingest_contract_validation.yml)
- Nightly Regression Suite (nightly.yml)
- HuggingFace model revision pinning (config/model_lock_manifest.yaml)
- Rollback procedures (docs/operations/ROLLBACK_PROCEDURES.md)
- ICC profile preservation in 16-bit TIFF
- SLERP interpolation (scene_builder.py)
- All source code TODOs cleaned up
- Curated Starlette 1.0 compatibility validation and merge (#1278)

### 🟢 CORRECT (No Action Required)
- 12 NotImplementedError instances (abstract methods, phase gates, limitations)
- 3 Test TODOs (observational performance tracking)
- 2 Security patterns (TODO_REPLACE regex matchers)

### 📦 ARCHIVED (Cleanup Complete)
- ~~`depth_canonical/` module~~ ✅ Removed from src/
- ~~`context_aware_rendering.py`~~ ✅ archive/scripts/
- ~~`lux_render_pipeline_plus_v3.py`~~ ✅ archive/scripts/pipelines/
- PR #98 action items → docs/_archive/2026-03-legacy-prs/
- Binary cleanup TODOs → Historical reference

### ⏳ PENDING (Action Required)
- Branch protection verification (P1, GitHub Admin task)
- Sample data upload (P2, low priority)
- ML lock generation trust model (`#1279`, packaging / requirements follow-up)

### 🔴 DEFERRED (No Action for v2.x)
- ComfyUI subprocess integration (P3)
- NVDIFFREC integration (P3)
- MaterialGAN integration (P3)

---

## Priority Reference

| Priority | Count | Definition | Examples |
|----------|-------|------------|----------|
| **P0** | 0 | Critical, release blocker | (All resolved) |
| **P1** | 1 | High, complete before next release | Branch protection verification |
| **P2** | 3 | Medium, schedule for v2.4.0+ | Sample uploads |
| **P3** | 3 | Low, deferred | ComfyUI, NVDIFFREC, MaterialGAN |
| **P4** | 1 | Enhancement, nice-to-have | Upscaling weights cache |

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
A: Of the 12 NotImplementedError instances, only 3 are technical debt requiring future work (NVDIFFREC, MaterialGAN, PBRFusion). The rest are intentional stubs (abstract methods, phase gates, platform limitations).

**Q: When should we update the inventory?**
A: Monthly or per-release. This review: 2026-03-25 (v2.5.0 planning).

**Q: Can we release v2.4.0 with pending items?**
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

**Last Updated:** 2026-03-26
**Next Review:** April 2026 (v2.5.0 planning)
**Authority:** Binding under architectural governance

---

**Quick Links:**
- [Full Inventory](../../analysis/TODO_INVENTORY.md) (1,798 lines, comprehensive)
- [Executive Summary](TODO_INVENTORY_EXECUTIVE_SUMMARY.md) (2 pages, key findings)
- [Agent Governance](agent_governance.md) (escalation protocol)
- [CHANGELOG.md](../../CHANGELOG.md) (what's been completed)
- [CONTRIBUTING.md](../../CONTRIBUTING.md) (process updates pending)
