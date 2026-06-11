# TODO Inventory Quick Reference

**Version:** 2.4.8 | **Date:** 2026-06-11 | **Status:** ACTIVE

**Full Inventory:** [docs/analysis/TODO_INVENTORY.md](../analysis/TODO_INVENTORY.md)
**Action Plan:** [docs/analysis/TODO_ACTION_PLAN.md](../analysis/TODO_ACTION_PLAN.md)

---

## At A Glance

| Metric | Value |
|--------|-------|
| **Source Code TODOs (`src/`)** | 0 ✅ |
| **Scanner-visible Test TODO comments** | 0 ✅ |
| **NotImplementedError** | 25 (all governed: abstract methods, phase gates, platform limits, fail-closed dispatch guards) |
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

**Recently Completed (2026-05-01):**
- TODO governance scanner wired into CI (`enforcement.yml` → `hf-revision-policy` job runs `scan_todo_inventory.py --check-governance`). Ungoverned TODOs now fail PR builds; replaces the manual monthly review cadence as the primary drift control. ✅
- Inventory recount: NotImplementedError baseline corrected from 12 → 25. All 25 instances were properly governed (abstract methods, phase gates, platform limits); the discrepancy was doc drift, not a code issue. ✅

**Recently Completed (2026-05-11):**
- TODO scanner can now refresh the tracked JSON snapshot directly with `python scripts/validation/scan_todo_inventory.py --write-snapshot`. ✅
- Inventory snapshot refreshed from live repo state: 24 governed `NotImplementedError` items, 0 ungoverned TODOs, and 1,570 files scanned. The 25 → 24 delta is from retired code, not a changed governance rule. ✅
- `docs/governance/todo_priority_schema.yaml` now records the current scanner baseline and snapshot command. ✅

**Recently Completed (2026-06-11):**
- Inventory snapshot refreshed from live repo state: 25 governed `NotImplementedError` items, 0 ungoverned TODOs, and 1,753 files scanned. The 24 → 25 delta is the governed ComfyUI unsupported-node executor guard, not an ungoverned backlog item. ✅
- `docs/analysis/TODO_INVENTORY.md`, `docs/analysis/TODO_ACTION_PLAN.md`, this quick reference, `docs/governance/todo_priority_schema.yaml`, `docs/README.md`, and the documentation map now agree on the live scanner baseline. ✅

---

## Immediate Actions (v2.4.0+ Planning)

**Total Effort:** ~4 hours
**Last Updated:** 2026-06-11

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
- 25 NotImplementedError instances (abstract methods, phase gates, limitations, fail-closed dispatch guards)
- 0 scanner-visible test TODO comments
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
- Technical debt: `"Single-view reconstruction not yet implemented (TODO_INVENTORY.md)"`

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
A: Of the 25 `NotImplementedError` instances the scanner reports, the majority are intentional stubs (abstract methods, bare-raise scaffolds, test protocol stubs, platform limitations, and fail-closed dispatch guards). A handful are actionable phase gates / known gaps (e.g., single-view 3D reconstruction in `spatial_ai/orchestration/pipeline.py`, GaussianBackend pending checkpoint integration, non-linear interpolation in `scene_builder.py`). The canonical, line-accurate list is the scanner output — `python scripts/validation/scan_todo_inventory.py --json` — not a hand-maintained name list (which is what caused prior drift). Note: NVDIFFREC and MaterialGAN are *not* `NotImplementedError` raises; those backends fall back to the heuristic generator at `material_backend.py` and are tracked separately under research-deferred work.

**Q: When should we update the inventory?**
A: CI now enforces `--check-governance` on every PR via `enforcement.yml`, so ungoverned TODOs cannot land. Manual narrative refresh (counts, completion logs) still occurs monthly or per-release. Refresh the tracked scanner snapshot with `python scripts/validation/scan_todo_inventory.py --write-snapshot`. This review: 2026-06-11.

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

**Last Updated:** 2026-06-11
**Next Review:** June 2026 (v2.5.0 planning); CI now enforces ungoverned-TODO blocks on every PR
**Authority:** Binding under architectural governance

---

**Quick Links:**
- [Full Inventory](../analysis/TODO_INVENTORY.md) (current scanner-governed inventory)
- [Agent Governance](agent_governance.md) (escalation protocol)
- [Monolith Decomposition Targets](MONOLITH_DECOMPOSITION_TARGETS.md) — ranked seam list (companion to [ADR-045](ADR-045-monolith-decomposition-residuals.md))
- [CHANGELOG.md](../../CHANGELOG.md) (what's been completed)
- [CONTRIBUTING.md](../../CONTRIBUTING.md) (process updates pending)
