# Development Roadmap: Q2 2026

**Status:** ACTIVE
**Version:** 1.4.1
**Date:** 2026-03-26
**Authority:** Architect Assessment
**Supersedes:** v1.3.0 (2026-03-23), v1.2.0 (2026-03-22), v1.1.0 (2026-03-22), v1.0.0 (2026-03-20)

---

## Executive Summary

This roadmap identifies the **highest-signal development priorities** for the Transformation Portal.

**Sources:**
- Q1 2026 Codebase Audit (7.6/10 overall score)
- TODO Inventory v2.2.0 (65 items, 3% action required)
- Portal Orchestrator Roadmap (Phases 2-7 complete)
- ADR-043 Orchestrator Decomposition (complete)
- ADR-044 Test Marker Enforcement (implemented, validation pending)
- Current quality-control workflow review (`build.yml`, `ci.yml`, `ci-quality-firewall.yml`, `quality-gate.yml`)
- Curated web-stack compatibility merge (`#1278`)

**Goal:** Raise overall codebase health from **7.6/10 → 8.6/10** by end of Q2 2026.

---

## Document Structure

This roadmap is organized by **delivery state** to ensure governance clarity:

| Section | Purpose |
|---------|---------|
| **Completed / Implemented Since Q1 Audit** | Work landed and may have minor validation or governance follow-through pending |
| **Carry-Forward Work** | Work partially landed, requiring normalization or validation |
| **Q2 Commitments** | Net-new or open work that must complete this quarter |
| **Deferred / Post-Q2** | Important but intentionally out of current scope |

---

## Priority and Status Definitions

### Priority (Sequencing Order)

| Priority | Definition |
|----------|------------|
| **Now** | Currently active or prerequisite for downstream work; must start immediately |
| **Next** | Queued for work once Now items unblock |
| **Later** | Important, scheduled for later in quarter or post-Q2 |

### Status (Delivery State)

| Status | Definition |
|--------|------------|
| **Open** | Work not yet started |
| **Partial** | Code/config landed, but normalization or validation incomplete |
| **Implemented** | Core work landed, awaiting verification |
| **Verified** | Behavior and targets proven in CI |
| **Complete** | All gates passed; docs synchronized |

---

## Completed / Implemented Since Q1 Audit

Work in this section has landed and is no longer a primary implementation track, though some items still have **validation or governance follow-through pending**.

### 1. Orchestrator Decomposition (ADR-043)

| Gate | Status | Evidence |
|------|--------|----------|
| Decision | ✅ Complete | ADR-043 approved |
| Implementation | ✅ Complete | Phases 2–7 landed |
| Validation | ✅ Complete | 180+ unit tests, integration tests pass |
| Governance | ✅ Complete | ADR-043 updated to COMPLETE status |

**Outcome:**
- Extracted modules: `validators/run_card_validator.py`, `artifact_manager.py`, `config_resolver.py`, `pipeline_coordinator.py`, `execution_engine.py`
- Total extracted LOC: ~2,720
- `orchestrator.py` remains at 5,676 LOC under a documented state-machine exception in ADR-043

### 2. Test Marker Retrofit (ADR-044)

| Gate | Status | Evidence |
|------|--------|----------|
| Decision | ✅ Complete | ADR-044 approved |
| Implementation | ✅ Complete | 100% marker coverage achieved (was 48.6%) |
| Validation | ✅ Complete | Pre-commit works; positive marker selection deployed (2026-03-23) |
| Governance | ✅ Complete | ADR-044 updated to IMPLEMENTED status |

**Outcome:**
- 137 files tagged with appropriate markers
- Pre-commit hook blocks unmarked new tests
- `check_test_markers.py --audit` validates coverage
- All quality-control workflows (build.yml, ci.yml, ci-quality-firewall.yml) migrated to positive marker selection

### 2A. Exact-Pinned Web Stack Compatibility Update

| Gate | Status | Evidence |
|------|--------|----------|
| Decision | ✅ Complete | Exact-pinned web stack treated as curated compatibility work, not routine Dependabot merge |
| Implementation | ✅ Complete | `pyproject.toml`, `requirements/base.in`, `requirements/base.txt`, `requirements/all.txt` updated |
| Validation | ✅ Complete | `make test-orchestrator-contract`, `make ci`, curated live orchestrator smoke |
| Governance | ✅ Complete | `#1275` closed; issue `#1277` and PR `#1278` documented and merged |

**Outcome:**
- Current validated runtime set: FastAPI `0.135.1`, Starlette `1.0.0`, Uvicorn `0.42.0`
- Invalid cross-platform ML lock regeneration was explicitly removed from the curated PR before merge
- Separate packaging follow-up is tracked in issue `#1279`
- Future exact-pin web-stack upgrades should follow the same issue-first / PR-second governance path

---

## Carry-Forward Work

Work in this section is **partially landed** and requires completion or normalization.

### 3. Quality Control Plane Canonicalization

**Priority:** Now
**Status:** Complete

**Problem Statement:**
The repository already encodes a **provisional canonical decision**: `ci.yml` explicitly states that pull-request gating lives in `build.yml`, while `ci.yml` serves post-merge validation. The quality-control plane normalization is now largely complete:
- ~~`build.yml` is the de facto PR gate and still uses legacy negative marker expressions~~ → ✅ Fixed (positive marker selection as of 2026-03-23)
- `ci.yml` and `ci-quality-firewall.yml` both run post-merge/post-CI validation with overlapping quality semantics
- `quality-gate.yml` remains an older helper workflow with its own quality behavior
- ~~Version-tag action refs are still present in `ci-quality-firewall.yml` and `quality-gate.yml`~~ → ✅ Fixed (SHA-pinned as of 2026-03-22)
- `-n auto` is enabled for pytest in `ci.yml`, but remains absent in `build.yml` and `ci-quality-firewall.yml`
- ~~Typecheck policy varies across the plane: hard-fail in `ci.yml` (critical modules), soft-fail in `ci-quality-firewall.yml`, absent in `build.yml` and `quality-gate.yml`~~ → ✅ `build.yml` now has hard-fail mypy (2026-03-23)

**Quality Control Plane Inventory:**

| Workflow | Trigger | Current Role | Branch-Protection Relevance | Marker Semantics | Typecheck Policy | Action Ref Style | Target Disposition |
|----------|---------|--------------|-----------------------------|------------------|------------------|------------------|-------------------|
| `build.yml` | PR, push, dispatch | De facto PR gate | Yes | ✅ Positive selection | ✅ Hard-fail mypy | SHA-pinned | Canonical |
| `ci.yml` | push | Post-merge validation | No | ✅ Positive selection | Hard-fail mypy (critical modules) | SHA-pinned | Align with canonical or narrow scope |
| `ci-quality-firewall.yml` | `workflow_run` | Post-CI verification | No | ✅ Positive selection | Soft-fail mypy | ✅ SHA-pinned | Align, narrow, or retire |
| `quality-gate.yml` | PR, push | Legacy helper workflow | Non-canonical / ambiguous | N/A | N/A | ✅ SHA-pinned | Retire or scope down |

**Scope note:** Workflows outside the quality-control plane (docs, security, nightly, deployment, automation) intentionally differ and are excluded from parity debt.

**Objective:**
Establish one canonical PR gating workflow (`build.yml`) with explicit policy, then align or retire conflicting quality-control workflows.

**Scope includes:**
- Workflow selection and role clarification
- Marker semantics normalization (negative → positive where policy requires) ✅ (2026-03-23)
- Runtime validation targets for canonical CI
- Documentation alignment (`CONTRIBUTING.md`, `docs/testing/STRATEGY.md`, workflow docs) ✅
- Action reference normalization within the quality-control plane ✅
- Typecheck policy normalization for branch-protection-relevant quality workflows ✅

**Target Outcome:**
- `build.yml` formally designated as canonical for branch protection ✅ (documented)
- All quality-control workflows either match canonical enforcement semantics or are retired/scoped down
- Marker semantics moved toward explicit positive selection for intended fast paths ✅ (2026-03-23)
- Runtime targets measured against canonical workflow only
- Quality-control workflow roles documented and non-overlapping ✅

**Acceptance Gates:**

| Gate | Criteria | Status |
|------|----------|--------|
| Decision | Formal control-plane inventory approved; canonical workflow and role boundaries documented | ✅ Complete |
| Implementation | Workflow alignment changes merged; obsolete workflows retired or narrowed; action refs normalized; typecheck normalized; marker selection normalized | ✅ Complete (2026-03-23) |
| Validation | Canonical CI runtime targets measured and reported | ⏳ Pending |
| Governance | Workflow documentation, `CONTRIBUTING.md`, and testing strategy aligned | ✅ Complete |

**Effort:** 12–16 hours
**Owner:** Architect + DevOps

---

### 4. Orchestrator Residual Slimming & Boundary Enforcement

**Priority:** Now
**Status:** Partial

**Problem Statement:**
ADR-043 is complete, but `orchestrator.py` still carries a large residual facade. The remaining risk is no longer decomposition feasibility; it is **boundary drift**:
- feature logic can creep back into the orchestrator,
- extracted module seams can erode over time,
- and maintainability gains can reverse if residual helpers are not ratcheted down.

**Objective:**
Prevent new feature logic from accumulating in `orchestrator.py` and continue moving residual helpers behind stable module boundaries.

**Acceptance Gates:**

| Gate | Criteria |
|------|----------|
| Decision | ✅ ADR-043 documents target architecture |
| Implementation | Residual helper logic continues to move behind extracted boundaries |
| Validation | Import graph remains acyclic; delegation/contract tests increase; no new responsibility domains appear |
| Governance | ADR-043 updated if boundary policy or state-machine exception changes |

**Primary indicators (blocking):**
1. No new responsibility domains added
2. No new feature logic added directly to `orchestrator.py`
3. Import graph remains acyclic
4. Delegation contract tests increase
5. Module boundaries remain stable

**Secondary indicator (directional):**
Reduce `orchestrator.py` by **~200 LOC per quarter** or document why the ratchet is infeasible.

**Target list:** Specific seam candidates for `orchestrator.py` (and other monoliths) live in [MONOLITH_DECOMPOSITION_TARGETS.md](MONOLITH_DECOMPOSITION_TARGETS.md); pattern is governed by [ADR-045](ADR-045-monolith-decomposition-residuals.md).

**Effort:** 4–8 hours ongoing per quarter
**Owner:** Architect (review) + Specialist (implementation)

---

## Q2 Commitments

Work in this section is **net-new or still open** and must complete this quarter.

### 5. Governance Synchronization

**Priority:** Now
**Status:** ✅ Complete (2026-03-23)

**Problem Statement:**
Policy and enforcement changes have landed, but supporting governance documents still encode stale assumptions. Current examples include:
- ~~`CONTRIBUTING.md` describing mypy as non-blocking~~ → ✅ Fixed (now documents hard-fail for ci.yml)
- ~~`CONTRIBUTING.md` describing an outdated coverage baseline~~ → ✅ Fixed (now shows 25.44%)
- ~~workflow control-plane semantics not yet documented~~ → ✅ Fixed (added CI/CD Control Plane section)
- ~~`docs/testing/STRATEGY.md` missing canonical workflow info~~ → ✅ Fixed (added Canonical Workflow section)
- documentation still teaches negative marker selection (intentional: current state; transition pending)

**Objective:**
Align all binding and operational documents with the current enforced state and the intended Q2 target state.

**Scope:**
- This roadmap
- `CONTRIBUTING.md` ✅
- `docs/testing/STRATEGY.md` ✅
- ADR index / status pages
- CI / workflow documentation

**Rule:**
No policy change is considered **Complete** until the corresponding governance docs are updated.

**Review Cadence and Update Triggers:**
- **Monthly review:** End of each sprint (see document footer)
- **Immediate update triggers:**
  - Any CI workflow policy change (marker selection, typecheck blocking, action refs)
  - Coverage threshold changes
  - New ADR acceptance affecting CI or testing
- **Owner responsibility:** Architect maintains binding docs within 1-week SLA of policy change

**Acceptance Gates:**

| Gate | Criteria | Status |
|------|----------|--------|
| Decision | Governance document scope and owners confirmed | ✅ Complete |
| Implementation | Documents updated | ✅ Complete (2026-03-23) |
| Validation | No contradictions remain between enforcement and documentation | ✅ Complete |
| Governance | Review cadence and update trigger documented | ✅ Complete |

**Effort:** 4–6 hours
**Owner:** Architect

---

### 6. Coverage Ramp Phase 1

**Priority:** Next
**Status:** Open

**Current Coverage:** 25.44% (6,314 of 24,820 statements)
**Phase 1 Target:** 28% (+636 statements)

**Phase 1 Focus Modules (high ROI):**

| Module | Statements | Current | Target |
|--------|-----------:|--------:|-------:|
| `cli/__init__.py` | ~250 | 0% | 60% |
| `config_loader.py` | 148 | 40% | 80% |
| `utils/input_validation.py` | 195 | 0% | 70% |
| `utils/recipe_validator.py` | 63 | 0% | 80% |

**Dependency:**
Benefits from Quality Control Plane Canonicalization for efficient test execution and clean runtime measurement.

**Acceptance Gates:**

| Gate | Criteria |
|------|----------|
| Decision | Target modules confirmed |
| Implementation | Tests added; overall coverage reaches 28% |
| Validation | `pytest-cov` reports target met |
| Governance | Coverage docs updated |

**Effort:** 15–20 hours
**Owner:** Specialist
**Milestone:** v2.4.0

---

## Deferred / Post-Q2

Work in this section is **important but intentionally out of current scope**.

### 7. Documentation Consolidation

**Priority:** Later
**Status:** Open

**Current State:**
- 719 markdown files across 99 directories
- Parallel ADR hierarchies
- 83% missing "Last Updated" metadata

**Target State:**
- ~200 canonical documents across ~30 directories
- Single ADR location with archive for superseded items
- All canonical docs carry metadata headers

**Rationale for Deferral:**
Best done after execution controls and governance surfaces are normalized.

**Effort:** 15–20 hours
**Owner:** Specialist
**Milestone:** v2.5.0

---

### 8. Circular Import Contract Hardening

**Priority:** Later
**Status:** Open

**Current State:**
Mitigated via `TYPE_CHECKING` guards across 8 files in `depth/backends/`.

**Target State:**
- Shared contracts in `lux_depth_v3/_contracts.py`
- Documented import-boundary pattern in the architecture guide
- CI check for new cross-module imports

**Rationale for Deferral:**
Best done after orchestrator boundaries and workflow governance are stabilized.

**Effort:** 8–12 hours
**Owner:** Architect
**Milestone:** v2.5.0

---

## Q2 Priority Stack

| Priority | Workstream | Status | Rationale |
|----------|------------|--------|-----------|
| Now | Quality Control Plane Canonicalization | ✅ Complete (action refs ✅, docs ✅, typecheck ✅, marker selection ✅) | CI ambiguity resolved; all QC workflows use positive marker selection |
| Now | Orchestrator Residual Slimming & Boundary Enforcement | Partial | Core maintainability risk remains active |
| Now | Governance Synchronization | ✅ Complete | Required for policy integrity |
| Next | Coverage Ramp Phase 1 | Open | High ROI once CI targeting is stable |
| Later | Documentation Consolidation | Open | Valuable, but not before execution controls are clean |
| Later | Circular Import Contract Hardening | Open | Important follow-on once boundaries stabilize |

---

## Leading Indicator Metrics

These metrics enable proactive steering rather than quarter-end audit.

### Workflow Parity Debt (Quality Control Plane Only)

**Scope:** `build.yml`, `ci.yml`, `ci-quality-firewall.yml`, `quality-gate.yml`

**Definition:**
A scoped workflow counts toward **Workflow Parity Debt** if it has at least one unresolved parity defect relative to the approved control-plane target state.

| Metric | Current | Target |
|--------|---------|--------|
| Quality-control workflows (scoped total) | 4 | N/A (informational) |
| QC workflows still using legacy negative marker selection | 0 | 0 |
| QC workflows with version-tag or mixed action refs | 0 | 0 |
| QC workflows with unresolved typecheck policy | 0 | 0 |
| QC workflows with divergent coverage / artifact behavior | Baseline by 2026-04-05 | 0 |

**Metric derivation (updated 2026-03-23):**
- **Marker selection (0):** ✅ All quality-control workflows now use positive marker selection (build.yml, ci.yml, ci-quality-firewall.yml); `quality-gate.yml` has no test jobs
- **Version-tag refs (0):** ✅ All quality-control workflows now use SHA-pinned action refs
- **Typecheck policy (0):** ✅ Normalized. "Unresolved" means policy not aligned with workflow's intended role:
  - `build.yml`: Hard-fail mypy (PR blocking gate) — **aligned**
  - `ci.yml`: Hard-fail mypy (post-merge validation) — **aligned**
  - `ci-quality-firewall.yml`: Soft-fail mypy — **intentionally different** (advisory workflow, not branch-protection relevant; soft-fail is appropriate for its role)
  - `quality-gate.yml`: No typecheck job — **N/A** (pre-commit style checks only)

  The count is 0 because all workflows have typecheck policies appropriate to their roles. `ci-quality-firewall.yml` uses soft-fail intentionally as an advisory workflow.

**Composite Metric:**
**Workflow Parity Debt = number of scoped workflows with ≥1 unresolved parity defect.**
Current baseline: **0** (all parity defects resolved as of 2026-03-23). Target: **0** ✅.

### Orchestrator Residual Debt

| Metric | Current | Target |
|--------|---------|--------|
| `orchestrator.py` LOC | 5,676 (verified 2026-03-23) | 5,476 |
| Responsibility domains in facade | 5 (verified) | ≤5 (no new domains) |
| Delegation / contract tests across extracted boundaries | ~180 | +10/quarter |

**Responsibility domain inventory (verified 2026-03-23):**
1. Depth stage execution (per-image backend fallback, cache management)
2. V2 stage execution (subprocess coordination)
3. PBR generation (texture pipeline)
4. Materials V3 execution (APEX quality gates)
5. Artifact persistence (run card, Merkle roots)

**Import health:** No direct circular imports; module imports successfully when dependencies are present.

### Governance Freshness

**Baseline established during Governance Synchronization (2026-03-22), updated 2026-03-23.**

| Metric | Current | Target |
|--------|---------|--------|
| Binding docs updated within SLA after policy change | ✅ 100% (updated 2026-03-23) | 100% within 1 week |
| Stale documents contradicting current enforcement | 0 (verified 2026-03-23) | 0 |

**Binding-doc scope:**
This roadmap ✅, `CONTRIBUTING.md` ✅, `docs/testing/STRATEGY.md` ✅, ADR status pages, and CI / workflow documentation.

**Recent updates (2026-03-23):**
- Typecheck policy documented in CONTRIBUTING.md (blocking gate)
- Canonical workflow table updated in STRATEGY.md
- Roadmap v1.3.0 with metrics refresh

---

## Quarterly Success Metrics (Lagging Indicators)

**Note:**
The Q1 2026 column reflects audit-time state (2026-03-20). The **Current** column reflects the state as of 2026-03-23.

| Metric | Q1 2026 (Audit) | Current | Q2 Target | Measurement |
|--------|-----------------|---------|-----------|-------------|
| Overall Score | 7.6/10 | 7.6/10 | 8.6/10 | Codebase audit |
| `orchestrator.py` LOC | 6,108 | 5,676 | 5,476 | `wc -l orchestrator.py` |
| Test marker coverage | 48.6% | 100% | 100% maintained | `check_test_markers.py --audit` |
| Code coverage | 25.44% | 25.44% | 28% | `pytest-cov` |
| Canonical CI time (`build.yml`) | 65–75 min | TBD | 40–50 min | GitHub Actions |
| Workflow Parity Debt (QC) | N/A | 0 (all parity defects resolved) | 0 | Quality-control workflows only |

---

## ADR Status

| ADR | Topic | Status |
|-----|-------|--------|
| ADR-043 | Orchestrator Decomposition | ✅ COMPLETE |
| ADR-044 | Test Marker Enforcement Policy | ✅ IMPLEMENTED (positive marker selection deployed 2026-03-23) |
| ADR-045 | CI/CD & Workflow Semantics | Required if Q2 changes branch-protection semantics, canonical workflow designation, marker taxonomy, or blocking / non-blocking policy |

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Quality control plane canonicalization disrupts existing CI | Low | Medium | Test changes in a separate workflow first; promote incrementally |
| Orchestrator responsibility creep | Medium | Medium | Enforce residual ratchet and ADR-043 review gate |
| Governance docs drift from enforcement | Medium | Low | Governance Synchronization workstream |
| Canonical CI runtime targets are not achieved | Low | Low | Measure early and adjust workflow topology or marker strategy |
| Retiring or narrowing `quality-gate.yml` breaks incidental tooling expectations | Low | Low | Document deprecation path and verify downstream dependencies before removal |

---

## Related Documents

- [Q1 2026 Codebase Audit](CODEBASE_AUDIT_2026_Q1.md)
- [TODO Inventory Quick Reference](TODO_INVENTORY_QUICK_REF.md)
- [Portal Orchestrator Roadmap](PORTAL_ORCHESTRATOR_ROADMAP.md)
- [ADR-043 Orchestrator Decomposition](ADR-043-orchestrator-decomposition.md)
- [ADR-044 Test Marker Enforcement](ADR-044-test-marker-enforcement.md)
- [Testing Strategy](../testing/STRATEGY.md)
- [Agent Governance](agent_governance.md)

---

**Document Owner:** Transformation Portal Architect
**Review Cadence:** Monthly (end of each sprint)
**Next Review:** April 2026

---

*This document is binding under architectural governance. Deviations require explicit Architect approval.*
