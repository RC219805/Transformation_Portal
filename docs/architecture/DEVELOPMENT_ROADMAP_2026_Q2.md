# Development Roadmap: Q2 2026

**Status:** ACTIVE
**Version:** 1.1.0
**Date:** 2026-03-22
**Authority:** Architect Assessment
**Supersedes:** v1.0.0 (2026-03-20)

---

## Executive Summary

This roadmap identifies the **highest-signal development priorities** for the Transformation Portal.

**Sources:**
- Q1 2026 Codebase Audit (7.6/10 overall score)
- TODO Inventory v2.2.0 (65 items, 3% action required)
- Portal Orchestrator Roadmap (Phases 2-7 complete)
- ADR-043 Orchestrator Decomposition (complete)
- ADR-044 Test Marker Enforcement (implemented, validation pending)

**Goal:** Raise overall codebase health from **7.6/10 → 8.6/10** by end of Q2 2026.

---

## Document Structure

This roadmap is organized by **delivery state** to ensure governance clarity:

| Section | Purpose |
|---------|---------|
| **Completed Since Q1 Audit** | Work landed and no longer part of active execution |
| **Carry-Forward Work** | Work partially landed, requiring normalization or validation |
| **Q2 Commitments** | Net-new or open work that must complete this quarter |
| **Deferred / Post-Q2** | Important but intentionally out of current scope |

---

## Priority and Status Definitions

### Priority (Sequencing Order)

| Priority | Definition |
|----------|------------|
| **Now** | Currently active or blocked by; must start immediately |
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

## Completed Since Q1 Audit

Work in this section is **done** and no longer requires active tracking.

### Orchestrator Decomposition (ADR-043)

| Gate | Status | Evidence |
|------|--------|----------|
| Decision | ✅ Complete | ADR-043 approved |
| Implementation | ✅ Complete | Phases 2-7 landed |
| Validation | ✅ Complete | 180+ unit tests, integration tests pass |
| Governance | ✅ Complete | ADR-043 updated to COMPLETE status |

**Outcome:**
- Extracted modules: `validators/run_card_validator.py`, `artifact_manager.py`, `config_resolver.py`, `pipeline_coordinator.py`, `execution_engine.py`
- Total extracted LOC: ~2,720
- Orchestrator remains at ~5,675 LOC (documented state-machine exception in ADR-043)

### Test Marker Retrofit (ADR-044)

| Gate | Status | Evidence |
|------|--------|----------|
| Decision | ✅ Complete | ADR-044 approved |
| Implementation | ✅ Complete | 95.1% coverage achieved (was 48.6%) |
| Validation | ⏳ Partial | Pre-commit works; CI runtime validation pending |
| Governance | ⏳ Partial | ADR-044 updated; CI marker selection pending |

**Outcome:**
- 137 files tagged with appropriate markers
- Pre-commit hook blocks unmarked new tests
- `check_test_markers.py --audit` validates coverage

### CI/CD Security Hardening

| Item | Status | Evidence |
|------|--------|----------|
| GitHub Actions SHA pinning | ✅ Complete | All actions pinned in `build.yml`, `ci.yml` |
| pytest-xdist parallelization | ✅ Complete | `-n auto` enabled in CI |
| mypy hard-fail (core/) | ✅ Complete | Type errors block merge |
| HuggingFace revision pinning | ✅ Complete | `config/model_lock_manifest.yaml` |

---

## Carry-Forward Work

Work in this section is **partially landed** and requires completion or validation.

### 1. Workflow Canonicalization

**Priority:** Now
**Status:** Open

**Problem Statement:**
Multiple CI workflows exist with divergent enforcement semantics:
- `build.yml`: PR gating workflow (lint, tests, manifest)
- `ci.yml`: Post-merge validation workflow
- Marker expressions differ between workflows
- Some workflows use negative marker selection (`not ml and not slow`)

**Objective:**
Establish one canonical PR gating workflow with consistent enforcement, then align or retire conflicting workflows.

**Target Outcome:**
- One workflow (`build.yml`) declared canonical for branch protection
- All active workflows match canonical enforcement semantics or are retired
- Floating action tags eliminated (already done)
- Runtime targets measured against canonical workflow only

**Acceptance Gates:**

| Gate | Criteria |
|------|----------|
| Decision | Canonical workflow named; parity requirements documented |
| Implementation | Workflow alignment changes merged |
| Validation | CI runtime targets measured on canonical workflow |
| Governance | Workflow documentation updated |

**Effort:** 8-12 hours
**Owner:** Architect + DevOps

---

### 2. Orchestrator Residual Slimming & Boundary Enforcement

**Priority:** Now
**Status:** Partial

**Problem Statement:**
While ADR-043 decomposition is complete, the orchestrator remains at ~5,675 LOC (documented state-machine exception). Ongoing vigilance is required to:
- Prevent responsibility creep back into the orchestrator
- Ensure extracted module boundaries remain durable
- Continue incremental slimming where feasible

**Objective:**
Stop new feature logic from accumulating in `orchestrator.py`; continue moving residual helpers behind module boundaries.

**Acceptance Gates:**

| Gate | Criteria |
|------|----------|
| Decision | ✅ ADR-043 documents target architecture |
| Implementation | ⏳ Residual helpers continue moving (ratchet target) |
| Validation | Import graph remains acyclic; delegation tests exist |
| Governance | ADR-043 updated if target changes |

**Ratchet Target:** Reduce orchestrator LOC by 200 lines/quarter or document why infeasible.

**Effort:** 4-8 hours ongoing per quarter
**Owner:** Architect (review) + Specialist (implementation)

---

### 3. Marker-Based CI Validation

**Priority:** Next
**Status:** Implemented

**Problem Statement:**
Test marker retrofit is complete (95.1% coverage), but CI runtime benefits are not yet proven:
- CI workflows still use negative marker selection
- Runtime targets (`pytest -m unit` in <3 min) not yet validated
- Positive marker selection migration not yet executed

**Objective:**
Prove that marker-based selection delivers the expected feedback-loop gains.

**Acceptance Gates:**

| Gate | Criteria |
|------|----------|
| Decision | ✅ ADR-044 approved |
| Implementation | ✅ Markers retrofitted; pre-commit enforces |
| Validation | ⏳ CI runtime targets proven in canonical workflow |
| Governance | ⏳ ADR-044 success criteria updated with measurements |

**Open Work:**
1. Migrate CI from negative selection to positive selection
2. Measure `pytest -m unit` runtime (target: <3 min)
3. Measure `pytest -m "unit or integration"` runtime (target: <10 min)
4. Document measurements in ADR-044

**Effort:** 4-6 hours
**Owner:** Specialist

---

## Q2 Commitments

Work in this section is **net-new or still open** and must complete this quarter.

### 4. Coverage Ramp Phase 1

**Priority:** Next
**Status:** Open

**Current Coverage:** 25.44% (6,314 of 24,820 statements)
**Phase 1 Target:** 28% (+636 statements)

**Phase 1 Focus Modules (high ROI):**

| Module | Statements | Current | Target |
|--------|-----------|---------|--------|
| `cli/__init__.py` | ~250 | 0% | 60% |
| `config_loader.py` | 148 | 40% | 80% |
| `utils/input_validation.py` | 195 | 0% | 70% |
| `utils/recipe_validator.py` | 63 | 0% | 80% |

**Dependency:** Benefits from Marker-Based CI Validation for efficient test execution.

**Acceptance Gates:**

| Gate | Criteria |
|------|----------|
| Decision | Target modules identified |
| Implementation | Tests added; coverage reaches 28% |
| Validation | pytest-cov reports target met |
| Governance | Coverage docs updated |

**Effort:** 15-20 hours
**Owner:** Specialist
**Milestone:** v2.4.0

---

### 5. Governance Synchronization

**Priority:** Now
**Status:** Open

**Problem Statement:**
Policy and enforcement changes have landed, but supporting governance documents may encode stale assumptions.

**Objective:**
Align all binding and operational documents after policy changes.

**Scope:**
- This roadmap
- `CONTRIBUTING.md`
- `docs/testing/STRATEGY.md`
- ADR index / status pages
- CI workflow documentation

**Rule:** No policy change is "complete" until corresponding governance docs are updated.

**Acceptance Gates:**

| Gate | Criteria |
|------|----------|
| Decision | Scope defined |
| Implementation | Documents updated |
| Validation | No contradictions between enforcement and docs |
| Governance | Review cadence documented |

**Effort:** 4-6 hours
**Owner:** Architect

---

## Deferred / Post-Q2

Work in this section is **important but intentionally out of current scope**.

### 6. Documentation Consolidation

**Priority:** Later
**Status:** Open

**Current State:**
- 719 markdown files across 99 directories
- Parallel ADR hierarchies
- 83% missing "Last Updated" metadata

**Target State:**
- ~200 canonical documents across ~30 directories
- Single ADR location with archive for superseded
- All canonical docs have metadata headers

**Rationale for Deferral:** Best done after execution controls are clean (Workflow Canonicalization, Governance Synchronization).

**Effort:** 15-20 hours
**Owner:** Specialist
**Milestone:** v2.5.0

---

### 7. Circular Import Contract Hardening

**Priority:** Later
**Status:** Open

**Current State:** Mitigated via `TYPE_CHECKING` guards (8 files in `depth/backends/`)

**Target State:**
- Shared contracts in `lux_depth_v3/_contracts.py`
- Documented pattern in architecture guide
- CI check for new cross-module imports

**Rationale for Deferral:** Best done against stabilized orchestrator boundaries.

**Effort:** 8-12 hours
**Owner:** Architect
**Milestone:** v2.5.0

---

## Q2 Priority Stack

| Priority | Workstream | Status | Rationale |
|----------|------------|--------|-----------|
| Now | Workflow Canonicalization | Open | Highest leverage; removes CI ambiguity |
| Now | Orchestrator Residual Slimming & Boundary Enforcement | Partial | Core maintainability risk still active |
| Now | Governance Synchronization | Open | Required for policy integrity |
| Next | Marker-Based CI Validation | Implemented | Policy exists; proof and runtime gains remain |
| Next | Coverage Ramp Phase 1 | Open | Strong ROI once CI targeting is stable |
| Later | Documentation Consolidation | Open | Valuable, but not before execution controls are clean |
| Later | Circular Import Contract Hardening | Open | Important follow-on, best done against stabilized boundaries |

---

## Leading Indicator Metrics

These metrics enable proactive steering rather than quarter-end audit.

### Workflow Parity Debt

| Metric | Current | Target |
|--------|---------|--------|
| Active workflows with conflicting enforcement semantics | TBD | 0 |
| Active workflows using floating action tags | 0 | 0 |
| Workflows whose test/type policy differs from canonical | TBD | 0 |

### Orchestrator Residual Debt

| Metric | Current | Target |
|--------|---------|--------|
| `orchestrator.py` LOC | ~5,675 | -200/quarter ratchet |
| Responsibility domains in facade | TBD | Document and track |
| Delegation contract tests | TBD | +10/quarter |

### Governance Freshness

| Metric | Current | Target |
|--------|---------|--------|
| Binding docs updated within SLA after policy change | TBD | 100% within 1 week |
| Stale documents contradicting current enforcement | TBD | 0 |

---

## Quarterly Success Metrics (Lagging Indicators)

| Metric | Q1 2026 | Q2 Target | Measurement |
|--------|---------|-----------|-------------|
| Overall Score | 7.6/10 | 8.6/10 | Codebase audit |
| Orchestrator LOC | 6,108 | 5,475 (-200 ratchet) | `wc -l orchestrator.py` |
| Test Marker Coverage | 48.6% | 95%+ | `check_test_markers.py --audit` |
| Code Coverage | 25.44% | 28% | pytest-cov |
| CI Time (canonical workflow) | 65-75 min | 40-50 min | GitHub Actions |
| Workflow Parity Debt | TBD | 0 | Manual audit |

---

## ADR Status

| ADR | Topic | Status |
|-----|-------|--------|
| ADR-043 | Orchestrator Decomposition | ✅ COMPLETE |
| ADR-044 | Test Marker Enforcement Policy | ✅ ACCEPTED (validation pending) |
| ADR-045 | CI/CD Optimization | Optional (SHA pinning complete) |

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Workflow canonicalization disrupts existing CI | Low | Medium | Test changes in separate workflow first |
| Orchestrator responsibility creep | Medium | Medium | Ratchet target + ADR-043 review gate |
| Governance docs drift from enforcement | Medium | Low | Governance Synchronization workstream |
| CI runtime targets not achieved | Low | Low | Measure early; adjust marker strategy if needed |

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
