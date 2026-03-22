# Development Roadmap: Q2 2026

**Status:** ACTIVE
**Version:** 1.2.0
**Date:** 2026-03-22
**Authority:** Architect Assessment
**Supersedes:** v1.1.0 (2026-03-22), v1.0.0 (2026-03-20)

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

## Completed / Implemented Since Q1 Audit

Work in this section has landed but may have **partial validation or governance pending**.

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
- Orchestrator remains at 5,676 LOC (documented state-machine exception in ADR-043)

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

**Open Work (see Quality Control Plane Canonicalization):**
- CI workflows still use negative marker selection (`not ml and not slow`)
- `CONTRIBUTING.md` still teaches `pytest -m "not ml and not slow"`
- Positive marker selection migration not yet executed

### CI/CD Security Hardening

| Item | Status | Evidence |
|------|--------|----------|
| GitHub Actions SHA pinning (`build.yml`) | ✅ Complete | All 17 actions pinned to commit SHA |
| GitHub Actions SHA pinning (`ci.yml`) | ✅ Complete | All actions pinned to commit SHA |
| GitHub Actions SHA pinning (other workflows) | ⏳ Partial | `ci-quality-firewall.yml`, `quality-gate.yml` use version tags |
| pytest-xdist parallelization (`ci.yml`) | ✅ Complete | `-n auto` enabled |
| pytest-xdist parallelization (`build.yml`) | ❌ Open | Not enabled in PR gating workflow |
| mypy hard-fail (`ci.yml`) | ✅ Complete | Type errors block (no `continue-on-error`) |
| mypy hard-fail (`ci-quality-firewall.yml`) | ❌ Open | Uses `continue-on-error: true` |
| HuggingFace revision pinning | ✅ Complete | `config/model_lock_manifest.yaml` |

**Note:** `build.yml` is designated as the canonical PR gating workflow (see `ci.yml` line 7 comment). `ci.yml` is the post-merge validation workflow. Formal documentation of this designation is pending (see Quality Control Plane Canonicalization workstream).

---

## Carry-Forward Work

Work in this section is **partially landed** and requires completion or validation.

### 1. Quality Control Plane Canonicalization

**Priority:** Now
**Status:** Partial

**Problem Statement:**
Multiple CI workflows exist with divergent enforcement semantics. The repo encodes a provisional canonical decision (`ci.yml` line 7: "pull_request disabled - CI Gate in build.yml is the required governance check"), but workflows are not fully normalized.

**Quality Control Plane Inventory:**

| Workflow | Trigger | Role | Blocking? | Marker Semantics | Typecheck Mode | Action Pinning | Target Disposition |
|----------|---------|------|-----------|------------------|----------------|----------------|-------------------|
| `build.yml` | PR, push, dispatch | PR gating | Yes | Negative (`not ml and not slow`) | N/A | ✅ SHA | Canonical |
| `ci.yml` | push only | Post-merge | No | Negative (`not ml and not slow`) | Hard-fail | ✅ SHA | Align with canonical |
| `ci-quality-firewall.yml` | workflow_run | Post-CI | No | Negative | Soft-fail | ⚠️ Mixed | Align or retire |
| `quality-gate.yml` | PR, push | Pre-commit | Yes | N/A | N/A | ❌ Version tags | Align or scope down |

**Note:** Workflows outside the quality control plane (docs, security, nightly, deployment, automation) intentionally differ and are not included in parity debt.

**Objective:**
Establish one canonical PR gating workflow (`build.yml`) with consistent enforcement, then align or retire conflicting quality-control workflows.

**Scope includes (folded from Marker-Based CI Validation):**
- Workflow selection and role clarification
- Marker semantics normalization (negative → positive)
- Runtime validation targets
- Documentation alignment (`CONTRIBUTING.md`, `docs/testing/STRATEGY.md`)
- Action pinning normalization

**Target Outcome:**
- `build.yml` declared canonical for branch protection
- All quality-control workflows match canonical enforcement semantics or are retired
- Marker semantics unified (positive selection: `unit and not slow`)
- Runtime targets measured against canonical workflow only

**Acceptance Gates:**

| Gate | Criteria |
|------|----------|
| Decision | ⏳ Provisional (build.yml designated; formal documentation pending) |
| Implementation | Workflow alignment changes merged; action pins normalized |
| Validation | CI runtime targets measured on canonical workflow |
| Governance | Workflow documentation updated; CONTRIBUTING.md aligned |

**Effort:** 12-16 hours (includes marker validation work)
**Owner:** Architect + DevOps

---

### 2. Orchestrator Residual Slimming & Boundary Enforcement

**Priority:** Now
**Status:** Partial

**Problem Statement:**
While ADR-043 decomposition is complete, the orchestrator remains at 5,676 LOC (documented state-machine exception). Ongoing vigilance is required to:
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

## Q2 Commitments

Work in this section is **net-new or still open** and must complete this quarter.

### 3. Coverage Ramp Phase 1

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

**Dependency:** Benefits from Quality Control Plane Canonicalization for efficient test execution.

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

### 4. Governance Synchronization

**Priority:** Now
**Status:** Open

**Problem Statement:**
Policy and enforcement changes have landed, but supporting governance documents encode stale assumptions. For example:
- `CONTRIBUTING.md` still says mypy is non-blocking
- `CONTRIBUTING.md` still describes 33% coverage baseline (actual: 25.44%, target: 28%)
- Docs teach `pytest -m "not ml and not slow"` while target state is positive selection

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

### 5. Documentation Consolidation

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

**Rationale for Deferral:** Best done after execution controls are clean (Quality Control Plane Canonicalization, Governance Synchronization).

**Effort:** 15-20 hours
**Owner:** Specialist
**Milestone:** v2.5.0

---

### 6. Circular Import Contract Hardening

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
| Now | Quality Control Plane Canonicalization | Partial | Highest leverage; removes CI ambiguity; includes marker validation |
| Now | Orchestrator Residual Slimming & Boundary Enforcement | Partial | Core maintainability risk still active |
| Now | Governance Synchronization | Open | Required for policy integrity |
| Next | Coverage Ramp Phase 1 | Open | Strong ROI once CI targeting is stable |
| Later | Documentation Consolidation | Open | Valuable, but not before execution controls are clean |
| Later | Circular Import Contract Hardening | Open | Important follow-on, best done against stabilized boundaries |

---

## Leading Indicator Metrics

These metrics enable proactive steering rather than quarter-end audit.

### Workflow Parity Debt (Quality Control Plane Only)

**Scope:** `build.yml`, `ci.yml`, `ci-quality-firewall.yml`, `quality-gate.yml`

**Note:** Workflows outside the quality control plane (docs, security, nightly, deployment, automation) intentionally differ and are excluded from parity debt.

| Metric | Current | Target |
|--------|---------|--------|
| Quality-control workflows (total) | 4 | N/A (informational) |
| QC workflows with conflicting marker semantics | 3 | 0 |
| QC workflows using version-tag actions | 2 | 0 |
| QC workflows with divergent typecheck blocking | 2 | 0 |
| QC workflows with divergent artifact/coverage behavior | TBD | 0 |

**Composite Metric:** Workflow Parity Debt = sum of non-conforming quality-control workflows. Target: 0.

### Orchestrator Residual Debt

| Metric | Current | Target |
|--------|---------|--------|
| `orchestrator.py` LOC | 5,676 | 5,476 (-200/quarter ratchet) |
| Responsibility domains in facade | 5 (depth, v2, PBR, materials, artifacts) | ≤5 (no new domains) |
| Delegation contract tests | ~180 (across extracted modules) | +10/quarter (190 by Q2 end) |

**Primary Indicators (blocking):**
1. No new responsibility domains added
2. No new feature logic added directly to `orchestrator.py`
3. Import graph remains acyclic
4. Delegation contract tests increase
5. Module boundaries remain stable

**Secondary Indicator (directional):** LOC ratchet target

**Domain Inventory:**
1. Depth stage execution (per-image backend fallback, cache management)
2. V2 stage execution (subprocess coordination)
3. PBR generation (texture pipeline)
4. Materials V3 execution (APEX quality gates)
5. Artifact persistence (run card, merkle roots)

### Governance Freshness

**Baseline to be established during Governance Synchronization workstream (by 2026-04-12).**

| Metric | Current | Target |
|--------|---------|--------|
| Binding docs updated within SLA after policy change | Audit by 04-12 | 100% within 1 week |
| Stale documents contradicting current enforcement | Audit by 04-12 | 0 |

**Scope of binding docs:** This roadmap, CONTRIBUTING.md, docs/testing/STRATEGY.md, ADR status pages, CI workflow documentation.

---

## Quarterly Success Metrics (Lagging Indicators)

**Note:** Q1 2026 column shows state at audit time (2026-03-20), before Q2 work began. Changes between audit and current state represent completed work documented in "Completed / Implemented Since Q1 Audit" section.

| Metric | Q1 2026 (Audit) | Current | Q2 Target | Measurement |
|--------|-----------------|---------|-----------|-------------|
| Overall Score | 7.6/10 | 7.6/10 | 8.6/10 | Codebase audit |
| Orchestrator LOC | 6,108 | 5,676 | 5,476 | `wc -l orchestrator.py` |
| Test Marker Coverage | 48.6% | 95.1% ✅ | 95%+ | `check_test_markers.py --audit` |
| Code Coverage | 25.44% | 25.44% | 28% | pytest-cov |
| CI Time (canonical) | 65-75 min | TBD | 40-50 min | GitHub Actions (`build.yml`) |
| Workflow Parity Debt (QC) | N/A | 7 (see metrics) | 0 | Quality-control workflows only |

---

## ADR Status

| ADR | Topic | Status |
|-----|-------|--------|
| ADR-043 | Orchestrator Decomposition | ✅ COMPLETE |
| ADR-044 | Test Marker Enforcement Policy | ✅ ACCEPTED (validation pending) |
| ADR-045 | CI/CD & Workflow Semantics | Required if Q2 changes: branch-protection semantics, canonical workflow designation, marker taxonomy, or blocking/non-blocking policy |

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Quality control plane canonicalization disrupts existing CI | Low | Medium | Test changes in separate workflow first |
| Orchestrator responsibility creep | Medium | Medium | Ratchet target + ADR-043 review gate |
| Governance docs drift from enforcement | Medium | Low | Governance Synchronization workstream |
| CI runtime targets not achieved | Low | Low | Measure early; adjust marker strategy if needed |
| quality-gate.yml retirement causes tooling breakage | Low | Low | Document deprecation path; verify no external dependencies |

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
