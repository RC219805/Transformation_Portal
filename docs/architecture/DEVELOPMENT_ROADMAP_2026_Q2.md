# Development Roadmap: Q2 2026

**Status:** ACTIVE  
**Version:** 1.0.0  
**Date:** 2026-03-20  
**Authority:** Architect Assessment  
**Supersedes:** Actionable items from [Q1 2026 Codebase Audit](CODEBASE_AUDIT_2026_Q1.md)

---

## Executive Summary

This roadmap identifies the **highest-signal development priorities** for the Transformation Portal, derived from:
- Q1 2026 Codebase Audit (7.6/10 overall score)
- TODO Inventory v2.1.0 (65 items, 6% action required)
- Portal Orchestrator Roadmap (Phase 5A complete)
- Testing Strategy documentation vs. actual enforcement gap

**Goal:** Raise overall codebase health from **7.6/10 → 8.6/10** by end of Q2 2026.

---

## Priority Matrix

| Priority | Focus Area | Impact | Effort | Dependency |
|----------|-----------|--------|--------|------------|
| 🔴 P0 | Orchestrator Decomposition | Maintainability | 40-60h | None |
| 🔴 P0 | Test Marker Enforcement | CI Efficiency | 10-15h | None |
| ⚠️ P1 | Coverage Ramp Phase 1 | Quality | 15-20h | P0 markers |
| ⚠️ P1 | CI/CD Optimization | DevX | 20-25h | None |
| 📋 P2 | Documentation Consolidation | Onboarding | 15-20h | None |
| 📋 P2 | Circular Import Hardening | Stability | 8-12h | P0 orchestrator |

---

## 🔴 Critical Priority (P0)

### 1. Orchestrator Decomposition

**Problem Statement:**
`src/transformation_portal/lux_depth_v3/orchestrator.py` contains 6,108 lines of code, with a single `EnhanceOrchestrator` class spanning 5,491 LOC. This violates single-responsibility principle and creates:
- Untestable monolithic code paths
- 40-50% cyclomatic complexity
- Onboarding friction (impossible to understand in isolation)
- Merge conflict hotspot

**Current Structure:**
```
EnhanceOrchestrator (5,491 LOC)
├── Configuration resolution (~1,000 LOC)
├── Pipeline coordination (~1,500 LOC)
├── Artifact management (~1,500 LOC)
├── Execution engine (~2,000 LOC)
└── Validation helpers (~500 LOC scattered)
```

**Target Structure (ADR-043):**
```
lux_depth_v3/
├── orchestrator.py (~500 LOC, facade)
├── config_resolver.py (~1,000 LOC)
├── pipeline_coordinator.py (~1,500 LOC)
├── artifact_manager.py (~1,500 LOC)
├── execution_engine.py (~2,000 LOC)
└── validators/
    └── run_card_validator.py (~500 LOC)
```

**Acceptance Criteria:**
- [ ] No single class exceeds 1,500 LOC
- [ ] All extracted modules have >80% unit test coverage
- [ ] No circular imports between new modules
- [ ] Existing CLI/API behavior unchanged (regression tests pass)
- [ ] ADR-043 documenting rationale and boundaries

**Effort:** 40-60 hours across 2-3 weeks
**Owner:** Specialist (implementation) + Architect (review)
**Milestone:** v2.4.0

---

### 2. Test Marker Enforcement

**Problem Statement:**
75% of tests (3,168 of 4,221 functions) lack pytest markers despite comprehensive `docs/testing/STRATEGY.md`. This prevents:
- Targeted test suite execution
- CI job parallelization
- Fast PR feedback loops

**Current State:**
| Marker | Count | Target |
|--------|-------|--------|
| No marker | 3,168 (75%) | 0% |
| `@pytest.mark.unit` | 10 | 2,500+ |
| `@pytest.mark.integration` | 5 | 200+ |
| `@pytest.mark.ml` | 50 | 150+ |
| `@pytest.mark.security` | 18 | 50+ |
| `@pytest.mark.benchmark` | 34 | (adequate) |

**Implementation Plan:**
1. **Batch 1 (Week 1):** Tag all `tests/unit/` with `@pytest.mark.unit` 
2. **Batch 2 (Week 1):** Tag all `tests/security/` with `@pytest.mark.security`
3. **Batch 3 (Week 2):** Tag all `tests/integration/` appropriately
4. **Batch 4 (Week 2):** Tag root-level tests based on actual test behavior
5. **Enforcement (Week 2):** Add pre-commit hook requiring marker on new tests

**Acceptance Criteria:**
- [ ] <5% of tests unmarked (only test infrastructure)
- [ ] Pre-commit hook blocks unmarked test additions
- [ ] CI can run `pytest -m unit` in <3 minutes
- [ ] CI can run `pytest -m "unit or integration"` in <10 minutes

**Effort:** 10-15 hours across 1-2 weeks
**Owner:** Specialist
**Milestone:** v2.3.1

---

## ⚠️ High Priority (P1)

### 3. Coverage Ramp Phase 1

**Current Coverage:** 25.44% (6,314 of 24,820 statements)
**Phase 1 Target:** 28% (+636 statements)
**Phase 2 Target:** 33% (+1,241 additional statements)

**Phase 1 Focus Modules (high ROI):**

| Module | Statements | Current | Target |
|--------|-----------|---------|--------|
| `cli/__init__.py` | ~250 | 0% | 60% |
| `config_loader.py` | 148 | 40% | 80% |
| `utils/input_validation.py` | 195 | 0% | 70% |
| `utils/recipe_validator.py` | 63 | 0% | 80% |

**Dependency:** Requires P0 test marker completion for efficient test execution.

**Effort:** 15-20 hours
**Owner:** Specialist
**Milestone:** v2.4.0

---

### 4. CI/CD Optimization

**Current State:**
- Full CI suite: 65-75 minutes
- Actions use version tags (@v6, @v7) not SHA pins
- Sequential test execution (no parallelization)
- mypy soft-fails (type errors don't block)

**Target State:**
- Full CI suite: 40-50 minutes (-35%)
- All actions pinned to commit SHAs
- Parallel test execution with pytest-xdist
- mypy hard-fail for critical modules

**Implementation Tasks:**

| Task | Effort | Impact | Status |
|------|--------|--------|--------|
| Pin all GitHub Actions to SHA | 3h | Security | Pending |
| Add pytest-xdist parallelization | 2h | 20-30% faster | Pending |
| Enable fail-fast: false | 1h | Better debugging | Pending |
| Make mypy hard-fail (core/) | 2h | Type safety | Pending |
| Add per-test timeouts | 1h | Stability | Pending |

**Effort:** 20-25 hours
**Owner:** DevOps + Architect
**Milestone:** v2.4.0

---

## 📋 Medium Priority (P2)

### 5. Documentation Consolidation

**Current State:**
- 719 markdown files across 99 directories
- Parallel ADR hierarchies (`architecture/`, `decisions/`, `adr/`)
- 83% missing "Last Updated" metadata
- 15+ stub documents (<50 lines)

**Target State:**
- ~200 canonical documents across ~30 directories
- Single ADR location with archive for superseded
- All canonical docs have metadata headers
- Stubs either completed or deleted

**Implementation Tasks:**
1. Archive historical session/PR docs to `docs/_archive/`
2. Consolidate ADRs to `docs/architecture/adr/`
3. Add metadata headers to top 50 docs
4. Delete or complete stub documents

**Effort:** 15-20 hours
**Owner:** Specialist
**Milestone:** v2.4.0

---

### 6. Circular Import Hardening

**Current State:** Mitigated via `TYPE_CHECKING` guards
- 8 files in `depth/backends/` import from `lux_depth_v3/`
- Fragile: can break with refactoring

**Target State:**
- Shared contracts in lightweight `lux_depth_v3/_contracts.py`
- Documented pattern in architecture guide
- CI check for new cross-module imports

**Dependency:** Partially blocked by P0 orchestrator decomposition.

**Effort:** 8-12 hours
**Owner:** Architect
**Milestone:** v2.5.0

---

## Quarterly Success Metrics

| Metric | Q1 2026 | Q2 Target | Measurement |
|--------|---------|-----------|-------------|
| Overall Score | 7.6/10 | 8.6/10 | Codebase audit |
| Orchestrator LOC | 6,108 | <2,000 | `wc -l orchestrator.py` |
| Test Marker Coverage | 25% | 95% | grep analysis |
| Code Coverage | 25.44% | 33% | pytest-cov |
| CI Time (full) | 65-75 min | 40-50 min | GitHub Actions |
| Doc Directories | 99 | ~30 | `find docs -type d` |

---

## ADR Requirements

This roadmap generates the following ADR needs:

| ADR | Topic | Status |
|-----|-------|--------|
| ADR-043 | Orchestrator Decomposition | Draft required |
| ADR-044 | Test Marker Enforcement Policy | Draft required |
| ADR-045 | CI/CD Optimization (Action Pinning) | Optional |

---

## Rollout Phases

### Phase 1: Foundation (Weeks 1-2)
- Start orchestrator analysis and decomposition planning
- Complete test marker retrofit (Batches 1-4)
- Pin GitHub Actions to SHAs

### Phase 2: Execution (Weeks 3-6)
- Execute orchestrator decomposition (incremental PRs)
- Add pytest-xdist parallelization
- Begin coverage ramp Phase 1

### Phase 3: Polish (Weeks 7-8)
- Documentation consolidation
- Circular import hardening
- Q2 retrospective and Q3 planning

---

## Risk Register

| Risk | Mitigation |
|------|------------|
| Orchestrator decomposition introduces regressions | Comprehensive integration tests before decomposition |
| Test marker retrofit is tedious and error-prone | Batch by directory, automated grep verification |
| CI changes break main branch | Test in separate workflow first |

---

## Related Documents

- [Q1 2026 Codebase Audit](CODEBASE_AUDIT_2026_Q1.md)
- [TODO Inventory Quick Reference](TODO_INVENTORY_QUICK_REF.md)
- [Portal Orchestrator Roadmap](PORTAL_ORCHESTRATOR_ROADMAP.md)
- [Testing Strategy](../testing/STRATEGY.md)
- [Coverage Improvement Plan](../guides/coverage-improvement-plan.md)

---

**Document Owner:** Transformation Portal Architect  
**Review Cadence:** Monthly (end of each sprint)  
**Next Review:** April 2026

---

*This document is binding under architectural governance. Deviations require explicit Architect approval.*
