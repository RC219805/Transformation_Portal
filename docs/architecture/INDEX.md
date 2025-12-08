# Architecture Hardening Plan: Document Index

**Version**: 1.0  
**Date**: 2025-12-08  
**Status**: 🟢 **COMPLETE - READY FOR IMPLEMENTATION**

---

## Quick Navigation

### 🚀 Start Here

**For Executives**: [Executive Summary](ARCHITECTURE_HARDENING_EXECUTIVE_SUMMARY.md)  
**For Architects**: [Complete Plan](ARCHITECTURE_HARDENING_PLAN.md)  
**For Developers**: [Platform Core Migration Guide](MIGRATION_GUIDE_PLATFORM_CORE.md)  
**For Security**: [Security Hardening Checklist](SECURITY_HARDENING_CHECKLIST.md)  
**For DevOps**: [CI/CD Integration Spec](CI_CD_INTEGRATION_SPEC.md)

---

## Document Catalog

### Core Planning Documents

| Document | Size | Purpose | Audience |
|----------|------|---------|----------|
| [ARCHITECTURE_HARDENING_PLAN.md](ARCHITECTURE_HARDENING_PLAN.md) | 47KB | Complete technical plan with all 6 PRs | Architects, Tech Leads |
| [ARCHITECTURE_HARDENING_EXECUTIVE_SUMMARY.md](ARCHITECTURE_HARDENING_EXECUTIVE_SUMMARY.md) | 12KB | High-level overview, success metrics | Executives, Product Managers |

### Architecture Decision Records (ADRs)

| ADR | Title | Size | Status | Related PR |
|-----|-------|------|--------|------------|
| [ADR-001](adrs/ADR-001-PLATFORM-CORE.md) | Platform Core Extraction | 10KB | Proposed | PR-2 |
| [ADR-002](adrs/ADR-002-STAGE-GRAPH.md) | Stage Graph Architecture | 11KB | Proposed | PR-3 |
| [ADR-003](adrs/ADR-003-SECURITY-HARDENING.md) | Security Hardening Strategy | 15KB | Proposed | PR-1 |

### Implementation Guides

| Guide | Size | Purpose | Audience |
|-------|------|---------|----------|
| [MIGRATION_GUIDE_PLATFORM_CORE.md](MIGRATION_GUIDE_PLATFORM_CORE.md) | 15KB | Step-by-step migration instructions | Developers |
| [CI_CD_INTEGRATION_SPEC.md](CI_CD_INTEGRATION_SPEC.md) | 16KB | CI/CD requirements for all PRs | DevOps, CI/CD Engineers |
| [SECURITY_HARDENING_CHECKLIST.md](SECURITY_HARDENING_CHECKLIST.md) | 14KB | Verification checklist with sign-off | Security Engineers |

---

## Reading Paths by Role

### Executive / Product Manager

1. **Executive Summary** (5 min read)
   - [ARCHITECTURE_HARDENING_EXECUTIVE_SUMMARY.md](ARCHITECTURE_HARDENING_EXECUTIVE_SUMMARY.md)
   - Get: Timeline, success metrics, business impact

2. **Risk Assessment** (optional, 10 min read)
   - [ARCHITECTURE_HARDENING_PLAN.md](ARCHITECTURE_HARDENING_PLAN.md) - Section: "Timeline & Risk Assessment"
   - Get: Risk matrix, mitigation strategies, rollback plan

**Total Time**: 5-15 minutes

### Architect / Tech Lead

1. **Complete Plan** (30 min read)
   - [ARCHITECTURE_HARDENING_PLAN.md](ARCHITECTURE_HARDENING_PLAN.md)
   - Get: All technical details, API contracts, migration strategy

2. **ADRs** (30 min read)
   - [ADR-001: Platform Core](adrs/ADR-001-PLATFORM-CORE.md)
   - [ADR-002: Stage Graph](adrs/ADR-002-STAGE-GRAPH.md)
   - [ADR-003: Security](adrs/ADR-003-SECURITY-HARDENING.md)
   - Get: Decision rationale, alternatives considered, consequences

**Total Time**: 1 hour

### Developer (Pipeline Contributor)

1. **Migration Guide** (20 min read)
   - [MIGRATION_GUIDE_PLATFORM_CORE.md](MIGRATION_GUIDE_PLATFORM_CORE.md)
   - Get: How to migrate existing code, before/after examples

2. **Platform Core ADR** (optional, 10 min read)
   - [ADR-001: Platform Core](adrs/ADR-001-PLATFORM-CORE.md)
   - Get: Why we're doing this, API contracts

**Total Time**: 20-30 minutes

### DevOps / CI Engineer

1. **CI/CD Spec** (25 min read)
   - [CI_CD_INTEGRATION_SPEC.md](CI_CD_INTEGRATION_SPEC.md)
   - Get: Job definitions, workflows, gates, badge setup

2. **Security Checklist** (optional, 15 min read)
   - [SECURITY_HARDENING_CHECKLIST.md](SECURITY_HARDENING_CHECKLIST.md) - Section: "Phase 2: CI Security Gates"
   - Get: Security gate implementation details

**Total Time**: 25-40 minutes

### Security Engineer

1. **Security ADR** (15 min read)
   - [ADR-003: Security Hardening](adrs/ADR-003-SECURITY-HARDENING.md)
   - Get: Security strategy, CVE mitigation, defense-in-depth

2. **Security Checklist** (20 min read)
   - [SECURITY_HARDENING_CHECKLIST.md](SECURITY_HARDENING_CHECKLIST.md)
   - Get: Verification steps, sign-off requirements

**Total Time**: 35 minutes

---

## PR Implementation Sequence

### PR-1: Security + Repo Hygiene (Week 1, Days 1-3)

**Primary Documents**:
- [ADR-003: Security Hardening](adrs/ADR-003-SECURITY-HARDENING.md)
- [SECURITY_HARDENING_CHECKLIST.md](SECURITY_HARDENING_CHECKLIST.md)
- [CI_CD_INTEGRATION_SPEC.md](CI_CD_INTEGRATION_SPEC.md) - Section: "PR-1"

**Implementation Steps**:
1. Purge sensitive artifacts (checklist Phase 1)
2. Implement `scripts/ci/enforce_safe_deps.py`
3. Update `.github/workflows/security-scan.yml`
4. Enable branch protection

**Success Criteria**: Zero CVEs, no secrets, CI gate enforced

---

### PR-2: Platform Core Extraction (Week 1-2)

**Primary Documents**:
- [ADR-001: Platform Core](adrs/ADR-001-PLATFORM-CORE.md)
- [MIGRATION_GUIDE_PLATFORM_CORE.md](MIGRATION_GUIDE_PLATFORM_CORE.md)
- [CI_CD_INTEGRATION_SPEC.md](CI_CD_INTEGRATION_SPEC.md) - Section: "PR-2"

**Implementation Steps**:
1. Create `transformation_portal/core/` module structure
2. Implement config, device, artifacts, security, observability
3. Write tests (90%+ coverage)
4. Migrate Lux Depth V2 (example)

**Success Criteria**: 66/66 tests pass, 90%+ core coverage, performance neutral

---

### PR-3: Stage Graph Refactor (Week 3-4)

**Primary Documents**:
- [ADR-002: Stage Graph](adrs/ADR-002-STAGE-GRAPH.md)
- [ARCHITECTURE_HARDENING_PLAN.md](ARCHITECTURE_HARDENING_PLAN.md) - Section: "PR-3"
- [CI_CD_INTEGRATION_SPEC.md](CI_CD_INTEGRATION_SPEC.md) - Section: "PR-3"

**Implementation Steps**:
1. Implement `core/pipeline/` (stage, graph, policy)
2. Refactor Lux Depth V2 to stage graph
3. Add performance benchmarks
4. Verify caching correctness

**Success Criteria**: 10x cache speedup, <5% overhead, tests pass

---

### PR-4: Performance + Profiling (Week 5)

**Primary Documents**:
- [ARCHITECTURE_HARDENING_PLAN.md](ARCHITECTURE_HARDENING_PLAN.md) - Section: "PR-4"
- [CI_CD_INTEGRATION_SPEC.md](CI_CD_INTEGRATION_SPEC.md) - Section: "PR-4"

**Implementation Steps**:
1. Add `core/device/profiler.py`
2. Add `core/processing/tiling.py`
3. Integrate profiler into pipelines
4. Add performance regression tests

**Success Criteria**: <5% profiler overhead, UHR support, regression tests

---

### PR-5: Validation-First Defaults (Week 6)

**Primary Documents**:
- [ARCHITECTURE_HARDENING_PLAN.md](ARCHITECTURE_HARDENING_PLAN.md) - Section: "PR-5"
- [CI_CD_INTEGRATION_SPEC.md](CI_CD_INTEGRATION_SPEC.md) - Section: "PR-5"

**Implementation Steps**:
1. Add `core/validation/` (report, metrics, comparison)
2. Integrate into pipelines (emit reports by default)
3. Add report collection to CI
4. Document validation integration

**Success Criteria**: 100% reproducibility, metrics computed, baseline comparison

---

### PR-6: Test Strategy (Week 7-8)

**Primary Documents**:
- [ARCHITECTURE_HARDENING_PLAN.md](ARCHITECTURE_HARDENING_PLAN.md) - Section: "PR-6"
- [CI_CD_INTEGRATION_SPEC.md](CI_CD_INTEGRATION_SPEC.md) - Section: "PR-6"

**Implementation Steps**:
1. Add `core/batch/` (checkpoint/resume)
2. Add fallback tests
3. Add edge case tests
4. Verify 85%+ coverage

**Success Criteria**: 85%+ coverage, fallback branches tested, checkpoint works

---

## Document Dependencies

```
ARCHITECTURE_HARDENING_EXECUTIVE_SUMMARY.md
    ↓
ARCHITECTURE_HARDENING_PLAN.md
    ↓
┌───────────────┬───────────────┬───────────────┐
│   ADR-001     │   ADR-002     │   ADR-003     │
│ Platform Core │ Stage Graph   │   Security    │
└───────┬───────┴───────┬───────┴───────┬───────┘
        │               │               │
        ↓               ↓               ↓
MIGRATION_GUIDE   CI_CD_SPEC    SECURITY_CHECKLIST
```

**Reading Order**:
1. Executive Summary (context)
2. Complete Plan (details)
3. ADRs (decisions)
4. Guides (implementation)

---

## Key Concepts Glossary

### Platform Core
Shared infrastructure module (`transformation_portal/core/`) providing config, device management, caching, security, and observability. Eliminates 50% code duplication.

### Stage Graph
Processing pipeline represented as DAG of cacheable, measurable stages. Enables 10-20x speedup on re-processing and stage-level observability.

### Policy Engine
Context-aware parameter selection (e.g., UHD images → enable tiling). Intelligently routes processing based on image characteristics.

### Reproducibility Manifest
Processing report with git commit, device info, model checksums, config hash, metrics. Enables defensible quality claims.

### Checkpoint/Resume
Batch processing with failure recovery. Jobs can be paused and resumed without losing progress.

### Defense-in-Depth
Multi-layer security: repo hygiene + CI gates + input validation + service authentication. No single point of failure.

---

## Quick Links

### Related Documentation

- [Lux Depth V2 README](../../lux_depth_v2/README.md) - Current production pipeline
- [Lux Depth V2 SECURITY.md](../../lux_depth_v2/SECURITY.md) - CVE-2024-27763 details
- [Root SECURITY.md](../../SECURITY.md) - Vulnerability disclosure process
- [Root README.md](../../README.md) - Project overview

### External References

- [CVE-2024-27763 Details](https://nvd.nist.gov/vuln/detail/CVE-2024-27763)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
- [OWASP API Security](https://owasp.org/www-project-api-security/)

---

## Status Dashboard

### Documentation Status: ✅ **COMPLETE**

| Document | Status | Approvals Needed |
|----------|--------|------------------|
| Complete Plan | ✅ Complete | Project Lead |
| Executive Summary | ✅ Complete | Executives |
| ADR-001 | ✅ Complete | Architect |
| ADR-002 | ✅ Complete | Architect |
| ADR-003 | ✅ Complete | Security Reviewer |
| Migration Guide | ✅ Complete | Tech Lead |
| CI/CD Spec | ✅ Complete | DevOps |
| Security Checklist | ✅ Complete | Security Reviewer |

### Implementation Status: 🔄 **PENDING APPROVAL**

**Next Action**: Stakeholder review and approval (1-2 days)

**Approval Checklist**:
- [ ] Project Lead reviewed Executive Summary
- [ ] Architect reviewed Complete Plan + ADRs
- [ ] Security Reviewer reviewed ADR-003 + Checklist
- [ ] DevOps reviewed CI/CD Spec
- [ ] Team meeting scheduled for kickoff

**Once Approved**: Create `feature/architecture-hardening-pr1-security` branch

---

## Feedback & Questions

**Document Issues**: Open GitHub issue with tag `architecture-hardening`  
**Questions**: Discussion in Architecture category  
**Urgent**: Contact project lead (see MAINTAINERS.md)

---

**Index Version**: 1.0  
**Last Updated**: 2025-12-08  
**Total Documentation**: 8 documents, 137KB  
**Estimated Implementation**: 6-8 weeks
