# CI Health and Tranche Execution — Architect Directive

**Authority**: Transformation Portal Architect
**Date**: 2026-02-04
**Status**: BINDING
**Effective**: Immediate

---

## Executive Summary

Epic #819 tracks 21 improvement opportunities; **7 are complete** (33%). The user's tranche plan proposes completing 3 more items (TEST-001, DOC-001, CI-001) to reach 11/21.

**However**, the repository has accumulated **265 files** with formatting drift (black/isort non-compliance) that will cause future CI failures. This directive mandates **Gate 0** (CI stabilization) before proceeding with throughput improvements.

**Decision**: No new tranche work begins until CI baseline is stable and enforceable.

---

## The Core Problem

Current CI status:
- ✅ Recent runs on main show **SUCCESS**
- ⚠️ **265 files** would be reformatted by black
- ⚠️ **Multiple files** fail isort checks
- ❓ Type checking enforcement is unclear

**Risk**: Without addressing formatting drift, every future PR will either:
1. Fail lint when strict enforcement is added, or
2. Accumulate more drift, making the problem worse

**Architectural principle violated**: "You cannot build throughput on unstable infrastructure."

---

## Mandated Approach: Gate 0 First

### Gate 0: CI Baseline Stabilization

**Timeline**: 2-3 days
**Owner**: Repository Maintainer
**Blocking**: All tranche work

**Required deliverables**:
1. **Baseline formatting PR**: Apply black + isort to entire codebase (one-time)
2. **CI enforcement PR**: Add black/isort checks to workflows (prevent future drift)
3. **Type checking policy**: Document enforcement strategy (may defer implementation)
4. **Coverage gate fix**: Resolve PR #822 or supersede it
5. **Documentation**: Update CONTRIBUTING.md with formatting standards

**Success criteria** (all must be true):
- Main branch CI shows all required checks green for 3+ consecutive commits
- `black --check src/ tests/` exits 0
- `isort --check-only src/ tests/` exits 0
- Type checking policy documented
- `.git-blame-ignore-revs` configured

**Detailed execution plan**: `docs/ci/GATE_0_CHECKLIST.md`

---

## Refined Tranche Plan (Post-Gate 0)

After Gate 0 completes, execute **one item per week** for 3 weeks:

### Week 1: TEST-001 — Shared Test Fixtures

**Deliverables**:
- `tests/conftest.py` with 3 fixture tiers (pure, IO, ML-optional)
- ≥5 reusable fixtures defined
- ≥3 test files refactored (10-20% LOC reduction)
- No new cross-test coupling

**Innovation**: Add 1-2 property-based tests (Hypothesis) for config parsing edge cases

**Success criteria**:
- Tests pass in both `pytest -m "not ml"` and `pytest -m "ml"` slices
- CI remains green
- Measured LOC reduction documented in PR

**Detailed plan**: `docs/architecture/TRANCHE_EXECUTION_PLAN.md` § Week 1

---

### Week 2: DOC-001 — Documentation Consolidation

**Deliverables**:
- `DOCUMENTATION_INDEX.md` in repository root (3-5 primary entry points)
- 1 canonical doc per topic (setup, quickstart, CI, security, architecture)
- Deprecated docs marked with redirect headers (30-day sunset)
- No directories with >3 files lacking README.md

**Canonicalization rule** (prevents taste-based decisions):
- Each topic has exactly one canonical doc
- Duplicates get 30-day deprecation notice before deletion
- Root README links to DOCUMENTATION_INDEX.md

**Success criteria**:
- DOCUMENTATION_INDEX.md exists
- All deprecated docs have redirect headers
- PR template updated to check documentation changes

**Detailed plan**: `docs/architecture/TRANCHE_EXECUTION_PLAN.md` § Week 2

---

### Week 3: CI-001 (Phase 1) — Workflow Consolidation

**Deliverables**:
- `docs/ci/WORKFLOW_MATRIX.md` documenting all workflows (completed in Gate 0)
- Identify and eliminate one duplicated execution path (e.g., lint in both build.yml and quality-gate.yml)
- Consolidate or clarify boundaries between workflows

**Consolidation strategy**:
- Option A: Merge quality-gate checks into build.yml
- Option B: Make quality-gate call reusable workflow from build.yml
- Option C: Clarify boundaries (build = blocking, quality-gate = reporting)

**Innovation** (optional): Add change-aware CI to skip ML tests when only docs change

**Success criteria**:
- One duplicated check eliminated (documented in PR)
- CI runtime same or improved
- No new required checks without stable naming

**Detailed plan**: `docs/architecture/TRANCHE_EXECUTION_PLAN.md` § Week 3

---

## Operational Discipline (Enforced)

### 1. Epic Accounting (Mandatory)

**Rule**: Every merged PR must update Epic #819 with:
- Checkbox status
- Link to merged PR
- Updated completion percentage

**Enforcement**: PR template includes:
```markdown
## Epic Tracking
- [ ] This PR updates Epic #819 with completion status and PR link
```

### 2. Auto-Close Issues (Mandatory)

**Rule**: Use `Fixes #<issue-number>` in PR descriptions

**Example**:
```markdown
## Summary
This PR implements shared test fixtures.

Fixes #<TEST-001-issue-number>
Part of Epic #819
```

### 3. PR Sizing (Enforced)

**Rule**: PRs must be <500 LOC except:
- Mechanical formatting (explicitly marked: `chore(format): ...`)
- Generated code (documented in description)
- Test fixtures (documented in description)

**Enforcement**: PR template includes size justification checkbox

---

## Architectural Governance

### Decision Authority

This directive is **binding** under Transformation Portal Architect authority:
- **CI/CD policy** (Gate 0, workflow consolidation)
- **Repository structure** (documentation canonicalization)
- **Cross-module integration** (test fixtures, shared contracts)
- **Security posture** (enforcement mechanisms, not just documentation)

### Specialist Collaboration

**Delegated to @transformation-portal-specialist**:
- Fixture implementation details (TEST-001)
- Test refactoring (TEST-001)
- Workflow YAML editing (CI-001)

**Retained by Architect**:
- Approval of approach and enforcement mechanisms
- Success criteria definition
- Timeline and scope control

### Deviation Protocol

**Any deviation from this plan requires**:
1. Explicit escalation to Architect
2. Written rationale (ADR if architectural impact)
3. Updated success criteria and timeline

**Silence is not approval.** The Specialist must wait for explicit Architect direction when escalated.

---

## Enforcement Mechanisms

### What Makes This Directive Enforceable

**Not just documentation**:
1. **Gate 0 blocks tranche work** (explicit timeline dependency)
2. **CI checks enforce standards** (black/isort in workflow, not just recommended)
3. **PR template enforces process** (epic tracking, size limits, doc updates)
4. **Success criteria are measurable** (LOC reduction, coverage thresholds, exit code checks)

### Monitoring and Compliance

**Weekly check-ins**:
- Epic #819 updated with progress
- Blockers escalated immediately
- Timeline slips documented with mitigation plan

**Monthly audits**:
- CI health metrics (runtime trends, flake rate)
- Documentation drift (orphaned files, missing READMEs)
- Workflow duplication (identify new consolidation targets)

---

## Risk Assessment and Mitigation

### Risk 1: Baseline Formatting PR Too Large

**Likelihood**: High (265 files is significant)
**Impact**: Medium (noisy git blame history, large diff to review)

**Mitigation**:
- `.git-blame-ignore-revs` preserves blame history
- Automated verification (test suite must pass)
- Clear commit message and PR description
- Single mechanical change (no logic modifications)

### Risk 2: CI Changes Break Existing Workflows

**Likelihood**: Medium
**Impact**: High (blocks all PRs)

**Mitigation**:
- Test enforcement in draft PR first
- Verify all checks pass before making required
- Rollback plan documented in Gate 0 checklist
- Gradual rollout (add checks, monitor, then make blocking)

### Risk 3: Tranche Work Takes Longer Than Estimated

**Likelihood**: Medium (always true for estimation)
**Impact**: Low (affects timeline, not quality)

**Mitigation**:
- Conservative time estimates (4-6 hours, not 2)
- Weekly cadence allows buffer
- Scope reduction option if needed (e.g., 3 fixtures instead of 5)
- Explicit escalation if blocked

---

## Success Metrics

### Gate 0 Metrics

- [ ] Formatting drift: 265 → 0 files
- [ ] CI success rate: ≥95% on main
- [ ] Test suite: 100% pass rate
- [ ] Documentation: 100% of policies documented

### Tranche 1 Metrics

- [ ] TEST-001: ≥5 fixtures, ≥3 files refactored, 10-20% LOC reduction
- [ ] DOC-001: 1 canonical doc per topic, DOCUMENTATION_INDEX.md exists
- [ ] CI-001: 1 duplication eliminated, CI runtime same or better

### Overall Progress

- **Baseline (2026-02-04)**: 7/21 complete (33%)
- **After Gate 0**: 7/21 complete (infrastructure stable)
- **After Tranche 1**: 10/21 complete (48%)
- **Projected velocity**: 3 items per 3 weeks (1/week)
- **Projected completion**: ~12 weeks (May 2026)

---

## Reference Documentation

**Created by this directive**:
- ✅ `docs/architecture/TRANCHE_EXECUTION_PLAN.md`: Full execution plan (8/21 → 11/21)
- ✅ `docs/ci/WORKFLOW_MATRIX.md`: CI workflow reference and audit
- ✅ `docs/ci/TYPE_CHECKING_POLICY.md`: Type enforcement strategy
- ✅ `docs/ci/GATE_0_CHECKLIST.md`: Day-by-day Gate 0 execution

**Referenced**:
- Epic #819: Improvement Opportunities Execution Plan
- PR #820: Quick wins delivery (7 items completed)
- PR #821: Concurrency control
- PR #822: Coverage artifact fix (under review)
- PR #823: SEC-001 command injection hardening (draft)
- `docs/architecture/agent_governance.md`: Governance policy

---

## Communication Plan

### Announcement (Post-Directive)

**To**: Repository maintainers, contributors
**Subject**: New CI Health Directive — Gate 0 Required Before Tranche Work

**Message**:
```markdown
## CI Health Directive — Action Required

The Transformation Portal Architect has issued a binding directive regarding CI health and Epic #819 execution.

### Key Points
- ⚠️ **Gate 0 (CI stabilization) is now mandatory** before continuing tranche work
- 📊 **265 files** have accumulated formatting drift (black/isort)
- ✅ **CI is currently green** but at risk of future failures
- 📅 **Timeline**: 2-3 days for Gate 0, then 3-week tranche (1 item/week)

### What This Means
1. No new tranche work (TEST-001, DOC-001, CI-001) starts until Gate 0 completes
2. Baseline formatting PR will touch 265 files (mechanical changes only)
3. All future PRs must pass black + isort checks

### What You Need to Do
- Review `docs/ci/GATE_0_CHECKLIST.md` for execution plan
- Configure git blame: `git config blame.ignoreRevsFile .git-blame-ignore-revs`
- Ensure your editor auto-formats with black + isort

### Timeline
- **Gate 0**: 2026-02-04 to 2026-02-07 (estimated)
- **Tranche 1**: 3 weeks starting after Gate 0 completion

### Questions?
- Read: `docs/architecture/TRANCHE_EXECUTION_PLAN.md`
- Escalate: @transformation-portal-architect
```

---

## Approval and Effective Date

**Approved by**: Transformation Portal Architect
**Date**: 2026-02-04
**Effective**: Immediate
**Review**: After Tranche 1 completion (3 weeks post-Gate 0)

**Binding status**: This directive is enforceable under the Transformation Portal Architect authority scope defined in `docs/architecture/agent_governance.md`.

**Amendments**: Any changes require explicit Architect approval and documented rationale (ADR if architectural impact).

---

**Next Action**: Execute Gate 0 (use `docs/ci/GATE_0_CHECKLIST.md`)
