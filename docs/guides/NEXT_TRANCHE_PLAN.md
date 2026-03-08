# Next Tranche Plan (8/21 → 11/21)

**Date:** 2026-02-04
**Current Progress:** 8/21 completed (38%)
**Epic:** #819

---

## Current State

### ✅ Quick Wins Complete (4/4)
All quick wins from the improvement opportunities epic are shipped:
- SEC-001: shell=True fix (PR #820)
- CI-002: pip cache (PR #820)
- CI-003: concurrency control (PR #821) ← **Just closed**
- TEST-003: test-integration target (PR #820)

### ✅ Additional Items Shipped (4)
- SEC-002: Input validation for config loader (PR #820)
- PERF-001: Complete YAML loading implementation (PR #820)
- PERF-003: xxHash as default hasher (PR #820)
- (One additional item from PR #820)

---

## Next Three Items (Highest Leverage)

### 1. TEST-001: Shared conftest.py Fixtures
**Priority:** HIGH
**Effort:** ~4 hours
**Impact:** Reduces test duplication, speeds new test creation

**Why This Matters:**
- Current tests duplicate fixture setup across modules
- New tests copy-paste existing patterns
- Maintenance burden multiplies with each new test file

**Deliverables:**
- Create `tests/conftest.py` with shared fixtures
- Migrate common fixtures from individual test files
- Document fixture usage patterns
- Reduce test boilerplate by ~30%

**Success Criteria:**
- At least 5 fixtures moved to shared conftest
- Zero test failures after migration
- Documentation updated with fixture reference

---

### 2. CI-001 (Phase 1): Workflow Consolidation
**Priority:** HIGH
**Effort:** ~6 hours
**Impact:** Reduced CI complexity, faster maintenance

**Scope (Incremental, Not Boil-the-Ocean):**
- Identify one duplicated lint/test path
- Consolidate into single workflow with matrix
- Preserve all existing coverage
- Document consolidation pattern

**Example Target:**
- Multiple workflows running flake8/pylint/bandit separately
- Consolidate to single "lint" workflow with job matrix

**Success Criteria:**
- At least 2 workflows merged into 1
- Total workflow count reduced
- CI runtime unchanged or improved
- No loss of coverage

---

### 3. DOC-001: Documentation Consolidation Quick Win
**Priority:** HIGH
**Effort:** ~3 hours
**Impact:** Improved discoverability, reduced confusion

**Tasks:**
- Remove duplicate documentation files
- Fix empty `docs/api` directory (remove or populate)
- Consolidate overlapping guides (setup, quickstart, etc.)
- Update DOCUMENTATION_INDEX.md

**Current Pain Points:**
- 5+ "setup" or "quickstart" guides with overlapping content
- Empty directories mislead users
- Hard to find canonical documentation

**Success Criteria:**
- At least 3 duplicate docs removed or merged
- All directories either populated or removed
- Single source of truth for each topic
- DOCUMENTATION_INDEX.md reflects current state

---

## Execution Strategy

### Cadence
- **1 item per week** (realistic pace with thorough testing)
- **PR per item** (keep changes reviewable)
- **Epic updates** (keep #819 progress visible)

### Order
1. **Week 1:** TEST-001 (foundation for better testing going forward)
2. **Week 2:** DOC-001 (quick win, high visibility)
3. **Week 3:** CI-001 Phase 1 (builds on CI-003 foundation)

### Quality Gates
Each PR must:
- ✅ Pass all existing CI checks
- ✅ Include tests (where applicable)
- ✅ Update documentation
- ✅ Reference epic #819

---

## Rationale: Why These Three?

### They're Force Multipliers
- **TEST-001:** Makes writing new tests easier (compounds over time)
- **CI-001:** Makes maintaining workflows easier (reduces cognitive load)
- **DOC-001:** Makes understanding the repo easier (reduces onboarding friction)

### They Build on CI-003
- CI-003 cleaned up concurrency waste
- CI-001 will reduce workflow sprawl
- Together they create a tighter, faster CI pipeline

### They Set Up Future Work
- Better tests → easier to validate refactors
- Cleaner CI → easier to add new checks
- Consolidated docs → easier to maintain as features evolve

---

## What's Not in This Tranche

**Deferred to Later:**
- TEST-002 (Add tests for untested modules) - depends on TEST-001
- PERF-002 (Batch depth processing) - larger refactor
- CI-001 Phase 2+ (full workflow consolidation) - iterative approach

**Reason:**
These three items (TEST-001, DOC-001, CI-001 Phase 1) move the repo from "more features" to "higher throughput" without introducing risk or scope creep.

---

## References

- **Epic:** #819
- **Source:** `docs/guides/IMPROVEMENT_OPPORTUNITIES.md`
- **CI-003 Completion:** `docs/ci/CI_003_COMPLETION.md`
- **Current State:** All quick wins complete (8/21 total)

---

## Next Steps

1. Create issue for TEST-001
2. Start implementation
3. Keep epic #819 updated with progress
4. Repeat for DOC-001 and CI-001 Phase 1

**Target:** 11/21 completed by end of month (52% progress)
