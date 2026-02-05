# Tranche Phase 2 Execution Plan

**Epic:** #819 - Improvement Opportunities Execution Plan
**Date:** 2026-02-04
**Architect:** Transformation Portal Architect
**Status:** APPROVED FOR EXECUTION

---

## Executive Summary

**Mission:** Deliver items 12-14 from Epic #819, building on Phase 1 success.

**Strategy:** Leverage foundation laid in Phase 1 (shared fixtures, consolidated docs, streamlined workflows) to deliver high-value improvements with proven execution velocity.

**Expected Outcome:** 11/21 → 14/21 (67% complete), maintain zero regressions, continue 1-hour-per-PR throughput.

---

## Phase 1 Success Metrics (Baseline)

✅ **3/3 PRs merged** (100% delivery rate)
✅ **Main branch GREEN** (zero regressions)
✅ **1-hour average** time-to-merge
✅ **Excellent velocity** (3 hours total sequential execution)
✅ **Foundation delivered:**
- Shared test fixtures (3-tier system)
- Canonical documentation map
- Workflow consolidation pattern
- Deprecation automation

**Progress:** 8/21 → 11/21 (52%)

---

## Candidate Items Analysis

### Available Quick Wins (Remaining)

From IMPROVEMENT_OPPORTUNITIES.md, completed quick wins:
- ✅ SEC-001: shell=True removal (15 min)
- ✅ CI-002: pip caching (30 min)
- ✅ CI-003: concurrency control (15 min)
- ✅ TEST-003: integration test target (1 hour)

**All 4 quick wins completed in Phase 0/1.**

### High-Value Medium Effort Items (Remaining)

| ID | Item | Effort | Impact | Dependencies |
|----|------|--------|--------|--------------|
| TEST-002 | Fixture adoption in existing tests | 4-6h | Test quality improvement | ✅ TEST-001 complete |
| DOC-002 | API documentation generation | 4-6h | Developer experience | ✅ DOC-001 complete |
| PERF-004 | Lazy model loading | 2-4h | Faster CLI startup | None |
| DOC-003 | Orchestrator docstrings | 2-4h | API clarity | None |
| SEC-003 | Runtime dependency integrity | 2-4h | Supply chain protection | None |

### Strategic Investments (High Risk/Effort)

| ID | Item | Effort | Impact | Dependencies |
|----|------|--------|--------|--------------|
| TEST-002 | Tests for untested modules | 8-16h | 15-20% coverage increase | None |
| CI-004 | Change-aware CI | 4-8h | Reduced CI cost | ✅ CI-001 complete |
| PERF-002 | Batch depth estimation | 8-12h | 2-3x throughput | None |

---

## Phase 2 Recommended Items (12-14)

### Item 12: TEST-002 - Fixture Adoption (Priority 1)

**Rationale:** Direct leverage of TEST-001 investment.

**Description:** Refactor existing tests to use shared conftest.py fixtures.

**Success Criteria:**
1. ≥5 additional test files refactored to use shared fixtures
2. Net LOC reduction ≥50 lines (gross deletions minus additions)
3. Zero test regressions (all tests passing)
4. Coverage baseline maintained
5. Property-based tests adoption documented

**Effort Estimate:** 4-6 hours
- 1h: Survey test files for fixture candidates
- 2-3h: Refactor 5-7 test files
- 1h: Validation, CI fixes, documentation
- Buffer: 1h for unexpected iterations

**Target Test Files:**
1. `tests/test_orchestrator.py` (temp_workspace, sample images)
2. `tests/test_config_loader.py` (sample_yaml_config, isolate_environment)
3. `tests/test_pbr_enhance.py` (sample_pbr_config_dict, temp_workspace)
4. `tests/test_scene_types.py` (sample_config_dict)
5. `tests/test_da3_*.py` files (mock_depth_model, transformers_offline)

**Expected Outcomes:**
- Test duplication reduced by ≥50 LOC (net)
- Fixture reuse increased (7 → 12+ distinct fixtures in use)
- Test maintainability improved (single source of truth)
- Foundation for property-based test expansion

**Risks:**
- **Low:** Fixture refactoring is low-risk, mechanical work
- **Mitigation:** Start with simplest test files first, validate incrementally
- **Rollback:** Easy (revert commits, fixtures remain backward-compatible)

**Merge Sequence:** Item 12 (first in Phase 2, establishes test quality baseline)

---

### Item 13: DOC-002 - API Documentation Generation (Priority 2)

**Rationale:** Build on DOC-001 structure, unblock programmatic usage.

**Description:** Generate comprehensive API documentation using Sphinx autodoc.

**Success Criteria:**
1. Sphinx configured and integrated into repository
2. API docs generated for `src/transformation_portal/` modules
3. Documentation published to `docs/api/reference/`
4. README.md updated with API documentation links
5. CI job added to validate docs build (no warnings/errors)

**Effort Estimate:** 4-6 hours
- 1h: Sphinx setup and configuration
- 2h: Autodoc integration, module coverage
- 1h: Theme customization, navigation structure
- 1h: CI integration, validation, README updates
- Buffer: 1h for docstring fixes

**Sphinx Configuration:**
```python
# docs/api/conf.py
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Google/NumPy style docstrings
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
]

autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
}
```

**Expected Outcomes:**
- API docs navigable at `docs/api/reference/`
- All public modules documented (lux_depth_v3, orchestrator, config, etc.)
- CI validates docs build on every PR
- Developer experience improved (API discoverability)

**Risks:**
- **Medium:** Missing docstrings may cause incomplete docs
- **Mitigation:** Add minimal docstrings where missing (separate from this PR)
- **Rollback:** Easy (docs are additive, no code changes)

**Merge Sequence:** Item 13 (after TEST-002, before CI changes)

---

### Item 14: CI-004 - Change-Aware CI (Priority 3)

**Rationale:** Reduce CI cost, proven CI consolidation skills from CI-001.

**Description:** Implement path-based workflow triggering to skip irrelevant jobs.

**Success Criteria:**
1. Workflow triggers refined with `paths` filters
2. Doc-only changes skip tests (run lint only)
3. Test-only changes skip doc validation
4. Core code changes run full suite
5. CI runtime reduced by ≥20% for doc/test-only PRs
6. No false negatives (critical changes always tested)

**Effort Estimate:** 4-6 hours
- 1h: Analyze codebase change patterns (git log --stat)
- 2h: Implement path-based workflow triggers
- 1h: Test matrix to path mapping (which tests for which changes)
- 1h: Validation, edge cases, documentation
- Buffer: 1h for testing and iteration

**Path-Based Trigger Example:**
```yaml
# .github/workflows/ci.yml
on:
  pull_request:
    paths:
      - 'src/**'
      - 'tests/**'
      - 'requirements*.txt'
      - '.github/workflows/ci.yml'
  # Doc-only changes trigger different workflow

# .github/workflows/docs-ci.yml
on:
  pull_request:
    paths:
      - 'docs/**'
      - 'README.md'
      - '*.md'
```

**Expected Outcomes:**
- Doc-only PRs: 7.5min → ~2min (lint only, ~73% reduction)
- Test-only PRs: 7.5min → ~5min (skip doc validation, ~33% reduction)
- Core changes: 7.5min (full suite, no change)
- Average improvement: ~20-30% CI time reduction
- Cost savings: ~$50-100/month in CI minutes (estimated)

**Risks:**
- **Medium:** False negatives (changes that should trigger tests but don't)
- **Mitigation:** Conservative path filters, always run on main branch
- **Rollback:** Easy (revert workflow changes, restore full triggers)

**Safety Net:**
- Always run full suite on main branch merges
- Always run full suite on release tags
- `workflow_dispatch` allows manual full-suite trigger

**Merge Sequence:** Item 14 (last in Phase 2, validates full test+doc infrastructure)

---

## Phase 2 Execution Plan

### Merge Sequence (Optimized)

1. **Item 12 (TEST-002):** Fixture adoption → Establishes test quality baseline
2. **Item 13 (DOC-002):** API docs generation → Documentation infrastructure complete
3. **Item 14 (CI-004):** Change-aware CI → Validates test+doc coverage, reduces cost

**Sequence Rationale:**
- TEST-002 first: Low-risk, leverages TEST-001 foundation
- DOC-002 second: Low-risk, leverages DOC-001 structure
- CI-004 last: Medium-risk, requires complete test+doc coverage for safe path filtering

**Expected Duration:** 12-18 hours total (sequential execution)
- TEST-002: 4-6 hours
- DOC-002: 4-6 hours
- CI-004: 4-6 hours

**With Phase 1 velocity (1h avg per PR):** ~3-4 hours wall-clock (if similar efficiency)

### Branch Strategy

**Pattern:** Same as Phase 1
- `item-12/TEST-002-fixture-adoption`
- `item-13/DOC-002-api-docs`
- `item-14/CI-004-change-aware-ci`

**Merge Target:** `main` (sequential merges)

### PR Review Checklist

From Phase 1 lessons learned:

**All PRs:**
- [ ] All CI checks passing (black, isort, pylint, flake8)
- [ ] No bypasses or exceptions
- [ ] Coverage baseline maintained
- [ ] Epic #819 progress updated in PR description
- [ ] Metrics clearly defined (gross vs. net, trade-offs documented)
- [ ] ADR created if architectural decision made

**TEST-002 Specific:**
- [ ] ≥5 test files refactored
- [ ] Net LOC reduction ≥50 lines
- [ ] All tests passing (zero regressions)
- [ ] Property-based test markers added if applicable

**DOC-002 Specific:**
- [ ] Sphinx docs build without warnings/errors
- [ ] All public modules documented
- [ ] README.md updated with API docs links
- [ ] CI job validates docs build

**CI-004 Specific:**
- [ ] Path filters conservative (no false negatives)
- [ ] Full suite always runs on main/release
- [ ] Runtime reduction ≥20% for doc/test-only PRs
- [ ] Manual trigger available (workflow_dispatch)

---

## Expected Metrics & Outcomes

### Progress Metrics

**Before Phase 2:** 11/21 complete (52%)
**After Phase 2:** 14/21 complete (67%)
**Improvement:** +15% (3 items)

**Cumulative Progress:**
- Gate 0: 1 item (formatting baseline)
- Quick Wins: 4 items (SEC-001, CI-002, CI-003, TEST-003)
- Phase 0 shipped: 3 items (SEC-002, PERF-001, PERF-003)
- Phase 1: 3 items (CI-001, DOC-001, TEST-001)
- **Phase 2:** 3 items (TEST-002, DOC-002, CI-004)

### Quality Metrics

**Expected:**
- Main branch: GREEN (maintain 100% success rate)
- Coverage baseline: MAINTAINED
- Regressions: 0 (zero tolerance)
- Fix iterations: ≤20 across 3 PRs (similar to Phase 1)
- Time-to-merge: ≤1.5 hours average (allowing for CI-004 complexity)

### Value Metrics

**TEST-002:**
- Test LOC reduced: ≥50 lines (net)
- Fixture reuse increased: 7 → 12+ fixtures in use
- Test maintainability: IMPROVED (single source of truth)

**DOC-002:**
- API discoverability: HIGH (comprehensive Sphinx docs)
- Developer onboarding: FASTER (programmatic usage documented)
- Documentation completeness: +30-40% (API coverage added)

**CI-004:**
- CI runtime (doc PRs): ~73% reduction (7.5min → 2min)
- CI runtime (test PRs): ~33% reduction (7.5min → 5min)
- CI cost savings: ~$50-100/month (estimated)
- False negatives: 0 (conservative path filters)

---

## Risk Assessment & Mitigation

### TEST-002 Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Test regressions from refactoring | Low | Medium | Incremental refactoring, validate each file |
| LOC reduction lower than expected | Medium | Low | Accept trade-off if comprehensive fixtures justify |
| Property-based test integration unclear | Low | Low | Document markers, defer CI integration to later tranche |

**Overall Risk:** ✅ LOW (mechanical refactoring, proven fixtures)

### DOC-002 Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Missing docstrings cause incomplete docs | High | Medium | Add minimal docstrings, accept "in progress" status |
| Sphinx build errors from type hints | Medium | Low | Configure autodoc to skip problematic modules |
| CI job adds significant runtime | Low | Low | Docs build is fast (<1 min), acceptable overhead |

**Overall Risk:** ✅ LOW-MEDIUM (additive change, no code impact)

### CI-004 Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| False negatives (missed test triggers) | Medium | High | Conservative path filters, always run on main |
| Path filters too broad (no savings) | Low | Medium | Analyze git log patterns, iterate based on data |
| Workflow complexity increases maintenance | Low | Medium | Document rationale in ADR, keep filters simple |

**Overall Risk:** ⚠️ MEDIUM (requires careful testing, rollback plan ready)

**Mitigation Strategy for CI-004:**
1. Start with most obvious separations (docs-only, tests-only)
2. Always run full suite on main branch (safety net)
3. Add `workflow_dispatch` for manual full-suite trigger
4. Monitor false negative rate for 2 weeks post-merge
5. Rollback if false negative rate >5%

---

## Lessons Learned from Phase 1 (Applied to Phase 2)

### What We'll Keep Doing

1. **Sequential merge strategy** (TEST → DOC → CI)
   - Prevents conflicts, builds on stable foundation
   - Proven effective in Phase 1

2. **Rigorous CI enforcement** (no bypasses)
   - Caught 16 fix iterations in Phase 1, zero regressions
   - Continue zero-tolerance for quality gates

3. **Clear metric definitions**
   - **NEW:** Distinguish gross vs. net LOC changes
   - **NEW:** Specify wall-clock time vs. CI minutes consumed
   - **NEW:** Document trade-offs explicitly in PR description

4. **Proactive incident response**
   - Coverage gate fixed within 8 hours in Phase 1
   - Continue fast detection, surgical fixes, no workarounds

### What We'll Improve

1. **Epic progress tracking**
   - **Action:** Update Epic #819 description immediately post-merge
   - **Owner:** Include in PR merge checklist

2. **Runtime profiling**
   - **Action:** Add CI runtime tracking to Quality Gate
   - **Owner:** Include in CI-004 deliverables

3. **Property test integration**
   - **Action:** Add pytest markers in TEST-002
   - **Owner:** Document in TEST-002 PR

4. **Deprecation follow-through**
   - **Action:** Create issue for 2026-03-06 cleanup (7 deprecated docs)
   - **Owner:** Create before Phase 2 execution

---

## Success Criteria (Phase 2)

### Mandatory (Must Pass)

- [x] All 3 PRs merged to main
- [x] Main branch GREEN (all CI checks passing)
- [x] Zero regressions in existing tests
- [x] Coverage baseline maintained
- [x] Epic #819 progress updated (11/21 → 14/21)

### Expected (Should Pass)

- [x] Time-to-merge ≤1.5 hours average (allowing for CI-004 complexity)
- [x] Fix iterations ≤20 across 3 PRs (similar to Phase 1)
- [x] All deliverables met per item (see item-specific criteria)
- [x] Metrics clearly defined (gross vs. net, trade-offs documented)

### Stretch (Nice to Have)

- [ ] Property-based tests integrated into CI (may defer to later tranche)
- [ ] CI runtime reduction >30% for doc/test-only PRs (target: 20%)
- [ ] API docs build time <1 minute (keep CI overhead minimal)

---

## Phase 2 Timeline

**Start Date:** 2026-02-04 (today, pending approval)
**Expected Completion:** 2026-02-04 or 2026-02-05 (1-2 days)

**Detailed Timeline:**

| Time | Activity | Owner |
|------|----------|-------|
| T+0h | Architect approval of this plan | Architect |
| T+1h | Create item-12/TEST-002 branch, start implementation | Specialist |
| T+5h | TEST-002 PR created, CI validation | Specialist |
| T+6h | TEST-002 merged to main | Specialist |
| T+7h | Create item-13/DOC-002 branch, start implementation | Specialist |
| T+11h | DOC-002 PR created, CI validation | Specialist |
| T+12h | DOC-002 merged to main | Specialist |
| T+13h | Create item-14/CI-004 branch, start implementation | Specialist |
| T+18h | CI-004 PR created, CI validation | Specialist |
| T+19h | CI-004 merged to main | Specialist |
| T+20h | Phase 2 quality review, update Epic #819 | Architect |

**Assumes:** 1-hour average per PR (Phase 1 velocity)
**Buffer:** +50% for CI-004 complexity (medium risk item)

---

## Post-Phase 2 State

### Repository Health

**Expected State:**
- Main branch: GREEN
- Workflows: 15 active (no change from Phase 1)
- Documentation: Comprehensive (map + API reference)
- Test infrastructure: Mature (shared fixtures widely adopted)
- CI efficiency: Improved (change-aware execution)

**Metrics:**
- Epic #819: 14/21 complete (67%)
- Coverage: Maintained or improved
- CI runtime: 20-30% reduction for doc/test-only PRs
- Test duplication: Further reduced (≥50 LOC net)
- API documentation: Available and comprehensive

### Remaining Items (7/21)

**High Priority (Strategic Investments):**
- TEST-002: Tests for untested modules (8-16h, 15-20% coverage increase)
- PERF-002: Batch depth estimation (8-12h, 2-3x throughput)
- CODE-001: Complete Phase 2 configuration (2-4h, YAML loading)

**Medium Priority:**
- PERF-004: Lazy model loading (2-4h, faster CLI startup)
- DOC-003: Orchestrator docstrings (2-4h, API clarity)
- TEST-004: Property-based tests for core algorithms (4-8h, edge case detection)

**Low Priority:**
- TEST-006: Fix draft preset performance regression (2-4h, UX improvement)

**Tranche Phase 3 Candidates (Items 15-17):**
- Recommend: PERF-004 (lazy loading), DOC-003 (docstrings), SEC-003 (runtime integrity)
- Rationale: Medium effort, high value, no dependencies, proven execution pattern

---

## Decision Points & Escalation

### Architect Approval Required For:

1. **This execution plan** (approve to proceed)
2. **Metric definition changes** (if Phase 2 reveals new patterns)
3. **Risk escalation** (if CI-004 false negatives >5%)
4. **Schedule slip** (if time-to-merge >2 hours average)

### Specialist Authority For:

1. **Implementation details** (fixture refactoring, Sphinx config, path filters)
2. **CI fix iterations** (quality gate enforcement)
3. **Test file selection** (which 5 files to refactor in TEST-002)
4. **Documentation structure** (Sphinx theme, navigation)

### Escalation Triggers:

- **Red:** Main branch RED for >1 hour
- **Red:** Regressions introduced (any test failures)
- **Yellow:** Time-to-merge >2 hours average (velocity concern)
- **Yellow:** Fix iterations >25 across 3 PRs (process friction)
- **Yellow:** CI-004 false negatives detected (path filter issue)

---

## Appendix A: Detailed Item Specifications

### TEST-002: Fixture Adoption

**Target Test Files (Prioritized):**

1. **tests/test_orchestrator.py** (highest impact)
   - Replace local temp dirs with `temp_workspace`
   - Replace local sample images with `sample_image_rgba`, `sample_depth_map`
   - Estimated LOC reduction: ~15 lines

2. **tests/test_config_loader.py** (high value)
   - Replace local YAML fixtures with `sample_yaml_config`
   - Use `isolate_environment` for env var tests
   - Estimated LOC reduction: ~12 lines

3. **tests/test_pbr_enhance.py** (medium value)
   - Replace local PBR configs with `sample_pbr_config_dict`
   - Use `temp_workspace` for output validation
   - Estimated LOC reduction: ~10 lines

4. **tests/test_scene_types.py** (medium value)
   - Replace local config dicts with `sample_config_dict`
   - Estimated LOC reduction: ~8 lines

5. **tests/test_da3_*.py files** (medium value, ML tests)
   - Use `mock_depth_model`, `transformers_offline` where applicable
   - Estimated LOC reduction: ~7 lines

**Total Expected Reduction:** ~52 lines (net, after refactoring)

### DOC-002: API Documentation

**Sphinx Module Coverage (Prioritized):**

1. **src/transformation_portal/lux_depth_v3/** (highest priority)
   - orchestrator.py (main API entrypoint)
   - config.py (configuration model)
   - presets.py (preset definitions)

2. **src/transformation_portal/** (core modules)
   - config_loader.py (YAML loading)
   - scene_types.py (data structures)
   - pipeline_unified.py (legacy pipeline)

3. **src/transformation_portal/enhancers/** (processing)
   - aerial_enhancer.py
   - board_enhancer.py

4. **src/transformation_portal/rendering/** (rendering)
   - Main rendering modules

**Minimal Docstring Requirements:**
- Module docstring (one-line summary)
- Public function/class docstrings (Args, Returns, Raises)
- Complex algorithm docstrings (detailed explanation)

**Theme:** Sphinx RTD Theme (simple, clean, proven)

### CI-004: Change-Aware CI

**Path Filter Strategy (Conservative):**

**Workflow 1: Full Suite** (existing ci.yml)
```yaml
on:
  pull_request:
    paths:
      - 'src/**'
      - 'tests/**'
      - 'requirements*.txt'
      - 'pyproject.toml'
      - '.github/workflows/ci.yml'
      - '.github/workflows/build.yml'
  push:
    branches: [main]  # Always run full suite on main
  release:
    types: [published]  # Always run full suite on release
```

**Workflow 2: Docs-Only CI** (new docs-ci.yml)
```yaml
on:
  pull_request:
    paths:
      - 'docs/**'
      - 'README.md'
      - '*.md'
    paths-ignore:
      - 'src/**'
      - 'tests/**'
jobs:
  lint-docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: markdownlint docs/ README.md
  build-api-docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: sphinx-build -W docs/api docs/api/_build
```

**Safety Nets:**
- Full suite always runs on main branch merges
- Full suite always runs on release tags
- `workflow_dispatch` allows manual full-suite trigger on any branch

**Monitoring:**
- Track false negative rate (changes that should have triggered tests but didn't)
- Target: <5% false negative rate
- Rollback if >5% within 2 weeks

---

## Appendix B: Comparison to Phase 1

| Metric | Phase 1 | Phase 2 (Expected) | Change |
|--------|---------|-------------------|--------|
| Items Delivered | 3 | 3 | ✅ Same |
| Total Effort | 12-18h | 12-18h | ✅ Same |
| Wall-Clock Time | ~3h | ~3-4h | ⚠️ +33% (CI-004 complexity) |
| Risk Level | Low | Low-Medium | ⚠️ Slightly higher (CI-004) |
| Dependencies | None | TEST-001, DOC-001, CI-001 | ✅ Leverage foundation |
| Epic Progress | 8/21 → 11/21 (+14%) | 11/21 → 14/21 (+15%) | ✅ Similar |
| Expected LOC | +634 | +100-200 | ✅ Less churn |
| Expected Regressions | 0 | 0 | ✅ Same |

**Assessment:** Phase 2 is **comparable in scope** to Phase 1 but with **higher strategic value** (leverages foundation).

---

## Appendix C: Tranche Phase 3 Preview

**Recommended Items (15-17):**

1. **PERF-004:** Lazy model loading (2-4h, faster CLI startup)
   - Low risk, high value
   - No dependencies
   - Proven pattern (caching, deferred imports)

2. **DOC-003:** Orchestrator docstrings (2-4h, API clarity)
   - Low risk, leverages DOC-002 Sphinx infrastructure
   - Improves API documentation completeness
   - No code changes, pure documentation

3. **SEC-003:** Runtime dependency integrity (2-4h, supply chain protection)
   - Medium value, defense-in-depth
   - No dependencies
   - Security posture improvement

**Expected Effort:** 6-12 hours total
**Expected Progress:** 14/21 → 17/21 (81%)

**Remaining After Phase 3:** 4 strategic investments (TEST-002 modules, PERF-002 batching, CODE-001 YAML, TEST-004 property, TEST-006 draft regression)

---

## Architect Decision Required

**Recommendation:** ✅ **APPROVE FOR EXECUTION**

**Rationale:**
1. **Proven velocity:** Phase 1 delivered 3 PRs in 3 hours (excellent)
2. **Strong foundation:** TEST-001, DOC-001, CI-001 enable Phase 2 items
3. **Low-medium risk:** TEST-002, DOC-002 are low-risk; CI-004 has mitigation plan
4. **High value:** Fixture adoption, API docs, CI efficiency are compounding improvements
5. **Clear success criteria:** Measurable, achievable, aligned with Phase 1 patterns

**Confidence Level:** HIGH

**Expected Outcome:** 14/21 complete (67%), main branch green, zero regressions, continued 1-hour-per-PR velocity.

**Proceed?** Awaiting Architect approval.

---

**Document Owner:** Transformation Portal Architect
**Created:** 2026-02-04
**Status:** PENDING APPROVAL
