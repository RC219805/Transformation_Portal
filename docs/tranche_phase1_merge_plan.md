# Tranche Phase 1 - PR Merge Analysis
**Generated:** 2026-02-04T09:46:00Z

## Executive Summary

All three PRs have CI failures that need addressing before merge. **None are currently green.**

**Recommended Action:** Fix remaining issues in all three PRs in parallel, then merge in this sequence once all are green:
1. **PR #827** (CI-001) - Foundation workflow changes
2. **PR #826** (DOC-001) - Documentation consolidation
3. **PR #825** (TEST-001) - Test infrastructure

---

## Current CI Status

### PR #825 - TEST-001: Shared conftest.py fixtures
**Branch:** `feature/test-001-shared-fixtures`
**Latest Commit:** 662bd166 (lint/pre-commit hygiene fixes)
**CI State:** ❌ **RED** - 3 failures

**Blocking Failures:**
- `CI Gate` - FAILURE
- `lint` (CI workflow) - FAILURE
- `pre-commit-checks` (Quality Gate) - FAILURE

**Passing Checks:** 21/25 workflows green (84%)

**Recent Fixes Applied:**
- Removed unused imports (settings, HealthCheck)
- Fixed temp_workspace docstring (added 'root' key)
- Made mock_depth_model deterministic

**Remaining Issues:** Lint/pre-commit failures need investigation

---

### PR #826 - DOC-001: Documentation consolidation
**Branch:** `feature/doc-001-consolidation`
**Latest Commit:** e5ccb224 (broken links + UTF-8 encoding)
**CI State:** ⚠️ **MOSTLY GREEN** - 2 failures, 3 in progress

**Blocking Failures:**
- `pre-commit-checks` (Quality Gate) - FAILURE
- `Golden Regression Tests` (Enforcement) - FAILURE
- `Layer 1 Tests (Fast)` (Enforcement) - FAILURE

**In Progress:** 3 test jobs still running (3.11/3.12 core/ml)

**Recent Fixes Applied:**
- Fixed broken link paths (ci/ → ci_cd/, fixes/TIFF* → root TIFF*)
- Added UTF-8 encoding to deprecate_docs.py
- Added missing ARCHITECTURE_PHILOSOPHY entry
- Removed trailing whitespace

**Remaining Issues:** Need to wait for in-progress tests; investigate failures

---

### PR #827 - CI-001 Phase 1: Workflow consolidation
**Branch:** `feature/ci-001-consolidation`
**Latest Commit:** a6dd0d5c (file location + ADR corrections)
**CI State:** ❌ **RED** - 3 failures

**Blocking Failures:**
- `CI Gate` - FAILURE
- `test (3.11, cpu, core)` - FAILURE
- `Golden Regression Tests` (Enforcement) - FAILURE
- `Layer 1 Tests (Fast)` (Enforcement) - FAILURE

**Cancelled:** 2 test jobs (3.11 ml, 3.12 core) - cancelled after first failure

**Recent Fixes Applied:**
- Moved disabled workflow to `.github/workflows-disabled/`
- Moved ADR to canonical location
- Documented Test PyPI publish removal

**Remaining Issues:** Core test failures need investigation; likely related to workflow consolidation changes

---

## Dependency Analysis

### Cross-PR Conflicts
**Status:** ✅ **No direct file conflicts**

- **TEST-001** creates `tests/conftest.py` (new file)
- **DOC-001** modifies documentation files and structure
- **CI-001** modifies workflow files (`.github/workflows/`)

**Git Conflict Risk:** Minimal - no overlapping file edits

### Logical Dependencies
**Status:** ⚠️ **Weak coupling exists**

1. **CI-001 → TEST-001**: Workflow consolidation changes how tests run
   - TEST-001 relies on CI executing tests correctly
   - Risk: If workflow changes break test execution, TEST-001 failures may be misleading

2. **DOC-001 → All**: Documentation changes include CI/test documentation
   - Low risk: mostly README updates

**Recommendation:** Fix and merge CI-001 first to establish stable test execution environment

---

## Risk Assessment

### PR #825 (TEST-001) - **MEDIUM RISK**
**Risk Factors:**
- Test infrastructure changes affect all future PRs
- Lint failures suggest code quality issues remain
- Mock fixture behavior must be deterministic

**Mitigation:**
- Address lint violations completely
- Verify all test tiers (pure/io/ml) work independently
- Run full test suite locally before final push

### PR #826 (DOC-001) - **LOW RISK**
**Risk Factors:**
- Documentation-only changes (no runtime impact)
- Golden regression test failure is unexpected for doc changes
- May indicate test fragility or environmental sensitivity

**Mitigation:**
- Investigate why doc changes trigger golden test failures
- May need to regenerate golden fixtures if docs affect output
- Pre-commit failures may be trailing whitespace or encoding issues

### PR #827 (CI-001) - **HIGH RISK**
**Risk Factors:**
- Workflow consolidation affects all future CI runs
- Test failures in core test suite
- Changes to CI execution may mask/reveal issues in other PRs

**Mitigation:**
- Verify disabled workflow is truly redundant
- Ensure no required checks were lost
- Test locally with `act` or similar workflow runner
- Document workflow behavior changes clearly

---

## Recommended Fix Sequence

### Phase 1: Parallel Investigation (NOW)
Execute these investigations simultaneously to identify root causes:

1. **PR #825**: Check lint/pre-commit failures
   ```bash
   git checkout feature/test-001-shared-fixtures
   pre-commit run --all-files
   flake8 tests/conftest.py
   pytest tests/ -v -ra -m "not ml and not slow"
   ```

2. **PR #826**: Wait for in-progress tests, investigate failures
   ```bash
   git checkout feature/doc-001-consolidation
   pre-commit run --all-files
   # Investigate why golden tests fail on doc changes
   pytest tests/ -v -ra -m "golden"
   ```

3. **PR #827**: Investigate core test failures
   ```bash
   git checkout feature/ci-001-consolidation
   # Run the failing test locally
   pytest tests/ -v -ra -m "not ml and not slow" --maxfail=1
   # Check if workflow consolidation broke test discovery
   ```

### Phase 2: Fix Application (SEQUENTIAL)
Once root causes are identified, apply fixes in this order:

1. **Fix PR #827 first** (CI-001)
   - Rationale: Establishes stable test execution environment
   - Ensures TEST-001 results are reliable
   - Prevents cascading failures in other PRs

2. **Fix PR #826 next** (DOC-001)
   - Rationale: Low-risk, documentation only
   - Validates CI-001 changes work for non-code PRs
   - Builds confidence before test infrastructure changes

3. **Fix PR #825 last** (TEST-001)
   - Rationale: Benefits from stable CI environment (CI-001)
   - Can validate against working baseline (DOC-001 merged)
   - Highest risk, deserves most stable foundation

### Phase 3: Merge Sequence (ONCE ALL GREEN)
**Critical:** Do not merge until ALL three PRs are green.

**Merge Order:**
1. **PR #827** (CI-001) - Foundation
2. **PR #826** (DOC-001) - Validation
3. **PR #825** (TEST-001) - Infrastructure

**Between Each Merge:**
- Wait for main branch CI to go green
- Rebase next PR on updated main
- Re-run CI to verify no conflicts with merged changes
- Only proceed when CI is green

---

## Common Failure Patterns to Investigate

### 1. Pre-commit Hook Failures
**Seen in:** PR #825, PR #826

**Likely Causes:**
- Trailing whitespace (partially fixed in DOC-001)
- Line length violations
- Import sorting issues
- File encoding problems (partially fixed in DOC-001)

**Investigation:**
```bash
pre-commit run --all-files --show-diff-on-failure
```

### 2. Golden Regression Test Failures
**Seen in:** PR #826, PR #827

**Likely Causes:**
- Documentation changes affecting output formatting
- Workflow changes affecting test execution environment
- Flaky test dependencies on execution order

**Investigation:**
```bash
pytest tests/ -v -ra -m "golden" --tb=short
# Check if golden fixtures need regeneration
```

### 3. Layer 1 Fast Test Failures
**Seen in:** PR #826, PR #827

**Likely Causes:**
- Import errors due to structural changes
- Missing test dependencies
- Environment variable assumptions

**Investigation:**
```bash
pytest tests/ -v -ra -m "not ml and not slow" -k "layer1"
```

---

## Post-Merge Actions

Once all three PRs are merged:

### 1. Update Epic #819
- Mark TEST-001, DOC-001, CI-001 as complete
- Update progress: 8/21 → 11/21 (52%)
- Add completion date and PR links

### 2. Close Associated Issues
- Close #825 (referenced by TEST-001 PR)
- Close #826 (referenced by DOC-001 PR)
- Close #818 (referenced by CI-001 PR)

### 3. Verify Integration
```bash
# On main after all merges
git checkout main
git pull origin main
pytest tests/ -v -ra -m "not ml and not slow" --maxfail=1
pre-commit run --all-files
```

### 4. Update Documentation
- Add entry to CHANGELOG.md for Tranche Phase 1 completion
- Update any affected architecture docs
- Verify DOCUMENTATION_MAP.md is accurate

---

## Validation Checklist

Before declaring Tranche Phase 1 complete:

- [ ] All three PRs show green CI
- [ ] No merge conflicts between PRs
- [ ] Main branch CI green after each merge
- [ ] Test suite passes locally on main
- [ ] Pre-commit hooks pass on main
- [ ] Epic #819 updated
- [ ] Issues #825, #826, #818 closed
- [ ] CHANGELOG.md updated
- [ ] No regression in existing functionality

---

## Notes

### Why This Sequence?

**CI-001 First:**
- Workflow changes are foundational
- Other PRs depend on reliable CI execution
- Failures here have cascading impact

**DOC-001 Second:**
- Low risk validation of CI-001 changes
- No code changes to complicate debugging
- Builds confidence before test infrastructure work

**TEST-001 Last:**
- Highest impact on future development
- Deserves most stable environment
- Benefits from validated CI and doc structure

### Parallel vs Sequential

**Investigation Phase:** Parallel is safe and efficient
- No cross-contamination between branches
- Faster time to understanding root causes

**Fix Phase:** Sequential is safer
- Each PR validates the previous one
- Reduces risk of compounding failures
- Easier rollback if issues arise

**Merge Phase:** Strictly sequential
- Prevents merge conflicts
- Ensures stable main branch
- Allows validation at each step

---
