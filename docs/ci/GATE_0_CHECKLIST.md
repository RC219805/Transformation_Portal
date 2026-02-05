# Gate 0: CI Baseline Stabilization Checklist

**Priority**: P0 (blocks all tranche work)  
**Owner**: Repository Maintainer  
**Timeline**: 2-3 days  
**Epic**: #819

---

## Context

**Problem**: Repository has accumulated 265 files with formatting drift (black/isort non-compliance). While CI is currently green on main, this creates a ticking time bomb where future PRs will fail lint if strict enforcement is added.

**Solution**: Apply one-time baseline formatting to establish clean slate for future PRs.

**References**:
- `docs/architecture/TRANCHE_EXECUTION_PLAN.md`: Full context
- `docs/ci/WORKFLOW_MATRIX.md`: CI workflow documentation
- `docs/ci/TYPE_CHECKING_POLICY.md`: Type enforcement strategy

---

## Day 1: Baseline Formatting

### 1.1 Pre-Flight Checks

- [ ] Verify current branch is up-to-date with main:
  ```bash
  git checkout main
  git pull origin main
  ```

- [ ] Verify test suite passes before formatting:
  ```bash
  pytest -v tests/ -ra -m "not ml and not slow" --maxfail=1
  ```
  **Expected**: All tests pass (establishes baseline)

- [ ] Measure formatting drift:
  ```bash
  black --check src/ tests/ 2>&1 | grep -c "would reformat"
  isort --check-only src/ tests/ 2>&1 | grep -c "ERROR"
  ```
  **Record counts in PR description**

### 1.2 Apply Baseline Formatting

- [ ] Create feature branch:
  ```bash
  git checkout -b chore/baseline-formatting
  ```

- [ ] Apply black formatting:
  ```bash
  black src/ tests/
  ```

- [ ] Apply isort formatting:
  ```bash
  isort src/ tests/
  ```

- [ ] Verify formatting compliance:
  ```bash
  black --check src/ tests/
  isort --check-only src/ tests/
  ```
  **Expected**: Both commands exit 0 (no errors)

### 1.3 Verify No Logic Changes

- [ ] Run full test suite (including ML tests if possible):
  ```bash
  # Core tests
  pytest -v tests/ -ra -m "not ml and not slow"
  
  # ML tests (if environment supports)
  pytest -v tests/ -ra -m "ml and not slow"
  ```
  **Critical**: All tests must pass. Formatting should never change logic.

- [ ] Spot-check a few files for unintended changes:
  ```bash
  git diff src/transformation_portal/core/config/validation.py
  git diff src/transformation_portal/config_loader.py
  ```
  **Look for**: Only whitespace/formatting changes, no logic modifications

### 1.4 Create Blame Ignore File

- [ ] Create `.git-blame-ignore-revs`:
  ```bash
  cat > .git-blame-ignore-revs << 'EOF'
  # Baseline formatting: black + isort applied to entire codebase
  # Applied on: 2026-02-04
  # PR: #[PR_NUMBER]
  [COMMIT_SHA_PLACEHOLDER]
  EOF
  ```
  **Note**: Update `[COMMIT_SHA_PLACEHOLDER]` after commit is created

- [ ] Add to git:
  ```bash
  git add .git-blame-ignore-revs
  ```

### 1.5 Commit and Push

- [ ] Commit formatting changes:
  ```bash
  git add src/ tests/ .git-blame-ignore-revs
  git commit -m "chore(format): Apply black + isort baseline (no logic changes)

  - Reformats 265 files to black standards
  - Applies isort to all src/ and tests/ modules
  - Zero functional changes (verified by test suite)
  - Establishes clean baseline for future PRs
  - Adds .git-blame-ignore-revs for git blame filtering

  Part of Gate 0 CI stabilization
  Epic #819"
  ```

- [ ] Push branch:
  ```bash
  git push origin chore/baseline-formatting
  ```

### 1.6 Create Pull Request

- [ ] Create PR via GitHub CLI:
  ```bash
  gh pr create \
    --title "chore(format): Apply black + isort baseline (no logic changes)" \
    --body "## Summary

  This PR applies baseline formatting to the entire codebase to eliminate formatting drift.

  ## Metrics
  - Files reformatted by black: 265
  - Files fixed by isort: [COUNT]
  - Test suite status: ✅ All tests pass

  ## Verification
  - [x] Test suite passes before formatting
  - [x] Test suite passes after formatting
  - [x] Only whitespace/formatting changes (no logic modifications)
  - [x] Added .git-blame-ignore-revs for git blame filtering

  ## Part of
  - Gate 0 CI Stabilization
  - Epic #819

  ## Next Steps
  After merge:
  1. Update .git-blame-ignore-revs with commit SHA
  2. Add black/isort enforcement to CI
  3. Document git blame usage in CONTRIBUTING.md" \
    --base main
  ```

- [ ] Request review from maintainer

- [ ] Wait for CI to pass (should be green)

---

## Day 2: Enforcement and Documentation

### 2.1 Update .git-blame-ignore-revs

- [ ] After PR is merged, get commit SHA:
  ```bash
  git checkout main
  git pull origin main
  git log --oneline -1  # Copy the SHA
  ```

- [ ] Update `.git-blame-ignore-revs`:
  ```bash
  # Replace [COMMIT_SHA_PLACEHOLDER] with actual SHA
  git checkout -b chore/update-blame-ignore-revs
  # Edit .git-blame-ignore-revs manually
  git add .git-blame-ignore-revs
  git commit -m "chore: Update .git-blame-ignore-revs with baseline formatting SHA"
  git push origin chore/update-blame-ignore-revs
  gh pr create --title "chore: Update .git-blame-ignore-revs with baseline formatting SHA" --base main
  ```

### 2.2 Add CI Enforcement

- [ ] Create branch for CI updates:
  ```bash
  git checkout main
  git pull origin main
  git checkout -b chore/enforce-formatting-in-ci
  ```

- [ ] Update `build.yml` to add formatting checks:
  ```yaml
  # Add after lint job or as new job
  format-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - uses: actions/setup-python@v6
        with:
          python-version: '3.12'
      - name: Install formatters
        run: |
          pip install black isort
      - name: Check black formatting
        run: black --check src/ tests/
      - name: Check isort
        run: isort --check-only src/ tests/
  ```

- [ ] Update `quality-gate.yml` (remove autopep8, add black/isort):
  ```yaml
  # Replace autopep8 step with:
  - name: Verify formatting
    run: |
      pip install black isort
      black --check src/ tests/
      isort --check-only src/ tests/
  ```

- [ ] Commit and push:
  ```bash
  git add .github/workflows/build.yml .github/workflows/quality-gate.yml
  git commit -m "chore(ci): Enforce black + isort in CI

  - Adds format-check job to build.yml
  - Updates quality-gate.yml to check (not fix) formatting
  - Ensures all PRs maintain formatting standards

  Part of Gate 0 CI stabilization"
  git push origin chore/enforce-formatting-in-ci
  ```

- [ ] Create PR:
  ```bash
  gh pr create \
    --title "chore(ci): Enforce black + isort in CI" \
    --body "## Summary

  Adds formatting enforcement to CI workflows after baseline formatting is complete.

  ## Changes
  - Adds format-check job to build.yml
  - Updates quality-gate.yml to verify (not auto-fix) formatting
  - Ensures future PRs maintain black + isort standards

  ## Part of
  - Gate 0 CI Stabilization
  - Epic #819

  ## Dependencies
  - Requires baseline formatting PR to be merged first" \
    --base main
  ```

### 2.3 Document Type Checking Decision

- [ ] Review `docs/ci/TYPE_CHECKING_POLICY.md`

- [ ] Make decision on type checking enforcement:
  - **Option A**: Non-blocking (recommended for Gate 0)
  - **Option B**: Narrow blocking (core modules only)
  - **Option C**: Deferred to Tranche 2

- [ ] Update policy document with decision and timeline

- [ ] Commit:
  ```bash
  git checkout -b docs/finalize-type-checking-policy
  git add docs/ci/TYPE_CHECKING_POLICY.md
  git commit -m "docs: Finalize type checking policy for Gate 0"
  git push origin docs/finalize-type-checking-policy
  gh pr create --title "docs: Finalize type checking policy" --base main
  ```

### 2.4 Review PR #822

- [ ] Check out PR #822:
  ```bash
  gh pr checkout 822
  ```

- [ ] Review code changes for coverage artifact fix

- [ ] Verify fix addresses Issue #815

- [ ] Decision:
  - [ ] **Merge**: If fix is correct and tests pass
  - [ ] **Request changes**: If issues found
  - [ ] **Close and create new issue**: If approach is wrong

- [ ] Document decision in PR comment

### 2.5 Update CONTRIBUTING.md

- [ ] Add git blame usage instructions:
  ```markdown
  ## Git Blame and Formatting Commits

  We maintain a `.git-blame-ignore-revs` file to exclude mechanical formatting commits from `git blame`.

  ### Configure git to use ignore-revs:
  ```bash
  git config blame.ignoreRevsFile .git-blame-ignore-revs
  ```

  ### GitHub blame view:
  GitHub automatically respects `.git-blame-ignore-revs` in the web UI.
  ```

- [ ] Commit and push:
  ```bash
  git checkout -b docs/add-blame-ignore-instructions
  git add CONTRIBUTING.md
  git commit -m "docs: Add git blame ignore-revs instructions"
  git push origin docs/add-blame-ignore-instructions
  gh pr create --title "docs: Add git blame ignore-revs instructions" --base main
  ```

---

## Day 3: Verification and Sign-Off

### 3.1 Verify CI Health

- [ ] Check CI status on main branch:
  ```bash
  gh run list --branch main --limit 5
  ```
  **Expected**: All recent runs show success

- [ ] Verify formatting checks are enforced:
  ```bash
  gh run view [LATEST_RUN_ID] --log | grep -E "(black|isort)"
  ```
  **Expected**: black and isort checks run and pass

- [ ] Check for any flaky tests:
  ```bash
  gh run list --workflow="build.yml" --limit 20 --json conclusion
  ```
  **Expected**: No failures or only isolated failures (not systemic)

### 3.2 Local Verification

- [ ] Pull latest main:
  ```bash
  git checkout main
  git pull origin main
  ```

- [ ] Verify formatting compliance:
  ```bash
  black --check src/ tests/
  isort --check-only src/ tests/
  ```
  **Expected**: Both exit 0

- [ ] Run full test suite:
  ```bash
  pytest -v tests/ -ra -m "not ml and not slow" --maxfail=1
  ```
  **Expected**: All tests pass

- [ ] Check coverage (if PR #822 is merged):
  ```bash
  pytest --cov=src/transformation_portal --cov-report=term-missing tests/
  ```
  **Expected**: Coverage report generated successfully

### 3.3 Update Documentation

- [ ] Update `docs/architecture/TRANCHE_EXECUTION_PLAN.md`:
  - [ ] Mark Gate 0 as complete
  - [ ] Add actual completion date
  - [ ] Link to merged PRs

- [ ] Update Epic #819:
  - [ ] Check off Gate 0 completion
  - [ ] Update progress metrics
  - [ ] Announce Tranche 1 start date

### 3.4 Create Gate 0 Completion Report

- [ ] Create issue comment or discussion post:
  ```markdown
  # Gate 0: CI Baseline Stabilization — COMPLETE ✅

  **Completion Date**: 2026-02-[DD]

  ## Delivered
  - ✅ Baseline formatting applied (265 files)
  - ✅ Black + isort enforcement added to CI
  - ✅ .git-blame-ignore-revs configured
  - ✅ Type checking policy documented
  - ✅ PR #822 [merged/closed/superseded]
  - ✅ CI green on main for 3+ consecutive commits

  ## Metrics
  - Formatting drift: 265 → 0 files
  - CI success rate: [X]%
  - Test suite: All passing

  ## Next Steps
  - Start Tranche 1: Week 1 (TEST-001 - Shared Test Fixtures)
  - Target start date: 2026-02-[DD+1]

  ## References
  - Baseline formatting PR: #[NUMBER]
  - CI enforcement PR: #[NUMBER]
  - docs/architecture/TRANCHE_EXECUTION_PLAN.md
  ```

- [ ] Post to Epic #819

### 3.5 Final Sign-Off

- [ ] Architect review and approval:
  - [ ] All success criteria met
  - [ ] CI stable for 3+ commits
  - [ ] Documentation complete
  - [ ] Enforcement mechanisms in place

- [ ] Close Gate 0 issue

- [ ] Create Tranche 1 tracking issue (TEST-001)

---

## Rollback Plan (If Needed)

If formatting PR causes unexpected issues:

1. **Immediate**: Revert the baseline formatting commit:
   ```bash
   git revert [BASELINE_COMMIT_SHA]
   git push origin main
   ```

2. **Investigate**: Identify which files caused issues

3. **Selective fix**: Re-apply formatting only to unproblematic files

4. **Document**: Update this checklist with lessons learned

---

## Success Criteria (Gate 0 Exit)

All must be true to exit Gate 0:

- [x] Main branch CI shows all required checks green for 3+ consecutive commits
- [x] `black --check src/ tests/` exits 0
- [x] `isort --check-only src/ tests/` exits 0
- [x] Type checking policy documented (enforcement may be deferred)
- [x] Coverage gate verified working (PR #822 merged or superseded)
- [x] .git-blame-ignore-revs configured and documented
- [x] CONTRIBUTING.md updated with formatting standards
- [x] Epic #819 updated with Gate 0 completion

---

## References

- Epic #819: Improvement Opportunities Execution Plan
- `docs/architecture/TRANCHE_EXECUTION_PLAN.md`: Full plan
- `docs/ci/WORKFLOW_MATRIX.md`: CI workflow reference
- `docs/ci/TYPE_CHECKING_POLICY.md`: Type enforcement strategy

---

**Tracked by**: [Issue #XXX]  
**Status**: [NOT_STARTED | IN_PROGRESS | COMPLETE]  
**Owner**: Repository Maintainer  
**Architect Approval**: [PENDING | APPROVED]
