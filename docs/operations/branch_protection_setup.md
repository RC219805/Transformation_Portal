# Branch Protection Configuration Guide

## Overview

This guide documents the **CI Gate pattern**: a stable branch protection approach that doesn't break when the test matrix changes.

## Problem Context

**The Old Way (Fragile):**
When branch protection requires matrix-expanded job names like `test (3.11, cpu, core)`, every matrix change breaks protection:
- Drop Python 3.10 → PRs stuck waiting for `test (3.10, cpu, core)` forever
- Add Python 3.13 → Admin must manually add new check
- Change test types → Admin must update all affected checks

**The New Way (Stable):**
Branch protection requires only `CI Gate`, which aggregates all upstream jobs. Matrix can evolve freely without admin intervention.

---

## CI Gate Pattern

### What is CI Gate?

`CI Gate` is a final aggregator job in `.github/workflows/build.yml` that:
1. **Depends on all critical jobs:** `lint`, `test`, `generate-manifest`
2. **Always runs:** Uses `if: always()` to run even when upstream jobs fail
3. **Checks all results:** Explicitly verifies each upstream job succeeded
4. **Reports one status:** Green if all pass, red if any fail
5. **Has a stable name:** Never changes when matrix evolves

### Implementation

```yaml
ci_gate:
  name: CI Gate
  runs-on: ubuntu-24.04
  needs: [lint, test, generate-manifest]
  if: ${{ always() }}
  timeout-minutes: 5

  steps:
    - name: Summarize upstream results
      run: |
        echo "lint result:  ${{ needs.lint.result }}"
        echo "test result:  ${{ needs.test.result }}"
        echo "manifest result: ${{ needs['generate-manifest'].result }}"

    - name: Enforce pass/fail
      run: |
        ok="true"
        if [ "${{ needs.lint.result }}" != "success" ]; then
          echo "❌ lint did not succeed"
          ok="false"
        fi
        if [ "${{ needs.test.result }}" != "success" ]; then
          echo "❌ test matrix did not succeed"
          ok="false"
        fi
        if [ "${{ needs['generate-manifest'].result }}" != "success" ]; then
          echo "❌ generate-manifest did not succeed"
          ok="false"
        fi
        if [ "$ok" != "true" ]; then
          echo "CI Gate: FAILED"
          exit 1
        fi
        echo "CI Gate: PASSED"
```

### Key Design Points

- **`if: always()`**: Gate runs even when upstream jobs fail/skip
- **Bracket syntax**: `needs['generate-manifest']` handles hyphenated job IDs
- **Explicit checks**: Each upstream job's result is checked for `success`
- **Clear messages**: Failure output shows which job failed

---

## Current Workflow Matrix (build.yml)

The CI Gate aggregates these jobs:

### Test Matrix
- `test (3.11, cpu, core)` - Core tests on Python 3.11
- `test (3.12, cpu, core)` - Core tests on Python 3.12
- `test (3.11, cpu, ml)` - ML tests on Python 3.11

### Other Jobs
- `lint` - Linting with Python 3.12
- `generate-manifest` - Montecito manifest generation

**You don't need to track these individually** — CI Gate does it for you.

## Required Status Checks Configuration

### Navigate to Settings
1. Go to repository: https://github.com/RC219805/Transformation_Portal
2. Click **Settings** → **Branches** → **main** (edit rule)
3. Scroll to **Require status checks to pass before merging**

### Enable CI Gate (Stable Aggregator)

**Require this check only:**
- ✅ `CI Gate`

**DO NOT require matrix-expanded checks:**
- ❌ `test (3.11, cpu, core)` — aggregated by CI Gate
- ❌ `test (3.12, cpu, core)` — aggregated by CI Gate
- ❌ `test (3.11, cpu, ml)` — aggregated by CI Gate
- ❌ `lint` — aggregated by CI Gate
- ❌ `generate-manifest` — aggregated by CI Gate
- ❌ `test (3.10, cpu, core)` — legacy check (Python 3.10 no longer supported)

**Why only CI Gate?**
- CI Gate depends on all critical jobs
- When matrix changes (add/remove Python versions, test types), CI Gate stays stable
- No more admin intervention when matrix evolves
- Clear failure reporting: CI Gate output shows which upstream job failed

### Additional Settings

**Recommended settings:**
- ✅ Require branches to be up to date before merging
- ✅ Require linear history (optional, based on team preference)
- ✅ Include administrators (enforce rules for all users)

## Verification

After updating branch protection:

1. Create a test PR or view an existing PR
2. Check that `CI Gate` appears in required status checks
3. Verify CI Gate runs and reports pass/fail based on upstream jobs
4. Click on CI Gate to view upstream job results
5. Confirm no "Expected — Waiting for status" issues

### Expected Behavior

**When all jobs pass:**
- CI Gate shows ✅ green
- PR is mergeable

**When any job fails:**
- CI Gate shows ❌ red
- CI Gate output identifies which job(s) failed
- PR is blocked

**When matrix changes (future):**
- CI Gate name stays stable
- No admin action needed
- Branch protection continues to work

## ci.yml Workflow Status

**Current Status:** Has expression error on line 214
**Issue:** Job names cannot use `env` context variables
**Recommendation:** Either fix the workflow or exclude from required checks until fixed

The `ci.yml` workflow duplicates functionality in `build.yml` and may be deprecated. Until resolved:
- Do **not** require any `ci.yml` checks in branch protection
- The primary workflow `build.yml` provides sufficient coverage

## Maintenance

### When Updating the Test Matrix

The CI Gate pattern **eliminates most maintenance**:

**Matrix changes that require NO admin action:**
- ✅ Add/remove Python versions (e.g., add 3.13, drop 3.11)
- ✅ Add/remove test types (e.g., add GPU tests)
- ✅ Add/remove device types (e.g., add `cuda`)
- ✅ Change matrix dimensions or combinations

**CI Gate changes that DO require admin action:**
- Only if you add/remove entire job categories (not matrix dimensions)
- Example: Add a new top-level job like `security-scan` and want CI Gate to require it

### Adding a New Job to CI Gate

If you add a new critical job that CI Gate should aggregate:

1. Add the job to `.github/workflows/build.yml`
2. Add the job name to `ci_gate.needs` array
3. Add result check in the "Enforce pass/fail" step
4. Update this documentation

**No branch protection changes needed** — CI Gate already required.

### Removing a Job from CI Gate

If you want to make a job optional (e.g., make `generate-manifest` non-blocking):

1. Remove job name from `ci_gate.needs` array in `.github/workflows/build.yml`
2. Remove result check from "Enforce pass/fail" step
3. Update this documentation

**No branch protection changes needed** — CI Gate already required.

### Matrix Change Checklist

When updating test matrix dimensions (Python versions, devices, test types):

- [ ] Update matrix in `.github/workflows/build.yml`
- [ ] Update this documentation (for reference)
- [ ] Test on draft PR
- [ ] Merge without admin intervention (CI Gate handles it)

When adding/removing top-level jobs:

- [ ] Update workflow in `.github/workflows/build.yml`
- [ ] Update `ci_gate.needs` and enforcement logic if critical
- [ ] Update this documentation
- [ ] Test on draft PR
- [ ] Announce to team if check categories change

## Historical Context

### CI Gate Pattern Introduction
- **Date:** February 2026
- **Reason:** Prevent branch protection breakage when test matrix evolves
- **Trigger:** PR #799 stuck on "Expected — Waiting for status" after Python 3.10 removal
- **Solution:** Stable aggregator job pattern used by major projects (Kubernetes, Terraform, etc.)

### Python 3.10 Deprecation (Catalyst Event)
- **Dropped in commits:** `99eb8341`, `82f7f92a`
- **Reason:** Python 3.10 EOL October 2026, repository standardizing on 3.11+
- **Date:** November 2024
- **Impact:** Exposed fragility of matrix-expanded check names in branch protection
- **Resolution:** CI Gate pattern prevents recurrence

## Related Files

- `.github/workflows/build.yml` - Primary CI workflow (source of truth for test matrix)
- `.github/workflows/ci.yml` - Legacy/duplicate workflow (currently broken)
- `README.md` - Documents Python 3.11+ requirement
- `pyproject.toml` - Declares `python_requires = ">=3.11"`

## Contact

For questions about CI/CD configuration, consult the repository maintainers or create an issue with the `ci/cd` label.
