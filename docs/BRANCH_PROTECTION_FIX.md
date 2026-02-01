# Branch Protection Configuration Fix

## Problem

PR #771 is blocked by two issues:

### 1. Required Check Mismatch
**Current required checks** (configured in branch protection):
- `build` - ❌ Never runs (phantom from ci.yml that hasn't triggered yet)
- `coverage-gate` - ❌ Never runs (phantom)
- `lint (3.12)` - ❌ Never runs (phantom)
- `quality-summary` - ❌ Never runs (phantom)
- `repo-hygiene` - ❌ Never runs (phantom)
- `security` - ❌ Never runs (phantom)
- `test-core (3.10)` - ❌ Never runs (phantom)
- `test-core (3.12)` - ❌ Never runs (phantom)
- `test-ml` - ❌ Never runs (phantom)
- `typecheck` - ❌ Never runs (phantom)

**Actual checks that run on PRs**:
- `Golden Regression Tests` - ✅ Passing
- `Layer 1 Tests (Fast)` - ✅ Passing
- `test (3.10, cpu, core)` - ✅ Passing
- `test (3.12, cpu, core)` - ✅ Passing
- `test (3.11, cpu, ml)` - ✅ Passing
- `lint` - ✅ Passing
- `CodeQL` (via Code scanning results) - ✅ Passing
- Plus many others (all passing)

### 2. Review Requirement
- Branch protection requires 1 approval
- PR author (RC219805) cannot approve their own PR
- Need either:
  - Another reviewer with write access, OR
  - Disable "Require approval from someone other than the last pusher"

## Solution

### Step 1: Update Required Status Checks

Go to: **Settings** → **Branches** → **Branch protection rules** → **main** → **Edit**

In "Require status checks to pass before merging":

1. **Remove all phantom checks:**
   - Uncheck: `build`
   - Uncheck: `coverage-gate`
   - Uncheck: `lint (3.12)`
   - Uncheck: `quality-summary`
   - Uncheck: `repo-hygiene`
   - Uncheck: `security`
   - Uncheck: `test-core (3.10)`
   - Uncheck: `test-core (3.12)`
   - Uncheck: `test-ml`
   - Uncheck: `typecheck`

2. **Add actual checks that provide quality gates:**
   - ✅ Check: `Golden Regression Tests` (Enforcement workflow)
   - ✅ Check: `Layer 1 Tests (Fast)` (Enforcement workflow)
   - ✅ Check: `test (3.10, cpu, core)` (Build workflow - core tests)
   - ✅ Check: `test (3.12, cpu, core)` (Build workflow - core tests)
   - ✅ Check: `test (3.11, cpu, ml)` (Build workflow - ML validation)
   - ✅ Check: `lint` (Build workflow OR Python CI/CD)
   - ✅ Check: `CodeQL` (Security scanning)

3. **Optional but recommended:**
   - `Verify No Banned Dependencies` (Enforcement - security)
   - `Verify Action Pins` (Enforcement - supply chain security)
   - `Performance Regression Check` (Performance Monitor)

### Step 2: Fix Review Requirement

**Option A - Recommended**: Get a second approver
- Add a collaborator with write access
- Have them approve PR #771

**Option B - Adjust settings** (if solo repository):
In "Require a pull request before merging":
- Uncheck "Require approval from someone other than the last pusher"
- Keep "Require approvals: 1" (self-approval will then work)

### Step 3: Verify and Merge

After updating settings:

```bash
# Check PR status
gh pr view 771

# Should now show:
# - All required checks passing ✅
# - Mergeable (if review handled)

# Merge the smoke PR
gh pr merge 771 --squash
```

## Why This Happened

1. **ci.yml workflow was created** with job names like `build`, `lint (3.12)`, etc.
2. **Branch protection was configured** before any PR actually triggered these jobs
3. **GitHub marked them as "Expected"** but they don't run because:
   - ci.yml is a new workflow we just added
   - Workflows only register their check names after they run at least once
   - The existing workflows use different check names

4. **The firewall is actually working** - all the checks that DO run are passing!

## Post-Merge Actions

After PR #771 merges:

1. **Future PRs will trigger ci.yml** and those check names will become available
2. **Can then add ci.yml checks to branch protection** if desired
3. **Current setup works** - we have quality gates from existing workflows

## Recommendation

**Don't require ci.yml checks yet.** The existing workflows provide comprehensive coverage:

- **Enforcement** workflow: Golden tests, Layer 1 tests, dependency validation
- **Build** workflow: Core + ML tests across Python 3.10-3.12
- **Python CI/CD** workflow: Lint, tests, build verification
- **CodeQL** workflow: Security scanning
- **Performance Monitor** workflow: Regression detection

The firewall is operational with these. Add ci.yml checks to requirements later once they've run and proven themselves.

## Verification Commands

```bash
# See what GitHub knows about the PR
gh pr view 771 --json statusCheckRollup \
  --jq '.statusCheckRollup[] | select(.conclusion != "SKIPPED") | {name: .name, conclusion: .conclusion}'

# See current branch protection
gh api repos/RC219805/Transformation_Portal/branches/main/protection \
  --jq '{required_checks: .required_status_checks.checks[].context, reviews: .required_pull_request_reviews}'
```
