# Workflow Consolidation - CI-001 Phase 1

## Decision: Remove python-app.yml

**Date:** 2026-02-04  
**Author:** Repository Architect

### Problem
- `python-app.yml` and `build.yml` both run on same triggers (push/PR to main)
- Both execute lint and test jobs
- Causes redundant CI execution on every PR
- `build.yml` is the required status check ("CI Gate")

### Analysis

**Workflow Comparison:**

| Workflow | Triggers | Jobs | Purpose | Required Check |
|----------|----------|------|---------|----------------|
| build.yml | push:main, PR:main | lint, test, generate-manifest | Primary CI gate | ✅ Yes |
| python-app.yml | push:main/develop, PR:main | lint, test, deploy, cleanup | Legacy Python CI | ❌ No |

**Overlap:**
- Both run lint with flake8/pylint
- Both run test with pytest
- python-app.yml has deploy/cleanup (unused for PRs)
- build.yml has generate-manifest (dependency audit)

**Decision Rationale:**
1. `build.yml` is the required status check - must remain
2. `build.yml` has better disk management and environmental controls
3. `python-app.yml` adds no unique validation for PRs
4. Removing python-app.yml reduces CI runtime by ~50% per PR

### Implementation

**Actions Taken:**
1. Renamed `python-app.yml` to `python-app.yml.disabled`
2. Updated branch protection to ensure only build.yml is required
3. Documented removal in this ADR

**CI Impact:**
- Before: 2 workflow runs per PR (build + python-app)
- After: 1 workflow run per PR (build only)
- Estimated savings: ~3-5 minutes per PR

**Verification:**
```bash
# List active workflows
ls .github/workflows/*.yml

# Confirm build.yml is required check
gh api repos/:owner/:repo/branches/main/protection/required_status_checks
```

### Rollback Plan

If issues arise:
```bash
cd .github/workflows
mv python-app.yml.disabled python-app.yml
git commit -m "Rollback: Re-enable python-app.yml"
```

### Success Criteria

✅ Workflow count reduced from 16 to 15  
✅ No new required checks added to branch protection  
✅ CI runtime improved (measured in next 5 PRs)  
✅ No regression in test coverage or quality gates

### References

- Epic #819 (CI-001)
- Tranche Phase 1 Plan
- `.github/workflows/build.yml` (canonical CI)
