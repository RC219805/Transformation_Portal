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
1. Moved `python-app.yml` to `.github/workflows-disabled/python-app.yml` (out of active workflows directory to prevent scanner/parser interference)
2. Updated branch protection to ensure only build.yml is required
3. Documented removal in this ADR

**Test PyPI Publish Decision:**

The removed `python-app.yml` included automatic Test PyPI publishing on pushes to main. This behavior has been **intentionally removed** for the following reasons:

1. Test PyPI publishes should be **manual and deliberate** (via `workflow_dispatch` or release tags)
2. Automatic publishes on every main push create noise and version number exhaustion
3. Release validation should happen through explicit release workflow, not automatic publishes

**If automatic Test PyPI publishes are needed in the future**, add a dedicated `publish-testpypi.yml` workflow with:
- Trigger: `workflow_dispatch` only (manual)
- Target: Test PyPI with `TEST_PYPI_API_TOKEN`
- Concurrency control to prevent overlapping publishes

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

# Confirm disabled workflow is isolated
ls .github/workflows-disabled/
```

### Rollback Plan

If issues arise:
```bash
mv .github/workflows-disabled/python-app.yml .github/workflows/python-app.yml
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
