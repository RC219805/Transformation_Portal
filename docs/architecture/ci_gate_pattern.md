# ADR: CI Gate Pattern for Stable Branch Protection

## Status
**Accepted** - February 2026

## Context

### The Problem

Branch protection rules in GitHub allow requiring specific status checks before merging. However, when these checks are tied to matrix-expanded job names, branch protection breaks every time the matrix changes.

**Example failure scenario (PR #799):**
1. Repository tested Python 3.10, 3.11, 3.12
2. Branch protection required: `test (3.10, cpu, core)`, `test (3.11, cpu, core)`, `test (3.12, cpu, core)`
3. Python 3.10 support was dropped in commits `99eb8341`, `82f7f92a` (November 2024)
4. Workflow updated to test only 3.11 and 3.12
5. **PRs stuck:** Branch protection still waiting for `test (3.10, cpu, core)` — status never arrives
6. **Admin intervention required:** Manual update of branch protection settings

**Recurring pattern:**
- Every Python version change requires admin intervention
- Every test type addition/removal requires admin intervention
- Every device type change requires admin intervention
- Matrix-expanded job names are inherently unstable

### Industry Practice

Major open-source projects use a "CI Gate" or "aggregator" pattern:
- **Kubernetes:** `pull-kubernetes-verify` aggregates ~100 checks
- **Terraform:** `ci-success` aggregates provider tests
- **GitHub CLI:** `ci` job aggregates lint, test, build
- **Docker:** `all-green` aggregates platform matrix

**Why it works:**
- Single stable check name for branch protection
- Matrix can evolve freely without admin intervention
- Clear failure reporting (which upstream job failed)
- Follows GitHub's own recommendations for matrix workflows

## Decision

Implement a **CI Gate** aggregator job in `.github/workflows/build.yml`:

```yaml
ci_gate:
  name: CI Gate
  runs-on: ubuntu-24.04
  needs: [lint, test, generate-manifest]
  if: ${{ always() }}
  timeout-minutes: 5
```

### Key Design Elements

1. **Stable Name:** Job name is `CI Gate` (not matrix-expanded)
2. **Always Runs:** Uses `if: always()` to run even when upstream jobs fail
3. **Explicit Dependencies:** Lists all critical jobs in `needs:`
4. **Result Checking:** Explicitly checks each upstream job for `result == 'success'`
5. **Clear Reporting:** Outputs which upstream job failed

### Implementation Details

**Upstream result checking:**
```bash
if [ "${{ needs.lint.result }}" != "success" ]; then
  echo "❌ lint did not succeed"
  ok="false"
fi
```

**Bracket syntax for hyphenated job IDs:**
```yaml
echo "manifest result: ${{ needs['generate-manifest'].result }}"
```

## Consequences

### Positive

✅ **Branch protection stability:** Matrix changes no longer require admin intervention
✅ **Developer velocity:** No more PRs stuck on "Expected — Waiting for status"
✅ **Maintainability:** Add/remove Python versions without breaking CI
✅ **Clarity:** Single check to monitor; failure shows which upstream job broke
✅ **Alignment:** Follows GitHub best practices and industry patterns
✅ **Future-proof:** Works for any matrix dimension (Python, device, test type)

### Negative

❌ **One more job:** Adds ~5-10 seconds to workflow runtime (negligible)
❌ **Indirection:** Must click into CI Gate to see which upstream job failed (acceptable UX trade-off)

### Neutral

- **Enforcement location shifts:** Branch protection now enforces "CI Gate passed" instead of individual checks
- **Admin action still required once:** Initial setup to require `CI Gate` and remove matrix checks
- **Documentation needed:** Team must understand CI Gate pattern (addressed in this ADR)

## Alternatives Considered

### Alternative 1: Manual Admin Updates
**Approach:** Continue requiring matrix-expanded checks; update branch protection manually when matrix changes.

**Rejected because:**
- Does not scale (every matrix change is blocked on admin)
- Fragile (easy to forget, causes PR blockages)
- Not maintainable long-term (admin becomes bottleneck)

### Alternative 2: Remove Branch Protection
**Approach:** Don't require status checks; rely on team discipline.

**Rejected because:**
- Defeats purpose of CI (no enforcement)
- High risk of merging broken code
- Not acceptable for production repository

### Alternative 3: Separate Workflows Per Python Version
**Approach:** Create `build-py311.yml`, `build-py312.yml` with stable names.

**Rejected because:**
- Massive duplication (each workflow ~300+ lines)
- Hard to maintain consistency across workflows
- Still breaks when adding new Python version (new workflow file needed)

### Alternative 4: GitHub Actions composite action
**Approach:** Create a composite action that runs all checks and reports single status.

**Rejected because:**
- Over-engineered for this use case
- Composite actions have limitations (can't use `needs:`)
- CI Gate job is simpler and more transparent

## Migration Plan

### Phase 1: Add CI Gate Job ✅
- Add `ci_gate` job to `.github/workflows/build.yml`
- Verify job runs and aggregates correctly
- Test on draft PR

### Phase 2: Update Documentation ✅
- Update `ADMIN_BRANCH_PROTECTION_UPDATE.md` with new instructions
- Update `docs/operations/branch_protection_setup.md`
- Update `docs/ci_cd/CI_CD_WORKFLOWS.md`
- Create this ADR

### Phase 3: Admin Action Required
- Remove matrix-expanded checks from branch protection:
  - ❌ `test (3.11, cpu, core)`
  - ❌ `test (3.12, cpu, core)`
  - ❌ `test (3.11, cpu, ml)`
  - ❌ `lint`
  - ❌ `generate-manifest`
- Add single check:
  - ✅ `CI Gate`

### Phase 4: Verification
- Confirm PRs show `CI Gate` as required check
- Verify CI Gate correctly reports pass/fail
- Test matrix change (add Python version) without admin intervention

## Testing Strategy

### Scenarios to Verify

1. **All upstream jobs pass:**
   - Expected: CI Gate passes (green ✅)
   - Expected: PR mergeable

2. **One upstream job fails:**
   - Expected: CI Gate fails (red ❌)
   - Expected: CI Gate output shows which job failed
   - Expected: PR blocked

3. **Matrix change (add Python version):**
   - Expected: New matrix job runs
   - Expected: CI Gate aggregates new job automatically
   - Expected: Branch protection still works (no admin action)

4. **Skipped job:**
   - Expected: CI Gate treats skipped as failure
   - Expected: PR blocked (unless skip is intentional)

## Future Considerations

### Making Jobs Optional

If a job should be allowed to fail (e.g., experimental tests):

**Option 1:** Remove from CI Gate aggregation
```yaml
needs: [lint, test]  # removed generate-manifest
```

**Option 2:** Allow specific job to fail
```yaml
if [ "${{ needs['generate-manifest'].result }}" != "success" ] && [ "${{ needs['generate-manifest'].result }}" != "skipped" ]; then
  echo "⚠️ generate-manifest failed (non-blocking)"
  # Don't set ok="false"
fi
```

### Adding New Critical Jobs

When adding a new job that CI Gate should aggregate:

1. Add job to workflow
2. Add to `ci_gate.needs` array
3. Add result check in "Enforce pass/fail" step
4. Update documentation
5. **No branch protection changes needed** (CI Gate already required)

### Matrix Evolution

The CI Gate pattern explicitly supports:
- ✅ Adding/removing Python versions
- ✅ Adding/removing devices (cpu, cuda, etc.)
- ✅ Adding/removing test types (core, ml, integration, etc.)
- ✅ Changing matrix combinations

All without admin intervention or branch protection updates.

## Related Documents

- **Implementation:** `.github/workflows/build.yml` (lines 341-382)
- **Admin Guide:** `ADMIN_BRANCH_PROTECTION_UPDATE.md`
- **Operations:** `docs/operations/branch_protection_setup.md`
- **CI/CD Overview:** `docs/ci_cd/CI_CD_WORKFLOWS.md`

## References

- GitHub Actions: [Using jobs in a workflow](https://docs.github.com/en/actions/using-jobs/using-jobs-in-a-workflow)
- GitHub: [Required status checks](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches#require-status-checks-before-merging)
- Kubernetes: [CI Signal](https://github.com/kubernetes/test-infra/blob/master/config/jobs/kubernetes/sig-release/cip/container-image-promoter.yaml)
- Terraform: [CI Success Pattern](https://github.com/hashicorp/terraform-provider-aws/blob/main/.github/workflows/ci.yml)

## Decision Authority

**Architect:** Transformation Portal Architect
**Date:** February 2026
**Status:** Binding architectural decision

This pattern is now the normative approach for branch protection in this repository. Future matrix changes should not require admin intervention. Deviations require explicit ADR.
