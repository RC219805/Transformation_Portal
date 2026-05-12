# Branch Protection Verification

> Current status note (2026-05-12): This file is a historical verification
> record for PR #804 plus non-binding policy notes. For live branch protection,
> verify GitHub repository metadata directly. A live GitHub API check on
> 2026-05-12 against `main` showed:
>
> - Required status check contexts: `CI Gate` only
> - Require branches to be up to date before merge: enabled
> - Require conversation resolution: enabled
> - Enforce protections for admins: enabled
> - Allow force pushes: disabled
> - Allow deletions: disabled
> - Require code-owner review: disabled
> - Require linear history: disabled
>
> Proof command:
>
> ```bash
> gh api repos/RC219805/Transformation_Portal/branches/main/protection
> ```

## Verification Status: ✅ COMPLETE

**Verified:** 2026-02-03T19:23:54Z
**PR:** #804
**Merge Commit:** 8e2b1402

**Note:** This document contains (a) a time-stamped verification record and (b) non-binding policy notes that may evolve.

### Proof Command
```bash
gh pr checks 804 --required --json name,state,workflow
```

### Result
```json
[
  {
    "name": "CI Gate",
    "state": "SUCCESS",
    "workflow": "CI (Lint, Tests & Manifest)"
  }
]
```

## Validated Behavior
- ✅ Only "CI Gate" is required (single stable check)
- ✅ No matrix-expanded checks (e.g., `test (3.11, cpu, core)`, `test (3.12, cpu, core)`)
- ✅ PR must be up to date with main before merge
- ✅ No phantom "Expected" checks

## Governance Trade-offs Documented

### Pre-merge Enforcement (Required)
- **CI Gate** is the single required check
- Coverage defined in `.github/workflows/build.yml` (ci_gate job dependencies)
- Current aggregation (as of 2026-02-03): `needs: [lint, test, generate-manifest]`
  - `lint`: runs on Python 3.12
  - `test`: matrix across supported Python versions (defined in build.yml) with cpu/core/ml test tiers
  - `generate-manifest`: artifact provenance validation
- **Note:** Aggregation subject to evolution; verify `.github/workflows/build.yml` for current state

### Post-merge Signal (Moved from PRs)
- **CI Quality Firewall** runs on push to main/develop only
- Rationale: Eliminated duplicate enforcement and noise on PRs
- Risk: Some failures may only be caught post-merge

### Signals Not Required by Branch Protection (as of verification)
- Security scans (CodeQL, dependency audit)
- Type checking
- Performance regression
- Repository hygiene

**Note:** Other repository policies (workflow approvals, code scanning alerts, CODEOWNERS) may still gate merges independently of branch protection.

**Decision Point:** Review which non-required checks should become pre-merge gates vs. post-merge/nightly validation.
