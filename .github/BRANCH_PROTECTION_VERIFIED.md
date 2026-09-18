# Branch Protection Verification

This file retains the PR #804 verification below as historical evidence.
Current workflow responsibilities are in the
[workflow matrix](../docs/ci/WORKFLOW_MATRIX.md).

## Current remote snapshot (2026-09-18 UTC)

A read-only GitHub API check against `main` returned:

- Required contexts: `CI Gate` and `Dependency Security`, each bound to
  GitHub Actions app `15368`.
- Strict up-to-date checking, conversation resolution, and admin enforcement:
  enabled.
- Code-owner review: disabled; required approving review count: zero.
- Active default-branch ruleset `9331244` ("Copilot review for default branch")
  also configures Copilot review on push, code-quality severity `all`, and
  CodeQL scanning with both alert thresholds set to `all`. These are separate
  from the two branch-protection status contexts.

```bash
gh api repos/RC219805/Transformation_Portal/branches/main/protection
gh api repos/RC219805/Transformation_Portal/rulesets/9331244
```

These settings can change independently of the repository. Re-read them before
merging; the old single-check record below does not describe current policy.
The post-CI firewall now runs through trusted `workflow_run` completion, not a
direct push trigger.

## Historical PR #804 record

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
