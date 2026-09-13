# Branch Protection Setup Guide

This guide separates observed branch-protection snapshots from configuration
recommendations. Re-read GitHub before relying on any recorded setting.

Read-only verification command:

```bash
gh api repos/RC219805/Transformation_Portal/branches/main/protection
```

Previous snapshot: 2026-06-03. Refreshed snapshot: **2026-09-12**. The refreshed
response requires `CI Gate` and `Dependency Security` (both bound to GitHub
Actions app ID 15368), strict up-to-date
checking, administrator enforcement, zero required approving reviews, stale
approval dismissal, and no Code Owner or last-push approval requirement.
Other settings below remain recommendations or the earlier dated snapshot
unless explicitly included in the refreshed response.

## Required Branch Protection Rules

### Navigate to Settings
1. Go to repository **Settings** → **Branches**
2. Click **Add rule** or edit existing rule for `main`

### Branch Name Pattern
```
main
```

### Protection Settings

#### ✅ Require Pull Request Reviews
- **Enable**: ✓ Require a pull request before merging
- **Required approving reviews**: 0 currently, 1+ recommended for governance-sensitive work
- **Dismiss stale pull request approvals**: ✓ enabled
- **Require review from Code Owners**: Disabled

#### ✅ Require Status Checks to Pass
- **Enable**: ✓ Require status checks to pass before merging
- **Require branches to be up to date**: ✓ enabled

**Required Status Checks**:
```
CI Gate
Dependency Security
```

**Note**: `CI Gate` is the stable branch-protection aggregator from
`.github/workflows/build.yml`. Do not add individual matrix jobs such as
`lint`, `test (3.11, cpu, core)`, or `generate-manifest` as separate required
checks unless the CI gate contract is intentionally changed.
`Dependency Security` is the independently required job in
`.github/workflows/security-unified.yml`. Its PR trigger must remain unfiltered
for `main`, and its dependency audit and source-policy checks must fail the job
on violations. A workflow comment alone does not configure merge protection.

#### ✅ Require Conversation Resolution
- **Enable**: ✓ enabled
- Ensures all review comments are addressed

#### ✅ Restrict Force Pushes
- **Enable**: ✓ force pushes are disabled
- Preserves commit history integrity

#### ⚠️ Require Linear History (Optional)
- **Current**: Disabled
- Exact-head squash merges remain the preferred project convention, but branch
  protection does not currently require linear history.

#### ✅ Enforce for Administrators
- **Current**: Enabled
- Admins are subject to the same branch protection rules.

#### ⚠️ Do Not Require Signed Commits (Optional)
- **Leave disabled** unless you have organizational policy
- Can be enabled for additional security

### Additional Settings

#### Allow Auto-Merge
- **Enable**: ✓ (optional, for developer convenience)
- PRs can be set to auto-merge once all checks pass

#### Allow Squash Merging
- **Enable**: ✓ (recommended)
- Keeps main branch history clean

#### Allow Rebase Merging
- **Enable**: ✓ (recommended)
- Alternative to squash for preserving individual commits

#### Allow Merge Commits
- **Disable**: ✗ (recommended)
- Prevents cluttered merge commit history

## Verification

Use read-only checks against the actual PR head before considering configuration
changes:

```bash
gh api repos/RC219805/Transformation_Portal/branches/main/protection
gh pr checks <PR-number> --required
gh pr view <PR-number> --json headRefOid,mergeStateStatus,reviewDecision
```

An expected check with no matching run needs trigger/name investigation; a
failing upstream job needs its own diagnosis. Do not change protection merely
to bypass a failure. Policy changes require explicit scope and approval.

## CODEOWNERS Setup (Optional)

Create `.github/CODEOWNERS` for automated review assignment:

```
# Global owners
* @repo-owner

# Architecture and contracts
/docs/architecture/ @architect-team
/src/transformation_portal/lux_depth_v3/config.py @architect-team

# CI/CD and infrastructure
/.github/workflows/ @devops-team @architect-team
/Dockerfile @devops-team
/docker-compose.yml @devops-team

# Security-critical files
/SECURITY.md @security-team @architect-team
/requirements/constraints.txt @architect-team
```

## Enforcement in CI

The canonical PR workflow (`.github/workflows/build.yml`) enforces quality
gates automatically. Its `CI Gate` job aggregates the blocking upstream jobs and
is required alongside the independent `Dependency Security` job.

### Critical Gates (BLOCKING)
- Lightweight checks
- Lint
- Test matrix
- Montecito manifest build
- CI contract validation
- Dependency security audit and checkpoint-loading source policy

### Advisory Gates (NON-BLOCKING but reported)
- Post-merge quality firewall signals
- Nightly and scheduled deep checks
- Advisory security/dependency reports not aggregated into `CI Gate`

## Troubleshooting

### Status checks not appearing
**Solution**: Run CI at least once. GitHub discovers available checks from workflow runs.

### Cannot find required check names
**Solution**: In branch protection settings, the check names are case-sensitive and must match exactly what appears in the workflow file's `name:` fields.

### Accidental force push to main
**Prevention**: Once branch protection is enabled, force pushes are blocked. Before enabling:
```bash
# Verify you're not on main
git branch --show-current

# If accidentally on main, create safety branch first
git checkout -b safety/backup-main
```

### Need to bypass protection (emergency)
**Admin override**: Admin enforcement is currently enabled, so emergency bypasses
require an explicit repository-settings change or an approved alternative
recovery path. Document the reason in an issue or post-incident review.

## Enforcement Checklist

Before considering this complete:

- [ ] Branch protection rule created for `main`
- [ ] `CI Gate` and `Dependency Security` are required, bound to the GitHub Actions app
- [ ] Pull request review policy intentionally set (currently 0 approvers; 1+ recommended)
- [ ] Conversation resolution required
- [ ] Admin enforcement enabled
- [ ] Force push disabled
- [ ] Linear history setting intentionally chosen (currently disabled; exact-head squash remains preferred)
- [ ] Verified by attempting to merge failing PR (blocks correctly)
- [ ] Verified by attempting force push (blocks correctly)
- [ ] CODEOWNERS file created (optional)
- [ ] Team documented the process

## Related Documentation

- [CONTRIBUTING.md](../../CONTRIBUTING.md) - Developer workflow
- [CI Workflow](../../.github/workflows/build.yml) - Quality gates implementation
- [Production Readiness](../deployment/PRODUCTION_READINESS.md) - Overall quality status

---

**Note**: Branch protection settings are stored in GitHub repository configuration, not in version control. Document changes to these settings in ADRs or team wikis.
