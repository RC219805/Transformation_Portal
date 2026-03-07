# Branch Protection Setup Guide

This guide documents the required branch protection rules for the `main` branch to enforce CI quality gates.

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
- **Required approving reviews**: 1
- **Dismiss stale pull request approvals**: ✓ (recommended)
- **Require review from Code Owners**: Optional (if using CODEOWNERS)

#### ✅ Require Status Checks to Pass
- **Enable**: ✓ Require status checks to pass before merging
- **Require branches to be up to date**: ✓ (recommended)

**Required Status Checks** (add these):
```
lint (3.12)
security
test-core (3.10)
test-core (3.12)
test-ml (3.11)
coverage-gate
build
repo-hygiene
quality-summary
```

**Note**: GitHub will auto-discover these check names after the first CI run. You may need to type them manually if they haven't run yet.

#### ✅ Require Conversation Resolution
- **Enable**: ✓ Require conversation resolution before merging
- Ensures all review comments are addressed

#### ✅ Restrict Force Pushes
- **Enable**: ✓ Do not allow force pushes
- Preserves commit history integrity

#### ✅ Require Linear History (Optional but Recommended)
- **Enable**: ✓ Require linear history
- Enforces clean, linear commit history via rebase or squash merges

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

After setting up branch protection:

1. Create a test PR with intentional failures:
   ```bash
   git checkout -b test/branch-protection
   # Make a change that fails lint
   echo "import os,sys" >> src/test_file.py
   git add . && git commit -m "test: verify branch protection"
   git push origin test/branch-protection
   ```

2. Open PR and verify:
   - CI checks are required
   - Cannot merge until checks pass
   - Cannot force push to main

3. Clean up:
   ```bash
   git checkout main
   git branch -D test/branch-protection
   git push origin --delete test/branch-protection
   ```

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

The CI workflow (`.github/workflows/ci.yml`) enforces quality gates automatically. The `quality-summary` job aggregates all results and blocks merge if critical gates fail.

### Critical Gates (BLOCKING)
- Lint
- Security scans
- Core tests
- Build verification
- Repo hygiene

### Advisory Gates (NON-BLOCKING but reported)
- Type checking
- Coverage warnings (diff coverage is enforced)
- ML tests (informational)

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
**Admin override**: Repository admins can bypass protection rules if absolutely necessary. This is logged and should be rare. Document the reason in an issue or post-incident review.

## Enforcement Checklist

Before considering this complete:

- [ ] Branch protection rule created for `main`
- [ ] All required status checks added
- [ ] Pull request reviews required (1+ approver)
- [ ] Force push disabled
- [ ] Linear history enabled (optional but recommended)
- [ ] Verified by attempting to merge failing PR (blocks correctly)
- [ ] Verified by attempting force push (blocks correctly)
- [ ] CODEOWNERS file created (optional)
- [ ] Team documented the process

## Related Documentation

- [CONTRIBUTING.md](../CONTRIBUTING.md) - Developer workflow
- [CI Workflow](../.github/workflows/ci.yml) - Quality gates implementation
- [Production Readiness](./PRODUCTION_READINESS.md) - Overall quality status

---

**Note**: Branch protection settings are stored in GitHub repository configuration, not in version control. Document changes to these settings in ADRs or team wikis.
