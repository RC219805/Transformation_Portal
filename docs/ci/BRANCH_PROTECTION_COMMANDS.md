# Branch Protection Configuration Commands

> **Historical record. Do not run these commands as current configuration.**
> The required branch-protection check is now the stable `CI Gate` aggregator,
> and live settings must be verified from GitHub before any change. Use
> [BRANCH_PROTECTION_SETUP.md](./BRANCH_PROTECTION_SETUP.md) for maintained
> setup and troubleshooting guidance.

**Generated**: 2026-02-01
**Purpose**: Complete quality firewall implementation with automated branch protection

---

## ⚡ Quick Setup (GitHub CLI)

### Prerequisites
```bash
# Verify gh CLI is installed and authenticated
gh auth status

# Verify you're in the correct repository
gh repo view
```

### Apply Branch Protection (Single Command)

```bash
gh api repos/:owner/:repo/branches/main/protection \
  --method PUT \
  --field required_status_checks[strict]=true \
  --field required_status_checks[contexts][]=lint (3.12) \
  --field required_status_checks[contexts][]=typecheck \
  --field required_status_checks[contexts][]=security \
  --field required_status_checks[contexts][]=test-core (3.10) \
  --field required_status_checks[contexts][]=test-core (3.12) \
  --field required_status_checks[contexts][]=test-ml \
  --field required_status_checks[contexts][]=coverage-gate \
  --field required_status_checks[contexts][]=build \
  --field required_status_checks[contexts][]=repo-hygiene \
  --field required_status_checks[contexts][]=quality-summary \
  --field required_pull_request_reviews[required_approving_review_count]=1 \
  --field required_pull_request_reviews[dismiss_stale_reviews]=true \
  --field required_pull_request_reviews[require_code_owner_reviews]=false \
  --field required_conversation_resolution[enabled]=true \
  --field enforce_admins[enabled]=false \
  --field restrictions=null \
  --field allow_force_pushes[enabled]=false \
  --field allow_deletions[enabled]=false \
  --field block_creations[enabled]=false \
  --field required_linear_history[enabled]=true \
  --field allow_fork_syncing[enabled]=true
```

### Alternative: JSON Payload Method

```bash
gh api repos/:owner/:repo/branches/main/protection \
  --method PUT \
  --input - <<'EOF'
{
  "required_status_checks": {
    "strict": true,
    "contexts": [
      "lint (3.12)",
      "typecheck",
      "security",
      "test-core (3.10)",
      "test-core (3.12)",
      "test-ml",
      "coverage-gate",
      "build",
      "repo-hygiene",
      "quality-summary"
    ]
  },
  "required_pull_request_reviews": {
    "required_approving_review_count": 1,
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": false
  },
  "required_conversation_resolution": {
    "enabled": true
  },
  "enforce_admins": false,
  "restrictions": null,
  "allow_force_pushes": {
    "enabled": false
  },
  "allow_deletions": {
    "enabled": false
  },
  "block_creations": {
    "enabled": false
  },
  "required_linear_history": {
    "enabled": true
  },
  "allow_fork_syncing": {
    "enabled": true
  }
}
EOF
```

---

## 🌐 Manual Setup (GitHub UI)

If you prefer to configure via the GitHub web interface:

### Step 1: Navigate to Settings
1. Go to repository on GitHub
2. Click **Settings** tab
3. Click **Branches** in left sidebar
4. Click **Add rule** (or edit existing `main` rule)

### Step 2: Configure Pattern
```
Branch name pattern: main
```

### Step 3: Enable Protections

#### ✅ Require Pull Request Reviews Before Merging
- ☑ Require a pull request before merging
- Required number of approvals: **1**
- ☑ Dismiss stale pull request approvals when new commits are pushed
- ☐ Require review from Code Owners (optional)

#### ✅ Require Status Checks to Pass Before Merging
- ☑ Require status checks to pass before merging
- ☑ Require branches to be up to date before merging

**Add required status checks:**
```
lint (3.12)
typecheck
security
test-core (3.10)
test-core (3.12)
test-ml
coverage-gate
build
repo-hygiene
quality-summary
```

**Important**: GitHub auto-discovers check names after first CI run. Type these manually if not appearing in dropdown.

#### ✅ Require Conversation Resolution Before Merging
- ☑ Require conversation resolution before merging

#### ✅ Require Linear History
- ☑ Require linear history

#### ✅ Do Not Allow Force Pushes
- ☑ Do not allow force pushes

#### ✅ Do Not Allow Deletions
- ☑ Do not allow deletions

### Step 4: Save Changes
- Click **Create** (or **Save changes**)

---

## ✅ Verification

### 1. Verify Protection is Active
```bash
# Check branch protection status
gh api repos/:owner/:repo/branches/main/protection | jq '.required_status_checks.contexts'

# Expected output:
# [
#   "lint (3.12)",
#   "typecheck",
#   "security",
#   "test-core (3.10)",
#   "test-core (3.12)",
#   "test-ml",
#   "coverage-gate",
#   "build",
#   "repo-hygiene",
#   "quality-summary"
# ]
```

### 2. Test Protection with Failing PR
```bash
# Create test branch with intentional failure
git checkout -b test/verify-protection
echo "import os,sys" >> test_lint_fail.py
git add test_lint_fail.py
git commit -m "test: verify branch protection blocks merge"
git push origin test/verify-protection

# Open PR
gh pr create --title "Test: Branch Protection" --body "Verify CI gates block merge"

# Verify:
# - CI checks appear as required
# - Cannot merge until checks pass
# - Merge button is disabled/blocked
```

### 3. Verify Force Push is Blocked
```bash
# Try to force push to main (should fail)
git checkout main
git commit --allow-empty -m "test: force push protection"
git push --force origin main

# Expected: Error! Push declined due to branch protection
```

### 4. Clean Up Test
```bash
git checkout main
git reset --hard HEAD~1
git branch -D test/verify-protection
gh pr close 1 --delete-branch  # Replace 1 with actual PR number
rm -f test_lint_fail.py
```

---

## 📊 Quality Gate Status Checks

### Critical Gates (BLOCKING)
These **must pass** for merge approval:

| Check Name | Purpose | Python Version |
|------------|---------|----------------|
| `lint (3.12)` | Code style (flake8, ruff) | 3.12 |
| `typecheck` | Type safety (mypy) | 3.12 |
| `security` | Vulnerability scan (bandit, safety) | 3.12 |
| `test-core (3.10)` | Core functionality tests (min Python) | 3.10 |
| `test-core (3.12)` | Core functionality tests (max Python) | 3.12 |
| `test-ml` | ML pipeline tests | 3.11 |
| `coverage-gate` | Code coverage thresholds | 3.12 |
| `build` | Package build verification | 3.12 |
| `repo-hygiene` | Repository cleanliness | 3.12 |
| `quality-summary` | Aggregate status report | 3.12 |

### Advisory Checks (NON-BLOCKING)
These run but **do not block** merge:
- Performance benchmarks (if configured)
- Documentation builds (if separate job)

---

## 🔐 Security Considerations

### Protection Enforcement
- **Admins NOT exempt**: `enforce_admins: false` allows emergency bypasses (logged)
- **Force pushes disabled**: Prevents history rewriting
- **Deletions disabled**: Prevents accidental branch deletion
- **Linear history required**: Enforces clean rebase/squash workflow

### Emergency Override Process
If admin bypass is needed (rare):
1. Document reason in GitHub issue
2. Notify team in Slack/communication channel
3. Perform change
4. Re-enable protection immediately
5. Post-incident review within 24 hours

---

## 🔧 Troubleshooting

### Problem: Status check names not appearing in UI
**Solution**:
- Run CI pipeline at least once
- GitHub discovers check names from actual workflow runs
- Check names are case-sensitive and must match workflow exactly

### Problem: `gh api` command fails with 403
**Solution**:
```bash
# Re-authenticate with required permissions
gh auth refresh -s admin:org,repo

# Verify authentication
gh auth status
```

### Problem: Branch protection updates but checks don't appear
**Solution**:
- Check workflow file syntax: `.github/workflows/ci.yml`
- Verify job names match exactly (including Python version in parentheses)
- Trigger CI manually: `gh workflow run ci.yml`

### Problem: Cannot merge even with passing checks
**Solution**:
- Verify all 10 required checks show ✅ green
- Check conversation resolution (all review comments resolved)
- Verify branch is up-to-date with main
- Check PR has required approval

---

## 📚 Related Documentation

- [BRANCH_PROTECTION_SETUP.md](./BRANCH_PROTECTION_SETUP.md) - Original setup guide
- [CI Workflow](../../.github/workflows/build.yml) - Quality gates implementation
- [CONTRIBUTING.md](../../CONTRIBUTING.md) - Developer workflow
- [CODE_QUALITY_SYSTEM.md](../guides/CODE_QUALITY_SYSTEM.md) - Quality standards

---

## 🎯 Success Criteria

Branch protection is correctly configured when:

- ✅ All 10 status checks are required
- ✅ Pull request review required (1+ approver)
- ✅ Conversation resolution required
- ✅ Force pushes disabled
- ✅ Linear history enforced
- ✅ Test PR with failures correctly blocks merge
- ✅ Test force push to main is rejected
- ✅ Documentation updated

---

**Note**: This configuration enforces the quality firewall at the repository level. Combined with CI gates, this ensures production stability and code quality standards.
