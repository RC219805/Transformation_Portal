# Branch Protection Verification

**Version:** 1.0.0
**Last Updated:** 2025-01-28
**Authority:** Transformation Portal Architect
**Review Frequency:** Quarterly or after policy changes

---

## Overview

Branch protection rules are critical repository governance controls that enforce:
- Code review requirements
- Quality gate enforcement
- Protection against accidental deletions or force pushes
- Linear history maintenance

**These settings are GitHub repository metadata and must be verified manually via the GitHub UI or API.**

This document provides:
1. Expected branch protection configuration for the `main` branch
2. Verification procedures
3. Troubleshooting guidance

---

## Expected Branch Protection Rules for `main`

### Core Requirements

The following protections **MUST** be enabled on the `main` branch:

#### ✅ Pull Request Requirements

- **Require pull request before merging:** ✅ ENABLED
  - Prevents direct commits to `main`
  - All changes must go through PR review process

- **Require approvals:** ✅ ENABLED
  - **Minimum approvals required:** 1
  - For Architect-level changes (security, dependency governance, ADRs): 2 approvals recommended
  - **Dismiss stale pull request approvals when new commits are pushed:** ✅ ENABLED
    - Ensures reviewers see latest code before merge

- **Require review from Code Owners:** ⚠️ OPTIONAL (Recommended if CODEOWNERS file exists)
  - If `CODEOWNERS` file is present, this should be enabled
  - Ensures domain experts review relevant changes

#### ✅ Status Check Requirements

- **Require status checks to pass before merging:** ✅ ENABLED

- **Required status checks:**
  1. **CI Gate** (`ci-gate` or primary CI workflow)
     - Validates: Build, core tests, linting, security checks
     - Status: REQUIRED

  2. **Lint (Critical)** (`lint-critical` or equivalent)
     - Validates: Flake8 critical errors, Black formatting, import validation
     - Status: REQUIRED

  3. **Tests (Core)** (`test-core` or equivalent)
     - Validates: Core test suite passes (non-ML, non-slow tests minimum)
     - Status: REQUIRED

  4. **Pre-commit Hooks** (if run in CI)
     - Validates: Trailing whitespace, EOF, YAML syntax, etc.
     - Status: REQUIRED

- **Require branches to be up to date before merging:** ⚠️ OPTIONAL
  - ✅ Recommended: Prevents merge conflicts and ensures tests run on final state
  - ⚠️ Trade-off: Increases friction for high-velocity repos (may require frequent rebases)
  - **Current Recommendation:** ENABLED for production repositories

#### ✅ Merge Strategy

- **Require conversation resolution before merging:** ✅ ENABLED
  - Ensures all review comments are addressed before merge

- **Require linear history:** ✅ ENABLED
  - Enforces clean, linear git history
  - Prevents merge commits in `main`
  - **Allowed merge methods:**
    - ✅ Squash and merge (RECOMMENDED for most PRs)
    - ✅ Rebase and merge (ALLOWED for clean feature branches)
    - ❌ Merge commit (DISABLED)

#### ✅ Protection Against Destructive Actions

- **Do not allow bypassing the above settings:** ✅ ENABLED
  - Prevents administrators from bypassing protections
  - **Exception:** May need to allow bypass for emergency hotfixes (with audit trail)

- **Do not allow force pushes:** ✅ ENABLED
  - Protects against history rewriting on `main`
  - **Exception:** Never allow - use `git revert` instead

- **Do not allow deletions:** ✅ ENABLED
  - Prevents accidental branch deletion

---

## Verification Procedures

### Option 1: GitHub UI Verification (Recommended)

**Prerequisites:**
- Repository admin or owner access
- GitHub account with appropriate permissions

**Steps:**

1. **Navigate to Branch Protection Settings:**
   ```
   GitHub Repository → Settings → Branches → Branch protection rules
   ```

2. **Select `main` branch rule:**
   - Click on the existing rule for `main`
   - Or click "Add rule" and enter `main` as branch name pattern

3. **Verify each setting matches expected configuration above:**
   - Use the checklist below as you verify each section

4. **Document any deviations:**
   - If settings differ from expected, document reason in `docs/governance/BRANCH_PROTECTION_DEVIATIONS.md`
   - Escalate to Architect for approval of deviations

**Verification Checklist:**

```
## Branch Protection Verification - [Date]

Verifier: [Your Name]
Date: [YYYY-MM-DD]

### Pull Request Requirements
- [ ] Require pull request before merging: ENABLED
- [ ] Require approvals: 1 minimum
- [ ] Dismiss stale approvals: ENABLED
- [ ] Require Code Owners review: [ENABLED/DISABLED - note if disabled]

### Status Checks
- [ ] Require status checks: ENABLED
- [ ] Required checks include:
  - [ ] CI Gate
  - [ ] Lint (critical)
  - [ ] Tests (core)
  - [ ] Pre-commit hooks (if applicable)
- [ ] Require branches up to date: [ENABLED/DISABLED]

### Merge Strategy
- [ ] Require conversation resolution: ENABLED
- [ ] Require linear history: ENABLED
- [ ] Allowed merge methods:
  - [ ] Squash merge: ENABLED
  - [ ] Rebase merge: ENABLED
  - [ ] Merge commit: DISABLED

### Protection
- [ ] Do not allow bypass: ENABLED
- [ ] Do not allow force push: ENABLED
- [ ] Do not allow deletion: ENABLED

### Deviations
[List any settings that differ from expected configuration and justification]

### Sign-off
- [ ] All critical protections verified
- [ ] Any deviations documented and approved
- [ ] Next review scheduled: [Date + 90 days]

Verified by: ___________________
Date: ___________________
```

### Option 2: GitHub CLI Verification (Programmatic)

**Prerequisites:**
- GitHub CLI (`gh`) installed
- Authenticated with repository access

**Commands:**

```bash
# 1. View current branch protection rules for main
gh api repos/:owner/:repo/branches/main/protection | jq '.'

# 2. Check specific protection settings
gh api repos/:owner/:repo/branches/main/protection | jq '{
  required_pull_request_reviews: .required_pull_request_reviews,
  required_status_checks: .required_status_checks,
  enforce_admins: .enforce_admins,
  restrictions: .restrictions,
  allow_force_pushes: .allow_force_pushes,
  allow_deletions: .allow_deletions,
  required_linear_history: .required_linear_history,
  required_conversation_resolution: .required_conversation_resolution
}'

# 3. Check required status checks specifically
gh api repos/:owner/:repo/branches/main/protection/required_status_checks | jq '.contexts'

# Expected output should include:
# - "ci-gate" (or primary CI workflow name)
# - "lint-critical"
# - "test-core"

# 4. Verify enforcement is strict
gh api repos/:owner/:repo/branches/main/protection/enforce_admins | jq '.enabled'
# Expected: true
```

**Automated Verification Script:**

Create `scripts/governance/verify_branch_protection.sh`:

```bash
#!/bin/bash
# Verify branch protection rules for main branch

set -euo pipefail

REPO_OWNER="${REPO_OWNER:-$(gh repo view --json owner -q .owner.login)}"
REPO_NAME="${REPO_NAME:-$(gh repo view --json name -q .name)}"
BRANCH="main"

echo "🔍 Verifying branch protection for ${REPO_OWNER}/${REPO_NAME}:${BRANCH}"

# Check if protection exists
if ! gh api "repos/${REPO_OWNER}/${REPO_NAME}/branches/${BRANCH}/protection" &>/dev/null; then
    echo "❌ ERROR: No branch protection rules found for ${BRANCH}"
    exit 1
fi

# Get protection config
PROTECTION=$(gh api "repos/${REPO_OWNER}/${REPO_NAME}/branches/${BRANCH}/protection")

# Verify required settings
echo ""
echo "Verifying required settings..."

# Check PR requirements
if echo "$PROTECTION" | jq -e '.required_pull_request_reviews != null' &>/dev/null; then
    echo "✅ Pull request reviews required"
    REQUIRED_APPROVALS=$(echo "$PROTECTION" | jq -r '.required_pull_request_reviews.required_approving_review_count')
    if [ "$REQUIRED_APPROVALS" -ge 1 ]; then
        echo "✅ Minimum approvals: $REQUIRED_APPROVALS"
    else
        echo "⚠️  WARNING: Minimum approvals is $REQUIRED_APPROVALS (expected >= 1)"
    fi
else
    echo "❌ ERROR: Pull request reviews NOT required"
    exit 1
fi

# Check status checks
if echo "$PROTECTION" | jq -e '.required_status_checks != null' &>/dev/null; then
    echo "✅ Status checks required"
    REQUIRED_CHECKS=$(echo "$PROTECTION" | jq -r '.required_status_checks.contexts[]')
    echo "   Required checks:"
    echo "$REQUIRED_CHECKS" | while read -r check; do
        echo "   - $check"
    done
else
    echo "⚠️  WARNING: Status checks NOT required"
fi

# Check linear history
if echo "$PROTECTION" | jq -e '.required_linear_history.enabled == true' &>/dev/null; then
    echo "✅ Linear history required"
else
    echo "⚠️  WARNING: Linear history NOT required"
fi

# Check force push protection
if echo "$PROTECTION" | jq -e '.allow_force_pushes.enabled == false' &>/dev/null; then
    echo "✅ Force pushes blocked"
else
    echo "❌ ERROR: Force pushes ALLOWED (should be blocked)"
    exit 1
fi

# Check deletion protection
if echo "$PROTECTION" | jq -e '.allow_deletions.enabled == false' &>/dev/null; then
    echo "✅ Deletions blocked"
else
    echo "❌ ERROR: Deletions ALLOWED (should be blocked)"
    exit 1
fi

# Check admin enforcement
if echo "$PROTECTION" | jq -e '.enforce_admins.enabled == true' &>/dev/null; then
    echo "✅ Admin bypass blocked"
else
    echo "⚠️  WARNING: Admins can bypass protections"
fi

echo ""
echo "✅ Branch protection verification complete"
```

**Usage:**

```bash
chmod +x scripts/governance/verify_branch_protection.sh
./scripts/governance/verify_branch_protection.sh
```

### Option 3: GitHub API Verification (Advanced)

For integration into CI/CD or automated governance checks:

```python
#!/usr/bin/env python3
"""Verify branch protection rules via GitHub API."""

import os
import sys
import requests

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_OWNER = os.environ.get("REPO_OWNER", "your-org")
REPO_NAME = os.environ.get("REPO_NAME", "Transformation_Portal")
BRANCH = "main"

def verify_branch_protection():
    """Verify branch protection rules for main branch."""
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/branches/{BRANCH}/protection"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"❌ ERROR: Failed to fetch branch protection (status {response.status_code})")
        return False

    protection = response.json()

    # Verify required settings
    checks_passed = True

    # Check PR reviews
    if "required_pull_request_reviews" not in protection:
        print("❌ Pull request reviews NOT required")
        checks_passed = False
    else:
        print("✅ Pull request reviews required")
        min_approvals = protection["required_pull_request_reviews"].get("required_approving_review_count", 0)
        if min_approvals >= 1:
            print(f"✅ Minimum approvals: {min_approvals}")
        else:
            print(f"⚠️  WARNING: Minimum approvals is {min_approvals} (expected >= 1)")

    # Check status checks
    if "required_status_checks" in protection:
        print("✅ Status checks required")
        contexts = protection["required_status_checks"].get("contexts", [])
        print(f"   Required checks ({len(contexts)}): {', '.join(contexts)}")
    else:
        print("⚠️  WARNING: Status checks NOT required")

    # Check linear history
    if protection.get("required_linear_history", {}).get("enabled"):
        print("✅ Linear history required")
    else:
        print("⚠️  WARNING: Linear history NOT required")

    # Check force push protection
    if not protection.get("allow_force_pushes", {}).get("enabled", True):
        print("✅ Force pushes blocked")
    else:
        print("❌ Force pushes ALLOWED")
        checks_passed = False

    # Check deletion protection
    if not protection.get("allow_deletions", {}).get("enabled", True):
        print("✅ Deletions blocked")
    else:
        print("❌ Deletions ALLOWED")
        checks_passed = False

    return checks_passed

if __name__ == "__main__":
    if not GITHUB_TOKEN:
        print("ERROR: GITHUB_TOKEN environment variable required")
        sys.exit(1)

    if verify_branch_protection():
        print("\n✅ All critical branch protections verified")
        sys.exit(0)
    else:
        print("\n❌ Branch protection verification FAILED")
        sys.exit(1)
```

**Usage:**

```bash
export GITHUB_TOKEN="your-personal-access-token"
python scripts/governance/verify_branch_protection.py
```

---

## When to Review

**Required Reviews:**

1. **Quarterly:** Every 90 days
   - Scheduled review to verify no drift
   - Update verification checklist with current date

2. **After Policy Changes:**
   - When ADRs modify governance requirements
   - When new quality gates are added to CI
   - When merge strategy changes

3. **After Repository Configuration Changes:**
   - When admins modify branch protection settings
   - When CI workflows are renamed or restructured

4. **Annual Governance Audit:**
   - Comprehensive review as part of annual security/governance audit

**Review Tracking:**

Maintain review log in `docs/governance/BRANCH_PROTECTION_REVIEW_LOG.md`:

```markdown
# Branch Protection Review Log

| Date       | Reviewer       | Status | Deviations | Next Review |
|------------|----------------|--------|------------|-------------|
| 2025-01-28 | Architect      | PASS   | None       | 2025-04-28  |
| 2024-10-28 | Lead Developer | PASS   | None       | 2025-01-28  |
```

---

## Troubleshooting

### Common Issues

#### Issue 1: "Status check not found"

**Symptom:** PR cannot merge because required status check is missing.

**Causes:**
- CI workflow was renamed
- Branch protection references old workflow name
- Workflow is not running on PR branches

**Resolution:**

1. Check which status checks are required:
   ```bash
   gh api repos/:owner/:repo/branches/main/protection/required_status_checks | jq '.contexts'
   ```

2. Check which status checks are available:
   ```bash
   gh pr checks <PR_NUMBER>
   ```

3. Update branch protection to use current workflow names:
   - Settings → Branches → Edit rule
   - Update required status checks to match current CI workflow names

#### Issue 2: "Required review missing"

**Symptom:** Cannot merge PR despite having approvals.

**Causes:**
- New commits pushed after approval (stale approval dismissed)
- Code Owners review required but not provided
- Reviewer lacks write access

**Resolution:**

1. Check PR approval status:
   ```bash
   gh pr view <PR_NUMBER> --json reviews
   ```

2. If stale approval dismissed:
   - Request re-approval from original reviewer
   - This is expected behavior (ensures reviewer sees latest code)

3. If Code Owners review needed:
   - Check `CODEOWNERS` file for required reviewers
   - Request review from Code Owner

#### Issue 3: "Branch not up to date"

**Symptom:** PR shows "Branch is out of date with base branch."

**Cause:** "Require branches to be up to date" is enabled.

**Resolution:**

1. Update PR branch with latest main:
   ```bash
   git checkout feature-branch
   git fetch origin
   git rebase origin/main
   git push --force-with-lease
   ```

2. Or use GitHub UI:
   - Click "Update branch" button on PR page

**Note:** This is expected behavior when "up to date" requirement is enabled.

---

## Emergency Bypass Procedure

**Authority Required:** Architect approval + documented justification

**Use Case:** Critical hotfix needed but branch protection blocks merge.

**Procedure:**

1. **Create Emergency Issue:**
   ```bash
   gh issue create \
       --title "EMERGENCY: Branch Protection Bypass for Hotfix #XXXX" \
       --label "emergency,governance" \
       --body "Justification: [Critical security fix / data loss prevention / etc.]

       Impact: [Description of issue requiring bypass]
       Hotfix PR: #XXXX
       Architect Approval: [Name]
       Duration: Temporary (will be re-enabled after merge)
       "
   ```

2. **Temporarily Disable Protection:**
   - Settings → Branches → Edit rule for `main`
   - Uncheck "Enforce for administrators" temporarily
   - Document in issue

3. **Merge Hotfix PR:**
   - Merge using approved method (preferably squash merge)
   - Document bypass in commit message

4. **Re-enable Protection Immediately:**
   - Settings → Branches → Edit rule for `main`
   - Re-check "Enforce for administrators"
   - Verify all other settings unchanged

5. **Document Bypass:**
   - Add entry to `docs/governance/BRANCH_PROTECTION_BYPASS_LOG.md`
   - Include: Date, reason, approver, PR number, duration

6. **Post-Mortem:**
   - Complete post-mortem within 24 hours (see `docs/operations/ROLLBACK_PROCEDURES.md#template-3`)
   - Identify process improvements to avoid future bypasses

**Bypass Log Format:**

```markdown
# Branch Protection Bypass Log

| Date       | Reason                          | Approver   | PR     | Duration | Post-Mortem |
|------------|---------------------------------|------------|--------|----------|-------------|
| 2025-01-28 | Critical security fix (CVE-XXX) | Architect  | #1234  | 15 min   | Issue #1235 |
```

---

## Integration with CONTRIBUTING.md

**Required Reference:**

Add the following section to `CONTRIBUTING.md`:

```markdown
## Branch Protection and Merge Requirements

The `main` branch is protected to ensure code quality and stability. All changes must:

1. **Go through Pull Request review**
   - Minimum 1 approval required
   - 2 approvals required for architectural changes (ADRs, security, dependencies)

2. **Pass all required CI checks:**
   - CI Gate (build + core tests)
   - Lint (critical errors)
   - Core test suite

3. **Resolve all review conversations**
   - Address all reviewer comments before merge

4. **Maintain linear history**
   - Use "Squash and merge" or "Rebase and merge"
   - Merge commits are disabled

5. **Keep branch up to date**
   - Rebase on latest `main` before merge

For full branch protection details, see: [Branch Protection Verification](docs/governance/BRANCH_PROTECTION_VERIFICATION.md)
```

---

## References

- **GitHub Documentation:** [About protected branches](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)
- **Rollback Procedures:** `docs/operations/ROLLBACK_PROCEDURES.md`
- **Agent Governance:** `docs/architecture/agent_governance.md`
- **ADR Template:** `docs/architecture/ADR_TEMPLATE.md`

---

## Version History

| Version | Date       | Changes                                         | Author    |
|---------|------------|-------------------------------------------------|-----------|
| 1.0.0   | 2025-01-28 | Initial branch protection verification guide    | Architect |

---

**Next Review:** 2025-04-28 (90 days from creation)
