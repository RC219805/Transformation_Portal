# GitHub Personal Access Token (PAT) Setup Guide

This guide explains how to create and configure a Personal Access Token (PAT) for automated dependency updates and other GitHub Actions workflows that need to create pull requests.

## Problem

GitHub Actions has a security restriction: workflows using the default `GITHUB_TOKEN` cannot create or approve pull requests. This prevents automated dependency update PRs from being created.

**Error you'll see:**
```
GitHub Actions is not permitted to create or approve pull requests.
```

## Solution: Create a Fine-Grained PAT

### Step 1: Generate a New PAT

1. Go to **GitHub Settings** → **Developer settings** → **Personal access tokens** → **Fine-grained tokens**
   - Direct link: https://github.com/settings/personal-access-tokens/new

2. Click **"Generate new token"**

3. Configure the token:

   **Token name:** `Transformation_Portal_Automation`
   
   **Expiration:** Choose appropriate duration (recommended: 90 days or 1 year)
   
   **Resource owner:** RC219805
   
   **Repository access:** Select "Only select repositories" → Choose `Transformation_Portal`

4. **Permissions** (Repository permissions):
   - **Contents:** Read and write ✅
   - **Metadata:** Read-only (automatic)
   - **Pull requests:** Read and write ✅
   - **Workflows:** Read and write ✅ (optional, for updating workflow files)

5. Click **"Generate token"**

6. **IMPORTANT:** Copy the token immediately - you won't be able to see it again!
   - Format: `github_pat_11...` (starts with `github_pat_`)

### Step 2: Add Token as Repository Secret

1. Go to your repository: https://github.com/RC219805/Transformation_Portal

2. Navigate to **Settings** → **Secrets and variables** → **Actions**

3. Click **"New repository secret"**

4. Configure the secret:
   - **Name:** `PAT_TOKEN`
   - **Value:** Paste the token you copied from Step 1

5. Click **"Add secret"**

### Step 3: Update Workflow to Use PAT

The workflow is already configured to use `PAT_TOKEN`. The key section is:

```yaml
- name: Create Pull Request
  uses: peter-evans/create-pull-request@v7
  with:
    token: ${{ secrets.PAT_TOKEN }}  # Uses PAT instead of GITHUB_TOKEN
    commit-message: "chore: update dependencies (automated)"
    title: "🔄 Automated Dependency Updates"
    # ... other settings
```

### Step 4: Enable Workflow Permissions

1. Go to **Repository Settings** → **Actions** → **General**

2. Scroll to **"Workflow permissions"**

3. Select:
   - ✅ **"Read and write permissions"**
   - ✅ **"Allow GitHub Actions to create and approve pull requests"**

4. Click **"Save"**

**Note:** Even with these settings enabled, you still need the PAT because the default `GITHUB_TOKEN` has limitations for security reasons.

## Verification

To verify the setup works:

1. **Manual trigger:**
   ```bash
   # Go to Actions tab → Dependency Updates → Run workflow
   ```

2. **Or wait for scheduled run:**
   - The workflow runs every Monday at 3 AM UTC
   - Check `.github/workflows/dependency-updates.yml` for schedule

3. **Expected result:**
   - Workflow completes successfully
   - A new PR is created: "🔄 Automated Dependency Updates"
   - PR contains updated `requirements/*.txt` files
   - Security report is attached

## Security Best Practices

### Token Rotation

- **Set expiration:** Don't create tokens that never expire
- **Rotate regularly:** Update the token every 3-6 months
- **Document expiration:** Add a calendar reminder 1 week before expiration

### Monitoring

- Review automated PRs before merging
- Check security reports for vulnerabilities
- Audit token usage in Settings → Developer settings → Personal access tokens

### Revocation

If the token is compromised:

1. Go to https://github.com/settings/personal-access-tokens
2. Find `Transformation_Portal_Automation`
3. Click **"Revoke"**
4. Generate a new token following Step 1-2 above
5. Update the `PAT_TOKEN` secret

## Troubleshooting

### Error: "Resource not accessible by personal access token"

**Cause:** Token doesn't have required permissions

**Fix:**
1. Go to https://github.com/settings/personal-access-tokens
2. Click on your token
3. Edit permissions to include:
   - Contents: Read and write
   - Pull requests: Read and write

### Error: "Bad credentials"

**Cause:** Token is expired or was revoked

**Fix:**
1. Generate a new token (Step 1)
2. Update the `PAT_TOKEN` secret (Step 2)

### Error: "Workflow requires the pull_requests permission"

**Cause:** Workflow permissions not enabled

**Fix:**
1. Go to Settings → Actions → General
2. Enable "Allow GitHub Actions to create and approve pull requests"

### PR Created but Empty or Missing Files

**Cause:** Workflow didn't generate changes or commit failed

**Debug:**
1. Check workflow logs in Actions tab
2. Look for errors in "Update Requirements" step
3. Verify `scripts/update_all_requirements.sh` exists and is executable

## Workflow Overview

The automated dependency update workflow:

1. **Schedule:** Runs every Monday at 3 AM UTC
2. **Checks:**
   - Updates all `requirements/*.txt` files
   - Runs `safety check` for vulnerabilities
   - Generates a security report
3. **Creates PR:**
   - Title: "🔄 Automated Dependency Updates"
   - Labels: `dependencies`, `automated`
   - Includes security scan results
4. **Review checklist:**
   - Check for breaking changes
   - Review security report
   - Validate tests pass
   - Verify compatibility

## Additional Resources

- [GitHub PAT Documentation](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)
- [Fine-Grained PAT Permissions](https://docs.github.com/en/rest/overview/permissions-required-for-fine-grained-personal-access-tokens)
- [GitHub Actions Secrets](https://docs.github.com/en/actions/security-guides/encrypted-secrets)
- [peter-evans/create-pull-request Action](https://github.com/peter-evans/create-pull-request)

## Quick Reference

| Item | Value |
|------|-------|
| **Secret Name** | `PAT_TOKEN` |
| **Token Permissions** | Contents: RW, Pull Requests: RW |
| **Repository** | `RC219805/Transformation_Portal` |
| **Workflow File** | `.github/workflows/dependency-updates.yml` |
| **Schedule** | Every Monday 3 AM UTC |

---

**Last Updated:** 2025-12-15  
**Maintainer:** RC219805
