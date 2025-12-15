# GitHub Personal Access Token (PAT) Setup Guide

**Purpose**: Enable automated dependency update PRs via GitHub Actions

**Issue**: GitHub Actions default `GITHUB_TOKEN` cannot create or approve PRs due to security restrictions

**Solution**: Create a fine-grained Personal Access Token (PAT) with minimal permissions

---

## Prerequisites

- GitHub account with admin access to `RC219805/Transformation_Portal`
- Access to repository Settings → Secrets and variables → Actions

---

## Step 1: Create Fine-Grained PAT

### 1.1 Navigate to Token Creation

1. Go to [GitHub Settings → Developer settings → Personal access tokens → Fine-grained tokens](https://github.com/settings/tokens?type=beta)
2. Click **"Generate new token"**

### 1.2 Configure Token Settings

**Token name**: `Transformation_Portal_Dependency_Bot`  
**Expiration**: `90 days` (recommended; set calendar reminder to regenerate)  
**Description**: `Automated dependency update PRs for Transformation_Portal`  

**Repository access**: `Only select repositories`  
→ Select: `RC219805/Transformation_Portal`

### 1.3 Set Repository Permissions

**Permissions** → Expand "Repository permissions":

| Permission | Access Level | Reason |
|------------|--------------|--------|
| **Contents** | Read and write | Clone repo, commit changes to automated branch |
| **Pull requests** | Read and write | Create PRs, add labels, assign reviewers |
| **Metadata** | Read-only | (Automatically granted) |

**All other permissions**: Leave as "No access"

### 1.4 Generate Token

1. Click **"Generate token"**
2. **⚠️ CRITICAL**: Copy the token immediately (it will only be shown once)
3. Store temporarily in a secure location (password manager, encrypted note)

**Token format**: `github_pat_***********************************************`

---

## Step 2: Add PAT to Repository Secrets

### 2.1 Navigate to Secrets

1. Go to [Transformation_Portal → Settings → Secrets and variables → Actions](https://github.com/RC219805/Transformation_Portal/settings/secrets/actions)
2. Click **"New repository secret"**

### 2.2 Create Secret

**Name**: `PAT_TOKEN`  
**Secret**: Paste the token copied in Step 1.4

Click **"Add secret"**

### 2.3 Verify Secret

- The secret should now appear in the list as `PAT_TOKEN`
- Value will be hidden (shows `***`)

---

## Step 3: Update Workflow to Use PAT

The workflow `.github/workflows/dependency-updates.yml` is already configured to use the PAT:

```yaml
- name: Create Pull Request
  uses: peter-evans/create-pull-request@v7
  with:
    token: ${{ secrets.PAT_TOKEN }}  # ← Uses the PAT instead of GITHUB_TOKEN
    commit-message: "chore: update dependencies (automated)"
    title: "🔄 Automated Dependency Updates"
    branch: automated/dependency-updates
    delete-branch: true
    labels: |
      dependencies
      automated
```

**No changes needed** — the workflow is already PAT-ready.

---

## Step 4: Verify Setup

### 4.1 Trigger Workflow Manually

1. Go to [Actions → Dependency Updates](https://github.com/RC219805/Transformation_Portal/actions/workflows/dependency-updates.yml)
2. Click **"Run workflow"** → **"Run workflow"**

### 4.2 Expected Behavior

- Workflow should complete successfully
- A new PR should be created: "🔄 Automated Dependency Updates"
- PR should have labels: `dependencies`, `automated`
- PR author should be `github-actions[bot]`

### 4.3 Troubleshooting

**Error: "GitHub Actions is not permitted to create or approve pull requests"**

→ PAT secret not found or misconfigured. Verify:
  - Secret name is exactly `PAT_TOKEN` (case-sensitive)
  - Secret value is a valid fine-grained PAT (starts with `github_pat_`)
  - Repository access includes `RC219805/Transformation_Portal`

**Error: "Resource not accessible by integration"**

→ PAT permissions insufficient. Verify:
  - `Contents`: Read and write
  - `Pull requests`: Read and write

**PR created but not labeled**

→ PAT may not have PR write permission. Regenerate token with correct permissions.

---

## Step 5: Maintenance

### 5.1 Token Expiration

**90-day expiration** (recommended):
- Set calendar reminder 7 days before expiration
- Regenerate token using same steps (1.1–1.4)
- Update `PAT_TOKEN` secret (Step 2.2)

### 5.2 Rotation Best Practices

1. **Generate new token** before old one expires
2. **Update secret** immediately
3. **Test workflow** manually before relying on scheduled runs
4. **Revoke old token** only after confirming new one works

### 5.3 Security Hygiene

- **Never commit** the PAT to the repository
- **Never share** the PAT in public channels
- **Revoke immediately** if compromised
- **Use minimal permissions** (only what's needed)
- **Regenerate regularly** (every 90 days or less)

---

## Appendix: Workflow Schedule

Current schedule (`.github/workflows/dependency-updates.yml`):

```yaml
on:
  schedule:
    - cron: '0 8 * * 1'  # Every Monday at 8:00 AM UTC
  workflow_dispatch:     # Allow manual trigger
```

**Automated runs**: Every Monday morning  
**Manual trigger**: Available anytime via Actions tab

---

## Appendix: Alternative — GitHub App (Advanced)

For multi-repo automation or team workflows, consider creating a **GitHub App** instead of a PAT:

**Advantages**:
- Fine-grained per-repository permissions
- Token auto-refresh (no manual rotation)
- Audit trail with app identity

**Disadvantages**:
- More complex setup
- Requires webhook endpoint for installation
- Overkill for single-repo automation

**When to use**: Managing 5+ repositories with automated workflows

---

## Summary Checklist

- [ ] Created fine-grained PAT with `Contents: RW` and `Pull requests: RW`
- [ ] Added PAT as `PAT_TOKEN` secret in repository settings
- [ ] Verified workflow uses `${{ secrets.PAT_TOKEN }}`
- [ ] Tested workflow manually and confirmed PR creation
- [ ] Set calendar reminder for token expiration (if < 90 days)
- [ ] Documented token creation date for rotation tracking

**Setup complete!** 🎉 Automated dependency PRs will now work seamlessly.

---

## Support

**Questions?** See:
- [GitHub PAT documentation](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)
- [Fine-grained PAT permissions](https://docs.github.com/en/rest/overview/permissions-required-for-fine-grained-personal-access-tokens)
- [peter-evans/create-pull-request docs](https://github.com/peter-evans/create-pull-request#action-inputs)

**Security concerns?** Contact repository admin or GitHub support.
