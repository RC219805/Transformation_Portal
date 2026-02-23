# Unblocking PR Workflows

This document explains the common blockers for PR workflows and how to resolve them.

## Quick Reference: Blocker Patterns

| Pattern | Symptom | Fix |
|---------|---------|-----|
| **A) Expected — Waiting** | Merge box shows "Expected — Waiting for status to be reported" | Re-anchor required checks in branch protection |
| **B) Checks failing** | Workflows run but fail with permission errors | Add explicit `permissions:` to workflow YAML |
| **C) No checks appear** | No workflows trigger on PR | Verify `on: pull_request:` trigger in workflow YAML |
| **D) Action Required** | Workflows show `action_required` conclusion | Maintainer must approve first-run for new contributors |

---

## Pattern A: "Expected — Waiting for status to be reported"

### Symptoms
- PR merge box shows "Expected — Waiting for status to be reported"
- Required check names don't appear in the Checks list
- Merging is blocked even though workflows seem to exist

### Root Cause
Branch protection is requiring a check name that no longer exists (renamed job, deleted workflow, or check only runs on different triggers).

### Fix
1. Go to **Settings → Branches → Branch protection rule for `main`**
2. Under **Required status checks**, look for:
   - Stale check names from old workflows
   - Check names that don't appear in PR's Checks list
3. Run the workflow at least once on a PR targeting `main`
4. Return to branch protection and **re-select** the required checks from the recognized list
5. **Do not** type check names manually

### Prevention
- Use **unique, stable job names** across all workflows
- Avoid generic names like `test` repeated in multiple workflows
- Example: Use `test (3.11, cpu, core)` instead of just `test`

---

## Pattern B: "Checks failing with permission errors"

### Symptoms
- Workflows run but jobs fail
- Error messages like:
  - `403: Resource not accessible by integration`
  - `The nested job is requesting 'X: write', but is only allowed 'X: read'`

### Root Cause
The `GITHUB_TOKEN` doesn't have required permissions for workflow operations.

### Fix: Add Explicit Permissions

Security best practice: Keep default token **restricted** and grant only what's needed.

**Key rule**: If you add a `permissions:` block, everything not listed becomes `none` (except `metadata: read`).

#### Minimal test-only job (read-only)
```yaml
permissions:
  contents: read
```

#### Job that comments on PRs
```yaml
permissions:
  contents: read
  pull-requests: write
```

#### Job that creates status checks
```yaml
permissions:
  contents: read
  checks: write
  statuses: write
```

#### CodeQL analysis
```yaml
permissions:
  security-events: write
  packages: read
  actions: read
  contents: read
```

### Edge Cases
- **PRs from public forks**: Get read-only token regardless of workflow permissions
- **Dependabot PRs**: May need `security-events: write` for code scanning uploads

---

## Pattern C: "No checks appear at all"

### Symptoms
- Opening/updating a PR shows no checks in the Checks section
- Actions tab shows no workflow runs for the PR

### Root Causes
1. GitHub Actions disabled for the repo
2. Workflow trigger doesn't include `pull_request`
3. Workflow file has invalid YAML
4. Workflow file not on default branch

### Fix

1. **Verify Actions enabled**: Settings → Actions → General
2. **Verify triggers include `pull_request`**:
```yaml
on:
  pull_request:
    branches: [main]
  push:
    branches: [main]
  workflow_dispatch: {}
```
3. **Validate YAML syntax**: Use `yamllint` or GitHub's workflow editor
4. **Ensure workflow is on default branch**: Workflows must exist on `main` to run on PRs

---

## Pattern D: "Action Required" (Copilot/First-Time Contributors)

### Symptoms
- All workflows show `conclusion: action_required`
- Jobs show `total_count: 0`
- PR is from Copilot agent or first-time contributor

### Root Cause
GitHub requires maintainer approval before running workflows from:
- First-time contributors
- Copilot coding agent PRs (depending on org settings)
- PRs that modify workflow files

### Fix
1. Navigate to the PR's Checks tab
2. Click "Approve and run" for each pending workflow
3. Or go to **Actions tab → Pending approvals** and approve all

### Prevention
- Configure org/repo settings to auto-approve workflows from trusted bots
- For Copilot agent: The approval is a one-time action per PR

---

## This Repository's Required Checks

The following checks are **required** for merging to `main`:

| Check Name | Workflow File | Purpose |
|------------|---------------|---------|
| `CI Gate` | `build.yml` | Main CI gate (lint, test, manifest) |
| `Analyze (python)` | `security-unified.yml` | CodeQL Python analysis |
| `Analyze (actions)` | `security-unified.yml` | CodeQL Actions analysis |

To verify these are correctly configured:
1. Open any passing PR
2. Note the exact check names in the Checks section
3. Ensure branch protection lists these exact names

---

## Validation Checklist

After making changes:

- [ ] Open/update a PR to trigger workflows
- [ ] Verify **PR → Checks** shows runs started
- [ ] Verify **Repo → Actions** shows workflow runs
- [ ] Verify **Branch protection** shows required checks being satisfied

---

## See Also

- [GitHub Docs: Required status checks](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)
- [GitHub Blog: Control permissions for GITHUB_TOKEN](https://github.blog/changelog/2021-04-20-github-actions-control-permissions-for-github_token/)
- [CodeQL: Missing workflow permissions](https://codeql.github.com/codeql-query-help/actions/actions-missing-workflow-permissions/)
