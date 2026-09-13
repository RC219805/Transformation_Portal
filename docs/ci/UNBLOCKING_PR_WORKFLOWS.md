# Unblocking PR Workflows

This document explains the common blockers for PR workflows and how to resolve them.

## Quick Reference: Blocker Patterns

| Pattern | Symptom | Fix |
|---------|---------|-----|
| **A) Expected — Waiting** | Merge box shows "Expected — Waiting for status to be reported" | Compare current required contexts with trigger/job names at the PR head |
| **B) Checks failing** | Workflows run but fail with permission errors | Trace the failing operation and event trust boundary before changing permissions |
| **C) No checks appear** | No workflows trigger on PR | Verify `on: pull_request:` trigger in workflow YAML |
| **D) Action Required** | Workflows show `action_required` conclusion | Maintainer must approve first-run for new contributors |

---

## Pattern A: "Expected — Waiting for status to be reported"

### Symptoms
- PR merge box shows "Expected — Waiting for status to be reported"
- Required check names don't appear in the Checks list
- Merging is blocked even though workflows seem to exist

### Possible Causes
A stale check name, an untriggered workflow, approval state, path filter, or a
run attached to a different commit can all leave an expected check pending.
Identify the actual cause before changing protection.

### Diagnosis

```bash
gh pr view <PR-number> --json headRefOid,statusCheckRollup
gh api repos/RC219805/Transformation_Portal/branches/main/protection
```

Match each run to the current head. The main protection snapshot read on
2026-09-12 required only `CI Gate`, with strict up-to-date checking. Its upstream
jobs are classified by `build.yml`; do not add matrix-expanded job names to
protection or reconfigure settings solely to bypass a pending/failing check.
A confirmed stale required context is a separate, explicitly authorized
administrative change.

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
4. **Inspect event-specific workflow availability**: PR workflows can come from
   the proposed change; events such as manual dispatch have default-branch
   availability requirements. Check the actual event instead of assuming every
   PR workflow must already be on `main`.

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

### Evidence boundary
An `action_required` run with zero jobs has not executed tests. Maintainer
approval and rerun requirements depend on repository policy and the current
change; do not assume approval persists for every subsequent commit or weaken
trust settings as a troubleshooting shortcut.

---

## This Repository's Required Checks

> **Authoritative Source**: The definitive list of required checks is defined in
> **Settings → Branches → Branch protection rules for `main`**.
>
> This documentation is explanatory, not declarative. Branch protection is the
> single source of truth for merge requirements.

### How to Find Current Required Checks

1. Go to **Settings → Branches → Edit rule for `main`**
2. Look under **Require status checks to pass before merging**
3. The checked items are the current required checks

### How to Verify Checks Are Working

1. Open any passing PR targeting `main`
2. Look at the **Checks** tab
3. Required checks will show with a ✓ when passing
4. If a required check shows "Expected", see Pattern A above

### Common Check Categories

| Category | Purpose | Workflow Location |
|----------|---------|-------------------|
| CI Gate | Lint, test, manifest validation | `build.yml` |
| Security Analysis | CodeQL scanning | `security-unified.yml`, `codeql.yml` |
| Quality Gates | Code quality enforcement | Various `*-gate.yml` files |

> **Note**: Specific check names may change as workflows evolve. Always verify
> against branch protection settings rather than relying on documentation.

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
