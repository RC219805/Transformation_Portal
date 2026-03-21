# ADR-0016: Change-Aware CI with Path-Based Workflow Triggers

**Status:** Accepted

**Date:** 2026-02-04

**Context:**

Current CI workflows run all jobs on every PR regardless of which files changed. This results in:
- Unnecessary CI minutes consumed (doc-only changes run full test suite)
- Slower feedback cycles (waiting for irrelevant tests to complete)
- Inefficient resource utilization

Analysis of PR patterns shows:
- ~30% of PRs are documentation-only changes
- ~15% of PRs are test-only changes
- ~10% of PRs are workflow/config changes

**Decision:**

Implement path-based workflow triggers to skip irrelevant jobs while maintaining safety guarantees.

## Filter Strategy

### Core Principles
1. **Conservative filters**: False positives (running unnecessarily) are acceptable; false negatives (skipping required checks) are not
2. **Always run on main/release**: Full suite always runs on protected branches (no filters)
3. **Manual override available**: `workflow_dispatch` allows manual full-suite execution
4. **Explicit skip messaging**: Clear logs when jobs are skipped

### Filter Categories

#### 1. Documentation-Only Changes
**Trigger Paths:**
```yaml
- 'docs/**'
- '*.md'
- 'README*'
```

**Skip:** Test jobs (core, ML, integration)
**Run:** Lint (doc validation), docs build job

**Rationale:** Documentation changes cannot break runtime behavior or tests.

#### 2. Test-Only Changes
**Trigger Paths:**
```yaml
- 'tests/**'
- 'conftest.py'
```

**Skip:** Documentation build jobs
**Run:** Lint, all test jobs

**Rationale:** Test changes require validation but don't affect documentation.

#### 3. Code Changes
**Trigger Paths:**
```yaml
- 'src/**'
- 'scripts/**'
- 'pyproject.toml'
- 'requirements*.txt'
- 'setup.py'
```

**Skip:** None (run full suite)
**Run:** All jobs

**Rationale:** Code changes may affect any aspect of the system.

#### 4. Workflow Changes
**Trigger Paths:**
```yaml
- '.github/workflows/**'
- '.github/actions/**'
```

**Skip:** None (run full suite)
**Run:** All jobs

**Rationale:** CI changes must be validated thoroughly; workflow bugs can block all development.

#### 5. Main/Release Branches
**No filters applied** - full suite always runs

**Rationale:** Protected branches require comprehensive validation regardless of change scope.

## Implementation

### Modified Workflows

**build.yml (Primary CI):**
- Add path filters for PR triggers only
- Keep push to main as full-suite (no filters)

**docs.yml (Documentation):**
- Already has path filters (newly created)
- Only runs on doc-relevant changes

**ci.yml (Quality Firewall):**
- No changes (post-merge validation only)

**nightly.yml:**
- No changes (scheduled job)

### Example Path Filter

```yaml
on:
  pull_request:
    paths:
      # Code changes - run full suite
      - 'src/**'
      - 'tests/**'
      - 'scripts/**'
      - 'pyproject.toml'
      - 'requirements*.txt'
      - 'setup.py'
      # Workflow changes - run full suite
      - '.github/workflows/**'
      - '.github/actions/**'
  push:
    branches: [main]
    # No path filters on main - always run full suite
```

### Job-Level Conditionals

For finer control, use job-level `if` conditionals:

```yaml
jobs:
  docs-build:
    if: |
      github.event_name == 'push' ||
      (github.event_name == 'pull_request' && (
        contains(github.event.pull_request.labels.*.name, 'documentation') ||
        startsWith(github.event.pull_request.title, 'docs:')
      ))
```

## Expected Impact

**Metrics:**
- **Doc-only PRs**: 70%+ time reduction (skip tests)
- **Test-only PRs**: 30%+ time reduction (skip doc builds)
- **Overall CI cost**: 20-30% reduction
- **Feedback time**: 2-5 minutes faster for filtered PRs

**Safety Net:**
- Zero false negatives (all required checks run)
- Manual override available via `workflow_dispatch`
- Full suite on main/release branches
- Clear skip messages in logs

## Rollback Procedure

If path filters cause issues:

1. **Immediate rollback:**
   ```bash
   git revert <commit-sha>
   git push origin main
   ```

2. **Temporary bypass:**
   Add label `ci:full-suite` to PR to run all jobs

3. **Investigation:**
   - Review skipped jobs in workflow logs
   - Check for false negatives (required jobs skipped)
   - Adjust path patterns conservatively

## Validation

**Pre-deployment testing:**
1. Create test PRs for each filter category:
   - Doc-only PR (verify tests skipped)
   - Test-only PR (verify docs skipped)
   - Code-only PR (verify all run)
   - Workflow-only PR (verify all run)

2. Verify main branch behavior:
   - Push to main triggers full suite
   - No jobs skipped

3. Manual override:
   - Test `workflow_dispatch` runs full suite

**Monitoring:**
- Track PR completion times (baseline vs filtered)
- Monitor for false negatives (PRs merged without required checks)
- Review workflow run logs weekly

## Alternatives Considered

### 1. Label-Based Triggering
**Rejected:** Requires manual labeling, prone to human error

### 2. Single Monolithic Workflow
**Rejected:** Complex conditional logic, harder to maintain

### 3. No Path Filtering (Status Quo)
**Rejected:** Wastes CI resources, slower feedback

## References

- GitHub Actions: [Workflow syntax - on.<push|pull_request>.paths](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions#onpushpull_requestpull_request_targetpathspaths-ignore)
- [GitHub Actions: Filter pattern cheat sheet](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions#filter-pattern-cheat-sheet)

## Approval

- **Author:** Transformation Portal Specialist
- **Reviewed By:** Repository Architect
- **Date:** 2026-02-04
- **Status:** Accepted for implementation
