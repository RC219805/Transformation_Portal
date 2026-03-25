# CI Workflow Matrix

**Purpose**: Canonical reference for all GitHub Actions workflows
**Owner**: Transformation Portal Architect
**Last Updated**: 2026-03-25

---

## Workflow Design Principles

1. **`build.yml` is the blocking CI gate** - All PR merge requirements go through this workflow
2. **Scheduled workflows own their domain** - `nightly.yml`, `ml-slow-suite.yml`, `performance-monitor.yml` are non-blocking validation
3. **Actions are SHA-pinned** - All third-party actions reference commit SHAs for supply-chain security
4. **Issue creation is deduplicated** - Automated workflows check for existing open issues before creating new ones

---

## Active Workflows

| Workflow | File | Trigger | Blocking | Purpose |
|----------|------|---------|----------|---------|
| **CI (Lint, Tests & Manifest)** | `build.yml` | PR, push to main, manual | ✅ Yes | Primary quality gate with preflight classifier |
| **Quality Gate** | `quality-gate.yml` | PR, push to main | ⚠️ Advisory | Formatting and structure checks |
| **Security Unified** | `security-unified.yml` | Schedule, PR, push to main, manual | ✅ Yes | Dependency scanning (pip-audit), security gates |
| **CodeQL** | `codeql.yml` | PR, push to main, schedule | ✅ Yes | GitHub semantic code analysis |
| **Enforcement** | `enforcement.yml` | PR, push to main/develop, schedule | ⚠️ Partial | Policy enforcement (action pins, banned deps, HF revisions, artifact boundary) |
| **Dependency Submission** | `dependency-submission.yml` | Push to main/develop, PR, manual | ❌ No | Submit dependencies to GitHub dependency graph |
| **Performance Monitor** | `performance-monitor.yml` | Schedule (3:30 AM UTC), manual | ❌ No | Performance regression tracking (schedule-only) |
| **Nightly** | `nightly.yml` | Schedule (2 AM UTC), manual | ❌ No | Extended validation: stress tests, benchmarks, memory, integration |
| **ML Slow Suite** | `ml-slow-suite.yml` | Schedule (3:30 AM UTC), manual | ❌ No | Slow ML test coverage |
| **AI Code Review** | `ai-code-review.yml` | PR | ❌ No | AI-powered code review comments (advisory) |
| **Issue Summarizer** | `summary.yml` | Issues, PRs, comments | ❌ No | AI-powered issue/PR summarization |
| **Smart Issue Management** | `smart-issue-management.yml` | Issues, PRs (opened/labeled) | ❌ No | AI-powered issue triage and labeling |

---

## Workflow Responsibilities

### Primary Quality Gate: `build.yml`

**Purpose**: Enforce code quality, type safety, and test coverage
**Runs on**: Every PR and push to main
**Blocking**: Yes (required for merge)

**Key Features**:
- **Preflight classifier**: Determines if full or lightweight suite runs based on changed files
- **SHA-pinned actions**: All third-party actions pinned to commit SHAs
- **Concurrency control**: Cancels outdated runs on new pushes

**Jobs**:
1. **Preflight**: Classify changes to determine suite scope
2. **Lightweight checks**: pip-tools cache, docs structure, sanity checks
3. **Lint** (Python 3.12): flake8, pylint, black, isort
4. **Test - Core** (Python 3.10, 3.12): `pytest -m "not ml and not slow and not benchmark"`
5. **Test - ML** (Python 3.11): `pytest -m "ml and not slow and not benchmark"`
6. **Manifest Validation**: MANIFEST.in correctness
7. **Coverage Gate**: Minimum coverage enforcement

### Enforcement: `enforcement.yml`

**Purpose**: Policy enforcement and governance checks
**Runs on**: PR, push to main/develop, nightly schedule
**Blocking**: Partial (some jobs are required)

**Key Features**:
- **Reliable change detection**: Uses `dorny/paths-filter` for accurate PR file detection
- **ML-aware**: Only runs ML tier when ML-related files change

**Jobs**:
1. **Changes**: Classify file changes for conditional job execution
2. **Action Pins**: Verify all workflow actions are SHA-pinned
3. **Banned Dependencies**: Check for prohibited packages
4. **HF Revision Policy**: Validate HuggingFace model revisions
5. **Layer 1 Tests**: Fast unit/regression tests
6. **Layer 2 ML Tests**: ML-specific tests (conditional)
7. **Golden Regression**: Golden path contract tests (conditional)
8. **Artifact Boundary**: Ensure no large artifacts in git

### Performance Monitor: `performance-monitor.yml`

**Purpose**: Performance regression detection
**Runs on**: Schedule (3:30 AM UTC daily), manual dispatch
**Blocking**: No

**Key Features**:
- **Schedule-only**: Does NOT run on PRs (baseline persistence requires cross-run storage)
- **Baseline reading**: Reads from `tools/benchmarks/baseline.json` if present (baseline refresh is manual)
- **Proper status classification**: Distinguishes no tests, passed, regression, and generic failure states
- **Deduplicated issues**: Checks for existing open issues before creating new ones

### Nightly Deep Checks: `nightly.yml`

**Purpose**: Extended validation suite
**Runs on**: Schedule (2 AM UTC daily), manual dispatch
**Blocking**: No

**Key Features**:
- **Concurrency control**: Prevents overlapping scheduled runs
- **Baseline reading**: Uses repo-stored baseline (`tools/benchmarks/baseline.json`) if available
- **Proper status classification**: Distinguishes regression vs generic benchmark failure
- **Deduplicated issues**: Updates existing issues instead of creating duplicates

**Jobs**:
1. **Stress Tests**: Long-running endurance tests
2. **Performance Benchmarks**: Regression testing with proper baseline handling
3. **Memory Leak Detection**: Memory growth profiling
4. **Deep Dependency Audit**: pip-audit + SBOM generation
5. **Full Integration Tests**: Complete integration suite
6. **Nightly Summary**: Aggregated results and failure notification

### ML Slow Suite: `ml-slow-suite.yml`

**Purpose**: Slow ML test coverage
**Runs on**: Schedule (3:30 AM UTC daily), manual dispatch
**Blocking**: No

**Key Features**:
- **Concurrency control**: Prevents overlapping scheduled runs
- **Model caching**: Caches HuggingFace models
- **Deduplicated issues**: Updates existing issues instead of creating duplicates

---

## Governance Notes

### Action Pinning

All workflows must pin third-party actions to commit SHAs:
```yaml
# ✅ Good - SHA pinned
- uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2

# ❌ Bad - floating tag
- uses: actions/checkout@v6
```

Enforcement: `enforcement.yml` → `action-pins` job

### Issue Deduplication

Automated failure notifications must check for existing open issues:
```javascript
const { data: issues } = await github.rest.issues.listForRepo({
  owner: context.repo.owner,
  repo: context.repo.repo,
  state: 'open',
  labels: 'relevant,labels',
  per_page: 10
});

const existingIssue = issues.find(i => i.title.includes('Expected Title'));
if (existingIssue) {
  // Update existing issue
} else {
  // Create new issue
}
```

### PR Change Detection

For conditional job execution on PRs, use `dorny/paths-filter` instead of unreliable `github.event.head_commit.modified`:
```yaml
- uses: dorny/paths-filter@de90cc6fb38fc0963ad72b210f1f284cd68cea36  # v3.0.2
  with:
    filters: |
      ml:
        - 'src/transformation_portal/ml/**'
```

---

## Change Log

| Date | Change | Rationale |
|------|--------|-----------|
| 2026-03-25 | Major update: Fixed enforcement.yml PR detection, performance-monitor.yml baseline handling, nightly.yml deduplication, pinned all actions | Address workflow correctness bugs and governance hygiene |
| 2026-02-04 | Initial creation | Baseline documentation |

---

**Maintained by**: Transformation Portal Architect
**Review Frequency**: Monthly (or after any workflow change)
