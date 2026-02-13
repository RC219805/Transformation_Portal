# CI Workflow Matrix

**Purpose**: Canonical reference for all GitHub Actions workflows
**Owner**: Transformation Portal Architect
**Last Updated**: 2026-02-04

---

## Active Workflows

| Workflow | File | Trigger | Blocking | Purpose | Key Checks |
|----------|------|---------|----------|---------|------------|
| **CI (Lint, Tests & Manifest)** | `build.yml` | PR, push to main | ✅ Yes | Primary quality gate | Lint (flake8, pylint), Tests (3.10, 3.12), Manifest validation |
| **Quality Gate** | `quality-gate.yml` | PR, push to main | ⚠️ Partial | Formatting enforcement | Autopep8, flake8 critical, pylint (non-blocking), markdown count |
| **Security Unified** | `security-unified.yml` | PR, push to main | ✅ Yes | Security scanning | Bandit, Safety, Trivy |
| **CodeQL** | `codeql.yml` | PR, push to main, schedule | ✅ Yes | Code security analysis | Static analysis for vulnerabilities |
| **Dependency Submission** | `dependency-submission.yml` | Push to main | ❌ No | Dependency graph | Submit dependencies to GitHub |
| **Performance Monitor** | `performance-monitor.yml` | Schedule (nightly) | ❌ No | Performance regression | Benchmark tracking |
| **Nightly** | `nightly.yml` | Schedule | ❌ No | Extended test suite | ML tests, integration tests |
| **Python App** | `python-app.yml` | PR, push to main | ⚠️ Unknown | Legacy/duplicate? | Needs audit |
| **Enforcement** | `enforcement.yml` | PR, push to main | ⚠️ Unknown | Policy enforcement | Needs audit |
| **Summary** | `summary.yml` | Workflow completion | ❌ No | Reporting | Aggregate workflow results |

---

## Workflow Responsibilities

### Primary Quality Gate: `build.yml`
**Purpose**: Enforce code quality, type safety, and test coverage
**Runs on**: Every PR and push to main
**Blocking**: Yes (required for merge)

**Jobs**:
1. **Lint** (Python 3.12):
   - flake8 (critical errors only: E9, F63, F7, F82)
   - pylint (on changed files, falls back to filtered full repo)
   - Caches pip dependencies for speed

2. **Test - Core** (Python 3.10, 3.12):
   - pytest with markers: `not ml and not slow`
   - Runs on minimal dependencies (`requirements-ci.txt`)
   - Produces coverage reports

3. **Test - ML** (Python 3.11):
   - pytest with markers: `ml and not slow`
   - Requires ML dependencies
   - Offline mode: `TRANSFORMERS_OFFLINE=1`

4. **Manifest Validation**:
   - Verifies MANIFEST.in correctness
   - Ensures package completeness

5. **Coverage Gate**:
   - Downloads coverage from test jobs
   - Combines reports
   - Enforces minimum coverage threshold

**Concurrency**: Cancels outdated runs on new pushes to same branch/PR

---

### Formatting Enforcement: `quality-gate.yml`
**Purpose**: Auto-fix and enforce formatting standards
**Runs on**: Every PR and push to main
**Blocking**: Partial (some checks non-blocking)

**Jobs**:
1. **Pre-commit Checks**:
   - Auto-fix trailing whitespace with autopep8 (max line length 127)
   - flake8 critical errors only
   - pylint (non-blocking, `continue-on-error: true`)
   - Markdown file count limit (max 10 in root)

**Current Status**: ⚠️ Does NOT enforce black or isort (creates drift risk)

**Recommendation**: Gate 0 should add:
```yaml
- name: Check black formatting
  run: black --check src/ tests/

- name: Check isort
  run: isort --check-only src/ tests/
```

---

### Security Scanning: `security-unified.yml`
**Purpose**: Detect vulnerabilities in code and dependencies
**Runs on**: Every PR and push to main
**Blocking**: Yes

**Jobs**:
1. **Bandit**: Python security linter
2. **Safety**: Dependency vulnerability scanner
3. **Trivy**: Container and filesystem vulnerability scanner

---

### Code Analysis: `codeql.yml`
**Purpose**: GitHub's semantic code analysis
**Runs on**: PR, push to main, weekly schedule
**Blocking**: Yes

**Languages**: Python

---

## Identified Duplication and Consolidation Opportunities

### Duplication 1: Lint Checks
- **build.yml** runs: flake8 (critical), pylint (on changed files)
- **quality-gate.yml** runs: flake8 (critical), pylint (non-blocking)

**Recommendation**: Consolidate lint to `build.yml` only. Make `quality-gate.yml` focus on formatting (black, isort).

---

### Duplication 2: Multiple Workflows on Same Trigger
- `build.yml`, `quality-gate.yml`, `security-unified.yml`, `python-app.yml`, `enforcement.yml` all trigger on `pull_request` and `push` to main

**Audit Required**: Determine if `python-app.yml` and `enforcement.yml` are:
- Legacy workflows (safe to remove)
- Specialized workflows (document distinct purpose)
- Duplicate workflows (consolidate into `build.yml`)

---

### Optimization 1: Change-Aware Execution
**Current**: All tests run on every change (regardless of affected code)

**Proposal**: Use `dorny/paths-filter` to conditionally run:
- ML tests only when `src/transformation_portal/lux_depth_v3/**` changes
- Doc checks only when `docs/**` or `**.md` changes
- Full suite when core modules or CI config changes

**Example**:
```yaml
- name: Detect changes
  id: changes
  uses: dorny/paths-filter@v2
  with:
    filters: |
      ml:
        - 'src/transformation_portal/lux_depth_v3/**'
      docs:
        - 'docs/**'
        - '**.md'
      core:
        - 'src/**'
        - 'tests/**'
        - '.github/workflows/**'

- name: Run ML tests
  if: steps.changes.outputs.ml == 'true' || steps.changes.outputs.core == 'true'
  run: pytest -m ml
```

**Impact**: Reduces CI minutes without compromising safety

---

## Workflow Naming and Branch Protection

### Required Status Checks (Branch Protection)
Current required checks for main branch (needs verification via GitHub settings):
- `CI (Lint, Tests & Manifest) / lint`
- `CI (Lint, Tests & Manifest) / test-core (3.10)`
- `CI (Lint, Tests & Manifest) / test-core (3.12)`
- `CI (Lint, Tests & Manifest) / test-ml (3.11)`
- `Security Unified / security-scan`
- `CodeQL`

**Critical Rule**: Never rename a required check without:
1. Updating branch protection settings first
2. Announcing change in PR description
3. Verifying merge queue still works

---

## CI Runtime Metrics (Baseline)

**Measured on**: 2026-02-04 (commit 1fe9e3c8)

| Workflow | Avg Runtime | Success Rate | Notes |
|----------|-------------|--------------|-------|
| build.yml | ~8-12 min | 95%+ | Stable |
| quality-gate.yml | ~2-3 min | 90%+ | Occasional markdown count failures |
| security-unified.yml | ~5-7 min | 95%+ | Stable |
| codeql.yml | ~10-15 min | 95%+ | Stable |

**Target**: Maintain or improve runtimes with consolidation changes

---

## CI Health Monitoring

### Green CI Criteria
A "green CI" state requires:
- All required checks passing on latest main commit
- No flaky tests (failures that pass on retry)
- No formatting drift (black/isort compliance)

### Red Flags
- Consistent failures in same job across multiple PRs
- Increasing runtime trend (>20% increase over 30 days)
- Coverage decreasing trend

### Monthly Audit
- Review CI runtime trends
- Identify flaky tests
- Update this matrix with any workflow changes

---

## Change Log

| Date | Change | Rationale |
|------|--------|-----------|
| 2026-02-04 | Initial creation | Baseline documentation for Gate 0 |

---

## References

- `docs/architecture/TRANCHE_EXECUTION_PLAN.md`: Gate 0 and tranche execution
- `docs/architecture/agent_governance.md`: Governance policy
- `.github/workflows/`: Workflow implementations

---

**Maintained by**: Transformation Portal Architect
**Review Frequency**: Monthly (or after any workflow change)
