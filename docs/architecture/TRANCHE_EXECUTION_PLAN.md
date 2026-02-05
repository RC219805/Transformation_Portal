# Tranche Execution Plan (8/21 → 11/21 complete)

**Architectural Decision**: Enforce CI health before throughput improvements  
**Authority**: Transformation Portal Architect  
**Date**: 2026-02-04  
**Status**: BINDING

---

## Executive Summary

Epic #819 tracks 21 improvement opportunities; **7 are complete** (33%). PRs #822 and #823 are currently open but **cannot be safely merged** while CI exhibits systemic failures in Lint, Type Check, and Coverage Gate.

**The fundamental constraint**: You cannot build throughput on unstable infrastructure.

This plan mandates a **Gate 0** (CI stabilization) before proceeding with the next tranche of improvements.

---

## Current State Assessment

### Completed Items (7/21) ✅
- SEC-001: Fixed `shell=True` in subprocess (PR #820, commit aca4eb2)
- CI-002: Added pip cache to build.yml (PR #820)
- CI-003: Added concurrency control (PR #821, commit 1fe9e3c)
- TEST-003: Added test-integration target (PR #820)
- SEC-002: Implemented input validation for config loader (PR #820)
- PERF-001: Completed YAML loading implementation (PR #820)
- PERF-003: Added xxHash as default hasher (PR #820)

### Open/In-Progress Items
- **PR #822**: Coverage artifact upload fix (targets Issue #815)
- **PR #823**: SEC-001 command injection hardening in `fix_quality_issues.py` (draft)

### CI Health Status ⚠️
**Reality check from live CI runs:**

1. **Quality Gate workflow** (`quality-gate.yml`): Recent runs show **success** on main
2. **Build workflow** (`build.yml`): Recent runs show **success** on main
3. **Local formatting check** reveals critical drift:
   - **Black would reformat 265 files** in `src/` and `tests/`
   - **isort failures** across core modules (imports incorrectly sorted)
   - **Type checking**: Enforcement unclear (not visible in recent workflow runs)

**Diagnosis**: CI is currently **green on main** but will fail on any PR that triggers strict formatting enforcement if those checks exist. The repository has accumulated **significant formatting debt** (265 files out of compliance with Black).

---

## Architectural Constraint: Gate 0 (Non-Negotiable)

### Gate 0: Establish Stable CI Baseline

**Priority**: P0 (blocks all other work)  
**Rationale**: Formatting drift of 265 files creates two failure modes:
1. **Silent drift**: PRs accumulate more formatting violations
2. **Surprise rejection**: Future PRs fail lint when strict enforcement is added

**Enforcement strategy required before Week 1 begins.**

### Strategy Decision Matrix

#### Option A: One-Time Baseline Formatting (RECOMMENDED)

**Execution**:
```bash
# Apply formatting to entire codebase
black src/ tests/
isort src/ tests/

# Commit as single mechanical PR
git checkout -b chore/baseline-formatting
git add src/ tests/
git commit -m "chore(format): Apply black + isort baseline (no logic changes)

- Reformats 265 files to black standards
- Applies isort to all modules
- Zero functional changes (verified by test suite)
- Establishes clean baseline for future PRs

Closes #<new-issue-number>"
```

**Pros**:
- Simplest long-term maintenance
- Clean slate for future PRs
- One-time blame history noise

**Cons**:
- Large diff (~265 files)
- Git blame requires `--ignore-revs` or GitHub's "Ignore commits" feature

**Mitigation for blame history**:
Create `.git-blame-ignore-revs` with the formatting commit SHA and document in `.github/CONTRIBUTING.md`.

**Success Criteria**:
- `black --check src/ tests/` passes
- `isort --check-only src/ tests/` passes
- All existing tests pass (no logic changes)
- CI workflows updated to enforce black + isort on future PRs

---

#### Option B: Ratcheting Enforcement (Complex, Deferred)

**Execution**:
- Modify lint job to check only changed files (diff-based)
- Keep formatting strict on new/modified code
- Create backlog item for incremental cleanup

**Pros**:
- Avoids large diff now
- Gradual improvement

**Cons**:
- More CI complexity (diff detection, edge cases)
- Partial compliance creates confusion
- Requires custom enforcement logic

**Recommendation**: Defer to future ADR if baseline approach proves unacceptable.

---

### Gate 0 Additional Requirements

1. **Resolve PR #822 status**:
   - If it correctly fixes coverage gate artifact flow → merge after baseline formatting
   - If it requires rework → close and create new issue with findings

2. **Type Check enforcement decision**:
   - Current state: No visible type check failures in recent CI runs
   - Required action: Audit `build.yml` and `quality-gate.yml` for type checking
   - Decision options:
     - Make type check **non-blocking** (warning-only) until coverage improves
     - Make type check **narrow** (specific modules only, blocking)
     - Make type check **strict** (all modules, blocking)
   - **Mandate**: Document decision in `docs/ci/TYPE_CHECKING_POLICY.md`

3. **CI workflow audit**:
   - Document which workflows enforce which checks (lint, format, type, test, coverage)
   - Identify any redundant execution paths
   - Create artifact: `docs/ci/WORKFLOW_MATRIX.md`

---

## Success Criteria (Gate 0 Exit Conditions)

**Required**:
- [ ] Main branch CI shows **all required checks green** for 3+ consecutive commits
- [ ] `black --check src/ tests/` exits 0
- [ ] `isort --check-only src/ tests/` exits 0
- [ ] Type checking policy documented and enforced consistently
- [ ] Coverage gate artifact flow verified (PR #822 merged or superseded)

**Evidence**:
- Screenshot of GitHub Actions summary showing all green checks
- Local verification commands in PR description

**Timeline**: 2-3 days (includes PR review cycle)

---

## Tranche 1 Execution Plan (Post-Gate 0)

**Scope**: Complete 3 items (8/21 → 11/21)  
**Timeline**: 3 weeks  
**Cadence**: 1 item per week, merged to main before next item begins

### Week 1: TEST-001 — Shared Test Fixtures

**Priority**: HIGH  
**Effort**: 4 hours (realistic with scope control)  
**Impact**: Strong force multiplier for future test development

#### Refined Scope

**Problem**: Test suite spans core + ML markers; CI runs split test jobs. Fixtures must not import heavy ML libraries or slow down core test jobs.

**Deliverables**:

1. **Create `tests/conftest.py` with three fixture tiers**:

   ```python
   # Tier 1: Pure (no IO, no heavy deps)
   @pytest.fixture
   def dummy_config():
       """Minimal config object for unit tests."""
       return {"preset": "standard", "output_dir": "/tmp/test"}
   
   @pytest.fixture
   def deterministic_rng():
       """Fixed RNG seed for reproducible tests."""
       import random
       random.seed(42)
       yield random
   
   # Tier 2: IO (tmp files, small test assets)
   @pytest.fixture
   def temp_output_dir(tmp_path):
       """Temporary directory for test outputs."""
       output = tmp_path / "output"
       output.mkdir()
       return output
   
   @pytest.fixture
   def sample_yaml_config(tmp_path):
       """Minimal valid YAML config file."""
       config_path = tmp_path / "config.yaml"
       config_path.write_text("preset: standard\nquality: 95\n")
       return config_path
   
   # Tier 3: Optional/ML (guarded with pytest.importorskip)
   @pytest.fixture
   def mock_depth_model():
       """Mock depth estimation model (ML tests only)."""
       pytest.importorskip("torch")
       # Return mock or lightweight model
   ```

2. **Consolidate duplicated patterns** across existing tests:
   - Temporary directory creation (→ use `temp_output_dir`)
   - Config file mocking (→ use `sample_yaml_config`)
   - Deterministic RNG setup (→ use `deterministic_rng`)

3. **Measure impact**:
   - Target: ≥5 fixtures defined
   - Target: ≥3 test files reduce LOC by 10-20%
   - Measure: `git diff --stat` before/after in PR description

#### Success Criteria

- [ ] `tests/conftest.py` exists with ≥5 reusable fixtures
- [ ] ≥3 test files refactored to use shared fixtures (LOC reduction documented)
- [ ] No new cross-test coupling (fixtures are narrow and composable)
- [ ] Tests pass in both `pytest -m "not ml"` and `pytest -m "ml"` slices
- [ ] CI remains green (core tests + ML tests on Python 3.10, 3.11, 3.12)

#### Innovation Opportunity (Optional)

Add 1-2 **property-based tests** using Hypothesis for:
- Config parsing edge cases (empty strings, special characters, max lengths)
- Path sanitization boundaries (relative paths, symlinks, directory traversal attempts)

**Rationale**: Property-based testing finds edge cases quickly and pays long-term dividends.

---

### Week 2: DOC-001 — Documentation Consolidation

**Priority**: HIGH  
**Effort**: 3 hours (constrained scope)  
**Impact**: Onboarding + maintainability

#### Refined Scope

**Problem**: Documentation consolidation fails when it becomes taste-based. Need **canonicalization rule**.

**Deliverables**:

1. **Create `DOCUMENTATION_INDEX.md` in repository root**:
   - Lists 3-5 primary entry points only
   - Each entry includes: title, path, intended audience, 1-sentence description
   
   Example:
   ```markdown
   # Documentation Index
   
   ## For Users
   - **README.md** — Quick start, installation, basic usage
   - **docs/user_guide/QUICKSTART.md** — Step-by-step first-time setup
   
   ## For Contributors
   - **CONTRIBUTING.md** — Development setup, PR guidelines, testing
   - **docs/architecture/README.md** — System design, module boundaries
   
   ## For Operators
   - **docs/deployment/DOCKER.md** — Containerized deployment
   - **docs/ci/WORKFLOW_MATRIX.md** — CI pipeline reference
   ```

2. **Canonicalize documentation by topic**:
   - For each topic (setup, quickstart, CI/CD, security), identify **one canonical doc**
   - Add deprecation headers to superseded docs:
     ```markdown
     > **⚠️ DEPRECATED**: This document has been superseded by `docs/path/to/canonical.md`.
     > Redirecting in 30 days (2026-03-06).
     ```
   - Delete duplicates **only after** 30-day redirect period

3. **Audit "mystery directories"**:
   - Any directory with >3 files but no README.md gets:
     - Either: a README.md explaining purpose
     - Or: removal (if obsolete/artifact directory)

#### Success Criteria

- [ ] `DOCUMENTATION_INDEX.md` exists in repository root
- [ ] 1 canonical doc identified per topic (setup, quickstart, CI, security, architecture)
- [ ] Deprecated docs marked with redirect headers (30-day sunset)
- [ ] No directories with >3 files lacking a README.md
- [ ] Root README.md links to DOCUMENTATION_INDEX.md

#### Enforcement

Update `.github/pull_request_template.md` to include:
```markdown
## Documentation Checklist
- [ ] If this PR changes user-facing behavior, I updated the canonical doc in DOCUMENTATION_INDEX.md
- [ ] If this PR adds a new directory with >3 files, I added a README.md
```

---

### Week 3: CI-001 (Phase 1) — Workflow Consolidation

**Priority**: HIGH  
**Effort**: 6 hours  
**Impact**: Reduced cognitive load, fewer flaky edges

#### Refined Scope

**Problem**: Multiple CI layers exist (build.yml, quality-gate.yml, etc.). Duplication is only acceptable if workflows serve distinct purposes with clear boundaries.

**Phase 1 Goal**: Identify and eliminate **one** duplicated execution path.

**Deliverables**:

1. **Create `docs/ci/WORKFLOW_MATRIX.md`**:
   ```markdown
   # CI Workflow Matrix
   
   | Workflow | Trigger | Purpose | Checks |
   |----------|---------|---------|--------|
   | build.yml | PR, push to main | Core gate | Lint, Type, Test (3.10/3.12), Coverage |
   | quality-gate.yml | PR, push to main | Quality enforcement | Format, Markdown count, Pylint |
   | security-unified.yml | PR, push to main | Security scan | Bandit, Safety, Trivy |
   | ... | ... | ... | ... |
   ```

2. **Identify duplication**:
   - Example: If both `build.yml` and `quality-gate.yml` run flake8, consolidate to **one** workflow
   - Document decision: "Lint enforcement lives in build.yml; quality-gate.yml focuses on formatting"

3. **Consolidate one duplicated path**:
   - Option A: Make `quality-gate.yml` call a reusable workflow from `build.yml`
   - Option B: Merge quality-gate checks into `build.yml` and delete `quality-gate.yml`
   - Option C: Clarify boundaries (build = blocking gate, quality = non-blocking reporting)

#### Success Criteria

- [ ] `docs/ci/WORKFLOW_MATRIX.md` exists and documents all workflows
- [ ] One duplicated check eliminated (documented in PR)
- [ ] CI runtime same or improved (measure total job minutes in GitHub Actions)
- [ ] No new required checks added to branch protection without stable naming

#### Innovation Opportunity (Optional)

**Change-Aware CI**:
```yaml
# Only run ML tests if src/transformation_portal/lux_depth_v3/ changes
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

- name: Run ML tests
  if: steps.changes.outputs.ml == 'true'
  run: pytest -m ml
```

**Impact**: Reduces CI cost without compromising safety.

---

## Operational Discipline (Tranche Hygiene)

### 1. Epic Accounting
**Rule**: Every merged PR **must** update Epic #819 checkbox count and link to the PR.

**Enforcement**:
- Add to PR template:
  ```markdown
  ## Epic Tracking
  - [ ] This PR updates Epic #819 with completion status and PR link
  ```

### 2. Auto-Close Issues
**Rule**: Use `Fixes #<issue-number>` in PR descriptions to auto-close issues on merge.

**Example**:
```markdown
## Summary
This PR implements shared test fixtures in `tests/conftest.py`.

Fixes #<TEST-001-issue-number>
Part of Epic #819
```

### 3. PR Sizing Rule
**Rule**: Keep PRs under **300-500 LOC** except:
- Mechanical formatting PRs (explicitly marked in title: `chore(format): ...`)
- Generated code (explicitly marked in description)
- Test data fixtures (explicitly marked in description)

**Enforcement**: PR template includes:
```markdown
## PR Size
- [ ] This PR is <500 LOC (excluding generated/mechanical changes)
- [ ] If >500 LOC, I've added a justification in the description
```

---

## Escalation and Decision Authority

**This plan is binding** under the Transformation Portal Architect authority scope:
- CI/CD policy and required gates (Gate 0)
- Cross-module integration contracts (test fixtures)
- Repository structure (documentation canonicalization)
- Dependency governance (workflow consolidation)

**Specialist collaboration**:
- Implementation details (fixture design, test refactoring) delegated to @transformation-portal-specialist
- Architect retains approval authority for approach and enforcement mechanisms

**Deviation protocol**:
- Any deviation from this plan requires:
  1. Explicit escalation to Architect
  2. Written rationale (ADR if architectural impact)
  3. Updated success criteria and timeline

---

## Metrics and Reporting

### Weekly Progress Report Template

```markdown
## Week N Progress Report

**Tranche Goal**: [Item name]

### Completed
- [ ] Deliverable 1
- [ ] Deliverable 2
- [ ] Success Criterion 1
- [ ] Success Criterion 2

### Blockers
- [None | Description of blocker + mitigation plan]

### Next Week
- [Preview of next item]

### Epic Status
- Completed: X/21 (Y%)
- In Progress: Z
- Blocked: 0
```

### Dashboard
Track in Epic #819 description:
```markdown
## Progress Dashboard
- **Baseline (2026-02-04)**: 7/21 complete (33%)
- **Gate 0 completion**: 2026-02-XX
- **Tranche 1 completion**: 2026-02-XX
- **Projected completion**: 2026-XX-XX
```

---

## Appendix: Gate 0 Execution Checklist

**Owner**: Repository maintainer  
**Timeline**: 2-3 days

- [ ] **Day 1**: Run baseline formatting
  - [ ] `black src/ tests/`
  - [ ] `isort src/ tests/`
  - [ ] Verify tests pass: `pytest -v tests/ -ra -m "not ml and not slow"`
  - [ ] Create PR: "chore(format): Apply black + isort baseline"
  - [ ] Add `.git-blame-ignore-revs` with formatting commit SHA

- [ ] **Day 2**: Update enforcement
  - [ ] Add black check to CI: `black --check src/ tests/`
  - [ ] Add isort check to CI: `isort --check-only src/ tests/`
  - [ ] Document type checking policy in `docs/ci/TYPE_CHECKING_POLICY.md`
  - [ ] Review PR #822 (coverage artifact fix)

- [ ] **Day 3**: Verification
  - [ ] Merge baseline formatting PR
  - [ ] Verify CI green on main (3+ consecutive commits)
  - [ ] Close Gate 0 issue
  - [ ] Announce Tranche 1 start date

---

## References

- Epic #819: Improvement Opportunities Execution Plan
- PR #820: Quick wins delivery (7 items)
- PR #821: Concurrency control
- PR #822: Coverage artifact fix
- PR #823: SEC-001 command injection hardening
- `docs/architecture/agent_governance.md`: Governance policy

---

**Approved**: Transformation Portal Architect  
**Effective**: 2026-02-04  
**Review**: After Tranche 1 completion (3 weeks post-Gate 0)
