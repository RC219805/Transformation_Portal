# ML CI Optimization: Architectural Review & Strategic Guidance

**Status:** Architect Review - Ready for Implementation
**Date:** 2026-02-18
**Authority:** Transformation Portal Architect
**Context:** ML CPU tests (Py3.11) taking 30+ minutes on every PR
**Goal:** Implement surgical CI optimization without breaking quality gates

---

## Executive Summary

**ARCHITECT DECISION: APPROVED WITH MODIFICATIONS**

The proposed ML CPU test optimization plan is **architecturally sound** and aligns with existing ADRs (ADR-031, ADR-032) and repository governance principles. The plan correctly identifies the core problem (30+ minute ML tests on every PR) and proposes appropriate tiering (PR default vs main push vs nightly).

**Key Modifications Required:**
1. **DO NOT delete `AGENT_TOOLSDIRECTORY`** - This is already a known anti-pattern (see incident 2026-02-03)
2. Implement path-based conditional execution with explicit governance
3. Create dedicated `requirements/ml-ci.txt` with CPU-optimized torch pins
4. Enforce test duration monitoring as Quality Firewall gate
5. Document CI tier contracts in ADR format

---

## 1. Alignment with Existing ADRs

### ADR-031: Test Dependency Isolation ✅

**Alignment:** STRONG

The proposal to split ML tests into a separate job and make them conditional **directly implements ADR-031 principles**:

```yaml
# Proposed structure aligns with ADR-031 isolation contract
test-core:
  # NO ML dependencies installed
  run: pytest -m "not ml"

test-ml:
  needs: [preflight, run_ml_check]
  if: needs.run_ml_check.outputs.run_ml == 'true'
  # ML dependencies installed ONLY when needed
  run: pytest -m "ml"
```

**ADR-031 Quote (line 100):**
> Core CI job runs without ML dependencies for speed (~30s vs ~5min)

Your proposal extends this principle: **core tests always run fast; ML tests run conditionally**.

**Enforcement Requirement:**
- The `run_ml` output MUST fail-safe to `true` on main/develop branches
- Path matcher MUST be validated in `test-isolation` job to prevent drift

---

### ADR-032: Dependency Pinning Strategy ✅

**Alignment:** STRONG with one REQUIRED change

The proposal to create a dedicated requirements file for ML CI aligns with ADR-032's pinning taxonomy:

**Proposed:**
```
requirements/ml-ci.txt  # CPU-optimized torch, pinned versions
```

**ADR-032 Constraint Style (line 88):**
> Range Pin (package>=X.Y,<Z) - DEFAULT for production dependencies

**ARCHITECT DIRECTIVE:**

Create `requirements/ml-ci.in` (not just .txt) to maintain the pip-compile workflow:

```ini
# requirements/ml-ci.in
# ML dependencies for CI - CPU-only, pinned for speed and determinism

# CPU-only PyTorch (explicit index URL required)
# Pin to specific versions for deterministic CI behavior
torch==2.10.0+cpu       # Strict pin: CI reproducibility
torchvision==0.25.0+cpu # Strict pin: CI reproducibility

# Include base ML deps but skip heavy optional deps
-c constraints.txt      # Enforce banned packages
transformers>=4.57.0,<6
diffusers>=0.38.0,<1
sentence-transformers>=3.1.0,<6  # CVE-73169 minimum
controlnet-aux>=0.0.6,<1
huggingface-hub>=0.19.0,<2

# Skip these heavy deps in CI:
# - accelerate (not needed for CPU tests)
# - coremltools (Apple Silicon only)
# - rawpy (tested separately)
# - efficientsam/open-clip-torch (tested in integration tier)
```

**Then compile:**
```bash
cd requirements && pip-compile ml-ci.in --output-file ml-ci.txt
```

**Rationale:**
- Maintains ADR-032 governance (abstract .in → compiled .txt)
- Pins torch/torchvision strictly (ADR-032 line 73: deterministic behavior critical)
- Preserves banned package enforcement via constraints.txt
- Enables quarterly review cycle (ADR-032 Appendix B)

---

## 2. ML-Relevant Paths for Conditional Execution

Based on codebase analysis, the `run_ml` output should trigger on changes to:

### Source Paths (HIGH CONFIDENCE)
```yaml
paths:
  # Core ML pipelines
  - 'src/transformation_portal/depth/**'
  - 'src/transformation_portal/depth_intelligence/**'
  - 'src/transformation_portal/upscaling/**'
  - 'src/transformation_portal/diffusion/**'
  - 'src/transformation_portal/spatial_ai/segmentation/**'
  - 'src/transformation_portal/spatial_ai/materials/**'  # Material classifier
  - 'src/transformation_portal/spatial_ai/reconstruction/**'  # Phase 6A Gaussian
  - 'src/transformation_portal/spatial_ai/ingest/**'  # Linear decoder (RAW/TIFF)

  # Style transfer and enhancement
  - 'src/transformation_portal/style_transfer/**'
  - 'src/transformation_portal/enhancers/**'
  - 'src/transformation_portal/neuroaesthetics/**'
  - 'src/transformation_portal/perceptual/**'

  # Core device/model utilities
  - 'src/transformation_portal/core/device/**'  # Device detection affects ML

  # VLM and foundation models
  - 'src/transformation_portal/vlm/**'
  - 'src/transformation_portal/foundation/**'
```

### Test Paths (HIGH CONFIDENCE)
```yaml
  # Any test marked with @pytest.mark.ml
  - 'tests/smoke/test_ml_stack_smoke.py'
  - 'tests/integration/test_depth_pro_integration.py'
  - 'tests/materials/test_segmentation_backend.py'
  - 'tests/spatial_ai/segmentation/**'
  - 'tests/spatial_ai/materials/**'
  - 'tests/spatial_ai/reconstruction/**'
  - 'tests/stage_graph/test_upscaling_stage.py'
  - 'tests/test_upscaling*.py'
  - 'tests/test_raw_loader.py'
  - 'tests/unit/depth/**'
```

### Dependency Paths (CRITICAL)
```yaml
  # Changes to ML dependencies MUST trigger ML tests
  - 'requirements/ml.in'
  - 'requirements/ml.txt'
  - 'requirements/ml-ci.in'  # NEW: proposed CI-specific deps
  - 'requirements/ml-ci.txt'  # NEW: proposed CI-specific deps
  - 'pyproject.toml'  # Changes to [project.optional-dependencies.ml]
```

### Configuration Paths (MEDIUM CONFIDENCE)
```yaml
  # CI config changes should trigger ML tests
  - '.github/workflows/ci-quality-firewall.yml'
  - '.github/workflows/nightly.yml'

  # ML-related config files
  - 'config/depth_*.yaml'
  - 'config/upscaling_*.yaml'
```

### FAIL-SAFE RULE

```yaml
run_ml:
  # ALWAYS run ML tests on main/develop branches
  # ALWAYS run ML tests if any path match is ambiguous
  # Default to TRUE on workflow_dispatch
  default: ${{ github.ref == 'refs/heads/main' || github.ref == 'refs/heads/develop' }}
```

**Rationale:**
- Main branch is Quality Firewall - ALL tests must run
- PRs can skip ML if no relevant paths changed
- Prevents regression from "clever" path matching

---

## 3. Current pyproject.toml Structure Assessment

### Strengths ✅

1. **Clear ML boundary** (line 38):
   ```toml
   [project.optional-dependencies]
   ml = [ ... ]
   ```
   ML dependencies are properly isolated as optional extras.

2. **pytest markers defined** (line 136-145):
   ```toml
   markers = [
       "slow: marks tests as slow",
       "ml: tests requiring ML models/large downloads",
       ...
   ]
   ```
   Repository already uses `@pytest.mark.ml` convention consistently.

3. **Coverage configuration** (line 148-183):
   Supports parallel coverage collection (required for separate test jobs).

### Gaps and Recommendations ⚠️

1. **Missing CI tier documentation in pyproject.toml**

   **RECOMMENDATION:** Add marker documentation for CI tiers:

   ```toml
   [tool.pytest.ini_options]
   markers = [
       "slow: marks tests as slow (deselect with '-m \"not slow\"')",
       "ml: tests requiring ML models/large downloads (CI: conditional on path match)",
       "unit: fast unit tests (<1s each)",
       "regression: regression tests with known fixtures",
       "integration: tests requiring multiple components (CI: runs on main push)",
       "golden: golden regression tests with curated fixtures",
       "stress: stress tests (CI: typically excludes; run manually or nightly)",
       "benchmark: performance benchmarks (CI: typically excludes; run manually or nightly)",
   ]
   ```

2. **No `--durations` flag in addopts**

   **RECOMMENDATION:** Add to pyproject.toml:

   ```toml
   [tool.pytest.ini_options]
   addopts = "--strict-markers --tb=short -p no:warnings --durations=20"
   ```

   This will surface the 20 slowest tests in every run, enabling quarantine decisions.

3. **Coverage enabled by default**

   Current `test-ml` job runs coverage (line 527):
   ```yaml
   --cov=src/transformation_portal
   ```

   **RECOMMENDATION:** Disable coverage for ML job (PR tier only):

   ```yaml
   test-ml:
     if: needs.run_ml_check.outputs.run_ml == 'true' && github.event_name == 'pull_request'
     run: |
       pytest -m "ml" --durations=20  # NO --cov
   ```

   **Rationale:**
   - ML tests are slow due to model loading, not complexity
   - Coverage data is low-signal for ML integration tests
   - Main branch can still run with coverage for quality metrics

---

## 4. Risk Assessment & Mitigations

### RISK 1: Path-based `run_ml` matcher has false negatives

**Severity:** HIGH (breaks Quality Firewall if ML changes undetected)

**Scenario:**
```python
# src/transformation_portal/cli/commands.py
# NOT in ML path list, but imports depth pipeline
from transformation_portal.depth import estimate_depth
```

**Mitigation:**
1. **Fail-safe to TRUE on main/develop** (see section 2)
2. **Weekly audit of `run_ml=false` PRs** (manual spot-check for 30 days)
3. **Add enforcement job:**

   ```yaml
   validate-ml-skip:
     name: Validate ML Test Skip Decision
     if: needs.run_ml_check.outputs.run_ml == 'false'
     steps:
       - name: Check for hidden ML imports
         run: |
           # Grep changed files for ML import patterns
           git diff origin/main --name-only | \
             xargs grep -l "from.*\(depth\|upscaling\|diffusion\|spatial_ai\)" && \
             echo "ERROR: ML imports detected but run_ml=false" && exit 1
   ```

### RISK 2: AGENT_TOOLSDIRECTORY deletion breaks CI (AGAIN)

**Severity:** CRITICAL (incident 2026-02-03 repeat)

**Current Code (line 485, 389):**
```yaml
- name: Free disk space
  run: |
    sudo rm -rf "$AGENT_TOOLSDIRECTORY"
```

**ARCHITECT DIRECTIVE: STOP DOING THIS**

**Evidence:**
- `docs/incidents/2026-02-03_ci_python_toolcache_deletion.md`
- Issue #806: "CI completely broken on main"
- PR #813: Root cause fix was to REMOVE this deletion

**The Problem:**
- GitHub Actions installs Python to `/opt/hostedtoolcache`
- Deleting this breaks subsequent Python commands
- Even if reordered, it's fragile and unnecessary

**REQUIRED CHANGE:**

Replace with **selective cleanup** that preserves toolcache:

```yaml
- name: Free disk space (selective)
  run: |
    # Remove large directories that are NOT toolcache
    sudo rm -rf /usr/share/dotnet
    sudo rm -rf /opt/ghc
    sudo rm -rf /usr/local/share/boost
    sudo docker image prune --all --force

    # CRITICAL: DO NOT delete AGENT_TOOLSDIRECTORY
    # See: docs/incidents/2026-02-03_ci_python_toolcache_deletion.md

    df -h  # Verify free space
```

**Alternative:**
Use GitHub's native disk cleanup action (if disk space is actually a problem):
```yaml
- uses: jlumbroso/free-disk-space@v1
  with:
    tool-cache: false  # NEVER delete tool cache
```

**If you must free space from toolcache:**
- Do it BEFORE `actions/setup-python@v5`
- Document rationale in workflow comment
- Add diagnostic step to verify Python survives

### RISK 3: scikit-learn force-reinstall loop

**Severity:** MEDIUM (wastes CI time, confuses dependency resolution)

**Current Pattern (hypothetical from proposal):**
```bash
# Some CI workflows do this unnecessarily
pip uninstall -y scikit-learn
pip install scikit-learn
```

**ARCHITECT DIRECTIVE: REMOVE THIS**

**Analysis:**
- `scikit-learn` is in `base.in` (line 19 of pyproject.toml): `scikit-learn>=1.0,<2`
- It's a CORE dependency, not ML-specific
- Force-reinstall suggests dependency conflict (anti-pattern)

**If this pattern exists, root cause is likely:**
1. Conflicting version pins in different requirements files
2. Over-constrained transitive dependencies
3. Mixing `pip install -e .` with `-r requirements.txt`

**Proper Fix:**
1. Audit all requirements files for sklearn version conflicts
2. Let pip's resolver handle it (it's good at this)
3. If conflicts persist, add explicit pin to `constraints.txt`

### RISK 4: "Fallback to not slow" rerun logic

**Severity:** MEDIUM (hides flaky tests, wastes CI resources)

**Pattern (hypothetical):**
```yaml
- name: Run ML tests
  run: pytest -m "ml"
  continue-on-error: true

- name: Retry without slow tests
  if: failure()
  run: pytest -m "ml and not slow"
```

**ARCHITECT DIRECTIVE: KILL THIS PATTERN**

**Why it's wrong:**
- Masks test flakes (ADR-033: Test Flake Management should quarantine instead)
- Reduces test coverage on retries (defeats the purpose)
- Makes CI results non-deterministic

**Proper Fix:**
1. **Quarantine flaky tests** (ADR-033 pattern):
   ```python
   @pytest.mark.ml
   @pytest.mark.flaky(reruns=3, reruns_delay=5)
   def test_flaky_ml_thing():
       ...
   ```

2. **Use `--durations=20` to identify slowpokes:**
   ```bash
   pytest -m "ml" --durations=20
   ```
   Output:
   ```
   === slowest 20 test durations ===
   120.45s test_sam2_segment_large_image
   45.23s test_depth_pro_4k_estimate
   ```

3. **Quarantine unreasonably slow tests:**
   ```python
   @pytest.mark.ml
   @pytest.mark.slow  # Exclude from PR CI
   def test_sam2_segment_large_image():
       ...
   ```

### RISK 5: Coverage overhead in ML job

**Severity:** LOW (acceptable cost for quality metrics)

**Current Behavior (line 527):**
```yaml
pytest -m "ml" --cov=src/transformation_portal
```

**Coverage overhead estimate:**
- ~10-15% runtime penalty for tracing
- Not significant compared to ML model load times

**RECOMMENDATION:**
Keep coverage ENABLED on main branch (Quality Firewall needs metrics).
Make coverage OPTIONAL on PR tier:

```yaml
test-ml:
  steps:
    - name: Run ML tests
      run: |
        if [ "${{ github.event_name }}" == "pull_request" ]; then
          # PR tier: skip coverage for speed
          pytest -m "ml" --durations=20
        else
          # Main/develop: collect coverage
          pytest -m "ml" --durations=20 --cov=src/transformation_portal
        fi
```

---

## 5. CI Tier Contracts (Proposed ADR Structure)

**ARCHITECT REQUIREMENT:**

Document CI tiers in a new ADR-034 before implementing:

```markdown
# ADR-034: CI Test Execution Tiers

## Tiers

### Tier 1: PR Default (Fast Feedback)
- **Trigger:** Every PR push
- **Tests:** Core only (`-m "not ml and not slow"`)
- **Coverage:** Full coverage, fail <25%
- **Timeout:** 20 minutes max
- **Goal:** Fast feedback loop for developers

### Tier 2: Main Push (Quality Firewall)
- **Trigger:** Push to main/develop
- **Tests:** Core + ML (`-m "not slow"`)
- **Coverage:** Full coverage + Codecov upload
- **Timeout:** 45 minutes max
- **Goal:** Definitive quality gate before merge

### Tier 3: Nightly (Comprehensive)
- **Trigger:** Scheduled (nightly) or workflow_dispatch
- **Tests:** All tests including slow (`-m "not benchmark"`)
- **Coverage:** Full coverage + historical trends
- **Timeout:** 120 minutes max
- **Goal:** Detect regressions in expensive/slow tests

### Tier 4: Benchmark (On-demand)
- **Trigger:** Manual workflow_dispatch or special label
- **Tests:** Performance benchmarks (`-m "benchmark"`)
- **Coverage:** No coverage (timing-sensitive)
- **Timeout:** 180 minutes max
- **Goal:** Validate performance budgets before release
```

---

## 6. Implementation Order (Phased Rollout)

**ARCHITECT-APPROVED SEQUENCE:**

### Phase 1: Safety First (NO RISKS)
**Goal:** Fix known anti-patterns before optimization

1. **Remove `AGENT_TOOLSDIRECTORY` deletion** (RISK 2)
   - File: `.github/workflows/ci-quality-firewall.yml` (lines 389, 485)
   - Action: Replace with selective cleanup (see RISK 2 mitigation)
   - Validation: CI passes with Python intact
   - ADR: Update incident report with "preventive removal"

2. **Audit for scikit-learn force-reinstall** (RISK 3)
   - Search: `grep -r "uninstall.*scikit" .github/workflows/`
   - Action: Remove if found, document rationale if required
   - Validation: Dependency resolution logs clean

3. **Kill fallback-to-not-slow logic** (RISK 4)
   - Search: `grep -r "continue-on-error" .github/workflows/ | grep pytest`
   - Action: Remove retry logic, quarantine flaky tests instead
   - Validation: Test results deterministic

**Exit Criteria:** 3 consecutive CI runs pass on main without safety issues

---

### Phase 2: Infrastructure (LOW RISK)
**Goal:** Create dedicated ML CI dependencies

4. **Create `requirements/ml-ci.in`** (ADR-032 alignment)
   - Content: CPU-only torch, reduced ML deps (see Section 1)
   - Compile: `cd requirements && pip-compile ml-ci.in`
   - Validation: `pip install -r requirements/ml-ci.txt` succeeds on Py3.11

5. **Update `pyproject.toml`** (test configuration)
   - Add: `--durations=20` to `addopts`
   - Update: Marker documentation for CI tiers
   - Validation: `pytest --help` shows updated config

6. **Test new ML dependencies locally**
   ```bash
   # Simulate CI environment
   python3.11 -m venv venv-ml-ci
   source venv-ml-ci/bin/activate
   pip install -r requirements/ml-ci.txt
   pytest -m "ml and not slow" --durations=20
   ```

**Exit Criteria:** ML tests pass with new dependencies, durations logged

---

### Phase 3: Conditional Execution (MEDIUM RISK)
**Goal:** Implement path-based ML test gating

7. **Add `run_ml_check` job** (new job in ci-quality-firewall.yml)
   ```yaml
   run_ml_check:
     name: Check if ML tests needed
     outputs:
       run_ml: ${{ steps.check-paths.outputs.run_ml }}
     steps:
       - uses: actions/checkout@v6
         with:
           fetch-depth: 2  # Need parent commit for diff

       - name: Check changed paths
         id: check-paths
         run: |
           # Fail-safe: always run on main/develop
           if [[ "${{ github.ref }}" == "refs/heads/main" ]] || \
              [[ "${{ github.ref }}" == "refs/heads/develop" ]]; then
             echo "run_ml=true" >> $GITHUB_OUTPUT
             echo "Reason: Main branch (Quality Firewall)"
             exit 0
           fi

           # Check for ML-relevant path changes
           ML_PATHS=(
             "src/transformation_portal/depth"
             "src/transformation_portal/upscaling"
             "src/transformation_portal/diffusion"
             "src/transformation_portal/spatial_ai/segmentation"
             "src/transformation_portal/spatial_ai/materials"
             "src/transformation_portal/spatial_ai/reconstruction"
             "src/transformation_portal/spatial_ai/ingest"
             "requirements/ml.in"
             "requirements/ml.txt"
             "requirements/ml-ci.in"
             "requirements/ml-ci.txt"
             "pyproject.toml"
           )

           changed_files=$(git diff --name-only HEAD~1 HEAD)

           for path in "${ML_PATHS[@]}"; do
             if echo "$changed_files" | grep -q "^$path"; then
               echo "run_ml=true" >> $GITHUB_OUTPUT
               echo "Reason: ML path changed: $path"
               exit 0
             fi
           done

           # Check for @pytest.mark.ml in changed test files
           test_changes=$(echo "$changed_files" | grep "^tests/.*\.py$" || true)
           if [ -n "$test_changes" ]; then
             if echo "$test_changes" | xargs grep -l "@pytest.mark.ml" 2>/dev/null; then
               echo "run_ml=true" >> $GITHUB_OUTPUT
               echo "Reason: ML test added/modified"
               exit 0
             fi
           fi

           echo "run_ml=false" >> $GITHUB_OUTPUT
           echo "Reason: No ML-relevant paths changed"
   ```

8. **Update `test-ml` job dependencies**
   ```yaml
   test-ml:
     needs: [preflight, run_ml_check]
     if: needs.run_ml_check.outputs.run_ml == 'true'
   ```

9. **Add validation job** (RISK 1 mitigation)
   ```yaml
   validate-ml-skip:
     name: Validate ML Skip Decision
     needs: run_ml_check
     if: needs.run_ml_check.outputs.run_ml == 'false'
     steps:
       - uses: actions/checkout@v6
         with:
           fetch-depth: 2

       - name: Audit skip decision
         run: |
           echo "ML tests skipped - auditing decision..."
           changed_files=$(git diff --name-only HEAD~1 HEAD)

           # Check for hidden ML imports
           if echo "$changed_files" | xargs grep -l "from.*\(depth\|upscaling\|diffusion\)" 2>/dev/null; then
             echo "WARNING: ML imports detected in changed files"
             echo "Consider adding changed paths to run_ml_check"
             # Non-blocking warning for first 30 days
           fi
   ```

**Exit Criteria:**
- ML tests run on path match (manual trigger test)
- ML tests skip when no ML paths changed (manual trigger test)
- Fail-safe verified: main branch always runs ML

---

### Phase 4: Optimization (LOW RISK)
**Goal:** Implement ML test speedups

10. **Switch ML job to `requirements/ml-ci.txt`**
    ```yaml
    test-ml:
      steps:
        - name: Install dependencies
          run: |
            pip install --upgrade pip wheel setuptools
            # Use CPU-optimized deps
            pip install -r requirements/ml-ci.txt
            pip install -e . --no-deps
    ```

11. **Disable coverage for PR tier**
    ```yaml
    - name: Run ML tests
      run: |
        if [ "${{ github.event_name }}" == "pull_request" ]; then
          pytest -m "ml" --durations=20  # No coverage
        else
          pytest -m "ml" --durations=20 --cov=src/transformation_portal
        fi
    ```

12. **Remove slow tests from PR tier**
    - Audit: `pytest --collect-only -q -m "ml and slow"`
    - Quarantine: Add `@pytest.mark.slow` to tests >60s
    - Update filter: `pytest -m "ml and not slow"`

**Exit Criteria:**
- ML tests complete in <15 minutes on PR tier
- Coverage still collected on main branch

---

### Phase 5: Documentation & Monitoring (REQUIRED)
**Goal:** Ensure long-term maintainability

13. **Create ADR-034: CI Test Execution Tiers**
    - Document 4 tiers (see Section 5)
    - Define SLAs (PR <20min, Main <45min, Nightly <120min)
    - Establish quarterly review cadence

14. **Update CONTRIBUTING.md**
    ````markdown
    ## Running Tests Locally

    ### Quick Smoke Test (PR tier)
    ```bash
    pytest -m "not ml and not slow" --maxfail=3
    ```

    ### Full Suite (Main tier)
    ```bash
    pytest -m "not slow" --maxfail=3
    ```

    ### ML Tests Only
    ```bash
    pip install -r requirements/ml-ci.txt
    pytest -m "ml and not slow" --durations=20
    ```
    ````

15. **Add performance monitoring**
    - Workflow: Log CI job durations to artifact
    - Alert: Notify if `test-ml` exceeds 20 minutes
    - Review: Quarterly audit of `--durations=20` output

**Exit Criteria:**
- ADR-034 merged
- Documentation updated
- Monitoring active

---

## 7. Quality Gates Preservation

**ARCHITECT REQUIREMENT:** All existing quality gates MUST remain enforceable

### Gate 1: Test Isolation (ADR-031)
**Status:** PRESERVED ✅

```yaml
test-isolation:
  # Still runs on every PR
  # Validates ML tests don't leak into core
```

### Gate 2: Dependency Constraints (ADR-032)
**Status:** ENHANCED ✅

```yaml
validate-dependency-constraints:
  # Now also validates requirements/ml-ci.in
  run: |
    bash scripts/validate_dependency_constraints.sh --all
```

### Gate 3: Coverage Threshold
**Status:** PRESERVED ✅

```yaml
coverage-gate:
  needs: [test-core, test-ml]
  # ML coverage collected on main, not PR tier
  # Threshold: 25% minimum (unchanged)
```

### Gate 4: Security Scanning
**Status:** UNCHANGED ✅

```yaml
security:
  # Bandit, gitleaks, safety checks
  # Run on every PR regardless of ML path match
```

### NEW Gate 5: ML Test Duration Budget (PROPOSED)

**RECOMMENDATION:** Add performance regression gate

```yaml
ml-duration-budget:
  name: ML Performance Budget
  needs: test-ml
  if: needs.test-ml.result == 'success'
  steps:
    - name: Check duration budget
      run: |
        # Extract duration from test-results artifact
        ml_duration=$(jq '.duration' test-results/ml-py3.11.json)
        budget_seconds=900  # 15 minutes

        if (( $(echo "$ml_duration > $budget_seconds" | bc -l) )); then
          echo "ERROR: ML tests exceeded budget"
          echo "Duration: ${ml_duration}s, Budget: ${budget_seconds}s"
          echo "Quarantine slow tests or optimize"
          exit 1
        fi
```

---

## 8. Additional Optimizations (Optional)

These are **NOT required** for Phase 1-5 but provide further gains:

### Optimization A: Cache Model Weights Across Runs

**Current:** Line 496-505 caches HuggingFace models

**Enhancement:**
```yaml
- uses: actions/cache@v5
  with:
    path: |
      ~/.cache/huggingface
      ~/.cache/torch
      ~/.cache/transformation_portal/segmentation
    key: models-${{ runner.os }}-${{ hashFiles('requirements/ml-ci.txt') }}-v2
```

**Benefit:** Reduce model download time from 5min to 30s on cache hit

---

### Optimization B: Parallel ML Test Shards

**Current:** Single `test-ml` job runs all ML tests serially

**Enhancement:**
```yaml
test-ml:
  strategy:
    matrix:
      shard: [1, 2, 3]
  steps:
    - run: pytest -m "ml" --shard ${{ matrix.shard }}/3
```

**Benefit:** Reduce ML test time by ~3x (if tests are independent)

**Risk:** Complexity, coverage combining, flake amplification

**ARCHITECT DECISION:** Defer to Phase 6 (after 30-day stability period)

---

### Optimization C: GPU Runner for ML Tests

**Current:** CPU-only torch on ubuntu-latest

**Enhancement:**
```yaml
test-ml:
  runs-on: [self-hosted, gpu]  # Or use GitHub's GPU runners
  steps:
    - run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Benefit:** GPU inference 10-50x faster than CPU

**Risk:** Cost, availability, driver compatibility

**ARCHITECT DECISION:** Requires cost-benefit analysis. Defer to Phase 6.

---

## 9. Monitoring & Success Metrics

**ARCHITECT REQUIREMENT:** Establish baselines before and after

### Pre-Optimization Baseline (Measure Now)
```bash
# Run on current main branch
time pytest -m "ml and not slow" --durations=20 2>&1 | tee baseline.log

# Extract metrics
grep "passed in" baseline.log
grep "slowest" baseline.log
```

**Expected metrics:**
- Total ML test time: ~30 minutes (as stated in proposal)
- Number of ML tests: `pytest --collect-only -q -m "ml" | tail -1`
- Slowest test: From `--durations=20` output

### Post-Optimization Targets
- **PR tier:** <15 minutes (50% reduction)
- **Main tier:** <25 minutes (17% reduction, coverage overhead)
- **Nightly tier:** <45 minutes (full suite)
- **Flake rate:** <2% (measure over 30 days)

### Monitoring Dashboard (Recommended)
```yaml
# Add to performance-monitor.yml workflow
- name: Log ML test metrics
  run: |
    echo "ml_test_duration_seconds $(jq '.duration' test-results.json)" >> metrics.txt
    echo "ml_test_count $(jq '.summary.total' test-results.json)" >> metrics.txt
```

**Integrate with:** Grafana, Datadog, or GitHub Actions dashboard

---

## 10. Rollback Plan

**ARCHITECT REQUIREMENT:** Define rollback criteria

### Rollback Triggers
1. **ML tests fail >10% of runs** (flake rate spike)
2. **Main branch CI blocked >2 hours** (critical path blocked)
3. **False negative on ML skip** (regression shipped without ML tests)
4. **Performance regression >20%** (optimization made things worse)

### Rollback Procedure
```bash
# Emergency: Revert to previous workflow
git revert <optimization-merge-commit>
git push origin main

# Controlled: Disable conditional execution
# .github/workflows/ci-quality-firewall.yml
run_ml_check:
  outputs:
    run_ml: true  # Force always-run
```

### Fallback: Nightly-Only ML
If optimization doesn't meet targets:
```yaml
test-ml:
  # Run only on nightly schedule
  if: github.event_name == 'schedule'
```

**Trade-off:** Faster PRs, but ML regressions detected next day

---

## 11. Final Architect Checklist

**Before implementation, verify:**

- [ ] ADR-034 (CI Test Execution Tiers) drafted and reviewed
- [ ] `AGENT_TOOLSDIRECTORY` deletion removed from all workflows
- [ ] `requirements/ml-ci.in` created with CPU-optimized pins
- [ ] `run_ml_check` fail-safe verified (always TRUE on main)
- [ ] ML path matcher validated against codebase structure
- [ ] Baseline metrics captured (current ML test duration)
- [ ] Rollback plan documented and tested
- [ ] Phase 1 (safety fixes) merged and validated before Phase 2
- [ ] Specialist consulted on ML-specific test quarantine decisions
- [ ] 30-day monitoring plan established (weekly skip audits)

---

## 12. Delegation Boundaries

**Architect retains authority over:**
- CI tier contracts and SLAs (ADR-034)
- Fail-safe behavior (always run on main)
- Quality gate thresholds (coverage, duration budgets)
- Rollback criteria

**Specialist can implement without escalation:**
- Specific test quarantine decisions (`@pytest.mark.slow`)
- ML path matcher refinements (within Section 2 scope)
- Coverage optimization tactics (as long as gate preserved)
- Performance profiling and bottleneck analysis

**Escalate to Architect if:**
- ML skip causes regression shipped to main
- Performance targets not met after Phase 4
- Path matcher has >5% false negative rate
- Need to relax quality gate thresholds

---

## Conclusion

**ARCHITECT DECISION: APPROVED FOR IMPLEMENTATION**

This optimization plan is architecturally sound, aligns with ADR-031 and ADR-032, and follows the repository's governance principles. The phased rollout ensures safety, and the fail-safe mechanisms prevent Quality Firewall degradation.

**Key Success Factors:**
1. Fix anti-patterns (AGENT_TOOLSDIRECTORY deletion) FIRST
2. Implement fail-safe (always run on main) BEFORE optimization
3. Monitor skip decisions for 30 days before declaring success
4. Document CI tiers in ADR-034 for long-term maintainability

**Expected Outcome:**
- PR feedback loop: 30min → 15min (50% faster)
- Main quality gate: Unchanged (all tests run)
- Nightly comprehensive: Unchanged (full coverage)
- Developer experience: Improved (faster iteration)

**Timeline:**
- Phase 1 (safety): 1 day
- Phase 2 (infrastructure): 2 days
- Phase 3 (conditional): 3 days (includes validation period)
- Phase 4 (optimization): 2 days
- Phase 5 (documentation): 1 day

**Total:** 9 working days with proper validation

---

**Approver:** Transformation Portal Architect
**Status:** Ready for Implementation
**Next Step:** Create ADR-034 and begin Phase 1
