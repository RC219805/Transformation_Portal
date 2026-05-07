# ML CI Optimization: Pre-Implementation Checklist

**Epic:** ML CPU Test Optimization (30min → 15min PR feedback)
**Authority:** Approved by Transformation Portal Architect
**Reference:** [ML_CI_OPTIMIZATION_STRATEGIC_REVIEW.md](./ML_CI_OPTIMIZATION_STRATEGIC_REVIEW.md)

---

## Prerequisites (DO NOT SKIP)

- [ ] **Read full strategic review:** `docs/architecture/ML_CI_OPTIMIZATION_STRATEGIC_REVIEW.md`
- [ ] **Read quick reference:** `docs/architecture/ML_CI_OPTIMIZATION_QUICKREF.md`
- [ ] **Capture baseline metrics:**
  ```bash
  time pytest -m "ml and not slow" --durations=20 > baseline_ml_tests.log 2>&1
  # Record: total time, test count, slowest test
  ```
- [ ] **Verify current CI is stable:**
  - [ ] Last 3 main branch CI runs passed
  - [ ] No active incidents or flakes >5%

---

## Phase 1: Safety First (CRITICAL - DO FIRST)

**Goal:** Fix known anti-patterns before optimization
**Risk Level:** HIGH if skipped
**Timeline:** 1 day

### 1.1 Remove AGENT_TOOLSDIRECTORY Deletion

- [ ] **Search for all occurrences:**
  ```bash
  grep -r "AGENT_TOOLSDIRECTORY" .github/workflows/
  ```
- [ ] **Edit `.github/workflows/ci-quality-firewall.yml`:**
  - [ ] Line ~389 (test-core job): Replace with selective cleanup
  - [ ] Line ~485 (test-ml job): Replace with selective cleanup
- [ ] **Replace with safe cleanup:**
  ```yaml
  - name: Free disk space (selective)
    run: |
      sudo rm -rf /usr/share/dotnet /opt/ghc /usr/local/share/boost
      sudo docker image prune --all --force
      # NEVER delete AGENT_TOOLSDIRECTORY
      # See: docs/incidents/2026-02-03_ci_python_toolcache_deletion.md
      df -h
  ```
- [ ] **Add comment referencing incident report**
- [ ] **Test locally (dry-run):**
  ```bash
  # Verify Python survives cleanup
  docker run --rm -v $PWD:/repo ubuntu:latest bash -c "
    rm -rf /usr/share/dotnet /opt/ghc /usr/local/share/boost
    which python3 && echo 'Python OK' || echo 'Python MISSING'
  "
  ```

### 1.2 Audit for Forced Reinstalls

- [ ] **Search for scikit-learn patterns:**
  ```bash
  grep -r "uninstall.*scikit-learn" .github/workflows/
  grep -r "reinstall.*scikit" .github/workflows/
  ```
- [ ] **If found, document rationale or remove:**
  - [ ] Check git history: `git log -p --grep="scikit-learn" -- .github/workflows/`
  - [ ] If no rationale exists, remove the pattern
  - [ ] If rationale exists, add inline comment

### 1.3 Kill Fallback-to-Not-Slow Logic

- [ ] **Search for retry patterns:**
  ```bash
  grep -r "continue-on-error.*pytest" .github/workflows/
  grep -A5 "if: failure()" .github/workflows/ | grep pytest
  ```
- [ ] **If found, remove and quarantine flaky tests:**
  ```python
  # Replace retry logic with proper quarantine
  @pytest.mark.flaky(reruns=3, reruns_delay=5)
  def test_previously_flaky():
      ...
  ```

### 1.4 Validation

- [ ] **Create PR with Phase 1 changes only**
- [ ] **Run CI on PR branch**
- [ ] **Verify:**
  - [ ] Python interpreter survives cleanup steps
  - [ ] No "No such file or directory" errors
  - [ ] Test results are deterministic (run twice, same outcome)
- [ ] **Merge Phase 1 PR**
- [ ] **Verify 3 consecutive green CI runs on main**

---

## Phase 2: Infrastructure

**Goal:** Create dedicated ML CI dependencies
**Risk Level:** LOW (additive changes)
**Timeline:** 2 days

### 2.1 Create requirements/ml-ci.in

- [ ] **Create file:** `requirements/ml-ci.in`
- [ ] **Content:**
  ```ini
  # ML dependencies for CI - CPU-only, pinned for speed/determinism
  # Compile with: pip-compile ml-ci.in --output-file ml-ci.txt

  # CPU-only PyTorch (strict pins for CI reproducibility)
  torch==2.10.0+cpu
  torchvision==0.25.0+cpu

  # ML dependencies with range pins
  -c constraints.txt  # Enforce banned packages (ADR-032)
  transformers>=4.57.0,<6
  diffusers>=0.38.0,<1
  sentence-transformers>=3.1.0,<6  # CVE-73169 minimum (ADR-032)
  controlnet-aux>=0.0.6,<1
  huggingface-hub>=0.19.0,<2

  # Heavy deps EXCLUDED from CI (tested separately):
  # - accelerate (not needed for CPU inference)
  # - coremltools (Apple Silicon only)
  # - rawpy (separate RAW loader tests)
  # - efficientsam/open-clip-torch (integration tier only)
  ```

### 2.2 Compile ml-ci.txt

- [ ] **Run pip-compile:**
  ```bash
  cd requirements
  pip-compile ml-ci.in --output-file ml-ci.txt --resolver=backtracking
  ```
- [ ] **Verify output:**
  - [ ] Check for `torch==2.10.0+cpu` (strict pin present)
  - [ ] Check for banned packages (should be absent)
  - [ ] Verify pip-compile header comment exists

### 2.3 Update pyproject.toml

- [ ] **Add --durations=20 to pytest config:**
  ```toml
  [tool.pytest.ini_options]
  addopts = "--strict-markers --tb=short -p no:warnings --durations=20"
  ```
- [ ] **Update marker documentation:**
  ```toml
  markers = [
      "slow: marks tests as slow (CI: excluded from PR tier)",
      "ml: tests requiring ML models (CI: conditional on path match)",
      ...
  ]
  ```

### 2.4 Local Validation

- [ ] **Create test venv:**
  ```bash
  python3.11 -m venv venv-ml-ci-test
  source venv-ml-ci-test/bin/activate
  pip install -r requirements/ml-ci.txt
  pip install -e . --no-deps
  ```
- [ ] **Run ML tests:**
  ```bash
  pytest -m "ml and not slow" --durations=20 -v
  ```
- [ ] **Verify:**
  - [ ] All ML tests pass
  - [ ] Durations logged (20 slowest tests shown)
  - [ ] No unexpected dependency errors

### 2.5 Validation

- [ ] **Create PR with Phase 2 changes**
- [ ] **Run CI (will still use old requirements)**
- [ ] **Verify:**
  - [ ] pip-compile artifacts are valid
  - [ ] pyproject.toml changes don't break collection
  - [ ] --durations=20 output visible in logs
- [ ] **Merge Phase 2 PR**

---

## Phase 3: Conditional Execution

**Goal:** Implement path-based ML test gating
**Risk Level:** MEDIUM (requires fail-safe validation)
**Timeline:** 3 days

### 3.1 Add run_ml_check Job

- [ ] **Edit `.github/workflows/ci-quality-firewall.yml`**
- [ ] **Insert after preflight job:**
  ```yaml
  run_ml_check:
    name: Check if ML tests needed
    needs: preflight
    if: needs.preflight.outputs.should_run == 'true'
    runs-on: ubuntu-latest
    outputs:
      run_ml: ${{ steps.check-paths.outputs.run_ml }}

    steps:
      - uses: actions/checkout@v6
        with:
          ref: ${{ needs.preflight.outputs.head_sha }}
          fetch-depth: 2  # Need parent for diff

      - name: Detect ML-relevant changes
        id: check-paths
        run: |
          # [Copy script from strategic review Section 6, Phase 3, step 7]
  ```

### 3.2 Update test-ml Job Dependencies

- [ ] **Edit test-ml job:**
  ```yaml
  test-ml:
    needs: [preflight, run_ml_check]
    if: needs.run_ml_check.outputs.run_ml == 'true'
  ```

### 3.3 Add Validation Job

- [ ] **Insert validate-ml-skip job:**
  ```yaml
  validate-ml-skip:
    name: Audit ML Skip Decision (30-day monitoring)
    needs: run_ml_check
    if: needs.run_ml_check.outputs.run_ml == 'false'
    steps:
      - name: Check for hidden ML imports
        run: |
          # [Copy script from strategic review Section 4, RISK 1 mitigation]
  ```

### 3.4 Fail-Safe Testing

- [ ] **Test Case 1: Main branch (MUST run ML)**
  ```bash
  git checkout main
  # Trigger CI via empty commit
  git commit --allow-empty -m "Test: ML always runs on main"
  git push
  ```
  - [ ] Verify: `run_ml=true` in CI logs
  - [ ] Verify: test-ml job executed

- [ ] **Test Case 2: PR with ML path changes (MUST run ML)**
  ```bash
  git checkout -b test-ml-path-match
  echo "# test" >> src/transformation_portal/depth/backends/depth_pro.py
  git commit -am "Test: ML path changed"
  git push -u origin test-ml-path-match
  # Create PR
  ```
  - [ ] Verify: `run_ml=true` in CI logs
  - [ ] Verify: test-ml job executed

- [ ] **Test Case 3: PR with NO ML changes (MUST skip ML)**
  ```bash
  git checkout -b test-ml-skip
  echo "# test" >> README.md
  git commit -am "Test: No ML paths changed"
  git push -u origin test-ml-skip
  # Create PR
  ```
  - [ ] Verify: `run_ml=false` in CI logs
  - [ ] Verify: test-ml job skipped
  - [ ] Verify: validate-ml-skip job ran (audit passed)

### 3.5 Validation

- [ ] **All 3 test cases passed**
- [ ] **Fail-safe verified (main always runs ML)**
- [ ] **Create tracking issue for 30-day monitoring:**
  - [ ] Week 1: Daily audit of skipped ML PRs
  - [ ] Week 2: Daily audit
  - [ ] Week 3-4: Weekly audit
- [ ] **Merge Phase 3 PR**

---

## Phase 4: Optimization

**Goal:** Implement ML test speedups
**Risk Level:** LOW (incremental improvements)
**Timeline:** 2 days

### 4.1 Switch to ml-ci.txt

- [ ] **Edit test-ml job in ci-quality-firewall.yml:**
  ```yaml
  - name: Install dependencies
    run: |
      python -m pip install --upgrade pip wheel setuptools
      # Use CPU-optimized deps (Phase 2 artifact)
      python -m pip install -r requirements/ml-ci.txt
      python -m pip install -e . --no-deps
  ```
- [ ] **Remove old torch install:**
  - [ ] Delete: `pip install "torch==X.Y.Z+cpu" --index-url ...`
  - [ ] Verify ml-ci.txt is used instead

### 4.2 Conditional Coverage

- [ ] **Edit test-ml job run step:**
  ```yaml
  - name: Run ML tests
    run: |
      if [ "${{ github.event_name }}" == "pull_request" ]; then
        # PR tier: no coverage (speed optimization)
        pytest -m "ml and not slow" --durations=20 -v
      else
        # Main tier: with coverage (quality gate)
        pytest -m "ml and not slow" --durations=20 --cov=src/transformation_portal -v
      fi
  ```

### 4.3 Quarantine Slow Tests

- [ ] **Identify slowpokes from --durations=20:**
  ```bash
  # Extract from baseline log
  grep "slowest 20" baseline_ml_tests.log -A 20
  ```
- [ ] **For tests >60s, add @pytest.mark.slow:**
  ```python
  @pytest.mark.ml
  @pytest.mark.slow  # Exclude from PR tier
  def test_expensive_operation():
      ...
  ```
- [ ] **Document quarantine decisions in PR:**
  - Test name
  - Duration (before quarantine)
  - Reason (model loading, large dataset, etc.)

### 4.4 Update Test Filters

- [ ] **Verify all pytest commands use:** `-m "ml and not slow"`
- [ ] **Check these files:**
  - [ ] `.github/workflows/ci-quality-firewall.yml`
  - [ ] `.github/workflows/nightly.yml` (should run slow tests)
  - [ ] `README.md` (update examples)
  - [ ] `CONTRIBUTING.md` (update examples)

### 4.5 Performance Baseline

- [ ] **Run optimized ML tests:**
  ```bash
  time pytest -m "ml and not slow" --durations=20 > optimized_ml_tests.log 2>&1
  ```
- [ ] **Compare baselines:**
  ```bash
  # Baseline (before): ~30 minutes
  # Optimized (after): <15 minutes (target)
  ```
- [ ] **Document improvement percentage**

### 4.6 Validation

- [ ] **Create PR with Phase 4 changes**
- [ ] **Run CI on PR branch**
- [ ] **Verify:**
  - [ ] ML tests complete in <15 minutes
  - [ ] Coverage NOT collected on PR tier
  - [ ] Slow tests excluded from PR tier
  - [ ] All tests still pass
- [ ] **Merge Phase 4 PR**
- [ ] **Monitor main branch CI:**
  - [ ] Coverage still collected on main
  - [ ] Quality gate threshold still enforced

---

## Phase 5: Documentation & Monitoring

**Goal:** Ensure long-term maintainability
**Risk Level:** NONE (documentation only)
**Timeline:** 1 day

### 5.1 Create ADR-034

- [ ] **Create file:** `docs/architecture/ADR-034-ci-test-execution-tiers.md`
- [ ] **Content sections:**
  - [ ] Context (30min ML tests problem)
  - [ ] Decision (4 tier structure: PR/Main/Nightly/Benchmark)
  - [ ] Tier SLAs (PR <20min, Main <45min, Nightly <120min)
  - [ ] Path matcher logic (ML-relevant paths)
  - [ ] Fail-safe rules (always run on main)
  - [ ] Monitoring plan (30-day audit)
  - [ ] Rollback criteria
- [ ] **Reference existing ADRs:**
  - [ ] ADR-031 (Test Dependency Isolation)
  - [ ] ADR-032 (Dependency Pinning Strategy)

### 5.2 Update CONTRIBUTING.md

- [ ] **Add section: "Running Tests Locally"**
- [ ] **Include tier examples:**
  ````markdown
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

### 5.3 Add Performance Monitoring

- [ ] **Create script:** `scripts/ci/log_test_metrics.sh`
  ```bash
  #!/bin/bash
  # Extract test metrics from pytest JSON report
  duration=$(jq '.duration' test-results.json)
  count=$(jq '.summary.total' test-results.json)
  echo "ml_test_duration_seconds $duration" >> metrics.txt
  echo "ml_test_count $count" >> metrics.txt
  ```
- [ ] **Integrate into workflow:**
  ```yaml
  - name: Log metrics
    if: always()
    run: bash scripts/ci/log_test_metrics.sh
  ```

### 5.4 Create Monitoring Dashboard (Optional)

- [ ] **Option A: GitHub Actions dashboard**
  - [ ] Use workflow artifacts to track duration over time
- [ ] **Option B: Grafana/Datadog**
  - [ ] Send metrics to external monitoring
- [ ] **Option C: CSV export**
  - [ ] Store metrics in repo artifact for manual analysis

### 5.5 Schedule 30-Day Reviews

- [ ] **Create calendar reminders:**
  - [ ] Week 1: Daily audit (Mon-Fri)
  - [ ] Week 2: Daily audit (Mon-Fri)
  - [ ] Week 3: Monday audit
  - [ ] Week 4: Friday audit + final report
- [ ] **Create GitHub issue template:** "ML Skip Audit - Week X"
  ```markdown
  ## ML Skip Audit - Week X

  **Date:** YYYY-MM-DD
  **Auditor:** @username

  ### PRs with run_ml=false
  - [ ] PR #XXX - Reviewed (reason: ___)
  - [ ] PR #YYY - Reviewed (reason: ___)

  ### False Negatives Found
  - None / [List with impact assessment]

  ### Path Matcher Refinements
  - None / [Proposed changes]
  ```

### 5.6 Final Checklist

- [ ] **ADR-034 reviewed and merged**
- [ ] **CONTRIBUTING.md updated**
- [ ] **Monitoring script deployed**
- [ ] **30-day audit schedule established**
- [ ] **Rollback plan documented and tested**

---

## Post-Implementation Validation (7 days)

### Daily Checks (Days 1-7)

- [ ] **Day 1:**
  - [ ] Monitor CI dashboard for failures
  - [ ] Check ML skip decisions (manual spot-check)
  - [ ] Verify PR feedback time <20min
- [ ] **Day 2:**
  - [ ] Repeat Day 1 checks
  - [ ] Review first audit report
- [ ] **Day 3-7:**
  - [ ] Daily dashboard check
  - [ ] No critical rollback triggers

### Success Criteria

- [ ] **PR feedback loop:** <15 minutes (50% improvement)
- [ ] **Main quality gate:** Unchanged (all tests run)
- [ ] **False negative rate:** <5% (path matcher accuracy)
- [ ] **Flake rate:** <2% (no regression)
- [ ] **Zero rollbacks:** No critical incidents

---

## Rollback Procedures

### Emergency Rollback (Critical Incident)

```bash
# If main CI broken >2 hours
git revert <optimization-merge-sha>
git push origin main
```

### Controlled Rollback (Performance Regression)

```yaml
# Edit ci-quality-firewall.yml
run_ml_check:
  outputs:
    run_ml: true  # Force always-run (bypass path matcher)
```

### Partial Rollback (Phase-specific)

- **Phase 4 regression:** Revert ml-ci.txt switch, keep conditional execution
- **Phase 3 false negatives:** Set fail-safe to always TRUE temporarily

---

## Sign-Off

- [ ] **Architect review:** Phases 1-5 implementation complete
- [ ] **Specialist review:** ML test coverage preserved
- [ ] **30-day monitoring:** Audit complete, no false negatives
- [ ] **Performance targets met:** PR <15min, Main <25min
- [ ] **Documentation complete:** ADR-034 merged, CONTRIBUTING.md updated

**Final Approval:**
- Architect: _________________ Date: _________
- Specialist: _________________ Date: _________

---

**Status:** Ready for Implementation
**Next Step:** Begin Phase 1 - Remove AGENT_TOOLSDIRECTORY deletion
