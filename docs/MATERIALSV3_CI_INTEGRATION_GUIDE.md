# MaterialsV3 CI/CD Integration Guide

**Date**: December 21, 2025
**Type**: Implementation Guide
**Priority**: CRITICAL 🔴
**Estimated Time**: 60-90 minutes

---

## Overview

This guide provides step-by-step instructions for integrating MaterialsV3 tests into the CI/CD pipeline. This is a **critical pre-requisite** for Phase 2 execution.

---

## Option 1: Quick Integration (Recommended)

### Step 1: Add MaterialsV3 Edge Case Tests to Core Tests

**File**: `.github/workflows/ci-consolidated.yml`
**Location**: After line 361 (in `test-core` job)

Add this step after "Run Core Tests":

```yaml
      - name: Run MaterialsV3 Edge Case Tests
        if: always()
        run: |
          pytest tests/test_materials_v3_edge_cases.py \
            -v \
            --tb=short \
            --timeout=300 \
            --maxfail=5
        timeout-minutes: 5

      - name: Verify Phase 1 Safety
        if: always()
        run: |
          python verify_phase1_safety.py --quick
        timeout-minutes: 5
```

**Explanation**:
- Runs edge case tests in every CI build
- 5-minute timeout (tests run in ~60 seconds)
- Uses `--maxfail=5` to stop after 5 failures
- `if: always()` ensures tests run even if core tests fail
- `verify_phase1_safety.py --quick` validates exception handling

### Step 2: Create Nightly Stress Test Workflow

**File**: `.github/workflows/materials-v3-stress-nightly.yml` (new file)

```yaml
name: MaterialsV3 Stress Tests (Nightly)

on:
  schedule:
    # Run at 2 AM UTC every day
    - cron: '0 2 * * *'
  workflow_dispatch:
    inputs:
      quick_mode:
        description: 'Run quick stress tests (reduced iterations)'
        required: false
        default: 'false'
        type: boolean

permissions:
  contents: read
  issues: write  # For failure notifications

jobs:
  stress-test:
    name: MaterialsV3 Stress Test Suite
    runs-on: ubuntu-24.04
    timeout-minutes: 90

    steps:
      - name: Maximize Disk Space
        run: |
          sudo rm -rf /usr/share/dotnet /opt/ghc /usr/local/lib/android
          sudo apt-get clean
          df -h

      - name: Checkout
        uses: actions/checkout@v6
        with:
          fetch-depth: 1

      - name: Setup Python
        uses: actions/setup-python@v6
        with:
          python-version: '3.11'

      - name: Install Dependencies
        run: |
          pip install --upgrade pip wheel
          pip install pytest pytest-timeout psutil
          pip install -c requirements/constraints.txt -r requirements.txt
          pip install -c requirements/constraints.txt -e .

      - name: Run Stress Tests
        run: |
          pytest tests/test_materials_v3_stress.py \
            -v \
            --tb=short \
            -m slow \
            --timeout=3600 \
            --maxfail=3
        timeout-minutes: 75

      - name: Generate Stress Test Report
        if: always()
        run: |
          python verify_phase1_safety.py --full > stress_test_report.txt

      - name: Upload Stress Test Results
        if: always()
        uses: actions/upload-artifact@v6
        with:
          name: stress-test-results-${{ github.run_id }}
          path: |
            stress_test_report.txt
            pytest-*.xml
          retention-days: 30

      - name: Notify on Failure
        if: failure()
        uses: actions/github-script@v7
        with:
          script: |
            github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: '🔴 MaterialsV3 Stress Test Failure - ${{ github.run_id }}',
              body: `
                ## Stress Test Failure Alert

                **Workflow**: MaterialsV3 Stress Tests (Nightly)
                **Run**: ${{ github.run_id }}
                **Date**: ${{ github.event.head_commit.timestamp }}

                The MaterialsV3 stress test suite has failed. This may indicate:
                - Memory leaks
                - Performance regressions
                - Stability issues under load

                **Action Required**: Review stress test logs and investigate failures.

                **Logs**: https://github.com/${{ github.repository }}/actions/runs/${{ github.run_id }}
              `,
              labels: ['bug', 'materials-v3', 'stress-test-failure', 'critical']
            });
```

**Explanation**:
- Runs nightly at 2 AM UTC
- Can also be triggered manually
- Creates GitHub issue on failure
- Uploads test results as artifacts
- 90-minute timeout for full stress suite

---

## Option 2: Comprehensive Integration (For Phase 2+)

### Step 1: Add Test Categories to Setup Job

**File**: `.github/workflows/ci-consolidated.yml`
**Location**: In `setup` job outputs (around line 86)

Add to outputs:

```yaml
      run-materials-v3-tests: ${{ steps.analyze.outputs.run_materials_v3 }}
```

Add to `analyze` step (after line 120):

```yaml
      - name: Analyze Test Requirements
        id: analyze
        run: |
          # ... existing code ...

          # MaterialsV3 tests if MaterialsV3 code or tests changed
          if echo "$FILES" | grep -E "(lux_depth_v2/materials_v3|test_materials_v3)" > /dev/null; then
            echo "run_materials_v3=true" >> $GITHUB_OUTPUT
          else
            echo "run_materials_v3=false" >> $GITHUB_OUTPUT
          fi
```

### Step 2: Create Dedicated MaterialsV3 Test Job

**File**: `.github/workflows/ci-consolidated.yml`
**Location**: After `test-core` job (around line 370)

```yaml
  # ===========================================================================
  # Stage 3.6: MaterialsV3 Edge Case Tests
  # ===========================================================================
  test-materials-v3:
    name: MaterialsV3 Edge Case Tests
    needs: [setup, lint]
    if: |
      always() &&
      needs.setup.outputs.run-materials-v3-tests == 'true' &&
      (needs.lint.result == 'success' || needs.lint.result == 'skipped')
    runs-on: ubuntu-24.04

    steps:
      - name: Checkout
        uses: actions/checkout@v6
        with:
          fetch-depth: 0

      - name: Setup Python
        uses: actions/setup-python@v6
        with:
          python-version: '3.11'

      - name: Install Dependencies
        run: |
          pip install --upgrade pip wheel
          pip install pytest pytest-timeout pytest-cov
          pip install -c requirements/constraints.txt -r requirements.txt
          pip install -c requirements/constraints.txt -e .

      - name: Run MaterialsV3 Edge Case Tests
        run: |
          pytest tests/test_materials_v3_edge_cases.py \
            -v \
            --tb=short \
            --timeout=300 \
            --cov=lux_depth_v2.materials_v3 \
            --cov-report=xml \
            --cov-report=term-missing
        timeout-minutes: 10

      - name: Verify Phase 1 Exception Handling
        run: |
          python verify_phase1_safety.py --quick
        timeout-minutes: 5

      - name: Upload Coverage
        uses: actions/upload-artifact@v6
        with:
          name: coverage-materials-v3
          path: coverage.xml
          retention-days: 7
```

---

## Implementation Checklist

### Pre-Implementation
- [ ] Review current CI/CD workflows
- [ ] Identify test execution time constraints
- [ ] Choose integration approach (Quick vs Comprehensive)

### Implementation
- [ ] Create feature branch: `git checkout -b feat/materials-v3-ci-integration`
- [ ] Edit `.github/workflows/ci-consolidated.yml` (Option 1 or 2)
- [ ] Create `.github/workflows/materials-v3-stress-nightly.yml` (new file)
- [ ] Test changes locally: `act -j test-core` (if using `act`)
- [ ] Commit changes: `git add .github/workflows/`
- [ ] Push and create PR

### Testing
- [ ] Verify edge case tests run in CI
- [ ] Check test execution time (<5 minutes)
- [ ] Verify failure handling (intentionally break a test)
- [ ] Test nightly workflow manually (workflow_dispatch)
- [ ] Verify stress test artifacts upload

### Post-Implementation
- [ ] Merge CI integration PR
- [ ] Monitor first nightly stress test run
- [ ] Update Phase 2 checklist: "Tests integrated into CI" ✅
- [ ] Proceed with Phase 2 execution

---

## Validation Commands

### Local Testing (Before Pushing)

```bash
# Test edge case tests locally
pytest tests/test_materials_v3_edge_cases.py -v --tb=short

# Test verification script
python verify_phase1_safety.py --quick

# Estimate CI execution time
time pytest tests/test_materials_v3_edge_cases.py -v
```

### CI Testing (After Push)

1. Create PR with CI changes
2. Check "Actions" tab for workflow run
3. Verify "MaterialsV3 Edge Case Tests" step passes
4. Check execution time (<5 minutes)
5. Verify failure handling (optional: create test failure)

---

## Failure Handling

### Edge Case Test Failure in CI

**Action**: **BLOCK MERGE** 🔴

**Triage**:
1. Check which test failed
2. Review test logs
3. Reproduce locally
4. Fix issue or skip test with documentation
5. Re-run CI

### Stress Test Failure (Nightly)

**Action**: **WARNING** 🟡

**Triage**:
1. Receive GitHub issue notification
2. Review stress test logs
3. Check for memory leaks, performance regressions
4. Schedule fix in next sprint
5. Don't block merges (unless critical)

### Skipped Tests

**Action**: **LOG ONLY** 📝

**Triage**:
1. Count skipped tests
2. Investigate if count increases
3. Document acceptable skips
4. Target <2 skipped tests

---

## Success Criteria

- [ ] Edge case tests run on every PR
- [ ] Test execution time <5 minutes
- [ ] Failures block merge
- [ ] Stress tests run nightly
- [ ] Artifacts uploaded on failure
- [ ] GitHub issues created on stress test failure
- [ ] Zero flaky tests

**Timeline**: 60-90 minutes
**Owner**: Architect / DevOps

---

## Rollback Plan

If CI integration causes issues:

```bash
# Revert CI changes
git revert <commit-sha>
git push

# Or: Remove test steps manually
# Edit .github/workflows/ci-consolidated.yml
# Remove "Run MaterialsV3 Edge Case Tests" step
```

**Rollback Time**: <10 minutes

---

## Next Steps After Integration

1. ✅ Update Phase 2 checklist: "Tests in CI/CD" → COMPLETE
2. ✅ Commit Phase 1 artifacts to version control
3. ✅ Proceed with Phase 2 execution (E2E validation)
4. ✅ Monitor first nightly stress test results

**Document Version**: 1.0
**Last Updated**: December 21, 2025
