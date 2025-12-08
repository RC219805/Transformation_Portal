# CI/CD Integration Specification

**Version**: 1.0  
**Date**: 2025-12-08  
**Related**: Architecture Hardening Plan

---

## Overview

This document specifies CI/CD integration requirements for the 6-priority architecture optimization plan. Each PR in the sequence has specific CI gates and validation requirements.

---

## PR-1: Security + Repo Hygiene

### New CI Jobs

#### 1. Security Gate (security-gate job)

**File**: `.github/workflows/security-scan.yml`

**Job Definition**:
```yaml
security-gate:
  name: Security Gate (CVE-2024-27763 Enforcement)
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    
    - name: Setup Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.10'
    
    - name: Install Dependencies
      run: pip install -r requirements-ci.txt
    
    - name: Enforce Safe Dependency Policy
      run: python scripts/ci/enforce_safe_deps.py
      
    - name: Check for Banned Imports
      run: |
        # Fail if basicsr/realesrgan/gfpgan imported anywhere
        if grep -r "import basicsr\|import realesrgan\|import gfpgan" \
           lux_depth_v2/ src/ --include="*.py"; then
          echo "❌ Banned package imports detected"
          exit 1
        fi
        echo "✅ No banned imports found"
```

#### 2. Secret Scanning (secret-scan job)

**Job Definition**:
```yaml
secret-scan:
  name: Secret Scanning
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0  # Full history for comprehensive scan
    
    - name: TruffleHog Secret Scan
      uses: trufflesecurity/trufflehog@v3
      with:
        path: ./
        base: ${{ github.event.repository.default_branch }}
        head: HEAD
        extra_args: --only-verified
    
    - name: Gitleaks Scan
      uses: gitleaks/gitleaks-action@v2
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

#### 3. Dependency Vulnerability Scan (vuln-scan job)

**Job Definition**:
```yaml
vuln-scan:
  name: Dependency Vulnerability Scan
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    
    - name: Setup Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.10'
    
    - name: Install Security Tools
      run: |
        pip install safety bandit pip-audit
    
    - name: Safety Check (Known Vulnerabilities)
      run: safety check --file requirements.txt --json --output safety-report.json
      continue-on-error: true
    
    - name: Bandit Scan (Code Security)
      run: bandit -r lux_depth_v2/ -ll -f json -o bandit-report.json
      continue-on-error: true
    
    - name: Pip Audit (Dependency Audit)
      run: pip-audit --require-hashes --desc --format json > pip-audit-report.json
      continue-on-error: true
    
    - name: Upload Security Reports
      uses: actions/upload-artifact@v4
      with:
        name: security-reports
        path: |
          safety-report.json
          bandit-report.json
          pip-audit-report.json
```

### Required Checks (Branch Protection)

**Branch**: `main`

**Required status checks**:
- ✅ `security-gate` (blocking)
- ✅ `secret-scan` (blocking)
- ⚠️ `vuln-scan` (non-blocking, informational)

**Settings**:
```yaml
# .github/settings.yml (via Probot Settings app)
branches:
  - name: main
    protection:
      required_status_checks:
        strict: true
        contexts:
          - security-gate
          - secret-scan
          - test (Python 3.10)
          - test (Python 3.11)
          - test (Python 3.12)
      required_pull_request_reviews:
        required_approving_review_count: 1
      enforce_admins: false
```

### Success Criteria

- ✅ Security gate fails if banned packages detected
- ✅ Secret scan detects test secrets (validation)
- ✅ CI runtime <2 minutes for security checks
- ✅ Zero false positives on clean code

---

## PR-2: Platform Core Extraction

### New CI Jobs

#### 1. Core Module Tests (test-core job)

**Job Definition**:
```yaml
test-core:
  name: Test Platform Core
  runs-on: ubuntu-latest
  strategy:
    matrix:
      python-version: ['3.10', '3.11', '3.12']
  steps:
    - uses: actions/checkout@v4
    
    - name: Setup Python
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install Dependencies
      run: |
        pip install -r requirements-dev.txt
        pip install -e .
    
    - name: Run Core Module Tests
      run: |
        pytest transformation_portal/core/ \
          --cov=transformation_portal.core \
          --cov-report=term \
          --cov-report=xml \
          --cov-fail-under=90 \
          -v
    
    - name: Upload Coverage
      uses: codecov/codecov-action@v4
      with:
        files: ./coverage.xml
        flags: core
```

#### 2. Integration Tests (test-integration job)

**Job Definition**:
```yaml
test-integration:
  name: Integration Tests (Core + Pipelines)
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    
    - name: Setup Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.10'
    
    - name: Install Dependencies
      run: pip install -r requirements.txt -e .
    
    - name: Run Integration Tests
      run: pytest tests/integration/ -v --tb=short
    
    - name: Verify Lux Depth V2 Still Works
      run: |
        # Quick smoke test
        python -c "from lux_depth_v2.pipeline import LuxDepthPipeline; print('✅ Import OK')"
```

### Coverage Requirements

**Platform Core**:
- Minimum: 90% line coverage
- Target: 95%+ line coverage
- Branch coverage: 85%+

**Integration Tests**:
- Verify core modules work with existing pipelines
- No regressions in Lux Depth V2 (66/66 tests must pass)

### Success Criteria

- ✅ Core module tests pass on Python 3.10, 3.11, 3.12
- ✅ 90%+ coverage on core modules
- ✅ Lux Depth V2 tests pass with core integration
- ✅ Zero performance regressions

---

## PR-3: Stage Graph Refactor

### New CI Jobs

#### 1. Stage Graph Tests (test-stage-graph job)

**Job Definition**:
```yaml
test-stage-graph:
  name: Test Stage Graph Infrastructure
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    
    - name: Setup Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.10'
    
    - name: Install Dependencies
      run: pip install -r requirements.txt -e .
    
    - name: Test Stage Infrastructure
      run: pytest transformation_portal/core/pipeline/ -v --cov
    
    - name: Test Cache Correctness
      run: pytest tests/test_caching.py -v
    
    - name: Test Policy Engine
      run: pytest tests/test_policy_engine.py -v
```

#### 2. Performance Benchmarks (benchmark job)

**Job Definition**:
```yaml
benchmark:
  name: Performance Benchmarks
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    
    - name: Setup Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.10'
    
    - name: Install Dependencies
      run: pip install -r requirements.txt -e .
    
    - name: Run Benchmarks
      run: |
        pytest tests/performance/ \
          --benchmark-only \
          --benchmark-json=benchmark-results.json
    
    - name: Compare Against Baseline
      run: |
        python scripts/ci/compare_benchmarks.py \
          --current benchmark-results.json \
          --baseline baseline-benchmarks.json \
          --fail-on-regression 5%
    
    - name: Upload Benchmark Results
      uses: actions/upload-artifact@v4
      with:
        name: benchmark-results
        path: benchmark-results.json
```

### Performance Baselines

**Lux Depth V2** (baseline: current monolithic implementation):
- Single image: 200ms (CPU, GitHub Actions)
- Cache miss overhead: <5%
- Cache hit speedup: >10x

**Regression Threshold**: 5% slower than baseline

### Success Criteria

- ✅ Stage graph tests pass
- ✅ Cache correctness validated
- ✅ Performance regression <5%
- ✅ Cache hit provides >10x speedup

---

## PR-4: Performance + Profiling Hooks

### New CI Jobs

#### 1. Performance Regression Tests (perf-regression job)

**Job Definition**:
```yaml
perf-regression:
  name: Performance Regression Tests
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    
    - name: Setup Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.10'
    
    - name: Install Dependencies
      run: pip install -r requirements.txt -e .
    
    - name: Run Performance Tests
      run: pytest tests/performance/ -v --tb=short
    
    - name: Verify Profiler Overhead
      run: |
        pytest tests/test_profiler_overhead.py -v
        # Assert <5% overhead when profiling enabled
```

#### 2. GPU Tests (Optional, Self-Hosted Runner)

**Job Definition**:
```yaml
gpu-tests:
  name: GPU Performance Tests
  runs-on: self-hosted  # Requires GPU runner
  if: ${{ github.event_name == 'push' && github.ref == 'refs/heads/main' }}
  steps:
    - uses: actions/checkout@v4
    
    - name: Setup Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.10'
    
    - name: Install Dependencies
      run: pip install -r requirements.txt -e .
    
    - name: Run GPU Tests
      run: pytest tests/gpu/ -v --tb=short
    
    - name: Benchmark GPU Throughput
      run: |
        python benchmarks/gpu_throughput.py \
          --images 100 \
          --output gpu-benchmark.json
```

### Success Criteria

- ✅ Performance tests pass on CPU
- ✅ Profiler overhead <5%
- ✅ GPU tests pass (if runner available)
- ✅ No memory leaks detected

---

## PR-5: Validation-First Defaults

### New CI Jobs

#### 1. Validation Report Tests (test-validation job)

**Job Definition**:
```yaml
test-validation:
  name: Test Validation Infrastructure
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    
    - name: Setup Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.10'
    
    - name: Install Dependencies
      run: pip install -r requirements.txt -e .
    
    - name: Test Report Generation
      run: pytest transformation_portal/core/validation/ -v
    
    - name: Test Metrics Computation
      run: pytest tests/test_metrics.py -v
    
    - name: Test Baseline Comparison
      run: pytest tests/test_baseline_comparison.py -v
```

#### 2. Report Artifact Collection (collect-reports job)

**Job Definition**:
```yaml
collect-reports:
  name: Collect Validation Reports
  runs-on: ubuntu-latest
  needs: test
  steps:
    - uses: actions/checkout@v4
    
    - name: Setup Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.10'
    
    - name: Install Dependencies
      run: pip install -r requirements.txt -e .
    
    - name: Process Test Images
      run: |
        python scripts/ci/process_validation_suite.py \
          --input tests/fixtures/ \
          --output reports/
    
    - name: Upload Reports
      uses: actions/upload-artifact@v4
      with:
        name: validation-reports
        path: reports/
    
    - name: Generate Summary
      run: |
        python scripts/ci/summarize_reports.py \
          --reports reports/ \
          --output $GITHUB_STEP_SUMMARY
```

### Success Criteria

- ✅ Report generation tests pass
- ✅ Metrics computation accurate
- ✅ Baseline comparison works
- ✅ Reports collected in CI

---

## PR-6: Test Strategy - Fill Coverage Gaps

### New CI Jobs

#### 1. Edge Case Tests (test-edge-cases job)

**Job Definition**:
```yaml
test-edge-cases:
  name: Edge Case & Fallback Tests
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    
    - name: Setup Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.10'
    
    - name: Install Dependencies
      run: pip install -r requirements.txt -e .
    
    - name: Test Fallback Paths
      run: pytest tests/test_fallbacks.py -v
    
    - name: Test Edge Cases
      run: pytest tests/test_edge_cases.py -v
    
    - name: Test Checkpoint/Resume
      run: pytest tests/test_batch_checkpoint.py -v
```

#### 2. Coverage Report (coverage job)

**Job Definition**:
```yaml
coverage:
  name: Code Coverage Report
  runs-on: ubuntu-latest
  needs: [test, test-core, test-stage-graph, test-edge-cases]
  steps:
    - uses: actions/checkout@v4
    
    - name: Setup Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.10'
    
    - name: Install Dependencies
      run: pip install -r requirements-dev.txt -e .
    
    - name: Generate Coverage Report
      run: |
        pytest --cov=transformation_portal \
               --cov=lux_depth_v2 \
               --cov-report=term \
               --cov-report=html \
               --cov-report=xml \
               tests/
    
    - name: Upload to Codecov
      uses: codecov/codecov-action@v4
      with:
        files: ./coverage.xml
    
    - name: Check Coverage Threshold
      run: |
        coverage report --fail-under=85
```

### Coverage Targets

**Overall**: 85%+ line coverage  
**Core Modules**: 90%+ line coverage  
**Lux Depth V2**: 80%+ line coverage  
**Fallback Branches**: 100% (critical paths)

### Success Criteria

- ✅ Edge case tests pass
- ✅ 85%+ overall coverage
- ✅ Fallback branches tested
- ✅ Checkpoint/resume works

---

## CI/CD Pipeline Summary

### Complete Job Dependency Graph

```
┌─────────────┐
│   PR Open   │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────┐
│  Parallel: Security + Linting + Tests  │
│  ┌──────────────┐  ┌───────────────┐   │
│  │ security-gate│  │  lint (flake8)│   │
│  └──────────────┘  └───────────────┘   │
│  ┌──────────────┐  ┌───────────────┐   │
│  │ secret-scan  │  │test (3.10-3.12)  │
│  └──────────────┘  └───────────────┘   │
│  ┌──────────────┐  ┌───────────────┐   │
│  │  vuln-scan   │  │   test-core   │   │
│  └──────────────┘  └───────────────┘   │
└─────────────┬───────────────────────────┘
              │
              ▼
       ┌──────────────┐
       │ All Pass?    │
       └──┬───────┬───┘
          │       │
      Yes │       │ No
          │       └──▶ ❌ Block Merge
          │
          ▼
    ┌─────────────┐
    │  benchmark  │ (optional)
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │   Ready to  │
    │    Merge    │
    └─────────────┘
```

### Total CI Runtime

**Fast Path** (no benchmarks): ~5-7 minutes  
**Full Path** (with benchmarks): ~10-12 minutes

**Optimization Strategies**:
- Parallel job execution
- Cached dependencies (`actions/cache`)
- Matrix strategy for Python versions
- Conditional jobs (e.g., GPU tests only on main)

---

## Badge Configuration

Add CI badges to README.md:

```markdown
[![Security Gate](https://github.com/RC219805/Transformation_Portal/workflows/Security%20Gate/badge.svg)](https://github.com/RC219805/Transformation_Portal/actions/workflows/security-scan.yml)
[![Tests](https://github.com/RC219805/Transformation_Portal/workflows/CI/badge.svg)](https://github.com/RC219805/Transformation_Portal/actions/workflows/ci-consolidated.yml)
[![Coverage](https://codecov.io/gh/RC219805/Transformation_Portal/branch/main/graph/badge.svg)](https://codecov.io/gh/RC219805/Transformation_Portal)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
```

---

## Rollback Strategy

### CI Failure Handling

**If security-gate fails**:
1. Check `scripts/ci/enforce_safe_deps.py` output
2. Verify `requirements.txt` doesn't include banned packages
3. Check for accidental imports in code

**If tests fail**:
1. Run locally: `pytest -v`
2. Check for environment-specific issues
3. Verify Python version compatibility

**If benchmarks regress**:
1. Review performance changes in PR
2. Check if regression is intentional (more features)
3. Update baseline if justified

### Emergency Bypass

**For critical hotfixes**:
```yaml
# Allow bypass of non-critical checks
if: ${{ github.event.label.name == 'hotfix' }}
```

---

## Maintenance

### Regular Tasks

**Weekly**:
- Review security scan reports
- Update vulnerability baselines

**Monthly**:
- Update Python versions in matrix
- Update action versions (`@v4` → `@v5`)
- Review benchmark baselines

**Quarterly**:
- Audit CI job efficiency
- Optimize slow jobs
- Update documentation

---

**Version**: 1.0  
**Last Updated**: 2025-12-08  
**Next Review**: 2025-12-22
