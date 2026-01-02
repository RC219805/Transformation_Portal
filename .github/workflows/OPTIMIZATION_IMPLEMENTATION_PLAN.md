# Workflow Optimization Implementation Plan
**Quick Reference Guide for Architects and DevOps**

## 📋 Executive Summary

**Current State:** 24 workflows, 5,024 lines, 300-500 Actions minutes/PR
**Target State:** 16 workflows, 3,000 lines, 80-120 Actions minutes/PR
**Expected Savings:** 60-70% reduction in CI runtime, 70-80% reduction in costs

---

## 🎯 Phase 1: Quick Wins (Week 1)

**Effort:** 1-2 hours total
**Expected Savings:** 17-29 minutes per PR

### Checklist

- [ ] **QW-1: Delete `depth_quality.yml`**
  ```bash
  git rm .github/workflows/depth_quality.yml
  git commit -m "Remove redundant depth_quality workflow (covered by ci-consolidated)"
  ```
  - **Reason:** Tests are already run in `ci-consolidated.yml`
  - **Savings:** 3-5 min/PR
  - **Risk:** None

- [ ] **QW-2: Disable MaterialsV3 PR trigger**
  ```yaml
  # .github/workflows/materialsv3_tests.yml
  # REMOVE these lines:
  # push:
  #   branches: [ main, develop ]
  # pull_request:
  #   branches: [ main, develop ]

  # KEEP only:
  on:
    schedule:
      - cron: '0 2 * * *'  # Nightly only
    workflow_dispatch:
  ```
  - **Reason:** Tests already run in `ci-consolidated.yml` on every PR
  - **Savings:** 8-15 min/PR
  - **Risk:** Low (still runs nightly)

- [ ] **QW-3: Add PyTorch caching**
  ```yaml
  # .github/workflows/ci-consolidated.yml
  # Line 680: Add BEFORE "Install ML Dependencies"
  - name: Cache PyTorch
    uses: actions/cache@v5
    with:
      path: ~/.cache/torch
      key: torch-cpu-${{ runner.os }}-${{ hashFiles('requirements/ml.txt') }}

  # Line 694: REMOVE --no-cache-dir
  # Before:
  pip install --no-cache-dir torch torchvision --index-url ...
  # After:
  pip install torch torchvision --index-url ...
  ```
  - **Savings:** 3-4 min/ML job
  - **Risk:** None

- [ ] **QW-4: Consolidate basicsr checks**
  ```bash
  # Keep checks in:
  # 1. ci-consolidated.yml (lint stage, line 265)
  # 2. ci-consolidated.yml (test-ml stage, line 730)

  # Remove from:
  # 3. security-scan.yml (lines 63-77) - REMOVE
  # 4. security-gates.yml - REMOVE entire duplicate check
  # 5. materialsv3_tests.yml - REMOVE
  # 6. Others - REMOVE
  ```
  - **Reason:** 8 checks is excessive, 2 is sufficient (one early, one in ML tests)
  - **Savings:** 1-2 min/PR (faster execution)
  - **Risk:** None (still checking twice)

- [ ] **QW-5: Parallelize lint and test-core**
  ```yaml
  # .github/workflows/ci-consolidated.yml
  # Line 292: REMOVE lint from dependencies
  # Before:
  test-core:
    needs: [setup, lint]

  # After:
  test-core:
    needs: [setup]  # Runs in parallel with lint
  ```
  - **Savings:** 2-3 min/PR (parallel execution)
  - **Risk:** None

---

## 🚀 Phase 2: Medium Effort (Week 2-3)

**Effort:** 18-25 hours total
**Expected Savings:** 35-51 minutes per PR

### High-Priority Tasks

- [ ] **ME-1: Create Reusable Setup Action (4 hours)**
  - Create `.github/actions/setup-python-deps/action.yml`
  - Standardize dependency installation across all workflows
  - Expected savings: 15-20 min/PR

- [ ] **ME-2: Merge Security Workflows (6 hours)**
  - Merge `security-scan.yml` + `security-gates.yml` → `security-comprehensive.yml`
  - Consolidate duplicate checks
  - Expected savings: 5-8 min/PR

- [ ] **ME-3: Implement Change Detection (8 hours)**
  - Create `.github/actions/detect-changes/action.yml`
  - Apply to 10+ workflows
  - Skip workflows on docs-only changes
  - Expected savings: 10-15 min/PR

- [ ] **ME-4: Add HuggingFace Model Caching (3 hours)**
  - Cache `~/.cache/huggingface` in ML jobs
  - Expected savings: 2-3 min/ML job

- [ ] **ME-5: Split Fast/Slow Tests (4 hours)**
  - Split `test-core` into `test-core-fast` and `test-core-slow`
  - Mark tests with pytest markers
  - Expected savings: 3-5 min faster feedback

---

## 🏗️ Phase 3: Strategic Improvements (Month 2)

**Effort:** 42-56 hours total
**Impact:** Qualitative (prevent production issues)

- [ ] **SI-1: Docker Build Validation (12 hours)**
  - Create `.github/workflows/docker-build.yml`
  - Test Docker builds on every change to Dockerfile

- [ ] **SI-2: Lux Depth V2 Service Tests (8 hours)**
  - Add FastAPI endpoint tests
  - Test `/health` and `/process` endpoints

- [ ] **SI-3: Nightly Stress Tests (16 hours)**
  - Create `.github/workflows/stress-tests.yml`
  - Memory leak detection (1000 image processing)

- [ ] **SI-4: Performance Dashboard (20 hours)**
  - Track workflow runtimes over time
  - Alert on performance regressions

---

## 📊 Success Criteria

### Quantitative Targets

| Metric | Current | Phase 1 Target | Phase 2 Target | Phase 3 Target |
|--------|---------|----------------|----------------|----------------|
| **PR CI Runtime** | 25-30 min | 15-20 min ✅ | 10-15 min ✅ | 8-12 min ✅ |
| **Actions Minutes/PR** | 300-400 | 180-240 ✅ | 120-180 ✅ | 80-120 ✅ |
| **First Test Feedback** | 8-12 min | 5-8 min ✅ | 2-4 min ✅ | 2-3 min ✅ |
| **Cache Hit Rate** | 20% | 40% ✅ | 60% ✅ | 80% ✅ |
| **Active Workflows** | 24 | 22 ✅ | 18 ✅ | 16 ✅ |

### Qualitative Targets

- ✅ Zero reduction in test coverage
- ✅ Zero security posture degradation
- ✅ Improved developer experience (faster feedback)
- ✅ Easier maintenance (standardized patterns)

---

## ⚠️ Risk Mitigation

### High-Risk Changes

**Merging Security Workflows:**
- ✅ Keep old workflows in `archived/` for 1 month
- ✅ Test thoroughly on feature branch first
- ✅ Monitor security scan results for 1 week
- ✅ Rollback plan: `git checkout HEAD~1 -- .github/workflows/security-*.yml`

**Change Detection Logic:**
- ✅ Extensive testing with various PR types
- ✅ Fallback to full run on detection errors
- ✅ Monitor false negatives for 2 weeks

### Low-Risk Changes (Safe to Deploy)

- Adding caching (worst case: cache miss, no regression)
- Parallelizing jobs (no functionality change)
- Removing duplicate workflows (tests covered elsewhere)
- Consolidating basicsr checks (2 checks is still safe)

---

## 🔄 Rollback Procedures

**If Phase 1 changes cause issues:**
```bash
# Restore deleted workflow
git checkout HEAD~1 -- .github/workflows/depth_quality.yml
git commit -m "Rollback: Restore depth_quality workflow"

# Revert MaterialsV3 PR triggers
git checkout HEAD~1 -- .github/workflows/materialsv3_tests.yml
git commit -m "Rollback: Re-enable MaterialsV3 PR triggers"
```

**If Phase 2 changes cause issues:**
```bash
# Restore old security workflows
git checkout HEAD~1 -- .github/workflows/security-scan.yml
git checkout HEAD~1 -- .github/workflows/security-gates.yml
git rm .github/workflows/security-comprehensive.yml
git commit -m "Rollback: Restore separate security workflows"
```

---

## 📈 Monitoring Plan

**Week 1 (Phase 1):**
- [ ] Monitor PR CI runtimes daily
- [ ] Check for test failures or coverage drops
- [ ] Verify cache hit rates improving

**Week 2-3 (Phase 2):**
- [ ] Monitor security scan results
- [ ] Check for missed test runs (change detection issues)
- [ ] Verify dependency installation times

**Month 2 (Phase 3):**
- [ ] Monitor Docker build success rates
- [ ] Check service endpoint test results
- [ ] Verify memory leak detection working

---

## 📞 Contact & Support

**Questions or Issues?**
- Review full analysis: `.github/workflows/WORKFLOW_OPTIMIZATION_ANALYSIS.md`
- Contact: Transformation Portal Architect
- Escalation: Open GitHub issue with label `ci-optimization`

---

**Document Version:** 1.0
**Last Updated:** 2026-01-02
**Status:** ✅ Ready for Implementation
