# CI/CD Optimization Plan

**Status**: Phase 1 Complete (2026-01-02)
**Target**: Reduce CI duration from 458s avg to <300s (35% improvement)
**Current Success Rate**: 68% → Target: 95%+

---

## ✅ Phase 1: Critical Failures Eliminated (Complete)

### Workflow Fixes (12 failures eliminated)
- ✅ Fixed `smart-issue-management.yml` - 0% → Skip when OPENAI_API_KEY missing
- ✅ Fixed `ai-code-review.yml` - 0% → Skip when OPENAI_API_KEY missing
- ✅ Fixed `summary.yml` - 0% → Skip when OPENAI_API_KEY missing

**Implementation**: Added `check-api-key` job to validate secret availability before running AI-dependent workflows. Workflows now skip cleanly instead of failing.

**Impact**:
- 12 workflow failures eliminated
- Overall success rate improved from 68% → Expected 85%+
- Reduced noise in CI dashboard

---

## 🚀 Phase 2: Performance Optimization (Next Priority)

### 2.1 Cache Optimization (Target: -60s)

**Current State**: CI reinstalls dependencies on every run
**Opportunity**: Aggressive caching of pip packages, pre-commit envs, test artifacts

```yaml
# Add to ci-consolidated.yml
- name: Cache pip packages
  uses: actions/cache@v4
  with:
    path: ~/.cache/pip
    key: pip-${{ runner.os }}-${{ hashFiles('requirements*.txt') }}
    restore-keys: |
      pip-${{ runner.os }}-

- name: Cache pre-commit
  uses: actions/cache@v4
  with:
    path: ~/.cache/pre-commit
    key: pre-commit-${{ hashFiles('.pre-commit-config.yaml') }}
```

**Estimated Impact**: 40-60s reduction per run

---

### 2.2 Parallel Job Optimization (Target: -90s)

**Current State**: Some jobs run sequentially that could run in parallel
**Opportunity**: Reorganize job dependencies to maximize parallelism

**Proposed Changes**:
```yaml
jobs:
  setup: # Fast change detection

  # Run in parallel after setup
  lint:
    needs: setup
  test-unit:
    needs: setup
  test-integration:
    needs: setup
  security-scan:
    needs: setup

  # Only after all tests pass
  build-package:
    needs: [lint, test-unit, test-integration, security-scan]
```

**Estimated Impact**: 60-90s reduction via parallelization

---

### 2.3 Test Selection Optimization (Target: -120s)

**Current State**: Runs all 2751 tests on every PR
**Opportunity**: Intelligent test selection based on changed files

```yaml
- name: Select tests
  id: test-selection
  run: |
    if [[ "${{ steps.changes.outputs.ml_any_changed }}" == "true" ]]; then
      echo "test_markers=ml" >> $GITHUB_OUTPUT
    elif [[ "${{ steps.changes.outputs.core_any_changed }}" == "true" ]]; then
      echo "test_markers=unit" >> $GITHUB_OUTPUT
    else
      echo "test_markers=smoke" >> $GITHUB_OUTPUT
    fi

- name: Run selected tests
  run: pytest -m "${{ steps.test-selection.outputs.test_markers }}"
```

**Estimated Impact**: 90-120s reduction for small PRs

---

### 2.4 Docker Layer Caching (Target: -30s)

**Current State**: Rebuilds Docker images from scratch
**Opportunity**: Layer caching and multi-stage builds

```dockerfile
# Use BuildKit cache mounts
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt
```

**Estimated Impact**: 20-30s reduction

---

## 📊 Expected Results After Phase 2

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| **Avg Duration** | 458s | <300s | -35% |
| **Success Rate** | 68% | 95%+ | +27% |
| **Cache Hit Rate** | 0% | 80%+ | +80% |
| **Parallel Jobs** | 4 | 8 | +100% |

---

## 🔧 Phase 3: Advanced Optimizations (Future)

### 3.1 Self-Hosted Runners
- **Target**: -50% duration via dedicated hardware
- **Cost**: $50-100/month for GitHub Actions minutes saved
- **ROI**: Positive if >1000 runs/month

### 3.2 Distributed Testing
- **Target**: -40% duration via pytest-xdist
- **Implementation**: Already installed, needs configuration
- **Impact**: Linear speedup with CPU cores

### 3.3 Incremental Linting
- **Target**: -20s via only linting changed files
- **Implementation**: Use `--from-ref` with ruff
- **Impact**: Proportional to change size

---

## 📈 Monitoring & Metrics

### Key Performance Indicators (KPIs)

Track via `scripts/workflow_health_check.py`:

1. **Duration Percentiles**
   - P50: <250s
   - P95: <400s
   - P99: <600s

2. **Success Rate by Workflow**
   - All workflows: >95%
   - Critical path (CI): >98%

3. **Cache Effectiveness**
   - Pip cache hit: >80%
   - Pre-commit cache hit: >90%
   - Docker layer cache hit: >70%

4. **Resource Utilization**
   - Runner minutes/month: <2000
   - Average concurrent jobs: 4-8
   - Queue time: <30s

---

## 🎯 Implementation Checklist

### Immediate (This Week)
- [x] Fix failing AI workflows (COMPLETE)
- [x] Add workflow status badges (COMPLETE)
- [ ] Implement pip caching
- [ ] Implement pre-commit caching
- [ ] Reorganize job dependencies

### Short-term (This Month)
- [ ] Implement test selection
- [ ] Add Docker layer caching
- [ ] Enable pytest-xdist parallelization
- [ ] Create CI performance dashboard

### Long-term (This Quarter)
- [ ] Evaluate self-hosted runners
- [ ] Implement incremental linting
- [ ] Add workflow duration alerts
- [ ] Optimize artifact retention policies

---

## 📚 References

- [GitHub Actions Cache Documentation](https://docs.github.com/en/actions/using-workflows/caching-dependencies-to-speed-up-workflows)
- [pytest-xdist Documentation](https://pytest-xdist.readthedocs.io/)
- [Docker BuildKit Cache](https://docs.docker.com/build/cache/)
- [Ruff Performance Guide](https://docs.astral.sh/ruff/linter/#performance)

---

## 🔄 Review Schedule

- **Weekly**: Monitor KPIs via workflow_health_check.py
- **Monthly**: Review optimization impact and adjust targets
- **Quarterly**: Evaluate new optimization opportunities

---

**Last Updated**: 2026-01-02
**Next Review**: 2026-01-09
**Owner**: Repository Maintainers
