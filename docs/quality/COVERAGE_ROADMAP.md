# Coverage Improvement Roadmap

**Status**: Active (as of 2026-02-04)
**Current Coverage**: 25.44%
**Target Coverage**: 33% → 50% → 70%

---

## Current Baseline

- **Combined Coverage**: 25.44% (core + ML tests)
- **Quality Gate Threshold**: 25% (hard fail)
- **Diff Coverage Requirement**: 80% (new/changed code only)

### Coverage by Component

| Component | Coverage | Priority |
|-----------|----------|----------|
| `lux_depth_v3/` | 68% | ✅ Good |
| `depth/` | 60% | 🟡 Moderate |
| `processors/material_response/` | 27% | 🔴 Critical |
| `pipelines/` | 20–73% (varies) | 🔴 Critical |
| `streaming/` | 9–31% | 🔴 Critical |
| `plugins/` | 0% | 🔴 Critical |

---

## Ratcheting Plan

### Phase 1: Stabilize at 25% (Complete)
- ✅ Fix coverage artifact collection (#832)
- ✅ Set realistic baseline (25%)
- ✅ Enforce 80% diff coverage on new code

### Phase 2: Reach 33% (Q1 2026)
**Focus**: High-value, low-hanging fruit

- [ ] Add integration tests for `pipelines/unified_luxury_pipeline.py` (+8%)
- [ ] Add unit tests for `processors/material_response/core.py` (+5%)
- [ ] Add tests for `streaming/stages.py` (+4%)
- [ ] **Target**: 33% total coverage

### Phase 3: Reach 50% (Q2 2026)
**Focus**: Core business logic

- [ ] Comprehensive tests for `depth_canonical/` pipeline
- [ ] Plugin system integration tests
- [ ] Streaming pipeline end-to-end tests
- [ ] **Target**: 50% total coverage

### Phase 4: Reach 70% (Q3 2026)
**Focus**: Edge cases and error paths

- [ ] All critical paths in `lux_render_pipeline.py`
- [ ] Error handling in `streaming/async_pipeline.py`
- [ ] CLI argument validation edge cases
- [ ] **Target**: 70% total coverage

---

## How to Contribute

### For New Code
- **Requirement**: 80% diff coverage (enforced by CI)
- Write tests before submitting PR
- Run `pytest --cov` locally before pushing

### For Legacy Code
- Pick a module from "Priority" table below
- Add tests incrementally (even 1 test helps!)
- Submit focused PRs: "test: add coverage for X module"

### Measuring Progress
```bash
# Run coverage locally
pytest --cov=src/transformation_portal --cov-report=term --cov-report=html

# View detailed report
open htmlcov/index.html

# Check specific module
pytest --cov=src/transformation_portal/pipelines --cov-report=term
```

---

## Module-Level Priorities

### Critical (0–20% coverage)
1. `plugins/` (0%) - Plugin loading and validation
2. `streaming/stages.py` (9%) - Stage execution logic
3. `vlm/` (0%) - Vision-language models
4. `style_transfer/` (0%) - Style transfer pipeline

### High (20–40%)
1. `processors/material_response/core.py` (27%)
2. `streaming/async_pipeline.py` (24%)
3. `depth/models/depth_anything_v2.py` (26%)

### Moderate (40–60%)
1. `depth/pipeline.py` (80%) - Already strong
2. `depth/backends/depth_pro.py` (48%)
3. `config_loader.py` (40%)

---

## CI Workflow

### Coverage Quality Gate (`.github/workflows/ci.yml`)

```yaml
- name: Check coverage thresholds
  run: |
    # Current baseline: 25% (as of 2026-02-04)
    coverage report --fail-under=25

- name: Check diff coverage
  run: |
    # New/changed code must be 80% covered
    diff-cover coverage.xml --compare-branch=origin/main --fail-under=80
```

### Updating the Baseline

When coverage improves sustainably (e.g., reaches 30% for 2+ weeks):

1. Update threshold in `.github/workflows/ci.yml`:
   ```yaml
   coverage report --fail-under=30
   ```

2. Update this document:
   ```markdown
   **Current Coverage**: 30%
   **Quality Gate Threshold**: 30%
   ```

3. Submit PR: `chore: raise coverage baseline to 30%`

---

## Notes

- **Diff coverage (80%)** is the primary ratcheting mechanism
  - Every PR with code changes must meet this
  - Prevents regression even if total coverage is low

- **Global baseline (25%)** prevents large drops
  - Only update when sustained improvement is proven
  - Should align with actual measured coverage on `main`

- **Coverage ≠ quality**, but it helps find:
  - Dead code
  - Untested error paths
  - Complex logic needing refactoring
