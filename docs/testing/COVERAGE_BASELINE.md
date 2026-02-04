# Test Coverage Baseline & Improvement Plan

## Current Status (2026-02-04)

**Combined Coverage**: 25.44%
- Core tests (Python 3.11-3.12): ~25%
- ML tests (Python 3.11): ~25%

**CI Enforcement**:
- Global minimum: 25% (enforced in `.github/workflows/ci.yml`)
- Diff coverage (PRs): 80% for new/changed lines
- Individual test runs: 20% minimum

## Coverage Distribution by Module

### Well-Covered Modules (>80%)
- `compliance/licensing.py`: 100%
- `depth/backends/cache.py`: 94.17%
- `depth/backends/protocol.py`: 91.84%
- `depth_canonical/config.py`: 91.28%
- `lux_depth_v3/pbr_cli.py`: 87.99%
- `lux_depth_v3/config.py`: 86.92%
- `utils/error_handling.py`: 89.04%
- `utils/format_utils.py`: 95.83%
- `stage_graph/stages/depth_pro.py`: 84.75%

### Critical Gaps (<20%)
- All `plugins/*` modules: 0%
- All `perceptual/*` modules: 0%
- All `neuroaesthetics/*` modules: 0%
- All `vlm/*` modules: 0%
- All `style_transfer/*` modules: 0%
- All `streaming/stages.py`: 9.38%
- `depth/processors/zone_tone_mapping.py`: 12.88%
- `depth/processors/depth_guided_filters.py`: 13.37%
- `depth/processors/numba_kernels.py`: 13.10%

## Improvement Roadmap

### Phase 1: Quick Wins (Target: 30% combined)
**Timeline**: 1 month

Priority targets:
1. **CLI layer** (`cli/__init__.py`): Currently 16.67%
   - Add command smoke tests
   - Test help output, version flags
   - Estimated gain: +2%

2. **Config loader** (`config_loader.py`): Currently 39.75%
   - Test YAML parsing edge cases
   - Test validation logic
   - Estimated gain: +1.5%

3. **Material response profiles** (`processors/material_response/profiles.py`): Currently 64.29%
   - Test all material presets
   - Estimated gain: +1%

4. **Stage graph basics** (`stage_graph/stage.py`, `stage_graph/graph.py`): Currently <35%
   - Unit tests for stage transitions
   - Policy validation tests
   - Estimated gain: +2%

**Expected total**: 28-30%

### Phase 2: Core Depth Pipeline (Target: 35%)
**Timeline**: 2 months

1. **Depth processors** (currently 13-57%):
   - `atmospheric_effects.py`: 57.94% → 75%
   - `zone_tone_mapping.py`: 12.88% → 50%
   - `depth_guided_filters.py`: 13.37% → 50%
   - Mock heavy Numba/torch operations
   - Estimated gain: +3%

2. **Orchestrator** (`lux_depth_v3/orchestrator.py`): 68.28% → 80%
   - Test error paths
   - Test checkpoint recovery
   - Estimated gain: +1%

3. **Depth backends** (currently 48-94%):
   - `depth_pro.py`: 48.51% → 65%
   - Test fallback logic
   - Estimated gain: +1.5%

**Expected total**: 33-35%

### Phase 3: Plugin System (Target: 40%)
**Timeline**: 3 months

1. Enable plugin system modules (currently 0%):
   - `plugins/loader.py`: 0% → 60%
   - `plugins/registry.py`: 0% → 60%
   - `plugins/validator.py`: 0% → 60%
   - Mock discovery, loading
   - Estimated gain: +3%

2. VLM and perceptual modules (currently 0%):
   - Basic smoke tests
   - Mock heavy model inference
   - Estimated gain: +2%

**Expected total**: 38-40%

### Phase 4: Streaming & Advanced (Target: 45%+)
**Timeline**: 4-6 months

1. Streaming pipeline (`streaming/*`): Currently 9-31%
   - Async workflow tests
   - Checkpoint/resume tests
   - Estimated gain: +3%

2. Style transfer (currently 0%):
   - Reference encoder tests
   - IP-Adapter integration tests
   - Estimated gain: +2%

**Expected total**: 43-45%

## Testing Strategy

### Unit Test Priorities
1. **Pure functions first**: Config parsing, validation, format utils
2. **Mock heavy dependencies**: Torch, transformers, FFmpeg, file I/O
3. **Test error paths**: Invalid inputs, missing files, device failures
4. **Avoid slow tests**: Mark ML/model tests as `@pytest.mark.ml`

### Integration Test Strategy
- Focus on happy-path workflows
- Use fixture images (small, fast)
- Mock external APIs (OpenAI, HuggingFace downloads)
- Run in separate CI job with extended timeout

### What NOT to Test Extensively
- Third-party library internals (torch, transformers)
- Rendering outputs (visual quality is subjective)
- Performance benchmarks (separate tooling)
- Deprecated/legacy code paths

## Monitoring & Enforcement

### CI Enforcement (Current)
```yaml
# .github/workflows/ci.yml
- Global threshold: 25% (fail if below)
- Diff coverage: 80% (fail if new code <80%)
- Individual jobs: 20% minimum
```

### Ratcheting Mechanism
- **Do not decrease** global threshold without explicit justification
- Increase threshold when sustained gains achieved:
  - 30% → update after Phase 1
  - 35% → update after Phase 2
  - 40% → update after Phase 3

### Coverage Reports
- Codecov integration: Tracks trends over time
- Artifacts uploaded: `htmlcov/`, `coverage.xml`
- Monthly review: Identify stagnant modules

## Contributing

When adding new code:
1. **Aim for 80% coverage** on new modules
2. **Test public APIs** thoroughly
3. **Mock expensive operations** (model inference, video encoding)
4. **Use fixtures** for common test data
5. **Mark appropriately**: `@pytest.mark.ml`, `@pytest.mark.slow`

When fixing bugs:
1. **Write failing test first** (TDD)
2. **Ensure coverage increases** (or stays same)
3. **Check diff-cover passes** (80% for changed lines)

## Historical Context

**Why is baseline 25%?**
- Repository evolved from research codebase
- Heavy ML/rendering components hard to test
- Initial focus on correctness over coverage
- Many modules experimental/exploratory

**Why enforce now?**
- Transition to production use
- Need regression protection
- Onboarding new contributors
- Preparing for v2.0.0 stability guarantees

## References

- [pytest documentation](https://docs.pytest.org/)
- [coverage.py docs](https://coverage.readthedocs.io/)
- [Codecov best practices](https://docs.codecov.com/docs/common-use-cases)
- Repository testing guide: `docs/testing/TESTING_GUIDE.md` (TODO)

---

**Last Updated**: 2026-02-04
**Next Review**: 2026-03-04 (after Phase 1 completion)
