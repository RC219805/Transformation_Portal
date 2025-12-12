# CI/CD Integration - Phase 2 Features

**Date**: December 12, 2025  
**Task**: Phase 2 Section 2.4 - CI/CD Integration Enhancements  
**Status**: ✅ Complete

---

## Overview

This document describes the Phase 2 CI/CD integration enhancements that enable controlled rollout of Phase 2 features (CLIP classification, Lighting Detection) with optional performance regression testing.

---

## Feature Flags

### Workflow Dispatch Inputs

The CI workflow (`ci-consolidated.yml`) now supports two new feature flags:

| Flag | Default | Description |
|------|---------|-------------|
| `enable_phase2_features` | `true` | Enable CLIP and Lighting Detection features in tests |
| `run_benchmark_regression` | `false` | Run Phase 2 performance benchmarks and regression checks |

### Usage

**Enable/Disable Phase 2 Features:**
```bash
# From GitHub UI: Actions > CI/CD Pipeline > Run workflow
# Set "Enable Phase 2 features" to true/false

# From gh CLI:
gh workflow run ci-consolidated.yml \
  --ref main \
  -f enable_phase2_features=false
```

**Run Benchmark Regression:**
```bash
gh workflow run ci-consolidated.yml \
  --ref main \
  -f run_benchmark_regression=true
```

---

## Conditional Model Downloads

### CLIP Model Caching

Phase 2 features require the CLIP model (`openai/clip-vit-base-patch32`). The CI workflow now conditionally downloads this model based on the `enable_phase2_features` flag.

**Implementation:**
```yaml
- name: Download Phase 2 Models (Conditional)
  if: env.ENABLE_PHASE2_FEATURES == 'true'
  run: |
    python -c "
    from transformers import CLIPProcessor, CLIPModel
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    "
  continue-on-error: true
```

**Benefits:**
- **Faster CI**: Skips ~200MB model download when Phase 2 features disabled
- **Offline Mode**: Tests can run with `TRANSFORMERS_OFFLINE=1` after initial cache
- **Graceful Degradation**: Model download failures don't fail the entire pipeline

### Model Size

| Model | Size | Cache Location |
|-------|------|----------------|
| CLIP ViT-B/32 | ~600MB | `~/.cache/huggingface/hub/` |

---

## Test Filtering

### Phase 2 Test Markers

Tests related to Phase 2 features should be marked with `@pytest.mark.phase2`:

```python
import pytest

@pytest.mark.phase2
def test_clip_classification():
    """Test CLIP-based material classification."""
    # ...
```

### Conditional Test Execution

The CI workflow filters tests based on the `enable_phase2_features` flag:

```yaml
- name: Run ML Tests
  run: |
    if [ "$ENABLE_PHASE2_FEATURES" == "true" ]; then
      TEST_FILTER="ml or slow"
    else
      TEST_FILTER="ml or slow and not phase2"
    fi
    
    pytest tests/ -k "$TEST_FILTER" -v
```

---

## Performance Benchmark Regression

### Overview

The optional benchmark regression job runs `bench/bench_phase2.py` and validates performance metrics against defined thresholds.

### Performance Thresholds

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| CLIP Classification | < 500ms | One-time per-image overhead, acceptable for quality gain |
| Pipeline Initialization | < 2.0s | Model loading dominates, threshold allows for variance |
| Peak Memory Usage | < 1200MB | Current: ~850MB, headroom for future features |

### Triggering Benchmark

**Manual Trigger:**
```bash
gh workflow run ci-consolidated.yml \
  --ref main \
  -f run_benchmark_regression=true
```

**Automatic Trigger (Future):**
- On PR to `main` when Phase 2 files changed
- Nightly scheduled run on `main` branch

### Regression Detection

The benchmark job automatically fails if any metric exceeds its threshold:

```python
# Check CLIP performance
if clip_time > MAX_CLIP_TIME_MS:
    failures.append(f'CLIP: {clip_time}ms > {MAX_CLIP_TIME_MS}ms')

# Check initialization time
if init_time > MAX_INIT_TIME_S:
    failures.append(f'{tier} init: {init_time}s > {MAX_INIT_TIME_S}s')
```

### Benchmark Artifacts

**Artifacts Uploaded:**
- `bench/results/phase2_benchmark_results.json` - Machine-readable results
- `docs/PHASE2_PERFORMANCE.md` - Human-readable report

**Retention:** 30 days (longer than default 7 days for trending analysis)

### PR Comments

When run on a PR, the benchmark job automatically posts results as a comment:

```markdown
## 📊 Phase 2 Performance Benchmark

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| CLIP Classification | 245.3ms | 500ms | ✅ |
| STANDARD Init | 1.01s | 2.0s | ✅ |
| STANDARD Memory | 847MB | 1200MB | ✅ |
| MAX Init | 1.02s | 2.0s | ✅ |
| MAX Memory | 851MB | 1200MB | ✅ |
| APEX Init | 1.04s | 2.0s | ✅ |
| APEX Memory | 855MB | 1200MB | ✅ |
```

---

## Change Detection

### Phase 2 File Patterns

The workflow detects changes to Phase 2-specific files:

```yaml
phase2:
  - 'lux_depth_v2/materials_v2.py'
  - 'lux_depth_v2/lighting_detector.py'
  - 'tests/test_phase2_*.py'
  - 'bench/bench_phase2.py'
```

**Behavior:**
- Changes to Phase 2 files → Run ML tests (CLIP model required)
- Changes to benchmark → Suggest running with `run_benchmark_regression=true`

---

## Environment Variables

### Global CI Environment

```yaml
env:
  ENABLE_PHASE2_FEATURES: ${{ github.event.inputs.enable_phase2_features || 'true' }}
  RUN_BENCHMARK_REGRESSION: ${{ github.event.inputs.run_benchmark_regression || 'false' }}
```

**Defaults:**
- `enable_phase2_features`: `true` (Phase 2 features enabled by default)
- `run_benchmark_regression`: `false` (Benchmark opt-in to avoid CI overhead)

---

## Migration Path

### Gradual Rollout (Complete)

- ✅ **Phase 1**: Feature flags added, default `true`
- ✅ **Phase 2**: Conditional model downloads implemented
- ✅ **Phase 3**: Benchmark regression job added
- ⏳ **Phase 4**: Enable automatic benchmark on Phase 2 file changes (future)
- ⏳ **Phase 5**: Add trend tracking for performance metrics (future)

### Rollback Plan

If Phase 2 features cause CI instability:

1. **Immediate**: Set default `enable_phase2_features=false`
2. **Short-term**: Fix underlying issue, validate with manual runs
3. **Long-term**: Re-enable default `true` after validation

---

## Best Practices

### For Developers

1. **Mark Phase 2 tests**: Use `@pytest.mark.phase2` for CLIP/Lighting tests
2. **Test locally first**: Run `bench/bench_phase2.py` before pushing
3. **Check thresholds**: Ensure your changes don't regress performance
4. **Use feature flags**: Test with Phase 2 disabled if adding core functionality

### For CI Maintainers

1. **Monitor benchmark trends**: Review artifacts for gradual performance degradation
2. **Update thresholds**: Adjust based on hardware changes or model updates
3. **Keep models cached**: Ensure CLIP model is pre-cached on CI runners when possible
4. **Document changes**: Update this file when adding new Phase 2 features

---

## Troubleshooting

### CLIP Model Download Fails

**Symptom:** "CLIP model download failed" warning in logs

**Solutions:**
1. Check HuggingFace Hub status
2. Verify network connectivity from GitHub Actions runners
3. Fallback: Tests requiring CLIP are automatically skipped
4. Manual fix: Pre-cache model in runner image (future optimization)

### Benchmark Regression False Positives

**Symptom:** Benchmark fails despite no code changes

**Causes:**
- CI runner variance (different CPU/memory)
- Cold cache vs warm cache
- GitHub Actions infrastructure changes

**Solutions:**
1. Re-run workflow with fresh cache
2. Review threshold appropriateness
3. Check for outliers in benchmark results
4. Consider 3-run averaging (future enhancement)

### Phase 2 Tests Skipped Unexpectedly

**Symptom:** Phase 2 tests not running when expected

**Checks:**
1. Verify `enable_phase2_features=true` in workflow
2. Check test markers: `pytest --collect-only -m phase2`
3. Ensure CLIP model downloaded successfully
4. Review test filter logic in workflow

---

## Related Documentation

- **Phase 2 Implementation Guide**: `lux_depth_v2/PHASE2_IMPLEMENTATION_GUIDE.md`
- **Benchmark Documentation**: `bench/README.md`
- **Performance Report**: `docs/PHASE2_PERFORMANCE.md` (generated by benchmark)
- **CLIP Classifier**: `src/transformation_portal/segmentation/clip_classifier.py`
- **Lighting Detector**: `lux_depth_v2/lighting_detector.py`

---

## Future Enhancements

### Planned (Future Phases)

1. **Automatic Benchmark on PR**: Run regression on Phase 2 file changes
2. **Trend Dashboard**: Track performance metrics over time
3. **GPU Benchmark Matrix**: Compare CUDA vs MPS vs CPU performance
4. **EfficientSAM Integration**: Add backend comparison when implemented
5. **Nightly Benchmarks**: Scheduled runs to catch gradual degradation

### Under Consideration

- **Multi-run Averaging**: Reduce variance in benchmark results
- **Artifact Comparison**: Diff current vs baseline benchmark results
- **Custom Thresholds per PR**: Allow temporary threshold increases with justification
- **Model Pre-caching**: Store CLIP model in runner image for faster CI

---

## Changelog

### 2025-12-12 - Initial Release

- Added `enable_phase2_features` workflow input (default: `true`)
- Added `run_benchmark_regression` workflow input (default: `false`)
- Implemented conditional CLIP model download
- Created `benchmark-phase2` job with regression checks
- Updated test filtering for Phase 2 features
- Added Phase 2 change detection patterns
- Created this documentation

---

## Contact

**Maintained by**: Transformation Portal Core Team  
**Questions**: Open an issue with label `ci/cd` or `phase2`  
**Related PRs**: See Phase 2 task tracking in project board
