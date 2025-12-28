# Throughput Validation System

## Overview

This document describes the automated throughput validation system implemented to address **P0-2** from the Performance Readiness Assessment (#606).

**Purpose**: Validate production claims of 127-400 images/hour batch processing throughput.

## Architecture

The throughput validation system consists of four components:

### 1. Test Suite (`tests/test_performance_throughput.py`)

pytest-based performance tests that measure real end-to-end pipeline throughput:

- **`test_throughput_standard_quality`**: Validates CPU baseline (target: >50 images/hour)
- **`test_throughput_max_quality`**: Validates GPU/CPU max quality (target: >30-100 images/hour)
- **`test_throughput_scaling`**: Validates linear scaling and detects memory leaks

**Features**:
- Processes 10 synthetic images through full pipeline
- Measures images/hour, seconds/image, and peak memory
- Adaptive thresholds based on hardware availability (GPU vs CPU)
- Conservative baselines to avoid false negatives in CI

### 2. Baseline Configuration (`bench/baselines/throughput_baseline.json`)

Defines minimum acceptable performance thresholds:

```json
{
  "baselines": {
    "standard_quality_cpu": {
      "min_images_per_hour": 50,
      "max_memory_mb": 2000
    },
    "max_quality_cpu": {
      "min_images_per_hour": 30,
      "max_memory_mb": 3000
    },
    "max_quality_gpu": {
      "min_images_per_hour": 100,
      "max_memory_mb": 3000
    }
  },
  "production_targets": {
    "cpu_standard": {"target_images_per_hour": 127},
    "gpu_max": {"target_images_per_hour": 400}
  }
}
```

**Philosophy**:
- **Baselines**: Conservative minimum thresholds (50-100 images/hour) that CI must meet
- **Production Targets**: Aspirational goals (127-400 images/hour) for informational comparison
- CI **fails** if baseline not met, **warns** if production target not reached

### 3. Validation Script (`scripts/validate_throughput.py`)

Command-line tool for comparing benchmark results against baselines:

```bash
python scripts/validate_throughput.py \
  --baseline bench/baselines/throughput_baseline.json \
  --current throughput_results.json \
  --quality standard \
  --max-regression 20
```

**Features**:
- Validates throughput and memory against baseline thresholds
- Detects regressions >20% (configurable)
- Compares against production targets (informational)
- Exits with code 1 if validation fails (blocks CI)

### 4. CI Integration (`.github/workflows/ci-consolidated.yml`)

**Job**: `test-throughput`  
**Stage**: 3.5 (between `test-core` and `test-ml`)  
**Triggers**: Pull requests and pushes to main

**Workflow**:
1. Run `pytest tests/test_performance_throughput.py` with benchmark output
2. Extract metrics from pytest-benchmark JSON
3. Validate against `bench/baselines/throughput_baseline.json`
4. Post results as PR comment
5. Upload artifacts for trend analysis

**PR Comment Example**:
```
📊 Throughput Validation Results

Standard Quality (CPU)
- Throughput: 67.3 images/hour
- Memory Peak: 1,234.5 MB
- Baseline: 50 images/hour minimum

✅ Status: Meets baseline requirements

Target: 127 images/hour (CPU production claim)
```

## Performance Budgets

The system uses performance budgets defined in `bench/config/performance_budgets.yaml`:

- **Per-operation budgets**: Depth estimation, material segmentation, upscaling
- **End-to-end budgets**: Throughput, latency percentiles, memory
- **Quality tier budgets**: Standard, Max, APEX quality levels
- **Hardware profiles**: GitHub Actions, Apple Silicon, NVIDIA GPU

## Usage

### Local Testing

```bash
# Run throughput tests
pytest tests/test_performance_throughput.py -v

# Run with benchmark output
pytest tests/test_performance_throughput.py --benchmark-json=results.json

# Validate against baseline
python scripts/validate_throughput.py \
  --baseline bench/baselines/throughput_baseline.json \
  --current results.json \
  --quality standard
```

### CI Behavior

**On Pull Request**:
1. Throughput tests run automatically
2. Results compared to baseline
3. PR comment posted with metrics
4. **CI blocks merge** if throughput < baseline

**On Push to Main**:
1. Same validation runs
2. Results archived for trend analysis
3. Future: Baseline auto-update on releases

## Baseline Philosophy

### Why Conservative Baselines?

The baselines are intentionally **conservative** (50-100 images/hour vs 127-400 production target):

1. **CI Environment Variance**: GitHub Actions runners vary in performance
2. **Avoid False Negatives**: Don't block PRs for minor performance fluctuations
3. **Safety Net**: Catch catastrophic regressions (e.g., 10x slowdown)
4. **Production Aspirational**: Track production targets separately for informational purposes

### Baseline Update Strategy

Baselines should be updated when:
1. **Intentional Optimization**: PR improves throughput, update baseline upward
2. **Hardware Change**: CI runner upgrade, recalibrate baselines
3. **Regression Fix**: After fixing a regression, restore baseline
4. **Never**: Don't lower baselines to make CI pass

## Regression Detection

### What Triggers CI Failure?

- Throughput < baseline minimum (e.g., < 50 images/hour)
- Memory > baseline maximum (e.g., > 2000 MB)
- Regression > 20% from previous baseline

### What Triggers CI Warning?

- Throughput < production target (informational)
- Memory approaching limits (85-95% of threshold)
- Variability > 30% between batch sizes (suggests instability)

## Future Enhancements

### Phase 1 (Current - P0-2)
- ✅ Automated throughput validation
- ✅ Baseline configuration
- ✅ CI integration with PR comments
- ✅ Performance budgets documentation

### Phase 2 (P1)
- [ ] Baseline versioning (`bench/baselines/v1.0.0/`)
- [ ] Historical trend tracking
- [ ] Automated baseline updates on releases
- [ ] Multi-environment baselines (CPU/GPU/Apple Silicon)

### Phase 3 (P2)
- [ ] Latency percentile validation (P50/P95/P99)
- [ ] Memory leak detection (100+ image batches)
- [ ] GPU performance validation (when GPU runner available)
- [ ] Performance dashboard integration

## Related Documentation

- `.github/workflows/PERFORMANCE_READINESS_ASSESSMENT.md` - Full assessment and roadmap
- `bench/config/performance_budgets.yaml` - Performance budget definitions
- `tests/test_performance_depth.py` - Individual operation benchmarks (P0-1)

## Integration with Existing Systems

This throughput validation complements existing performance infrastructure:

- **`performance-monitor.yml`**: Runs pytest-benchmark tests (P0-1 completed in #607)
- **Phase 2 Benchmark**: Initialization and CLIP timing (manual dispatch)
- **Throughput Validation**: End-to-end pipeline throughput (automatic on PR)

Together, these systems provide comprehensive performance coverage:
1. **Unit-level**: Individual operations (depth, upscale, etc.)
2. **Integration-level**: Pipeline initialization and overhead
3. **System-level**: End-to-end throughput validation ← **This PR**

## Maintenance

**Owner**: Transformation Portal Architect  
**Review Cadence**: Quarterly baseline review  
**Update Triggers**: Major feature changes, CI environment changes, regression fixes

---

**Last Updated**: 2025-12-28  
**Version**: 1.0.0  
**PR**: #608 (Automated Throughput Validation - P0-2)
