# Benchmark Tests for lux_depth_v3

This directory contains performance benchmark tests for the lux_depth_v3 pipeline,
as part of the v2.1 Optimization Roadmap.

## Overview

The benchmark suite provides:
- **Baseline performance measurement** (p50/p95 runtime, throughput)
- **Memory tracking** (peak RSS, incremental usage)
- **Output invariants validation** (dtype, range, shape checks)
- **Regression detection** (automated threshold checks)

## Running Benchmarks

### Run all benchmarks
```bash
pytest tests/benchmarks/ -v -m benchmark
```

### Run specific benchmark test
```bash
pytest tests/benchmarks/test_lux_depth_v3_perf_smoke.py::TestLuxDepthV3PerformanceBaseline::test_single_image_baseline_runtime -v -s
```

### Exclude benchmarks from regular test runs
```bash
pytest tests/ -v -m "not benchmark"
```

## Benchmark Tests

### `test_lux_depth_v3_perf_smoke.py`

**Purpose:** Fast smoke tests for performance baseline and regression detection.

**Tests:**
- `test_single_image_baseline_runtime`: Measures p50/p95 runtime for single images
- `test_batch_processing_baseline`: Measures batch throughput
- `test_output_invariants_smoke`: Validates output correctness
- `test_memory_baseline_peak_tracking`: Tracks peak memory usage (requires psutil, skips in core CI)
- `test_no_model_reinitialization_guard`: Placeholder for backend singleton checks (L1.0)
- `test_p95_latency_regression_threshold`: Regression guard (to be populated in L1.x)

**Execution Time:** <3 seconds (target: <10s)

**Dependencies:**
- No model downloads (uses synthetic backend fallback)
- No network calls
- Fully deterministic synthetic fixtures
- Memory tracking test requires `psutil` (skips gracefully if unavailable in core CI)

## Optimization Roadmap Integration

These benchmarks support the lux_depth_v3 v2.1 Optimization Roadmap:

**Milestone 0: Baselines & Guardrails**
- ✅ **L0.0**: Benchmark harness (this directory)
- 🔜 **L0.1**: Memory regression + peak tracking
- 🔜 **L0.2**: Output invariants & golden-delta tests

**Milestone 1: Throughput Wins**
- 🔜 **L1.0**: Backend warm pool + singleton registry cache
- 🔜 **L1.1**: Mixed precision inference path
- 🔜 **L1.2**: Channels-last + contiguity discipline
- 🔜 **L1.3**: torch.compile (optional, safe fallback)

**Future Milestones:** L2.x (Scalability), L3.x (Quality-safe optimizations)

## JSON Output Format

Benchmarks produce machine-readable JSON for automated regression detection:

```json
{
  "test": "single_image_baseline",
  "fixture": "512x512",
  "megapixels": 0.26,
  "p50_ms": 55.0,
  "p95_ms": 56.0,
  "per_mp_ms": 210.0
}
```

## CI Integration

Benchmarks use the `@pytest.mark.benchmark` marker:
- Excluded from fast PR gating CI by default (`-m "not benchmark"`)
- Run in nightly/deep checks or manually
- Future: Add explicit regression thresholds in CI

## Design Principles

1. **Fast execution** (<10s target, achieved <3s)
2. **Fully offline** (no model downloads, no network)
3. **Deterministic** (same fixtures, same results)
4. **Machine-readable** (JSON output for automation)
5. **Isolated** (no coupling to spatial_ai or other pipelines)

## Adding New Benchmarks

When adding new benchmark tests:

1. Mark with `@pytest.mark.benchmark`
2. Use deterministic synthetic fixtures (no random, no downloads)
3. Keep execution time fast (<5s per test)
4. Produce JSON output for regression tracking
5. Document expected performance characteristics
6. Add to this README

## Questions?

See `docs/architecture/optimization_roadmap.md` (to be created) or
the original issue for the v2.1 Optimization Roadmap.
