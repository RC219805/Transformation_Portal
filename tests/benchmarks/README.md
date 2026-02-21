# Benchmark Tests for lux_depth_v3

This directory contains performance benchmark tests for the lux_depth_v3 pipeline,
as part of the v2.1 Optimization Roadmap.

## Overview

The benchmark suite provides:
- **Baseline performance measurement** (p50/p95 runtime, throughput)
- **Memory tracking** (peak RSS via polling thread, processing-only window)
- **Output invariants validation** (dtype, range, shape checks)
- **Regression detection** (automated threshold checks - planned L0.2)

## Running Benchmarks

### Run all benchmarks
```bash
pytest tests/benchmarks/ -v -m benchmark
```

### Run specific benchmark test
```bash
pytest tests/benchmarks/test_lux_depth_v3_perf_smoke.py::TestLuxDepthV3PerformanceBaseline::test_single_image_cold_start_p95 -v -s
```

### Exclude benchmarks from regular test runs
```bash
pytest tests/ -v -m "not benchmark"
```

## CI Execution Policy

**Current State (Enforced):** Benchmarks are EXCLUDED from PR gating CI.

**Marker Expression:** `"not ml and not slow and not benchmark"`

**Rationale:**
1. **Determinism**: PR gating must be deterministic; CI runner variance breaks this invariant
2. **Governance Alignment**: Repository policy requires benchmarks out of fast PR feedback loops
3. **Separation of Concerns**: Performance regression detection belongs in nightly/scheduled workflows
4. **Fast Feedback**: Benchmark tests add latency and variance to critical PR gating

**Where Benchmarks Run:**
- Nightly performance workflows (`.github/workflows/nightly.yml`)
- Manual runs via `pytest -m benchmark`
- Scheduled deep-check workflows

**To Include Benchmark Tests (Not Recommended):**
Only do this if you need to temporarily debug performance issues in CI. Update `.github/workflows/build.yml`:
```yaml
# NOT RECOMMENDED: Remove "and not benchmark" from marker expressions
markexpr: "not ml and not slow"  # Don't do this!
```

**Governance Reference:** `.github/copilot-instructions.md` (Testing Taxonomy)

## Benchmark Tests

### `test_lux_depth_v3_perf_smoke.py`

**Purpose:** Fast smoke tests for performance baseline and regression detection.

**Performance Tests:**

1. **`test_single_image_cold_start_p95`**
   - **Type:** COLD-START measurement
   - **What:** Creates new orchestrator per run
   - **Includes:** Initialization overhead (directory creation, config parsing, backend instantiation)
   - **Use Case:** Worst-case single-image workflow, one-off processing
   - **Artifact:** `baseline_cold_start.json`

2. **`test_single_image_steady_state_p95`**
   - **Type:** STEADY-STATE measurement
   - **What:** Reuses orchestrator across runs, includes warm-up
   - **Excludes:** Initialization overhead
   - **Use Case:** Best-case throughput, batch workflows, long-running processes
   - **Artifact:** `baseline_steady_state.json`

3. **`test_batch_throughput_baseline`**
   - **Type:** BATCH THROUGHPUT measurement
   - **What:** Processes multiple different images sequentially
   - **Use Case:** Sustained production throughput
   - **Artifact:** `baseline_batch.json`

**Invariant Tests:**

4. **`test_output_invariants_smoke`**
   - Validates dtype, range, shape, no NaNs/inf
   - Checks both .npy and .png outputs
   - Ensures dimension preservation

**Memory Tests:**

5. **`test_memory_peak_rss_baseline`**
   - **Type:** PEAK RSS via polling thread (true high-water mark)
   - **What:** Polls RSS at ~5ms intervals during processing to capture peak
   - **Semantic:** Processing-only — baseline taken after orchestrator init
   - **Guarantee:** First-sample barrier ensures polling starts before workload
   - **Captures:** Transient allocation spikes missed by post-completion snapshots
   - **Use Case:** Detecting memory leaks and regression in allocation patterns
   - **Requires:** psutil (skips gracefully if unavailable)
   - **Artifact:** `baseline_memory.json` (includes `sample_count`, `sampling_interval_s`)

**Guard Tests:**

6. **`test_no_model_reinitialization_guard`**
   - Placeholder for L1.0 backend singleton checks

7. **`test_cold_start_p95_regression_threshold`**
   - Placeholder for L0.2 regression detection with % tolerance

8. **`test_steady_state_p95_regression_threshold`**
   - Placeholder for L0.2 regression detection with % tolerance

**Execution Time:** <2 seconds (target: <10s)

**Dependencies:**
- No model downloads (uses pinned synthetic backend)
- No network calls
- Fully deterministic vectorized fixtures (NumPy-based)
- Memory tracking test requires `psutil` (skips gracefully if unavailable)

**Measurement Methodology:**
- **Cold-start**: New orchestrator per run (includes initialization)
- **Steady-state**: Reused orchestrator + warm-up (excludes initialization)
- **Backend pinning**: Explicitly uses `depth_backend="synthetic"` for reproducibility
- **Fixture generation**: Vectorized NumPy (avoids Python loop contamination)
- **Timing**: Uses `time.perf_counter()` for monotonic high-resolution measurements
- **Percentiles**: Computed with `np.percentile()` for accuracy
- **Assertions**: Relaxed warnings instead of hard failures (prevents CI runner variance issues)
- **Artifact persistence**: Supports `BENCHMARK_ARTIFACTS_DIR` env var for cross-run baseline storage

**Known Limitations:**
- Memory test uses polling-thread peak RSS (~5ms sampling interval); sub-millisecond spikes may be missed
- Baseline persistence not yet implemented (each test writes to tmp_path)
- Absolute time assertions removed per architectural review
- Cross-run regression detection planned for L0.2

## Optimization Roadmap Integration

These benchmarks support the lux_depth_v3 v2.1 Optimization Roadmap:

**Milestone 0: Baselines & Guardrails**
- ✅ **L0.0**: Benchmark harness (this directory)
- ✅ **L0.1**: Peak RSS tracking (polling-thread high-water mark)
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
- Included in PR gating CI (marker expression `"not ml and not slow"` does NOT exclude `benchmark`)
- Assertions are warnings-only until L0.2 implements baseline comparison with % tolerance
- Dedicated benchmark runs can be invoked with `-m "benchmark"`

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
