# Phase 6 Reconstruction Performance Budgets

## Purpose

Define nightly regression budgets for the reconstruction rasterizer hot paths.
The enforcement tier is governed by [Performance Gate Policy](GATE_POLICY.md).
These budgets are enforced by:

- `tests/spatial_ai/reconstruction/test_performance_budgets.py`
- `.github/workflows/ml-slow-suite.yml`

## Budget Surface (CPU Reference)

Fixture profile:

- 96 gaussians
- 64x64 render size
- float32 tensors
- forward and backward timing loops with warmup

Budgets in milliseconds:

### Forward Render

- `p50 <= 80`
- `p95 <= 150`
- `max <= 220`

### Backward Pass

- `p50 <= 220`
- `p95 <= 420`
- `max <= 650`

## Metric Tracking

When `TP_RECON_PERF_METRICS_FILE` is set, the budget test emits JSON metrics:

- `forward_cpu`
- `backward_cpu`
- measured percentiles and budget thresholds

In nightly CI this is written to:

- `test-results/reconstruction_performance_metrics.json`

and uploaded as an artifact for trend review.

## Operational Notes

- These tests are marked `ml`, `slow`, and `benchmark` by design.
- They are excluded from fast PR lanes and run in the nightly slow-ML lane.
- Threshold updates should be data-driven and documented in this file.
