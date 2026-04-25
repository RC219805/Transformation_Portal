# Depth Backend Benchmark Protocol

This benchmark governs the Depth Pro and DA3 comparison used by APEX quality work.

## Roles

- `da3_metric`: commercial-safe baseline.
- `depth_pro`: research-quality reference, gated by the existing non-commercial and Apple Depth Pro research-license acknowledgements.

Depth Pro is a quality yardstick, not a deployable default.

## Command

```bash
.venv/bin/python tools/benchmark_depth_backends.py \
  --evalset evalsets/picacho_apex \
  --backends da3-metric,depth_pro \
  --quality-tier apex \
  --output-dir output/depth_backend_benchmark \
  --emit-comparison-report on
```

Depth Pro remains `license_blocked` unless the operator explicitly passes:

```bash
--non-commercial-ok true \
--accept-apple-depth-pro-research-license true
```

## Report

The tool emits `depth_backend_comparison_report.json` with:

- backend id
- license tier
- per-asset status
- depth edge score
- boundary halo risk
- architectural plausibility
- runtime
- provenance, skip, and error fields

The default command is offline-safe and records assets as `not_executed` until a live execution runner is wired in. Unit tests must not download models.

## Promotion Rule

Depth stack changes should show a report-backed improvement on the same evalset before becoming APEX defaults. A backend can be faster or easier to operate, but APEX promotion requires quality evidence against this benchmark.
