# Performance Ledger System

Production-grade performance tracking and regression detection for the Transformation Portal.

## Overview

The Performance Ledger System captures **scene-dependent** performance characteristics with investor-grade rigor:

- **Honest time accounting**: No teleported time, all overhead measured
- **Scene-aware bucketing**: Pool scenes ≠ Aerial scenes ≠ Interiors
- **Phase-level timing**: Know where time is spent (load, inference, postprocess, write)
- **SQLite-backed ledger**: Efficient queries, atomic writes, crash-safe
- **Quality Firewall integration**: BLOCK/WARN/PASS verdicts based on buckets

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Orchestrator/Pipeline (instrumented with timing contexts)  │
└────────────┬────────────────────────────────────────────────┘
             │
             │ PerformanceCapsule
             ▼
┌─────────────────────────────────────────────────────────────┐
│ Performance Ledger (SQLite backend)                         │
│  - Append-only logging                                      │
│  - Indexed queries (scene_type, device, timestamp)          │
│  - Statistics computation                                   │
│  - Regression detection                                     │
└────────────┬────────────────────────────────────────────────┘
             │
             │ Query API
             ▼
┌─────────────────────────────────────────────────────────────┐
│ Analysis Tools                                              │
│  - Bucket matching                                          │
│  - Regression detection                                     │
│  - Performance reports                                      │
│  - Quality Firewall enforcement                             │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Instrument Your Pipeline

```python
from transformation_portal.metrics import (
    PerformanceCapsule,
    timing_context,
    compute_config_hash,
    compute_dimension_adjustment,
)

# Capture timings
timings = {}

with timing_context("load_decode", timings):
    image = load_image(path)

with timing_context("inference", timings):
    depth = backend.compute(image)

with timing_context("write_depth", timings):
    write_depth_map(depth, output_path)

# Total time
timings["total"] = sum(timings.values())

# Create capsule
capsule = PerformanceCapsule(
    image_id=Path(path).stem,
    image_path=str(path),
    input_hash=compute_hash(image),
    original_shape=original.shape[:2],
    enforced_shape=enforced.shape[:2],
    pixel_count=enforced.shape[0] * enforced.shape[1],
    dimension_adjustment=compute_dimension_adjustment(
        original.shape[:2], enforced.shape[:2]
    ),
    backend_id="da3",
    device="mps",
    timings=timings,
    scene_type="pool",  # Classify scene
    config_hash=compute_config_hash(config),
    pipeline_version="2.0.0",
    firewall_status="unknown",  # To be determined
)
```

### 2. Log to Ledger

```python
from transformation_portal.metrics.ledger import PerformanceLedger

ledger = PerformanceLedger(Path("./performance.db"))
ledger.log_capsule(capsule)
```

Or via CLI:

```bash
# Save capsule to JSON
echo '{"image_id": "test", ...}' > capsule.json

# Log to ledger
python -m transformation_portal.metrics.ledger log \
  --capsule capsule.json \
  --ledger-db performance.db
```

### 3. Query Historical Data

```python
# Query pool scenes on MPS from last 30 days
from datetime import datetime, timedelta, timezone

cutoff = datetime.now(timezone.utc) - timedelta(days=30)

capsules = ledger.query_capsules(
    scene_type="pool",
    device="mps",
    min_captured_at=cutoff.isoformat(),
)

print(f"Found {len(capsules)} pool scenes")
```

Or via CLI:

```bash
python -m transformation_portal.metrics.ledger query \
  --ledger-db performance.db \
  --scene-type pool \
  --device mps \
  --min-days 30 \
  --output results.json
```

### 4. Detect Regressions

```python
from transformation_portal.metrics.ledger import detect_regression

# Compare current to historical
result = detect_regression(current_capsule, historical_capsules)

if result["status"].startswith("regression"):
    print(f"⚠️  Regression detected: {result['message']}")
    sys.exit(1)
```

Or via CLI:

```bash
python -m transformation_portal.metrics.ledger regression \
  --ledger-db performance.db \
  --capsule current_capsule.json \
  --baseline-days 30
# Exit code 1 if regression detected
```

### 5. Generate Reports

```bash
python -m transformation_portal.metrics.ledger report \
  --ledger-db performance.db \
  --output performance_report.md \
  --min-days 7
```

## Performance Buckets

Buckets define scene-specific thresholds for Quality Firewall enforcement:

| Bucket Name           | Filters                                          | p50   | p95   | Description                          |
|-----------------------|--------------------------------------------------|-------|-------|--------------------------------------|
| `aerial_large_mps`    | scene=aerial, pixels≥20M, device=mps             | 8.5s  | 12.0s | Large aerial high-frequency texture  |
| `pool_medium_mps`     | scene=pool, pixels≥10M, device=mps               | 11.0s | 15.0s | Pool scenes with reflections         |
| `interior_standard_mps`| scene=interior, pixels≤15M, device=mps          | 7.0s  | 10.0s | Standard architectural interiors     |
| `generic_large`       | pixels≥20M                                       | 10.0s | 15.0s | Fallback for large images            |
| `generic_medium`      | 5M≤pixels<20M                                    | 6.0s  | 10.0s | Fallback for medium images           |

**Firewall Logic:**
- `runtime > p95` → **BLOCK** (regression)
- `runtime > p50 × 1.5` → **WARN** (investigate)
- `runtime ≤ p50` → **PASS** (nominal)

## CLI Reference

### `log` - Log a performance capsule

```bash
python -m transformation_portal.metrics.ledger log \
  --capsule capsule.json \
  --ledger-db performance.db
```

### `query` - Query historical capsules

```bash
python -m transformation_portal.metrics.ledger query \
  --ledger-db performance.db \
  --scene-type pool \
  --device mps \
  --backend-id da3 \
  --min-days 30 \
  --limit 100 \
  --output results.json
```

### `regression` - Detect performance regression

```bash
python -m transformation_portal.metrics.ledger regression \
  --ledger-db performance.db \
  --capsule current_capsule.json \
  --baseline-days 30
```

**Exit codes:**
- `0` - No regression
- `1` - Regression detected

### `report` - Generate performance report

```bash
python -m transformation_portal.metrics.ledger report \
  --ledger-db performance.db \
  --output report.md \
  --min-days 7
```

### `prune` - Prune old entries

```bash
python -m transformation_portal.metrics.ledger prune \
  --ledger-db performance.db \
  --days-to-keep 90
```

## Schema Stability

The `PerformanceCapsule` schema is **contract-stable** (v1.0.0).

Breaking changes require:
- Version bump
- Migration plan
- Contract test updates

See `tests/test_performance_capsule_contract.py` for enforced invariants.

## Examples

See `examples/performance_ledger_example.py` for a complete working example.

```bash
python examples/performance_ledger_example.py
```

## Integration with Quality Firewall

```python
from transformation_portal.metrics import get_bucket_for_capsule

# After creating capsule with timings
bucket = get_bucket_for_capsule(capsule)

if bucket:
    total_sec = capsule.timings["total"]

    if total_sec > bucket.p95_threshold_sec:
        capsule.firewall_status = "block"
        raise PerformanceRegressionError(
            f"Exceeded p95 threshold ({bucket.p95_threshold_sec:.2f}s)"
        )
    elif total_sec > bucket.p50_threshold_sec * 1.5:
        capsule.firewall_status = "warn"
        logger.warning(f"Significantly above p50 threshold")
    else:
        capsule.firewall_status = "pass"
```

## Performance Insights (APEX Research Workflow)

From `docs/PERFORMANCE_ANALYSIS_20260207.md`:

**Key Findings:**
- Pool scenes: 11.49s (2.38× slower than interiors) - specular highlights + reflections
- Aerial scenes: 8.11s (1.68× slower) - high-frequency texture everywhere
- Interior scenes: 4.83s (fastest) - simpler geometry, less texture

**Scene content drives performance, not just pixel count.**

**Overhead is honest:** 0.52s / 46.74s = 1.1% (excellent)

## Development

Run contract tests:

```bash
pytest -v tests/test_performance_capsule_contract.py
```

Expected: 22 tests passing, enforcing schema stability.

## Next Steps

1. **Instrument orchestrator** with phase-level timing
2. **Run baseline collection** (10× APEX workflow)
3. **Add to CI** (nightly performance regression checks)
4. **Optimize hot paths** (see `docs/PERFORMANCE_ANALYSIS_20260207.md`)

## Related Documentation

- `docs/PERFORMANCE_ANALYSIS_20260207.md` - Performance analysis and optimization roadmap
- `docs/APEX_RESEARCH_WORKFLOW_REPORT_20260207.md` - APEX workflow execution report
- `QUALITY_FIREWALL_QUICK_REF.md` - Quality Firewall bucket definitions
- `tests/test_performance_capsule_contract.py` - Schema contract tests

---

**Version:** 2.0.0
**Status:** Production-ready
**Maintainer:** Transformation Portal Architect
