# Performance Baselines

This directory contains versioned performance baselines for regression detection.

## Directory Structure

```
baselines/
├── README.md                            # This file
├── v2.0.0-post-pr841.json              # Production baseline (captured 2026-02-05)
└── archive/                             # Historical baselines
    └── (future archives)
```

## Baseline Governance

**Update Policy:**
- Baselines require Architect approval (PR review)
- No automated baseline updates
- Major/minor version changes require new baseline
- Patch versions inherit baseline unless performance fix

**Current Baselines:**

| File | Version | Backend | Tier | Images | Environment | Status |
|------|---------|---------|------|--------|-------------|--------|
| `v2.0.0-post-pr841.json` | v2.0.0 | DA3 | standard | 20 | macOS M1, Python 3.11, MPS | ✅ Active |

## Active Baseline: v2.0.0-post-pr841

**Created:** 2026-02-05  
**Environment:**
- OS: macOS 14.2 (Darwin 25.2.0, arm64)
- Python: 3.11.14
- PyTorch: 2.10.0
- Device: Apple MPS (M-series GPU)

**Dataset:**
- 20 images from `input_images/750_picacho`
- Mix of JPEG and TIFF formats
- Representative production workload

**Statistics:**
- Count: 20 images
- Mean: 13.89s per image
- Median: 11.82s
- p90: 22.05s
- p95: 30.43s
- Success rate: 100%

**Model Configuration:**
- Backend: Depth Anything V3 (DA3)
- Model: depth-anything-v3-metric-large
- Quality tier: standard

**Notes:**
- Baseline captured after PR #841 (input hygiene improvements)
- Represents production performance on Apple Silicon

## Capture New Baseline

```bash
# Run production batch
lux-depth-v3 \
  --input-dir ./input_images \
  --output-dir ./output/baseline_run \
  --quality-tier standard

# Capture baseline
python tools/performance_ledger.py \
  --manifests-dir ./output/baseline_run/manifests \
  --output ./docs/performance/baselines/v2.1.0-baseline.json \
  --version "v2.1.0" \
  --backend "da3" \
  --quality-tier "standard"

# Commit with approval
git add docs/performance/baselines/v2.1.0-baseline.json
git commit -m "perf: capture v2.1.0 DA3 baseline"
# Create PR for Architect review
```

## Regression Thresholds

Per ADR-023, the following thresholds trigger regression alerts:

- **p95 > 10% worse:** Tail latency regression
- **mean > 15% worse:** Average performance regression
- **failure_rate > 0%:** Any new failures

## Usage

**Compare against baseline:**
```bash
python tools/performance_ledger.py \
  --baseline docs/performance/baselines/v2.0.0-post-pr841.json \
  --compare output/my_run/manifests \
  --output perf_report.md
```

**Exit codes:**
- 0: No regressions detected
- 1: Regressions detected (CI should fail)

## Reference

See [Performance Monitoring Guide](../README.md) for usage instructions.
