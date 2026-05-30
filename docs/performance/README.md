# Performance Monitoring Guide

## Overview

The Transformation Portal performance ledger provides regression detection for pipeline runtime performance. It parses manifests from batch runs, computes statistics, and compares against baselines to detect performance degradation.

**Status:** Phase 2 tooling (v2.0.1+)
**Enforcement:** Manual workflow, not a CI gate

Current gate authority: [Performance Gate Policy](GATE_POLICY.md).

---

## Quick Start

### Capture Baseline

After a successful production run:

```bash
python tools/performance_ledger.py \
  --manifests-dir ./output/lux_depth_v3_prod/manifests \
  --output ./docs/performance/baselines/baseline_v2.0.0_da3_apex.json
```

### Compare Against Baseline

After an experimental run:

```bash
python tools/performance_ledger.py \
  --baseline ./docs/performance/baselines/baseline_v2.0.0_da3_apex.json \
  --compare ./output/experimental_run/manifests \
  --output ./output/perf_report.md \
  --emit-json ./output/perf_current.json
```

---

## Baseline Management

### Baseline Naming Convention

```
baseline_{version}_{backend}_{tier}.json
```

**Examples:**
- `baseline_v2.0.0_da3_apex.json` - v2.0.0 with DA3, APEX quality tier
- `baseline_v2.0.0_da3_standard.json` - v2.0.0 with DA3, standard quality
- `baseline_v2.1.0_depth_pro_experimental.json` - v2.1.0 with Depth Pro (experimental)

### Baseline Directory Structure

```
docs/performance/baselines/
├── README.md                            # This file
├── baseline_v2.0.0_da3_apex.json       # Production baseline (20 images, post-PR #841)
├── baseline_v2.0.0_da3_standard.json   # Standard tier baseline
└── archive/
    └── baseline_v1.9.0_da3.json        # Historical baselines
```

### Baseline Governance

**Update Policy:**
1. Baselines are versioned and committed to repo
2. Updates require Architect approval (PR review)
3. No automated baseline updates (prevents inflation)
4. Breaking changes must update baseline filename (new version)

**When to Update:**
- Major/minor version releases (v2.0.0 → v2.1.0)
- Backend changes (DA3 → Depth Pro)
- Quality tier changes (standard → apex)
- Hardware changes (M1 → M4, CPU → GPU)

**When NOT to Update:**
- Patch releases (v2.0.0 → v2.0.1) - unless performance fix
- CI runs (use existing baseline)
- Individual developer runs

---

## Regression Thresholds

### Default Policy

```python
REGRESSION_THRESHOLDS = {
    "p95_worsening_pct": 10,      # p95 > 10% slower = regression
    "mean_worsening_pct": 15,     # mean > 15% slower = regression
    "failure_rate_increase": 0,   # Any failures = regression
}
```

**Rationale:**
- **p95:** Captures tail latency (worst 5% of images)
- **mean:** Captures overall throughput
- **failure_rate:** Zero tolerance for new failures

### Interpretation

| Metric | Baseline | Current | Change | Regression? |
|--------|----------|---------|--------|-------------|
| p95 | 30.43s | 33.00s | +8.4% | ❌ No (< 10%) |
| p95 | 30.43s | 34.00s | +11.7% | ✅ Yes (> 10%) |
| mean | 13.92s | 15.50s | +11.3% | ❌ No (< 15%) |
| mean | 13.92s | 16.50s | +18.5% | ✅ Yes (> 15%) |
| failure_rate | 0.0% | 0.5% | +0.5% | ✅ Yes (> 0%) |

### Custom Thresholds

Override via CLI flags:

```bash
python tools/performance_ledger.py \
  --baseline baseline.json \
  --compare ./manifests \
  --p95-threshold 15 \
  --mean-threshold 20 \
  --output report.md
```

---

## Baseline Schema

```json
{
  "version": "v2.0.0",
  "backend": "depth_anything_v3",
  "quality_tier": "apex",
  "environment": {
    "python": "3.11.0",
    "torch": "2.1.0",
    "device": "mps",
    "os": "macOS-14.0-arm64",
    "cpu": "Apple M1 Max",
    "memory_gb": 64
  },
  "statistics": {
    "count": 20,
    "mean_sec": 13.92,
    "median_sec": 11.82,
    "p90_sec": 28.50,
    "p95_sec": 30.43,
    "min_sec": 5.21,
    "max_sec": 30.83,
    "success_rate": 1.0,
    "total_sec": 278.4,
    "overhead_sec": 0.62
  },
  "captured_at": "2026-02-04T06:55:48Z",
  "captured_by": "tools/performance_ledger.py v1.0",
  "notes": "Post-PR #841 production validation, 750_Picacho_Kitchen dataset"
}
```

---

## Output Formats

### Markdown Report

```markdown
# Performance Comparison Report

**Baseline:** v2.0.0 (DA3, APEX)
**Current:** experimental_run (20 images)
**Environment:** macOS-14.0-arm64, Python 3.11.0, torch 2.1.0, device=mps

## Statistics

| Metric | Baseline | Current | Change | Status |
|--------|----------|---------|--------|--------|
| Mean | 13.92s | 15.50s | +11.3% | ✅ OK |
| Median | 11.82s | 12.10s | +2.4% | ✅ OK |
| p90 | 28.50s | 29.00s | +1.8% | ✅ OK |
| p95 | 30.43s | 34.00s | +11.7% | ⚠️ REGRESSION |
| Min | 5.21s | 5.50s | +5.6% | ✅ OK |
| Max | 30.83s | 35.00s | +13.5% | ✅ OK |
| Success Rate | 100.0% | 100.0% | 0.0% | ✅ OK |

## Regressions Detected

⚠️ **p95 regression:** 30.43s → 34.00s (+11.7%, threshold 10.0%)

## Recommendation

**DO NOT MERGE** - Performance regression detected in p95 (tail latency).
Investigate slowdown before merging changes.
```

### JSON Output

```json
{
  "baseline_id": "v2.0.0_da3_apex",
  "current_run": {
    "count": 20,
    "mean_sec": 15.50,
    "median_sec": 12.10,
    "p90_sec": 29.00,
    "p95_sec": 34.00,
    "min_sec": 5.50,
    "max_sec": 35.00,
    "success_rate": 1.0
  },
  "changes": {
    "mean_pct": 11.3,
    "median_pct": 2.4,
    "p95_pct": 11.7
  },
  "regressions": [
    {
      "metric": "p95_sec",
      "baseline": 30.43,
      "current": 34.00,
      "change_pct": 11.7,
      "threshold_pct": 10.0,
      "status": "regression"
    }
  ],
  "verdict": "regression_detected"
}
```

---

## Usage Examples

### Scenario 1: Capture Initial Baseline

```bash
# Run production batch
lux-depth-v3 \
  --input-dir ./input_images \
  --output-dir ./output/baseline_run \
  --quality-tier apex

# Capture baseline
python tools/performance_ledger.py \
  --manifests-dir ./output/baseline_run/manifests \
  --output ./docs/performance/baselines/baseline_v2.0.0_da3_apex.json

# Commit baseline
git add docs/performance/baselines/baseline_v2.0.0_da3_apex.json
git commit -m "perf: capture v2.0.0 DA3 APEX baseline (20 images)"
```

### Scenario 2: Pre-Merge Regression Check

```bash
# On feature branch, run experimental batch
lux-depth-v3 \
  --input-dir ./input_images \
  --output-dir ./output/feature_test \
  --quality-tier apex

# Compare against baseline
python tools/performance_ledger.py \
  --baseline ./docs/performance/baselines/baseline_v2.0.0_da3_apex.json \
  --compare ./output/feature_test/manifests \
  --output ./output/perf_check.md

# Review report
cat ./output/perf_check.md

# If OK, merge. If regression, investigate.
```

### Scenario 3: Nightly Performance Tracking

```bash
# In CI nightly workflow
python tools/performance_ledger.py \
  --baseline ./docs/performance/baselines/baseline_v2.0.0_da3_apex.json \
  --compare ./output/nightly_run/manifests \
  --output ./output/nightly_perf_report.md \
  --emit-json ./output/nightly_perf.json

# Post report to Slack/GitHub
# (not a gate, just visibility)
```

---

## Troubleshooting

### "No manifests found"

**Cause:** Manifests directory is empty or path is wrong.

**Solution:**
```bash
# Verify manifests exist
ls -la ./output/your_run/manifests/

# Check for JSON files
find ./output/your_run/manifests/ -name "*.json"

# Ensure manifests were written during batch run
# (orchestrator writes one manifest per image)
```

### "Baseline schema validation failed"

**Cause:** Baseline JSON is corrupted or has wrong schema.

**Solution:**
```bash
# Validate baseline JSON syntax
python -m json.tool < baseline.json

# Re-capture baseline if corrupted
python tools/performance_ledger.py \
  --manifests-dir ./output/known_good_run/manifests \
  --output ./baseline_fixed.json
```

### "Environment mismatch warning"

**Cause:** Baseline and current run have different environments (Python version, torch version, device).

**Example:**
```
WARNING: Environment mismatch detected:
  Baseline: Python 3.11.0, torch 2.1.0, device=mps
  Current:  Python 3.12.0, torch 2.2.0, device=mps

  Comparison may not be valid. Consider using environment-specific baseline.
```

**Solution:**
- Use environment-specific baselines (e.g., `baseline_v2.0.0_da3_apex_py312.json`)
- Accept warning if intentional upgrade
- Ignore if investigating environment-specific performance

---

## Future Enhancements

Potential improvements for v2.1.0+:

- **CI Integration:** Optional workflow to run ledger on nightly builds
- **Historical Tracking:** Trend analysis across multiple runs
- **Per-Image Analysis:** Identify which images regressed
- **Slack/GitHub Notifications:** Auto-post reports
- **HTML Dashboard:** Visual comparison charts
- **Multi-Baseline Comparison:** Compare against multiple baselines (v2.0.0 vs v2.1.0)

---

## References

- [ADR-023: Post-PR #841 Hardening Strategy](../architecture/decisions/ADR-023-post-pr841-hardening.md)
- [Performance Ledger Implementation](../../tools/performance_ledger.py)
- [Manifest Schema Documentation](../../src/transformation_portal/lux_depth_v3/manifest.py)

---

**Last Updated:** 2026-02-05
**Maintainer:** Transformation Portal Architect
