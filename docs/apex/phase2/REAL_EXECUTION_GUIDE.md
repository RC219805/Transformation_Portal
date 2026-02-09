# APEX Real Execution Guide

## Overview

APEX now supports **hybrid execution modes**:

* **Synthetic mode** (PR/push lane): Fast, deterministic validation without ML dependencies
* **Real mode** (manual/nightly): Actual pipeline execution with torch/transformers/models

This guide explains when to use each mode and how to interpret results.

---

## Execution Modes

### Synthetic Mode (Default for PRs)

**When**: Automatically used for all PR and push events

**Characteristics**:
* No ML dependencies required (fast CI)
* Validates schema, contracts, aggregation logic
* Produces deterministic mock data (10.0s timings)
* PR comments clearly labeled `[SYNTHETIC DATA]`

**Purpose**: Catch contract regressions, schema drift, and workflow logic errors without the cost/variance of real ML execution.

**Trade-off**: Does not detect performance regressions in actual inference.

### Real Mode (Manual Testing & Nightly Monitoring)

**When**:
* Manual trigger via `workflow_dispatch` with `mode=real`
* Scheduled nightly runs (Sundays 00:00 UTC)

**Characteristics**:
* Requires ML dependencies (`pip install -e ".[ml]"` ~5GB)
* Uses actual orchestrator/depth backends/enhancement pipeline
* Real timing measurements (variable, environment-dependent)
* Produces non-synthetic performance capsules

**Purpose**: Gather real performance data, detect actual regressions, calibrate thresholds.

**Trade-off**: Slower (~2-5 min for 3 images), requires ML infrastructure, introduces noise.

---

## How to Use

### Manual Real Run (workflow_dispatch)

1. Go to: **Actions → APEX Performance Matrix → Run workflow**
2. Configure inputs:
   * **mode**: `real`
   * **backend_id**: `da3` (or `depth_pro`, `mock`)
   * **sample_size**: `3` (number of test images)
   * **device**: `cpu` (or `cuda` if GPU runner available)
3. Click "Run workflow"
4. Monitor job logs for real timing data
5. Check PR comment / artifacts for ledger DB

### Nightly Real Run (automatic)

* Runs every Sunday at midnight UTC
* Uses `mode=real` automatically
* Default: `backend_id=da3`, `sample_size=3`, `device=cpu`
* Results archived in workflow artifacts (90 days retention)

### Local Real Run (development)

```bash
python scripts/apex_matrix_runner.py \
  --run-id "local-$(date +%s)" \
  --commit-sha "$(git rev-parse HEAD)" \
  --workflow-versions v1 v2 \
  --zones local \
  --input-dir ./tests/fixtures/apex_images \
  --sample-size 3 \
  --output-dir ./apex_results_local \
  --ledger-db ./apex_local.db \
  --backend-id da3 \
  --device cpu
```

**Prerequisites**:
```bash
pip install -e ".[ml]"  # Install torch + transformers + backend deps
```

---

## Interpreting Results

### Synthetic Mode Results

Example PR comment:
```
🎯 APEX Performance Report [SYNTHETIC DATA]

⚠️ This report uses mock data (dry-run mode)
Real pipeline integration tracked in docs/apex/phase2/REAL_PIPELINE_INTEGRATION.md

✅ APEX Performance Verdict: PASS
```

**Key indicators**:
* `[SYNTHETIC DATA]` banner at top
* All timings = exactly 10.0s
* Status = informational only (never blocks)

**What to check**:
* Schema validation (no errors during aggregation)
* PR comment renders correctly
* Ledger DB structure is valid

### Real Mode Results

Example (manual/nightly run):
```
🎯 APEX Performance Report

✅ APEX Performance Verdict: PASS (mode: shadow)

V1 vs V2 Comparison
Bucket          V1 p95   V2 p95   Delta     Status
generic_large   8.23s    9.45s    +14.8%    ⚠️ Near threshold
unknown         2.11s    2.09s    -0.9%     ✅ Improved
```

**Key indicators**:
* No `[SYNTHETIC DATA]` banner
* Variable timings (realistic distributions)
* Actual backend/device used shown in metadata

**What to check**:
* p95 latencies within expected ranges (e.g., <15s for generic_large on CPU)
* No sudden spikes or anomalies
* Timing distributions look plausible (not all identical)

---

## Phase Roadmap

### Phase 2 (Current): Hybrid CI

* ✅ Real execution path implemented
* ✅ Hybrid mode (PR=synthetic, manual/nightly=real)
* ✅ Backend-aware dependency validation
* ⏳ Nightly data collection (2-4 weeks)

### Phase 3 (Next): Calibration

* Analyze nightly data distributions
* Set conservative thresholds (p95 + 20% margin)
* Document variance/noise characteristics
* Prepare enforcement criteria

### Phase 4 (Future): Enforcement

* Switch from shadow → enforce mode
* Block merges on performance regressions
* Require minimum sample size (n≥20 for enforcement)
* Continuous threshold tuning

---

## Troubleshooting

### "Backend requires ML dependencies" error

**Symptom**: `RuntimeError: Backend 'da3' requires ML dependencies: transformers`

**Fix**:
```bash
pip install -e ".[ml]"
```

**Why**: Real mode requires torch + backend-specific packages. Synthetic mode does not.

### "No images found" error

**Symptom**: `ValueError: No images found in ./tests/fixtures/apex_images`

**Fix**: Verify test fixtures exist:
```bash
ls tests/fixtures/apex_images/
# Should show: apex_test_aerial.jpg, apex_test_interior.jpg, apex_test_pool.jpg
```

### Slow CI / timeout

**Symptom**: Job exceeds 30min timeout

**Context**: This should only happen in real mode. PR/push lanes use synthetic mode and complete in ~1-2 minutes.

**Fix (if real mode)**:
* Reduce `--sample-size` (default=3)
* Optimize backend initialization
* Use faster device if available

---

## Best Practices

### For Reviewers

* **PR lane (synthetic)**: Expect `[SYNTHETIC DATA]` label. Check contract/schema correctness, not performance.
* **Real runs (manual/nightly)**: Look for anomalies, not absolute numbers. Noise is expected.

### For Contributors

* **Don't disable synthetic mode in PR lane**: It keeps CI fast and deterministic.
* **Use manual real runs sparingly**: They're for validation/debugging, not routine checks.
* **Report performance anomalies**: If nightly shows sudden spike, create an issue.

### For APEX Maintainers

* **Monitor nightly trends**: Track p95 distributions over weeks.
* **Tune thresholds conservatively**: Add margin for variance (e.g., +20% safety).
* **Document noise sources**: Runner contention, model load variance, etc.

---

## FAQ

**Q: Why not run real mode on every PR?**

A: Cost + variance. ML deps are ~5GB, runs take 2-5 min per matrix cell, and performance can vary 10-15% on shared runners. Synthetic mode validates contracts in <2 min with zero variance.

**Q: How accurate is synthetic mode?**

A: For contracts/schema: 100%. For performance: 0% (it's mock data). Use real mode to measure actual performance.

**Q: Can I run real mode locally without ML deps?**

A: No. Real mode requires `pip install -e ".[ml]"`. Use `--dry-run` if you want to test without ML.

**Q: What backends are supported?**

A:
* `da3`: Depth Anything V3 (default, requires transformers)
* `depth_pro`: Apple Depth Pro (requires depth_pro package, research license)
* `mock`: No-op backend (testing only)

**Q: How do I add more test images?**

A: Place images in `tests/fixtures/apex_images/`. Keep them small (<1MB) and non-sensitive. Commit to repo.

---

## Related Documentation

* [APEX Production Readiness](../APEX_PRODUCTION_READINESS.md)
* [Performance Contract](../APEX_CONTRACT.md)
* [Phase 1.1 Truth Alignment](../phase1/PHASE1.1_TRUTH_ALIGNMENT.md)
* [Phase 3 Backend-Aware Deps](../../issues/875)
* [Tier 1 Registry API](../tier1/REGISTRY_API_MIGRATION.md)
