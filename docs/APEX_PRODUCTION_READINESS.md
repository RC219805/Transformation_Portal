# APEX Production Readiness Status

**Status:** ✅ Scaffolding Complete | 🚧 Real Pipeline Integration Pending

**Current Version:** v1.0.0 (Dry-Run / Schema Validation)

**Last Updated:** 2026-02-08

---

## Executive Summary

APEX (Architectural Photo Enhancement eXecution) performance observability platform is **production-ready as scaffolding** but requires **real pipeline integration** before it can be used for actual regression gating.

**What works today:**
- ✅ Complete data contracts (RunSpec → Observation → Judgement)
- ✅ Schema-aware ledger with migration support (v3.0.0)
- ✅ Multi-zone aggregation and worst-zone tracking
- ✅ Regression detection with baseline comparison
- ✅ CI/CD workflows with idempotent PR comments
- ✅ Dashboard generation and GitHub Pages deployment
- ✅ Synthetic data generation for testing

**What requires real work:**
- 🚧 Integration with actual V1/V2 pipelines (currently `NotImplementedError`)
- 🚧 Real performance measurements (currently mock timings)
- 🚧 Production gate enforcement (currently shadow mode only)

---

## Current Execution Mode

### Dry-Run (Synthetic Data)

The APEX matrix runner (`scripts/apex_matrix_runner.py`) currently operates in **dry-run mode only**:

```python
# From apex_matrix_runner.py line 115
raise NotImplementedError("Actual pipeline integration not yet implemented. Use --dry-run for testing.")
```

**What this means:**
- All CI runs use `--dry-run` flag
- Performance capsules contain **mock timings** (hardcoded 10.0s total)
- Outputs are labeled **SYNTHETIC DATA** in PR comments
- Gate verdicts are **informational only** (shadow mode)

**Why this is intentional:**
- Validates end-to-end schema/plumbing correctness
- Tests ledger persistence, aggregation, and reporting
- Provides working examples for development
- **Does not mislead** about real performance

---

## Production Integration Checklist

To move from "scaffolding" to "production gating":

### 1. Real Pipeline Execution (Critical)

**File:** `scripts/apex_matrix_runner.py` (line 109-115)

**Current state:**
```python
# TODO: Integrate with actual pipeline runner
raise NotImplementedError("...")
```

**Required changes:**
```python
# V1: Lux Depth V3 pipeline
if run_spec.workflow_version == "v1":
    from transformation_portal.pipelines.lux_depth_v3 import run_pipeline
    capsules = run_pipeline(config=..., zone=zone)

# V2: New depth-aware pipeline
elif run_spec.workflow_version == "v2":
    from transformation_portal.pipelines.v2_runner import run_pipeline
    capsules = run_pipeline(config=..., zone=zone)
```

**Testing strategy:**
1. Start with single-image smoke tests per zone
2. Add batch validation (6-10 images)
3. Verify timing synchronization (GPU/MPS/CPU)
4. Validate capsule schema matches expectations

---

### 2. Remove --dry-run from CI (Critical)

**File:** `.github/workflows/apex_performance.yml` (line 59)

**Current state:**
```yaml
--dry-run  # Remove when ready for production gates
```

**Required changes:**
1. Wire real pipeline execution (step 1 above)
2. Test locally with real images
3. Run in CI on a feature branch
4. Verify PR comment shows real measurements
5. Only then: remove `--dry-run` and `--synthetic` flags

---

### 3. Enable Enforcement Mode (Optional)

**File:** `.github/workflows/apex_performance.yml` (line 150)

**Current state:**
```yaml
--mode shadow  # Informational only
```

**Required changes:**
```yaml
--mode enforce  # Block merges on failure
```

**Recommendation:**
- Keep shadow mode for at least 2 weeks after real integration
- Monitor false positive rate
- Tune thresholds (`DEFAULT_BUCKETS` in `performance_capsule.py`)
- Only enforce after stable baseline established

---

### 4. Minimum Sample Size Protection (Recommended)

**Current issue:**
Reports show `Count=1` with p95 values, which is statistically meaningless.

**File:** `src/transformation_portal/metrics/aggregator.py`

**Add validation:**
```python
MIN_SAMPLES_FOR_PERCENTILES = 20

def compute_bucket_stats(...) -> BucketStats:
    if len(total_times) < MIN_SAMPLES_FOR_PERCENTILES:
        return BucketStats(
            ...
            pass_fail="insufficient_data",  # Don't gate
            p50=None, p95=None, p99=None  # Mark as invalid
        )
```

---

### 5. Fix Percentile Calculation (Recommended)

**Current issue:**
Median uses `total_times[n // 2]` which is incorrect for even-sized samples.

**File:** `src/transformation_portal/metrics/aggregator.py` (line 76-79)

**Fix:**
```python
import statistics

p50 = statistics.median(total_times)
p95 = statistics.quantiles(total_times, n=100)[94]  # True 95th percentile
p99 = statistics.quantiles(total_times, n=100)[98]
```

Or use a proper quantile sketch library (e.g., `datasketches`, `t-digest`).

---

## Schema Correctness (Already Hardened)

The following issues identified in review have been **fixed**:

### ✅ Run-Scoped Aggregation
**Fixed in:** `scripts/apex_aggregate_ledger.py` (commit 6d707d88)

```python
# Now schema-aware: filters by run_id if column exists
if "run_id" in columns:
    cursor = conn.execute(
        "SELECT capsule_json FROM performance_capsules WHERE run_id = ?",
        (run_id,)
    )
```

### ✅ Schema-Aware Gate Enforcement
**Fixed in:** `scripts/apex_enforce_gate.py` (commit 6d707d88)

```python
# Checks for commit_sha column before filtering
cursor = conn.execute("PRAGMA table_info(apex_runs)")
columns = [row[1] for row in cursor.fetchall()]

if "commit_sha" in columns:
    # Use full filter
else:
    # Fallback for legacy schema
```

### ✅ Deterministic Rebuild
**Fixed in:** `scripts/apex_rebuild_ledger.py` (commit 6d707d88)

```bash
# Clean mode: truncate before ingest to avoid duplicates
python scripts/apex_rebuild_ledger.py --clean
```

### ✅ Synthetic Data Labeling
**Fixed in:** `scripts/apex_pr_comment.py` (commit 6d707d88)

PR comments now show prominent warning:

```markdown
> ⚠️ **SYNTHETIC DATA (DRY-RUN MODE)**
>
> This report uses mock performance capsules for testing...
```

---

## Advanced Features (Future Work)

These are **research-grade upgrades** that scale the system beyond basic gating:

### 1. Mergeable Quantile Sketches

**Why:** Compute accurate percentiles across zones without storing all samples.

**Options:**
- [KLL sketch](https://arxiv.org/abs/1603.05346) (optimal for streaming)
- [t-digest](https://github.com/tdunning/t-digest) (excellent tail accuracy)

**Benefits:**
- Constant memory per bucket
- Mergeable across zones
- Sub-millisecond percentile queries

### 2. Statistical Drift Detection

**Why:** Replace hard thresholds with principled change detection.

**Options:**
- ADWIN (Adaptive Windowing for concept drift)
- Sequential Probability Ratio Test (SPRT)
- Bayesian change point detection

**Benefits:**
- Automatically adapts to workload changes
- Reduces false positive rate
- Provides confidence intervals

### 3. Multi-Grade SLO Thresholds

**Why:** Capture both typical and tail behavior.

**Pattern (from Google SRE):**
```yaml
buckets:
  pool_large_mps:
    p90: 8.0s   # "Most requests"
    p95: 10.0s  # "Almost all requests"
    p99: 15.0s  # "Worst acceptable tail"
```

---

## Current Test Coverage

### Unit Tests
- ✅ `tests/test_apex_contracts.py` (RunSpec/Observation/Judgement)
- ✅ `tests/test_apex_aggregator.py` (Per-zone/global/worst-zone)
- ✅ `tests/test_apex_gate.py` (Enforcement modes)
- ✅ `tests/test_apex_ledger.py` (Schema migration v2→v3)
- ✅ `tests/test_timing_context.py` (GPU sync correctness)

### Integration Tests
- ✅ `tests/test_apex_dashboard.py` (Data generation + HTML)
- ✅ `tests/test_performance_capsule_contract.py` (Schema validation)

### CI Tests
- ✅ Dry-run matrix execution (V1 × V2 × zones)
- ✅ Ledger rebuild + aggregation
- ✅ PR comment generation
- ✅ Gate evaluation (shadow mode)

**Missing (intentionally):**
- ❌ Real pipeline execution tests (requires implementation)
- ❌ Large-scale stress tests (> 100 images)
- ❌ Multi-zone correctness (requires cloud infra)

---

## Decision Log

### Why keep dry-run scaffolding in main?

**Decision:** Merge APEX as scaffolding, gate on real pipelines later.

**Rationale:**
1. **Schema correctness is valuable now** (migrations, contracts)
2. **Plumbing is production-grade** (deterministic, testable)
3. **Clear labeling prevents misuse** (SYNTHETIC warnings)
4. **Integration is separable work** (can be feature-branched)

**Alternatives considered:**
- ❌ Block merge until real integration → delays schema/workflow value
- ❌ Fork into separate branch → diverges from main, harder to integrate

---

## Migration Path

When real pipeline integration lands:

1. **Feature branch integration**
   ```bash
   git checkout -b feature/apex-real-pipeline-integration
   # Implement run_apex_for_config() real execution path
   # Test locally with 6-10 real images
   ```

2. **Smoke test PR**
   - One zone, one workflow version
   - Verify real timings in ledger
   - Check PR comment shows accurate data

3. **Full matrix PR**
   - Both V1 and V2
   - Multiple zones (local, then cloud)
   - Shadow mode for 1-2 weeks

4. **Enforcement PR**
   - Switch to `--mode enforce`
   - Update docs to reflect "production gating"
   - Celebrate 🎉

---

## Contact / Questions

- **Architecture:** See `docs/adr/002-apex-performance-workflow.md`
- **Schema:** See `src/transformation_portal/metrics/contracts.py`
- **CI:** See `.github/workflows/apex_performance.yml`
- **Issues:** Tag `@transformation-portal-architect` agent

---

**Signature:** APEX Phase 3 scaffolding complete. Real integration is the remaining 20% of work that unlocks 80% of production value.
