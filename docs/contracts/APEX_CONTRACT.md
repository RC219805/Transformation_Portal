# APEX Performance Contract v1.0.0

**Status:** Enforced
**Effective Date:** 2026-02-08
**Review Cycle:** Quarterly (Next: 2026-05-08)

---

## Purpose

This document defines the **formal performance contract** for APEX (Architectural Photo Enhancement eXecution). It specifies what APEX measures, how it judges performance, and what actions it takes based on those judgments.

---

## Contract Layers

### Layer 1: Intent (RunSpec)

What we declare we will test:
- `run_id`: Unique run identifier
- `commit_sha`: Exact code version
- `workflow_version`: `"v1"` or `"v2"`
- `zones`: Execution zones (list)
- `device`: `"mps"`, `"cuda"`, or `"cpu"`
- `backend_id`: Depth backend (`"da3"`, `"depth_pro"`, etc.)

**Immutability:** RunSpec is frozen after declaration.

---

### Layer 2: Reality (Observation)

What actually happened during execution:
- `capsules`: List of `PerformanceCapsule` objects
- `zone`: Where the observation was captured
- `errors`: List of error messages (if any)
- `captured_at`: ISO8601 timestamp

**Integrity:** Each observation is scoped to a single `(run_id, commit_sha, zone)`.

---

### Layer 3: Judgement

What we do with the measurements:

#### Verdict Domain

**Valid verdicts:**
- `"pass"`: Performance within thresholds
- `"warn"`: Performance degraded but acceptable
- `"fail"`: Performance violation requiring action

**Sample sufficiency:**
- If `count < 20`: verdict is `"pass"` with metadata `is_insufficient_data: true`
- Never block on insufficient data
- Render as 📊 INSUFFICIENT DATA in PR comments

#### Aggregation

**Per-zone statistics (BucketStats):**
- `count`: Number of samples
- `p50`, `p95`, `p99`: Percentiles (index-based, not interpolated)
- `mean`, `min`, `max`: Distribution bounds
- `threshold_p50`, `threshold_p95`: Configured limits
- `pass_fail`: Verdict string

**Global statistics:**
- Computed across all zones
- `worst_zone_p95`: Maximum p95 across zones
- `worst_zone_name`: Zone with maximum p95

---

## Enforcement Modes

### 1. `shadow` (Current Default)

- ✅ Captures measurements
- ✅ Writes ledger
- ✅ Posts PR comment
- ❌ Does not block CI

**Use:** Validation, tuning, baseline collection

---

### 2. `enforce`

- ✅ All `shadow` behaviors
- ✅ Blocks CI on `"fail"` verdict
- ❌ Only if `count >= 20` per bucket

**Use:** Production gating

---

### 3. `disabled`

- ❌ No measurements
- ❌ No ledger writes
- ❌ No PR comment

**Use:** Temporary bypass during incidents

---

## Data Integrity Rules

### Scoping

All aggregation queries **MUST** filter by:
```sql
WHERE run_id = ? AND commit_sha = ?
```

**Rationale:** Prevents contamination from prior runs.

---

### Synthetic Data Isolation

**Current state (v1.0.0):**
- Synthetic data marked via metadata in JSON
- PR comments labeled `[SYNTHETIC DATA]`
- Baseline comparisons excluded (future)

**Future considerations:**
- Add `is_synthetic` column to ledger schema
- Default filter: `WHERE is_synthetic = 0`
- Require explicit flag to include synthetic data

---

## Schema Stability

### Current Version: v3.0.0

**Tables:**
- `performance_capsules`: Raw observations
- `apex_runs`: Aggregated bucket stats (one row per bucket per zone)

**Migration policy:**
- Breaking changes require version bump
- Backward-compatible changes allowed within major version
- Schema introspection (`PRAGMA table_info`) required for portability

---

## Percentile Computation

**Current implementation (v1.0.0):**
- Index-based selection (not interpolated)
- `p50 = sorted_values[n // 2]`
- `p95 = sorted_values[int(n * 0.95)]`
- `p99 = sorted_values[int(n * 0.99)]`

**Future enhancement:**
- Interpolated percentiles for even sample counts
- True median for p50 (already implemented in aggregator)

---

## PR Comment Contract

### Verdict Banner

```
✅ APEX Performance Verdict: PASS
⚠️ APEX Performance Verdict: WARN
❌ APEX Performance Verdict: FAIL
```

### Required Sections

1. **Metadata:** `run_id`, `commit_sha`, mode
2. **V1 vs V2 Comparison Table**
3. **Per-workflow Performance Tables**
4. **Zone Heatmap** (collapsed)
5. **Worst Offenders** (if any)
6. **Synthetic Data Label** (if applicable)

---

## Minimum Sample Size Protection

**Rule:** `n < 20` produces non-blocking verdict.

**Implementation:**
```python
if count < 20:
    return BucketStats(
        ...,
        pass_fail="pass",
        # Metadata stored separately or in ledger JSON
    )
```

**Rendering:**
```
📊 INSUFFICIENT DATA (n=3, need 20)
```

---

## Real Pipeline Integration

**Current state:**
- Dry-run mode only (`--dry-run` flag)
- Generates synthetic capsules
- PR comments labeled `[SYNTHETIC DATA]`

**Requirements for production:**
- [ ] Wire matrix runner to real orchestrator
- [ ] Remove `--dry-run` default from CI
- [ ] Validate SHA attribution
- [ ] Collect 1-2 weeks shadow data
- [ ] Tune thresholds based on reality

**Tracked in:** `docs/APEX_REAL_PIPELINE_INTEGRATION.md`

---

## Definitions of Done

### PR Merge Readiness

- [x] Contract document exists (this file)
- [x] Verdict semantics enforced (`pass|warn|fail` only)
- [x] Minimum sample size protection active
- [x] Aggregation scoped by `run_id` + `commit_sha`
- [x] Synthetic data labeled in PR comments
- [ ] CI green on latest commit
- [ ] PR comment observed with correct labels

### Production Readiness

- [ ] Real pipeline wired
- [ ] Shadow mode data collected (2+ weeks)
- [ ] Thresholds tuned to reality
- [ ] False positive rate < 5%
- [ ] Rollback plan documented

---

## Governance

### Contract Changes

**Requires:**
- ADR documenting rationale
- Migration plan for breaking changes
- Test coverage update
- Version bump

**Approval:** Maintainer + 1 reviewer

### Threshold Changes

**Requires:**
- Explanation in PR
- Data justifying change
- No approval for < 10% adjustment
- Approval for >= 10% adjustment

### Mode Changes

**Shadow → Enforce:**
- Requires 2 weeks clean shadow data
- Sign-off from team lead

**Enforce → Disabled:**
- Requires incident justification
- Time-boxed (< 24 hours)

---

## Version History

| Version | Date       | Changes                                      |
|---------|------------|----------------------------------------------|
| 1.0.0   | 2026-02-08 | Initial contract (shadow mode, synthetic OK) |

---

## References

- **Architecture:** `docs/APEX_ARCHITECTURE.md`
- **Integration Plan:** `docs/APEX_REAL_PIPELINE_INTEGRATION.md`
- **Quality Firewall:** `QUALITY_FIREWALL_QUICK_REF.md`
- **Performance Ledger:** `docs/PERFORMANCE_LEDGER_README.md`

---

**Signed (Code Enforcement):**
- `scripts/apex_enforce_gate.py` (mode handling)
- `src/transformation_portal/metrics/aggregator.py` (min sample check)
- `scripts/apex_aggregate_ledger.py` (scoping enforcement)

**Next Review:** 2026-05-08
