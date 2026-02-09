# APEX Performance Contract v1.0.0

**Status:** Official Contract
**Last Updated:** 2026-02-08
**Binding Version:** v1.0.0 (APEX scaffolding baseline)

---

## Purpose

This document defines the **formal contract** for the APEX (Architectural Photo Enhancement eXecution) performance observability platform.

APEX is not a dashboard. APEX is a **judge** that determines whether performance changes are acceptable.

---

## Contract Boundary

### Inputs (What APEX Consumes)

| Input | Format | Authority | Notes |
|-------|--------|-----------|-------|
| **RunSpec** | Immutable dataclass | CI workflow | Declares intent (workflow version, zones, commit SHA) |
| **Performance Capsules** | Serialized JSON | Pipeline execution | Contains measured timings + metadata |
| **Bucket Definitions** | YAML | `performance_capsule.py::DEFAULT_BUCKETS` | Scene/device/backend categorization |
| **Baseline Data** | SQLite query | Performance ledger | Historical runs for regression comparison |

### Outputs (What APEX Produces)

| Output | Format | Consumer | Binding Contract |
|--------|--------|----------|------------------|
| **Judgement** | Dataclass | Gate enforcement | PASS / WARN / FAIL verdict |
| **RegressionReport** | Dataclass | PR comment generator | Delta vs baseline with explanations |
| **BucketStats** | Dataclass | Dashboard | Per-bucket p50/p95/p99 + thresholds |
| **PR Comment** | Markdown | GitHub PR UI | Human-readable verdict + heatmap |

---

## Verdicts (The Decision Layer)

APEX produces exactly **three verdicts**:

### 1. **PASS** ✅

**Definition:** All tested buckets meet their p95 thresholds AND no bucket regresses > 10% vs baseline.

**Contract:**
- Merge is **allowed** (in enforce mode)
- No blocking comments
- Green checkmark in PR

**Example:**
```
V2 Gate: PASSED ✅
All buckets within limits. Max regression: +2.3%
```

---

### 2. **WARN** ⚠️

**Definition:** At least one bucket exceeds its p95 threshold by ≤15% OR regression vs baseline is 10-15%.

**Contract:**
- Merge is **allowed** (informational only)
- Warning icon in PR comment
- Explanation required in comment

**Example:**
```
V2 Gate: WARNING ⚠️
Bucket 'pool_large_mps' p95: 11.2s (limit: 10.0s, +12%)
Consider investigating before merge.
```

---

### 3. **FAIL** ❌

**Definition:** At least one bucket exceeds its p95 threshold by >15% OR regression vs baseline is >15%.

**Contract:**
- Merge is **blocked** (in enforce mode)
- Red X in PR comment
- Must fix before merge (or override with justification)

**Example:**
```
V2 Gate: FAILED ❌
Bucket 'aerial_large_mps' p95: 14.5s (limit: 10.0s, +45%)
BLOCKING: Performance regression detected.
```

---

## Threshold Contract

### Bucket Thresholds (Stable API)

Thresholds are defined in `DEFAULT_BUCKETS` and versioned alongside the schema.

**Contract guarantees:**
- Threshold changes require explicit version bump
- No retroactive threshold changes
- Baseline comparisons use **frozen thresholds from baseline run**

**Current thresholds (v1.0.0):**

| Bucket Name | Scene | Device | p95 Threshold | p50 Threshold |
|-------------|-------|--------|---------------|---------------|
| `pool_large_mps` | Pool | MPS | 10.0s | 6.0s |
| `aerial_large_mps` | Aerial | MPS | 10.0s | 6.0s |
| `interior_medium_mps` | Interior | MPS | 8.0s | 5.0s |
| `generic_large` | Any | Any | 15.0s | 10.0s |

---

## Statistical Validity Requirements

### Minimum Sample Size

**Contract:** Percentiles (p50/p95/p99) require **≥20 samples** per bucket.

**Enforcement:**
- If `count < 20`: verdict is `insufficient_data`
- `insufficient_data` never blocks (treated as shadow mode)
- PR comment displays: "⚠️ Insufficient data (n=X)"

**Rationale:** With n < 20, percentiles are not statistically meaningful.

---

### Percentile Calculation Method

**Contract:** Use **true percentiles** (not index-based approximations).

**Guaranteed behavior:**
- p50 (median): Properly handles even-sized samples (average of two middle values)
- p95: Linear interpolation between 94th and 95th percentile positions
- p99: Linear interpolation between 98th and 99th percentile positions

**Reference implementation:** `src/transformation_portal/metrics/aggregator.py::compute_bucket_stats()`

---

## Regression Detection

### Baseline Selection

**Contract:** Baseline is the **most recent successful run** for the same `(workflow_version, zone, bucket_name)` tuple.

**Query logic:**
```sql
SELECT p95 FROM apex_runs
WHERE workflow_version = ?
  AND zone = ?
  AND bucket_name = ?
  AND pass_fail = 'pass'
ORDER BY timestamp DESC
LIMIT 1
```

**Fallback:** If no baseline exists, regression comparison is skipped (verdict based on thresholds only).

---

### Regression Tolerance

**Contract:**

| Regression Delta | Verdict |
|------------------|---------|
| ≤ 10% | PASS |
| 10-15% | WARN |
| > 15% | FAIL |

**Example:**
- Baseline p95: 10.0s
- Current p95: 11.0s
- Delta: +10% → **WARN**

---

## Zone Awareness

### Worst-Zone Semantics

**Contract:** The **worst-zone p95** is the maximum p95 across all tested zones for a given bucket.

**Example:**
```
Bucket: pool_large_mps
- Zone "us-west-2a": p95 = 9.5s
- Zone "us-east-1a": p95 = 12.3s
- Worst-zone p95: 12.3s (used for gating)
```

**Rationale:** User experience is defined by the slowest zone.

---

### Global Rollup

**Contract:** Global stats are **weighted averages** across zones (not a separate run).

**Not currently implemented** (placeholder for Phase 4).

---

## Schema Stability

### Backward Compatibility

**Contract:** APEX must support **schema v2 and v3** during migration.

**Enforcement:**
- All queries use `PRAGMA table_info()` to detect columns
- Missing columns trigger graceful fallback (not crashes)
- Tests validate migration path explicitly

**Breaking change policy:**
- Schema v4+ requires ADR approval
- Deprecated schemas supported for ≥ 2 release cycles

---

## Data Retention

**Contract:**

| Data Type | Retention | Storage |
|-----------|-----------|---------|
| Raw capsules | 90 days | SQLite ledger |
| Aggregated stats | 1 year | SQLite ledger |
| Weekly backups | Indefinite | GitHub Releases |

---

## Gate Modes

APEX supports **three enforcement modes**:

### 1. **Enforce Mode**

**Contract:**
- FAIL verdicts **block PR merge**
- CI step exits with non-zero code
- GitHub required check fails

**Usage:**
```bash
python scripts/apex_enforce_gate.py --mode enforce
```

---

### 2. **Shadow Mode** (Default)

**Contract:**
- Verdicts are **informational only**
- CI step always succeeds
- PR comment posted, but no blocking

**Usage:**
```bash
python scripts/apex_enforce_gate.py --mode shadow
```

---

### 3. **Disabled Mode**

**Contract:**
- Gate logic **not executed**
- CI step always succeeds
- No PR comment posted

**Usage:**
```bash
python scripts/apex_enforce_gate.py --mode disabled
```

---

## Version Contract

**APEX Contract Version:** v1.0.0
**Schema Version:** v3.0.0
**Baseline Compatibility:** v2.0.0+

**Contract changes require:**
1. Version bump in this document
2. Migration guide in CHANGELOG
3. Contract tests updated

---

## Prohibited Behaviors

APEX **must never**:

1. ❌ Silently ignore data (log + fail instead)
2. ❌ Change thresholds retroactively
3. ❌ Mix data from different commits
4. ❌ Gate on <20 samples
5. ❌ Change verdicts based on who authored the PR

---

## Emergency Override

**Contract:** Gate enforcement can be overridden via:

```yaml
# In PR description:
APEX-OVERRIDE: Performance regression justified by <reason>
```

**Requirements:**
- Must include justification
- Requires maintainer approval
- Logged in ledger as `override=true`

---

## Audit Trail

**Contract:** Every gate decision is recorded in:

1. **Performance ledger** (`apex_runs` table)
2. **PR comment** (Markdown)
3. **GitHub Actions logs**

**Retention:** 90 days minimum (per data retention policy)

---

## Signature

This contract is **binding** for all APEX operations.

Changes to this contract require:
- ADR approval
- Consensus from 2+ maintainers
- Version bump

**Contract Authority:** Transformation Portal Governance  
**Effective Date:** 2026-02-08  
**Next Review:** 2026-05-08 (quarterly)

---

## Related Governance Documents

- **[APEX Governance Framework (ADR-026)](../architecture/decisions/ADR-026-APEX-governance-framework.md)** - Architectural design for policy-as-code
- **[Governance User Guide](GOVERNANCE_USER_GUIDE.md)** - How to interact with APEX governance (waivers, budget changes, incidents)
- **[Governance Implementation Summary](GOVERNANCE_IMPLEMENTATION_SUMMARY.md)** - Implementation status and next steps
- **[Performance Budgets Policy](policy/performance_budgets.yaml)** - Performance thresholds (versioned)
- **[Enforcement Policy](policy/enforcement_policy.yaml)** - Statistical methods and evidence gates
- **[Governance Rules](policy/governance_rules.yaml)** - Waivers, incidents, budget evolution
- **[Workload Suites](policy/workload_suites.yaml)** - Canonical test workloads (Golden/Canary/Fuzz)
