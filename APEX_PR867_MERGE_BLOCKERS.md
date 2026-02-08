# PR #867 Merge Blockers - Resolution Plan

**Status:** Blocking Issues Identified
**Date:** 2026-02-08
**Reviewer:** System Architecture Review

---

## Executive Summary

PR #867 contains **excellent foundation work** (APEX Contract v1.0.0, comprehensive docs), but has **3 critical merge blockers** that must be resolved to avoid shipping "production theater."

**Verdict:** **REQUEST CHANGES** (fix 3 blockers → instant approval)

---

## Critical Blocker #1: Contradictory "Complete" Claims

### Issue

Documentation claims:
- ✅ "APEX End-to-End Workflow: COMPLETE"
- ✅ "Blocking issues: 0"

But `scripts/apex_matrix_runner.py` Line 115:
```python
raise NotImplementedError("Actual pipeline integration not yet implemented. Use --dry-run for testing.")
```

**Impact:** Future contributors/reviewers will assume enforcement is real when it's synthetic.

### Fix Required

**Option A (Recommended for scaffolding merge):**

1. **Update all "complete" language:**
   ```diff
   - APEX End-to-End Workflow: COMPLETE
   + APEX Governance Framework: COMPLETE (scaffolding)
   + Real Pipeline Integration: PENDING (see APEX_REAL_PIPELINE_INTEGRATION.md)
   ```

2. **Add upfront validation in `apex_matrix_runner.py`:**
   ```python
   # After args = parser.parse_args()
   if not args.dry_run:
       logger.error("❌ Real pipeline integration not yet implemented")
       logger.error("   Use --dry-run to test APEX scaffolding")
       logger.error("   Track progress: docs/APEX_REAL_PIPELINE_INTEGRATION.md")
       return 1
   ```

3. **Label all PR comments as SYNTHETIC:**
   In `scripts/apex_pr_comment.py`, line ~60:
   ```python
   lines.append("# 🎯 APEX Performance Report [SYNTHETIC DATA]")
   lines.append("")
   lines.append("> ⚠️ **This report uses mock data (dry-run mode)**")
   lines.append("> Real pipeline integration tracked in docs/APEX_REAL_PIPELINE_INTEGRATION.md")
   lines.append("")
   ```

**Option B (Production merge):**
- Implement real pipeline wiring per `APEX_REAL_PIPELINE_INTEGRATION.md`
- Remove `--dry-run` from CI
- Update docs to claim "complete"

---

## Critical Blocker #2: Minimum Sample Size Not Enforced

### Issue

APEX Contract v1.0.0 (Line 122) states:
```
**Contract:** Percentiles (p50/p95/p99) require ≥20 samples per bucket.
```

But current PR comment shows:
```
Count: 1 | p95: 10.00s ✅
```

**Impact:** Verdicts on n=1 are statistically meaningless and can produce false confidence or noise.

### Fix Required

**In `src/transformation_portal/metrics/aggregator.py::compute_bucket_stats()`:**

```python
def compute_bucket_stats(
    bucket: PerformanceBucket,
    capsules: List[PerformanceCapsule],
    min_samples: int = 20,  # Contract-mandated minimum
) -> BucketStats:
    """Compute statistics for a performance bucket.

    Args:
        bucket: Performance bucket definition
        capsules: List of performance capsules
        min_samples: Minimum samples required for valid statistics

    Returns:
        BucketStats with pass_fail = 'insufficient_data' if count < min_samples
    """
    # ... existing filtering logic ...

    n = len(total_times)

    # Enforce contract: n < 20 → insufficient_data
    if n < min_samples:
        return BucketStats(
            bucket_name=bucket.name,
            count=n,
            p50=0.0,
            p95=0.0,
            p99=0.0,
            mean=0.0,
            min=0.0,
            max=0.0,
            threshold_p50=bucket.threshold_p50,
            threshold_p95=bucket.threshold_p95,
            pass_fail="insufficient_data",  # Never blocks
        )

    # ... rest of existing logic ...
```

**In `scripts/apex_pr_comment.py`:**

Update status icon logic:
```python
def get_status_icon(status: str) -> str:
    """Get status icon for pass/warn/fail."""
    return {
        "pass": "✅",
        "warn": "⚠️",
        "fail": "❌",
        "insufficient_data": "📊",  # New: data icon
    }.get(status, "❓")
```

Add explanation in table:
```python
if row["pass_fail"] == "insufficient_data":
    status_cell += f" (n={row['count']}, need {min_samples})"
```

---

## Critical Blocker #3: Run Scoping Not Enforced in Aggregation

### Issue

Copilot flagged:
> `apex_aggregate_ledger.py` accepts `run_id/commit_sha` but loads **all rows** from `performance_capsules`

**Impact:** Once the DB contains multiple runs (reruns, local dev, artifacts from different PRs), aggregation will produce incorrect mixed-run statistics.

### Fix Required

**In `scripts/apex_aggregate_ledger.py` (or inline workflow aggregation):**

```python
# BEFORE (BROKEN):
rows = conn.execute("SELECT capsule_json FROM performance_capsules").fetchall()

# AFTER (CORRECT):
rows = conn.execute("""
    SELECT capsule_json
    FROM performance_capsules
    WHERE run_id = ? AND commit_sha = ?
""", (run_id, commit_sha)).fetchall()
```

**Schema check required:**
```python
def table_has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Check if a table has a specific column."""
    cursor = conn.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cursor.fetchall())

# Then:
where_clauses = []
params = []

if table_has_column(conn, "performance_capsules", "run_id"):
    where_clauses.append("run_id = ?")
    params.append(run_id)

if table_has_column(conn, "performance_capsules", "commit_sha"):
    where_clauses.append("commit_sha = ?")
    params.append(commit_sha)

where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
query = f"SELECT capsule_json FROM performance_capsules WHERE {where_sql}"
rows = conn.execute(query, params).fetchall()
```

---

## Non-Blocking Improvements (Can Be Post-Merge)

### 1. Copilot: Truncate Aggregates in Clean Mode

**File:** `scripts/apex_rebuild_ledger.py`

Currently only truncates `performance_capsules`, should also clear `apex_runs`.

**Recommended fix (post-merge is fine):**
```python
tables_to_truncate = ["performance_capsules", "apex_runs"]
for table in tables_to_truncate:
    if table_exists(conn, table):
        logger.warning(f"CLEAN MODE: Truncating {table}")
        conn.execute(f"DELETE FROM {table}")
```

### 2. Copilot: Weekly Backup Needs `actions: read`

**File:** `.github/workflows/apex_performance.yml`

```diff
  weekly_backup:
    permissions:
      contents: write
+     actions: read  # Required for artifact download
```

### 3. Copilot: Integration Doc Has Placeholder Issue Numbers

**File:** `docs/APEX_REAL_PIPELINE_INTEGRATION.md`

Replace `#XXX` with real issue once created, or remove placeholder.

### 4. Copilot: Schema Version Mismatch in Docs

Multiple docs claim schema v2.0.0, but ledger.py has `SCHEMA_VERSION = 3`.

**Global find/replace:**
```bash
find docs -name "*.md" -exec sed -i '' 's/schema v2\.0\.0/schema v3.0.0/g' {} +
```

---

## Decision Matrix

| Path | Blockers to Fix | Timeline | Merge Confidence |
|------|----------------|----------|------------------|
| **Option A: Scaffolding Merge** | 1, 2, 3 | 2-4 hours | High (honest scope) |
| **Option B: Production Merge** | 1, 2, 3 + real wiring | 1-2 weeks | Very High |

---

## Recommended Next Steps (Option A)

1. **Implement 3 blocker fixes above** (~2-4 hours)
2. **Run CI to validate fixes**
3. **Update PR description:**
   ```
   ## APEX Governance Framework (Scaffolding Complete)

   This PR establishes the full APEX performance observability architecture:
   - ✅ Contract v1.0.0 (formal binding contract)
   - ✅ CI wiring (shadow mode, dry-run)
   - ✅ Ledger schema v3.0.0
   - ✅ PR comment generator
   - ✅ Dashboard deployment
   - 🚧 Real pipeline integration: PENDING (tracked separately)

   **Current State:** All components functional with synthetic data
   **Next Phase:** Wire real pipeline execution
   ```

4. **Create GitHub issue** from `APEX_REAL_PIPELINE_INTEGRATION.md`
5. **Merge with confidence**

---

## Contract Compliance Check

| Contract Requirement | Status | Notes |
|---------------------|--------|-------|
| Minimum 20 samples | ❌ Not enforced | Blocker #2 |
| SHA-pure data | ❌ Not enforced | Blocker #3 |
| Verdicts deterministic | ⚠️ Partial | Works with fixes |
| Mode isolation | ✅ Correct | shadow/enforce/disabled |
| Audit trail | ✅ Correct | Ledger + comment + logs |

---

## Approval Criteria

Once the 3 blockers are fixed, this PR becomes:

**APPROVED** with confidence because:
1. Contract formalization is excellent
2. Architecture is sound and well-documented
3. Scope is honest (scaffolding, not production)
4. Path forward is clear and tracked
5. Non-blocking issues are documented

---

## Signature

**Reviewed By:** Architecture + Systems Review
**Recommendation:** Fix 3 blockers → instant merge
**Confidence:** High (foundation is solid)
**Date:** 2026-02-08
