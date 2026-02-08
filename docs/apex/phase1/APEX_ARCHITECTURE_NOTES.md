# APEX Architecture Notes - Runner vs Ingester Pattern

## Discovered Clean Architecture Pattern

The APEX system implements a separation between **event generation** and **state storage**:

```
┌─────────────────────────────────────────────────────────────┐
│  Runner (apex_matrix_runner.py)                             │
│  ├─ Executes pipelines                                      │
│  ├─ Captures timing/metrics                                 │
│  └─ Writes Observation JSON (immutable event log)           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Observation JSON Files (event stream)                      │
│  ├─ observation_v1_local.json                               │
│  ├─ observation_v2_local.json                               │
│  └─ Timestamped, immutable, reproducible                    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Aggregator/Ingester (apex_aggregate_ledger.py)             │
│  ├─ Reads observation JSONs                                 │
│  ├─ Transforms into normalized DB schema                    │
│  └─ Writes to performance_capsules table (queryable state)  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  SQLite DB (apex_performance.db)                            │
│  ├─ performance_capsules (raw metrics)                      │
│  ├─ apex_runs (aggregated stats)                            │
│  └─ Query layer for gate evaluation                         │
└─────────────────────────────────────────────────────────────┘
```

## Why This Matters

### Benefits of Separation

1. **Testability**: Can validate runner output without DB dependency
2. **Replay**: Can re-ingest observation JSONs with different aggregation logic
3. **Audit Trail**: Immutable event log preserves raw data
4. **Debugging**: Can inspect observation files directly (human-readable JSON)
5. **Schema Migration**: Can transform old observations to new DB schema

### Current Implementation Status (2026-02-08)

**Smoke Test Results:**
- Runner: ✅ Writes observation JSON correctly
- Ingester: ⚠️ Not tested in smoke test
- DB Capsules: ❌ Count = 0 (expected, since ingester wasn't run)

**CI Workflow:**
- Current: Runner only (writes observations to ./apex_results/)
- Future: Add aggregation step to populate DB before gate evaluation

## Fixing the Validation Gap

The original validation summary claimed "real capsules generated" but actually meant "real observation JSON generated". The DB capsules are a separate step.

### Correct Statement of Facts

1. ✅ Runner executes real pipeline (V1/V2 workflows)
2. ✅ Runner captures accurate timing data (`is_synthetic: false`)
3. ✅ Runner writes observation JSON to disk
4. ❌ Smoke test did NOT run aggregation/ingestion
5. ❌ DB capsule count = 0 (expected, not a failure)

### What CI Actually Needs

For the gate to work in CI, the workflow must:

1. Run matrix_runner (generates observations)
2. Run aggregation (ingests observations → DB)
3. Run gate evaluation (queries DB)
4. Generate PR comment (reads gate verdict)

**Current CI coverage:** Steps 1, 3, 4 ✅
**Missing:** Step 2 (aggregation/ingestion)

This is likely why the workflow was designed with separate jobs for collection vs aggregation.

## Terminology Clarity

| Term | Meaning | File Location |
|------|---------|---------------|
| **Observation** | Immutable event record from one matrix run | `apex_results/observation_v1_local.json` |
| **Capsule** | Single-image performance record (in memory or JSON) | Inside observation.capsules[] |
| **DB Capsule** | Row in `performance_capsules` table | SQLite DB |
| **Aggregated Run** | Summary stats in `apex_runs` table | SQLite DB |

**Common confusion:** "capsule" can refer to either the in-memory/JSON object OR the database row. Context matters.

## Design Pattern Recognition

This is a variant of **Event Sourcing**:

- **Events**: Observation JSON files (append-only, immutable)
- **Projections**: DB tables (queryable, derived state)
- **Replay**: Can rebuild DB from observation history

It's also similar to **CQRS** (Command/Query Responsibility Segregation):

- **Write Model**: Runner produces observations
- **Read Model**: DB supports gate queries

## Recommendations

### For Testing

1. **Smoke test should validate end-to-end OR document scope clearly**
   - Option A: Run runner + aggregator, assert DB count > 0
   - Option B: Run runner only, assert observation exists, document DB as out-of-scope

2. **Contract tests should verify both layers**
   - Runner contract: produces valid observation JSON
   - Aggregator contract: consumes observation, writes DB capsules

### For Documentation

1. Update validation reports to use precise terminology
2. Distinguish between "observation capsule" and "DB capsule"
3. Document the two-phase architecture explicitly

### For CI

1. Verify aggregation step exists in workflow (or add it)
2. Add verification that DB is populated before gate runs
3. Consider adding health check: `SELECT COUNT(*) >= expected_count`

## Example: Complete Smoke Test

```bash
# Phase 1: Run pipeline
python scripts/apex_matrix_runner.py \
  --run-id "smoke-001" \
  --commit-sha "$(git rev-parse HEAD)" \
  --workflow-versions v1 \
  --input-dir ./tests/fixtures \
  --sample-size 3 \
  --output-dir /tmp/apex_smoke \
  --ledger-db /tmp/apex_smoke.db

# Verify observations written
test -f /tmp/apex_smoke/observation_v1_local.json || exit 1

# Phase 2: Ingest to DB
python scripts/apex_aggregate_ledger.py \
  --input-dir /tmp/apex_smoke \
  --ledger-db /tmp/apex_smoke.db \
  --run-id "smoke-001" \
  --commit-sha "$(git rev-parse HEAD)"

# Verify DB populated
CAPSULE_COUNT=$(sqlite3 /tmp/apex_smoke.db "SELECT COUNT(*) FROM performance_capsules;")
test "$CAPSULE_COUNT" -ge 3 || exit 1

echo "✅ End-to-end smoke test PASS"
```

## Conclusion

The runner-only smoke test was **correct for its scope** but the validation summary **overstated what was tested**.

Going forward:
- Be precise about what each test validates
- Document the event-sourcing pattern as intentional design
- Add aggregation smoke test to cover full workflow

---

**Pattern discovered:** 2026-02-08
**Documentation accuracy fix:** 2026-02-08
**Design pattern:** Event Sourcing + CQRS (unintentional but good)
