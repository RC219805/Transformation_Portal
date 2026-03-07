# APEX End-to-End Architecture

**Version:** 1.0.0
**Status:** Production-Ready Foundation
**Last Updated:** 2026-02-07

## Executive Summary

APEX (Architectural Photo Enhancement eXecution) is now a **production-grade performance observability platform** for the Transformation Portal. This document describes the complete end-to-end workflow from instrumentation to CI gating.

### What Changed

**Before (v1.0):**
- Ad-hoc performance tracking with CSV files
- No multi-zone awareness
- No V1/V2 comparison capability
- Manual regression detection
- No CI enforcement

**After (v2.0):**
- **Contract-driven architecture** (RunSpec → Observation → Judgement)
- **Zone-aware aggregation** (per-zone, global, worst-zone)
- **Dual-run support** (V1 + V2 on same commit)
- **Automated regression detection** with configurable thresholds
- **CI gate integration** with shadow/enforce modes
- **Production ledger** with SQLite backend and migration support

---

## Three-Layer Contract Architecture

The system is built on three immutable contracts:

### 1. RunSpec (Intent Layer)

**What we intend to test** - declared before execution.

```python
@dataclass(frozen=True)
class RunSpec:
    run_id: str                        # Unique run identifier
    commit_sha: str                    # Git commit SHA
    workflow_version: Literal["v1", "v2"]  # Pipeline version
    zones: List[str]                   # Deployment zones
    device: str                        # "mps", "cuda", "cpu"
    backend_id: str                    # "da3", "depth_pro"
    scene_type: Optional[str]          # Optional scene filter
    timestamp: str                     # ISO8601 timestamp
    config_hash: str                   # Configuration fingerprint
```

**Immutability:** RunSpec is frozen - it represents intent, not results.

### 2. Observation (Reality Layer)

**What actually happened** - captured during/after execution.

```python
@dataclass
class Observation:
    run_spec: RunSpec                  # Linked to intent
    zone: Optional[str]                # Zone where captured
    capsules: List[PerformanceCapsule] # Raw measurements
    phase_timings: Dict[str, float]    # Pipeline-level timings
    resource_metadata: Dict[str, Any]  # Memory, GPU util, etc.
    errors: List[str]                  # Execution errors
    captured_at: str                   # Timestamp
```

**Relationship:** One RunSpec → Many Observations (one per zone).

### 3. Judgement (Decision Layer)

**What we do with it** - computed from Observations.

```python
@dataclass
class Judgement:
    run_id: str
    workflow_version: Literal["v1", "v2"]
    zone: Optional[str]                # None = global judgement
    bucket_stats: Dict[str, BucketStats]
    regression_report: Optional[RegressionReport]
    pass_fail: Literal["pass", "warn", "fail"]
    explanation: str
    worst_zone_p95: Optional[float]    # Critical for gating
    worst_zone_name: Optional[str]
```

**Relationship:** One Observation → One Judgement (per zone or global).

---

## Zone Awareness

### What Are Zones?

Zones represent **fault/latency/capacity boundaries** in the deployment topology:

- **Kubernetes:** topology zones (e.g., `us-west-2a`, `us-east-1b`)
- **AWS:** Availability Zones via IMDSv2
- **On-prem:** Racks, data centers
- **Local:** "local" (default fallback)

### Why Zones Matter

**User experience is bounded by the worst zone**, not the average:

- If zone A is fast (8s) and zone B is slow (15s), users in zone B have a poor experience
- Global p95 (11s) hides this variance
- **We gate on worst-zone p95**, not global p95

### Zone Resolution Priority

```python
ZoneResolver.resolve() -> str
```

1. Explicit override parameter (for testing)
2. `APEX_ZONE` environment variable
3. Kubernetes topology zone (`KUBE_NODE_ZONE` or `/etc/podinfo/zone`)
4. AWS EC2 availability zone (IMDSv2)
5. "local" (fallback)

**Never raises** - always returns a string.

---

## Workflow Version (V1 vs V2)

### Dual-Run Architecture

APEX supports running **V1 and V2 on the same commit** for direct comparison:

```bash
# Matrix run: V1 + V2 across 3 zones
python scripts/apex_matrix_runner.py \
    --run-id abc123 \
    --commit-sha abc123 \
    --workflow-versions v1 v2 \
    --zones us-west-2a us-west-2b us-east-1a
```

### Storage Model

V1 and V2 results are stored with different `workflow_version` tags:

```sql
SELECT * FROM apex_runs WHERE run_id = 'abc123';
-- Returns rows for both v1 and v2 with same run_id
```

### Comparison Strategy

- **V1 baseline:** Stable, production-tested workflow
- **V2 candidate:** New workflow under evaluation
- **Shadow mode:** V2 runs alongside V1 but doesn't block CI
- **Enforce mode:** V2 must not regress vs V1 to merge

---

## Aggregation Pipeline

### Per-Zone Aggregation

```python
from transformation_portal.metrics.aggregator import compute_per_zone_stats

per_zone = compute_per_zone_stats(capsules, buckets)
# {
#   "us-west-2a": {"pool_medium_mps": BucketStats(...), ...},
#   "us-west-2b": {"pool_medium_mps": BucketStats(...), ...},
# }
```

### Global Aggregation

```python
global_stats = compute_global_stats(capsules, buckets)
# {"pool_medium_mps": BucketStats(...), ...}
```

### Worst-Zone Detection

```python
worst_zone, worst_p95 = compute_worst_zone_p95(per_zone)
# ("us-west-2b", 15.3)  # Critical for gating
```

**Gate on worst-zone p95**, not global p95.

---

## Regression Detection

### Comparison Modes

1. **Commit-to-commit:** Current commit vs previous commit
2. **Branch-to-main:** Feature branch vs main branch
3. **V1-to-V2:** V2 workflow vs V1 baseline

### Thresholds

```python
DEFAULT_WARN_THRESHOLD = 0.10  # 10% regression triggers warning
DEFAULT_FAIL_THRESHOLD = 0.15  # 15% regression triggers failure
```

### Example: V1 vs V2

```python
from transformation_portal.metrics.comparator import detect_v1_v2_regression

report = detect_v1_v2_regression(
    v2_stats=v2_stats,
    v1_stats=v1_stats,
    run_id="abc123",
    commit_sha="abc123",
)

print(report.status)  # "pass", "warn", or "fail"
print(report.max_regression)  # 0.08 (8% regression)
print(report.max_regression_bucket)  # "pool_medium_mps"
```

---

## CI Gate Logic

### Gate Rules (Evaluated in Order)

1. **Bucket threshold violation:** Any bucket exceeds p95 threshold
2. **Worst-zone p95:** Worst-zone p95 > threshold (user experience gate)
3. **Regression:** Regression vs baseline > threshold (quality gate)

### Gate Modes

| Mode       | Behavior                                      | Use Case                  |
|------------|-----------------------------------------------|---------------------------|
| `enforce`  | Block if any rule fails                       | V1 production enforcement |
| `shadow`   | Log warnings but don't block                  | V2 rollout / monitoring   |
| `disabled` | Always pass (no-op)                           | Temporary bypass          |

### Example: Gate Evaluation

```python
from transformation_portal.metrics.gate import evaluate_gate

result = evaluate_gate(
    judgement=judgement,
    worst_zone_p95_threshold=15.0,  # seconds
    max_regression_threshold=0.15,  # 15%
    mode="enforce",
)

if result.should_block:
    print(f"GATE BLOCKED: {result.explanation}")
    sys.exit(1)
```

---

## CI Workflow (GitHub Actions)

### Matrix Job Graph

```
┌─────────────┐
│ Build + Test│
└──────┬──────┘
       │
┌──────▼──────────────────────────────────┐
│ APEX Matrix Run                          │
│ - V1 × zones                            │
│ - V2 × zones                            │
└──────┬──────────────────────────────────┘
       │
┌──────▼──────────────────────────────────┐
│ Aggregate                                │
│ - Per-zone stats                        │
│ - Global stats                          │
│ - Worst-zone p95                        │
└──────┬──────────────────────────────────┘
       │
┌──────▼──────────────────────────────────┐
│ Compare to Baseline                      │
│ - Query historical data                 │
│ - Detect regressions                    │
└──────┬──────────────────────────────────┘
       │
┌──────▼──────────────────────────────────┐
│ Gate                                     │
│ - Evaluate rules                        │
│ - Block if fail                         │
└──────┬──────────────────────────────────┘
       │
┌──────▼──────────────────────────────────┐
│ Publish                                  │
│ - PR comment                            │
│ - Dashboard update                      │
└─────────────────────────────────────────┘
```

### Example Workflow (`.github/workflows/apex_matrix.yml`)

```yaml
name: APEX Performance Matrix

on:
  pull_request:
  push:
    branches: [main]

jobs:
  apex_matrix:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        zone: [us-west-2a, us-west-2b, us-east-1a]
        workflow_version: [v1, v2]
    steps:
      - uses: actions/checkout@v4
      - name: Run APEX
        run: |
          python scripts/apex_matrix_runner.py \
            --run-id ${{ github.sha }} \
            --commit-sha ${{ github.sha }} \
            --workflow-versions ${{ matrix.workflow_version }} \
            --zones ${{ matrix.zone }} \
            --output-dir ./apex_results
      - uses: actions/upload-artifact@v4
        with:
          name: apex-${{ matrix.workflow_version }}-${{ matrix.zone }}
          path: ./apex_results

  apex_gate:
    needs: apex_matrix
    runs-on: ubuntu-latest
    steps:
      - uses: actions/download-artifact@v4
      - name: Aggregate and Gate
        run: |
          python scripts/apex_aggregate_and_gate.py \
            --run-id ${{ github.sha }} \
            --commit-sha ${{ github.sha }} \
            --mode enforce
      - name: Generate PR Comment
        if: github.event_name == 'pull_request'
        run: |
          python scripts/apex_pr_comment.py \
            --run-id ${{ github.sha }} \
            --commit-sha ${{ github.sha }} \
            --ledger-db ./apex_performance.db \
            --output comment.md
```

---

## Schema Evolution and Migration

### PerformanceCapsule v2.0.0

**New fields:**
- `workflow_version: Literal["v1", "v2"]` (default: "v1")
- `zone: Optional[str]` (default: None)

**Migration:**
```python
# Automatic migration in from_dict()
if schema_version == "1.0.0":
    data.setdefault("workflow_version", "v1")
    data.setdefault("zone", None)
    data["schema_version"] = "2.0.0"
```

### Ledger Schema v2

**New columns:**
- `workflow_version TEXT DEFAULT 'v1'`
- `zone TEXT`

**New table:**
- `apex_runs` for aggregated run-level stats

**Migration:**
```sql
-- Automatic migration in _migrate_schema()
ALTER TABLE performance_capsules ADD COLUMN workflow_version TEXT DEFAULT 'v1';
ALTER TABLE performance_capsules ADD COLUMN zone TEXT;
```

---

## Security and Integrity

### Artifact Integrity (Planned)

```python
# Hash all APEX result blobs
result_hash = hashlib.sha256(result_blob).hexdigest()

# Sign with HMAC (requires shared secret)
import hmac
signature = hmac.new(secret_key, result_blob, hashlib.sha256).hexdigest()
```

### RBAC for Threshold Changes (Planned)

- Require approval for bucket threshold changes
- Track changes in audit log
- Prevent unauthorized threshold relaxation

### Data Minimization

- No PII in logs or artifacts
- Image paths are anonymized where possible
- Capsules contain only performance metadata

---

## Operational Guide

### Running Locally

```bash
# Dry-run mode (uses mock data)
python scripts/apex_matrix_runner.py \
    --run-id local-test \
    --commit-sha $(git rev-parse HEAD) \
    --workflow-versions v1 v2 \
    --zones local \
    --output-dir ./apex_results \
    --dry-run

# Generate PR comment
python scripts/apex_pr_comment.py \
    --run-id local-test \
    --commit-sha $(git rev-parse HEAD) \
    --ledger-db ./apex_performance.db \
    --output comment.md
```

### Querying Ledger

```bash
# Query recent V1 runs
python -m transformation_portal.metrics.ledger query \
    --ledger-db ./apex_performance.db \
    --workflow-version v1 \
    --min-days 7 \
    --output recent_v1.json

# Detect regression
python -m transformation_portal.metrics.ledger regression \
    --ledger-db ./apex_performance.db \
    --capsule current_capsule.json \
    --baseline-days 30
```

### Pruning Old Data

```bash
# Prune entries older than 90 days
python -m transformation_portal.metrics.ledger prune \
    --ledger-db ./apex_performance.db \
    --days-to-keep 90
```

---

## Future Enhancements

### Phase 2: Distribution-Friendly Metrics

- Emit histogram/sketch structures for better aggregation
- Support HDRHistogram or SplineSketch
- Tag with all dimensions (workflow_version, zone, bucket, device, backend)

### Phase 3: Dashboard Integration

- Time series: p50/p95 per bucket over time
- Zone heatmap: (zone × bucket) showing p95 vs threshold
- Regression alerts with drill-down

### Phase 4: Advanced Security

- Artifact signing and verification
- RBAC for threshold changes
- Rate limiting to prevent APEX spam

---

## Compliance and Governance

### ADR Binding

This architecture is governed by:
- **ADR-025-APEX-end-to-end.md** (this implementation)

### Contract Stability Promise

- RunSpec, Observation, Judgement are **contract-stable**
- Changes require version bump and migration plan
- Backward compatibility required for ledger queries

### Quality Firewall Integration

- APEX gates enforce Quality Firewall thresholds
- Bucket thresholds are contract-stable (requires ADR to change)
- Regression thresholds are configurable (default: 10% warn, 15% fail)

---

## Success Metrics

After implementation:

✅ Single commit runs V1 and V2 across multiple zones
✅ All results stored with workflow_version + zone tags
✅ Per-zone, global, and worst-zone stats computed
✅ Regression detection works for V1→V2 comparison
✅ Gate blocks PRs if worst-zone p95 > threshold
✅ PR comments show human-readable performance summary
✅ Dashboard can visualize zone×bucket heatmap (schema ready)
✅ All timing uses unified `torch.accelerator.synchronize()`
✅ Security guardrails (integrity, RBAC) designed

---

## References

- **APEX Executive Summary:** `docs/APEX_EXECUTIVE_SUMMARY_20260207.md`
- **APEX Workflow Design:** `docs/APEX_WORKFLOW_DESIGN.md`
- **Performance Analysis:** `docs/PERFORMANCE_ANALYSIS_20260207.md`
- **ADR-025:** `docs/architecture/decisions/ADR-025-APEX-end-to-end.md`

---

**Document Owner:** Transformation Portal Architect
**Review Cycle:** Quarterly or when contracts change
**Next Review:** 2026-05-07
