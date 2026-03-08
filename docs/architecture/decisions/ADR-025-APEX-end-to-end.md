# ADR-025: APEX End-to-End Workflow Architecture

**Status:** Accepted
**Date:** 2026-02-07
**Deciders:** Transformation Portal Architect
**Supersedes:** ADR-019 (partial - APEX instrumentation baseline)

---

## Context and Problem Statement

APEX (Architectural Photo Enhancement eXecution) performance tracking has evolved from basic instrumentation to a mission-critical observability platform. However, as of 2026-02-07, key gaps prevent production use:

1. **No V2 workflow support:** V2 is a forked pipeline, not a first-class dimension
2. **No zone awareness:** Cannot detect per-zone variance or worst-zone p95
3. **Manual regression detection:** No automated baseline comparison
4. **No CI enforcement:** Performance regressions slip through
5. **No dual-run capability:** Cannot compare V1 vs V2 on same commit

**Business Impact:**
- V2 rollout blocked by lack of performance comparison
- Zone-specific performance issues hidden by global averages
- Regressions discovered too late (post-merge, post-release)
- No enforcement mechanism for Quality Firewall thresholds

---

## Decision Drivers

1. **V2 Adoption Enablement:** Must support side-by-side V1/V2 comparison
2. **User Experience Focus:** Gate on worst-zone p95, not global average
3. **Automation:** CI must detect and block regressions automatically
4. **Contract Stability:** Changes must not break existing data or workflows
5. **Security:** Artifact integrity and RBAC for threshold changes
6. **Operational Simplicity:** Easy to run locally and in CI

---

## Considered Options

### Option 1: Extend Existing Schema (In-Place Evolution) ✅ CHOSEN

**Approach:**
- Add `workflow_version` and `zone` fields to PerformanceCapsule
- Extend ledger schema with new columns and `apex_runs` table
- Implement migration logic for backward compatibility
- Build contracts layer (RunSpec, Observation, Judgement)

**Pros:**
- Minimal disruption to existing instrumentation
- Backward compatible with v1.0.0 data
- Incremental migration path

**Cons:**
- Schema migration complexity
- Need to maintain two schema versions during transition

### Option 2: Greenfield Rewrite

**Approach:**
- New `apex_v2` module with clean-slate schema
- Parallel operation with legacy `metrics` module
- Eventual migration to new schema

**Pros:**
- No migration complexity
- Clean architecture without legacy baggage

**Cons:**
- Dual maintenance burden
- Data fragmentation (old vs new)
- Delayed V2 adoption (requires full rewrite first)

**Rejected:** Too disruptive and delays V2 rollout.

### Option 3: External Observability Platform

**Approach:**
- Push metrics to external system (Prometheus, Datadog, etc.)
- Use external dashboards and alerting

**Pros:**
- Rich visualization and alerting out-of-box
- Industry-standard tooling

**Cons:**
- Dependency on external service
- Cost and operational complexity
- Data leaves repository boundary
- No offline operation

**Rejected:** Adds external dependency and violates data governance.

---

## Decision Outcome

**Chosen Option:** Option 1 (Extend Existing Schema)

### Rationale

1. **Fastest path to V2 adoption:** Extends proven instrumentation
2. **Backward compatible:** Existing v1.0.0 data migrates seamlessly
3. **Self-contained:** No external dependencies
4. **Contract-driven:** RunSpec/Observation/Judgement provide clear layering
5. **CI-ready:** Scripts can be integrated into existing workflows

---

## Implementation Design

### Three-Layer Contract Architecture

```
┌─────────────────────────────────────────────────┐
│ RunSpec (Intent Layer - Immutable)              │
│ - What we intend to test                        │
│ - run_id, commit_sha, workflow_version, zones   │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│ Observation (Reality Layer)                     │
│ - What actually happened                        │
│ - run_spec + zone + capsules + errors           │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│ Judgement (Decision Layer)                      │
│ - What we do with it                            │
│ - bucket_stats + regression_report + pass/fail  │
└─────────────────────────────────────────────────┘
```

### Zone Resolution Strategy

**Priority:**
1. Explicit override (for testing)
2. `APEX_ZONE` environment variable
3. Kubernetes topology zone (`KUBE_NODE_ZONE` or Downward API)
4. AWS EC2 availability zone (IMDSv2)
5. "local" (fallback)

**Safe by design:** Never raises, always returns a string.

### Workflow Version Semantics

| Value | Meaning                              | Current Status    |
|-------|--------------------------------------|-------------------|
| `v1`  | Lux Depth V3 pipeline (production)   | Stable, enforced  |
| `v2`  | Next-gen pipeline (under evaluation) | Shadow mode       |

**Dual-run model:** Same `run_id`, different `workflow_version` tags.

### Bucket Specificity Scoring

Extended to include workflow_version and zone:

```python
workflow_version: +10  # Critical for V1/V2 comparison
scene_type:       +10  # Primary discriminator
zone:             +5   # Deployment topology
device:           +5   # Hardware-specific
backend_id:       +5   # Model-specific
pixel_count:      +3   # Counts once (not min+max separately)
```

### Aggregation Pipeline

```
Raw Capsules
    ↓
Per-Zone Stats (zone → bucket → p50/p95/p99)
    ↓
Global Stats (bucket → p50/p95/p99)
    ↓
Worst-Zone Detection (max p95 across zones)
    ↓
Ledger Storage (apex_runs table)
```

### Gate Logic

**Rules (evaluated in order):**
1. Bucket threshold violation (any bucket exceeds p95 threshold)
2. Worst-zone p95 > threshold (user experience gate)
3. Regression > threshold (quality regression gate)

**Modes:**
- `enforce`: Block if any rule fails (V1 production)
- `shadow`: Log warnings but don't block (V2 rollout)
- `disabled`: Always pass (temporary bypass)

---

## Schema Changes

### PerformanceCapsule v2.0.0

**New fields:**
```python
workflow_version: Literal["v1", "v2"] = "v1"
zone: Optional[str] = None
```

**Migration:**
```python
if schema_version == "1.0.0":
    data.setdefault("workflow_version", "v1")
    data.setdefault("zone", None)
    data["schema_version"] = "2.0.0"
```

### Ledger Schema v2

**New columns in `performance_capsules`:**
```sql
workflow_version TEXT DEFAULT 'v1',
zone TEXT
```

**New table:**
```sql
CREATE TABLE apex_runs (
    run_id TEXT NOT NULL,
    commit_sha TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    workflow_version TEXT NOT NULL,
    zone TEXT,
    bucket_name TEXT NOT NULL,
    p50 REAL NOT NULL,
    p95 REAL NOT NULL,
    p99 REAL,
    count INTEGER NOT NULL,
    threshold_p50 REAL NOT NULL,
    threshold_p95 REAL NOT NULL,
    pass_fail TEXT NOT NULL,
    raw_capsules_json TEXT,
    PRIMARY KEY (run_id, workflow_version, zone, bucket_name)
);
```

---

## Timing Improvements

### Unified GPU Sync

**Before:**
```python
if device == "mps":
    torch.mps.synchronize()
elif device == "cuda":
    torch.cuda.synchronize()
```

**After:**
```python
# PyTorch 2.4+ unified API
if hasattr(torch, "accelerator"):
    torch.accelerator.synchronize()
else:
    # Fallback for older versions
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
```

**Benefit:** Device-agnostic, future-proof.

---

## CI Integration

### Matrix Workflow

```yaml
jobs:
  apex_matrix:
    strategy:
      matrix:
        workflow_version: [v1, v2]
        zone: [us-west-2a, us-west-2b, us-east-1a]
    steps:
      - name: Run APEX
        run: |
          python scripts/apex_matrix_runner.py \
            --workflow-versions ${{ matrix.workflow_version }} \
            --zones ${{ matrix.zone }}

  apex_gate:
    needs: apex_matrix
    steps:
      - name: Aggregate and Gate
        run: python scripts/apex_aggregate_and_gate.py
      - name: PR Comment
        run: python scripts/apex_pr_comment.py
```

---

## Security Considerations

### Artifact Integrity (Phase 1 Foundation)

- Hash all APEX result blobs with SHA256
- Store hashes in ledger
- Verify on read (planned for Phase 2)

### RBAC for Threshold Changes (Planned)

- Require approval workflow for bucket threshold changes
- Track changes in audit log
- Prevent unauthorized threshold relaxation

### Data Minimization

- No PII in logs or artifacts
- Image paths anonymized where possible
- Capsules contain only performance metadata

---

## Migration Plan

### Phase 1: Foundation (This ADR) ✅

- [x] Extend PerformanceCapsule schema to v2.0.0
- [x] Add workflow_version and zone fields
- [x] Implement ZoneResolver
- [x] Create contracts layer (RunSpec, Observation, Judgement)
- [x] Extend ledger schema with migration support
- [x] Implement aggregator module
- [x] Implement comparator module
- [x] Implement gate module
- [x] Create CI orchestration scripts
- [x] Write comprehensive documentation

### Phase 2: CI Integration (Next Sprint)

- [ ] Integrate matrix runner with actual pipelines
- [ ] Add GitHub Actions workflows
- [ ] Deploy to staging environment
- [ ] Run shadow-mode validation (V2 doesn't block)

### Phase 3: V2 Enforcement (After Validation)

- [ ] Switch V2 from shadow to warn mode
- [ ] Collect V2 baseline data (30 days)
- [ ] Switch V2 to enforce mode
- [ ] Deprecate V1 (if V2 proves superior)

### Phase 4: Advanced Features

- [ ] Dashboard integration
- [ ] Artifact signing and verification
- [ ] RBAC for threshold changes
- [ ] Distribution-friendly metrics (histograms)

---

## Consequences

### Positive

1. **V2 Adoption Unblocked:** Side-by-side comparison with V1
2. **Zone Awareness:** Detect and gate on worst-zone p95
3. **Automated Enforcement:** CI blocks regressions automatically
4. **Contract Stability:** Clear layering and migration path
5. **Self-Contained:** No external dependencies

### Negative

1. **Schema Migration Complexity:** Need to maintain v1/v2 compatibility
2. **Testing Burden:** Must test across zones and workflow versions
3. **Operational Learning Curve:** New concepts (zones, dual-run)

### Neutral

1. **Ledger Growth:** `apex_runs` table grows with each run (mitigated by pruning)
2. **CI Runtime:** Matrix runs increase CI time (parallelizable)

---

## Rollback Plan

If critical issues discovered:

1. **Instrumentation:** v2.0.0 schema backward-compatible with v1.0.0 readers
2. **Ledger:** Migration is reversible (columns can be ignored)
3. **Gate:** Disable gate via `mode="disabled"`
4. **Scripts:** Old scripts still work (ignore new fields)

**Maximum Rollback Time:** < 1 hour (disable gate, revert scripts).

---

## Compliance and Governance

### Quality Firewall Integration

- Gate enforces Quality Firewall thresholds
- Bucket thresholds are contract-stable
- Changes require ADR and review

### Documentation Requirements

- [x] End-to-end architecture doc
- [x] CI integration guide
- [x] Zone concepts guide
- [x] ADR (this document)

### Testing Requirements

- [ ] Contract validation tests
- [ ] Zone resolver tests
- [ ] Dual-write tests (V1/V2)
- [ ] Aggregation tests
- [ ] Gate logic tests
- [ ] Migration tests

---

## References

- **APEX Executive Summary:** `docs/historical/APEX_EXECUTIVE_SUMMARY_20260207.md`
- **APEX Workflow Design:** `docs/architecture/APEX_WORKFLOW_DESIGN.md`
- **Performance Analysis:** `docs/performance/PERFORMANCE_ANALYSIS_20260207.md`
- **End-to-End Architecture:** `docs/architecture/APEX_END_TO_END_ARCHITECTURE.md`
- **ADR-019:** Previous APEX instrumentation baseline

---

## Approval

**Architect Decision:** ACCEPTED
**Date:** 2026-02-07
**Rationale:** Provides fastest path to V2 adoption with minimal disruption. Contract-driven design ensures long-term maintainability. Zone awareness addresses real user experience variance.

**Review Cycle:** Quarterly or when contracts change
**Next Review:** 2026-05-07
