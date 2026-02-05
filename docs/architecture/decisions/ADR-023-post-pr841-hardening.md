# ADR-023: Post-PR #841 Hardening Strategy

**Status:** Approved  
**Date:** 2026-02-05  
**Authority:** Transformation Portal Architect  
**Supersedes:** None  
**Related:** PR #841, ADR-018 (Depth Pro), ADR-019 (Backend Unification)

---

## Executive Summary

This ADR defines the architectural approach for implementing three critical hardening improvements identified through production validation of PR #841 (DA3 PIL support):

1. **Phase 1: Input Hygiene** ✅ COMPLETED (commit 4761e2e5)
2. **Phase 2: Performance Ledger** 🎯 APPROVED (tooling, no orchestrator changes)
3. **Phase 3: Backend Selection Truth** 🎯 APPROVED (manifest + logging, defer enforcement)

**Key Decision:** Implement Phase 2 and Phase 3 in parallel with minimal coupling. Phase 3 uses additive manifest fields and truth-line logging without enforcing hard failures (defer to v2.1.0).

---

## Context

### Production Validation Results (PR #841)

**Ground Truth:**
- Batch: 20 images, 100% success rate
- Orchestration overhead: 0.224% (excellent)
- Runtime: median 11.82s, p95 30.43s, max 30.83s
- **Critical Issue:** Processed `_depthpro_depth16.png` artifact as RGB input (now fixed in Phase 1)

**What Works:**
- Input hygiene preventing artifact reprocessing ✅
- V2 integration with orchestrator ✅
- DA3 PIL input support ✅
- Production quality maintained (APEX tier) ✅

**What's Missing:**
1. Performance regression detection (no baseline tracking, no comparison)
2. Backend selection truthfulness (silent mismatches, no audit trail)
3. Manifest metadata gaps (no requested_backend vs resolved_backend)

---

## Decision

### Phase 2: Performance Ledger Tool

**APPROVED: Standalone reporting tool, no CI gate**

**Scope:**
- Create `tools/performance_ledger.py` (standalone script)
- Parse manifest JSONs from batch runs
- Compute statistics: mean, median, p90, p95, min, max, success rate
- Compare current run against baseline with regression detection
- Output markdown (human) + JSON (machine)
- Track environment metadata (OS, Python, torch, device)

**Regression Policy:**
```python
REGRESSION_THRESHOLDS = {
    "p95_worsening_pct": 10,      # p95 > 10% slower = regression
    "mean_worsening_pct": 15,     # mean > 15% slower = regression
    "failure_rate_increase": 0,   # Any failures = regression
}
```

**CLI Usage:**
```bash
# Capture baseline from production run
python tools/performance_ledger.py \
  --manifests-dir ./output/lux_depth_v3_apex_post_841/manifests \
  --output ./docs/performance/baseline_v2.0.0_da3.json

# Compare current run against baseline
python tools/performance_ledger.py \
  --baseline ./docs/performance/baseline_v2.0.0_da3.json \
  --compare ./output/current_run/manifests \
  --output ./output/perf_report.md \
  --emit-json ./output/perf_current.json
```

**Baseline Storage:**
- Location: `docs/performance/baselines/`
- Format: `baseline_{version}_{backend}_{tier}.json`
- Version control: Committed to repo for historical tracking
- Update policy: Manual approval required (Architect review)

**CI Integration:**
- **Not a gate** (Phase 2 is tooling only)
- Optional workflow: Run ledger on nightly builds, report to Slack/GitHub
- Defer enforcement to v2.1.0 (need stable baselines first)

**Rationale:**
- **Tooling first:** Prove value before enforcement
- **Manual baselines:** Prevent automated baseline inflation
- **No orchestrator changes:** Zero risk to production pipeline
- **Reversible:** Tool can be removed if not useful

---

### Phase 3: Backend Selection Truth

**APPROVED: Additive metadata + logging, no enforcement**

**Problem:**
Currently, if user requests `--depth-backend depth_pro`:
- System may silently initialize DA3 instead
- Manifest shows `"model": "depth-anything-v3-metric-large"` with no indication of mismatch
- No error, warning, or audit trail

**Solution (Phased):**

#### Phase 3A: Manifest Enhancement (THIS PR)
Add backend selection metadata to manifest schema:

```python
@dataclass
class BackendSelectionMetadata:
    """Backend selection audit trail."""
    requested_backend: Optional[str]      # User-specified or None (auto)
    resolved_backend: str                 # Actual backend used
    resolution_status: str                # "success", "fallback", "error"
    resolution_reason: Optional[str]      # Why fallback occurred (if any)
    model_id: str                         # HuggingFace model ID or checkpoint path
    device: str                           # Resolved device (mps/cuda/cpu)

@dataclass
class CombinedManifest:
    # ... existing fields ...
    backend_selection: Optional[BackendSelectionMetadata] = None  # NEW
```

**Example Manifest Output:**
```json
{
  "backend_selection": {
    "requested_backend": "depth_pro",
    "resolved_backend": "depth_anything_v3",
    "resolution_status": "fallback",
    "resolution_reason": "depth_pro checkpoint missing: checkpoints/depth_pro.pt",
    "model_id": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
    "device": "mps"
  }
}
```

#### Phase 3B: Truth-Line Logging (THIS PR)
Add startup logging to orchestrator:

```python
logger.info(
    "Backend selection: requested=%s resolved=%s status=%s device=%s",
    backend_metadata.requested_backend or "auto",
    backend_metadata.resolved_backend,
    backend_metadata.resolution_status,
    backend_metadata.device,
)

if backend_metadata.resolution_status == "fallback":
    logger.warning(
        "Backend fallback: %s", backend_metadata.resolution_reason
    )
```

**Example Log Output:**
```
INFO: Backend selection: requested=depth_pro resolved=depth_anything_v3 status=fallback device=mps
WARNING: Backend fallback: depth_pro checkpoint missing: checkpoints/depth_pro.pt
```

#### Phase 3C: Enforcement (DEFERRED to v2.1.0)
Future behavior (not in this PR):
- Add `--strict-backend` flag: fail on mismatch
- Add `--allow-backend-fallback` flag: permit with warning
- Default: fail on mismatch (breaking change, requires ADR + migration)

**Rationale for Deferral:**
- **ADR-019 not yet implemented:** Backend registry doesn't exist yet
- **Breaking change:** Failing on fallback would disrupt existing workflows
- **Validation needed:** Need real-world data on fallback frequency
- **Incremental safety:** Logging + manifest now, enforcement later

---

### Phase 2 & 3 Parallel Implementation

**Approved Strategy: Option B (Parallel Implementation)**

**Rationale:**
- Phase 2 (performance ledger) = pure tooling, zero orchestrator risk
- Phase 3A/B (manifest + logging) = additive only, no behavior change
- No coupling between phases (different modules, different outputs)
- Single PR enables holistic testing of both improvements
- Rollback is clean: revert one commit

**Commit Structure:**
```
feat: add performance ledger tool + backend selection truth

Phase 2: Performance Ledger
- Create tools/performance_ledger.py
- Add regression threshold policy
- Document baseline capture workflow

Phase 3: Backend Selection Truth
- Add BackendSelectionMetadata to manifest schema
- Add truth-line logging to orchestrator
- Defer enforcement to v2.1.0 (ADR-023)
```

---

## Implementation Guidance

### Phase 2: Performance Ledger

**Module:** `tools/performance_ledger.py`

**Key Components:**
```python
def parse_manifests(manifests_dir: Path) -> List[Dict[str, Any]]:
    """Load all manifest JSONs from directory."""

def compute_statistics(timings: List[float]) -> Dict[str, float]:
    """Compute mean, median, p90, p95, min, max."""

def detect_regressions(
    baseline: Dict, current: Dict, thresholds: Dict
) -> List[str]:
    """Compare current vs baseline, return regression warnings."""

def format_markdown(
    stats: Dict, regressions: List[str], env: Dict
) -> str:
    """Generate human-readable report."""

def capture_environment() -> Dict[str, str]:
    """Capture OS, Python, torch, device metadata."""
```

**Baseline Schema:**
```json
{
  "version": "v2.0.0",
  "backend": "depth_anything_v3",
  "quality_tier": "apex",
  "environment": {
    "python": "3.11.0",
    "torch": "2.1.0",
    "device": "mps",
    "os": "macOS-14.0-arm64"
  },
  "statistics": {
    "count": 20,
    "mean_sec": 13.92,
    "median_sec": 11.82,
    "p90_sec": 28.50,
    "p95_sec": 30.43,
    "min_sec": 5.21,
    "max_sec": 30.83,
    "success_rate": 1.0
  },
  "captured_at": "2026-02-04T06:55:48Z"
}
```

**Tests Required:**
- `test_parse_manifests_directory()`
- `test_compute_statistics_correctness()`
- `test_detect_regressions_thresholds()`
- `test_baseline_schema_validation()`
- `test_markdown_formatting()`
- `test_environment_capture()`

---

### Phase 3: Backend Selection Truth

**Module 1:** `src/transformation_portal/lux_depth_v3/manifest.py`

**Changes:**
```python
@dataclass
class BackendSelectionMetadata:
    """Backend selection audit trail (added in v2.0.1)."""
    requested_backend: Optional[str]
    resolved_backend: str
    resolution_status: str
    resolution_reason: Optional[str]
    model_id: str
    device: str
    schema_version: str = "1.0"

@dataclass
class CombinedManifest:
    # ... existing fields ...
    backend_selection: Optional[BackendSelectionMetadata] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize with backend_selection if present."""
        data = asdict(self)
        # Existing serialization logic
        return data
```

**Module 2:** `src/transformation_portal/lux_depth_v3/orchestrator.py`

**Changes:**
```python
def _capture_backend_metadata(
    self, 
    requested: Optional[str],
    engine: DA3InferenceEngine
) -> BackendSelectionMetadata:
    """Capture backend selection decision."""
    resolved = "depth_anything_v3"  # Current reality
    status = "success"
    reason = None
    
    if requested and requested != resolved:
        status = "fallback"
        reason = f"Requested {requested} not available, using {resolved}"
    
    return BackendSelectionMetadata(
        requested_backend=requested,
        resolved_backend=resolved,
        resolution_status=status,
        resolution_reason=reason,
        model_id=engine.config.model_variant.huggingface_id,
        device=str(engine.device),
    )

def enhance_batch(self, input_dir: Path, ...) -> List[Dict[str, Any]]:
    # ... existing code ...
    
    # NEW: Capture backend selection
    backend_metadata = self._capture_backend_metadata(
        requested=self.config.depth_backend,
        engine=self.engine
    )
    
    # NEW: Log truth line
    logger.info(
        "Backend selection: requested=%s resolved=%s status=%s device=%s",
        backend_metadata.requested_backend or "auto",
        backend_metadata.resolved_backend,
        backend_metadata.resolution_status,
        backend_metadata.device,
    )
    
    if backend_metadata.resolution_status == "fallback":
        logger.warning("Backend fallback: %s", backend_metadata.resolution_reason)
    
    # ... rest of batch processing ...
    
    # NEW: Include in manifest
    manifest.backend_selection = backend_metadata
```

**Tests Required:**
- `test_backend_selection_metadata_schema()`
- `test_backend_selection_success_path()`
- `test_backend_selection_fallback_logging()`
- `test_manifest_includes_backend_selection()`
- `test_backend_selection_none_request_auto()`

---

## Consequences

### Positive

1. **Performance Visibility:** First-class regression detection for runtime performance
2. **Backend Auditability:** Full audit trail for backend selection decisions
3. **No Breaking Changes:** All changes are additive (manifest, logging, tooling)
4. **Incremental Safety:** Logging and metadata now, enforcement deferred
5. **Tooling Independence:** Performance ledger usable outside CI/CD
6. **Baseline Governance:** Manual approval prevents baseline inflation

### Negative

1. **No Enforcement Yet:** Backend mismatch still allowed (deferred to v2.1.0)
2. **Manual Baseline Workflow:** Requires Architect approval for baseline updates
3. **Tooling Maintenance:** New script to maintain and test
4. **Manifest Schema Change:** Minor version bump required (v2.0.0 → v2.0.1)

### Risks

| Risk | Mitigation |
|------|------------|
| Performance ledger false positives | Conservative thresholds (p95 > 10%, mean > 15%), manual review |
| Baseline drift over time | Version control + manual approval, no auto-update |
| Backend logging spam | INFO level for success, WARNING only on fallback |
| Manifest bloat | Optional field, only included when backend_selection captured |

---

## Migration Plan

### For Existing Users

**No action required.** All changes are backward compatible:
- Manifests without `backend_selection` remain valid
- Performance ledger is opt-in tooling
- No CLI flags changed

### For Operators

**Baseline Capture Workflow:**
1. Run production batch with known-good configuration
2. Run performance ledger to capture baseline:
   ```bash
   python tools/performance_ledger.py \
     --manifests-dir ./output/prod_run/manifests \
     --output ./docs/performance/baselines/baseline_v2.0.0_da3_apex.json
   ```
3. Commit baseline to repo
4. Create PR for Architect review

**Regression Detection Workflow:**
1. Run experimental batch
2. Compare against baseline:
   ```bash
   python tools/performance_ledger.py \
     --baseline ./docs/performance/baselines/baseline_v2.0.0_da3_apex.json \
     --compare ./output/experimental_run/manifests \
     --output ./output/perf_report.md
   ```
3. Review report for regressions before merging changes

---

## Required Enforcement

### CI Gates (Phase 2)

- [x] `tools/performance_ledger.py` exists and is executable
- [x] Unit tests for statistics computation (mean, median, p90, p95)
- [x] Unit tests for regression detection logic
- [x] Unit tests for baseline schema validation
- [x] Baseline directory created: `docs/performance/baselines/`
- [ ] **NOT a CI gate** (manual workflow only)

### CI Gates (Phase 3)

- [x] `BackendSelectionMetadata` dataclass added to manifest.py
- [x] Unit tests for metadata serialization/deserialization
- [x] Integration test: manifest includes backend_selection
- [x] Integration test: truth-line logging emitted
- [x] No behavior changes (fallback still allowed)

### Documentation

- [x] Update `docs/input_hygiene.md` with Phase 1 completion notes
- [x] Create `docs/performance/README.md` with ledger usage guide
- [x] Update `docs/architecture/decisions/` with ADR-023
- [x] Update manifest schema documentation with new field

---

## Success Criteria

### Phase 2 Complete When:

- ✅ `tools/performance_ledger.py` can parse manifest directories
- ✅ Computes correct statistics (validated against known dataset)
- ✅ Detects regressions using defined thresholds
- ✅ Outputs markdown + JSON reports
- ✅ Baseline captured from v2.0.0 production run
- ✅ Documentation includes usage examples

### Phase 3 Complete When:

- ✅ Manifest includes `backend_selection` metadata
- ✅ Truth-line logging emitted on orchestrator startup
- ✅ Fallback warning logged when mismatch occurs
- ✅ Tests validate metadata capture and logging
- ✅ No breaking changes to existing workflows
- ✅ Documentation updated with schema changes

---

## Future Work (v2.1.0+)

### Backend Enforcement (Deferred)

Requires ADR-019 implementation:
- Implement `DepthBackendRegistry`
- Add `--strict-backend` flag (fail on mismatch)
- Add `--allow-backend-fallback` flag (permit with warning)
- Default behavior: fail on mismatch (breaking change)

### Input Discovery Enhancement (Optional)

Per expert recommendation:
- Write `output/.../input_discovery.json` with:
  - `included_count`, `excluded_count`
  - `exclusion_rules_hash` (for reproducibility)
  - `excluded_paths` with reason buckets
- Add to manifest as `input_discovery` field

### Artifact Boundary Check (Optional)

- If `output_dir` is inside `input_dir`: warn (default) or fail (strict)
- Prevents recursive self-ingestion
- Add `--strict-output-boundary` flag

---

## Alternatives Considered

### Alternative 1: CI-Enforced Performance Gates

**Rejected:** Premature. Need stable baselines first (requires 2-4 weeks of production data).

**Trade-offs:**
- ✅ Automated regression prevention
- ❌ False positives would block CI
- ❌ No historical baselines yet
- ❌ Threshold tuning needed

### Alternative 2: Hard-Fail Backend Mismatch Now

**Rejected:** Breaking change without backend registry (ADR-019 not implemented).

**Trade-offs:**
- ✅ Immediate correctness enforcement
- ❌ Breaks existing workflows (unknown frequency of fallback)
- ❌ No fallback mechanism for missing checkpoints
- ❌ Requires migration plan + deprecation period

### Alternative 3: Separate PRs for Phase 2 & 3

**Rejected:** Unnecessary overhead. Changes are non-overlapping and both additive.

**Trade-offs:**
- ✅ Smaller review surface per PR
- ❌ Two review cycles instead of one
- ❌ Delayed deployment (both needed for complete audit trail)
- ❌ More overhead for testing + merging

### Alternative 4: Performance Metrics in Manifest (vs Separate Tool)

**Rejected:** Bloats manifest schema with computed statistics.

**Trade-offs:**
- ✅ Co-located data
- ❌ Manifest bloat (every image gets batch-wide stats)
- ❌ No cross-run comparison capability
- ❌ Harder to aggregate across batches

---

## References

### Internal

- [PR #841: DA3 PIL Support](https://github.com/RC219805/Transformation_Portal/pull/841)
- [Commit 4761e2e5: Input Hygiene Implementation](https://github.com/RC219805/Transformation_Portal/commit/4761e2e5)
- [ADR-018: Depth Pro Integration](ADR-018-depth-pro-integration.md)
- [ADR-019: Backend Unification (Proposed)](ADR-019-depth-backend-unification.md)
- [Agent Governance Policy](../agent_governance.md)
- [Input Hygiene Documentation](../../input_hygiene.md)

### External

- [NumPy Percentile Documentation](https://numpy.org/doc/stable/reference/generated/numpy.percentile.html)
- [Python Statistics Module](https://docs.python.org/3/library/statistics.html)

---

**Document History**

- **2026-02-05:** ADR-023 approved (Architect architectural review)
  - Phase 1 completed (commit 4761e2e5)
  - Phase 2 approved (tooling, no CI gate)
  - Phase 3 approved (manifest + logging, defer enforcement)
