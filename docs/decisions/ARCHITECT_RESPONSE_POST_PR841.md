# Post-PR #841 Hardening: Architectural Review Complete

**Date:** 2026-02-05  
**Authority:** Transformation Portal Architect  
**Status:** ✅ APPROVED - Ready for Implementation

---

## Executive Summary

I have reviewed the expert's "make-it-bulletproof" analysis and **APPROVED** the three-phase hardening plan with architectural guidance and implementation boundaries.

**Decisions:**
- ✅ Phase 1: Input Hygiene (COMPLETED - commit 4761e2e5)
- ✅ Phase 2: Performance Ledger (APPROVED - tooling only, no CI gate)
- ✅ Phase 3: Backend Selection Truth (APPROVED - manifest + logging, defer enforcement)

**Implementation Strategy:** Parallel implementation (Option B) - both Phase 2 & 3 in single PR.

---

## Key Architectural Decisions

### 1. Performance Ledger (Phase 2)

**✅ APPROVED as standalone tooling**

**Scope:**
- Create `tools/performance_ledger.py` (standalone script)
- Parse manifests, compute statistics (mean, median, p90, p95, min, max)
- Compare against baseline with regression detection
- Output markdown (human) + JSON (machine)

**Regression Thresholds:**
```python
REGRESSION_THRESHOLDS = {
    "p95_worsening_pct": 10,      # p95 > 10% slower = regression
    "mean_worsening_pct": 15,     # mean > 15% slower = regression
    "failure_rate_increase": 0,   # Any failures = regression
}
```

**Baseline Governance:**
- Location: `docs/performance/baselines/`
- Format: `baseline_{version}_{backend}_{tier}.json`
- Update policy: Manual approval required (Architect review)
- **NOT a CI gate** (manual workflow only for now)

**Rationale:**
- Prove value before enforcement
- No orchestrator changes = zero risk
- Reversible if not useful

---

### 2. Backend Selection Truth (Phase 3)

**✅ APPROVED with additive changes only**

**Phase 3A: Manifest Enhancement**
Add `BackendSelectionMetadata` to manifest schema:
```python
@dataclass
class BackendSelectionMetadata:
    requested_backend: Optional[str]      # User-specified or None (auto)
    resolved_backend: str                 # Actual backend used
    resolution_status: str                # "success", "fallback", "error"
    resolution_reason: Optional[str]      # Why fallback occurred
    model_id: str                         # HuggingFace model ID
    device: str                           # Resolved device (mps/cuda/cpu)
```

**Phase 3B: Truth-Line Logging**
Add startup logging to orchestrator:
```
INFO: Backend selection: requested=depth_pro resolved=depth_anything_v3 status=fallback device=mps
WARNING: Backend fallback: depth_pro checkpoint missing: checkpoints/depth_pro.pt
```

**Phase 3C: Enforcement (DEFERRED to v2.1.0)**
Future behavior (not in this PR):
- Add `--strict-backend` flag: fail on mismatch
- Add `--allow-backend-fallback` flag: permit with warning
- Default: fail on mismatch (breaking change, requires ADR + migration)

**Rationale for Deferral:**
- ADR-019 (Backend Registry) not yet implemented
- Breaking change requires validation period
- Logging + manifest now, enforcement later

---

### 3. Implementation Strategy

**✅ APPROVED: Option B (Parallel Implementation)**

**Rationale:**
- Phase 2 (performance ledger) = pure tooling, zero orchestrator risk
- Phase 3A/B (manifest + logging) = additive only, no behavior change
- No coupling between phases (different modules, different outputs)
- Single PR enables holistic testing

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

## Deliverables Created

### 1. Architecture Decision Record
**File:** `docs/architecture/decisions/ADR-023-post-pr841-hardening.md`

Comprehensive ADR documenting:
- Context and problem statement
- Decisions for Phase 2 & 3
- Regression threshold policy
- Baseline governance rules
- Manifest schema changes
- Migration plan and consequences
- Success criteria

### 2. Performance Monitoring Guide
**File:** `docs/performance/README.md`

User-facing documentation:
- Quick start guide
- Baseline management workflow
- Regression threshold interpretation
- Usage examples (baseline capture, comparison, nightly tracking)
- Troubleshooting guide

### 3. Baseline Directory Structure
**Files:**
- `docs/performance/baselines/README.md` (baseline governance)
- `docs/performance/baselines/` (directory for versioned baselines)

### 4. Performance Ledger Tool Skeleton
**File:** `tools/performance_ledger.py`

Implementation skeleton with:
- Dataclasses (EnvironmentMetadata, Statistics, Baseline, Regression)
- Core functions (parse_manifests, compute_statistics, detect_regressions, format_markdown)
- CLI argument parsing
- TODO markers for manifest schema integration

### 5. Implementation Guide
**File:** `docs/architecture/decisions/ADR-023-implementation-guide.md`

Specialist-facing guide with:
- Step-by-step implementation checklist
- Code snippets for manifest.py and orchestrator.py
- Testing requirements (unit + integration)
- Manual testing procedures
- Rollback plan
- Success criteria

---

## Validation of Expert's Analysis

### ✅ Agreed With Expert

1. **Prioritization:** Phase 1 (input hygiene) → Phase 2 (ledger) → Phase 3 (backend truth)
2. **Regression Thresholds:** p95 > 10%, mean > 15%, failure_rate > 0%
3. **Environment Tracking:** OS, Python, torch, device metadata
4. **Baseline Governance:** Manual approval, no auto-update
5. **Backend Truth Line:** Log requested vs resolved on startup

### 📝 Architectural Refinements

1. **Phase 2 CI Integration:** Expert suggested immediate CI gate, **Architect deferred** (need stable baselines first, 2-4 weeks production data)
2. **Phase 3 Enforcement:** Expert suggested immediate hard-fail on mismatch, **Architect deferred** (requires ADR-019 backend registry, breaking change needs migration)
3. **Implementation Strategy:** Expert suggested sequential PRs, **Architect approved parallel** (non-overlapping changes, single review cycle)

### 🔧 Additional Guidance Provided

1. **Baseline Storage:** Versioned in `docs/performance/baselines/` (not `tools/baselines/`)
2. **Manifest Schema Version:** Bump to v2.0.1 (additive change, backward compatible)
3. **Rollback Safety:** All changes additive, single revert command
4. **Testing Strategy:** Unit tests for ledger, integration tests for manifest + logging

---

## Implementation Readiness

### Prerequisites Satisfied

- [x] Phase 1 completed (input hygiene)
- [x] Production validation successful (20/20 images)
- [x] Expert analysis reviewed
- [x] Architectural decisions documented (ADR-023)
- [x] Implementation skeleton created

### Ready to Implement

**Specialist can proceed with:**
1. Completing `tools/performance_ledger.py` (manifest parsing + main workflow)
2. Adding `BackendSelectionMetadata` to `manifest.py`
3. Modifying `orchestrator.py` to capture + log backend selection
4. Writing unit tests for both phases
5. Integration testing with real batch runs

**Implementation guide:** `docs/architecture/decisions/ADR-023-implementation-guide.md`

---

## Risk Assessment

### Low Risk (Phase 2)

- **Pure tooling** - no orchestrator changes
- **Opt-in workflow** - not a CI gate
- **Reversible** - delete tool if not useful

### Low Risk (Phase 3)

- **Additive changes** - new manifest field, logging
- **No behavior changes** - fallback still allowed
- **Backward compatible** - old manifests remain valid

### Mitigations in Place

1. **No enforcement yet:** Backend mismatch logged, not blocked (defer to v2.1.0)
2. **Manual baselines:** Architect approval prevents baseline inflation
3. **Conservative thresholds:** p95 > 10%, mean > 15% (reduces false positives)
4. **Rollback plan:** Single commit revert restores prior state

---

## Success Criteria

### Phase 2 Complete When:

- ✅ `tools/performance_ledger.py` parses manifests correctly
- ✅ Statistics computation validated against known dataset
- ✅ Regression detection works with defined thresholds
- ✅ Baseline can be captured from v2.0.0 production run
- ✅ Unit tests pass
- ✅ Documentation complete

### Phase 3 Complete When:

- ✅ Manifest includes `backend_selection` metadata
- ✅ Truth-line logging emitted on startup
- ✅ Fallback warning logged when mismatch occurs
- ✅ Unit tests pass
- ✅ Integration test validates logging + manifest
- ✅ Backward compatibility verified

### Both Phases Complete When:

- ✅ All tests passing
- ✅ CI green
- ✅ No breaking changes to existing workflows
- ✅ Architect final review approved

---

## Next Steps for Specialist

1. **Read Implementation Guide:** `docs/architecture/decisions/ADR-023-implementation-guide.md`
2. **Complete Phase 2:** Implement manifest parsing in `performance_ledger.py`
3. **Complete Phase 3:** Add backend selection to manifest + orchestrator
4. **Write Tests:** Unit + integration tests per checklist
5. **Manual Testing:** Run full batch, verify logging + manifest
6. **Submit PR:** Single PR with both Phase 2 & 3 (parallel implementation)

---

## Conclusion

The expert's analysis is sound and actionable. I have provided architectural boundaries, governance rules, and implementation guidance to ensure safe, reversible deployment.

**Approval:** Proceed with implementation per ADR-023 and implementation guide.

**Questions?** Escalate to Architect before deviating from approved plan.

---

**Files Changed:**
- Created: `docs/architecture/decisions/ADR-023-post-pr841-hardening.md`
- Created: `docs/architecture/decisions/ADR-023-implementation-guide.md`
- Created: `docs/performance/README.md`
- Created: `docs/performance/baselines/README.md`
- Created: `tools/performance_ledger.py` (skeleton)

**Status:** SUCCEEDED
