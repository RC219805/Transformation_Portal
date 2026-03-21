# ADR-019 Integration: Architect Approval Summary

**Date:** 2026-02-05
**Status:** ✅ APPROVED FOR IMMEDIATE INTEGRATION
**Authority:** Transformation Portal Architect
**Document:** ADR-019-REVISED-DECISION.md

---

## Executive Decision

**APPROVE MINIMAL INTEGRATION** of ADR-019 Backend Registry orchestrator integration.

**Scope:** Orchestrator + DA3Backend adapter + tests + docs (~2 dev days)

**Rationale:** Depth Pro checkpoint verified operational, all blockers resolved.

---

## What Changed?

### Previous Decision (Deferral)
**From:** ADR-019-IMPLEMENTATION-STATUS.md
**Rationale:** "Depth Pro not operational" → defer to v2.1.0

### New Information
✅ **Checkpoint available:** `checkpoints/depth_pro.pt` (1.77 GB)
✅ **Dependencies installed:** `import depth_pro` succeeds
✅ **Backend operational:** `backend.ensure_available()` succeeds
✅ **Checkpoint verified:** Loads correctly, 1119 keys

### Revised Decision (Approval)
**From:** ADR-019-REVISED-DECISION.md
**Rationale:** All blockers resolved → proceed immediately

---

## Approved Scope

### ✅ In Scope (This PR)

1. **DA3Backend Adapter** (`src/transformation_portal/depth/backends/depth_anything_v3.py`)
   - Wrap existing `DA3InferenceEngine`
   - Implement `DepthBackend` protocol
   - Maintain backward compatibility

2. **Orchestrator Integration** (`src/transformation_portal/lux_depth_v3/orchestrator.py`)
   - Replace hardcoded `DA3InferenceEngine` with `DepthBackendRegistry`
   - Backend selection: `config.depth_backend or "depth_anything_v3"`
   - Fallback: `depth_pro → da3` if unavailable
   - Update metadata capture

3. **Backend Availability Checking**
   - Pre-flight: `backend.ensure_available()` on init
   - Graceful fallback with warning

4. **Tests**
   - Unit: `DA3Backend` adapter
   - Integration: Orchestrator with both backends
   - Regression: DA3 behavior unchanged
   - Fallback: `depth_pro → da3` graceful degradation

5. **Documentation**
   - README: `--depth-backend` usage
   - CLI reference updates
   - Depth Pro license/requirements guide

### ⏸️ Out of Scope (Future PRs)

- `--strict-backend` enforcement (ADR-024)
- `--list-backends` command
- Depth Pro preset configuration
- CI testing with Depth Pro
- Automatic checkpoint download

---

## Implementation Checklist

### Phase 1: DA3Backend Adapter (4 hours)
- [ ] Create `src/transformation_portal/depth/backends/depth_anything_v3.py`
- [ ] Implement `DepthBackend` protocol
- [ ] Wrap `DA3InferenceEngine`
- [ ] Write unit tests (`tests/unit/depth/backends/test_da3_backend.py`)
- [ ] Verify backward compatibility

### Phase 2: Orchestrator Integration (4 hours)
- [ ] Import `DepthBackendRegistry` in orchestrator
- [ ] Replace `DA3InferenceEngine` with `registry.get_backend()`
- [ ] Implement backend selection logic
- [ ] Implement fallback policy
- [ ] Update compute path: `self.depth_backend.compute()`
- [ ] Update metadata capture: `self.depth_backend.name`

### Phase 3: Testing (3 hours)
- [ ] Write integration tests (`tests/integration/test_orchestrator_backend_selection.py`)
- [ ] Test default backend (DA3)
- [ ] Test explicit DA3 selection
- [ ] Test Depth Pro selection (if checkpoint available)
- [ ] Test fallback behavior
- [ ] Run regression tests (ensure DA3 unchanged)
- [ ] Run license enforcement tests

### Phase 4: Documentation (2 hours)
- [ ] Update README.md with backend selection guide
- [ ] Add Depth Pro requirements section
- [ ] Document license flags
- [ ] Add CLI examples
- [ ] Update CLI reference

### Phase 5: Validation (3 hours)
- [ ] Manual test: Default backend
- [ ] Manual test: Explicit DA3
- [ ] Manual test: Depth Pro (if checkpoint available)
- [ ] Manual test: Fallback behavior
- [ ] Verify logs show backend selection
- [ ] Verify manifests record backend metadata

---

## Verification Commands

```bash
# 1. Verify Depth Pro operational
python -c "
from transformation_portal.depth.backends.depth_pro import DepthProBackend
backend = DepthProBackend()
backend.ensure_available()
print('✅ Depth Pro operational')
"

# 2. Test DA3 backend
pytest tests/unit/depth/backends/test_da3_backend.py -v

# 3. Test orchestrator integration
pytest tests/integration/test_orchestrator_backend_selection.py -v -m ml

# 4. Test default backend (DA3)
lux-depth-v3 --input-dir ./test_images --output-dir ./output_test

# 5. Test Depth Pro backend
lux-depth-v3 \
  --input-dir ./test_images \
  --output-dir ./output_depth_pro \
  --depth-backend depth_pro \
  --accept-apple-depth-pro-research-license true \
  --non-commercial-ok true

# 6. Test fallback
lux-depth-v3 \
  --input-dir ./test_images \
  --output-dir ./output_fallback \
  --depth-backend nonexistent_backend
# Should warn and use DA3
```

---

## Success Criteria

**Integration is COMPLETE when:**

- ✅ `DA3Backend` implements `DepthBackend` protocol
- ✅ Registry returns both `depth_anything_v3` and `depth_pro` backends
- ✅ Orchestrator uses registry (not hardcoded `DA3InferenceEngine`)
- ✅ Backend selection works via `--depth-backend` flag
- ✅ Fallback works: `depth_pro → da3` if unavailable
- ✅ Truth-line logs capture backend decisions
- ✅ Manifests record backend metadata
- ✅ All tests pass (unit + integration + regression)
- ✅ Documentation updated
- ✅ Manual validation successful for both backends

---

## Risk Assessment

### Risk Profile: LOW ✅

| Risk Factor | Status | Mitigation |
|-------------|--------|------------|
| Depth Pro unavailable | 🟢 **RESOLVED** | Verified operational |
| No second backend | 🟢 **RESOLVED** | Two backends ready |
| Breaks DA3 | 🟡 **LOW** | Thin wrapper + regression tests |
| No user value | 🟢 **RESOLVED** | Unlocks Depth Pro |

### Rollback Plan

If issues found:
1. Revert orchestrator changes
2. Keep `DA3InferenceEngine` direct call
3. Backend infrastructure remains (no breaking changes)

---

## Timeline

**Target:** Single PR, ~2 developer days (16 hours)

**Breakdown:**
- 🔧 4h: DA3Backend adapter + tests
- 🔧 4h: Orchestrator integration
- 🔧 3h: Integration tests
- 🔧 2h: Documentation
- 🔧 3h: Manual validation

---

## Architect Decision Record

**Decision:** APPROVED for immediate integration

**Justification:**
1. All technical blockers resolved
2. Risk profile favorable (LOW)
3. User value significant (unlocks Depth Pro)
4. Architecture validated with real backends
5. Minimal scope, surgical change
6. Comprehensive testing planned
7. Clear rollback path

**Alternatives Rejected:**
- ❌ Keep deferral: Delays user value, blocks architecture validation
- ❌ Full integration: Larger scope, ADR-024 is separate decision

**Enforcement:**
- This decision is binding
- Implementation must follow approved scope
- Tests must be comprehensive
- Documentation must be complete
- No scope creep beyond defined boundaries

**Sign-off:** Transformation Portal Architect, 2026-02-05

---

## References

**Primary Document:** [ADR-019-REVISED-DECISION.md](ADR-019-REVISED-DECISION.md)

**Related ADRs:**
- [ADR-019: Depth Backend Unification](../ADR-019-depth-backend-unification.md)
- [ADR-024: Backend Enforcement Strategy](ADR-024-backend-enforcement-strategy.md)
- [ADR-018: Depth Pro Integration](../ADR-018-depth-pro-integration.md)

**Superseded:**
- [ADR-019-IMPLEMENTATION-STATUS.md](ADR-019-IMPLEMENTATION-STATUS.md) (deferral decision)

---

## Next Actions

1. **Specialist:** Implement per approved scope
2. **Architect:** Review PR when ready
3. **User:** Manual validation after merge

**Questions/Escalations:** Direct to Architect via governance protocol
