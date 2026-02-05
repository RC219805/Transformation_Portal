# ADR-024: Backend Selection Enforcement Strategy

**Status:** Approved
**Date:** 2026-02-05
**Authority:** Transformation Portal Architect
**Supersedes:** None
**Related:** ADR-023 (Post-PR #841 Hardening), ADR-019 (Backend Unification - Proposed)

---

## Executive Summary

This ADR defines the enforcement strategy for backend selection mismatches, addressing the deferred enforcement from ADR-023 Phase 3.

**Key Decision:** **REJECT immediate enforcement**. Current logging and manifest metadata are sufficient. Defer enforcement to v2.1.0 when ADR-019 (Backend Registry) is implemented with proper fallback mechanisms.

**Rationale:** Enforcement without fallback infrastructure creates brittleness. The current warning-based approach provides transparency without blocking legitimate workflows.

---

## Context

### Current State (Post-ADR-023 Phase 3)

ADR-023 Phase 3 implemented:
- ✅ `BackendSelectionMetadata` in manifest schema
- ✅ Truth-line logging (`INFO` on success, `WARNING` on fallback)
- ✅ Full audit trail of requested vs resolved backend
- ❌ **No enforcement** (fallback always allowed)

**Current Behavior:**
```bash
# User requests Depth Pro (not yet available)
lux-depth-v3 --depth-backend depth_pro input/ output/

# System logs:
INFO: Backend selection: requested=depth_pro resolved=depth_anything_v3 status=fallback device=mps
WARNING: Backend fallback: Requested 'depth_pro' not available, using 'depth_anything_v3' (ADR-019 not yet implemented)

# Process continues successfully with DA3
```

**Manifest Output:**
```json
{
  "backend_selection": {
    "requested_backend": "depth_pro",
    "resolved_backend": "depth_anything_v3",
    "resolution_status": "fallback",
    "resolution_reason": "Requested 'depth_pro' not available, using 'depth_anything_v3' (ADR-019 not yet implemented)",
    "model_id": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
    "device": "mps"
  }
}
```

### User Request

The user requested implementation of:
1. **Depth Pro routing fix**: "ensure requested backend matches resolved backend"
2. **Resolution caps**: "safe inference resolution limits"

### Problem Analysis

**Is Backend Routing Actually Broken?**

No. The current behavior is:
- **Transparent**: Logs show exactly what happened
- **Auditable**: Manifest records the decision
- **Predictable**: Always falls back to DA3 (the only available backend)
- **Non-blocking**: Users can proceed with work

**What Would Enforcement Fix?**

Enforcement would catch:
- User typos in `--depth-backend` flag (e.g., `depth_pra` instead of `depth_pro`)
- Misconfigured CI/automation expecting specific backend
- Missing checkpoint files when backend is required

**What Would Enforcement Break?**

Without fallback infrastructure (ADR-019):
- Users cannot gracefully degrade if preferred backend unavailable
- No ability to express "try X, fallback to Y if unavailable"
- Checkpoint availability becomes hard dependency (brittle)
- Testing becomes harder (requires all backends present)

---

## Decision

### Depth Pro Routing: **REJECT Enforcement (Defer to v2.1.0)**

**Do NOT implement enforcement now.** Current logging and manifest metadata are sufficient.

**Rationale:**

1. **No Backend Registry Yet**: ADR-019 (DepthBackendRegistry) is not implemented
2. **No Fallback Infrastructure**: No way to express "preferred backend with fallback"
3. **Only One Backend Available**: DA3 is the only backend; enforcement would just fail
4. **Current Solution is Adequate**: Warnings + manifest metadata provide transparency
5. **Enforcement is Breaking Change**: Requires ADR approval + migration plan

**When to Revisit:**

Revisit enforcement in v2.1.0 when:
- ADR-019 implemented (DepthBackendRegistry with backend availability checking)
- Depth Pro integration complete (second backend available)
- Fallback policies defined (prefer X, fallback to Y with warning)
- Backend availability checking implemented (fail fast if required backend missing)

**Future Design (v2.1.0):**

```python
@dataclass
class EnhanceConfig:
    # Current (backward compatible default)
    strict_backend: bool = False  # New flag, defaults to permissive

    # Alternative naming options:
    # enforce_backend: bool = False
    # allow_backend_fallback: bool = True  # Inverted logic
```

**Future Behavior:**

```bash
# Strict mode (fail on mismatch)
lux-depth-v3 --strict-backend --depth-backend depth_pro input/ output/
# ERROR: Backend mismatch: requested 'depth_pro', resolved 'depth_anything_v3'
# Reason: Depth Pro checkpoint not found: checkpoints/depth_pro.pt
# Solutions:
#   - Download checkpoint: lux-depth-v3 --download-models depth_pro
#   - Allow fallback: Remove --strict-backend flag
#   - Use DA3 explicitly: --depth-backend da3

# Permissive mode (default, current behavior)
lux-depth-v3 --depth-backend depth_pro input/ output/
# WARNING: Backend fallback: requested 'depth_pro', using 'depth_anything_v3'
# Process continues...
```

---

### Resolution Caps: **REJECT Implementation**

**Do NOT implement resolution caps.** Insufficient evidence of need.

**Evidence from Production Validation:**

From ADR-023 validation (20 images):
- **Median**: 11.82s
- **p90**: 28.50s
- **p95**: 30.43s
- **Max**: 30.83s (PrimaryBedroom images)
- **Success rate**: 100%

**Analysis:**

1. **No Performance Problem**: Only 10% of images exceed 25s (2/20)
2. **Acceptable Runtimes**: 30s max is well within reasonable bounds
3. **No SLA Violations**: No user complaints or timeout failures
4. **Expert Opinion**: Called it "premature optimization" in original analysis
5. **Risk > Benefit**: Quality degradation risk outweighs unproven benefit

**What Would Resolution Caps Fix?**

Hypothetically:
- Prevent worst-case timeouts (no evidence this is a problem)
- Cap memory usage (no evidence of OOM failures)
- Enforce time budgets (no time budget defined)

**What Would Resolution Caps Break?**

Definitely:
- **Quality degradation** on high-resolution inputs (known cost)
- **User surprise** if images silently downscaled
- **Complexity** in code and configuration
- **Maintenance burden** of tuning caps per backend/tier

**When to Revisit:**

Revisit resolution caps if:
- Users report timeout failures (> 60s runtimes)
- OOM failures on specific hardware configurations
- SLA requirement established (e.g., "p95 < 20s")
- Performance regression detected in production

**Alternative Approach (If Needed Later):**

Instead of hard caps, implement:
1. **Resolution logging** in manifests (track correlation with runtime)
2. **Performance budgets** in quality tiers (apex = no limits, fast = aggressive caps)
3. **Opt-in downscaling** via `--max-resolution` flag (explicit user control)

**Example (If Implemented in v2.2.0):**

```python
@dataclass
class EnhanceConfig:
    # Opt-in resolution cap (None = no limit)
    max_resolution: Optional[int] = None  # Max width or height in pixels

    # Example usage:
    # --max-resolution 2048  (downscale inputs > 2048px on longest side)
```

---

## Consequences

### Positive

1. **Avoid Premature Enforcement**: No breaking changes until fallback infrastructure ready
2. **Maintain Flexibility**: Users can proceed despite backend unavailability
3. **Clear Audit Trail**: Existing logging and manifests provide transparency
4. **Defer Complexity**: Wait until ADR-019 implementation to add enforcement
5. **Prevent Quality Degradation**: No resolution downscaling without proven need

### Negative

1. **No Hard Enforcement**: User typos in `--depth-backend` won't fail loudly
2. **Deferred Enforcement**: v2.1.0 earliest for strict mode
3. **No Resolution Protection**: Worst-case runtimes unbounded (current max: 30s)

### Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| User confusion about fallback | Low | Low | Clear warning logging, manifest metadata |
| Timeout on extreme resolutions | Low | Medium | Monitor production, add caps if needed |
| Backend enforcement delay | Certain | Low | Acceptable; current solution works |

---

## Alternatives Considered

### Alternative 1: Implement `--strict-backend` Now

Add opt-in enforcement flag immediately.

**Rejected Reasons:**
- No backend registry to check availability
- No fallback policies defined
- Only one backend available (DA3)
- Enforcement would just fail immediately
- Better to wait for ADR-019 implementation

### Alternative 2: Implement Quality-Tier-Based Resolution Caps

Add resolution caps tied to quality tiers:
- `fast`: 1024px max
- `standard`: 2048px max
- `apex`: no limits

**Rejected Reasons:**
- No evidence of performance problem
- Median runtime 11.82s is acceptable
- Only 10% of images > 25s
- Risk of quality degradation
- Premature optimization per expert

### Alternative 3: Add Resolution Logging Only

Log input resolution in manifest without caps.

**Rejected Reasons:**
- Adds manifest complexity for unproven benefit
- Can add later if performance regression detected
- Current metadata sufficient for debugging

### Alternative 4: Backend Enforcement via Config Validation

Validate backend availability in `EnhanceConfig.validate()`.

**Rejected Reasons:**
- Requires backend registry (ADR-019)
- No graceful fallback mechanism
- Breaking change without migration path

---

## Migration Plan

### Phase 1: Current State (v2.0.0) ✅ COMPLETE

- ✅ Truth-line logging (ADR-023 Phase 3)
- ✅ Backend metadata in manifests
- ✅ Fallback warnings
- ✅ No enforcement (permissive default)

### Phase 2: Backend Registry (v2.1.0) - ADR-019

Prerequisites for enforcement:
1. Implement `DepthBackendRegistry`
2. Add backend availability checking
3. Define fallback policies
4. Implement checkpoint discovery

### Phase 3: Opt-In Enforcement (v2.1.0)

1. Add `--strict-backend` flag (opt-in)
2. Fail if strict mode + backend mismatch
3. Provide actionable error messages
4. Comprehensive testing (unit + integration)
5. Documentation updates

### Phase 4: Default Enforcement (v3.0.0) - Breaking Change

1. Change default: `strict_backend = True`
2. Add `--allow-backend-fallback` flag for permissive mode
3. Deprecation period (2 minor versions)
4. Migration guide for users

### Resolution Caps: **NO MIGRATION PLAN** (Not Implementing)

If needed in future:
1. Gather production evidence (runtime vs resolution correlation)
2. Define SLA/performance requirements
3. Implement opt-in flag (`--max-resolution`)
4. Tie to quality tiers if appropriate

---

## Required Enforcement

### CI Gates

**No new CI gates required.** Current state is acceptable.

Existing enforcement (ADR-023):
- ✅ Backend metadata in manifests (schema validation)
- ✅ Truth-line logging (integration tests)
- ✅ Fallback warnings (log assertion tests)

### Documentation

- ✅ Update ADR-023 to reference ADR-024 for enforcement deferral
- ✅ Document decision to reject immediate enforcement
- ✅ Document decision to reject resolution caps
- ✅ Define criteria for future implementation

---

## Success Criteria

### Immediate (v2.0.0)

- ✅ User understands current state is intentional (not a bug)
- ✅ Clear architectural guidance provided
- ✅ No premature implementation of unproven features
- ✅ Path forward defined for v2.1.0

### Future (v2.1.0)

- ADR-019 implementation complete
- Backend registry with availability checking
- `--strict-backend` flag available (opt-in)
- Comprehensive error messages with solutions

### Future (v3.0.0)

- Default strict enforcement
- Fallback policies mature and tested
- Clear migration path for users

---

## Specific Answers to Implementation Questions

### 1. Flag naming (if implemented in v2.1.0)

**Recommendation:** `--strict-backend`

**Rationale:**
- Clear intent (strict = enforce)
- Matches existing `--strict-inputs` pattern
- Shorter than `--enforce-backend`
- More explicit than `--no-backend-fallback` (double negative)

### 2. Error messages (if implemented in v2.1.0)

**Yes, suggest solutions:**

```
ERROR: Backend mismatch in strict mode
Requested: depth_pro
Resolved: depth_anything_v3
Reason: Depth Pro checkpoint not found: checkpoints/depth_pro.pt

Solutions:
  1. Download checkpoint: lux-depth-v3 --download-models depth_pro
  2. Allow fallback: Remove --strict-backend flag
  3. Use DA3 explicitly: --depth-backend da3
```

### 3. Backend availability check location (v2.1.0)

**Recommendation:** New module `src/transformation_portal/depth/backends/registry.py`

**Rationale:**
- Aligns with ADR-019 design
- Centralized backend management
- Separate from `backend_selection.py` (policy vs implementation)

### 4. Deprecation path (v2.1.0 → v3.0.0)

**Yes, add deprecation warning:**

```python
if not config.strict_backend:
    warnings.warn(
        "Backend fallback will be disallowed by default in v3.0.0. "
        "Set strict_backend=True to adopt future behavior, or use "
        "--allow-backend-fallback flag to maintain permissive mode.",
        FutureWarning,
        stacklevel=2,
    )
```

### 5. Testing scope (v2.1.0)

**Unit tests:**
- Backend availability checking logic
- Config validation with strict mode
- Error message generation

**Integration tests:**
- Full pipeline with `--strict-backend` (expect failure)
- Full pipeline with fallback (expect warning)
- Backend registry selection

**CI:**
- Test strict mode in CI (expect failure when backend missing)
- Test permissive mode in CI (current default)
- Do NOT require strict mode by default (breaking change)

---

## References

### Internal ADRs

- [ADR-023: Post-PR #841 Hardening Strategy](ADR-023-post-pr841-hardening.md)
- [ADR-019: Depth Backend Unification (Proposed)](../ADR-019-depth-backend-unification.md)
- [Agent Governance Policy](../agent_governance.md)

### Performance Data

- [v2.0.0 Production Validation](../../performance/baselines/v2.0.0-post-hardening-validation.json)
- Median: 11.82s, p95: 30.43s, max: 30.83s (20 images, 100% success)

### Expert Analysis

From original expert analysis:
> "Your max runtimes are ~30s. If those correspond to huge resolutions, you can implement 'safe inference resolution' (max_side / max_pixels) to cap worst-case cost without harming the median much."

Expert assessment: "Premature optimization" unless proven necessary.

---

## Document History

- **2026-02-05:** ADR-024 created (Architect decision)
  - Reject immediate backend enforcement (defer to v2.1.0)
  - Reject resolution caps (insufficient evidence)
  - Define future implementation path (ADR-019 prerequisite)
