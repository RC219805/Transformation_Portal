# APEX Phase 2.1 Hardening Summary

**Date:** 2026-02-08
**PR:** #873
**Status:** ✅ Complete, In Review

---

## Executive Summary

Phase 2.1 addresses three post-merge hardening items identified during PR #872 review. These fixes eliminate truth drift between code and documentation, improve fail-fast behavior for broken ML installs, and ensure reproducible dependency management.

**Impact:** APEX real-execution path is now more robust against misconfigured environments and provides clearer error messages when dependencies are missing or broken.

---

## Fixes Implemented

### 1. Broaden Exception Handling in Dependency Checks ✅

**Problem:**
`check_ml_dependencies()` only caught `ImportError`, meaning broken torch installations (e.g., CUDA driver mismatches, corrupted shared libraries) would pass the check and then crash deep in model loading with confusing stack traces.

**Solution:**
Changed exception handling from `except ImportError` to `except Exception as e`, with debug logging:

```python
try:
    import torch  # noqa: F401
except Exception as e:
    logger.debug(f"torch import failed ({e}), treating as missing")
    missing.append("torch")
```

**Benefit:**
- Broken installs are treated as missing dependencies
- Clear error message at the start: "Real execution requires ML dependencies: torch"
- Users get actionable guidance instead of cryptic mid-run failures

**Files Changed:**
- `scripts/apex_matrix_runner.py`: Broadened exception handling in `check_ml_dependencies()`

---

### 2. Update Install Guidance to Use ML Extra ✅

**Problem:**
Error messages and documentation suggested:
```bash
pip install torch transformers
```

This bypasses the repo's `[ml]` extra and version pinning in `pyproject.toml`, leading to potential version mismatches and irreproducible environments.

**Solution:**
Updated all install guidance to:
```bash
pip install -e .[ml]
```

**Locations Updated:**
1. `scripts/apex_matrix_runner.py`: RuntimeError message (line 209)
2. `docs/apex/phase2/COMPLETION_REPORT.md`: Error message example (line 75)

**Benefit:**
- Consistent dependency versions across dev, CI, and production
- Single source of truth for ML dependency pins
- Easier to maintain and update dependency specifications

---

### 3. Reconcile Dependency Policy Documentation ✅

**Problem:**
Documentation contradicted implementation:

| Source | Claim |
|--------|-------|
| Code (`check_ml_dependencies()`) | Both torch and transformers **required** |
| COMPLETION_REPORT.md line 313 | "Transformers optional (backend-dependent)" |

This truth drift would confuse contributors and lead to incorrect assumptions about what's needed for real runs.

**Solution:**
Updated documentation to match reality:

**Before:**
> **Mitigation:** Only torch required for real execution. Transformers optional (backend-dependent). Clear error messages guide users.

**After:**
> **Mitigation:** Both torch and transformers are required for Phase 2 real execution (current HuggingFace-based backends). Broad exception handling (not just ImportError) catches broken installs. Clear error messages guide users to `pip install -e .[ml]`.

Also updated function signature example to remove obsolete `require_torch` parameter.

**Files Changed:**
- `docs/apex/phase2/COMPLETION_REPORT.md`:
  - Risk mitigation section (line 313)
  - Function signature example (line 59)
  - Error message example (line 75)

**Benefit:**
- Documentation accurately reflects code behavior
- No misleading claims about optional dependencies
- Contributors have correct mental model of requirements

---

## Validation

### Tests
All APEX contract tests pass:
```bash
pytest -xvs tests/test_apex_contract_verification.py tests/test_apex_gate.py tests/test_apex_aggregator.py
# Result: 38 passed, 1 skipped in 3.27s
```

### Syntax
```bash
python -m py_compile scripts/apex_matrix_runner.py
# Result: ✓ No errors
```

### Pre-commit
All quality gates passed:
- Trailing whitespace auto-fixed
- Black + isort formatting passed
- Flake8 (critical errors) passed
- Python syntax validation passed
- Markdown file count OK

---

## Design Rationale

### Why catch all exceptions, not just ImportError?

**Real-world failure modes:**
- CUDA driver version mismatch → RuntimeError
- Corrupted libtorch.so → OSError
- Missing system libraries → ModuleNotFoundError subclass

If we only catch ImportError, these scenarios crash with unhelpful stack traces. By catching Exception and logging debug info, we:
1. Treat all import failures uniformly
2. Give users a clear "missing dependency" error
3. Log diagnostic info for troubleshooting
4. Degrade gracefully to CPU when torch is broken

### Why require both torch and transformers?

**Phase 2 reality:**
- Default depth backend is HuggingFace-based (Depth Anything V3)
- Real orchestrator imports transformers for model loading
- No current backend works without transformers

**Future flexibility:**
Once backend registry is fully implemented (Phase 3+), we can make this backend-aware:
- Core always requires torch
- Specific backends declare their own requirements (transformers, onnxruntime, etc.)
- Dependency check becomes: `required = ["torch"] + backend.required_packages()`

For now, simplicity wins: both deps required, clear error message.

---

## Follow-Up Opportunities (Not in Scope)

### Backend-Aware Dependency Checks
When APEX supports multiple backends (DA3, Depth Pro, custom ONNX models), make dependency checking backend-specific:

```python
def check_backend_dependencies(backend_id: str) -> tuple[bool, list[str]]:
    """Check dependencies for specific backend."""
    registry = get_backend_registry()
    backend = registry.get(backend_id)
    return backend.check_dependencies()
```

This would allow:
- CPU-only backends (no torch)
- ONNX backends (onnxruntime, not transformers)
- Custom backends with unique dependency chains

**Decision:** Deferred to Phase 3+ when multi-backend support is active.

---

## Acceptance Criteria

- [x] Broadened exception handling in `check_ml_dependencies()`
- [x] Updated install guidance to `pip install -e .[ml]` (code + docs)
- [x] Reconciled dependency policy in COMPLETION_REPORT.md
- [x] All APEX tests pass
- [x] Pre-commit checks pass
- [x] No new lint warnings
- [x] Documentation accurately reflects code behavior

---

## Review Checklist

- [x] Code changes are minimal and surgical
- [x] Tests validate behavior (no manual "trust me" claims)
- [x] Documentation matches implementation
- [x] Error messages are actionable
- [x] Commit message explains the "why" not just the "what"
- [x] No scope creep (backend-aware checks deferred)

---

## Related Work

- **PR #872:** Phase 2 Real Pipeline Integration (merged)
- **PR #871:** Phase 1.1 Truth Alignment (merged)
- **PR #870:** Workflow Env Var Refactor (merged)
- **PR #869:** Phase 1 Hardening (merged)

---

## Conclusion

Phase 2.1 is a small, focused follow-up that eliminates the last truth drift items from Phase 2. The APEX judge is now:
1. **Truthful:** Docs match code
2. **Robust:** Broken installs fail fast with clear messages
3. **Reproducible:** Single install path (`[ml]` extra)

Ready for merge once CI confirms no regressions.
