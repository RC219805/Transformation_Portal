# APEX Phase 3: Backend-Aware Dependency Validation - Implementation Summary

**Issue**: #875
**Branch**: `feat/apex-backend-aware-deps`
**Date**: 2026-02-08

## What Was Implemented

### 1. Backend Protocol Extension

Added `required_packages()` method to the `DepthBackend` protocol:
- Returns list of import module names required by backend
- torch is handled separately by APEX runner (always required)
- Backends only declare additional dependencies

**Files Modified**:
- `src/transformation_portal/depth/backends/protocol.py`

### 2. Backend Implementations

Implemented `required_packages()` for existing backends:

- **DA3Backend**: Returns `["transformers"]` (HuggingFace model)
- **DepthProBackend**: Returns `["depth_pro"]` (Apple's depth_pro package)

**Files Modified**:
- `src/transformation_portal/depth/backends/da3.py`
- `src/transformation_portal/depth/backends/depth_pro.py`

### 3. APEX Matrix Runner Update

Updated `check_ml_dependencies()` to be backend-aware:
- Now takes `backend_id: str` parameter
- Fetches backend from registry
- Constructs requirements as `["torch"] + backend.required_packages()`
- Dedupes while preserving order
- Catches all Exception types (not just ImportError) for broken installs

**Key Changes**:
- torch always required for real execution
- transformers only required for HF-based backends
- Backend-specific error messages include backend ID
- Fallback to strict check (torch + transformers) for unknown backends

**Files Modified**:
- `scripts/apex_matrix_runner.py`

### 4. Module Exports

Exported `get_registry()` from backends module for testing/inspection:

**Files Modified**:
- `src/transformation_portal/depth/backends/__init__.py`

## Current Status

✅ **Core Implementation Complete**:
- Protocol defined
- Existing backends implemented
- APEX runner integrated
- Backward compatible (existing HF backends still require transformers)

⏳ **Test Coverage** (Partial):
- Created `tests/test_apex_backend_deps.py` with 8 self-contained unit tests
- All tests passing (backend-aware dependency resolution validated)
- Tests cover: HF backends, non-HF backends, broken installs, unknown backends

## Example Usage

```python
# DA3 backend (requires transformers)
python scripts/apex_matrix_runner.py \
  --backend-id da3 \
  --input-dir ./tests/fixtures \
  --sample-size 3

# Error if transformers missing:
# "Backend 'da3' requires ML dependencies: transformers"
```

```python
# Future non-HF backend (no transformers required)
python scripts/apex_matrix_runner.py \
  --backend-id onnx \
  --input-dir ./tests/fixtures \
  --sample-size 3

# Only requires torch, not transformers
```

## Benefits

1. **No False Negatives**: Non-HF backends won't fail dependency check for transformers
2. **Clear Error Messages**: Backend ID included in error messages
3. **Extensible**: New backends just implement `required_packages()`
4. **Backward Compatible**: Existing DA3/Depth Pro behavior unchanged

## Next Steps (Issue #875 Checklist)

Remaining work tracked in Issue #875:

- [ ] Refine test mocking strategy for `check_ml_dependencies()`
- [ ] Add integration test with actual backend instances
- [ ] Update Phase 2 completion report to reference Phase 3
- [ ] Document `required_packages()` protocol in backend docs

## Files Changed

```
scripts/apex_matrix_runner.py                          +65 -27
src/transformation_portal/depth/backends/__init__.py   +2 -1
src/transformation_portal/depth/backends/da3.py        +10 -0
src/transformation_portal/depth/backends/depth_pro.py  +10 -0
src/transformation_portal/depth/backends/protocol.py   +22 -0
tests/test_apex_backend_deps.py (new)                  +167 -0
```

## Validation

- ✅ Python syntax checks pass
- ✅ Existing APEX tests pass (30 passed, 1 skipped)
- ✅ No regressions in contract verification or aggregation tests
- ✅ New backend dependency tests pass (8/8)
- ✅ Local and CI test suite green after classmethod fix
