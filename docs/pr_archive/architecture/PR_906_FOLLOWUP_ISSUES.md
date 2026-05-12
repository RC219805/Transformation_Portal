# PR #906 Follow-Up Issues

**Source:** Final Architectural Verification (docs/pr_archive/architecture/PR_906_FINAL_VERIFICATION.md)
**Created:** 2026-02-11
**Priority:** MEDIUM (complete before Phase 2)
**Status:** NOT STARTED

---

## Overview

While PR #906 successfully fixes all three critical bugs (P1-A, P1-B, P2-C), the final architectural verification identified **four additional issues** that should be addressed before Phase 2 begins.

These are **non-blocking** for merging PR #906, but should be fixed promptly.

---

## Issue #4: Ensemble Test Uses Duplicate Model Names

**Severity:** MEDIUM
**Category:** Test Coverage Gap
**File:** `tests/depth/backends/test_ensemble.py:118-128`

### Problem
```python
def test_variance_weighted_fusion_synthetic(self):
    models = [
        ModelConfig(name="synthetic", weight=0.5),
        ModelConfig(name="synthetic", weight=0.5),  # ❌ Same name
    ]
```

When `_run_models()` stores results in a dict using `results[model_config.name] = result`, the second model overwrites the first. The test only actually tests with **one model**, not two.

### Impact
- Reduced test coverage for multi-model fusion
- Test claims to test variance fusion but only uses one model
- Could miss bugs in multi-model scenarios

### Recommended Fix

**Option A (Preferred):** Use distinct model names in test:
```python
models = [
    ModelConfig(name="synthetic", weight=0.5),
    ModelConfig(name="synthetic_2", weight=0.5),
]
```

**Option B:** Properly mock two different backends that return different results.

### Notes
- The P1-B test (`test_variance_weighted_fusion_actually_uses_variance`) correctly uses distinct names ("model_a", "model_b"), so the fusion algorithm IS adequately tested
- This is a quality improvement, not a critical fix

---

## Issue #5: LinearDecoder `validate_contract=False` Is Non-Functional

**Severity:** MEDIUM
**Category:** API Contract Violation
**File:** `src/transformation_portal/spatial_ai/ingest/linear_decoder.py`

### Problem

The `LinearDecoder` constructor accepts a `validate_contract` parameter to allow gamma override:

```python
def __init__(self, gamma: float = 1.0, validate_contract: bool = True):
    if validate_contract and abs(gamma - 1.0) > 1e-6:
        raise ValueError(...)  # Only raised if validate_contract=True
```

However, `LinearIngestResult.__post_init__` always enforces gamma==1.0:

```python
def __post_init__(self):
    if abs(self.gamma - 1.0) > 1e-6:
        raise ValueError(...)  # ❌ Always raises, no awareness of validate_contract
```

### Impact
- `validate_contract=False` doesn't work as documented
- Creates `LinearDecoder(gamma=1.5, validate_contract=False)` succeeds, but calling `.decode()` raises error
- API is misleading

### Recommended Fix

**Option A (Preferred):** Remove `validate_contract` parameter entirely:
```python
class LinearDecoder:
    def __init__(self, gamma: float = 1.0, bit_depth: int = 32):
        if abs(gamma - 1.0) > 1e-6:
            raise ValueError("Linear ingest requires gamma=1.0 (SpatialCaptureV1 contract)")
        # ...
```

**Rationale:** If gamma==1.0 is non-negotiable for the SpatialCaptureV1 contract, don't pretend it's overrideable.

**Option B:** Thread `validate_contract` through to result:
```python
@dataclass
class LinearIngestResult:
    # ... existing fields ...
    _validate_contract: bool = field(default=True, repr=False)

    def __post_init__(self):
        if self._validate_contract and abs(self.gamma - 1.0) > 1e-6:
            raise ValueError(...)
```

Then pass it when creating result:
```python
result = LinearIngestResult(
    # ... fields ...
    _validate_contract=self.validate_contract
)
```

### Notes
- Phase 1 doesn't rely on gamma overrides, so not blocking
- The enforcement itself is correct; the API is just confusing

---

## Issue #6: EXR Fallback Clips HDR Values

**Severity:** MEDIUM-LOW
**Category:** Documentation/Honesty Issue
**File:** `src/transformation_portal/spatial_ai/ingest/linear_decoder.py:442`

### Problem

When OpenEXR is not installed, `_save_exr` falls back to 16-bit TIFF and **clips HDR values >1.0**:

```python
except ImportError:
    logger.warning("OpenEXR not available, using TIFF fallback for HDR output")
    # Convert to uint16 for TIFF (lossy for HDR >1.0)
    img_uint16 = np.clip(linear_rgb * 65535, 0, 65535).astype(np.uint16)
    img = Image.fromarray(img_uint16, mode="RGB")
    img.save(output_path, format="TIFF", compression="lzw")
```

### Impact
- Violates "HDR preservation" claim if OpenEXR not installed
- Research users may unknowingly train on clipped data
- Warning is printed but data is already corrupted

### Example
```python
linear_rgb = np.array([[[2.0, 0.5, 0.3]]])  # HDR value 2.0
# Fallback: 2.0 * 65535 = 131070 → clipped to 65535 → normalized = 1.0
# Result: 2.0 becomes 1.0 (50% data loss)
```

### Recommended Fix

**Option A (Preferred):** Fail loudly instead of silently clipping:
```python
except ImportError:
    # Check if we have HDR data
    if linear_rgb.max() > 1.0:
        raise RuntimeError(
            f"HDR data detected (max value: {linear_rgb.max():.2f}) but OpenEXR not installed.\n"
            "Cannot preserve HDR range in fallback TIFF format.\n\n"
            "Install OpenEXR: pip install OpenEXR\n"
            "Or use emit_exr=False to skip EXR export."
        )

    # SDR data: safe to save as 16-bit TIFF
    logger.warning("OpenEXR not available, using TIFF fallback (SDR data only)")
    img_uint16 = np.clip(linear_rgb * 65535, 0, 65535).astype(np.uint16)
    # ...
```

**Option B:** Use 32-bit float TIFF (preserves HDR, less portable):
```python
except ImportError:
    logger.warning("OpenEXR not available, using 32-bit float TIFF fallback")
    output_path = output_dir / f"{stem}_linear.tiff"

    # Save each channel as 32-bit float TIFF
    # Note: PIL doesn't support multi-channel float TIFF, need to use tifffile
    import tifffile
    tifffile.imwrite(output_path, linear_rgb, compression='lzw', photometric='rgb')
```

**Option C:** Document limitation in ADR-026 and require OpenEXR installation.

### Notes
- Phase 1 has OpenEXR in `requirements.txt`, so this path shouldn't execute
- More of a defensive coding / research honesty issue
- Could affect future users who skip optional dependencies

---

## Issue #7: `required_packages()` Docstring Mismatch

**Severity:** LOW
**Category:** Documentation Inconsistency
**File:** `src/transformation_portal/depth/backends/ensemble.py:503-512`

### Problem

Docstring says "torch + transformers", code returns only `["transformers"]`:

```python
@classmethod
def required_packages(cls) -> list[str]:
    """Return required import modules for ensemble.

    Ensemble requires at least torch + transformers (for DA3).  # ❌ Docstring
    Depth Pro is optional (graceful degradation).

    Returns:
        List of required module names.
    """
    return ["transformers"]  # ❌ Only transformers
```

### Impact
- Minor documentation inconsistency
- Code is likely correct (torch handled by APEX runner per protocol line 219)
- Could confuse future maintainers

### Recommended Fix

Update docstring to match code:
```python
@classmethod
def required_packages(cls) -> list[str]:
    """Return required import modules for ensemble.

    Ensemble requires transformers (for DA3 backend).
    torch is handled by the APEX runner and not listed here.
    Depth Pro is optional (graceful degradation).

    Returns:
        List of required module names (["transformers"]).
    """
    return ["transformers"]
```

### Notes
- Lowest priority issue
- No functional impact
- Should align with DepthBackend protocol documentation

---

## Implementation Plan

### Phase 1: Quick Wins (Est: 1-2 hours)
1. Fix Issue #7 (docstring) - trivial
2. Fix Issue #4 (test names) - simple refactor

### Phase 2: API Decisions (Est: 2-3 hours)
3. Fix Issue #5 (validate_contract) - requires decision on Option A vs B
4. Fix Issue #6 (EXR fallback) - requires decision on failure mode

### Testing
- Add regression tests for issues #5 and #6
- Verify existing tests still pass
- Run full APEX test suite

### Documentation
- Update ADR-026 if behavior changes
- Update README if API changes
- Add notes to CHANGELOG

---

## Acceptance Criteria

### Issue #4
- [ ] `test_variance_weighted_fusion_synthetic` uses distinct model names
- [ ] Test actually exercises multi-model fusion
- [ ] Test still passes

### Issue #5
- [ ] Decision made: remove parameter OR thread through to result
- [ ] API behavior matches documentation
- [ ] Regression test added

### Issue #6
- [ ] Decision made: fail loudly OR use 32-bit TIFF OR document limitation
- [ ] HDR data not silently corrupted
- [ ] Clear error message if OpenEXR required
- [ ] Regression test added

### Issue #7
- [ ] Docstring matches actual return value
- [ ] Aligns with DepthBackend protocol documentation

---

## Review Notes

**Architect Approval Required:** Yes (for issues #5 and #6 due to API/behavior changes)
**Specialist Implementation:** Yes (can implement after Architect decision)
**Target Completion:** Before Phase 2 work begins
**Blocking:** No (for Phase 1 merge)

---

## References

- Full analysis: `docs/pr_archive/architecture/PR_906_FINAL_VERIFICATION.md`
- Original review: `docs/pr_archive/architecture/PR_906_ARCHITECTURAL_REVIEW.md`
- PR #906: feat/apex-research-ultra-phase1
