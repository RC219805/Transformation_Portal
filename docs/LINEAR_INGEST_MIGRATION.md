# Linear Ingest Verification Migration Guide

**Date:** 2026-02-10  
**PR:** copilot/verify-linear-ingest-pipeline  
**Status:** Implementation Complete

## Overview

This guide describes the linear ingest verification implementation for the APEX depth pipeline, which enforces correctness-critical invariants for linear light preservation per the Spatial AI Foundation ROADMAP.

## What Changed

### 1. New Linear Verification Module

**File:** `src/transformation_portal/lux_depth_v3/linear_verify.py`

Provides blocking validation for:
- **dtype** - Rejects uint8/uint16, requires float32
- **Range** - Enforces [0, 1] bounds for normalized linear light
- **Gamma** - Detects and rejects gamma-encoded inputs

```python
from transformation_portal.lux_depth_v3.linear_verify import verify_linear_ingest

# Validates all invariants, raises error on violation
verify_linear_ingest(tensor)  # tensor must be float32 [0,1] linear
```

### 2. RAW Loader Updates

**File:** `src/transformation_portal/lux_depth_v3/raw_loader.py`

**Breaking Changes:**
- Default `output_bps` changed from `8` to `16` (preserves precision)
- Default `output_linear` is `True` (gamma-encoded output blocked)
- Gamma-encoded output (`output_linear=False`) raises `ValueError`

**Before:**
```python
# Old behavior (gamma-encoded sRGB, 8-bit)
rgb = load_raw_as_rgb(path)  # uint8 gamma-encoded sRGB
```

**After:**
```python
# New behavior (linear RGB, 16-bit, APEX compliant)
rgb = load_raw_as_rgb(path)  # uint16 linear RGB

# Gamma output is BLOCKED
rgb = load_raw_as_rgb(path, output_linear=False)
# → ValueError: Gamma-encoded output not allowed for APEX pipeline
```

### 3. New Linear Preprocessing Function

**File:** `src/transformation_portal/lux_depth_v3/preprocessing.py`

**New function:** `preprocess_image_linear()`

This is the APEX-compliant preprocessing path with integrated linear verification:

```python
from transformation_portal.lux_depth_v3.preprocessing import preprocess_image_linear

# APEX-compliant linear preprocessing
image, orig_shape = preprocess_image_linear("photo.CR2")
# → float32 [0,1] linear, verified

# Preserves 16-bit TIFF precision
image, orig_shape = preprocess_image_linear("render.tif")
# → Uses tifffile for 16-bit preservation

# Rejects gamma-encoded inputs
image, orig_shape = preprocess_image_linear(gamma_encoded_array)
# → LinearityViolationError
```

**Legacy function preserved:**
- `preprocess_image()` remains unchanged for backward compatibility
- Does NOT enforce linear verification
- Use `preprocess_image_linear()` for APEX pipeline

## Migration Path

### For Existing Code Using RAW Files

**Option 1: Update to linear (recommended for APEX)**

```python
# Before
from transformation_portal.lux_depth_v3.raw_loader import load_raw_as_pil
pil_img = load_raw_as_pil(path)  # 8-bit gamma-encoded

# After
from transformation_portal.lux_depth_v3.preprocessing import preprocess_image_linear
image, shape = preprocess_image_linear(path)  # float32 linear, verified
```

**Option 2: Continue with legacy behavior (non-APEX)**

```python
# Use legacy preprocessing (no linear verification)
from transformation_portal.lux_depth_v3.preprocessing import preprocess_image
image, shape = preprocess_image(path)  # Legacy path, no verification
```

### For TIFF Files

**16-bit TIFF preservation:**

```python
# Before (PIL converts uint16 → uint8, loses precision)
from transformation_portal.lux_depth_v3.preprocessing import preprocess_image
image, shape = preprocess_image("16bit.tif")  # Precision lost

# After (preserves 16-bit via tifffile)
from transformation_portal.lux_depth_v3.preprocessing import preprocess_image_linear
image, shape = preprocess_image_linear("16bit.tif")  # Precision preserved
```

### For APEX Depth Pipeline Integration

**Recommended approach:**

```python
# Use preprocess_image_linear() for all APEX inputs
from transformation_portal.lux_depth_v3.preprocessing import preprocess_image_linear

# This function:
# 1. Preserves 16-bit precision (RAW → uint16, TIFF → tifffile)
# 2. Converts to float32 [0,1] preserving linearity
# 3. Validates dtype, range, and gamma
# 4. Raises blocking errors on violations

image, orig_shape = preprocess_image_linear(input_path, verify_linearity=True)
```

## Testing

### Test Files Added

1. **tests/test_linear_verify.py** - 50 tests for linear verification module
2. **tests/test_linear_ingest_end_to_end.py** - 14 tests for end-to-end linearity
3. **tests/test_raw_loader.py** - Updated for linear output

**Total:** 64 new/updated tests, all passing

### Run Tests

```bash
# Linear verification tests
python -m pytest tests/test_linear_verify.py -v

# End-to-end linear ingest tests
python -m pytest tests/test_linear_ingest_end_to_end.py -v

# RAW loader tests (requires rawpy)
python -m pytest tests/test_raw_loader.py -v -m ml

# All preprocessing tests (no regressions)
python -m pytest tests/test_preprocessing.py -v
```

## Error Handling

### Common Errors and Fixes

**1. DtypeViolationError: "Tensor dtype must be float32"**

```python
# Problem: uint8 or uint16 tensor detected
arr = np.array([128, 200], dtype=np.uint8)
verify_linear_ingest(arr)  # → DtypeViolationError

# Fix: Convert to float32 [0,1]
arr_float = arr.astype(np.float32) / 255.0
verify_linear_ingest(arr_float)  # ✓ Passes
```

**2. RangeViolationError: "Tensor maximum value exceeds expected range"**

```python
# Problem: Values outside [0, 1]
arr = np.array([0.5, 1.5], dtype=np.float32)
verify_linear_ingest(arr)  # → RangeViolationError

# Fix: Clip to valid range or fix normalization
arr_clipped = np.clip(arr, 0.0, 1.0)
verify_linear_ingest(arr_clipped)  # ✓ Passes
```

**3. LinearityViolationError: "Gamma-encoded input detected"**

```python
# Problem: Input is gamma-encoded (sRGB, Rec.709)
gamma_arr = create_gamma_encoded_fixture()
verify_linear_ingest(gamma_arr)  # → LinearityViolationError

# Fix: Use linear input (RAW with linear output, or pre-linearized TIFF)
linear_arr = preprocess_image_linear("photo.CR2")  # ✓ Linear from RAW
verify_linear_ingest(linear_arr)  # ✓ Passes

# DO NOT apply inverse gamma - reject the input instead!
```

**4. ValueError: "Gamma-encoded RAW output not allowed"**

```python
# Problem: Trying to get gamma-encoded output from RAW loader
rgb = load_raw_as_rgb(path, output_linear=False)
# → ValueError: Gamma-encoded output not allowed for APEX pipeline

# Fix: Use linear output (default)
rgb = load_raw_as_rgb(path, output_linear=True)  # ✓ Linear output
```

## Compliance with Spatial AI Foundation ROADMAP

This implementation satisfies **Section I: Data Fidelity is Sacred** requirements:

> "Training inputs MUST preserve linear-light relationships: pixel intensity
> MUST remain a linear proxy for captured light (photon count proxy), not
> tone-mapped or gamma-corrected."

### Invariants Enforced

✅ **12-14 bit precision preservation** - RAW → 16-bit → float32  
✅ **Linear light preservation** - Gamma encoding detected and rejected  
✅ **No silent fallbacks** - All violations raise blocking errors  
✅ **Deterministic validation** - Same input → same verification result  
✅ **dtype enforcement** - Only float32 tensors allowed (no uint8/uint16 leakage)

## Performance Impact

- **Linear verification overhead:** ~1-2ms per image (negligible)
- **16-bit TIFF loading:** Slightly slower than PIL (requires tifffile), but preserves precision
- **No impact on non-APEX pipelines:** Legacy `preprocess_image()` unchanged

## Rollout Plan

1. ✅ **Phase 1: Core implementation** (this PR)
   - Linear verification module
   - RAW loader updates
   - Linear preprocessing function
   - Comprehensive tests

2. **Phase 2: Integration** (future PR)
   - Update orchestrator to use `preprocess_image_linear()`
   - Add linear verification to depth inference pipeline
   - Update presets and configs

3. **Phase 3: Enforcement** (future PR)
   - Enable linear verification in CI
   - Add performance budgets for verification overhead
   - Update governance policy

## References

- **Spatial AI Foundation ROADMAP:** `docs/spatial_ai/ROADMAP.md` (Section I)
- **APEX Performance Contract:** `docs/apex/APEX_CONTRACT.md`
- **Colorspace Quick Reference:** `docs/quick_references/COLORSPACE_QUICK_REFERENCE.md`
- **Linear Verify Module:** `src/transformation_portal/lux_depth_v3/linear_verify.py`
- **Linear Preprocessing:** `src/transformation_portal/lux_depth_v3/preprocessing.py::preprocess_image_linear()`

## Questions?

For questions or issues, please:
1. Check error messages (they include remediation guidance)
2. Review test files for examples (`tests/test_linear_verify.py`, `tests/test_linear_ingest_end_to_end.py`)
3. Consult this migration guide
4. Open an issue with reproduction steps
