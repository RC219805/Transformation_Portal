# EXIF Normalization Fix - Implementation Summary

## Problem Statement

The V2 pipeline was experiencing runtime failures after running for ~16 seconds, with the manifest showing `exif_normalized: false`. This was causing EXIF orientation mismatches between the depth map and input image, leading to shape validation failures in V2.

## Root Cause Analysis

1. **Semantic Mismatch**: The `normalize_exif_orientation()` function always created a normalized file but returned `True` only if an EXIF tag existed. This caused the manifest to incorrectly show `exif_normalized: false` when the file was actually normalized.

2. **Missing Validation**: There was no preflight check before V2 to catch depth/image shape mismatches, leading to cryptic 16-second crashes instead of immediate, clear errors.

3. **Incomplete Manifest**: The `normalized_path` field was only set when EXIF tags existed, not always, making it unclear which file V2 should use.

## Solution Implementation

### 1. Fixed Normalization Return Value
**File**: `lux_depth_v3/enhance/preprocessing.py`

Changed `normalize_exif_orientation()` to always return `True` because it always creates a normalized file:

```python
# Before:
return has_exif_orientation  # True only if EXIF tag existed

# After:
return True  # Always true - file is always normalized
```

This ensures the manifest correctly reflects reality: the file is **always** normalized via `ImageOps.exif_transpose()`, regardless of whether an EXIF tag was present.

### 2. Added Preflight Validation
**File**: `lux_depth_v3/enhance/preprocessing.py`

New function `validate_depth_image_alignment()` catches errors before V2 runs:

```python
def validate_depth_image_alignment(image_path: Path, depth_path: Path) -> None:
    """Validate that depth and image have matching dimensions.

    Checks:
    - Image and depth shapes match (H, W)
    - Depth is uint16 (not uint8 or float)
    - Depth is single-channel (not RGB)

    Raises clear error messages for EXIF orientation mismatches.
    """
```

This validation runs **before** V2 is invoked, providing:
- **Immediate failure** (milliseconds vs. 16 seconds)
- **Clear error messages** pointing to the root cause
- **Actionable guidance** for fixing the issue

### 3. Updated Orchestrator
**File**: `lux_depth_v3/enhance/orchestrator.py`

Three key changes:

```python
# 1. Always set exif_normalized=True in manifest
input=InputMetadata(
    image_path=str(image_input.path),
    image_sha256=input_sha256,
    exif_normalized=True,  # Always true - file is always normalized
    normalized_path=str(normalized_path),  # Always set
)

# 2. Call preflight validation before V2
if depth_path and depth_path.exists():
    try:
        validate_depth_image_alignment(normalized_path, depth_path)
    except ValueError as e:
        logger.error(f"Preflight validation failed: {e}")
        raise
```

### 4. Comprehensive Testing
**File**: `lux_depth_v3/tests/test_exif_normalization.py`

Added 5 new preflight validation tests:
- ✅ `test_valid_alignment`: Matching dimensions pass
- ✅ `test_shape_mismatch_detected`: EXIF mismatch detected
- ✅ `test_depth_wrong_dtype`: uint8 depth rejected
- ✅ `test_depth_multichannel`: RGB depth rejected
- ✅ `test_missing_depth_file`: Missing depth handled

All 19 tests passing:
```
19 passed, 2 warnings in 0.21s
```

### 5. Updated Documentation
**File**: `lux_depth_v3/enhance/HASH_MODE_GUIDE.md`

Updated manifest examples to show the new structure:

```json
{
  "input": {
    "image_path": "/path/to/image.jpg",
    "image_sha256": "abc123...",
    "exif_normalized": true,  // ✓ Always true now
    "normalized_path": "/path/to/tmp_inputs/image_normalized.png"  // ✓ Always set
  }
}
```

## Impact & Benefits

### Before Fix
- ❌ Manifest shows `exif_normalized: false` even when normalized
- ❌ V2 crashes after ~16 seconds with cryptic error
- ❌ No way to detect EXIF orientation issues early
- ❌ Users need to debug complex pipeline failures

### After Fix
- ✅ Manifest correctly shows `exif_normalized: true`
- ✅ Preflight validation fails immediately (milliseconds)
- ✅ Clear error messages point to root cause
- ✅ Users get actionable guidance

### Example Error Message

**Before (cryptic, after 16 seconds):**
```
[Generic V2 error after 16 seconds]
```

**After (clear, immediate):**
```
Preflight validation failed: Image/depth shape mismatch (likely EXIF orientation issue):
  Image (normalized): (3600, 6000) (H, W)
  Depth:              (6000, 3600) (H, W)

This usually means:
  1. EXIF normalization was not applied to the input image, OR
  2. Depth was generated from a different version of the input, OR
  3. The normalized file path was not used consistently.

Expected behavior: Both should match because depth is generated
from the same normalized file that V2 will process.
```

## Files Changed

1. **lux_depth_v3/enhance/preprocessing.py**
   - Fixed `normalize_exif_orientation()` return value
   - Added `validate_depth_image_alignment()` function

2. **lux_depth_v3/enhance/orchestrator.py**
   - Always set `exif_normalized=True` in manifest
   - Always set `normalized_path` in manifest
   - Added preflight validation call before V2

3. **lux_depth_v3/tests/test_exif_normalization.py**
   - Updated existing tests
   - Added 5 new preflight validation tests

4. **lux_depth_v3/enhance/HASH_MODE_GUIDE.md**
   - Updated manifest examples

5. **lux_depth_v3/examples/test_exif_fix.py**
   - Created demonstration script

## Validation

### Linting
```bash
$ flake8 lux_depth_v3/enhance/preprocessing.py lux_depth_v3/enhance/orchestrator.py
# No errors
```

### Tests
```bash
$ pytest lux_depth_v3/tests/test_exif_normalization.py -v
# 19 passed, 2 warnings in 0.21s
```

### Demonstration
```bash
$ python3 lux_depth_v3/examples/test_exif_fix.py
# ALL TESTS PASSED ✓
```

## Future Recommendations

1. **Monitoring**: Track preflight validation failures to identify common EXIF issues
2. **Metrics**: Add timing metrics to show preflight validation saves ~16 seconds per failure
3. **Documentation**: Update user guides with EXIF normalization best practices
4. **Upstream**: Consider submitting EXIF handling improvements to V2 pipeline

## References

- **Problem Statement**: Issue description in PR
- **Decision Guide**: `docs/DECISION_GUIDE.md`
- **Manifest Schema**: `lux_depth_v3/enhance/manifest.py`
- **Testing Guide**: `lux_depth_v3/enhance/TESTING_STRATEGY.md`
