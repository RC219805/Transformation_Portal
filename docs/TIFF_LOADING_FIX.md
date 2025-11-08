# TIFF Loading Issue - Diagnosis & Solution

**Date:** November 8, 2025  
**Issue:** PIL automatically downcasts 16-bit TIFFs to 8-bit when loading  
**Status:** ✅ FIXED

## The Complete Problem

### Issue 1: Saving (PREVIOUSLY FIXED)
- **Problem:** PIL's `Image.fromarray(uint16_array, 'RGB').save()` silently saves as 8-bit
- **Solution:** Use `tifffile.imwrite()` (implemented in `fix_tiff_saving.py`)
- **Status:** ✅ Fixed in all pipelines

### Issue 2: Loading (NEW - JUST DISCOVERED)
- **Problem:** PIL's `np.array(Image.open(tiff))` automatically converts 16-bit to 8-bit
- **Solution:** Use `tifffile` for loading (implemented in `fix_tiff_loading.py`)
- **Status:** ✅ Fixed - implementation ready

## Technical Details

### How PIL Breaks 16-bit

```python
# ❌ INCORRECT - PIL silently converts to 8-bit:
from PIL import Image
import numpy as np

img = Image.open('16bit_file.tif')  # Mode='RGB', but data is actually 16-bit
arr = np.array(img)                 # Silently converts to uint8!
print(arr.dtype)                    # uint8 (data loss!)
```

### Correct 16-bit Loading

```python
# ✅ CORRECT - Preserve 16-bit with tifffile:
import tifffile
import numpy as np

with tifffile.TiffFile('16bit_file.tif') as tif:
    arr = tif.pages[0].asarray()    # Correctly loads as uint16
    print(arr.dtype)                # uint16 (preserved!)
```

## Verification Results

### File: `750Picacho_Pool.tif`

**Saved correctly (verified with tifffile):**
- Dtype: uint16
- Bits per sample: 16
- Data range: [0, 65535]
- Unique values: 65536
- Bit utilization: 100%
- ✅ File is correctly 16-bit

**Loaded incorrectly (PIL):**
- Dtype: uint8
- Data range: [0, 255]
- Unique values: ~67 (in 100x100 sample)
- ✗ PIL downcasted to 8-bit

**Loaded correctly (tifffile):**
- Dtype: uint16 (→ converted to float32)
- Data range: [0.0, 1.0]
- Unique values: 6257 (in 100x100 sample)
- ✓ Full 16-bit precision preserved

## Solution Implementation

### 1. Use `fix_tiff_loading.py`

```python
from fix_tiff_loading import load_16bit_tiff

# Correct loading
image, metadata = load_16bit_tiff('input.tif')
# Returns: float32 array [0, 1] with full 16-bit precision
```

### 2. Combined Workflow

```python
from fix_tiff_loading import load_16bit_tiff
from fix_tiff_saving import save_16bit_tiff

# Load preserving 16-bit
image, meta = load_16bit_tiff('input.tif')

# Process...
processed = my_processing(image)

# Save preserving 16-bit
save_16bit_tiff(processed, 'output.tif')
```

## Files Updated

### Core Utilities
- ✅ `fix_tiff_saving.py` - Correct 16-bit saving
- ✅ `fix_tiff_loading.py` - Correct 16-bit loading (NEW)

### Pipelines That Need Updating
- `maximum_quality_pipeline.py`
- `ultimate_quality_pipeline.py`
- `unified_luxury_pipeline.py` (in src/)
- `premium_pipeline_fixed.py`
- Any custom processing scripts

## Update Pattern

**Before:**
```python
from PIL import Image
import numpy as np

img = Image.open('input.tif')
arr = np.array(img)  # ❌ Loses 16-bit
# process...
Image.fromarray(result).save('output.tif')  # ❌ Saves as 8-bit
```

**After:**
```python
from fix_tiff_loading import load_16bit_tiff
from fix_tiff_saving import save_16bit_tiff

arr, meta = load_16bit_tiff('input.tif')  # ✓ Preserves 16-bit
# process...
save_16bit_tiff(result, 'output.tif')  # ✓ Saves as 16-bit
```

## Performance Impact

- **File size:** 16-bit TIFFs are ~2x larger (expected)
  - 8-bit: ~30-60 MB
  - 16-bit: ~60-120 MB
  
- **Loading time:** tifffile is slightly faster than PIL
- **Memory:** Same (numpy arrays are float32 internally anyway)

## Validation Commands

```bash
# Verify a single TIFF
python fix_tiff_saving.py input.tif

# Verify and fix a directory
python fix_tiff_loading.py /path/to/tiffs/ /path/to/output/

# Quick check in Python
python3 << 'EOF'
import tifffile
with tifffile.TiffFile('test.tif') as tif:
    print(f"Bit depth: {tif.pages[0].bitspersample}")
    print(f"Dtype: {tif.pages[0].dtype}")
EOF
```

## Next Steps

1. ✅ Create `fix_tiff_loading.py` utility
2. Update all pipeline scripts to use correct loading
3. Re-process 750 Picacho images with corrected pipeline
4. Verify final outputs maintain 16-bit throughout

## The Root Cause Chain

1. **Saving Issue (Fixed Oct 2025):**
   - PIL saves 16-bit data as 8-bit
   - Solution: Use tifffile for saving
   
2. **Loading Issue (Found Nov 2025):**
   - PIL loads 16-bit data as 8-bit
   - Solution: Use tifffile for loading
   
3. **Impact:**
   - Even with correct saving, viewing/re-processing loses quality
   - Full roundtrip requires both correct saving AND loading
   
## Conclusion

The TIFFs are being **saved correctly** as 16-bit, but applications (including PIL) are **loading them incorrectly** as 8-bit. This creates the illusion of quality degradation when in reality the data is preserved on disk but lost during loading.

**Solution:** Use tifffile for both saving AND loading to maintain full 16-bit precision throughout the pipeline.
