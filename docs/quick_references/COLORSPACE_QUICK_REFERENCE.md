# Colorspace Quick Reference

## Status: ✓ Linear Output Confirmed + APEX Linear Ingest Verification

The pro_pipeline now correctly outputs **16-bit linear TIFF** files.

**NEW:** The APEX depth pipeline now enforces linear ingest verification end-to-end per Spatial AI Foundation requirements.

## Quick Commands

```bash
# Linear 16-bit TIFF (default - for compositing)
python3 pro_pipeline.py process image.tiff --preset pool-luxury

# Gamma-encoded 8-bit JPEG (for web)
python3 pro_pipeline.py process image.tiff --preset pool-luxury --gamma --format jpg

# 32-bit float linear (for HDR)
python3 pro_pipeline.py process image.tiff --preset pool-luxury --linear --bits 32
```

## Linear Ingest Verification (APEX Pipeline)

The APEX depth pipeline (`lux_depth_v3`) now enforces strict linear light preservation:

### What is Verified

1. **dtype validation** - Rejects uint8/uint16 tensors (only float32 allowed)
2. **Range validation** - Enforces [0, 1] bounds for normalized linear light
3. **Gamma detection** - Rejects gamma-encoded inputs (sRGB, Rec.709, etc.)
4. **End-to-end linearity** - Validates from RAW/TIFF → final tensor

### How to Use

```python
from transformation_portal.lux_depth_v3.preprocessing import preprocess_image_linear
from transformation_portal.lux_depth_v3.linear_verify import verify_linear_ingest

# Preprocess with linear verification (APEX compliant)
image, orig_shape = preprocess_image_linear("photo.CR2")  # RAW file
image, orig_shape = preprocess_image_linear("render.tif")  # 16-bit TIFF

# Manual verification (if needed)
verify_linear_ingest(tensor)  # Raises error if not linear
```

### RAW File Handling

RAW files are now processed as **linear 16-bit RGB** by default:

```python
from transformation_portal.lux_depth_v3.raw_loader import load_raw_as_rgb

# Linear output (APEX compliant, default)
rgb = load_raw_as_rgb("photo.CR2", output_linear=True, output_bps=16)
# → uint16 [0, 65535] linear RGB

# Gamma output (BLOCKED for APEX)
rgb = load_raw_as_rgb("photo.CR2", output_linear=False)
# → ValueError: Gamma-encoded output not allowed for APEX pipeline
```

### TIFF File Handling

16-bit TIFF files preserve precision via `tifffile`:

```python
# 16-bit TIFF → float32 [0,1] preserving precision
image, shape = preprocess_image_linear("render.tif")
# Uses tifffile to load uint16 → float32/65535

# 8-bit formats (PNG, JPEG) also supported
image, shape = preprocess_image_linear("photo.jpg")
```

## Verification

```python
import tifffile
import numpy as np

# Check colorspace
arr = tifffile.imread("output.tiff")
print(f"Type: {arr.dtype}")  # Should be uint16
print(f"Range: {arr.min()}-{arr.max()}")  # Should be 0-65535
```

Expected for linear:
- Data type: `uint16` (16-bit) or `float32` (32-bit)
- Range: 0-65535 (16-bit) or 0.0-1.0 (32-bit float)
- Mean: Lower than gamma-encoded (linear is darker)
- **APEX Pipeline:** All tensors are float32 [0, 1] with linear verification

### Linear Verification Errors

If you encounter linear verification errors in the APEX pipeline:

```python
# Error: DtypeViolationError
# → Tensor is uint8 or uint16, not float32
# Fix: Use preprocess_image_linear() instead of preprocess_image()

# Error: RangeViolationError
# → Values outside [0, 1]
# Fix: Check input normalization (should be /255 for uint8, /65535 for uint16)

# Error: LinearityViolationError
# → Input appears gamma-encoded (sRGB, Rec.709)
# Fix: Use linear RAW files or pre-linearized TIFF
# DO NOT apply inverse gamma - reject the input instead
```

## Why Linear?

- ✓ Mathematical correctness for blending/compositing
- ✓ Matches 3D render output
- ✓ Professional VFX/post-production standard
- ✓ No banding in gradients
- ✓ Preserves full dynamic range

## When to Use Each

| Use Case | Colorspace | Format | Command |
|----------|-----------|--------|---------|
| Compositing | Linear | 16-bit TIFF | `--linear --bits 16` |
| Color grading | Linear | 16-bit TIFF | `--linear --bits 16` |
| HDR workflow | Linear | 32-bit TIFF | `--linear --bits 32` |
| Web/email | Gamma (sRGB) | JPEG | `--gamma --format jpg` |
| Print | Gamma (sRGB) | 16-bit TIFF | `--gamma --bits 16` |
| Archive | Linear | 16-bit TIFF | `--linear --bits 16` |

## Files

- **Test output:** `processed_images/pool_pro_linear/750Picacho_Pool_compatible_pool-luxury.tiff`
- **Documentation:** `LINEAR_COLORSPACE_IMPLEMENTATION.md`
- **Code:** `pro_pipeline.py` (modified with linear support)
