# Colorspace Quick Reference

## Status: ✓ Linear Output Confirmed

The pro_pipeline now correctly outputs **16-bit linear TIFF** files.

## Quick Commands

```bash
# Linear 16-bit TIFF (default - for compositing)
python3 pro_pipeline.py process image.tiff --preset pool-luxury

# Gamma-encoded 8-bit JPEG (for web)
python3 pro_pipeline.py process image.tiff --preset pool-luxury --gamma --format jpg

# 32-bit float linear (for HDR)
python3 pro_pipeline.py process image.tiff --preset pool-luxury --linear --bits 32
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

