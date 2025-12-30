# DA3 Integration Quick Reference

## ✅ Status: COMPLETE

Basic monocular depth estimation is **fully operational** with the Depth Anything 3 (DA3) integration.

## Quick Test

```bash
# Test with existing image
python test_da3_quick.py test_output/da3_basic/test_image.png

# Test with your own image
python test_da3_quick.py path/to/your/image.jpg output_dir/
```

## Python API

```python
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Required workaround

from lux_depth_v3.config import ModelVariant, DA3Config
from lux_depth_v3.inference import DA3InferenceEngine
from lux_depth_v3.input_manager import ImageInput
from pathlib import Path

# Configure and run
config = DA3Config(model_variant=ModelVariant.DA3_LARGE_V1_1)
engine = DA3InferenceEngine(config, commercial_use=False)

result = engine.infer([ImageInput(path=Path("image.jpg"))])

# Access results
print(f"Depth: {result.depth.shape}")      # (1, H, W)
print(f"Confidence: {result.conf.shape}")  # (1, H, W)
```

## Available Models

| Model | Params | License | Use Case |
|-------|--------|---------|----------|
| DA3-LARGE-1.1 | 0.35B | CC-BY-NC-4.0 | **Recommended** - Best balance |
| DA3-GIANT-1.1 | 1.15B | CC-BY-NC-4.0 | High quality |
| DA3NESTED-GIANT-LARGE-1.1 | 1.40B | CC-BY-NC-4.0 | Full features (metric + pose + GS) |
| DA3-BASE | 0.12B | CC-BY-NC-4.0 | Lightweight |
| DA3METRIC-LARGE | 0.35B | CC-BY-NC-4.0 | Metric depth specialist |
| DA3MONO-LARGE | 0.35B | **Apache-2.0** | Commercial use |

## Performance

**M4 Max (MPS backend):**
- Model load: ~2s (one-time)
- Inference: ~1.3s per 512x512 image
- Throughput: ~0.7 images/sec

## Files Changed

1. **lux_depth_v3/da3_wrapper.py**
   - Added model name mapping (`VARIANT_TO_API_NAME`)
   - Updated `_prepare_images()` for ImageInput support

2. **lux_depth_v3/inference.py**
   - Fixed model variant translation
   - Removed `-1.1` suffix for API compatibility

## Tests

```bash
# Unit tests
cd lux_depth_v3
pytest tests/test_da3_api.py -v
# Result: ✅ 19/19 passed

# Quick integration test
python test_da3_quick.py test_output/da3_basic/test_image.png
```

## Documentation

- **Complete Guide:** `DA3_INTEGRATION_COMPLETE.md`
- **Summary:** `DA3_INTEGRATION_SUMMARY.md`
- **lux_depth_v3 Docs:** `lux_depth_v3/INTEGRATION_GUIDE.md`

## Next Steps (Optional)

The following features are **already implemented** in the infrastructure but not tested in this integration:

- Multi-view depth estimation
- Camera pose prediction
- Gaussian Splatting export
- GLB/NPZ export formats
- CLI batch processing

These can be enabled/tested as needed in future sessions.

## Troubleshooting

**Issue:** `OMP: Error #15: Initializing libomp.dylib`
**Fix:** Set `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` before imports (already done in test scripts)

**Issue:** `gsplat` warning
**Fix:** Optional - only needed for Gaussian Splatting workflows

**Issue:** Model not found
**Fix:** Models auto-download on first use. DA3-LARGE-1.1 and DA3NESTED-GIANT-LARGE-1.1 are pre-cached.

---

**Date:** 2025-12-19
**Status:** ✅ Production Ready
