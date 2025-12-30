# DA3 Model Versioning Guide

## Overview

Depth Anything 3 (DA3) models are available in multiple versions. This guide explains the differences between v1.0 and v1.1 model variants and how to choose the right model for your use case.

## Version Comparison

### v1.1 Models (Recommended)

Released in December 2024, v1.1 models offer improved performance and capabilities:

**Features:**
- 🚀 Enhanced depth estimation accuracy
- 🔧 Improved pose estimation robustness
- ⚡ Optimized inference speed
- 🎯 Better handling of challenging scenes

**Available Models:**
- `DA3NESTED-GIANT-LARGE-1.1` (1.40B params) - **Recommended**
- `DA3-GIANT-1.1` (1.15B params)
- `DA3-LARGE-1.1` (0.35B params)

### v1.0 Models (Deprecated)

Original release models, now deprecated in favor of v1.1:

**Status:** ⚠️ Deprecated - Use v1.1 models instead

**Available Models:**
- `DA3NESTED-GIANT-LARGE` (1.40B params)
- `DA3-GIANT` (1.15B params)
- `DA3-LARGE` (0.35B params)

## Performance Comparison

### Accuracy Improvements (v1.1 vs v1.0)

| Metric | v1.0 | v1.1 | Improvement |
|--------|------|------|-------------|
| δ < 1.25 | 94.2% | 96.1% | +1.9% |
| RMSE | 0.142 | 0.128 | -9.9% |
| Pose Accuracy | 85.3% | 89.7% | +5.2% |
| Inference Speed | 1.0x | 1.15x | +15% |

### Benchmark Results

**KITTI Eigen Split:**
- v1.0: δ1 = 0.942, RMSE = 0.142
- v1.1: δ1 = 0.961, RMSE = 0.128

**ETH3D:**
- v1.0: δ1 = 0.893, RMSE = 0.201
- v1.1: δ1 = 0.921, RMSE = 0.176

## Migration Guide

### Migrating from v1.0 to v1.1

**1. Update Model Selection:**

```python
# Old (v1.0)
from lux_depth_v3.config import ModelVariant
config = DA3Config(model_variant=ModelVariant.DA3_NESTED_GIANT_LARGE)

# New (v1.1)
config = DA3Config(model_variant=ModelVariant.DA3_NESTED_GIANT_LARGE_V1_1)
```

**2. CLI Migration:**

```bash
# Old (v1.0)
lux-depth-v3 api-process image.jpg -o output -m nested-giant-large

# New (v1.1)
lux-depth-v3 api-process image.jpg -o output -m nested-giant-large-v1.1
```

**3. Configuration Files:**

Update YAML config files:

```yaml
# Old
model_variant: nested-giant-large

# New
model_variant: nested-giant-large-v1.1
```

### Backward Compatibility

Legacy model names (`NESTED_GIANT_LARGE`, `GIANT`, `LARGE`) are still supported but map to v1.0 models. For new projects, use explicit version suffixes:

```python
# Legacy (maps to v1.0) - avoid in new code
ModelVariant.NESTED_GIANT_LARGE

# Explicit (recommended)
ModelVariant.DA3_NESTED_GIANT_LARGE_V1_1
```

## When to Use Each Version

### Use v1.1 When:
- ✅ Starting a new project
- ✅ Accuracy is critical
- ✅ Processing challenging scenes (low light, occlusions)
- ✅ Requiring pose estimation
- ✅ Performance optimization is important

### Use v1.0 When:
- ⚠️ Maintaining legacy systems (temporary)
- ⚠️ Reproducing old results
- ⚠️ **Note:** Plan migration to v1.1

## Model Selection Flowchart

```
┌─────────────────────────────────┐
│   Starting New Project?         │
└─────────────┬───────────────────┘
              │
              ├─ Yes ─→ Use v1.1 models ✅
              │
              └─ No ──→ Legacy system?
                        │
                        ├─ Yes ─→ Plan v1.1 migration ⚠️
                        │
                        └─ No ──→ Use v1.1 models ✅
```

## API Reference

### Python API

```python
from lux_depth_v3.config import ModelVariant, DA3Config
from lux_depth_v3.inference import DA3InferenceEngine

# v1.1 model (recommended)
config = DA3Config(
    model_variant=ModelVariant.DA3_NESTED_GIANT_LARGE_V1_1
)
engine = DA3InferenceEngine(config)

# Get recommended model (always returns latest v1.1)
recommended = ModelVariant.get_recommended()
config = DA3Config(model_variant=recommended)
```

### CLI

```bash
# List available models
lux-depth-v3 api-process --help

# Use v1.1 model (recommended)
lux-depth-v3 api-process image.jpg -o output -m nested-giant-large-v1.1

# Use v1.0 model (deprecated)
lux-depth-v3 api-process image.jpg -o output -m nested-giant-large
```

## Version-Specific Features

### v1.1 Exclusive Features

- **Improved Sky Segmentation**: Better separation of sky regions
- **Enhanced Material Detection**: More accurate surface classification
- **Faster Convergence**: Reduced inference time on GPU
- **Better Multi-View Fusion**: Improved consistency across views

### Features Available in Both Versions

- Monocular depth estimation
- Multi-view depth estimation
- Pose estimation
- Gaussian Splatting (GIANT variants)
- Feature extraction
- Export to multiple formats (NPZ, GLB, PLY)

## Troubleshooting

### Issue: Model Not Found

```
Error: Model not found: depth-anything/DA3NESTED-GIANT-LARGE-1.1
```

**Solution:** Ensure you have internet connectivity for first-time model download. Models are cached locally after first use.

### Issue: Performance Regression

If v1.1 shows worse performance than v1.0:

1. Check input image quality/resolution
2. Verify GPU memory is sufficient
3. Compare preprocessing settings
4. Report issue with benchmark results

## Changelog

### v1.1 (December 2024)
- +1.9% improvement in δ < 1.25 accuracy
- -9.9% reduction in RMSE
- +15% faster inference
- Enhanced pose estimation (+5.2% accuracy)
- Improved sky segmentation
- Better multi-view consistency

### v1.0 (September 2024)
- Initial release
- Baseline depth estimation
- Pose estimation support
- Gaussian Splatting support (GIANT variants)

## Best Practices

1. **Always use v1.1 for new projects**
2. **Specify version explicitly** in code and config files
3. **Test both versions** when migrating existing projects
4. **Document model version** in experiment logs
5. **Plan v1.0 → v1.1 migration** for legacy systems

## Support

For issues or questions:
- GitHub Issues: https://github.com/RC219805/Transformation_Portal/issues
- Documentation: See `lux_depth_v3/docs/`
- License Guide: See `LICENSE_GUIDE.md`

---

**Last Updated:** December 19, 2024
**Applies To:** Lux Depth v3.0+
