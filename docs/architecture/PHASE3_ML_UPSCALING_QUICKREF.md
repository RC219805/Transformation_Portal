# ML Upscaling Quick Reference

## Overview

The upscaler backend registry provides **ML-powered super-resolution** as an optional alternative to bicubic upscaling.

**Golden Path**: Bicubic (fast, no ML dependencies)
**ML Tier**: Real-ESRGAN (superior quality, requires ML dependencies)

## CLI Usage

### Bicubic (Default - Always Available)

```bash
lux-depth-v3 \
  --input-dir ./input \
  --output-dir ./output \
  --enable-v2 on
  # --v2-upscaler defaults to "bicubic"
```

### Real-ESRGAN (ML-Powered)

```bash
# Install ML dependencies first
pip install basicsr

# Use Real-ESRGAN
lux-depth-v3 \
  --input-dir ./input \
  --output-dir ./output \
  --enable-v2 on \
  --v2-upscaler realesrgan \
  --v2-device cuda  # or mps for Apple Silicon
```

## Python API

### Basic Usage

```python
from transformation_portal.upscaling import UpscalerRegistry
import numpy as np

# Get registry
registry = UpscalerRegistry()

# Bicubic backend (always available)
upscaler = registry.get("bicubic")
image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
upscaled = upscaler.upscale(image, scale_factor=2.0)
print(upscaled.shape)  # (2000, 2000, 3)

# Real-ESRGAN backend (with graceful fallback)
upscaler = registry.get(
    "realesrgan",
    device="cuda",
    model="RealESRGAN_x2plus",
    fallback_to_bicubic=True
)
upscaled = upscaler.upscale(image, scale_factor=2.0)
```

### List Available Backends

```python
registry = UpscalerRegistry()
backends = registry.list_backends()
print(backends)
# {'bicubic': {'requires_ml': False}, 'realesrgan': {'requires_ml': True}}
```

### Check Backend Availability

```python
if registry.has_backend("realesrgan"):
    print("Real-ESRGAN available")
else:
    print("Real-ESRGAN not available (missing ML deps)")
```

## Backend Comparison

| Feature | Bicubic | Real-ESRGAN |
|---------|---------|-------------|
| **ML Dependencies** | ❌ None | ✅ torch, basicsr |
| **Quality** | Good | Excellent |
| **Speed (4K→8K)** | ~15ms | ~8-12s (GPU) |
| **Throughput** | ~200/hour | ~300/hour (GPU) |
| **Memory** | ~50MB | ~2-4GB GPU |
| **License** | HPND | BSD-3-Clause |
| **Commercial Safe** | ✅ Yes | ✅ Yes |

## Real-ESRGAN Models

### RealESRGAN_x2plus (Default)
- **Best for**: 2x upscaling (1920→3840)
- **Weight size**: ~17MB
- **Quality**: Excellent detail preservation
- **Recommended scale**: 2.0

### RealESRGAN_x4plus
- **Best for**: 4x upscaling (1920→7680)
- **Weight size**: ~64MB
- **Quality**: Excellent for large upscales
- **Recommended scale**: 4.0

## Installation

### Core (No ML)
```bash
# Already included in base requirements
pip install -r requirements/base.txt
```

### ML Dependencies
```bash
# Option 1: Install ML dependencies
pip install basicsr

# Option 2: Install from ml requirements
pip install -r requirements/ml.txt
```

## Model Weights

Model weights are **auto-downloaded** on first use:

- **Source**: https://github.com/xinntao/Real-ESRGAN/releases
- **Location**: `weights/RealESRGAN_x2plus.pth` or `weights/RealESRGAN_x4plus.pth`
- **License**: BSD-3-Clause (commercial-safe)

## Graceful Fallback

The system automatically falls back to bicubic if:
1. Real-ESRGAN backend requested but ML dependencies missing
2. Model loading fails
3. Inference fails

Example:
```python
# Request Real-ESRGAN with fallback enabled
upscaler = registry.get("realesrgan", fallback_to_bicubic=True)
# If ML deps missing, automatically uses bicubic (no error)
```

## Performance Tips

### For Speed (Bicubic)
```bash
--v2-upscaler bicubic
# ~200 images/hour for 4K→8K
```

### For Quality (Real-ESRGAN)
```bash
--v2-upscaler realesrgan --v2-device cuda
# ~300 images/hour for 4K→8K (GPU)
# ~10-30 images/hour (CPU)
```

### For Apple Silicon
```bash
--v2-upscaler realesrgan --v2-device mps
# 2-3x faster than CPU
```

## Troubleshooting

### "Backend 'realesrgan' requires ML dependencies"
**Solution**: Install basicsr
```bash
pip install basicsr
```

### "Failed to download model weights"
**Solution**: Download manually
```bash
mkdir -p weights
curl -L -o weights/RealESRGAN_x2plus.pth \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth
```

### Real-ESRGAN too slow on CPU
**Solution**: Use GPU or fall back to bicubic
```bash
--v2-upscaler bicubic  # Much faster
```

### Out of memory on GPU
**Solution**: Use bicubic or reduce batch size
```bash
--v2-upscaler bicubic
```

## Stage Graph Integration

```python
from transformation_portal.stage_graph.stages import UpscalingStage
from transformation_portal.stage_graph.stage import StageContext

# Create upscaling stage
stage = UpscalingStage(
    scale_factor=2.0,
    backend="realesrgan",  # or "bicubic"
    version="1.0.0",
)

# Run stage
context = StageContext(
    device="cuda",
    artifacts={"enhanced_image": image},
)
result = stage.compute(context)

# Get upscaled image
upscaled = result.artifacts["upscaled_image"]
```

## Files

### Module Structure
```
src/transformation_portal/upscaling/
├── __init__.py                    # Public API
├── protocol.py                    # UpscalerBackend protocol
├── registry.py                    # UpscalerRegistry
└── backends/
    ├── __init__.py
    ├── bicubic.py                 # Core backend
    └── realesrgan.py              # ML backend
```

### Integration Points
- **CLI**: `src/transformation_portal/lux_depth_v3/__main__.py` (--v2-upscaler flag)
- **Config**: `src/transformation_portal/lux_depth_v3/config.py` (v2_upscaler_backend field)
- **Stage**: `src/transformation_portal/stage_graph/stages/upscaling.py`

## License

| Component | License | Commercial |
|-----------|---------|------------|
| Bicubic (PIL) | HPND | ✅ Yes |
| Real-ESRGAN Model | BSD-3-Clause | ✅ Yes |
| BasicSR | Apache 2.0 | ✅ Yes |

## References

- **Paper**: [Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data](https://arxiv.org/abs/2107.10833)
- **Code**: https://github.com/xinntao/Real-ESRGAN
- **License**: https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE
- **Implementation Report**: `docs/architecture/PHASE3_ML_UPSCALING_IMPLEMENTATION_REPORT.md`
