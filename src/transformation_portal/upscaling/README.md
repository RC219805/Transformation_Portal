# Upscaler Backend Registry

ML-powered super-resolution upscaling with graceful fallback.

## Overview

The upscaler backend registry provides a **plugin-based architecture** for image upscaling with:
- **Golden Path**: Bicubic (fast, always available, no ML dependencies)
- **ML Tier**: Real-ESRGAN (superior quality, optional, commercial-safe)
- **Graceful Fallback**: Automatic degradation if ML dependencies unavailable

## Quick Start

### Bicubic (Always Available)

```python
from transformation_portal.upscaling import UpscalerRegistry
import numpy as np

# Get registry
registry = UpscalerRegistry()

# Get bicubic backend
upscaler = registry.get("bicubic")

# Upscale image
image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
upscaled = upscaler.upscale(image, scale_factor=2.0)
print(upscaled.shape)  # (2000, 2000, 3)
```

### Real-ESRGAN (Requires ML Dependencies)

**⚠️ CURRENTLY UNAVAILABLE**: Real-ESRGAN backend is disabled due to CVE-2024-27763.

```bash
# Real-ESRGAN is currently disabled due to security vulnerability
# The code below is for reference only - it will raise ImportError

# Install ML dependencies (BLOCKED)
# pip install basicsr  # CVE-2024-27763

# Get Real-ESRGAN backend (will fail)
# registry = UpscalerRegistry()
# upscaler = registry.get(
#     "realesrgan",
#     device="cuda",
#     model="RealESRGAN_x2plus",
#     half_precision=False,
# )
```

### With Graceful Fallback

```python
# Request Real-ESRGAN with automatic fallback to bicubic
upscaler = registry.get("realesrgan", fallback_to_bicubic=True)
# If ML deps missing, automatically uses bicubic (no error)
upscaled = upscaler.upscale(image, scale_factor=2.0)
```

## Backends

### Bicubic (`bicubic`)

**Golden Path** - Always available, no dependencies.

- **Algorithm**: OpenCV's bicubic interpolation (cv2.INTER_CUBIC)
- **Dependencies**: None (OpenCV in base requirements)
- **Performance**: ~100-200 images/hour for 4K→8K
- **Memory**: ~50MB per image
- **Quality**: Good for 2x, acceptable for 4x
- **License**: BSD-3-Clause (or Apache 2.0 depending on OpenCV version) (commercial-safe)

### Real-ESRGAN (`realesrgan`)

**ML Tier** - Superior quality, requires ML dependencies.

**⚠️ CURRENTLY UNAVAILABLE**: Real-ESRGAN backend is temporarily disabled due to CVE-2024-27763 in the BasicSR dependency. A vendored safe implementation will be added in a future update. Use `bicubic` backend as the current production path.

- **Algorithm**: Real-ESRGAN (RRDB network with perceptual loss)
- **Dependencies**: torch, basicsr
- **Performance**: ~10-30 images/hour for 4K→8K (GPU), ~2-5/hour (CPU)
- **Memory**: ~2-4GB GPU memory
- **Quality**: Excellent detail preservation, especially textures
- **License**: BSD-3-Clause (commercial-safe)
- **Models**:
  - `RealESRGAN_x2plus`: Best for 2x upscaling (~17MB)
  - `RealESRGAN_x4plus`: Best for 4x upscaling (~64MB)

## Architecture

### Protocol-Based Design

All backends implement the `UpscalerBackend` protocol:

```python
class UpscalerBackend(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def requires_ml(self) -> bool: ...

    def upscale(
        self,
        image: np.ndarray,  # (H, W, 3)
        scale_factor: float,  # 1.0-4.0
    ) -> np.ndarray: ...
```

### Registry Pattern

The `UpscalerRegistry` provides:
- Backend discovery: `list_backends()`, `has_backend()`, `available_backend_ids()`
- Backend instantiation: `get(backend_name, **kwargs)`
- Graceful fallback: `get(..., fallback_to_bicubic=True)`

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

## CLI Usage

```bash
# Golden Path (bicubic, default)
lux-depth-v3 \
  --input-dir ./input \
  --output-dir ./output \
  --enable-v2 on

# ML Tier (Real-ESRGAN)
lux-depth-v3 \
  --input-dir ./input \
  --output-dir ./output \
  --enable-v2 on \
  --v2-upscaler realesrgan \
  --v2-device cuda
```

## API Reference

### UpscalerRegistry

```python
class UpscalerRegistry:
    def get(
        self,
        backend_name: str,
        device: str = "cpu",
        fallback_to_bicubic: bool = True,
        **kwargs,
    ) -> UpscalerBackend:
        """Get upscaler backend with optional fallback."""
        ...

    def list_backends(self) -> Dict[str, Dict[str, Any]]:
        """List all registered backends with metadata."""
        ...

    def available_backend_ids(self) -> list[str]:
        """Get list of all registered backend IDs."""
        ...

    def has_backend(self, backend_id: str) -> bool:
        """Check if backend is registered."""
        ...
```

### BicubicUpscaler

```python
class BicubicUpscaler:
    def upscale(
        self,
        image: np.ndarray,
        scale_factor: float,
    ) -> np.ndarray:
        """Upscale using bicubic interpolation."""
        ...
```

### RealESRGANUpscaler

```python
class RealESRGANUpscaler:
    def __init__(
        self,
        device: str = "cpu",
        model: str = "RealESRGAN_x2plus",
        half_precision: bool = False,
    ):
        """Initialize Real-ESRGAN upscaler."""
        ...

    def upscale(
        self,
        image: np.ndarray,
        scale_factor: float,
    ) -> np.ndarray:
        """Upscale using Real-ESRGAN."""
        ...
```

## Performance

### Bicubic

| Resolution | Scale | Time | Throughput | Memory |
|------------|-------|------|------------|--------|
| 1920x1080 → 3840x2160 | 2.0x | ~5ms | ~200/hour | ~50MB |
| 3840x2160 → 7680x4320 | 2.0x | ~15ms | ~240/hour | ~50MB |
| 1920x1080 → 7680x4320 | 4.0x | ~20ms | ~180/hour | ~50MB |

### Real-ESRGAN (CURRENTLY UNAVAILABLE)

**Note**: Real-ESRGAN backend is currently disabled due to security vulnerability CVE-2024-27763 in BasicSR dependency. Benchmarks preserved for future reference when safe implementation is available.

| Resolution | Scale | Time (GPU) | Time (CPU) | Throughput (GPU) | Memory (GPU) |
|------------|-------|------------|------------|------------------|--------------|
| 1920x1080 → 3840x2160 | 2.0x | ~2-3s | ~15-20s | ~1200/hour | ~2-4GB |
| 3840x2160 → 7680x4320 | 2.0x | ~8-12s | ~60-90s | ~300/hour | ~2-4GB |

## License

All components are commercial-safe:

| Component | License | Commercial Use | Status |
|-----------|---------|----------------|--------|
| Bicubic (OpenCV) | Apache 2.0 | ✅ Yes | Active |
| Real-ESRGAN Model | BSD-3-Clause | ✅ Yes | Suspended (CVE-2024-27763) |
| BasicSR | Apache 2.0 | ⚠️ Blocked | CVE-2024-27763 |

**Security Note**: BasicSR dependency is blocked due to CVE-2024-27763 (command injection vulnerability). Real-ESRGAN backend will be re-enabled when a safe vendored implementation is available.

## Examples

See `examples/upscaling_comparison.py` for a complete comparison script.

```bash
# Compare bicubic vs Real-ESRGAN
python examples/upscaling_comparison.py --backend both --device cuda

# Test bicubic only
python examples/upscaling_comparison.py --backend bicubic

# Test Real-ESRGAN only
python examples/upscaling_comparison.py --backend realesrgan --device cuda
```

## Testing

```bash
# Run all tests (bicubic only, no ML deps required)
pytest tests/test_upscaling.py -v

# Run with Real-ESRGAN (CURRENTLY UNAVAILABLE - tests will skip)
# Real-ESRGAN tests are disabled due to CVE-2024-27763 in BasicSR
pytest tests/test_upscaling.py -v -m ml
```

**Note**: Real-ESRGAN integration tests have been removed. They will be re-added when a safe vendored implementation is available.

## References

- **Paper**: [Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data](https://arxiv.org/abs/2107.10833)
- **Code**: https://github.com/xinntao/Real-ESRGAN
- **License**: https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE
