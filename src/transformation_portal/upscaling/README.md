# Upscaler Backend Registry

ML-powered super-resolution upscaling with graceful fallback.

## Overview

The upscaler backend registry provides a **plugin-based architecture** for image upscaling with:
- **Golden Path**: Bicubic (fast, always available, no ML dependencies)
- **ML Tier**: Real-ESRGAN is currently disabled by the repository availability guard.
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

**Core backend** - Requires the core NumPy/OpenCV environment; no model download.

- **Algorithm**: OpenCV's bicubic interpolation (cv2.INTER_CUBIC)
- **Dependencies**: NumPy and OpenCV from the governed core environment
- **License**: BSD-3-Clause (or Apache 2.0 depending on OpenCV version) (commercial-safe)

### Real-ESRGAN (`realesrgan`)

**ML Tier** - Superior quality, requires ML dependencies.

**⚠️ CURRENTLY UNAVAILABLE**: Real-ESRGAN backend is temporarily disabled due to CVE-2024-27763 in the BasicSR dependency. Re-enabling it requires separate implementation and dependency-policy review. Use `bicubic` backend as the current production path.

- **Algorithm**: Real-ESRGAN (RRDB network with perceptual loss)
- **Dependencies**: torch, basicsr
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

## CLI boundary

The Lux Depth V3 CLI does not expose `--v2-upscaler` or `--v2-device`.
`--enable-v2 on` enables the separate enhancement stage; it does not prove
that this registry performed upscaling. Use the Python registry API above for
this component and the [Lux CLI guide](../../../docs/cli/LUX_DEPTH_V3_CLI_GUIDE.md)
for supported command-line options.

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

## Performance evidence

No reproducible benchmark run accompanies this guide. Measure the actual
backend, input dimensions, scale, runtime, and output quality before making
throughput or memory claims. Registry fallback returns bicubic when the
requested backend is unavailable and `fallback_to_bicubic=True`; inspect
`upscaler.name` so a fallback is not reported as Real-ESRGAN inference.

## License

All components are commercial-safe:

| Component | License | Commercial Use | Status |
|-----------|---------|----------------|--------|
| Bicubic (OpenCV) | Apache 2.0 | ✅ Yes | Active |
| Real-ESRGAN Model | BSD-3-Clause | ✅ Yes | Suspended (CVE-2024-27763) |
| BasicSR | Apache 2.0 | ⚠️ Blocked | CVE-2024-27763 |

**Security Note**: BasicSR dependency is blocked due to CVE-2024-27763 (command injection vulnerability). Any re-enablement requires a separately reviewed implementation and dependency lane.

## Examples

See `examples/upscaling_comparison.py` for a complete comparison script.

```bash
# Compare bicubic vs Real-ESRGAN
python examples/upscaling_comparison.py --backend both --device cuda

# Test bicubic only
python examples/upscaling_comparison.py --backend bicubic

# Disabled Real-ESRGAN request (may fall back; not ML inference evidence)
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
