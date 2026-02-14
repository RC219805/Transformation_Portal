"""Upscaler backend registry and protocols.

Provides a registry-based system for selecting upscaling backends with
graceful fallback to bicubic when ML dependencies are unavailable.

Usage:
    >>> from transformation_portal.upscaling import UpscalerRegistry
    >>> registry = UpscalerRegistry()
    >>>
    >>> # Golden Path: bicubic (always available)
    >>> upscaler = registry.get("bicubic")
    >>>
    >>> # ML tier: Real-ESRGAN (requires ML dependencies)
    >>> upscaler = registry.get("realesrgan", device="cuda", model="RealESRGAN_x2plus")
    >>>
    >>> # Auto-fallback if ML deps missing
    >>> upscaler = registry.get("realesrgan", fallback_to_bicubic=True)  # -> bicubic

Example:
    >>> import numpy as np
    >>> from transformation_portal.upscaling import UpscalerRegistry
    >>>
    >>> # Create registry
    >>> registry = UpscalerRegistry()
    >>>
    >>> # Get backend
    >>> upscaler = registry.get("bicubic")
    >>>
    >>> # Upscale image
    >>> image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
    >>> upscaled = upscaler.upscale(image, scale_factor=2.0)
    >>> upscaled.shape
    (2000, 2000, 3)
"""

from .protocol import UpscalerBackend
from .registry import UpscalerRegistry, get_registry

__all__ = [
    "UpscalerBackend",
    "UpscalerRegistry",
    "get_registry",
]

# Version
__version__ = "1.0.0"
