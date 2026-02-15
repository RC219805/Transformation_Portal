"""Upscaler backends."""

from .bicubic import BicubicUpscaler

__all__ = ["BicubicUpscaler"]

# Optional ML backend (lazy import)
try:
    from .realesrgan import RealESRGANUpscaler  # noqa: F401

    __all__.append("RealESRGANUpscaler")
except ImportError:
    pass
