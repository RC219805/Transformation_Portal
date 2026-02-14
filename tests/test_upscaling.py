"""Test upscaler backends."""

import numpy as np
import pytest

from transformation_portal.upscaling import UpscalerRegistry


def _check_ml_deps_available() -> bool:
    """Check if ML dependencies are available."""
    try:
        import torch  # noqa: F401
        from basicsr.archs.rrdbnet_arch import RRDBNet  # noqa: F401

        return True
    except ImportError:
        return False


def test_bicubic_upscaler():
    """Test bicubic upscaler (always available)."""
    registry = UpscalerRegistry()

    # Get bicubic backend
    upscaler = registry.get("bicubic")

    assert upscaler.name == "bicubic"
    assert upscaler.requires_ml is False

    # Test upscaling
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    upscaled = upscaler.upscale(image, scale_factor=2.0)

    assert upscaled.shape == (200, 200, 3)
    assert upscaled.dtype == np.uint8


def test_bicubic_upscaler_float32():
    """Test bicubic upscaler with float32 input."""
    registry = UpscalerRegistry()
    upscaler = registry.get("bicubic")

    # Test with float32 [0, 1]
    image = np.random.rand(100, 100, 3).astype(np.float32)
    upscaled = upscaler.upscale(image, scale_factor=2.0)

    assert upscaled.shape == (200, 200, 3)
    assert upscaled.dtype == np.float32
    assert upscaled.min() >= 0.0
    assert upscaled.max() <= 1.0


def test_registry_list_backends():
    """Test registry backend listing."""
    registry = UpscalerRegistry()
    backends = registry.list_backends()

    assert "bicubic" in backends
    assert backends["bicubic"]["requires_ml"] is False


def test_registry_fallback():
    """Test graceful fallback to bicubic when backend unavailable."""
    registry = UpscalerRegistry()

    # Request unknown backend with fallback
    upscaler = registry.get("unknown_backend", fallback_to_bicubic=True)

    # Should fallback to bicubic
    assert upscaler.name == "bicubic"


def test_registry_no_fallback():
    """Test error when backend unavailable and fallback disabled."""
    registry = UpscalerRegistry()

    # Request unknown backend without fallback
    with pytest.raises(ValueError, match="Unknown upscaler backend"):
        registry.get("unknown_backend", fallback_to_bicubic=False)


def test_default_alias():
    """Test 'default' alias for bicubic."""
    registry = UpscalerRegistry()
    upscaler = registry.get("default")

    assert upscaler.name == "bicubic"


@pytest.mark.skipif(
    not _check_ml_deps_available(),
    reason="ML dependencies not installed",
)
def test_realesrgan_upscaler():
    """Test Real-ESRGAN upscaler (requires ML deps)."""
    registry = UpscalerRegistry()

    # Get Real-ESRGAN backend
    upscaler = registry.get("realesrgan", device="cpu", model="RealESRGAN_x2plus")

    assert upscaler.name == "realesrgan"
    assert upscaler.requires_ml is True

    # Test upscaling (small image to avoid long test time)
    image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    upscaled = upscaler.upscale(image, scale_factor=2.0)

    assert upscaled.shape == (100, 100, 3)
    assert upscaled.dtype == np.uint8
