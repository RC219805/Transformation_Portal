"""Test upscaling color preservation (regression test for BGR swap bug).

This test validates that the upscaling backends preserve RGB channel ordering
and don't accidentally swap red/blue channels (the infamous BGR/RGB confusion).
"""

import numpy as np
import pytest


def test_bicubic_preserves_red_channel():
    """Bicubic should preserve pure red (regression: BGR swap would make it blue)."""
    from transformation_portal.upscaling.backends.bicubic import BicubicUpscaler

    upscaler = BicubicUpscaler()

    # Create a pure RED image (255, 0, 0) - RGB format
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, :, 0] = 255  # Red channel

    upscaled = upscaler.upscale(image, scale_factor=2.0)

    # Assert red channel is dominant (within tolerance for interpolation)
    assert upscaled[:, :, 0].mean() > 250, "Red channel should be preserved"
    assert upscaled[:, :, 1].mean() < 5, "Green should stay zero"
    assert upscaled[:, :, 2].mean() < 5, "Blue should stay zero (BGR swap would flip this)"


def test_bicubic_preserves_blue_channel():
    """Bicubic should preserve pure blue (regression: BGR swap would make it red)."""
    from transformation_portal.upscaling.backends.bicubic import BicubicUpscaler

    upscaler = BicubicUpscaler()

    # Create a pure BLUE image (0, 0, 255) - RGB format
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, :, 2] = 255  # Blue channel

    upscaled = upscaler.upscale(image, scale_factor=2.0)

    # Assert blue channel is dominant
    assert upscaled[:, :, 2].mean() > 250, "Blue channel should be preserved"
    assert upscaled[:, :, 0].mean() < 5, "Red should stay zero (BGR swap would flip this)"
    assert upscaled[:, :, 1].mean() < 5, "Green should stay zero"


def test_bicubic_float32_preserves_channels():
    """Bicubic should preserve channels for float32 (no precision loss regression)."""
    from transformation_portal.upscaling.backends.bicubic import BicubicUpscaler

    upscaler = BicubicUpscaler()

    # Create a subtle gradient in red channel only
    image = np.zeros((64, 64, 3), dtype=np.float32)
    image[:, :, 0] = np.linspace(0.0, 1.0, 64).reshape(1, -1)  # Horizontal gradient in red

    upscaled = upscaler.upscale(image, scale_factor=2.0)

    # Red channel should have gradient, others should be zero
    assert upscaled[:, :, 0].mean() > 0.4, "Red gradient should be preserved"
    assert upscaled[:, :, 0].max() > 0.95, "Red peak should be preserved"
    assert upscaled[:, :, 1].max() < 0.01, "Green should stay zero"
    assert upscaled[:, :, 2].max() < 0.01, "Blue should stay zero"


def _check_ml_deps_available() -> bool:
    """Check if ML dependencies are available."""
    try:
        import torch  # noqa: F401

        # Note: basicsr is blocked due to CVE-2024-27763
        # This will always return False until a safe alternative is implemented
        return False
    except ImportError:
        return False


@pytest.mark.skipif(
    not _check_ml_deps_available(),
    reason="ML dependencies not installed",
)
@pytest.mark.ml
def test_realesrgan_preserves_red_channel(monkeypatch):
    """Real-ESRGAN should preserve pure red (critical regression test)."""
    import logging

    logger = logging.getLogger(__name__)

    # Mock the model loading to avoid weight downloads
    def mock_load_model(self):
        """Mock model load - passthrough."""
        import torch

        class MockModel:
            def eval(self):
                return self

            def to(self, device):
                return self

            def half(self):
                return self

            def __call__(self, x):
                # Just return scaled-up input (nearest) for color preservation test
                import torch.nn.functional as F

                return F.interpolate(x, scale_factor=2, mode="nearest")

            netscale = 2

        self._model = MockModel()
        logger.info(f"Real-ESRGAN MOCKED for offline testing: {self._model_name}")

    from transformation_portal.upscaling.backends.realesrgan import RealESRGANUpscaler

    monkeypatch.setattr(RealESRGANUpscaler, "_load_model", mock_load_model)

    upscaler = RealESRGANUpscaler(device="cpu", model="RealESRGAN_x2plus")

    # Create a pure RED image (255, 0, 0) - RGB format
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, :, 0] = 255  # Red channel

    upscaled = upscaler.upscale(image, scale_factor=2.0)

    # This would FAIL with the old BGR swap bug (red would become blue)
    assert upscaled[:, :, 0].mean() > 250, "Red channel should be preserved (not swapped to blue)"
    assert upscaled[:, :, 2].mean() < 5, "Blue should stay zero (old bug would make this 255)"


@pytest.mark.skipif(
    not _check_ml_deps_available(),
    reason="ML dependencies not installed",
)
@pytest.mark.ml
def test_realesrgan_preserves_blue_channel(monkeypatch):
    """Real-ESRGAN should preserve pure blue (critical regression test)."""
    import logging

    logger = logging.getLogger(__name__)

    # Mock the model loading
    def mock_load_model(self):
        """Mock model load - passthrough."""
        import torch

        class MockModel:
            def eval(self):
                return self

            def to(self, device):
                return self

            def half(self):
                return self

            def __call__(self, x):
                import torch.nn.functional as F

                return F.interpolate(x, scale_factor=2, mode="nearest")

            netscale = 2

        self._model = MockModel()
        logger.info(f"Real-ESRGAN MOCKED for offline testing: {self._model_name}")

    from transformation_portal.upscaling.backends.realesrgan import RealESRGANUpscaler

    monkeypatch.setattr(RealESRGANUpscaler, "_load_model", mock_load_model)

    upscaler = RealESRGANUpscaler(device="cpu", model="RealESRGAN_x2plus")

    # Create a pure BLUE image (0, 0, 255) - RGB format
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, :, 2] = 255  # Blue channel

    upscaled = upscaler.upscale(image, scale_factor=2.0)

    # This would FAIL with the old BGR swap bug (blue would become red)
    assert upscaled[:, :, 2].mean() > 250, "Blue channel should be preserved (not swapped to red)"
    assert upscaled[:, :, 0].mean() < 5, "Red should stay zero (old bug would make this 255)"
