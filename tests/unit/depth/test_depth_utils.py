"""Unit coverage for depth map utility functions."""

from __future__ import annotations

import numpy as np
import pytest

from transformation_portal.depth.utils.depth_utils import CV2_AVAILABLE, smooth_depth

pytestmark = pytest.mark.unit

requires_cv2 = pytest.mark.skipif(not CV2_AVAILABLE, reason="opencv-python required")


def _noisy_ramp(size: int = 32, dtype=np.float32) -> np.ndarray:
    rng = np.random.default_rng(0)
    ramp = np.tile(np.linspace(0.0, 1.0, size), (size, 1)) + rng.normal(0, 0.05, (size, size))
    return ramp.astype(dtype)


def _ramp(lo: float = 0.2, hi: float = 0.9, size: int = 32) -> np.ndarray:
    return np.tile(np.linspace(lo, hi, size, dtype=np.float32), (size, 1))


@requires_cv2
@pytest.mark.parametrize("sigma", [2.5, 3.0, 5.0])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_smooth_depth_median_supports_float_depth_at_large_and_even_apertures(sigma, dtype):
    """Regression (D-03): cv2.medianBlur rejects float input beyond aperture 5 and even apertures."""
    from scipy.ndimage import median_filter

    depth = _noisy_ramp(dtype=dtype)
    out = smooth_depth(depth, method="median", sigma=sigma)

    ksize = (int(sigma * 2) + 1) | 1
    assert out.shape == depth.shape
    assert out.dtype == depth.dtype
    np.testing.assert_array_equal(out, median_filter(depth, size=ksize, mode="nearest"))


@requires_cv2
@pytest.mark.parametrize("sigma", [1.0, 2.0])
def test_smooth_depth_median_small_float32_apertures_match_opencv(sigma):
    import cv2

    depth = _noisy_ramp()
    expected = cv2.medianBlur(depth, int(sigma * 2) + 1)
    np.testing.assert_array_equal(smooth_depth(depth, method="median", sigma=sigma), expected)


@requires_cv2
def test_smooth_depth_median_uint8_matches_opencv():
    import cv2

    depth = (_noisy_ramp().clip(0, 1) * 255).astype(np.uint8)
    np.testing.assert_array_equal(smooth_depth(depth, method="median", sigma=5.0), cv2.medianBlur(depth, 11))
