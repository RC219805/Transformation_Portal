"""Unit coverage for depth map utility functions (smoothing and hole inpainting)."""

from __future__ import annotations

import numpy as np
import pytest

from transformation_portal.depth.utils.depth_utils import CV2_AVAILABLE, inpaint_depth_holes, smooth_depth

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


@requires_cv2
@pytest.mark.parametrize("method", ["telea", "ns"])
def test_inpaint_depth_holes_fills_nan_holes_in_original_range(method):
    """Regression (D-04): NaN holes used to poison min/max and return an all-zero map."""
    depth = _ramp()
    holed = depth.copy()
    holed[10:14, 10:14] = np.nan

    out = inpaint_depth_holes(holed, method=method)

    assert np.isfinite(out).all()
    valid = ~np.isnan(holed)
    np.testing.assert_array_equal(out[valid], depth[valid])
    # Filled values come from the neighbourhood, in the input's depth units.
    np.testing.assert_allclose(out[10:14, 10:14], depth[10:14, 10:14], atol=0.05)


@requires_cv2
def test_inpaint_depth_holes_zero_holes_keep_valid_pixels_exact():
    """Regression (D-04): valid pixels used to be renormalised to [0, 1] including the 0-holes."""
    depth = _ramp()
    holed = depth.copy()
    holed[5:8, 20:23] = 0.0

    out = inpaint_depth_holes(holed)

    valid = holed != 0
    np.testing.assert_array_equal(out[valid], depth[valid])
    assert float(out.min()) >= 0.2 - 1e-6
    assert float(out.max()) <= 0.9 + 1e-6
    np.testing.assert_allclose(out[5:8, 20:23], depth[5:8, 20:23], atol=0.05)


@requires_cv2
def test_inpaint_depth_holes_explicit_mask_and_unknown_method():
    depth = _ramp(lo=1.0, hi=10.0)
    mask = np.zeros(depth.shape, dtype=np.uint8)
    mask[0:3, 0:3] = 1

    out = inpaint_depth_holes(depth, mask=mask)

    np.testing.assert_array_equal(out[mask == 0], depth[mask == 0])
    assert float(out[0:3, 0:3].min()) >= 1.0 - 1e-6
    with pytest.raises(ValueError):
        inpaint_depth_holes(depth, method="bogus")
