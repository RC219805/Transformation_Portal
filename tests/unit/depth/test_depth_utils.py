"""Unit coverage for depth map utility functions (zoning, smoothing and hole inpainting)."""

from __future__ import annotations

import numpy as np
import pytest

from transformation_portal.depth.utils.depth_utils import CV2_AVAILABLE, create_depth_zones, inpaint_depth_holes, smooth_depth

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


def _half_zero_ramp() -> np.ndarray:
    depth = np.zeros((16, 32), dtype=np.float32)
    depth[:, 16:] = np.linspace(0.01, 1.0, 16, dtype=np.float32)[None, :]
    return depth


@pytest.mark.parametrize("smooth_sigma", [0.0, 2.0])
def test_create_depth_zones_log_assigns_every_pixel_and_spreads_zones(smooth_sigma):
    """Regression (D-05): zero depth stretched the log range and left pixels in no zone."""
    depth = _half_zero_ramp()

    zones = create_depth_zones(depth, num_zones=3, method="log", smooth_sigma=smooth_sigma)

    np.testing.assert_allclose(np.sum(zones, axis=0), 1.0, atol=1e-5)
    hard = create_depth_zones(depth, num_zones=3, method="log", smooth_sigma=0.0)
    # Zero depth belongs to the nearest zone and the positive ramp is split
    # across all zones instead of collapsing into the last one.
    assert (hard[0][:, :16] == 1.0).all()
    assert all(float(z[:, 16:].sum()) > 0 for z in hard)


def test_create_depth_zones_log_includes_minimum_of_positive_depth():
    """Regression (D-05): the +1e-8 log boundary excluded the minimum-depth pixels from every zone."""
    depth = np.tile(np.linspace(0.05, 1.0, 20, dtype=np.float32), (4, 1))

    zones = create_depth_zones(depth, num_zones=4, method="log", smooth_sigma=0.0)

    np.testing.assert_allclose(np.sum(zones, axis=0), 1.0, atol=1e-5)
    assert (zones[0][:, 0] == 1.0).all()


@pytest.mark.parametrize("method", ["percentile", "linear"])
def test_create_depth_zones_percentile_and_linear_partition_unchanged(method):
    depth = _noisy_ramp()

    zones = create_depth_zones(depth, num_zones=3, method=method, smooth_sigma=0.0)

    if method == "percentile":
        bounds = np.percentile(depth, np.linspace(0, 100, 4))
    else:
        bounds = np.linspace(depth.min(), depth.max(), 4)
    expected = [
        (depth >= bounds[0]) & (depth < bounds[1]),
        (depth >= bounds[1]) & (depth < bounds[2]),
        (depth >= bounds[2]) & (depth <= bounds[3]),
    ]
    for zone, mask in zip(zones, expected):
        np.testing.assert_allclose(zone, mask.astype(np.float32), atol=1e-6)


@requires_cv2
def test_smooth_depth_bilateral_keeps_metric_depth_range():
    """Regression (D-06): bilateral smoothing renormalised metric depth to [0, 1]."""
    depth = np.tile(np.linspace(2.0, 30.0, 256, dtype=np.float32), (8, 1))

    out = smooth_depth(depth, method="bilateral", sigma=2.0)

    assert out.dtype == np.float32
    assert float(out.min()) >= 2.0 - 1e-4
    assert float(out.max()) <= 30.0 + 1e-4
    # Within a few 8-bit quantisation steps of the 28 m span.
    np.testing.assert_allclose(out, depth, atol=3 * 28.0 / 255.0)


@requires_cv2
def test_smooth_depth_bilateral_unit_range_output_unchanged():
    """Depth already spanning exactly [0, 1] keeps the historical 8-bit bilateral result."""
    import cv2

    depth = _noisy_ramp().clip(0.0, 1.0)
    depth[0, 0], depth[0, 1] = 0.0, 1.0
    expected = cv2.bilateralFilter((depth * 255).astype(np.uint8), d=10, sigmaColor=0.1 * 255, sigmaSpace=5.0)

    out = smooth_depth(depth, method="bilateral", sigma=5.0)

    np.testing.assert_array_equal(out, expected.astype(np.float32) / 255.0)


@requires_cv2
def test_smooth_depth_bilateral_constant_depth_stays_constant():
    depth = np.full((8, 8), 7.5, dtype=np.float32)

    np.testing.assert_array_equal(smooth_depth(depth, method="bilateral", sigma=2.0), depth)


@pytest.mark.parametrize("explicit_mask", [False, True])
def test_inpaint_depth_holes_without_opencv_treats_non_finite_pixels_as_holes(explicit_mask, monkeypatch):
    """D-04: without OpenCV, NaN/inf pixels are holes too and must not poison the fill value."""
    monkeypatch.setattr("transformation_portal.depth.utils.depth_utils.CV2_AVAILABLE", False)
    depth = _ramp(lo=1.0, hi=10.0)
    holed = depth.copy()
    holed[4, 4] = np.nan
    holed[5, 5] = np.inf
    mask = None
    hole = ~np.isfinite(holed)
    if explicit_mask:
        mask = np.zeros(depth.shape, dtype=np.uint8)
        mask[0:3, 0:3] = 1
        hole |= mask.astype(bool)

    out = inpaint_depth_holes(holed, mask=mask)

    assert np.isfinite(out).all()
    np.testing.assert_array_equal(out[~hole], depth[~hole])
    np.testing.assert_allclose(out[hole], holed[~hole].mean(), rtol=1e-6)


@requires_cv2
def test_inpaint_depth_holes_constant_depth_keeps_its_value():
    """D-04: constant valid depth has no span to normalise over; holes take that constant, not 0 or 1."""
    depth = np.full((16, 16), 5.0, dtype=np.float32)
    depth[4:6, 4:6] = 0.0

    np.testing.assert_array_equal(inpaint_depth_holes(depth), np.full((16, 16), 5.0, dtype=np.float32))


@requires_cv2
def test_inpaint_depth_holes_without_valid_pixels_returns_zeros():
    out = inpaint_depth_holes(np.full((8, 8), np.nan, dtype=np.float32))

    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, np.zeros((8, 8), dtype=np.float32))
