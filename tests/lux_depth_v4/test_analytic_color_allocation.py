"""Bound analytic decode scratch while preserving historical float32 samples."""

from __future__ import annotations

import numpy as np
import pytest

from transformation_portal.lux_depth_v4 import color_preparation as color

pytestmark = pytest.mark.unit


def _historical(samples):
    rgb = np.repeat(samples[..., None], 3, axis=2) if samples.ndim == 2 else samples
    values = np.asarray(rgb, dtype=np.float32)
    return np.where(values <= 0.04045, values / 12.92, ((np.maximum(values, 0) + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _assert_exact(samples, monkeypatch, *, pixels=7):
    expected = _historical(samples)
    original = samples.copy()
    convert = color.srgb_to_linear
    calls = []

    def bounded(tile):
        calls.append(tile.shape[:2])
        assert tile.shape[0] * tile.shape[1] <= pixels
        return convert(tile)

    monkeypatch.setattr(color, "_TILE_PIXELS", pixels)
    monkeypatch.setattr(color, "srgb_to_linear", bounded)
    result = color.apply_input_color(samples, color.PreparedColor("srgb", "explicit", "convert"))
    assert result.shape == (*samples.shape[:2], 3) and result.dtype == np.float32
    np.testing.assert_array_equal(result.view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(samples, original)
    assert not np.shares_memory(result, samples)
    assert calls and sum(height * width for height, width in calls) == samples.shape[0] * samples.shape[1]


@pytest.mark.parametrize("gray", [False, True])
@pytest.mark.parametrize("layout", ["c", "fortran", "strided", "negative", "transpose"])
@pytest.mark.parametrize("shape", [(1, 23), (23, 1), (7, 13), (13, 7), (1, 1)])
def test_analytic_conversion_is_bit_exact_for_bounded_two_axis_tiles(monkeypatch, gray, layout, shape):
    values = np.linspace(-0.25, 1.75, np.prod(shape) * (1 if gray else 3), dtype=np.float32)
    samples = values.reshape(shape if gray else (*shape, 3))
    if layout == "fortran":
        samples = np.asfortranarray(samples)
    elif layout == "strided":
        samples = samples[:, ::2]
    elif layout == "negative":
        samples = samples[::-1, ::-1]
    elif layout == "transpose":
        samples = samples.swapaxes(0, 1)
    _assert_exact(samples, monkeypatch)


@pytest.mark.parametrize("gray", [False, True])
def test_threshold_neighbors_signed_zero_and_extended_values_preserve_bits(monkeypatch, gray):
    threshold = np.float32(0.04045)
    values = np.array(
        [
            -10,
            -0.0,
            0,
            np.nextafter(threshold, np.float32(0)),
            threshold,
            np.nextafter(threshold, np.float32(1)),
            1,
            1.25,
            100,
        ],
        np.float32,
    )
    samples = values.reshape((3, 3)) if gray else np.repeat(values.reshape((3, 3, 1)), 3, axis=2)
    _assert_exact(samples, monkeypatch, pixels=2)


@pytest.mark.parametrize("gray", [False, True])
@pytest.mark.parametrize("shape", [(256, 256), (1, 65536), (65536, 1)])
def test_exhaustive_integer_16bit_ramp_matches_historical_float32_transfer(monkeypatch, gray, shape):
    ramp = np.arange(65536, dtype=np.uint16).astype(np.float32).reshape(shape)
    ramp /= 65535.0
    samples = ramp if gray else np.repeat(ramp[..., None], 3, axis=2)
    _assert_exact(samples, monkeypatch, pixels=257)


def test_rgba_view_preserves_source_pixels_and_separate_alpha(monkeypatch):
    rgba = np.linspace(0, 1, 11 * 17 * 4, dtype=np.float32).reshape((11, 17, 4))
    original = rgba.copy()
    _assert_exact(rgba[..., :3], monkeypatch)
    np.testing.assert_array_equal(rgba, original)
    np.testing.assert_array_equal(rgba[..., 3], original[..., 3])


def test_linear_path_keeps_original_alias_and_gray_expansion(monkeypatch):
    def forbidden(_samples):
        pytest.fail("Linear identity must not invoke the analytic transfer")

    monkeypatch.setattr(color, "srgb_to_linear", forbidden)
    prepared = color.PreparedColor("linear_srgb", "explicit", "identity")
    rgb = np.arange(18, dtype=np.float32).reshape((2, 3, 3))
    assert color.apply_input_color(rgb, prepared) is rgb
    gray = rgb[..., 0]
    np.testing.assert_array_equal(color.apply_input_color(gray, prepared), np.repeat(gray[..., None], 3, axis=2))
