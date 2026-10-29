"""Exact integer proxy quantization for corrected geometry snapshots."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from transformation_portal.lux_depth_v3 import preprocessing

pytestmark = pytest.mark.unit


class _IntegerWorkspaceOnly(np.ndarray):
    """Reject a native floating-point quantization buffer at the array seam."""

    def astype(self, dtype, *args, **kwargs):
        if np.issubdtype(np.dtype(dtype), np.floating):
            raise AssertionError("Proxy quantization allocated a native floating-point buffer")
        return super().astype(dtype, *args, **kwargs)


@pytest.mark.parametrize("noncontiguous", [False, True])
def test_uint8_proxy_reuses_samples_without_conversion(noncontiguous):
    samples = np.arange(256, dtype=np.uint8).reshape(8, 8, 4)
    if noncontiguous:
        samples = samples[:, ::-1]
    original = samples.copy()

    result = preprocessing._geometry_proxy_uint8(samples)

    assert result is samples
    np.testing.assert_array_equal(result, original)


@pytest.mark.parametrize("layout", ["contiguous", "transposed", "reversed"])
def test_uint16_proxy_matches_both_float_formulas_for_every_sample(layout):
    samples = np.arange(65536, dtype=np.uint16).reshape(256, 256)
    if layout == "transposed":
        samples = samples.T
    elif layout == "reversed":
        samples = samples[::-1, ::-1]
    original = samples.copy()
    tiff_oracle = (samples.astype(np.float64) * (255.0 / 65535)).round().astype(np.uint8)
    png_oracle = np.rint(samples.astype(np.float64) / 257).astype(np.uint8)

    result = preprocessing._geometry_proxy_uint8(samples)

    assert result.dtype == np.uint8
    assert not np.shares_memory(result, samples)
    np.testing.assert_array_equal(result, tiff_oracle)
    np.testing.assert_array_equal(result, png_oracle)
    np.testing.assert_array_equal(samples, original)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_proxy_quantization_never_requests_native_floating_workspace(dtype):
    samples = np.arange(256, dtype=dtype).reshape(8, 8, 4).view(_IntegerWorkspaceOnly)
    result = preprocessing._geometry_proxy_uint8(samples)
    assert result.dtype == np.uint8


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("orientation", [1, 6])
def test_tiff_proxy_retains_exact_rgb_alpha_and_orientation(tmp_path, monkeypatch, dtype, orientation):
    tifffile = pytest.importorskip("tifffile")
    ramp = np.arange(65536, dtype=np.uint16).reshape(256, 256)
    samples = np.stack((ramp, np.roll(ramp, 17, axis=0), 65535 - ramp, np.roll(ramp, 23, axis=1)), axis=2)
    if dtype == np.uint8:
        samples = np.rint(samples.astype(np.float64) / 257).astype(np.uint8)
    source = tmp_path / "proxy.tif"
    tifffile.imwrite(
        source,
        samples,
        photometric="rgb",
        extrasamples="unassalpha",
        metadata=None,
        extratags=[(274, "H", 1, orientation, False)],
    )
    _assert_snapshot_proxy(source, samples, orientation, monkeypatch)


def test_png16_proxy_retains_exact_rgb_and_alpha(tmp_path, monkeypatch):
    imagecodecs = pytest.importorskip("imagecodecs")
    ramp = np.arange(65536, dtype=np.uint16).reshape(256, 256)
    samples = np.stack((ramp, np.roll(ramp, 17, axis=0), 65535 - ramp, np.roll(ramp, 23, axis=1)), axis=2)
    source = tmp_path / "proxy.png"
    source.write_bytes(imagecodecs.png_encode(samples))
    _assert_snapshot_proxy(source, samples, 1, monkeypatch)


def _assert_snapshot_proxy(source, samples, orientation, monkeypatch):
    original_bytes = source.read_bytes()
    source_bits = samples.dtype.itemsize * 8
    expected = samples if source_bits == 8 else np.rint(samples.astype(np.float64) / 257).astype(np.uint8)
    if orientation == 6:
        expected = np.rot90(expected, -1)
    original_quantizer = preprocessing._geometry_proxy_uint8
    quantized_dtypes = []

    def guarded_quantization(pixels):
        quantized_dtypes.append(pixels.dtype)
        return original_quantizer(pixels.view(_IntegerWorkspaceOnly))

    monkeypatch.setattr(preprocessing, "_geometry_proxy_uint8", guarded_quantization)
    result = preprocessing.preprocess_geometry_snapshot(source, verify_snapshot=True)

    assert quantized_dtypes == [samples.dtype]
    assert result.source_bit_depth == source_bits
    assert result.transform.orientation == orientation
    assert result.transform.source_sha256 == hashlib.sha256(original_bytes).hexdigest()
    np.testing.assert_array_equal(result.canonical_rgb, expected[..., :3])
    np.testing.assert_array_equal(result.alpha, expected[..., 3])
    np.testing.assert_array_equal(result.pixels, expected[2:254, 2:254, :3].astype(np.float32) / 255)
    assert source.read_bytes() == original_bytes
