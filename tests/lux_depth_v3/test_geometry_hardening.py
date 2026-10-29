"""Malformed photographic metadata cannot authorize corrected geometry."""

from __future__ import annotations

import hashlib
import struct
import zlib
from dataclasses import replace

import numpy as np
import pytest
import tifffile
from PIL import Image

from transformation_portal.lux_depth_v3.geometry import GeometryPreprocessResult, GeometryTransform
from transformation_portal.lux_depth_v3.preprocessing import preprocess_geometry_snapshot

pytestmark = pytest.mark.unit


def _chunk(kind, payload):
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))


def _png16(samples):
    """Write independent unfiltered uint16 PNG bytes, without a raster codec."""
    channels = 1 if samples.ndim == 2 else samples.shape[2]
    color_type = {1: 0, 3: 2, 4: 6}[channels]
    header = struct.pack(">IIBBBBB", samples.shape[1], samples.shape[0], 16, color_type, 0, 0, 0)
    rows = b"".join(b"\x00" + row.astype(">u2").tobytes() for row in samples)
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", header) + _chunk(b"IDAT", zlib.compress(rows)) + _chunk(b"IEND", b"")


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_png_precision_authority_is_not_replaced_by_8bit_proxy_precision(tmp_path, channels):
    samples = np.full((31, 45), 30000, np.uint16)
    if channels != 1:
        samples = np.repeat(samples[..., None], channels, axis=2)
    path = tmp_path / "native16.png"
    path.write_bytes(_png16(samples))
    original = path.read_bytes()
    result = preprocess_geometry_snapshot(path)
    assert result.source_bit_depth == 16
    assert result.canonical_rgb.dtype == np.uint8
    assert result.transform.source_sha256 == hashlib.sha256(original).hexdigest()
    assert path.read_bytes() == original
    assert (result.alpha is not None) is (channels == 4)


def test_uint16_gray_proxy_uses_full_sample_range_instead_of_clipping_at_255(tmp_path):
    samples = np.tile(np.array([0, 10000, 30000, 60000, 65535], np.uint16), (31, 9))
    path = tmp_path / "gray16.png"
    path.write_bytes(_png16(samples))
    result = preprocess_geometry_snapshot(path)
    expected = np.rint(samples.astype(np.float64) / 257).astype(np.uint8)
    np.testing.assert_array_equal(result.canonical_rgb, np.repeat(expected[..., None], 3, axis=2))


@pytest.mark.parametrize("mode,key", [("L", 128), ("RGB", (19, 53, 91))])
def test_png_color_key_alpha_survives_orientation_before_materials_admission(tmp_path, mode, key):
    shape = (31, 45) if mode == "L" else (31, 45, 3)
    samples = np.full(shape, 7, np.uint8)
    samples[3:9, 5:13] = key
    exif = Image.Exif()
    exif[274] = 2
    path = tmp_path / "color-key.png"
    Image.fromarray(samples, mode=mode).save(path, transparency=key, exif=exif)
    result = preprocess_geometry_snapshot(path)
    expected = np.full((31, 45), 255, np.uint8)
    expected[3:9, 5:13] = 0
    np.testing.assert_array_equal(result.alpha, expected[:, ::-1])
    assert result.transform.orientation == 2


def test_tiff_pixels_decode_once_even_with_verification_and_orientation(tmp_path, monkeypatch):
    samples = np.full((31, 45, 3), 30000, np.uint16)
    path = tmp_path / "oriented16.tif"
    tifffile.imwrite(path, samples, photometric="rgb", metadata=None, extratags=[(274, "H", 1, 6, False)])
    original = tifffile.TiffPage.asarray
    calls = []

    def counted(page, *args, **kwargs):
        calls.append(page.shape)
        return original(page, *args, **kwargs)

    monkeypatch.setattr(tifffile.TiffPage, "asarray", counted)
    result = preprocess_geometry_snapshot(path, verify_snapshot=True)
    assert calls == [(31, 45, 3)]
    assert result.source_bit_depth == 16
    assert result.transform.orientation == 6
    assert result.canonical_rgb.shape == (45, 31, 3)


@pytest.mark.parametrize("orientation", [1.0, 6.5])
def test_tiff_float_orientation_is_rejected_instead_of_coerced(tmp_path, orientation):
    path = tmp_path / "invalid-orientation.tif"
    tifffile.imwrite(
        path,
        np.full((31, 45, 3), 70, np.uint8),
        photometric="rgb",
        metadata=None,
        extratags=[(274, "d", 1, orientation, False)],
    )
    original = path.read_bytes()
    with pytest.raises(ValueError):
        preprocess_geometry_snapshot(path)
    assert path.read_bytes() == original


@pytest.mark.parametrize("restore", ["restore_numeric", "restore_mask"])
def test_finite_float64_model_overflow_cannot_create_nonfinite_float32_evidence(restore):
    transform = GeometryTransform("a" * 64, (14, 14), 1)
    values = np.full((14, 14), np.finfo(np.float64).max)
    with pytest.raises(ValueError, match="finite float32"):
        getattr(transform, restore)(values)


@pytest.mark.parametrize("shape,target", [((1, 1), 10**1000), ((10**1000, 1), 1), ((1, 10**1000), None)])
def test_declared_geometry_budget_is_checked_before_floating_point_arithmetic(shape, target):
    with pytest.raises(ValueError, match="pixel budget"):
        GeometryTransform("a" * 64, shape, 1, target)


@pytest.mark.parametrize(
    "mutation",
    [
        "depth_bool",
        "depth_float",
        "depth_unsupported",
        "model_dtype",
        "model_nan",
        "model_low",
        "model_high",
        "rgb_dtype",
        "alpha_dtype",
        "alpha_shape",
    ],
)
def test_preprocess_result_rejects_malformed_declared_sample_evidence(mutation):
    transform = GeometryTransform("a" * 64, (14, 14), 1)
    result = GeometryPreprocessResult(
        np.full((14, 14, 3), 0.3, np.float32), np.full((14, 14, 3), 70, np.uint8), None, transform
    )
    changes = {}
    if mutation.startswith("depth"):
        changes["source_bit_depth"] = {"depth_bool": True, "depth_float": 8.0, "depth_unsupported": 32}[mutation]
    elif mutation.startswith("model"):
        if mutation == "model_dtype":
            changes["pixels"] = result.pixels.astype(np.float64)
        else:
            model = result.pixels.copy()
            model[0, 0, 0] = {"model_nan": np.nan, "model_low": -0.1, "model_high": 1.1}[mutation]
            changes["pixels"] = model
    elif mutation == "rgb_dtype":
        changes["canonical_rgb"] = result.canonical_rgb.astype(np.float32)
    elif mutation == "alpha_dtype":
        changes["alpha"] = np.full((14, 14), 1, np.float32)
    else:
        changes["alpha"] = np.full((13, 14), 255, np.uint8)
    with pytest.raises(ValueError):
        replace(result, **changes)
