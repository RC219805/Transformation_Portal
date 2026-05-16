"""Unit coverage for depth normalization and mask loading."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from transformation_portal.depth import tools

pytestmark = pytest.mark.unit


def _write_depth(path, values: np.ndarray) -> None:
    Image.fromarray(values.astype(np.uint16), mode="I;16").save(path)


def test_load_depth_normalized_supports_percentile_histogram_and_linear(tmp_path) -> None:
    depth_path = tmp_path / "villa_depth16.png"
    values = np.linspace(0, 4095, 16 * 16, dtype=np.float32).reshape(16, 16)
    _write_depth(depth_path, values)

    percentile = tools.load_depth_normalized(str(depth_path), method="percentile", use_cache=False)
    histogram = tools.load_depth_normalized(str(depth_path), method="histogram", use_cache=False)
    linear = tools.load_depth_normalized(str(depth_path), method="linear", use_cache=False)

    for normalized in (percentile, histogram, linear):
        assert normalized.shape == (16, 16)
        assert 0.0 <= float(normalized.min()) <= float(normalized.max()) <= 1.0


def test_load_depth_normalized_resizes_to_target_size(tmp_path) -> None:
    depth_path = tmp_path / "villa_depth16.png"
    _write_depth(depth_path, np.arange(16 * 16, dtype=np.uint16).reshape(16, 16))

    normalized = tools.load_depth_normalized(str(depth_path), target_size=(4, 6), use_cache=False)

    assert normalized.shape == (4, 6)
    assert 0.0 <= float(normalized.min()) <= float(normalized.max()) <= 1.0


def test_load_depth_normalized_returns_cached_copy(tmp_path) -> None:
    tools._depth_cache.clear()
    depth_path = tmp_path / "villa_depth16.png"
    _write_depth(depth_path, np.full((8, 8), 1000, dtype=np.uint16))

    first = tools.load_depth_normalized(str(depth_path), method="linear", use_cache=True)
    first[0, 0] = 1.0
    _write_depth(depth_path, np.full((8, 8), 5000, dtype=np.uint16))
    cached = tools.load_depth_normalized(str(depth_path), method="linear", use_cache=True)

    assert cached[0, 0] != 1.0
    assert tools._depth_cache.stats()["hits"] == 1


def test_load_mask_missing_path_returns_zero_mask(tmp_path) -> None:
    missing = tmp_path / "missing_mask.png"

    mask = tools.load_mask(str(missing), "sky", (4, 6), use_cache=False)

    assert mask.shape == (4, 6)
    assert np.count_nonzero(mask) == 0


def test_load_mask_none_returns_zero_mask() -> None:
    mask = tools.load_mask(None, "building", (4, 6), use_cache=False)

    assert mask.shape == (4, 6)
    assert np.count_nonzero(mask) == 0


def test_load_mask_reads_l_mode_and_resizes(tmp_path) -> None:
    mask_path = tmp_path / "villa_mask_sky.png"
    Image.fromarray(np.full((8, 8), 128, dtype=np.uint8), mode="L").save(mask_path)

    mask = tools.load_mask(str(mask_path), "sky", (4, 6), use_cache=False)

    assert mask.shape == (4, 6)
    assert mask.mean() == pytest.approx(128.0 / 255.0, abs=0.01)


def test_load_mask_uses_rgba_alpha_channel(tmp_path) -> None:
    mask_path = tmp_path / "villa_mask_sky.png"
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    rgba[..., 3] = 204
    Image.fromarray(rgba, mode="RGBA").save(mask_path)

    mask = tools.load_mask(str(mask_path), "sky", (8, 8), use_cache=False)

    assert mask.mean() == pytest.approx(204.0 / 255.0)


def test_load_mask_converts_rgb_masks_to_grayscale(tmp_path) -> None:
    mask_path = tmp_path / "villa_mask_building.png"
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    Image.fromarray(rgb, mode="RGB").save(mask_path)

    mask = tools.load_mask(str(mask_path), "building", (8, 8), use_cache=False)

    assert mask.shape == (8, 8)
    assert 0.0 < float(mask.mean()) < 1.0


def test_load_mask_corrupt_file_falls_back_to_zero_mask(tmp_path) -> None:
    mask_path = tmp_path / "villa_mask_sky.png"
    mask_path.write_bytes(b"not an image")

    mask = tools.load_mask(str(mask_path), "sky", (4, 6), use_cache=False)

    assert mask.shape == (4, 6)
    assert np.count_nonzero(mask) == 0


def test_load_mask_returns_cached_copy(tmp_path) -> None:
    tools._mask_cache.clear()
    mask_path = tmp_path / "villa_mask_sky.png"
    Image.fromarray(np.full((8, 8), 255, dtype=np.uint8), mode="L").save(mask_path)

    first = tools.load_mask(str(mask_path), "sky", (8, 8), use_cache=True)
    first[0, 0] = 0.0
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8), mode="L").save(mask_path)
    cached = tools.load_mask(str(mask_path), "sky", (8, 8), use_cache=True)

    assert cached[0, 0] == 1.0
    assert tools._mask_cache.stats()["hits"] == 1


def test_load_mask_accepts_uniform_rgb_without_warning(tmp_path, caplog: pytest.LogCaptureFixture) -> None:
    # When all three channels match, the "differing channels" debug log
    # branch (lines 532-535) must NOT fire — the silent uniform-RGB path
    # at line 538 should win.
    mask_path = tmp_path / "villa_mask_building.png"
    rgb = np.full((8, 8, 3), 200, dtype=np.uint8)
    Image.fromarray(rgb, mode="RGB").save(mask_path)

    caplog.set_level("DEBUG", logger="depth_tools")
    mask = tools.load_mask(str(mask_path), "building", (8, 8), use_cache=False)

    assert mask.shape == (8, 8)
    assert mask.mean() == pytest.approx(200.0 / 255.0, abs=0.01)
    assert not any("differing channels" in record.message for record in caplog.records)


def test_load_mask_converts_palette_mode_to_grayscale(tmp_path) -> None:
    # 'P' (palette) mode is none of L/RGBA/RGB, so the else-branch
    # convert("L") on line 538 must run.
    mask_path = tmp_path / "villa_mask_sky.png"
    Image.fromarray(np.full((8, 8), 7, dtype=np.uint8), mode="P").save(mask_path)

    mask = tools.load_mask(str(mask_path), "sky", (4, 4), use_cache=False)

    assert mask.shape == (4, 4)
    assert 0.0 <= float(mask.min()) <= float(mask.max()) <= 1.0
