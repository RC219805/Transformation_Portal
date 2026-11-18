"""Regression coverage for the depth pipeline's image load/save range handling (finding D-24)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from transformation_portal.depth.utils import image_utils
from transformation_portal.depth.utils.image_utils import load_image, save_image

pytestmark = [pytest.mark.unit, pytest.mark.regression]


@pytest.fixture(params=[False, True], ids=["pil", "opencv"])
def writer_backend(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    if request.param:
        pytest.importorskip("cv2")
    monkeypatch.setattr(image_utils, "CV2_AVAILABLE", request.param)
    return "opencv" if request.param else "pil"


def test_load_image_scales_near_black_8bit_frame_by_dtype(tmp_path: Path) -> None:
    dark = np.zeros((8, 8, 3), np.uint8)
    dark[2:4, 2:4] = 1
    Image.fromarray(dark).save(tmp_path / "dark.png")

    loaded = load_image(tmp_path / "dark.png", normalize=True)

    assert loaded.dtype == np.float32
    np.testing.assert_allclose(float(loaded.max()), 1.0 / 255.0, rtol=1e-6)


def test_load_image_regular_8bit_frame_is_unchanged(tmp_path: Path) -> None:
    rgb = np.random.default_rng(7).integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    rgb[0, 0, 0] = 255
    Image.fromarray(rgb).save(tmp_path / "rgb.png")

    loaded = load_image(tmp_path / "rgb.png", normalize=True)

    np.testing.assert_array_equal(loaded, rgb.astype(np.float32) / 255.0)


def test_save_image_float_overshoot_is_not_reinterpreted_as_8bit(tmp_path: Path, writer_backend: str) -> None:
    image = np.full((8, 8, 3), 0.6, np.float32)
    image[0, 0, 0] = 1.000001

    save_image(image, tmp_path / "over.png")
    back = np.asarray(Image.open(tmp_path / "over.png"))

    assert back[0, 0, 0] == 255
    assert set(np.unique(back[1:, 1:]).tolist()) == {int(np.float32(0.6) * 255)}


def test_save_image_near_black_roundtrip_keeps_code_values(tmp_path: Path, writer_backend: str) -> None:
    dark = np.zeros((8, 8, 3), np.uint8)
    dark[2:4, 2:4] = 1
    Image.fromarray(dark).save(tmp_path / "dark.png")

    save_image(load_image(tmp_path / "dark.png", normalize=True), tmp_path / "dark_out.png")

    np.testing.assert_array_equal(np.asarray(Image.open(tmp_path / "dark_out.png")), dark)


def test_save_image_in_range_float_keeps_existing_quantisation(tmp_path: Path, writer_backend: str) -> None:
    image = np.random.default_rng(11).random((8, 8, 3)).astype(np.float32)

    save_image(image, tmp_path / "in_range.png")

    np.testing.assert_array_equal(np.asarray(Image.open(tmp_path / "in_range.png")), (image * 255).astype(np.uint8))


def test_save_image_scales_uint16_instead_of_wrapping(tmp_path: Path, writer_backend: str) -> None:
    image = np.full((4, 4, 3), 65000, np.uint16)
    image[0, 0] = 30000

    save_image(image, tmp_path / "deep.png")
    back = np.asarray(Image.open(tmp_path / "deep.png"))

    assert back[0, 0, 0] == round(30000 / 257)
    assert int(back[1:, 1:].min()) == round(65000 / 257)


def test_load_image_scales_raw_16bit_frame_by_dtype(tmp_path: Path) -> None:
    """A raw 16-bit decode is normalised by 65535; dividing by 255 left values far above 1.0."""
    deep = np.zeros((8, 8), np.uint16)
    deep[2:4, 2:4] = 40000
    Image.fromarray(deep).save(tmp_path / "deep.png")

    loaded = load_image(tmp_path / "deep.png", color_space="BGR", normalize=True)

    assert loaded.dtype == np.float32
    np.testing.assert_allclose(float(loaded.max()), 40000 / 65535, rtol=1e-6)
