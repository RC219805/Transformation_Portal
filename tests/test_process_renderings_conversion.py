from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.transformation_portal.rendering.process_renderings_750 import (
    CONVERTIBLE_IMAGE_SUFFIXES,
    SUPPORTED_IMAGE_SUFFIXES,
    _load_rgb,
    convert_renderings_to_jpeg,
    ensure_supported_renderings,
)

pytestmark = pytest.mark.unit


def _write_dummy_image(path: Path) -> None:
    # Create a 1D array of 16 values from 0 to 255
    grayscale_values = np.linspace(0, 255, 16, dtype=np.uint8)
    # Reshape to a 4x4 2D array
    grayscale_image = grayscale_values.reshape(4, 4)
    # Add a channel dimension to make it (4, 4, 1)
    grayscale_image_3d = grayscale_image[:, :, None]
    # Repeat the single channel 3 times to create an RGB image (4, 4, 3)
    rgb_image = np.repeat(grayscale_image_3d, 3, axis=2)
    Image.fromarray(rgb_image, mode="RGB").save(path)


def test_convert_renderings_to_jpeg_creates_jpg(tmp_path: Path) -> None:
    source = tmp_path / "example.tif"
    _write_dummy_image(source)

    converted_dir = convert_renderings_to_jpeg(tmp_path)
    converted_file = converted_dir / "example.jpg"

    assert converted_file.exists()
    assert converted_file.suffix == ".jpg"


def test_ensure_supported_renderings_returns_original_when_already_supported(tmp_path: Path) -> None:
    source = tmp_path / "example.jpg"
    _write_dummy_image(source)

    normalized = ensure_supported_renderings(tmp_path)
    assert normalized == tmp_path


def test_supported_and_convertible_sets_are_disjoint() -> None:
    assert not (CONVERTIBLE_IMAGE_SUFFIXES & SUPPORTED_IMAGE_SUFFIXES)


def test_convert_renderings_to_jpeg_scales_16bit_greyscale_tiff(tmp_path: Path) -> None:
    """Regression for H-05: a 16-bit grey TIFF (mode I;16) was clipped at 255 and converted to white."""
    Image.fromarray(np.full((16, 16), 32896, dtype=np.uint16)).save(tmp_path / "grey16.tif")

    converted_dir = convert_renderings_to_jpeg(tmp_path)

    with Image.open(converted_dir / "grey16.jpg") as converted:
        pixels = np.asarray(converted).astype(np.int16)
    assert np.abs(pixels - 128).max() <= 2


def test_load_rgb_scales_16bit_greyscale(tmp_path: Path) -> None:
    path = tmp_path / "grey16.png"
    Image.fromarray(np.array([[0, 16448, 32896, 65535]], dtype=np.uint16)).save(path)

    rgb = _load_rgb(path)

    assert rgb.shape == (1, 4, 3)
    assert np.allclose(rgb[0, :, 0], np.array([0, 64, 128, 255]) / 255.0)
