from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from process_renderings_750 import (
    CONVERTIBLE_IMAGE_SUFFIXES,
    SUPPORTED_IMAGE_SUFFIXES,
    convert_renderings_to_jpeg,
    ensure_supported_renderings,
)


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
