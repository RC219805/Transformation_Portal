"""Unit coverage for depth tools validation and save helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from transformation_portal.depth import tools

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("color", "expected"),
    [
        ((0.1, 0.2, 0.3), (0.1, 0.2, 0.3)),
        ((255.0, 128.0, 0.0), (1.0, 128.0 / 255.0, 0.0)),
    ],
)
def test_validate_color_accepts_unit_and_byte_ranges(
    color: tuple[float, float, float], expected: tuple[float, float, float]
) -> None:
    assert tools.validate_color(color, "accent") == pytest.approx(expected)


@pytest.mark.parametrize(
    ("color", "match"),
    [
        ((1.0, 0.5), "3 components"),
        ((300.0, 0.0, 0.0), "0..1 or 0..255"),
        ((-0.1, 0.0, 0.0), "0..1 range"),
    ],
)
def test_validate_color_rejects_invalid_shapes_and_ranges(color: tuple[float, ...], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        tools.validate_color(color, "accent")  # type: ignore[arg-type]


def test_validate_file_exists_accepts_readable_files(tmp_path) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"not an image but readable")

    tools.validate_file_exists(str(source), "Source")


def test_validate_file_exists_rejects_missing_paths(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Source not found"):
        tools.validate_file_exists(str(tmp_path / "missing.png"), "Source")


def test_validate_file_exists_rejects_directories(tmp_path) -> None:
    with pytest.raises(ValueError, match="Source is not a file"):
        tools.validate_file_exists(str(tmp_path), "Source")


def test_save_image_rgb_accepts_default_tiff_alias(tmp_path) -> None:
    image = np.full((8, 8, 3), 0.5, dtype=np.float32)

    saved = tools.save_image_rgb(str(tmp_path / "render.tiff"), image, fmt="tiff")

    assert saved.endswith((".tif", ".png"))
    assert (tmp_path / Path(saved).name).exists()


def test_save_image_rgb_rejects_unknown_format(tmp_path) -> None:
    image = np.full((8, 8, 3), 0.5, dtype=np.float32)

    with pytest.raises(ValueError, match="Unsupported format"):
        tools.save_image_rgb(str(tmp_path / "render.bmp"), image, fmt="bmp")
