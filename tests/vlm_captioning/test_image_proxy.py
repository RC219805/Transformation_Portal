from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from transformation_portal.vlm_captioning.image_proxy import build_vlm_image_proxy

pytestmark = pytest.mark.unit


def _write_tiff(path: Path, *, size: tuple[int, int] = (3200, 1600), value: int = 1024) -> None:
    image = Image.new("I;16", size)
    image.putdata([value] * (size[0] * size[1]))
    image.save(path)


def test_tiff_source_builds_png_rgb_proxy(tmp_path: Path) -> None:
    source = tmp_path / "pool_water_stone_001_master16 copy.tif"
    _write_tiff(source)

    proxy = build_vlm_image_proxy(source, tmp_path / "out")

    assert proxy.proxy_path.suffix == ".png"
    assert proxy.mode == "RGB"
    assert proxy.format == "png"
    assert proxy.width == 1600
    assert proxy.height == 800
    with Image.open(proxy.proxy_path) as image:
        assert image.mode == "RGB"
        assert image.size == (1600, 800)


def test_jpeg_source_builds_png_rgb_proxy(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    Image.new("RGB", (120, 60), (20, 80, 140)).save(source, format="JPEG")

    proxy = build_vlm_image_proxy(source, tmp_path / "out")

    assert proxy.proxy_path.suffix == ".png"
    assert proxy.width == 120
    assert proxy.height == 60
    assert proxy.mode == "RGB"


def test_proxy_filename_and_dimensions_are_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "source.tif"
    _write_tiff(source, size=(3000, 1000), value=2048)

    first = build_vlm_image_proxy(source, tmp_path / "out")
    second = build_vlm_image_proxy(source, tmp_path / "out")

    assert first.proxy_path == second.proxy_path
    assert first.proxy_sha256 == second.proxy_sha256
    assert (first.width, first.height) == (1600, 533)
    assert first.proxy_path.name.startswith("source_")
    assert first.proxy_path.name.endswith("_proxy.png")


def test_invalid_input_path_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="VLM source image not found"):
        build_vlm_image_proxy(tmp_path / "missing.tif", tmp_path / "out")


def test_output_hash_changes_when_input_changes(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (32, 32), (10, 20, 30)).save(source)
    first = build_vlm_image_proxy(source, tmp_path / "out")

    Image.new("RGB", (32, 32), (200, 80, 40)).save(source)
    second = build_vlm_image_proxy(source, tmp_path / "out")

    assert first.source_sha256 != second.source_sha256
    assert first.proxy_sha256 != second.proxy_sha256
    assert first.proxy_path != second.proxy_path
