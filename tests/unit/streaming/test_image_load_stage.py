"""Unit coverage for ImageLoadStage contracts."""

from __future__ import annotations

import asyncio
import builtins
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from tests.unit.streaming._helpers import RecordingIOPool
from transformation_portal.streaming.stages import ImageLoadStage

pytestmark = pytest.mark.unit


class FakeImage:
    """Context-manager image stub that supports np.array(image)."""

    format = "JPEG"
    mode = "RGB"
    size = (6, 8)

    def __init__(self, *, exif: dict[int, Any] | BaseException | None = None) -> None:
        self._exif_fixture = exif
        self._array = np.zeros((8, 6, 3), dtype=np.uint8)

    def __enter__(self) -> "FakeImage":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def __array__(self, dtype=None) -> np.ndarray:
        return self._array.astype(dtype) if dtype is not None else self._array

    def _getexif(self):
        if isinstance(self._exif_fixture, BaseException):
            raise self._exif_fixture
        return self._exif_fixture


def test_image_load_stage_loads_png_metadata(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    source = np.zeros((8, 6, 3), dtype=np.uint8)
    source[..., 0] = 255
    Image.fromarray(source, mode="RGB").save(image_path)

    image_data = ImageLoadStage(load_exif=False)._load_sync(image_path)

    assert image_data.path == image_path
    assert image_data.array.shape == (8, 6, 3)
    assert image_data.array.dtype == np.uint8
    assert image_data.metadata["original_path"] == str(image_path)
    assert image_data.metadata["filename"] == "sample.png"
    assert image_data.metadata["format"] == "PNG"
    assert image_data.metadata["mode"] == "RGB"
    assert image_data.metadata["size"] == (6, 8)
    assert image_data.metadata["loaded_with"] == "PIL"
    assert image_data.metadata["dtype"] == "uint8"
    assert image_data.metadata["shape"] == (8, 6, 3)
    assert image_data.metadata["memory_mb"] > 0


def test_image_load_stage_tiff_falls_back_to_pil_when_tifffile_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "sample.tiff"
    source = np.full((8, 6), 120, dtype=np.uint8)
    Image.fromarray(source, mode="L").save(image_path, format="TIFF")
    original_import = builtins.__import__

    def blocked_tifffile_import(name: str, *args: Any, **kwargs: Any):
        if name == "tifffile":
            raise ImportError("tifffile intentionally unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_tifffile_import)

    image_data = ImageLoadStage(load_exif=False)._load_sync(image_path)

    assert image_data.array.shape == (8, 6)
    assert image_data.metadata["format"] == "TIFF"
    assert image_data.metadata["loaded_with"] == "PIL"
    assert image_data.metadata["dtype"] == "uint8"


def test_image_load_stage_uses_tifffile_for_tiff_when_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "sample.tif"
    image_path.write_bytes(b"placeholder")
    fake_tifffile = types.SimpleNamespace(
        imread=lambda path: np.full((8, 6), 42, dtype=np.uint8),
    )
    monkeypatch.setitem(sys.modules, "tifffile", fake_tifffile)

    image_data = ImageLoadStage(load_exif=False)._load_sync(image_path)

    assert image_data.array.shape == (8, 6)
    assert image_data.metadata["format"] == "TIFF"
    assert image_data.metadata["loaded_with"] == "tifffile"
    assert image_data.metadata["dtype"] == "uint8"


def test_image_load_stage_converts_16bit_images_to_float32(tmp_path: Path) -> None:
    image_path = tmp_path / "depth.png"
    source = np.linspace(0, 65535, 8 * 6, dtype=np.uint16).reshape(8, 6)
    Image.fromarray(source, mode="I;16").save(image_path)

    image_data = ImageLoadStage(load_exif=False, convert_16bit=True)._load_sync(image_path)

    assert image_data.array.shape == (8, 6)
    assert image_data.array.dtype == np.float32
    assert 0.0 <= float(image_data.array.min()) <= float(image_data.array.max()) <= 1.0
    assert image_data.metadata["converted_from"] == "uint16"
    assert image_data.metadata["dtype"] == "float32"


def test_image_load_stage_can_preserve_16bit_dtype(tmp_path: Path) -> None:
    image_path = tmp_path / "depth.png"
    source = np.full((8, 6), 1024, dtype=np.uint16)
    Image.fromarray(source, mode="I;16").save(image_path)

    image_data = ImageLoadStage(load_exif=False, convert_16bit=False)._load_sync(image_path)

    assert image_data.array.dtype == np.uint16
    assert image_data.metadata["dtype"] == "uint16"
    assert "converted_from" not in image_data.metadata


def test_image_load_stage_ignores_exif_extraction_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"placeholder")
    monkeypatch.setattr(Image, "open", lambda path: FakeImage(exif=OSError("bad exif block")))

    image_data = ImageLoadStage(load_exif=True)._load_sync(image_path)

    assert image_data.metadata["format"] == "JPEG"
    assert "exif" not in image_data.metadata


def test_image_load_stage_records_supported_exif_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"placeholder")
    monkeypatch.setattr(
        Image,
        "open",
        lambda path: FakeImage(exif={1: "camera", 2: 42, 3: object(), 4: b"raw"}),
    )

    image_data = ImageLoadStage(load_exif=True)._load_sync(image_path)

    assert image_data.metadata["exif"] == {1: "camera", 2: 42, 4: b"raw"}


def test_image_load_stage_process_uses_injected_worker_pool(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.fromarray(np.zeros((8, 6, 3), dtype=np.uint8), mode="RGB").save(image_path)
    pool = RecordingIOPool()
    stage = ImageLoadStage(load_exif=False, worker_pool=pool)  # type: ignore[arg-type]

    image_data = asyncio.run(stage.process(image_path))

    assert pool.calls == [image_path]
    assert image_data.path == image_path
    assert image_data.array.shape == (8, 6, 3)


def test_image_load_stage_startup_shutdown_manage_owned_worker_pool() -> None:
    def marker() -> str:
        return "ready"

    async def runner() -> None:
        stage = ImageLoadStage(max_concurrent=1)
        assert stage._worker_pool is None
        await stage.startup()
        worker_pool = stage._worker_pool
        assert worker_pool is not None
        assert await worker_pool.run_io(marker) == "ready"
        await stage.shutdown()
        with pytest.raises(RuntimeError, match="WorkerPool not active"):
            await worker_pool.run_io(marker)

    asyncio.run(runner())


def test_image_load_stage_process_uses_default_executor_without_worker_pool(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.fromarray(np.zeros((8, 6, 3), dtype=np.uint8), mode="RGB").save(image_path)
    stage = ImageLoadStage(load_exif=False)

    image_data = asyncio.run(stage.process(image_path))

    assert image_data.path == image_path
    assert image_data.metadata["loaded_with"] == "PIL"
