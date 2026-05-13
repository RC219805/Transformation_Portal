"""Unit coverage for ImageSaveStage contracts."""

from __future__ import annotations

import asyncio
import builtins
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from tests.unit.streaming._helpers import RecordingIOPool
from transformation_portal.streaming.stages import ImageData, ImageLoadStage, ImageSaveStage, create_luxury_pipeline_stages

pytestmark = pytest.mark.unit


def test_image_save_stage_writes_float_tiff(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    image_array = np.arange(8 * 6 * 3, dtype=np.float32).reshape(8, 6, 3) / np.float32(8 * 6 * 3 - 1)
    image_data = ImageData(
        array=image_array,
        path=tmp_path / "frame.png",
    )
    stage = ImageSaveStage(output_dir=output_dir, output_format="TIFF", suffix="_float")

    saved = stage._save_sync(image_data)

    assert saved == output_dir / "frame_float.tiff"
    assert saved.exists()
    assert saved.stat().st_size > 0


def test_image_save_stage_tiff_falls_back_to_pil_when_tifffile_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    image_data = ImageData(
        array=np.full((8, 6, 3), 127, dtype=np.uint8),
        path=tmp_path / "frame.png",
    )
    stage = ImageSaveStage(output_dir=output_dir, output_format="TIFF", suffix="_fallback")
    original_import = builtins.__import__

    def blocked_tifffile_import(name: str, *args: Any, **kwargs: Any):
        if name == "tifffile":
            raise ImportError("tifffile intentionally unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_tifffile_import)

    saved = stage._save_sync(image_data)

    assert saved == output_dir / "frame_fallback.tiff"
    assert saved.exists()
    with Image.open(saved) as image:
        assert image.format == "TIFF"
        assert image.size == (6, 8)


def test_image_save_stage_writes_float_png_as_uint8(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    image_data = ImageData(
        array=np.full((8, 6, 3), 0.5, dtype=np.float32),
        path=tmp_path / "frame.tiff",
    )
    stage = ImageSaveStage(output_dir=output_dir, output_format="PNG", suffix="_preview")

    saved = stage._save_sync(image_data)

    assert saved == output_dir / "frame_preview.png"
    with Image.open(saved) as image:
        assert image.mode == "RGB"
        assert image.size == (6, 8)
        assert np.asarray(image).dtype == np.uint8


def test_image_save_stage_process_creates_output_dir_and_updates_metadata(tmp_path: Path) -> None:
    output_dir = tmp_path / "nested" / "output"
    image_data = ImageData(
        array=np.full((8, 6, 3), 0.25, dtype=np.float32),
        path=tmp_path / "frame.png",
    )

    async def runner() -> ImageData:
        stage = ImageSaveStage(output_dir=output_dir, output_format="PNG", suffix="_done")
        await stage.startup()
        try:
            return await stage.process(image_data)
        finally:
            await stage.shutdown()

    result = asyncio.run(runner())

    output_path = output_dir / "frame_done.png"
    assert output_dir.is_dir()
    assert output_path.exists()
    assert result.metadata["output_path"] == str(output_path)


def test_image_save_stage_process_uses_injected_worker_pool(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    pool = RecordingIOPool(record_result=True)
    image_data = ImageData(
        array=np.zeros((8, 6, 3), dtype=np.uint8),
        path=tmp_path / "frame.png",
    )
    stage = ImageSaveStage(
        output_dir=output_dir,
        output_format="PNG",
        suffix="_pool",
        worker_pool=pool,  # type: ignore[arg-type]
    )

    result = asyncio.run(stage.process(image_data))

    output_path = output_dir / "frame_pool.png"
    assert pool.calls == [output_path]
    assert result.metadata["output_path"] == str(output_path)
    assert output_path.exists()


def test_image_save_stage_process_uses_default_executor_without_worker_pool(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    image_data = ImageData(
        array=np.zeros((8, 6, 3), dtype=np.uint8),
        path=tmp_path / "frame.png",
    )
    stage = ImageSaveStage(output_dir=output_dir, output_format="PNG", suffix="_direct")

    result = asyncio.run(stage.process(image_data))

    output_path = output_dir / "frame_direct.png"
    assert result.metadata["output_path"] == str(output_path)
    assert output_path.exists()


def test_image_save_stage_defaults_unknown_format_to_tiff_extension(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    image_data = ImageData(
        array=np.zeros((8, 6, 3), dtype=np.uint8),
        path=tmp_path / "frame.png",
    )
    stage = ImageSaveStage(output_dir=output_dir, output_format="unknown", suffix="_fallback")

    saved = stage._save_sync(image_data)

    assert saved == output_dir / "frame_fallback.tiff"
    assert saved.exists()


def test_create_luxury_pipeline_stages_can_build_load_save_only_subset(tmp_path: Path) -> None:
    stages = create_luxury_pipeline_stages(
        output_dir=tmp_path / "out",
        enable_depth=False,
        enable_material=False,
        enable_color_grading=False,
    )

    assert [stage.name for stage in stages] == ["image_load", "image_save"]
    assert isinstance(stages[0], ImageLoadStage)
    assert isinstance(stages[1], ImageSaveStage)


def test_image_save_stage_writes_jpeg_with_quality(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    image_data = ImageData(
        array=np.full((8, 6, 3), 190, dtype=np.uint8),
        path=tmp_path / "frame.png",
    )
    stage = ImageSaveStage(output_dir=output_dir, output_format="JPEG", quality=80, suffix="_display")

    saved = stage._save_sync(image_data)

    assert saved == output_dir / "frame_display.jpg"
    with Image.open(saved) as image:
        assert image.format == "JPEG"
        assert image.size == (6, 8)
