"""Smoke coverage for streaming stage implementations."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Pytest markers
pytestmark = [
    pytest.mark.unit,
]

from transformation_portal.streaming.stages import ImageData, ImageLoadStage, ImageSaveStage


def test_image_load_stage_loads_rgb_image_and_metadata(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    source = np.array(
        [
            [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            [[12, 34, 56], [78, 90, 123], [200, 150, 100]],
        ],
        dtype=np.uint8,
    )
    Image.fromarray(source, mode="RGB").save(image_path)

    async def runner() -> None:
        stage = ImageLoadStage(load_exif=False)
        await stage.startup()
        try:
            result = await stage(image_path)
        finally:
            await stage.shutdown()

        assert result.success
        assert result.data is not None
        assert result.data.path == image_path
        assert result.data.array.shape == (2, 3, 3)
        assert result.data.metadata["filename"] == "sample.png"
        assert result.data.metadata["format"] == "PNG"
        assert result.data.metadata["mode"] == "RGB"
        assert result.data.metadata["loaded_with"] == "PIL"
        assert result.data.metadata["dtype"] == "uint8"
        assert result.data.metadata["shape"] == (2, 3, 3)

    asyncio.run(runner())


def test_image_save_stage_writes_expected_output_suffix(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    image_data = ImageData(
        array=np.full((2, 2, 3), 127, dtype=np.uint8),
        path=tmp_path / "frame.png",
    )

    async def runner() -> None:
        stage = ImageSaveStage(output_dir=output_dir, output_format="PNG", suffix="_smoke")
        await stage.startup()
        try:
            result = await stage(image_data)
        finally:
            await stage.shutdown()

        assert result.success
        assert result.data is not None

        output_path = Path(result.data.metadata["output_path"])
        assert output_path == output_dir / "frame_smoke.png"
        assert output_path.exists()

        with Image.open(output_path) as saved:
            assert saved.mode == "RGB"
            assert saved.size == (2, 2)

    asyncio.run(runner())
