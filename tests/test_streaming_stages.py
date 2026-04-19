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

import transformation_portal.depth.models as depth_models
from transformation_portal.streaming.stages import DepthEstimationStage, ImageData, ImageLoadStage, ImageSaveStage


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


def test_depth_stage_missing_runtime_fails_closed_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_import_error(**_kwargs):
        raise ImportError("missing depth runtime")

    monkeypatch.setattr(depth_models, "load_depth_model", _raise_import_error)

    async def runner() -> None:
        stage = DepthEstimationStage()
        with pytest.raises(DepthEstimationStage.DepthBackendUnavailableError, match="Depth backend unavailable"):
            await stage.startup()

    asyncio.run(runner())


def test_depth_stage_allows_synthetic_only_with_explicit_opt_in(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _raise_import_error(**_kwargs):
        raise ImportError("missing depth runtime")

    monkeypatch.setattr(depth_models, "load_depth_model", _raise_import_error)

    image_data = ImageData(
        array=np.ones((4, 4, 3), dtype=np.float32),
        path=tmp_path / "frame.png",
    )

    async def runner() -> None:
        stage = DepthEstimationStage(allow_synthetic_depth=True)
        await stage.startup()
        try:
            result = await stage(image_data)
        finally:
            await stage.shutdown()

        assert result.success
        assert result.data is not None
        assert result.data.metadata["synthetic_output"] is True
        assert result.data.metadata["depth_capability"]["executed_backend"] == "synthetic"
        assert result.data.metadata["depth_capability"]["availability_state"] == "synthetic_opt_in"

    asyncio.run(runner())


def test_depth_stage_invalid_model_size_propagates_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_value_error(**_kwargs):
        raise ValueError("Unsupported depth model_size 'giant'. Expected one of: small, base, large.")

    monkeypatch.setattr(depth_models, "load_depth_model", _raise_value_error)

    async def runner() -> None:
        stage = DepthEstimationStage(model_size="giant", allow_synthetic_depth=True)
        with pytest.raises(ValueError, match="Unsupported depth model_size 'giant'"):
            await stage.startup()

    asyncio.run(runner())


def test_depth_stage_uses_estimate_depth_contract_when_real_model_present(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = {"estimate_depth": 0}

    class FakeDepthModel:
        model_revision = "rev-123"

        class variant:
            value = "depth-anything/Depth-Anything-V2-Small-hf"

        def estimate_depth(self, image: np.ndarray) -> dict:
            calls["estimate_depth"] += 1
            return {
                "depth": np.full(image.shape[:2], 0.5, dtype=np.float32),
                "depth_raw": np.full(image.shape[:2], 12.0, dtype=np.float32),
                "metadata": {"backend": "pytorch_cpu"},
            }

    monkeypatch.setattr(depth_models, "load_depth_model", lambda **_kwargs: FakeDepthModel())

    image_data = ImageData(
        array=np.ones((4, 4, 3), dtype=np.float32),
        path=tmp_path / "frame.png",
    )

    async def runner() -> None:
        stage = DepthEstimationStage(cache_model=False)
        await stage.startup()
        try:
            result = await stage(image_data)
        finally:
            await stage.shutdown()

        assert result.success
        assert calls["estimate_depth"] == 1
        assert np.allclose(result.data.depth_map, 0.5)
        assert result.data.metadata["depth_capability"]["executed_backend"] == "pytorch_cpu"
        assert result.data.metadata["depth_capability"]["model_repo_id"] == "depth-anything/Depth-Anything-V2-Small-hf"

    asyncio.run(runner())
