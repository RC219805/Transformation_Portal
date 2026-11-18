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
from transformation_portal.streaming.stages import (
    ColorGradingStage,
    DepthEstimationStage,
    ImageData,
    ImageLoadStage,
    ImageSaveStage,
    MaterialResponseStage,
)


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


# ---------------------------------------------------------------------------
# ColorGradingStage .cube LUT application (H-03: channel order + trilinear lookup)
# ---------------------------------------------------------------------------


def _write_cube(path: Path, size: int, transfer) -> None:
    """Write a spec-conformant .cube file: red index fastest, then green, then blue."""
    lines = ['TITLE "test"', f"LUT_3D_SIZE {size}"]
    for b in range(size):
        for g in range(size):
            for r in range(size):
                out = transfer(r / (size - 1), g / (size - 1), b / (size - 1))
                lines.append(" ".join(f"{v:.6f}" for v in out))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _grade(stage: ColorGradingStage, array: np.ndarray, tmp_path: Path) -> np.ndarray:
    return stage._apply_lut_sync(ImageData(array=array.copy(), path=tmp_path / "frame.png")).array


_LUT_PIXELS = np.array([[[1.0, 0.25, 0.0], [0.0, 0.5, 1.0], [0.2, 0.4, 0.6]]], dtype=np.float32)


def test_color_grading_identity_lut_returns_input_without_posterising(tmp_path: Path) -> None:
    cube = tmp_path / "identity.cube"
    _write_cube(cube, 17, lambda r, g, b: (r, g, b))
    stage = ColorGradingStage(lut_path=cube, intensity=1.0)
    stage._load_lut()

    np.testing.assert_allclose(_grade(stage, _LUT_PIXELS, tmp_path), _LUT_PIXELS, atol=1e-5)

    ramp = np.repeat(np.linspace(0.0, 1.0, 1000, dtype=np.float32)[None, :, None], 3, axis=2)
    graded_ramp = _grade(stage, ramp, tmp_path)
    np.testing.assert_allclose(graded_ramp, ramp, atol=1e-5)
    assert len(np.unique(graded_ramp[..., 1])) == 1000


def test_color_grading_red_blue_swap_lut_swaps_red_and_blue(tmp_path: Path) -> None:
    cube = tmp_path / "swap_rb.cube"
    _write_cube(cube, 5, lambda r, g, b: (b, g, r))
    stage = ColorGradingStage(lut_path=cube, intensity=1.0)
    stage._load_lut()

    image = np.random.default_rng(3).random((8, 8, 3), dtype=np.float32)
    np.testing.assert_allclose(_grade(stage, image, tmp_path), image[..., ::-1], atol=1e-5)

    half = ColorGradingStage(lut_path=cube, intensity=0.25)
    half._load_lut()
    np.testing.assert_allclose(
        _grade(half, _LUT_PIXELS, tmp_path), 0.75 * _LUT_PIXELS + 0.25 * _LUT_PIXELS[..., ::-1], atol=1e-5
    )


def test_color_grading_stage_matches_4k_pipeline_lut_lookup(tmp_path: Path) -> None:
    from transformation_portal.pipelines.rendering_4k.stages import _apply_lut

    cube = tmp_path / "multilinear.cube"
    _write_cube(cube, 3, lambda r, g, b: (r * g, g * (1.0 - b), 0.25 + 0.5 * b * r))

    async def runner() -> np.ndarray:
        stage = ColorGradingStage(lut_path=cube, intensity=1.0)
        await stage.startup()
        try:
            result = await stage(ImageData(array=(_LUT_PIXELS * 255).astype(np.uint8), path=tmp_path / "frame.png"))
        finally:
            await stage.shutdown()
        assert result.success
        assert result.data.metadata["color_graded"] is True
        return result.data.array

    graded = asyncio.run(runner())
    expected = _apply_lut((_LUT_PIXELS * 255).astype(np.uint8).astype(np.float32) / 255.0, cube, strength=1.0)
    np.testing.assert_allclose(graded, expected, atol=1e-5)


@pytest.mark.parametrize("shape", [(48, 64), (64, 64)])
def test_material_response_stage_handles_2d_greyscale(shape: tuple, tmp_path: Path) -> None:
    """M-13: (H, W) greyscale input must not broadcast masks against the image width."""
    grey = np.random.default_rng(0).random(shape).astype(np.float32)
    depth = np.tile(np.linspace(0.0, 1.0, shape[1], dtype=np.float32), (shape[0], 1))

    stage = MaterialResponseStage()
    flat = stage._enhance_sync(ImageData(array=grey.copy(), path=tmp_path / "g.png", depth_map=depth))
    channel = stage._enhance_sync(ImageData(array=grey[..., None].copy(), path=tmp_path / "g.png", depth_map=depth))

    assert flat.array.shape == shape
    assert flat.metadata["material_enhanced"] is True
    assert np.isfinite(flat.array).all()
    assert flat.array.min() >= 0.0 and flat.array.max() <= 1.0
    # Same result as the equivalent single-channel (H, W, 1) image.
    np.testing.assert_allclose(flat.array, channel.array[..., 0], atol=1e-6)
