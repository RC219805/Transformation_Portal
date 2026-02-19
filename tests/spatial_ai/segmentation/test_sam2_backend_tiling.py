"""Tiling routing tests that do not require torch."""

from unittest.mock import Mock, patch

import numpy as np

from transformation_portal.spatial_ai.segmentation.contracts import (
    MaskMetadata,
    SegmentationInput,
    SegmentationResult,
)
from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend
from transformation_portal.spatial_ai.segmentation.tiling.config import SegmentationTilingConfig
from transformation_portal.spatial_ai.segmentation.tiling.engine import TiledSegmentationEngine
from transformation_portal.spatial_ai.segmentation.tiling.types import (
    BBox,
    SoftMaskPatch,
    TileInstance,
    TileManifest,
    TileSpec,
)


def test_segment_routes_to_tiled_engine_when_enabled(tmp_path):
    checkpoint = tmp_path / "sam2_test.pt"
    checkpoint.touch()
    backend = SAM2Backend(
        model_size="base",
        checkpoint_path=str(checkpoint),
        device="cpu",
        tiling=SegmentationTilingConfig(enabled=True),
    )
    fake_result = SegmentationResult(
        masks=np.ones((1, 16, 16), dtype=bool),
        scores=np.array([0.9], dtype=np.float32),
        metadata=[MaskMetadata(area=256, bbox=(0, 0, 16, 16), stability_score=0.9)],
    )
    mock_engine = Mock()
    mock_engine.run.return_value = fake_result
    backend.tiled_engine = mock_engine

    seg_input = SegmentationInput(image=np.random.rand(16, 16, 3).astype(np.float32), gamma=1.0, mode="auto")

    with patch.object(backend, "_load_model"):
        result = backend.segment(seg_input)

    assert result is fake_result
    mock_engine.run.assert_called_once()


def test_video_mode_does_not_route_to_tiling(tmp_path):
    checkpoint = tmp_path / "sam2_test.pt"
    checkpoint.touch()
    backend = SAM2Backend(
        model_size="base",
        checkpoint_path=str(checkpoint),
        device="cpu",
        tiling=SegmentationTilingConfig(enabled=True),
    )
    video_dir = tmp_path / "frames"
    video_dir.mkdir()
    fake_result = SegmentationResult(
        masks=np.ones((1, 8, 8), dtype=bool),
        scores=np.array([1.0], dtype=np.float32),
        metadata=[MaskMetadata(area=64, bbox=(0, 0, 8, 8), stability_score=1.0)],
        temporal_ids=np.array([1], dtype=np.int32),
    )
    seg_input = SegmentationInput(
        image=None,
        gamma=1.0,
        mode="video",
        video_path=str(video_dir),
        prompts={"frame_idx": 0, "object_id": 1, "points": [[1, 1]], "labels": [1]},
    )

    with patch.object(backend, "_load_model"), patch.object(
        backend, "_segment_video", return_value=fake_result
    ) as mock_video:
        result = backend.segment(seg_input)

    assert result is fake_result
    mock_video.assert_called_once_with(seg_input)


def test_tiling_bypassed_when_mode_not_in_apply_to_modes(tmp_path):
    checkpoint = tmp_path / "sam2_test.pt"
    checkpoint.touch()
    backend = SAM2Backend(
        model_size="base",
        checkpoint_path=str(checkpoint),
        device="cpu",
        tiling=SegmentationTilingConfig(enabled=True, apply_to_modes=("auto",)),
    )
    fake_result = SegmentationResult(
        masks=np.ones((1, 16, 16), dtype=bool),
        scores=np.array([0.8], dtype=np.float32),
        metadata=[MaskMetadata(area=256, bbox=(0, 0, 16, 16), stability_score=0.8)],
    )
    backend.tiled_engine = Mock()
    seg_input = SegmentationInput(
        image=np.random.rand(16, 16, 3).astype(np.float32),
        gamma=1.0,
        mode="points",
        prompts={"points": [[5, 5]], "labels": [1]},
    )
    with patch.object(backend, "_load_model"), patch.object(
        backend, "_segment_prompted", return_value=fake_result
    ) as mock_prompted:
        result = backend.segment(seg_input)
    assert result is fake_result
    mock_prompted.assert_called_once_with(seg_input)
    backend.tiled_engine.run.assert_not_called()


def test_tiling_zero_area_instances_returns_empty_result(tmp_path):
    checkpoint = tmp_path / "sam2_test.pt"
    checkpoint.touch()
    backend = SAM2Backend(model_size="base", checkpoint_path=str(checkpoint), device="cpu")
    engine = backend._build_default_tiled_engine()

    class _StubBackend:
        name = "stub"
        device = "cpu"

        def global_seed_pass(self, **kwargs):
            del kwargs
            return None

        def segment_tile(self, **kwargs):
            del kwargs
            return (
                TileInstance(
                    local_id="i0",
                    score=0.7,
                    stability_score=0.7,
                    soft_mask=SoftMaskPatch(
                        bbox=BBox(0, 0, 8, 8),
                        values=np.zeros((8, 8), dtype=np.float32),
                        space="prob",
                    ),
                ),
            )

    class _StubPlanner:
        def plan(self, *, image_hash, W, H, config, global_hints, prompts, mode):
            del config, global_hints, prompts, mode
            tile = TileSpec(tile_id="t0", bbox=BBox(0, 0, W, H), overlap_px=0, pad_mode="reflect")
            return TileManifest(
                image_hash=image_hash,
                W=W,
                H=H,
                tile_size_px=W,
                overlap_px=0,
                stride_px=W,
                policy="uniform",
                seed=0,
                tiles=(tile,),
            )

    engine = TiledSegmentationEngine(planner=_StubPlanner(), merger=engine.merger, validator=engine.validator)
    seg_input = SegmentationInput(image=np.random.rand(8, 8, 3).astype(np.float32), gamma=1.0, mode="auto")
    result = engine.run(
        backend=_StubBackend(),
        seg_input=seg_input,
        image_hash="abc",
        config=SegmentationTilingConfig(enabled=True),
    )
    assert result.masks.shape == (0, 8, 8)
    assert result.scores.shape == (0,)
    assert result.metadata == []


def test_stable_image_hash_is_deterministic_for_identical_input(tmp_path):
    checkpoint = tmp_path / "sam2_test.pt"
    checkpoint.touch()
    backend = SAM2Backend(model_size="base", checkpoint_path=str(checkpoint), device="cpu")
    image = np.random.rand(10, 12, 3).astype(np.float32)
    h1 = backend._stable_image_hash(image)
    h2 = backend._stable_image_hash(image.copy())
    assert h1 == h2


def test_translate_prompts_to_tile_clamps_bounds(tmp_path):
    checkpoint = tmp_path / "sam2_test.pt"
    checkpoint.touch()
    backend = SAM2Backend(model_size="base", checkpoint_path=str(checkpoint), device="cpu")
    tile = TileSpec(tile_id="t0", bbox=BBox(10, 20, 30, 40), overlap_px=0, pad_mode="reflect")

    points = backend._translate_prompts_to_tile({"points": [[5, 100]]}, tile, "points")
    assert points["points"][0] == [0.0, 19.0]

    bbox = backend._translate_prompts_to_tile({"bbox": [0, 0, 100, 100]}, tile, "bbox")
    assert bbox["bbox"] == [0.0, 0.0, 20.0, 20.0]


def test_unload_clears_backend_references(tmp_path):
    checkpoint = tmp_path / "sam2_test.pt"
    checkpoint.touch()
    backend = SAM2Backend(model_size="base", checkpoint_path=str(checkpoint), device="cpu")
    backend._model = object()
    backend._mask_generator = object()
    backend._image_predictor = object()
    backend._video_predictor = object()
    backend._material_classifier = object()
    backend.unload()
    assert backend._model is None
    assert backend._mask_generator is None
    assert backend._image_predictor is None
    assert backend._video_predictor is None
    assert backend._material_classifier is None
