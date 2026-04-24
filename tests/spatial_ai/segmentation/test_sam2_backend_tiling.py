"""Tiling routing tests that do not require torch."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from transformation_portal.spatial_ai.segmentation.contracts import MaskMetadata, SegmentationInput, SegmentationResult
from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend
from transformation_portal.spatial_ai.segmentation.tiling.config import (
    InstanceMergeConfig,
    MergeConfig,
    SegmentationTilingConfig,
    ValidationConfig,
)
from transformation_portal.spatial_ai.segmentation.tiling.engine import TiledSegmentationEngine
from transformation_portal.spatial_ai.segmentation.tiling.merger import BinaryUnionTileMerger
from transformation_portal.spatial_ai.segmentation.tiling.planner import UniformTilingPlanner
from transformation_portal.spatial_ai.segmentation.tiling.types import (
    BBox,
    SoftMaskPatch,
    TileInstance,
    TileManifest,
    TileSpec,
)
from transformation_portal.spatial_ai.segmentation.tiling.validator import SeamMergeValidator

pytestmark = pytest.mark.unit


class _TwoTilePlanner:
    def __init__(self, left: TileSpec, right: TileSpec):
        self.left = left
        self.right = right

    def plan(self, *, image_hash, W, H, config, global_hints, prompts, mode):
        del config, global_hints, prompts, mode
        return TileManifest(
            image_hash=image_hash,
            W=W,
            H=H,
            tile_size_px=8,
            overlap_px=2,
            stride_px=6,
            policy="uniform",
            seed=0,
            tiles=(self.left, self.right),
        )


class _TileInstanceBackend:
    name = "stub"
    device = "cpu"

    def __init__(self, instances_by_tile):
        self.instances_by_tile = instances_by_tile

    def global_seed_pass(self, **kwargs):
        del kwargs
        return None

    def segment_tile(self, **kwargs):
        tile_spec = kwargs["tile_spec"]
        return tuple(self.instances_by_tile[tile_spec.tile_id])


def _tile_instance(
    *,
    tile_id: str,
    width: int,
    height: int,
    label: str | None = None,
    confidence: float | None = None,
    score: float = 0.8,
    stability_score: float = 0.9,
) -> TileInstance:
    return TileInstance(
        local_id=f"{tile_id}:0",
        score=score,
        stability_score=stability_score,
        soft_mask=SoftMaskPatch(
            bbox=BBox(0, 0, width, height),
            values=np.ones((height, width), dtype=np.float32),
            space="prob",
        ),
        material_label=label,
        material_confidence=confidence,
    )


def _run_two_tile_merge(*, left_instance, right_instance, config, left_tile=None, right_tile=None, width=14):
    backend = SAM2Backend.__new__(SAM2Backend)
    engine = backend._build_default_tiled_engine()
    if left_tile is None:
        left_tile = TileSpec(tile_id="left", bbox=BBox(0, 0, 8, 8), overlap_px=2, pad_mode="reflect")
    if right_tile is None:
        right_tile = TileSpec(tile_id="right", bbox=BBox(6, 0, 14, 8), overlap_px=2, pad_mode="reflect")
    engine = TiledSegmentationEngine(
        planner=_TwoTilePlanner(left_tile, right_tile),
        merger=engine.merger,
        validator=engine.validator,
    )
    seg_input = SegmentationInput(image=np.zeros((8, width, 3), dtype=np.float32), gamma=1.0, mode="auto")
    return engine.run(
        backend=_TileInstanceBackend({"left": [left_instance], "right": [right_instance]}),
        seg_input=seg_input,
        image_hash="abc",
        config=config,
    )


def _run_single_tile_merge(*, instances, config):
    backend = SAM2Backend.__new__(SAM2Backend)
    engine = backend._build_default_tiled_engine()
    tile = TileSpec(tile_id="tile", bbox=BBox(0, 0, 8, 8), overlap_px=0, pad_mode="reflect")

    class _SingleTilePlanner:
        def plan(self, *, image_hash, W, H, config, global_hints, prompts, mode):
            del config, global_hints, prompts, mode
            return TileManifest(
                image_hash=image_hash,
                W=W,
                H=H,
                tile_size_px=8,
                overlap_px=0,
                stride_px=8,
                policy="uniform",
                seed=0,
                tiles=(tile,),
            )

    engine = TiledSegmentationEngine(
        planner=_SingleTilePlanner(),
        merger=engine.merger,
        validator=engine.validator,
    )
    seg_input = SegmentationInput(image=np.zeros((8, 8, 3), dtype=np.float32), gamma=1.0, mode="auto")
    return engine.run(
        backend=_TileInstanceBackend({"tile": instances}),
        seg_input=seg_input,
        image_hash="abc",
        config=config,
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

    with patch.object(backend, "_load_model"), patch.object(backend, "_segment_video", return_value=fake_result) as mock_video:
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
    with (
        patch.object(backend, "_load_model"),
        patch.object(backend, "_segment_prompted", return_value=fake_result) as mock_prompted,
    ):
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


def test_default_tiled_engine_uses_extracted_components():
    backend = SAM2Backend.__new__(SAM2Backend)
    engine = backend._build_default_tiled_engine()

    assert isinstance(engine.planner, UniformTilingPlanner)
    assert isinstance(engine.merger, BinaryUnionTileMerger)
    assert isinstance(engine.validator, SeamMergeValidator)


def test_tiling_merges_large_smooth_region_across_tile_overlap():
    left = _tile_instance(tile_id="left", width=8, height=8, label="sky", confidence=0.9, score=0.8)
    right = _tile_instance(tile_id="right", width=8, height=8, label="sky", confidence=0.7, score=0.6)

    result = _run_two_tile_merge(
        left_instance=left,
        right_instance=right,
        config=SegmentationTilingConfig(
            enabled=True,
            tile_size_px=8,
            overlap_px=2,
            merge=MergeConfig(instance_merge=InstanceMergeConfig(enabled=True, iou_threshold=0.35)),
        ),
    )

    assert result.masks.shape == (1, 8, 14)
    assert int(result.masks[0].sum()) == 112
    assert result.metadata[0].bbox == (0, 0, 14, 8)
    assert result.metadata[0].material_label == "sky"
    assert result.metadata[0].material_confidence == pytest.approx(0.8)
    assert result.scores[0] == pytest.approx(0.7)


@pytest.mark.parametrize("window", ["hann", "cosine", "linear"])
def test_tiling_window_modes_preserve_smooth_binary_union(window):
    left = _tile_instance(tile_id="left", width=8, height=8, label="sky", confidence=0.9)
    right = _tile_instance(tile_id="right", width=8, height=8, label="sky", confidence=0.9)

    result = _run_two_tile_merge(
        left_instance=left,
        right_instance=right,
        config=SegmentationTilingConfig(
            enabled=True,
            tile_size_px=8,
            overlap_px=2,
            merge=MergeConfig(
                window=window,
                instance_merge=InstanceMergeConfig(enabled=True, iou_threshold=0.35),
            ),
        ),
    )

    assert result.masks.shape == (1, 8, 14)
    assert int(result.masks[0].sum()) == 112
    assert result.metadata[0].bbox == (0, 0, 14, 8)


def test_tiling_merge_avoids_full_image_bool_masks_per_candidate(monkeypatch):
    import transformation_portal.spatial_ai.segmentation.tiling.merger as merger_module

    class _NumpyProxy:
        def __init__(self, wrapped):
            self._wrapped = wrapped
            self.full_image_bool_zeros = 0

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

        def zeros(self, shape, *args, **kwargs):
            dtype = kwargs.get("dtype", args[0] if args else None)
            if shape == (8, 14) and dtype is bool:
                self.full_image_bool_zeros += 1
            return self._wrapped.zeros(shape, *args, **kwargs)

    proxy = _NumpyProxy(np)
    monkeypatch.setattr(merger_module, "np", proxy)

    left = _tile_instance(tile_id="left", width=8, height=8, label="sky")
    right = _tile_instance(tile_id="right", width=8, height=8, label="sky")

    result = _run_two_tile_merge(
        left_instance=left,
        right_instance=right,
        config=SegmentationTilingConfig(
            enabled=True,
            tile_size_px=8,
            overlap_px=2,
            merge=MergeConfig(instance_merge=InstanceMergeConfig(enabled=True, iou_threshold=0.35)),
        ),
    )

    assert result.masks.shape == (1, 8, 14)
    assert proxy.full_image_bool_zeros == 1


def test_tiling_parallel_matches_serial_output_ordering():
    class _PerTileBackend:
        name = "stub"
        device = "cpu"

        def global_seed_pass(self, **kwargs):
            del kwargs
            return None

        def segment_tile(self, **kwargs):
            tile_spec = kwargs["tile_spec"]
            tile_idx = int(tile_spec.tile_id.split("_")[1])
            return (
                TileInstance(
                    local_id=f"{tile_spec.tile_id}:0",
                    score=0.5 + tile_idx * 0.01,
                    stability_score=0.9,
                    soft_mask=SoftMaskPatch(
                        bbox=BBox(0, 0, tile_spec.bbox.w, tile_spec.bbox.h),
                        values=np.ones((tile_spec.bbox.h, tile_spec.bbox.w), dtype=np.float32),
                        space="prob",
                    ),
                ),
            )

    def run(max_concurrency):
        engine = TiledSegmentationEngine(
            planner=UniformTilingPlanner(),
            merger=BinaryUnionTileMerger(),
            validator=SeamMergeValidator(),
        )
        return engine.run(
            backend=_PerTileBackend(),
            seg_input=SegmentationInput(image=np.zeros((4, 4, 3), dtype=np.float32), gamma=1.0, mode="auto"),
            image_hash="abc",
            config=SegmentationTilingConfig(
                enabled=True,
                tile_size_px=2,
                overlap_px=0,
                max_concurrency=max_concurrency,
                merge=MergeConfig(instance_merge=InstanceMergeConfig(enabled=False)),
            ),
        )

    serial = run(1)
    parallel = run(3)

    assert parallel.scores.tolist() == serial.scores.tolist()
    assert [item.bbox for item in parallel.metadata] == [item.bbox for item in serial.metadata]
    assert np.array_equal(parallel.masks, serial.masks)


def test_tiling_validator_rejects_excessive_seam_discontinuity():
    manifest = TileManifest(
        image_hash="abc",
        W=8,
        H=8,
        tile_size_px=8,
        overlap_px=2,
        stride_px=6,
        policy="uniform",
        seed=0,
        tiles=(),
    )
    stats = {
        "seam_metrics": {
            "merged_pair_count": 1,
            "max_merged_discontinuity": 0.5,
            "mean_merged_discontinuity": 0.5,
        },
        "warnings": [],
    }
    ok, details = SeamMergeValidator().validate(
        manifest=manifest,
        merge_stats=stats,
        config=SegmentationTilingConfig(
            enabled=True,
            validation=ValidationConfig(enabled=True, seam_discontinuity_threshold=0.25),
        ),
    )

    assert ok is False
    assert details["ok"] is False
    assert "exceeds threshold" in stats["warnings"][0]


def test_tiling_does_not_merge_conflicting_material_labels():
    left = _tile_instance(tile_id="left", width=8, height=8, label="sky", confidence=0.9)
    right = _tile_instance(tile_id="right", width=8, height=8, label="water", confidence=0.8)

    result = _run_two_tile_merge(
        left_instance=left,
        right_instance=right,
        config=SegmentationTilingConfig(
            enabled=True,
            tile_size_px=8,
            overlap_px=2,
            merge=MergeConfig(instance_merge=InstanceMergeConfig(enabled=True, iou_threshold=0.35)),
        ),
    )

    assert result.masks.shape == (2, 8, 14)
    assert [item.material_label for item in result.metadata] == ["sky", "water"]


def test_tiling_preserves_instances_when_instance_merge_disabled():
    left = _tile_instance(tile_id="left", width=8, height=8, label="sky")
    right = _tile_instance(tile_id="right", width=8, height=8, label="sky")

    result = _run_two_tile_merge(
        left_instance=left,
        right_instance=right,
        config=SegmentationTilingConfig(
            enabled=True,
            tile_size_px=8,
            overlap_px=2,
            merge=MergeConfig(instance_merge=InstanceMergeConfig(enabled=False, iou_threshold=0.35)),
        ),
    )

    assert result.masks.shape == (2, 8, 14)
    assert [item.area for item in result.metadata] == [64, 64]


def test_tiling_does_not_merge_same_tile_instances():
    first = _tile_instance(tile_id="tile", width=8, height=8, label="sky")
    second = _tile_instance(tile_id="tile", width=8, height=8, label="sky", score=0.6)

    result = _run_single_tile_merge(
        instances=[first, second],
        config=SegmentationTilingConfig(
            enabled=True,
            tile_size_px=8,
            overlap_px=0,
            merge=MergeConfig(instance_merge=InstanceMergeConfig(enabled=True, iou_threshold=0.35)),
        ),
    )

    assert result.masks.shape == (2, 8, 8)
    assert [item.area for item in result.metadata] == [64, 64]


def test_tiling_skips_non_overlapping_tile_pairs():
    left = _tile_instance(tile_id="left", width=8, height=8, label="sky")
    right = _tile_instance(tile_id="right", width=8, height=8, label="sky")

    result = _run_two_tile_merge(
        left_instance=left,
        right_instance=right,
        left_tile=TileSpec(tile_id="left", bbox=BBox(0, 0, 8, 8), overlap_px=0, pad_mode="reflect"),
        right_tile=TileSpec(tile_id="right", bbox=BBox(8, 0, 16, 8), overlap_px=0, pad_mode="reflect"),
        width=16,
        config=SegmentationTilingConfig(
            enabled=True,
            tile_size_px=8,
            overlap_px=0,
            merge=MergeConfig(instance_merge=InstanceMergeConfig(enabled=True, iou_threshold=0.0)),
        ),
    )

    assert result.masks.shape == (2, 8, 16)
    assert [item.area for item in result.metadata] == [64, 64]


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
