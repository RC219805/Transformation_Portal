"""Tiling routing tests that do not require torch."""

import weakref
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
    TileSegmentationResult,
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
    values: np.ndarray | None = None,
) -> TileInstance:
    if values is None:
        mask_values = np.ones((height, width), dtype=np.float32)
    else:
        mask_values = np.asarray(values, dtype=np.float32)
        if mask_values.shape != (height, width):
            raise ValueError(f"tile instance values must have shape {(height, width)}, got {mask_values.shape}")
    return TileInstance(
        local_id=f"{tile_id}:0",
        score=score,
        stability_score=stability_score,
        soft_mask=SoftMaskPatch(
            bbox=BBox(0, 0, width, height),
            values=mask_values,
            space="prob",
        ),
        material_label=label,
        material_confidence=confidence,
    )


def test_tile_instance_fixture_rejects_mismatched_mask_shape():
    values = np.ones((4, 8), dtype=np.float32)

    with pytest.raises(ValueError, match=r"shape \(8, 4\), got \(4, 8\)"):
        _tile_instance(tile_id="tile", width=4, height=8, values=values)


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


def test_tiling_merges_large_smooth_region_across_tile_overlap_when_global_iou_is_low():
    left = _tile_instance(tile_id="left", width=8, height=8, label="sky", confidence=0.9, score=0.8)
    right = _tile_instance(tile_id="right", width=8, height=8, label="sky", confidence=0.7, score=0.6)
    global_iou = (2 * 8) / ((8 * 8) + (8 * 8) - (2 * 8))

    assert global_iou < 0.35

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
    assert np.all(result.masks[0, :, 6:8])
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


@pytest.mark.parametrize("window", ["hann", "cosine", "linear"])
def test_tiling_window_modes_preserve_one_pixel_overlap_seams(window):
    left = _tile_instance(tile_id="left", width=4, height=8, label="sky", confidence=0.9)
    right = _tile_instance(tile_id="right", width=4, height=8, label="sky", confidence=0.9)

    result = _run_two_tile_merge(
        left_instance=left,
        right_instance=right,
        left_tile=TileSpec(tile_id="left", bbox=BBox(0, 0, 4, 8), overlap_px=1, pad_mode="reflect"),
        right_tile=TileSpec(tile_id="right", bbox=BBox(3, 0, 7, 8), overlap_px=1, pad_mode="reflect"),
        width=7,
        config=SegmentationTilingConfig(
            enabled=True,
            tile_size_px=4,
            overlap_px=1,
            merge=MergeConfig(
                window=window,
                instance_merge=InstanceMergeConfig(enabled=True, iou_threshold=0.35),
            ),
        ),
    )

    assert result.masks.shape == (1, 8, 7)
    assert int(result.masks[0].sum()) == 56
    assert np.all(result.masks[0, :, 3])
    assert result.metadata[0].bbox == (0, 0, 7, 8)


def test_tiling_does_not_merge_same_label_regions_without_overlap_continuity():
    left_values = np.zeros((8, 8), dtype=np.float32)
    left_values[:, :7] = 1.0
    right_values = np.zeros((8, 8), dtype=np.float32)
    right_values[:, 1:] = 1.0
    left = _tile_instance(tile_id="left", width=8, height=8, label="sky", confidence=0.9, values=left_values)
    right = _tile_instance(tile_id="right", width=8, height=8, label="sky", confidence=0.9, values=right_values)

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
    assert not np.any(result.masks[0] & result.masks[1])
    assert np.all(result.masks[0, :, 6])
    assert not np.any(result.masks[0, :, 7])
    assert not np.any(result.masks[1, :, 6])
    assert np.all(result.masks[1, :, 7])
    assert [item.bbox for item in result.metadata] == [(0, 0, 7, 8), (7, 0, 7, 8)]


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


class _GroundTruthCropBackend:
    """Report, per tile, the visible crop of full-image masks as full-tile patches (the SAM2 ``segment_tile`` shape)."""

    name = "stub"
    device = "cpu"

    def __init__(self, masks_for_tile, *, tight=False):
        self._masks_for_tile = masks_for_tile
        self._tight = tight

    def global_seed_pass(self, **kwargs):
        del kwargs
        return None

    def segment_tile(self, **kwargs):
        tile_spec = kwargs["tile_spec"]
        box = tile_spec.bbox
        instances = []
        for idx, full_mask in enumerate(self._masks_for_tile(tile_spec)):
            crop = full_mask[box.y0 : box.y1, box.x0 : box.x1]
            if not crop.any():
                continue
            patch_bbox = BBox(0, 0, box.w, box.h)
            if self._tight:
                rows, cols = np.nonzero(crop)
                patch_bbox = BBox(int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1)
                crop = crop[patch_bbox.y0 : patch_bbox.y1, patch_bbox.x0 : patch_bbox.x1]
            instances.append(
                TileInstance(
                    local_id=f"{tile_spec.tile_id}:{idx}",
                    score=0.9,
                    stability_score=0.9,
                    soft_mask=SoftMaskPatch(bbox=patch_bbox, values=crop.astype(np.float32), space="prob"),
                )
            )
        return tuple(instances)


def _run_default_engine_on_ground_truth(masks_for_tile, *, width, height, tile_size_px=200, overlap_px=50):
    config = SegmentationTilingConfig(enabled=True, tile_size_px=tile_size_px, overlap_px=overlap_px)
    config.global_pass.enabled = False
    engine = SAM2Backend.__new__(SAM2Backend)._build_default_tiled_engine()
    return engine.run(
        backend=_GroundTruthCropBackend(masks_for_tile),
        seg_input=SegmentationInput(image=np.zeros((height, width, 3), dtype=np.float32), gamma=1.0, mode="auto"),
        image_hash="abc",
        config=config,
    )


def _rect_mask(height, width, rows, cols):
    mask = np.zeros((height, width), dtype=bool)
    mask[rows[0] : rows[1], cols[0] : cols[1]] = True
    return mask


def test_tiling_keeps_nested_part_and_whole_as_separate_instances_across_seam():
    """A part inside the overlap band is 100 % contained in the other tile's whole mask; that is not agreement."""
    height, width = 60, 300  # tiles [0, 200) and [150, 300); overlap band [150, 200)
    sofa = _rect_mask(height, width, (10, 50), (20, 280))
    cushion = _rect_mask(height, width, (20, 40), (160, 190))

    result = _run_default_engine_on_ground_truth(lambda tile: [sofa, cushion], width=width, height=height)

    assert result.masks.shape == (2, height, width)
    assert np.array_equal(result.masks[0], sofa)
    assert np.array_equal(result.masks[1], cushion)
    assert [item.area for item in result.metadata] == [int(sofa.sum()), int(cushion.sum())]


def test_tiling_unions_same_tile_parts_before_blending_across_tiles():
    """One tile splits an object in two, the neighbour sees one mask: no confident pixel may be dropped."""
    height, width = 60, 300
    left_part = _rect_mask(height, width, (10, 50), (20, 170))
    right_part = _rect_mask(height, width, (10, 50), (170, 200))
    whole = left_part | right_part

    def masks_for_tile(tile):
        return [left_part, right_part] if tile.bbox.x0 == 0 else [whole]

    result = _run_default_engine_on_ground_truth(masks_for_tile, width=width, height=height)

    assert result.masks.shape == (1, height, width)
    assert np.array_equal(result.masks[0], whole)
    assert result.metadata[0].area == int(whole.sum())


def test_tiling_reproduces_ground_truth_when_all_tiles_agree():
    """Four tiles meeting in one corner, a whole with a nested part in the corner and an object seen by one tile."""
    size = 300  # 2 x 2 tiles of 200 px with a 50 px overlap; all four share [150, 200) x [150, 200)
    floor = _rect_mask(size, size, (20, 280), (20, 280))
    rug = _rect_mask(size, size, (165, 185), (165, 185))
    lamp = _rect_mask(size, size, (5, 15), (5, 15))

    result = _run_default_engine_on_ground_truth(lambda tile: [floor, rug, lamp], width=size, height=size)

    assert result.masks.shape == (3, size, size)
    assert np.array_equal(result.masks[0], floor)
    assert np.array_equal(result.masks[1], rug)
    assert np.array_equal(result.masks[2], lamp)


def _merge_ground_truth_directly(masks_for_tile, *, width, height, tight, tile_size_px=200, overlap_px=50):
    config = SegmentationTilingConfig(enabled=True, tile_size_px=tile_size_px, overlap_px=overlap_px)
    manifest = UniformTilingPlanner().plan(
        image_hash="abc", W=width, H=height, config=config, global_hints=None, prompts=None, mode="auto"
    )
    backend = _GroundTruthCropBackend(masks_for_tile, tight=tight)
    tile_results = [
        TileSegmentationResult(
            image_hash="abc",
            tile_id=tile.tile_id,
            tile_spec=tile,
            instances=backend.segment_tile(tile_spec=tile),
            runtime_s=0.0,
        )
        for tile in manifest.tiles
    ]
    return BinaryUnionTileMerger().merge(
        image_hash="abc",
        W=width,
        H=height,
        manifest=manifest,
        tile_results=tile_results,
        global_hints=None,
        merge_config=config.merge,
    )


def test_tiling_merge_does_not_depend_on_patch_crop():
    """Full-tile patches (what SAM2 emits) and tight-bbox patches of the same masks merge identically.

    Covers the band metrics (two tiles that see the same object shifted by five rows, plus a whole with a nested
    part) and the per-tile vote (a tile that splits an object in two).
    """
    height, width = 60, 300
    shifted_left = _rect_mask(height, width, (0, 15), (150, 200))
    shifted_right = _rect_mask(height, width, (5, 20), (150, 200))
    sofa = _rect_mask(height, width, (25, 58), (20, 280))
    cushion = _rect_mask(height, width, (30, 50), (160, 190))
    left_part = _rect_mask(height, width, (25, 58), (20, 170))
    right_part = _rect_mask(height, width, (25, 58), (170, 200))

    def masks_for_tile(tile):
        if tile.bbox.x0 == 0:
            return [shifted_left, sofa, cushion, left_part, right_part]
        return [shifted_right, sofa, cushion, left_part | right_part]

    full = _merge_ground_truth_directly(masks_for_tile, width=width, height=height, tight=False)
    tight = _merge_ground_truth_directly(masks_for_tile, width=width, height=height, tight=True)

    assert full[0].shape == tight[0].shape
    assert np.array_equal(full[0], tight[0])
    assert np.array_equal(full[1], tight[1])
    assert full[3]["seam_metrics"] == tight[3]["seam_metrics"]
    assert full[3]["seam_metrics"]["merged_pair_count"] > 0


def test_sam2_segment_tile_emits_tight_bool_patches_that_merge_like_full_tile_patches(tmp_path):
    """SAM2Backend.segment_tile keeps each instance's tight bbox only, and the merge result is unchanged."""
    height, width = 60, 300
    shifted_left = _rect_mask(height, width, (0, 15), (150, 200))
    shifted_right = _rect_mask(height, width, (5, 20), (150, 200))
    sofa = _rect_mask(height, width, (25, 58), (20, 280))
    cushion = _rect_mask(height, width, (30, 50), (160, 190))
    left_part = _rect_mask(height, width, (25, 58), (20, 170))
    right_part = _rect_mask(height, width, (25, 58), (170, 200))

    def masks_for_tile(tile):
        if tile.bbox.x0 == 0:
            return [shifted_left, sofa, cushion, left_part, right_part]
        return [shifted_right, sofa, cushion, left_part | right_part]

    checkpoint = tmp_path / "sam2_test.pt"
    checkpoint.touch()
    backend = SAM2Backend(model_size="base", checkpoint_path=str(checkpoint), device="cpu")
    config = SegmentationTilingConfig(enabled=True, tile_size_px=200, overlap_px=50)
    manifest = UniformTilingPlanner().plan(
        image_hash="abc", W=width, H=height, config=config, global_hints=None, prompts=None, mode="auto"
    )
    tile_results = []
    for tile in manifest.tiles:
        crops = [mask[tile.bbox.y0 : tile.bbox.y1, tile.bbox.x0 : tile.bbox.x1] for mask in masks_for_tile(tile)]
        crops = [crop for crop in crops if crop.any()]
        tile_seg = SegmentationResult(
            masks=np.stack(crops),
            scores=np.full((len(crops),), 0.9, dtype=np.float32),
            metadata=[MaskMetadata(area=int(crop.sum()), bbox=(0, 0, 1, 1), stability_score=0.9) for crop in crops],
        )
        with patch.object(backend, "_segment_auto", return_value=tile_seg):
            instances = backend.segment_tile(
                tile_linear=np.zeros((tile.bbox.h, tile.bbox.w, 3), dtype=np.float32),
                image_hash="abc",
                tile_spec=tile,
                mode="auto",
                prompts=None,
                global_hints=None,
                rng_seed=0,
            )
        for instance, crop in zip(instances, crops, strict=True):
            rows, cols = np.nonzero(crop)
            patch_bbox = instance.soft_mask.bbox
            assert (patch_bbox.x0, patch_bbox.y0, patch_bbox.x1, patch_bbox.y1) == (
                cols.min(),
                rows.min(),
                cols.max() + 1,
                rows.max() + 1,
            )
            assert instance.soft_mask.values.dtype == np.bool_
            assert instance.soft_mask.values.base is None  # owns its pixels, not a view of the tile's mask stack
            assert np.array_equal(
                instance.soft_mask.values, crop[patch_bbox.y0 : patch_bbox.y1, patch_bbox.x0 : patch_bbox.x1]
            )
        tile_results.append(
            TileSegmentationResult(image_hash="abc", tile_id=tile.tile_id, tile_spec=tile, instances=instances, runtime_s=0.0)
        )

    tight = BinaryUnionTileMerger().merge(
        image_hash="abc",
        W=width,
        H=height,
        manifest=manifest,
        tile_results=tile_results,
        global_hints=None,
        merge_config=config.merge,
    )
    full = _merge_ground_truth_directly(masks_for_tile, width=width, height=height, tight=False)

    assert np.array_equal(tight[0], full[0])
    assert np.array_equal(tight[1], full[1])
    assert tight[2] == full[2]
    assert tight[3] == full[3]


_SEAM_HEIGHT, _SEAM_WIDTH = 60, 700  # tiles of 400 px; with a 50 px overlap: [0, 400) and [350, 700)
_SEAM_OBJECT_COLS = (360, 390)  # one small object inside the overlap band of the first two tiles
_SEAM_OBJECT = _rect_mask(_SEAM_HEIGHT, _SEAM_WIDTH, (20, 40), _SEAM_OBJECT_COLS)


class _SeamDisagreementBackend:
    """The first tile and its neighbours place one small object 10 px apart; the two masks still share two thirds.

    The merger joins the pair (IoU 0.5) and the seam validator rejects it (discontinuity 1/3). With
    ``agree_from_overlap_px`` every tile reports the true object once the tiles overlap that much. The overlap of
    every tile the engine asks for is recorded.
    """

    name = "stub"
    device = "cpu"

    def __init__(self, *, agree_from_overlap_px=None):
        self._agree_from_overlap_px = agree_from_overlap_px
        self.overlaps_seen = []

    def global_seed_pass(self, **kwargs):
        del kwargs
        return None

    def segment_tile(self, **kwargs):
        tile_spec = kwargs["tile_spec"]
        self.overlaps_seen.append(tile_spec.overlap_px)
        box = tile_spec.bbox
        col0, col1 = _SEAM_OBJECT_COLS
        if self._agree_from_overlap_px is None or tile_spec.overlap_px < self._agree_from_overlap_px:
            shift = -5 if box.x0 == 0 else 5
            col0, col1 = col0 + shift, col1 + shift
        crop = _rect_mask(_SEAM_HEIGHT, _SEAM_WIDTH, (20, 40), (col0, col1))[box.y0 : box.y1, box.x0 : box.x1]
        if not crop.any():
            return ()
        return (
            TileInstance(
                local_id=f"{tile_spec.tile_id}:0",
                score=0.9,
                stability_score=0.9,
                soft_mask=SoftMaskPatch(bbox=BBox(0, 0, box.w, box.h), values=crop.astype(np.float32), space="prob"),
            ),
        )


def _seam_config(*, overlap_px=50, validation=None):
    config = SegmentationTilingConfig(
        enabled=True, tile_size_px=400, overlap_px=overlap_px, validation=validation or ValidationConfig()
    )
    config.global_pass.enabled = False
    return config


def _run_default_engine_on_seam_object(backend, config):
    engine = SAM2Backend.__new__(SAM2Backend)._build_default_tiled_engine()
    return engine.run(
        backend=backend,
        seg_input=SegmentationInput(image=np.zeros((_SEAM_HEIGHT, _SEAM_WIDTH, 3), dtype=np.float32), gamma=1.0, mode="auto"),
        image_hash="abc",
        config=config,
    )


def test_tiling_retries_with_doubled_overlap_when_seam_validation_fails():
    """A pair the merger joined but the validator rejects is re-tiled (max_retries=1, auto_increase_overlap)."""
    backend = _SeamDisagreementBackend(agree_from_overlap_px=100)
    config = _seam_config()

    result = _run_default_engine_on_seam_object(backend, config)

    # Two tiles at the configured 50 px, then [0, 400) and [300, 700) at 100 px.
    assert backend.overlaps_seen == [50, 50, 100, 100]
    assert result.masks.shape == (1, _SEAM_HEIGHT, _SEAM_WIDTH)
    assert np.array_equal(result.masks[0], _SEAM_OBJECT)
    assert config.overlap_px == 50


def test_tiling_releases_the_rejected_attempt_before_the_retry_segments():
    """The retry must not run while the rejected attempt's (N, H, W) mask stack is still referenced."""

    class _RecordingMerger(BinaryUnionTileMerger):
        def __init__(self):
            self.mask_stacks = []

        def merge(self, **kwargs):
            merged = super().merge(**kwargs)
            self.mask_stacks.append(weakref.ref(merged[0]))
            return merged

    class _RetryObservingBackend(_SeamDisagreementBackend):
        def __init__(self, merger):
            super().__init__(agree_from_overlap_px=100)
            self._merger = merger
            self.rejected_stack_alive_during_retry = []

        def segment_tile(self, **kwargs):
            if kwargs["tile_spec"].overlap_px == 100:
                self.rejected_stack_alive_during_retry.append(self._merger.mask_stacks[0]() is not None)
            return super().segment_tile(**kwargs)

    merger = _RecordingMerger()
    backend = _RetryObservingBackend(merger)
    engine = TiledSegmentationEngine(planner=UniformTilingPlanner(), merger=merger, validator=SeamMergeValidator())

    result = engine.run(
        backend=backend,
        seg_input=SegmentationInput(image=np.zeros((_SEAM_HEIGHT, _SEAM_WIDTH, 3), dtype=np.float32), gamma=1.0, mode="auto"),
        image_hash="abc",
        config=_seam_config(),
    )

    assert len(merger.mask_stacks) == 2
    assert backend.rejected_stack_alive_during_retry == [False, False]
    assert np.array_equal(result.masks[0], _SEAM_OBJECT)


def test_tiling_fails_closed_when_seam_validation_retries_are_exhausted():
    backend = _SeamDisagreementBackend()

    with pytest.raises(
        RuntimeError,
        match=r"SAM2 tiling validation failed after 1 retry \(overlap_px 50 -> 100\): .*0\.333 exceeds threshold 0\.250",
    ):
        _run_default_engine_on_seam_object(backend, _seam_config())

    assert backend.overlaps_seen == [50, 50, 100, 100]


def test_tiling_retry_overlap_stops_at_half_the_tile_size():
    backend = _SeamDisagreementBackend()

    with pytest.raises(RuntimeError, match=r"SAM2 tiling validation failed after 2 retries \(overlap_px 50 -> 200\): "):
        _run_default_engine_on_seam_object(backend, _seam_config(validation=ValidationConfig(max_retries=5)))

    assert backend.overlaps_seen == [50] * 2 + [100] * 2 + [200] * 3


@pytest.mark.parametrize(
    ("overlap_px", "validation", "tiles"),
    [
        (50, ValidationConfig(max_retries=0), 2),
        (50, ValidationConfig(auto_increase_overlap=False), 2),
        (200, ValidationConfig(), 3),
    ],
    ids=["no-retries-allowed", "overlap-increase-disabled", "overlap-already-half-the-tile"],
)
def test_tiling_does_not_repeat_an_identical_tiling_after_seam_validation_fails(overlap_px, validation, tiles):
    backend = _SeamDisagreementBackend()

    with pytest.raises(RuntimeError, match=r"SAM2 tiling validation failed: SAM2 tiling seam discontinuity"):
        _run_default_engine_on_seam_object(backend, _seam_config(overlap_px=overlap_px, validation=validation))

    assert backend.overlaps_seen == [overlap_px] * tiles


def test_tiling_returns_the_merged_pair_without_retry_when_seam_validation_is_disabled():
    backend = _SeamDisagreementBackend()

    result = _run_default_engine_on_seam_object(backend, _seam_config(validation=ValidationConfig(enabled=False)))

    assert backend.overlaps_seen == [50, 50]
    assert result.masks.shape == (1, _SEAM_HEIGHT, _SEAM_WIDTH)


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [
        (1536, 1536, [(0, 0, 1536, 1536)]),
        (1281, 1281, [(0, 0, 1281, 1281)]),
        (2816, 1536, [(0, 0, 1536, 1536), (1280, 0, 2816, 1536)]),
        (2817, 1300, [(0, 0, 1536, 1300), (1280, 0, 2816, 1300), (2560, 0, 2817, 1300)]),
    ],
    ids=["exactly-one-tile", "one-pixel-past-the-stride", "two-tiles-reach-the-edge", "edge-tile-still-needed"],
)
def test_planner_stops_at_the_first_tile_that_reaches_the_image_edge(width, height, expected):
    """No tile lies wholly inside an earlier one (the 1x1 px and 256 px slivers past the edge tile)."""
    config = SegmentationTilingConfig(enabled=True)  # 1536 px tiles, 256 px overlap, stride 1280
    manifest = UniformTilingPlanner().plan(
        image_hash="abc", W=width, H=height, config=config, global_hints=None, prompts=None, mode="auto"
    )

    assert [(t.bbox.x0, t.bbox.y0, t.bbox.x1, t.bbox.y1) for t in manifest.tiles] == expected
    assert [t.tile_id for t in manifest.tiles] == [f"tile_{i}" for i in range(len(expected))]
    covered = np.zeros((height, width), dtype=bool)
    for tile in manifest.tiles:
        covered[tile.bbox.y0 : tile.bbox.y1, tile.bbox.x0 : tile.bbox.x1] = True
    assert covered.all()


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
