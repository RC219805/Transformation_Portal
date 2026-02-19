"""Tiling routing tests that do not require torch."""

from unittest.mock import Mock, patch

import numpy as np

from transformation_portal.spatial_ai.segmentation.contracts import MaskMetadata, SegmentationInput, SegmentationResult
from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend
from transformation_portal.spatial_ai.segmentation.tiling.config import SegmentationTilingConfig


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
