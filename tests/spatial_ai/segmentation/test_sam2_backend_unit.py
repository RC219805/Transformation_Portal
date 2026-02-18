"""Unit tests for SAM2 backend with mocked dependencies (Phase 4B).

These tests mock ML components for fast execution without model loading.

NOTE: Mocking strategy needs refinement. Current issues:
- Patching at sam2 package level works for imports
- But torch.load still tries to load empty checkpoint file
- Need to also mock torch.load or use better fixture strategy
- For now, rely on integration tests for coverage

TODO: Fix mocking to work properly without real checkpoints.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from transformation_portal.spatial_ai.segmentation.contracts import MaskMetadata, SegmentationInput, SegmentationResult

# Module-level availability check
try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed (optional dependency)")


@pytest.fixture
def mock_checkpoint(tmp_path):
    """Create a fake checkpoint file."""
    checkpoint = tmp_path / "sam2_test.pt"
    checkpoint.touch()
    return checkpoint


class TestSAM2BackendInit:
    """Test SAM2Backend initialization."""

    def test_init_base_model(self, mock_checkpoint):
        """Test initialization with base model."""
        from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

        backend = SAM2Backend(model_size="base", checkpoint_path=str(mock_checkpoint), device="cpu")

        assert backend.model_size == "base"
        assert backend.device == "cpu"
        assert backend.checkpoint_path == mock_checkpoint

    def test_init_large_model(self, mock_checkpoint):
        """Test initialization with large model."""
        from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

        backend = SAM2Backend(model_size="large", checkpoint_path=str(mock_checkpoint), device="cpu")

        assert backend.model_size == "large"

    def test_init_invalid_model_size(self, mock_checkpoint):
        """Test invalid model size raises ValueError."""
        from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

        with pytest.raises(ValueError, match="Invalid model_size"):
            SAM2Backend(model_size="invalid", checkpoint_path=str(mock_checkpoint), device="cpu")

    def test_init_missing_checkpoint(self):
        """Test missing checkpoint file raises FileNotFoundError."""
        from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

        with pytest.raises(FileNotFoundError, match="checkpoint not found"):
            SAM2Backend(model_size="base", checkpoint_path="/nonexistent/checkpoint.pt", device="cpu")

    def test_init_default_checkpoint_path(self, monkeypatch):
        """Test default checkpoint path resolution."""
        from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

        # Mock Path.exists to avoid FileNotFoundError
        mock_exists = Mock(return_value=True)
        monkeypatch.setattr(Path, "exists", mock_exists)

        backend = SAM2Backend(model_size="large", device="cpu")

        assert backend.checkpoint_path.name == "sam2_hiera_large.pt"
        assert "checkpoints" in str(backend.checkpoint_path)


class TestSAM2BackendAutoMode:
    """Test auto mode with mocked SAM2AutomaticMaskGenerator."""

    @patch("sam2.build_sam.build_sam2")
    @patch("sam2.automatic_mask_generator.SAM2AutomaticMaskGenerator")
    def test_auto_mode_success(self, mock_mask_gen_cls, mock_build, mock_checkpoint):
        """Test auto mode returns correct result structure."""
        from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

        # Mock model
        mock_model = Mock()
        mock_build.return_value = mock_model

        # Mock mask generator
        mock_gen = Mock()
        fake_masks = [
            {
                "segmentation": np.ones((64, 64), dtype=bool),
                "bbox": [10, 10, 20, 20],
                "area": 400,
                "predicted_iou": 0.95,
                "stability_score": 0.92,
            },
            {
                "segmentation": np.zeros((64, 64), dtype=bool),
                "bbox": [30, 30, 15, 15],
                "area": 225,
                "predicted_iou": 0.88,
                "stability_score": 0.87,
            },
        ]
        mock_gen.generate.return_value = fake_masks
        mock_mask_gen_cls.return_value = mock_gen

        # Create backend and run
        backend = SAM2Backend(model_size="base", checkpoint_path=str(mock_checkpoint), device="cpu")

        test_image = np.random.rand(64, 64, 3).astype(np.float32)
        seg_input = SegmentationInput(image=test_image, gamma=1.0, mode="auto")

        result = backend.segment(seg_input)

        # Verify result structure
        assert isinstance(result, SegmentationResult)
        assert result.masks.shape == (2, 64, 64)
        assert len(result.scores) == 2
        assert len(result.metadata) == 2
        assert result.scores[0] == 0.95
        assert result.metadata[0].area == 400


class TestSAM2BackendPromptedMode:
    """Test prompted mode with mocked SAM2ImagePredictor."""

    @patch("sam2.build_sam.build_sam2")
    @patch("sam2.sam2_image_predictor.SAM2ImagePredictor")
    def test_points_mode_success(self, mock_predictor_cls, mock_build, mock_checkpoint):
        """Test points mode returns correct result."""
        from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

        # Mock model
        mock_model = Mock()
        mock_build.return_value = mock_model

        # Mock predictor
        mock_pred = Mock()
        fake_masks = np.ones((1, 64, 64), dtype=bool)
        fake_scores = np.array([0.92])
        fake_logits = np.random.rand(1, 64, 64)
        mock_pred.predict.return_value = (fake_masks, fake_scores, fake_logits)
        mock_predictor_cls.return_value = mock_pred

        # Create backend and run
        backend = SAM2Backend(model_size="base", checkpoint_path=str(mock_checkpoint), device="cpu")

        test_image = np.random.rand(64, 64, 3).astype(np.float32)
        seg_input = SegmentationInput(
            image=test_image, gamma=1.0, mode="points", prompts={"points": [[10, 20]], "labels": [1]}
        )

        result = backend.segment(seg_input)

        # Verify
        assert result.masks.shape[0] == 1
        assert result.scores[0] == 0.92
        mock_pred.set_image.assert_called_once()
        mock_pred.predict.assert_called_once()

    @patch("sam2.build_sam.build_sam2")
    @patch("sam2.sam2_image_predictor.SAM2ImagePredictor")
    def test_bbox_mode_success(self, mock_predictor_cls, mock_build, mock_checkpoint):
        """Test bbox mode converts bbox to corner points."""
        from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

        # Mock model
        mock_model = Mock()
        mock_build.return_value = mock_model

        # Mock predictor
        mock_pred = Mock()
        fake_masks = np.ones((1, 64, 64), dtype=bool)
        fake_scores = np.array([0.89])
        fake_logits = np.random.rand(1, 64, 64)
        mock_pred.predict.return_value = (fake_masks, fake_scores, fake_logits)
        mock_predictor_cls.return_value = mock_pred

        # Create backend and run
        backend = SAM2Backend(model_size="base", checkpoint_path=str(mock_checkpoint), device="cpu")

        test_image = np.random.rand(64, 64, 3).astype(np.float32)
        seg_input = SegmentationInput(
            image=test_image, gamma=1.0, mode="bbox", prompts={"bbox": [10, 10, 30, 40]}  # x1, y1, x2, y2
        )

        result = backend.segment(seg_input)

        # Verify bbox was converted to corner points
        assert result.masks.shape[0] == 1
        mock_pred.predict.assert_called_once()

        # Check that predict was called with corner points (labels 2 and 3)
        call_args = mock_pred.predict.call_args
        labels = call_args[1].get("point_labels")
        assert labels is not None
        assert 2 in labels  # Top-left corner
        assert 3 in labels  # Bottom-right corner


class TestSAM2BackendVideoMode:
    """Test video mode with mocked SAM2VideoPredictor."""

    @patch("sam2.build_sam.build_sam2")
    @patch("sam2.build_sam.build_sam2_video_predictor")
    def test_video_mode_success(self, mock_build_video, mock_build, mock_checkpoint, tmp_path):
        """Test video mode tracks object across frames."""
        from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

        # Mock models
        mock_model = Mock()
        mock_build.return_value = mock_model

        mock_video_pred = Mock()
        mock_build_video.return_value = mock_video_pred

        # Mock inference state and video segments
        fake_state = {"num_frames": 3, "video_height": 64, "video_width": 64}
        mock_video_pred.init_state.return_value = fake_state

        fake_segments = {
            0: {1: np.ones((64, 64), dtype=bool)},
            1: {1: np.ones((64, 64), dtype=bool)},
            2: {1: np.ones((64, 64), dtype=bool)},
        }
        mock_video_pred.propagate_in_video.return_value = fake_segments

        # Create fake video frames
        video_dir = tmp_path / "frames"
        video_dir.mkdir()
        for i in range(3):
            (video_dir / f"{i:05d}.jpg").touch()

        # Create backend and run
        backend = SAM2Backend(model_size="base", checkpoint_path=str(mock_checkpoint), device="cpu")

        seg_input = SegmentationInput(
            image=None,
            gamma=1.0,
            mode="video",
            video_path=str(video_dir),
            prompts={"frame_idx": 0, "object_id": 1, "points": [[10, 20]], "labels": [1]},
        )

        result = backend.segment(seg_input)

        # Verify
        assert result.masks.shape == (3, 64, 64)
        assert len(result.scores) == 3
        assert len(result.metadata) == 3
        assert all(result.temporal_ids == 1)  # Same object tracked

        mock_video_pred.init_state.assert_called_once()
        mock_video_pred.add_new_points.assert_called_once()
        mock_video_pred.propagate_in_video.assert_called_once()
        mock_video_pred.reset_state.assert_called_once()


class TestSAM2BackendEdgeCases:
    """Test edge cases and error handling."""

    @patch("sam2.build_sam.build_sam2")
    @patch("sam2.automatic_mask_generator.SAM2AutomaticMaskGenerator")
    def test_empty_auto_results(self, mock_gen_cls, mock_build, mock_checkpoint):
        """Test auto mode with no masks found."""
        from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

        # Mock model
        mock_model = Mock()
        mock_build.return_value = mock_model

        # Mock empty results
        mock_gen = Mock()
        mock_gen.generate.return_value = []  # No masks
        mock_gen_cls.return_value = mock_gen

        backend = SAM2Backend(model_size="base", checkpoint_path=str(mock_checkpoint), device="cpu")

        test_image = np.zeros((64, 64, 3), dtype=np.float32)  # Empty image
        seg_input = SegmentationInput(image=test_image, gamma=1.0, mode="auto")

        result = backend.segment(seg_input)

        # Should return empty result
        assert result.masks.shape == (0, 64, 64)
        assert len(result.scores) == 0
        assert len(result.metadata) == 0

    def test_invalid_video_path(self, mock_checkpoint):
        """Test video mode with non-existent path."""
        from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

        backend = SAM2Backend(model_size="base", checkpoint_path=str(mock_checkpoint), device="cpu")

        seg_input = SegmentationInput(
            image=None,
            gamma=1.0,
            mode="video",
            video_path="/nonexistent/path",
            prompts={"frame_idx": 0, "object_id": 1, "points": [[10, 20]], "labels": [1]},
        )

        with pytest.raises(FileNotFoundError, match="Video path not found"):
            backend.segment(seg_input)
