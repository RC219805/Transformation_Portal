"""Unit tests for SAM2 backend with mocked dependencies (Phase 4B).

These tests mock internal segmentation methods to avoid model loading.
This provides fast unit tests without complex ML mocking.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from transformation_portal.spatial_ai.segmentation.contracts import MaskMetadata, SegmentationInput, SegmentationResult

# Module-level availability check
try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = [pytest.mark.ml, pytest.mark.skipif(not HAS_TORCH, reason="torch not installed (optional dependency)")]

PINNED_SAM21_LARGE_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
PINNED_SAM21_LARGE_SHA256 = "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"


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

        assert backend.checkpoint_path.name == "sam2.1_hiera_large.pt"
        assert "checkpoints" in str(backend.checkpoint_path)
        assert backend.model_config == PINNED_SAM21_LARGE_CONFIG
        assert backend.expected_sha256 == PINNED_SAM21_LARGE_SHA256

    def test_init_large_model_allows_explicit_integrity_overrides(self, mock_checkpoint):
        """Explicit SAM2 model config and checksum overrides must win over defaults."""
        from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

        backend = SAM2Backend(
            model_size="large",
            checkpoint_path=str(mock_checkpoint),
            model_config="configs/custom/sam2_large.yaml",
            expected_sha256="b" * 64,
            device="cpu",
        )

        assert backend.model_config == "configs/custom/sam2_large.yaml"
        assert backend.expected_sha256 == "b" * 64


class TestSAM2BackendAutoMode:
    """Test auto mode by mocking internal _segment_auto method."""

    def test_auto_mode_returns_results(self, mock_checkpoint):
        """Test auto mode calls _segment_auto and returns results."""
        from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

        backend = SAM2Backend(model_size="base", checkpoint_path=str(mock_checkpoint), device="cpu")

        # Create fake result
        fake_masks = np.ones((2, 64, 64), dtype=bool)
        fake_scores = np.array([0.95, 0.88], dtype=np.float32)
        fake_metadata = [
            MaskMetadata(area=400, bbox=(10, 10, 20, 20), stability_score=0.95),
            MaskMetadata(area=225, bbox=(30, 30, 15, 15), stability_score=0.88),
        ]
        fake_result = SegmentationResult(masks=fake_masks, scores=fake_scores, metadata=fake_metadata)

        # Mock both _load_model and _segment_auto
        with patch.object(backend, "_load_model"):  # Skip model loading
            with patch.object(backend, "_segment_auto", return_value=fake_result) as mock_auto:
                test_image = np.random.rand(64, 64, 3).astype(np.float32)
                seg_input = SegmentationInput(image=test_image, gamma=1.0, mode="auto")

                result = backend.segment(seg_input)

                # Verify
                mock_auto.assert_called_once_with(seg_input)
                assert result.masks.shape == (2, 64, 64)
                assert result.scores[0] == pytest.approx(0.95)
                assert result.metadata[0].area == 400


class TestSAM2BackendPromptedMode:
    """Test prompted mode by mocking internal _segment_prompted method."""

    def test_points_mode_returns_results(self, mock_checkpoint):
        """Test points mode calls _segment_prompted and returns results."""
        from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

        backend = SAM2Backend(model_size="base", checkpoint_path=str(mock_checkpoint), device="cpu")

        # Create fake result
        fake_masks = np.ones((1, 64, 64), dtype=bool)
        fake_scores = np.array([0.92], dtype=np.float32)
        fake_metadata = [MaskMetadata(area=500, bbox=(5, 5, 30, 30), stability_score=0.92)]
        fake_result = SegmentationResult(masks=fake_masks, scores=fake_scores, metadata=fake_metadata)

        # Mock both _load_model and _segment_prompted
        with patch.object(backend, "_load_model"):
            with patch.object(backend, "_segment_prompted", return_value=fake_result) as mock_prompted:
                test_image = np.random.rand(64, 64, 3).astype(np.float32)
                seg_input = SegmentationInput(
                    image=test_image, gamma=1.0, mode="points", prompts={"points": [[10, 20]], "labels": [1]}
                )

                result = backend.segment(seg_input)

                # Verify
                mock_prompted.assert_called_once_with(seg_input)
                assert result.masks.shape[0] == 1
                assert result.scores[0] == pytest.approx(0.92)

    def test_bbox_mode_returns_results(self, mock_checkpoint):
        """Test bbox mode calls _segment_prompted and returns results."""
        from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

        backend = SAM2Backend(model_size="base", checkpoint_path=str(mock_checkpoint), device="cpu")

        # Create fake result
        fake_masks = np.ones((1, 64, 64), dtype=bool)
        fake_scores = np.array([0.89], dtype=np.float32)
        fake_metadata = [MaskMetadata(area=600, bbox=(10, 10, 30, 40), stability_score=0.89)]
        fake_result = SegmentationResult(masks=fake_masks, scores=fake_scores, metadata=fake_metadata)

        # Mock both _load_model and _segment_prompted
        with patch.object(backend, "_load_model"):
            with patch.object(backend, "_segment_prompted", return_value=fake_result) as mock_prompted:
                test_image = np.random.rand(64, 64, 3).astype(np.float32)
                seg_input = SegmentationInput(image=test_image, gamma=1.0, mode="bbox", prompts={"bbox": [10, 10, 30, 40]})

                result = backend.segment(seg_input)

                # Verify
                mock_prompted.assert_called_once_with(seg_input)
                assert result.masks.shape[0] == 1

    def test_segment_prompted_orders_multimask_results_by_score(self, mock_checkpoint):
        """Prompted mode should expose the highest-confidence proposal first."""
        from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

        backend = SAM2Backend(model_size="base", checkpoint_path=str(mock_checkpoint), device="cpu")
        backend._image_predictor = Mock()

        masks = np.zeros((3, 64, 64), dtype=np.float32)
        masks[0, 31:33, 31:33] = 1.0
        masks[1, 12:52, 12:52] = 1.0
        masks[2, 20:44, 20:44] = 1.0
        scores = np.array([0.10, 0.95, 0.60], dtype=np.float32)

        backend._image_predictor.predict.return_value = (masks, scores, None)

        seg_input = SegmentationInput(
            image=np.random.rand(64, 64, 3).astype(np.float32),
            gamma=1.0,
            mode="points",
            prompts={"points": [[32, 32]], "labels": [1]},
        )

        result = backend._segment_prompted(seg_input)

        assert result.scores.tolist() == pytest.approx([0.95, 0.60, 0.10])
        assert result.metadata[0].area == 1600
        assert result.masks[0].sum() == 1600

    def test_segment_prompted_uses_normalized_prediction_extraction(self, mock_checkpoint):
        """Prompted mode should route predictor outputs through the SAM2 extraction helper."""
        from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

        backend = SAM2Backend(model_size="base", checkpoint_path=str(mock_checkpoint), device="cpu")
        backend._image_predictor = Mock()

        masks = np.zeros((2, 32, 32), dtype=np.float32)
        masks[0, 8:24, 8:24] = 1.0
        masks[1, 10:22, 10:22] = 1.0
        scores = np.array([0.4, 0.8], dtype=np.float32)
        backend._image_predictor.predict.return_value = (masks, scores, None)

        seg_input = SegmentationInput(
            image=np.random.rand(32, 32, 3).astype(np.float32),
            gamma=1.0,
            mode="bbox",
            prompts={"bbox": [6, 6, 26, 26]},
        )

        with patch.object(backend, "_extract_sam2_predictions", wraps=backend._extract_sam2_predictions) as extractor:
            result = backend._segment_prompted(seg_input)

        extractor.assert_called_once()
        assert result.scores.tolist() == pytest.approx([0.8, 0.4])
        assert result.metadata[0].stability_score == pytest.approx(0.8)


class TestSAM2BackendVideoMode:
    """Test video mode by mocking internal _segment_video method."""

    def test_video_mode_returns_results(self, mock_checkpoint, tmp_path):
        """Test video mode calls _segment_video and returns results."""
        from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

        backend = SAM2Backend(model_size="base", checkpoint_path=str(mock_checkpoint), device="cpu")

        # Create fake video frames
        video_dir = tmp_path / "frames"
        video_dir.mkdir()
        for i in range(3):
            (video_dir / f"{i:05d}.jpg").touch()

        # Create fake result with temporal IDs
        fake_masks = np.ones((3, 64, 64), dtype=bool)
        fake_scores = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        fake_metadata = [MaskMetadata(area=100, bbox=(10, 10, 20, 20), stability_score=1.0) for _ in range(3)]
        fake_temporal_ids = np.array([1, 1, 1], dtype=np.int32)
        fake_result = SegmentationResult(
            masks=fake_masks, scores=fake_scores, metadata=fake_metadata, temporal_ids=fake_temporal_ids
        )

        # Mock both _load_model and _segment_video
        with patch.object(backend, "_load_model"):
            with patch.object(backend, "_segment_video", return_value=fake_result) as mock_video:
                seg_input = SegmentationInput(
                    image=None,
                    gamma=1.0,
                    mode="video",
                    video_path=str(video_dir),
                    prompts={"frame_idx": 0, "object_id": 1, "points": [[10, 20]], "labels": [1]},
                )

                result = backend.segment(seg_input)

                # Verify
                mock_video.assert_called_once_with(seg_input)
                assert result.masks.shape == (3, 64, 64)
                assert len(result.scores) == 3
                assert all(result.temporal_ids == 1)


class TestSAM2BackendEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_auto_results(self, mock_checkpoint):
        """Test auto mode with no masks found."""
        from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

        backend = SAM2Backend(model_size="base", checkpoint_path=str(mock_checkpoint), device="cpu")

        # Create empty result
        fake_masks = np.empty((0, 64, 64), dtype=bool)
        fake_scores = np.array([], dtype=np.float32)
        fake_metadata = []
        fake_result = SegmentationResult(masks=fake_masks, scores=fake_scores, metadata=fake_metadata)

        # Mock both _load_model and _segment_auto
        with patch.object(backend, "_load_model"):
            with patch.object(backend, "_segment_auto", return_value=fake_result):
                test_image = np.zeros((64, 64, 3), dtype=np.float32)
                seg_input = SegmentationInput(image=test_image, gamma=1.0, mode="auto")

                result = backend.segment(seg_input)

                # Should return empty result
                assert result.masks.shape == (0, 64, 64)
                assert len(result.scores) == 0
                assert len(result.metadata) == 0

    def test_unsupported_mode_raises_error(self, mock_checkpoint):
        """Test unsupported mode raises ValueError."""
        from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

        backend = SAM2Backend(model_size="base", checkpoint_path=str(mock_checkpoint), device="cpu")

        # Create input with invalid mode (bypass contract validation for test)
        test_image = np.random.rand(64, 64, 3).astype(np.float32)

        # Manually create input to bypass validation
        seg_input = SegmentationInput.__new__(SegmentationInput)
        seg_input.image = test_image
        seg_input.gamma = 1.0
        seg_input.mode = "invalid_mode"  # type: ignore
        seg_input.prompts = None
        seg_input.video_path = None
        seg_input.prev_masks = None
        seg_input.frame_idx = None

        # Mock _load_model but let segment() run to check mode validation
        with patch.object(backend, "_load_model"):
            with pytest.raises(ValueError, match="Unsupported mode"):
                backend.segment(seg_input)


class TestSAM2BackendIntegrity:
    """Checksum enforcement for the canonical SAM 2.1 large checkpoint."""

    def test_load_raises_typed_integrity_error_on_sha_mismatch(self, mock_checkpoint, monkeypatch):
        """A canonical-load checksum mismatch must fail closed with the typed error."""
        import sys
        from types import ModuleType

        from transformation_portal.spatial_ai.segmentation.sam2_backend import (
            SAM2Backend,
            SAM2CheckpointIntegrityError,
        )

        backend = SAM2Backend(model_size="large", checkpoint_path=str(mock_checkpoint), device="cpu")

        sam2_module = ModuleType("sam2")
        build_module = ModuleType("sam2.build_sam")
        predictor_module = ModuleType("sam2.sam2_image_predictor")
        automask_module = ModuleType("sam2.automatic_mask_generator")

        build_module.build_sam2 = lambda **kwargs: object()
        predictor_module.SAM2ImagePredictor = lambda model: object()
        automask_module.SAM2AutomaticMaskGenerator = lambda **kwargs: object()
        sam2_module.build_sam = build_module
        sam2_module.sam2_image_predictor = predictor_module
        sam2_module.automatic_mask_generator = automask_module
        monkeypatch.setitem(sys.modules, "sam2", sam2_module)
        monkeypatch.setitem(sys.modules, "sam2.build_sam", build_module)
        monkeypatch.setitem(sys.modules, "sam2.sam2_image_predictor", predictor_module)
        monkeypatch.setitem(sys.modules, "sam2.automatic_mask_generator", automask_module)
        monkeypatch.setattr(
            "transformation_portal.spatial_ai.segmentation.sam2_backend._compute_file_sha256",
            lambda path: "0" * 64,
        )

        with pytest.raises(SAM2CheckpointIntegrityError, match="SHA-256 mismatch"):
            backend._load_model()
