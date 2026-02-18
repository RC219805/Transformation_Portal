"""Tests for SAM2 backend with material classification integration (Phase 4D)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from transformation_portal.spatial_ai.segmentation.contracts import MaskMetadata, SegmentationInput, SegmentationResult
from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend


class TestSAM2MaterialClassificationIntegration:
    """Test SAM2 backend material classification integration."""

    def test_backend_init_without_material_classification(self, tmp_path):
        """Test that backend initializes without material classification by default."""
        # Create dummy checkpoint
        checkpoint = tmp_path / "test.pt"
        checkpoint.write_bytes(b"dummy")

        backend = SAM2Backend(
            model_size="base",
            device="cpu",
            checkpoint_path=str(checkpoint),
            enable_material_classification=False,
        )

        assert not backend.enable_material_classification
        assert backend._material_classifier is None

    def test_backend_init_with_material_classification(self, tmp_path):
        """Test that backend initializes material classifier when enabled."""
        # Create dummy checkpoint
        checkpoint = tmp_path / "test.pt"
        checkpoint.write_bytes(b"dummy")

        backend = SAM2Backend(
            model_size="base",
            device="cpu",
            checkpoint_path=str(checkpoint),
            enable_material_classification=True,
            material_confidence_threshold=0.5,
        )

        assert backend.enable_material_classification
        assert backend._material_classifier is not None
        assert backend._material_classifier.confidence_threshold == 0.5

    @patch.object(SAM2Backend, "_load_model")
    @patch.object(SAM2Backend, "_segment_auto")
    def test_auto_mode_without_material_classification(
        self,
        mock_segment_auto,
        mock_load_model,
        tmp_path,
    ):
        """Test auto mode without material classification (baseline behavior)."""
        checkpoint = tmp_path / "test.pt"
        checkpoint.write_bytes(b"dummy")

        # Create backend WITHOUT material classification
        backend = SAM2Backend(
            model_size="base",
            device="cpu",
            checkpoint_path=str(checkpoint),
            enable_material_classification=False,
        )

        # Mock _segment_auto to return result WITHOUT material labels
        mock_result = SegmentationResult(
            masks=np.ones((2, 64, 64), dtype=bool),
            scores=np.array([0.9, 0.85], dtype=np.float32),
            metadata=[
                MaskMetadata(area=100, bbox=(10, 10, 20, 20), stability_score=0.95),
                MaskMetadata(area=200, bbox=(30, 30, 25, 25), stability_score=0.92),
            ],
        )
        mock_segment_auto.return_value = mock_result

        # Run segmentation
        seg_input = SegmentationInput(
            image=np.ones((64, 64, 3), dtype=np.float32),
            gamma=1.0,
            mode="auto",
        )
        result = backend.segment(seg_input)

        # Verify no material labels
        assert len(result.metadata) == 2
        assert result.metadata[0].material_label is None
        assert result.metadata[0].material_confidence is None
        assert result.metadata[1].material_label is None
        assert result.metadata[1].material_confidence is None

    @patch.object(SAM2Backend, "_load_model")
    def test_auto_mode_with_material_classification(self, mock_load_model, tmp_path):
        """Test auto mode WITH material classification."""
        checkpoint = tmp_path / "test.pt"
        checkpoint.write_bytes(b"dummy")

        # Create backend WITH material classification
        backend = SAM2Backend(
            model_size="base",
            device="cpu",
            checkpoint_path=str(checkpoint),
            enable_material_classification=True,
            material_confidence_threshold=0.3,
        )

        # Mock the mask generator
        backend._mask_generator = MagicMock()
        backend._mask_generator.generate.return_value = [
            {
                "segmentation": np.ones((64, 64), dtype=bool),
                "predicted_iou": 0.9,
                "stability_score": 0.95,
                "area": 1000,
                "bbox": [10, 10, 30, 30],  # xyxy
            },
            {
                "segmentation": np.ones((64, 64), dtype=bool),
                "predicted_iou": 0.85,
                "stability_score": 0.92,
                "area": 800,
                "bbox": [40, 40, 55, 55],  # xyxy
            },
        ]

        # Mock material classifier
        backend._material_classifier.is_available = MagicMock(return_value=True)
        backend._material_classifier.classify_masks = MagicMock(
            return_value=[
                ("wood floor", 0.87),
                ("marble surface", 0.72),
            ]
        )

        # Run segmentation
        seg_input = SegmentationInput(
            image=np.ones((64, 64, 3), dtype=np.float32),
            gamma=1.0,
            mode="auto",
        )
        result = backend.segment(seg_input)

        # Verify material labels are populated
        assert len(result.metadata) == 2
        assert result.metadata[0].material_label == "wood floor"
        assert result.metadata[0].material_confidence == pytest.approx(0.87)
        assert result.metadata[1].material_label == "marble surface"
        assert result.metadata[1].material_confidence == pytest.approx(0.72)

        # Verify classifier was called
        backend._material_classifier.classify_masks.assert_called_once()

    @patch.object(SAM2Backend, "_load_model")
    def test_auto_mode_with_material_classification_unavailable(self, mock_load_model, tmp_path):
        """Test auto mode when material classification enabled but CLIP unavailable."""
        checkpoint = tmp_path / "test.pt"
        checkpoint.write_bytes(b"dummy")

        # Create backend WITH material classification enabled
        backend = SAM2Backend(
            model_size="base",
            device="cpu",
            checkpoint_path=str(checkpoint),
            enable_material_classification=True,
        )

        # Mock the mask generator
        backend._mask_generator = MagicMock()
        backend._mask_generator.generate.return_value = [
            {
                "segmentation": np.ones((64, 64), dtype=bool),
                "predicted_iou": 0.9,
                "stability_score": 0.95,
                "area": 1000,
                "bbox": [10, 10, 30, 30],
            }
        ]

        # Mock material classifier as unavailable
        backend._material_classifier.is_available = MagicMock(return_value=False)

        # Run segmentation
        seg_input = SegmentationInput(
            image=np.ones((64, 64, 3), dtype=np.float32),
            gamma=1.0,
            mode="auto",
        )
        result = backend.segment(seg_input)

        # Verify no material labels (graceful fallback)
        assert len(result.metadata) == 1
        assert result.metadata[0].material_label is None
        assert result.metadata[0].material_confidence is None

    @patch.object(SAM2Backend, "_load_model")
    def test_auto_mode_with_low_confidence_materials(self, mock_load_model, tmp_path):
        """Test auto mode with materials below confidence threshold."""
        checkpoint = tmp_path / "test.pt"
        checkpoint.write_bytes(b"dummy")

        backend = SAM2Backend(
            model_size="base",
            device="cpu",
            checkpoint_path=str(checkpoint),
            enable_material_classification=True,
            material_confidence_threshold=0.5,  # High threshold
        )

        backend._mask_generator = MagicMock()
        backend._mask_generator.generate.return_value = [
            {
                "segmentation": np.ones((64, 64), dtype=bool),
                "predicted_iou": 0.9,
                "stability_score": 0.95,
                "area": 1000,
                "bbox": [10, 10, 30, 30],
            }
        ]

        # Mock material classifier returning low confidence
        backend._material_classifier.is_available = MagicMock(return_value=True)
        backend._material_classifier.classify_masks = MagicMock(
            return_value=[
                (None, None),  # Below threshold
            ]
        )

        seg_input = SegmentationInput(
            image=np.ones((64, 64, 3), dtype=np.float32),
            gamma=1.0,
            mode="auto",
        )
        result = backend.segment(seg_input)

        # Verify no material label (below threshold)
        assert result.metadata[0].material_label is None
        assert result.metadata[0].material_confidence is None

    @patch.object(SAM2Backend, "_load_model")
    def test_prompted_mode_with_material_classification(self, mock_load_model, tmp_path):
        """Test prompted mode with material classification."""
        checkpoint = tmp_path / "test.pt"
        checkpoint.write_bytes(b"dummy")

        backend = SAM2Backend(
            model_size="base",
            device="cpu",
            checkpoint_path=str(checkpoint),
            enable_material_classification=True,
        )

        # Mock image predictor
        backend._image_predictor = MagicMock()
        backend._image_predictor.set_image = MagicMock()
        backend._image_predictor.predict = MagicMock(
            return_value=(
                np.ones((3, 64, 64), dtype=bool),  # 3 masks
                np.array([0.95, 0.90, 0.85], dtype=np.float32),  # scores
                None,  # logits (unused)
            )
        )

        # Mock material classifier
        backend._material_classifier.is_available = MagicMock(return_value=True)
        backend._material_classifier.classify_masks = MagicMock(
            return_value=[
                ("glass", 0.82),
                ("steel", 0.76),
                (None, None),  # Third mask below threshold
            ]
        )

        seg_input = SegmentationInput(
            image=np.ones((64, 64, 3), dtype=np.float32),
            gamma=1.0,
            mode="points",
            prompts={"points": [[32, 32]], "labels": [1]},
        )
        result = backend.segment(seg_input)

        # Verify material labels
        assert len(result.metadata) == 3
        assert result.metadata[0].material_label == "glass"
        assert result.metadata[0].material_confidence == pytest.approx(0.82)
        assert result.metadata[1].material_label == "steel"
        assert result.metadata[1].material_confidence == pytest.approx(0.76)
        assert result.metadata[2].material_label is None
        assert result.metadata[2].material_confidence is None
