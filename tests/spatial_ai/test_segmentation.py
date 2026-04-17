"""Tests for spatial_ai segmentation module (Phase 5 coverage).

Tests for:
- SegmentationInput contract validation
- SegmentationResult contract validation
- MaskMetadata validation
- MaskProcessor operations

All tests use mocks - no ML model downloads or GPU requirements.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from transformation_portal.spatial_ai.segmentation.contracts import (
    MaskMetadata,
    SegmentationInput,
    SegmentationResult,
)
from transformation_portal.spatial_ai.segmentation.mask_processor import MaskProcessor

pytestmark = [pytest.mark.unit, pytest.mark.ml]


@pytest.fixture
def linear_image():
    """Create a linear RGB image (gamma=1.0) for testing."""
    return np.random.rand(256, 256, 3).astype(np.float32)


@pytest.fixture
def sample_masks():
    """Create sample boolean masks."""
    masks = np.zeros((3, 256, 256), dtype=bool)
    masks[0, 50:100, 50:100] = True  # 2500 pixels
    masks[1, 100:200, 100:200] = True  # 10000 pixels
    masks[2, 150:180, 150:180] = True  # 900 pixels
    return masks


@pytest.fixture
def sample_scores():
    """Create sample confidence scores."""
    return np.array([0.95, 0.87, 0.72], dtype=np.float32)


@pytest.fixture
def sample_metadata():
    """Create sample mask metadata."""
    return [
        MaskMetadata(area=2500, bbox=(50, 50, 50, 50), stability_score=0.9),
        MaskMetadata(area=10000, bbox=(100, 100, 100, 100), stability_score=0.85),
        MaskMetadata(area=900, bbox=(150, 150, 30, 30), stability_score=0.7),
    ]


class TestSegmentationInput:
    """Test SegmentationInput contract validation."""

    def test_valid_auto_mode(self, linear_image):
        """Test valid input for auto mode."""
        seg_input = SegmentationInput(
            image=linear_image,
            gamma=1.0,
            mode="auto",
        )

        assert seg_input.mode == "auto"
        assert seg_input.gamma == 1.0
        assert seg_input.image.dtype == np.float32

    def test_valid_points_mode(self, linear_image):
        """Test valid input for points mode."""
        prompts = {"points": [[128, 128]], "labels": [1]}
        seg_input = SegmentationInput(
            image=linear_image,
            gamma=1.0,
            mode="points",
            prompts=prompts,
        )

        assert seg_input.mode == "points"
        assert seg_input.prompts == prompts

    def test_valid_bbox_mode(self, linear_image):
        """Test valid input for bbox mode."""
        prompts = {"bbox": [50, 50, 200, 200]}
        seg_input = SegmentationInput(
            image=linear_image,
            gamma=1.0,
            mode="bbox",
            prompts=prompts,
        )

        assert seg_input.mode == "bbox"

    def test_valid_video_mode(self):
        """Test valid input for video mode."""
        prompts = {"frame_idx": 0, "object_id": 1, "points": [[128, 128]], "labels": [1]}
        seg_input = SegmentationInput(
            image=None,
            gamma=1.0,
            mode="video",
            video_path="/path/to/video.mp4",
            prompts=prompts,
        )

        assert seg_input.mode == "video"
        assert seg_input.video_path == "/path/to/video.mp4"

    def test_invalid_gamma_raises(self, linear_image):
        """Test that non-linear gamma is rejected."""
        with pytest.raises(ValueError, match="gamma=1.0"):
            SegmentationInput(
                image=linear_image,
                gamma=2.2,  # sRGB gamma
                mode="auto",
            )

    def test_invalid_dtype_raises(self):
        """Test that non-float32 image is rejected."""
        uint8_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="float32"):
            SegmentationInput(
                image=uint8_image,
                gamma=1.0,
                mode="auto",
            )

    def test_invalid_shape_raises(self):
        """Test that invalid image shape is rejected."""
        grayscale = np.random.rand(256, 256).astype(np.float32)

        with pytest.raises(ValueError, match="\\(H, W, 3\\)"):
            SegmentationInput(
                image=grayscale,
                gamma=1.0,
                mode="auto",
            )

    def test_points_mode_requires_prompts(self, linear_image):
        """Test that points mode requires prompts."""
        with pytest.raises(ValueError, match="requires prompts"):
            SegmentationInput(
                image=linear_image,
                gamma=1.0,
                mode="points",
            )

    def test_bbox_mode_requires_prompts(self, linear_image):
        """Test that bbox mode requires prompts."""
        with pytest.raises(ValueError, match="requires prompts"):
            SegmentationInput(
                image=linear_image,
                gamma=1.0,
                mode="bbox",
            )

    def test_video_mode_requires_video_path(self, linear_image):
        """Test that video mode requires video_path."""
        prompts = {"frame_idx": 0, "object_id": 1, "points": [[128, 128]], "labels": [1]}

        with pytest.raises(ValueError, match="requires video_path"):
            SegmentationInput(
                image=None,
                gamma=1.0,
                mode="video",
                prompts=prompts,
            )

    def test_video_mode_requires_prompts(self):
        """Test that video mode requires prompts."""
        with pytest.raises(ValueError, match="requires prompts"):
            SegmentationInput(
                image=None,
                gamma=1.0,
                mode="video",
                video_path="/path/to/video.mp4",
            )

    def test_non_video_mode_requires_image(self):
        """Test that non-video modes require image."""
        with pytest.raises(ValueError, match="requires image"):
            SegmentationInput(
                image=None,
                gamma=1.0,
                mode="auto",
            )

    def test_prev_masks_validation(self, linear_image):
        """Test prev_masks validation."""
        valid_prev_masks = np.zeros((2, 256, 256), dtype=bool)
        seg_input = SegmentationInput(
            image=linear_image,
            gamma=1.0,
            mode="auto",
            prev_masks=valid_prev_masks,
        )
        assert seg_input.prev_masks is not None

    def test_prev_masks_wrong_dtype_raises(self, linear_image):
        """Test that prev_masks with wrong dtype is rejected."""
        wrong_dtype = np.zeros((2, 256, 256), dtype=np.uint8)

        with pytest.raises(ValueError, match="bool"):
            SegmentationInput(
                image=linear_image,
                gamma=1.0,
                mode="auto",
                prev_masks=wrong_dtype,
            )

    def test_prev_masks_dimension_mismatch_raises(self, linear_image):
        """Test that prev_masks with mismatched dimensions is rejected."""
        wrong_size = np.zeros((2, 128, 128), dtype=bool)

        with pytest.raises(ValueError, match="must match image dims"):
            SegmentationInput(
                image=linear_image,
                gamma=1.0,
                mode="auto",
                prev_masks=wrong_size,
            )


class TestMaskMetadata:
    """Test MaskMetadata contract validation."""

    def test_valid_metadata(self):
        """Test valid metadata creation."""
        metadata = MaskMetadata(
            area=1000,
            bbox=(10, 20, 50, 60),
            stability_score=0.85,
        )

        assert metadata.area == 1000
        assert metadata.bbox == (10, 20, 50, 60)
        assert metadata.stability_score == 0.85

    def test_metadata_with_material(self):
        """Test metadata with material classification."""
        metadata = MaskMetadata(
            area=5000,
            bbox=(0, 0, 100, 100),
            stability_score=0.9,
            material_label="wood",
            material_confidence=0.78,
        )

        assert metadata.material_label == "wood"
        assert metadata.material_confidence == 0.78

    def test_invalid_area_raises(self):
        """Test that non-positive area is rejected."""
        with pytest.raises(ValueError, match="area must be positive"):
            MaskMetadata(area=0, bbox=(0, 0, 10, 10), stability_score=0.5)

        with pytest.raises(ValueError, match="area must be positive"):
            MaskMetadata(area=-100, bbox=(0, 0, 10, 10), stability_score=0.5)

    def test_invalid_stability_score_raises(self):
        """Test that out-of-range stability score is rejected."""
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            MaskMetadata(area=100, bbox=(0, 0, 10, 10), stability_score=1.5)

        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            MaskMetadata(area=100, bbox=(0, 0, 10, 10), stability_score=-0.1)

    def test_invalid_material_confidence_raises(self):
        """Test that out-of-range material confidence is rejected."""
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            MaskMetadata(
                area=100,
                bbox=(0, 0, 10, 10),
                stability_score=0.5,
                material_label="wood",
                material_confidence=1.2,
            )

    def test_invalid_bbox_raises(self):
        """Test that invalid bounding box is rejected."""
        with pytest.raises(ValueError, match="width/height must be positive"):
            MaskMetadata(area=100, bbox=(0, 0, 0, 10), stability_score=0.5)

        with pytest.raises(ValueError, match="width/height must be positive"):
            MaskMetadata(area=100, bbox=(0, 0, 10, -5), stability_score=0.5)


class TestSegmentationResult:
    """Test SegmentationResult contract validation."""

    def test_valid_result(self, sample_masks, sample_scores, sample_metadata):
        """Test valid result creation."""
        result = SegmentationResult(
            masks=sample_masks,
            scores=sample_scores,
            metadata=sample_metadata,
        )

        assert result.masks.shape == (3, 256, 256)
        assert len(result.scores) == 3
        assert len(result.metadata) == 3

    def test_valid_result_with_temporal_ids(self, sample_masks, sample_scores, sample_metadata):
        """Test valid result with temporal IDs."""
        temporal_ids = np.array([0, 1, 2], dtype=np.int32)
        result = SegmentationResult(
            masks=sample_masks,
            scores=sample_scores,
            metadata=sample_metadata,
            temporal_ids=temporal_ids,
        )

        assert result.temporal_ids is not None
        np.testing.assert_array_equal(result.temporal_ids, [0, 1, 2])

    def test_invalid_masks_dtype_raises(self, sample_scores, sample_metadata):
        """Test that non-bool masks are rejected."""
        float_masks = np.random.rand(3, 256, 256).astype(np.float32)

        with pytest.raises(ValueError, match="bool"):
            SegmentationResult(
                masks=float_masks,
                scores=sample_scores,
                metadata=sample_metadata,
            )

    def test_invalid_masks_shape_raises(self, sample_scores, sample_metadata):
        """Test that 2D masks are rejected."""
        masks_2d = np.zeros((256, 256), dtype=bool)

        with pytest.raises(ValueError, match="\\(N, H, W\\)"):
            SegmentationResult(
                masks=masks_2d,
                scores=sample_scores,
                metadata=sample_metadata,
            )

    def test_invalid_scores_shape_raises(self, sample_masks, sample_metadata):
        """Test that mismatched scores shape is rejected."""
        wrong_scores = np.array([0.9, 0.8], dtype=np.float32)

        with pytest.raises(ValueError, match="Scores shape"):
            SegmentationResult(
                masks=sample_masks,
                scores=wrong_scores,
                metadata=sample_metadata,
            )

    def test_invalid_scores_range_raises(self, sample_masks, sample_metadata):
        """Test that out-of-range scores are rejected."""
        bad_scores = np.array([0.9, 1.5, 0.7], dtype=np.float32)

        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            SegmentationResult(
                masks=sample_masks,
                scores=bad_scores,
                metadata=sample_metadata,
            )

    def test_invalid_metadata_length_raises(self, sample_masks, sample_scores):
        """Test that mismatched metadata length is rejected."""
        short_metadata = [
            MaskMetadata(area=100, bbox=(0, 0, 10, 10), stability_score=0.5),
        ]

        with pytest.raises(ValueError, match="Metadata length"):
            SegmentationResult(
                masks=sample_masks,
                scores=sample_scores,
                metadata=short_metadata,
            )

    def test_invalid_temporal_ids_shape_raises(self, sample_masks, sample_scores, sample_metadata):
        """Test that mismatched temporal IDs shape is rejected."""
        wrong_ids = np.array([0, 1], dtype=np.int32)

        with pytest.raises(ValueError, match="Temporal IDs shape"):
            SegmentationResult(
                masks=sample_masks,
                scores=sample_scores,
                metadata=sample_metadata,
                temporal_ids=wrong_ids,
            )


class TestMaskProcessor:
    """Test MaskProcessor class."""

    def test_processor_initialization(self):
        """Test processor initialization."""
        processor = MaskProcessor(
            min_area=200,
            min_stability=0.6,
            iou_threshold=0.4,
        )

        assert processor.min_area == 200
        assert processor.min_stability == 0.6
        assert processor.iou_threshold == 0.4

    def test_invalid_min_area_raises(self):
        """Test that invalid min_area is rejected."""
        with pytest.raises(ValueError, match="min_area must be positive"):
            MaskProcessor(min_area=0)

    def test_invalid_min_stability_raises(self):
        """Test that invalid min_stability is rejected."""
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            MaskProcessor(min_stability=1.5)

    def test_invalid_iou_threshold_raises(self):
        """Test that invalid iou_threshold is rejected."""
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            MaskProcessor(iou_threshold=-0.1)

    def test_filter_masks(self, sample_masks, sample_scores, sample_metadata):
        """Test filtering masks by area and stability."""
        result = SegmentationResult(
            masks=sample_masks,
            scores=sample_scores,
            metadata=sample_metadata,
        )

        processor = MaskProcessor(min_area=1000, min_stability=0.75)
        filtered = processor.filter_masks(result)

        # Only first two masks should pass (area >= 1000, stability >= 0.75)
        assert filtered.masks.shape[0] == 2
        assert len(filtered.metadata) == 2

    def test_filter_masks_empty_result(self, sample_masks, sample_scores, sample_metadata):
        """Test filtering returns empty result when nothing passes."""
        result = SegmentationResult(
            masks=sample_masks,
            scores=sample_scores,
            metadata=sample_metadata,
        )

        processor = MaskProcessor(min_area=100000, min_stability=0.99)
        filtered = processor.filter_masks(result)

        assert filtered.masks.shape[0] == 0
        assert len(filtered.metadata) == 0
        assert len(filtered.scores) == 0

    def test_refine_masks(self):
        """Test morphological mask refinement."""
        # Create mask with noise
        masks = np.zeros((1, 100, 100), dtype=bool)
        masks[0, 40:60, 40:60] = True
        masks[0, 10, 10] = True  # Single pixel noise

        processor = MaskProcessor()
        refined = processor.refine_masks(masks, kernel_size=3)

        assert refined.dtype == bool
        assert refined.shape == masks.shape
        # Single pixel noise should be removed
        assert not refined[0, 10, 10]

    def test_refine_masks_invalid_dtype_raises(self):
        """Test that refining non-bool masks raises."""
        processor = MaskProcessor()
        float_masks = np.random.rand(1, 100, 100).astype(np.float32)

        with pytest.raises(ValueError, match="bool"):
            processor.refine_masks(float_masks)

    def test_refine_masks_invalid_shape_raises(self):
        """Test that refining 2D masks raises."""
        processor = MaskProcessor()
        masks_2d = np.zeros((100, 100), dtype=bool)

        with pytest.raises(ValueError, match="\\(N, H, W\\)"):
            processor.refine_masks(masks_2d)

    def test_track_temporal(self):
        """Test temporal tracking with IoU matching."""
        # Previous frame masks
        prev_masks = np.zeros((2, 100, 100), dtype=bool)
        prev_masks[0, 20:40, 20:40] = True  # ID 0
        prev_masks[1, 60:80, 60:80] = True  # ID 1
        prev_ids = np.array([0, 1], dtype=np.int32)

        # Current frame masks (slightly shifted)
        current_masks = np.zeros((2, 100, 100), dtype=bool)
        current_masks[0, 25:45, 25:45] = True  # Should match ID 0
        current_masks[1, 65:85, 65:85] = True  # Should match ID 1

        processor = MaskProcessor(iou_threshold=0.3)
        current_ids = processor.track_temporal(current_masks, prev_masks, prev_ids)

        # IDs should be preserved for matching masks
        assert current_ids.dtype in [np.int32, np.int64]
        assert len(current_ids) == 2

    def test_track_temporal_new_object(self):
        """Test temporal tracking assigns new ID for non-matching masks."""
        prev_masks = np.zeros((1, 100, 100), dtype=bool)
        prev_masks[0, 10:30, 10:30] = True
        prev_ids = np.array([0], dtype=np.int32)

        current_masks = np.zeros((2, 100, 100), dtype=bool)
        current_masks[0, 10:30, 10:30] = True  # Should match ID 0
        current_masks[1, 70:90, 70:90] = True  # New object, should get new ID

        processor = MaskProcessor(iou_threshold=0.5)
        current_ids = processor.track_temporal(current_masks, prev_masks, prev_ids)

        assert current_ids[0] == 0  # Matched
        assert current_ids[1] == 1  # New ID

    def test_resolve_overlaps(self):
        """Test resolving overlapping masks."""
        masks = np.zeros((2, 100, 100), dtype=bool)
        masks[0, 30:70, 30:70] = True  # Low confidence
        masks[1, 40:60, 40:60] = True  # High confidence (overlap)
        scores = np.array([0.5, 0.9], dtype=np.float32)

        processor = MaskProcessor()
        resolved = processor.resolve_overlaps(masks, scores)

        # Higher score mask should take precedence in overlap region
        assert resolved[1, 50, 50]  # High score mask keeps its pixels
        # Low score mask should lose pixels in overlap
        # Check that some pixels were removed from low score mask
        assert resolved[0].sum() < masks[0].sum()

    def test_resolve_overlaps_invalid_dtype_raises(self):
        """Test that resolving non-bool masks raises."""
        processor = MaskProcessor()

        with pytest.raises(ValueError, match="bool"):
            processor.resolve_overlaps(
                np.random.rand(2, 100, 100).astype(np.float32),
                np.array([0.5, 0.9], dtype=np.float32),
            )

    def test_resolve_overlaps_shape_mismatch_raises(self):
        """Test that mismatched shapes raise."""
        processor = MaskProcessor()
        masks = np.zeros((2, 100, 100), dtype=bool)
        wrong_scores = np.array([0.5, 0.9, 0.7], dtype=np.float32)

        with pytest.raises(ValueError, match="same N"):
            processor.resolve_overlaps(masks, wrong_scores)

    def test_compute_iou_matrix(self):
        """Test IoU matrix computation."""
        masks_a = np.zeros((2, 100, 100), dtype=bool)
        masks_a[0, 0:50, 0:50] = True
        masks_a[1, 50:100, 50:100] = True

        masks_b = np.zeros((2, 100, 100), dtype=bool)
        masks_b[0, 0:50, 0:50] = True  # Same as masks_a[0]
        masks_b[1, 0:25, 0:25] = True  # Partial overlap with masks_a[0]

        processor = MaskProcessor()
        iou_matrix = processor._compute_iou_matrix(masks_a, masks_b)

        assert iou_matrix.shape == (2, 2)
        # Perfect match
        assert iou_matrix[0, 0] == pytest.approx(1.0)
        # Partial overlap
        assert 0 < iou_matrix[0, 1] < 1
        # No overlap
        assert iou_matrix[1, 1] == 0.0
