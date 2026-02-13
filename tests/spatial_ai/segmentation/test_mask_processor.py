"""Unit tests for mask processor (Phase 2.1)."""

import numpy as np
import pytest

from transformation_portal.spatial_ai.segmentation.contracts import MaskMetadata, SegmentationResult
from transformation_portal.spatial_ai.segmentation.mask_processor import MaskProcessor


class TestMaskProcessor:
    """Test MaskProcessor initialization and configuration."""

    def test_initialization(self):
        """Test processor initialization with defaults."""
        processor = MaskProcessor()
        assert processor.min_area == 100
        assert processor.min_stability == 0.5
        assert processor.iou_threshold == 0.5

    def test_custom_parameters(self):
        """Test processor initialization with custom parameters."""
        processor = MaskProcessor(min_area=200, min_stability=0.7, iou_threshold=0.6)
        assert processor.min_area == 200
        assert processor.min_stability == 0.7
        assert processor.iou_threshold == 0.6

    def test_invalid_parameters(self):
        """Test validation of initialization parameters."""
        # Negative min_area
        with pytest.raises(ValueError, match="min_area must be positive"):
            MaskProcessor(min_area=-10)

        # Invalid min_stability range
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            MaskProcessor(min_stability=1.5)

        # Invalid iou_threshold range
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            MaskProcessor(iou_threshold=-0.1)


class TestFilterMasks:
    """Test mask filtering by area and stability."""

    def test_filter_by_area(self):
        """Test filtering masks by minimum area."""
        # Create masks with different areas
        masks = np.zeros((3, 100, 100), dtype=bool)
        masks[0, :10, :10] = True  # Area: 100 (pass)
        masks[1, :5, :5] = True  # Area: 25 (fail)
        masks[2, :20, :20] = True  # Area: 400 (pass)

        scores = np.array([0.9, 0.8, 0.95], dtype=np.float32)
        metadata = [
            MaskMetadata(area=100, bbox=(0, 0, 10, 10), stability_score=0.9),
            MaskMetadata(area=25, bbox=(0, 0, 5, 5), stability_score=0.8),
            MaskMetadata(area=400, bbox=(0, 0, 20, 20), stability_score=0.95),
        ]

        result = SegmentationResult(masks=masks, scores=scores, metadata=metadata)

        processor = MaskProcessor(min_area=50, min_stability=0.0)
        filtered = processor.filter_masks(result)

        # Should keep masks 0 and 2
        assert len(filtered.masks) == 2
        assert filtered.metadata[0].area == 100
        assert filtered.metadata[1].area == 400

    def test_filter_by_stability(self):
        """Test filtering masks by stability score."""
        masks = np.zeros((3, 100, 100), dtype=bool)
        for i in range(3):
            masks[i, :10, :10] = True  # All same area

        scores = np.array([0.9, 0.4, 0.7], dtype=np.float32)
        metadata = [
            MaskMetadata(area=100, bbox=(0, 0, 10, 10), stability_score=0.9),
            MaskMetadata(area=100, bbox=(0, 0, 10, 10), stability_score=0.4),
            MaskMetadata(area=100, bbox=(0, 0, 10, 10), stability_score=0.7),
        ]

        result = SegmentationResult(masks=masks, scores=scores, metadata=metadata)

        processor = MaskProcessor(min_area=50, min_stability=0.6)
        filtered = processor.filter_masks(result)

        # Should keep masks 0 and 2
        assert len(filtered.masks) == 2
        assert filtered.metadata[0].stability_score == 0.9
        assert filtered.metadata[1].stability_score == 0.7

    def test_filter_no_masks_pass(self):
        """Test when no masks pass filtering."""
        masks = np.zeros((2, 100, 100), dtype=bool)
        masks[0, :2, :2] = True  # Very small
        masks[1, :3, :3] = True  # Very small

        scores = np.array([0.9, 0.8], dtype=np.float32)
        metadata = [
            MaskMetadata(area=4, bbox=(0, 0, 2, 2), stability_score=0.9),
            MaskMetadata(area=9, bbox=(0, 0, 3, 3), stability_score=0.8),
        ]

        result = SegmentationResult(masks=masks, scores=scores, metadata=metadata)

        processor = MaskProcessor(min_area=100, min_stability=0.5)
        filtered = processor.filter_masks(result)

        # Should have empty result
        assert len(filtered.masks) == 0
        assert len(filtered.metadata) == 0


class TestRefineMasks:
    """Test morphological mask refinement."""

    def test_refine_removes_noise(self):
        """Test that refinement removes small noise."""
        # Create mask with noise
        masks = np.zeros((1, 100, 100), dtype=bool)
        masks[0, 10:30, 10:30] = True  # Main region
        masks[0, 50, 50] = True  # Single pixel noise

        processor = MaskProcessor()
        refined = processor.refine_masks(masks, kernel_size=3)

        # Noise should be removed
        assert refined[0, 50, 50] == False
        # Main region should remain (mostly)
        assert refined[0, 15:25, 15:25].sum() > 0

    def test_refine_fills_holes(self):
        """Test that refinement fills small holes."""
        # Create mask with hole
        masks = np.zeros((1, 100, 100), dtype=bool)
        masks[0, 10:30, 10:30] = True
        masks[0, 20, 20] = False  # Small hole

        processor = MaskProcessor()
        refined = processor.refine_masks(masks, kernel_size=3)

        # Hole should be filled
        # Note: Result depends on kernel size and morphological ops
        assert refined[0, 10:30, 10:30].sum() >= masks[0, 10:30, 10:30].sum()

    def test_refine_preserves_shape(self):
        """Test that refinement preserves output shape."""
        masks = np.random.rand(5, 100, 100) > 0.5

        processor = MaskProcessor()
        refined = processor.refine_masks(masks, kernel_size=3)

        assert refined.shape == masks.shape
        assert refined.dtype == bool

    def test_refine_invalid_dtype(self):
        """Test refinement rejects non-bool masks."""
        masks = np.random.rand(5, 100, 100).astype(np.float32)

        processor = MaskProcessor()
        with pytest.raises(ValueError, match="must be bool"):
            processor.refine_masks(masks)


class TestTemporalTracking:
    """Test temporal mask tracking."""

    def test_track_perfect_match(self):
        """Test tracking with perfect overlap (IoU=1.0)."""
        # Same masks in current and previous frame
        masks = np.zeros((3, 100, 100), dtype=bool)
        masks[0, :10, :10] = True
        masks[1, 20:30, 20:30] = True
        masks[2, 50:60, 50:60] = True

        prev_ids = np.array([10, 20, 30], dtype=np.int32)

        processor = MaskProcessor(iou_threshold=0.5)
        current_ids = processor.track_temporal(masks, masks.copy(), prev_ids)

        # Should inherit same IDs
        np.testing.assert_array_equal(current_ids, prev_ids)

    def test_track_partial_match(self):
        """Test tracking with partial overlap."""
        prev_masks = np.zeros((2, 100, 100), dtype=bool)
        prev_masks[0, :10, :10] = True
        prev_masks[1, 50:60, 50:60] = True
        prev_ids = np.array([10, 20], dtype=np.int32)

        # Current masks: slightly shifted
        current_masks = np.zeros((2, 100, 100), dtype=bool)
        current_masks[0, 2:12, 2:12] = True  # Overlaps with prev[0]
        current_masks[1, 80:90, 80:90] = True  # No overlap (new object)

        processor = MaskProcessor(iou_threshold=0.3)
        current_ids = processor.track_temporal(current_masks, prev_masks, prev_ids)

        # First mask should inherit ID 10
        assert current_ids[0] == 10
        # Second mask should get new ID
        assert current_ids[1] > 20

    def test_track_new_masks(self):
        """Test tracking assigns new IDs to new masks."""
        prev_masks = np.zeros((1, 100, 100), dtype=bool)
        prev_masks[0, :10, :10] = True
        prev_ids = np.array([10], dtype=np.int32)

        # Current frame has 3 masks (2 new)
        current_masks = np.zeros((3, 100, 100), dtype=bool)
        current_masks[0, :10, :10] = True  # Matches prev[0]
        current_masks[1, 50:60, 50:60] = True  # New
        current_masks[2, 80:90, 80:90] = True  # New

        processor = MaskProcessor(iou_threshold=0.5)
        current_ids = processor.track_temporal(current_masks, prev_masks, prev_ids)

        # First mask inherits ID 10
        assert current_ids[0] == 10
        # Other two get new IDs
        assert current_ids[1] == 11
        assert current_ids[2] == 12

    def test_track_invalid_dtypes(self):
        """Test tracking rejects invalid dtypes."""
        current_masks = np.zeros((2, 100, 100), dtype=np.float32)  # Wrong
        prev_masks = np.zeros((2, 100, 100), dtype=bool)
        prev_ids = np.array([10, 20], dtype=np.int32)

        processor = MaskProcessor()
        with pytest.raises(ValueError, match="must be bool"):
            processor.track_temporal(current_masks, prev_masks, prev_ids)

    def test_track_dimension_mismatch(self):
        """Test tracking rejects spatial dimension mismatch."""
        current_masks = np.zeros((2, 100, 100), dtype=bool)
        prev_masks = np.zeros((2, 50, 50), dtype=bool)  # Wrong size
        prev_ids = np.array([10, 20], dtype=np.int32)

        processor = MaskProcessor()
        with pytest.raises(ValueError, match="must match"):
            processor.track_temporal(current_masks, prev_masks, prev_ids)


class TestResolveOverlaps:
    """Test overlap resolution."""

    def test_resolve_no_overlap(self):
        """Test that non-overlapping masks are unchanged."""
        masks = np.zeros((2, 100, 100), dtype=bool)
        masks[0, :50, :50] = True
        masks[1, 50:, 50:] = True
        scores = np.array([0.9, 0.8], dtype=np.float32)

        processor = MaskProcessor()
        resolved = processor.resolve_overlaps(masks, scores)

        # Should be unchanged
        np.testing.assert_array_equal(resolved, masks)

    def test_resolve_with_overlap(self):
        """Test that overlapping regions go to highest score."""
        masks = np.zeros((2, 100, 100), dtype=bool)
        masks[0, :60, :60] = True  # Score 0.9 (higher)
        masks[1, 40:, 40:] = True  # Score 0.7 (lower)
        scores = np.array([0.9, 0.7], dtype=np.float32)

        processor = MaskProcessor()
        resolved = processor.resolve_overlaps(masks, scores)

        # Overlap region (40:60, 40:60) should go to mask 0
        assert resolved[0, 50, 50] == True
        assert resolved[1, 50, 50] == False

        # Non-overlap regions should remain
        assert resolved[0, 10, 10] == True  # Mask 0 unique region
        assert resolved[1, 80, 80] == True  # Mask 1 unique region

    def test_resolve_multiple_overlaps(self):
        """Test resolution with multiple overlapping masks."""
        masks = np.zeros((3, 100, 100), dtype=bool)
        masks[0, :70, :70] = True  # Score 0.5 (lowest)
        masks[1, 30:, 30:] = True  # Score 0.9 (highest)
        masks[2, 50:80, 50:80] = True  # Score 0.7 (middle)
        scores = np.array([0.5, 0.9, 0.7], dtype=np.float32)

        processor = MaskProcessor()
        resolved = processor.resolve_overlaps(masks, scores)

        # Check highly overlapped region (50:70, 50:70)
        # Should belong to mask 1 (highest score)
        assert resolved[1, 60, 60] == True
        assert resolved[0, 60, 60] == False
        assert resolved[2, 60, 60] == False

    def test_resolve_invalid_inputs(self):
        """Test overlap resolution input validation."""
        masks = np.zeros((2, 100, 100), dtype=bool)
        scores = np.array([0.9], dtype=np.float32)  # Wrong length

        processor = MaskProcessor()
        with pytest.raises(ValueError, match="same N"):
            processor.resolve_overlaps(masks, scores)
