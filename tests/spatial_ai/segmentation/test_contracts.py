"""Unit tests for segmentation contracts (Phase 2.1)."""

import numpy as np
import pytest

from transformation_portal.spatial_ai.segmentation.contracts import MaskMetadata, SegmentationInput, SegmentationResult


class TestSegmentationInput:
    """Test SegmentationInput contract validation."""

    def test_valid_auto_mode(self):
        """Test valid automatic segmentation input."""
        image = np.random.rand(512, 512, 3).astype(np.float32)
        seg_input = SegmentationInput(
            image=image,
            gamma=1.0,
            mode="auto",
        )
        assert seg_input.gamma == 1.0
        assert seg_input.mode == "auto"

    def test_gamma_enforcement(self):
        """Test gamma=1.0 enforcement (SpatialCaptureV1 contract)."""
        image = np.random.rand(512, 512, 3).astype(np.float32)

        with pytest.raises(ValueError, match="gamma=1.0"):
            SegmentationInput(
                image=image,
                gamma=2.2,  # Invalid
                mode="auto",
            )

    def test_dtype_enforcement(self):
        """Test float32 dtype requirement."""
        image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="float32"):
            SegmentationInput(
                image=image,
                gamma=1.0,
                mode="auto",
            )

    def test_shape_validation(self):
        """Test image shape validation (H, W, 3)."""
        # Wrong number of dimensions
        with pytest.raises(ValueError, match="\\(H, W, 3\\)"):
            SegmentationInput(
                image=np.random.rand(512, 512).astype(np.float32),
                gamma=1.0,
                mode="auto",
            )

        # Wrong number of channels
        with pytest.raises(ValueError, match="\\(H, W, 3\\)"):
            SegmentationInput(
                image=np.random.rand(512, 512, 4).astype(np.float32),
                gamma=1.0,
                mode="auto",
            )

    def test_prompted_mode_requires_prompts(self):
        """Test that prompted modes require prompts."""
        image = np.random.rand(512, 512, 3).astype(np.float32)

        # Points mode without prompts
        with pytest.raises(ValueError, match="requires prompts"):
            SegmentationInput(
                image=image,
                gamma=1.0,
                mode="points",
            )

        # Bbox mode without prompts
        with pytest.raises(ValueError, match="requires prompts"):
            SegmentationInput(
                image=image,
                gamma=1.0,
                mode="bbox",
            )

    def test_video_mode_requires_prev_masks(self):
        """Test that video mode requires prev_masks."""
        image = np.random.rand(512, 512, 3).astype(np.float32)

        with pytest.raises(ValueError, match="requires prev_masks"):
            SegmentationInput(
                image=image,
                gamma=1.0,
                mode="video",
            )

    def test_prev_masks_validation(self):
        """Test prev_masks shape and dtype validation."""
        image = np.random.rand(512, 512, 3).astype(np.float32)

        # Wrong dtype
        with pytest.raises(ValueError, match="must be bool"):
            SegmentationInput(
                image=image,
                gamma=1.0,
                mode="video",
                prev_masks=np.random.rand(5, 512, 512).astype(np.float32),
            )

        # Wrong shape (2D instead of 3D)
        with pytest.raises(ValueError, match="\\(N, H, W\\)"):
            SegmentationInput(
                image=image,
                gamma=1.0,
                mode="video",
                prev_masks=np.random.rand(512, 512) > 0.5,
            )

        # Spatial dimensions mismatch
        with pytest.raises(ValueError, match="must match image dims"):
            SegmentationInput(
                image=image,
                gamma=1.0,
                mode="video",
                prev_masks=np.random.rand(5, 256, 256) > 0.5,
            )


class TestMaskMetadata:
    """Test MaskMetadata validation."""

    def test_valid_metadata(self):
        """Test valid mask metadata."""
        metadata = MaskMetadata(
            area=1000,
            bbox=(10, 20, 100, 200),
            stability_score=0.95,
        )
        assert metadata.area == 1000
        assert metadata.stability_score == 0.95

    def test_area_validation(self):
        """Test area must be positive."""
        with pytest.raises(ValueError, match="area must be positive"):
            MaskMetadata(
                area=0,
                bbox=(10, 20, 100, 200),
                stability_score=0.95,
            )

    def test_stability_score_range(self):
        """Test stability score must be in [0, 1]."""
        # Too low
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            MaskMetadata(
                area=1000,
                bbox=(10, 20, 100, 200),
                stability_score=-0.1,
            )

        # Too high
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            MaskMetadata(
                area=1000,
                bbox=(10, 20, 100, 200),
                stability_score=1.5,
            )

    def test_bbox_validation(self):
        """Test bounding box width/height must be positive."""
        # Zero width
        with pytest.raises(ValueError, match="width/height must be positive"):
            MaskMetadata(
                area=1000,
                bbox=(10, 20, 0, 200),
                stability_score=0.95,
            )

        # Negative height
        with pytest.raises(ValueError, match="width/height must be positive"):
            MaskMetadata(
                area=1000,
                bbox=(10, 20, 100, -50),
                stability_score=0.95,
            )

    def test_material_confidence_validation(self):
        """Test material confidence must be in [0, 1] if provided."""
        # Valid
        metadata = MaskMetadata(
            area=1000,
            bbox=(10, 20, 100, 200),
            stability_score=0.95,
            material_label="wood",
            material_confidence=0.85,
        )
        assert metadata.material_confidence == 0.85

        # Invalid (too high)
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            MaskMetadata(
                area=1000,
                bbox=(10, 20, 100, 200),
                stability_score=0.95,
                material_label="wood",
                material_confidence=1.2,
            )


class TestSegmentationResult:
    """Test SegmentationResult contract validation."""

    def test_valid_result(self):
        """Test valid segmentation result."""
        masks = np.random.rand(5, 512, 512) > 0.5
        scores = np.random.rand(5).astype(np.float32)
        metadata = [
            MaskMetadata(
                area=1000,
                bbox=(10, 20, 100, 200),
                stability_score=float(scores[i]),
            )
            for i in range(5)
        ]

        result = SegmentationResult(
            masks=masks,
            scores=scores,
            metadata=metadata,
        )
        assert result.masks.shape[0] == 5
        assert len(result.metadata) == 5

    def test_masks_dtype_validation(self):
        """Test masks must be bool dtype."""
        masks = np.random.rand(5, 512, 512).astype(np.float32)  # Wrong dtype
        scores = np.random.rand(5).astype(np.float32)
        metadata = [MaskMetadata(area=1000, bbox=(10, 20, 100, 200), stability_score=0.9) for _ in range(5)]

        with pytest.raises(ValueError, match="bool dtype"):
            SegmentationResult(masks=masks, scores=scores, metadata=metadata)

    def test_masks_shape_validation(self):
        """Test masks must be (N, H, W)."""
        # Wrong number of dimensions
        masks = np.random.rand(512, 512) > 0.5
        scores = np.array([0.9], dtype=np.float32)
        metadata = [MaskMetadata(area=1000, bbox=(10, 20, 100, 200), stability_score=0.9)]

        with pytest.raises(ValueError, match="\\(N, H, W\\)"):
            SegmentationResult(masks=masks, scores=scores, metadata=metadata)

    def test_scores_range_validation(self):
        """Test scores must be in [0, 1]."""
        masks = np.random.rand(5, 512, 512) > 0.5
        scores = np.array([0.9, 1.2, 0.8, 0.7, 0.6], dtype=np.float32)  # 1.2 is invalid
        metadata = [MaskMetadata(area=1000, bbox=(10, 20, 100, 200), stability_score=0.9) for _ in range(5)]

        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            SegmentationResult(masks=masks, scores=scores, metadata=metadata)

    def test_metadata_length_validation(self):
        """Test metadata length must match N."""
        masks = np.random.rand(5, 512, 512) > 0.5
        scores = np.random.rand(5).astype(np.float32)
        metadata = [MaskMetadata(area=1000, bbox=(10, 20, 100, 200), stability_score=0.9) for _ in range(3)]  # Wrong length

        with pytest.raises(ValueError, match="Metadata length must match"):
            SegmentationResult(masks=masks, scores=scores, metadata=metadata)

    def test_temporal_ids_validation(self):
        """Test temporal IDs shape and dtype validation."""
        masks = np.random.rand(5, 512, 512) > 0.5
        scores = np.random.rand(5).astype(np.float32)
        metadata = [MaskMetadata(area=1000, bbox=(10, 20, 100, 200), stability_score=0.9) for _ in range(5)]

        # Valid
        temporal_ids = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        result = SegmentationResult(masks=masks, scores=scores, metadata=metadata, temporal_ids=temporal_ids)
        assert result.temporal_ids is not None

        # Wrong shape
        with pytest.raises(ValueError, match="shape must be"):
            SegmentationResult(
                masks=masks,
                scores=scores,
                metadata=metadata,
                temporal_ids=np.array([0, 1, 2], dtype=np.int32),  # Wrong length
            )

        # Wrong dtype
        with pytest.raises(ValueError, match="must be int32/int64"):
            SegmentationResult(
                masks=masks,
                scores=scores,
                metadata=metadata,
                temporal_ids=np.array([0, 1, 2, 3, 4], dtype=np.float32),
            )
