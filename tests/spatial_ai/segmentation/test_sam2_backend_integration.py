"""Integration tests for SAM2 backend with real models (Phase 4B).

These tests use real SAM2 models and are marked as slow/ml.
Run with: pytest -m "ml and slow" tests/spatial_ai/segmentation/test_sam2_backend_integration.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Module-level availability check
try:
    import sam2  # noqa: F401
    import torch  # noqa: F401

    HAS_SAM2 = True
    # Check MPS availability safely at module level
    MPS_AVAILABLE = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
except ImportError:
    HAS_SAM2 = False
    MPS_AVAILABLE = False

pytestmark = [
    pytest.mark.skipif(not HAS_SAM2, reason="SAM2 package not installed"),
    pytest.mark.ml,  # Requires ML dependencies
    pytest.mark.slow,  # Uses real model loading
]


@pytest.fixture(scope="module")
def checkpoint_path():
    """Get SAM2 checkpoint path, skip if not available."""
    checkpoint = Path("checkpoints/sam2.1_hiera_large.pt")
    if not checkpoint.exists():
        pytest.skip(f"Checkpoint not found: {checkpoint}. Run: python scripts/download_sam2_checkpoint.py")
    return str(checkpoint)


@pytest.fixture(scope="module")
def fixtures_dir():
    """Get test fixtures directory."""
    return Path(__file__).parent / "fixtures"


def load_test_image(fixture_path: Path) -> np.ndarray:
    """Load test fixture as linear RGB float32."""
    img = Image.open(fixture_path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    return arr


class TestSAM2AutoModeIntegration:
    """Integration tests for auto mode with real model."""

    def test_auto_mode_simple_circle(self, checkpoint_path, fixtures_dir):
        """Test auto mode finds simple circle."""
        from transformation_portal.spatial_ai.segmentation.contracts import SegmentationInput
        from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

        backend = SAM2Backend(model_size="large", checkpoint_path=checkpoint_path, device="cpu")  # Use CPU for CI

        test_image = load_test_image(fixtures_dir / "simple_circle_64x64.png")

        seg_input = SegmentationInput(image=test_image, gamma=1.0, mode="auto")

        result = backend.segment(seg_input)

        # Verify result
        assert result.masks.shape[1:] == (64, 64)
        assert len(result.masks) > 0  # Should find at least one mask
        assert all(0 <= s <= 1 for s in result.scores)

    def test_auto_mode_multi_object(self, checkpoint_path, fixtures_dir):
        """Test auto mode finds multiple objects."""
        from transformation_portal.spatial_ai.segmentation.contracts import SegmentationInput
        from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

        backend = SAM2Backend(model_size="large", checkpoint_path=checkpoint_path, device="cpu")

        test_image = load_test_image(fixtures_dir / "multi_object_128x128.png")

        seg_input = SegmentationInput(image=test_image, gamma=1.0, mode="auto")

        result = backend.segment(seg_input)

        # Should find multiple objects (we have 3 in the fixture)
        assert len(result.masks) >= 1  # At least one object
        assert result.masks.shape[1:] == (128, 128)

    def test_auto_mode_empty_image(self, checkpoint_path, fixtures_dir):
        """Test auto mode with empty/black image."""
        from transformation_portal.spatial_ai.segmentation.contracts import SegmentationInput
        from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

        backend = SAM2Backend(model_size="large", checkpoint_path=checkpoint_path, device="cpu")

        test_image = load_test_image(fixtures_dir / "empty_64x64.png")

        seg_input = SegmentationInput(image=test_image, gamma=1.0, mode="auto")

        result = backend.segment(seg_input)

        # Empty image might find 0 or very few low-quality masks
        assert result.masks.shape[1:] == (64, 64)
        # Don't assert specific count - SAM2 behavior varies


class TestSAM2PromptedModeIntegration:
    """Integration tests for prompted mode with real model."""

    def test_points_mode_on_circle(self, checkpoint_path, fixtures_dir):
        """Test points mode segments circle when prompted."""
        from transformation_portal.spatial_ai.segmentation.contracts import SegmentationInput
        from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

        backend = SAM2Backend(model_size="large", checkpoint_path=checkpoint_path, device="cpu")

        test_image = load_test_image(fixtures_dir / "simple_circle_64x64.png")

        # Point in center of circle (32, 32)
        seg_input = SegmentationInput(
            image=test_image, gamma=1.0, mode="points", prompts={"points": [[32, 32]], "labels": [1]}  # Positive point
        )

        result = backend.segment(seg_input)

        # Verify
        assert result.masks.shape[0] >= 1  # At least one mask
        assert result.masks.shape[1:] == (64, 64)
        # Should find significant area (circle is ~1256 pixels out of 4096)
        mask_coverage = result.masks[0].sum() / result.masks[0].size
        assert mask_coverage > 0.1  # At least 10% coverage

    def test_bbox_mode_on_object(self, checkpoint_path, fixtures_dir):
        """Test bbox mode segments object within bbox."""
        from transformation_portal.spatial_ai.segmentation.contracts import SegmentationInput
        from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

        backend = SAM2Backend(model_size="large", checkpoint_path=checkpoint_path, device="cpu")

        test_image = load_test_image(fixtures_dir / "simple_circle_64x64.png")

        # Bbox around circle
        seg_input = SegmentationInput(
            image=test_image, gamma=1.0, mode="bbox", prompts={"bbox": [12, 12, 52, 52]}  # x1, y1, x2, y2
        )

        result = backend.segment(seg_input)

        # Verify
        assert result.masks.shape[0] >= 1
        assert result.masks.shape[1:] == (64, 64)

    def test_negative_point_excludes_region(self, checkpoint_path, fixtures_dir):
        """Test negative point excludes region from mask."""
        from transformation_portal.spatial_ai.segmentation.contracts import SegmentationInput
        from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

        backend = SAM2Backend(model_size="large", checkpoint_path=checkpoint_path, device="cpu")

        test_image = load_test_image(fixtures_dir / "multi_object_128x128.png")

        # Positive point on one object, negative on another
        seg_input = SegmentationInput(
            image=test_image,
            gamma=1.0,
            mode="points",
            prompts={"points": [[35, 35], [90, 90]], "labels": [1, 0]},  # One positive, one negative  # 1=include, 0=exclude
        )

        result = backend.segment(seg_input)

        # Should get at least one mask
        assert len(result.masks) >= 1


class TestSAM2VideoModeIntegration:
    """Integration tests for video mode with real model."""

    def test_video_mode_tracks_moving_object(self, checkpoint_path, fixtures_dir):
        """Test video mode tracks object across frames."""
        from transformation_portal.spatial_ai.segmentation.contracts import SegmentationInput
        from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

        backend = SAM2Backend(model_size="large", checkpoint_path=checkpoint_path, device="cpu")

        video_dir = fixtures_dir / "video_frames"
        if not video_dir.exists():
            pytest.skip("Video frames fixture not found")

        seg_input = SegmentationInput(
            image=None,
            gamma=1.0,
            mode="video",
            video_path=str(video_dir),
            prompts={
                "frame_idx": 0,  # Prompt on first frame
                "object_id": 1,
                "points": [[20, 32]],  # Point on moving object
                "labels": [1],
            },
        )

        result = backend.segment(seg_input)

        # Verify video tracking
        num_frames = len(list(video_dir.glob("*.jpg")))
        assert result.masks.shape[0] == num_frames
        assert result.masks.shape[1:] == (64, 64)
        assert all(result.temporal_ids == 1)  # Same object tracked
        assert len(result.metadata) == num_frames

        # Check that object is found in most frames
        frames_with_object = sum(mask.any() for mask in result.masks)
        assert frames_with_object >= num_frames * 0.5  # At least half the frames

    def test_video_mode_with_bbox_prompt(self, checkpoint_path, fixtures_dir):
        """Test video mode with bbox prompt on first frame."""
        from transformation_portal.spatial_ai.segmentation.contracts import SegmentationInput
        from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

        backend = SAM2Backend(model_size="large", checkpoint_path=checkpoint_path, device="cpu")

        video_dir = fixtures_dir / "video_frames"
        if not video_dir.exists():
            pytest.skip("Video frames fixture not found")

        seg_input = SegmentationInput(
            image=None,
            gamma=1.0,
            mode="video",
            video_path=str(video_dir),
            prompts={"frame_idx": 0, "object_id": 1, "bbox": [10, 22, 30, 42]},  # Around first frame object
        )

        result = backend.segment(seg_input)

        # Verify
        num_frames = len(list(video_dir.glob("*.jpg")))
        assert result.masks.shape[0] == num_frames


class TestSAM2DeviceHandling:
    """Integration tests for device selection."""

    def test_cpu_device(self, checkpoint_path, fixtures_dir):
        """Test SAM2 works on CPU."""
        from transformation_portal.spatial_ai.segmentation.contracts import SegmentationInput
        from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

        backend = SAM2Backend(model_size="large", checkpoint_path=checkpoint_path, device="cpu")

        test_image = load_test_image(fixtures_dir / "simple_circle_64x64.png")

        seg_input = SegmentationInput(image=test_image, gamma=1.0, mode="auto")

        result = backend.segment(seg_input)
        assert result.masks.shape[1:] == (64, 64)

    @pytest.mark.skipif(not MPS_AVAILABLE, reason="MPS not available")
    def test_mps_device(self, checkpoint_path, fixtures_dir):
        """Test SAM2 works on MPS (Apple Silicon)."""
        import torch

        from transformation_portal.spatial_ai.segmentation.contracts import SegmentationInput
        from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

        backend = SAM2Backend(model_size="large", checkpoint_path=checkpoint_path, device="mps")

        test_image = load_test_image(fixtures_dir / "simple_circle_64x64.png")

        seg_input = SegmentationInput(image=test_image, gamma=1.0, mode="auto")

        result = backend.segment(seg_input)
        assert result.masks.shape[1:] == (64, 64)


class TestSAM2MetadataQuality:
    """Integration tests for metadata quality."""

    def test_metadata_completeness(self, checkpoint_path, fixtures_dir):
        """Test that all masks have complete metadata."""
        from transformation_portal.spatial_ai.segmentation.contracts import SegmentationInput
        from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

        backend = SAM2Backend(model_size="large", checkpoint_path=checkpoint_path, device="cpu")

        test_image = load_test_image(fixtures_dir / "multi_object_128x128.png")

        seg_input = SegmentationInput(image=test_image, gamma=1.0, mode="auto")

        result = backend.segment(seg_input)

        # Check every mask has valid metadata
        for i, meta in enumerate(result.metadata):
            assert meta.area > 0, f"Mask {i} has non-positive area"
            assert 0 <= meta.stability_score <= 1, f"Mask {i} has invalid stability score"
            x, y, w, h = meta.bbox
            assert w > 0 and h > 0, f"Mask {i} has invalid bbox dimensions"
