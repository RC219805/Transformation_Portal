"""Tests for SAM2 segmentation backend (Phase 3)."""

from __future__ import annotations

import numpy as np
import pytest

# Module-level availability check
try:
    import sam2  # noqa: F401

    HAS_SAM2 = True
except ImportError:
    HAS_SAM2 = False

pytestmark = [pytest.mark.ml, pytest.mark.skipif(not HAS_SAM2, reason="SAM2 package not installed (optional dependency)")]


@pytest.fixture
def test_image():
    """Simple RGB test image."""
    return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8).astype(np.float32) / 255.0


@pytest.fixture
def sam2_backend():
    """SAM2 backend instance (CPU for CI)."""
    from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

    return SAM2Backend(model_size="large", checkpoint_path="checkpoints/sam2.1_hiera_large.pt", device="cpu")


def test_sam2_backend_init():
    """Test SAM2Backend initialization."""
    from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

    backend = SAM2Backend(model_size="large", device="cpu")
    assert backend.model_size == "large"
    assert backend.device == "cpu"
    assert backend.checkpoint_path is not None


def test_sam2_backend_invalid_model_size():
    """Test invalid model size raises ValueError."""
    from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

    with pytest.raises(ValueError, match="Invalid model_size"):
        SAM2Backend(model_size="invalid", device="cpu")


def test_sam2_backend_methods_exist(sam2_backend):
    """Test required methods exist."""
    assert hasattr(sam2_backend, "segment")
    assert hasattr(sam2_backend, "_segment_auto")
    assert hasattr(sam2_backend, "_segment_prompted")
    assert hasattr(sam2_backend, "_segment_video")


@pytest.mark.slow
def test_sam2_auto_mode_shape_validation(sam2_backend, test_image):
    """Test auto mode with shape validation (no actual inference)."""
    from transformation_portal.spatial_ai.segmentation.contracts import SegmentationInput

    seg_input = SegmentationInput(image=test_image, gamma=1.0, mode="auto")

    # This test verifies inputs are accepted
    # Actual inference is slow and tested manually
    assert seg_input.image.shape == (512, 512, 3)
    assert seg_input.gamma == 1.0


@pytest.mark.slow
def test_sam2_prompted_mode_validation(test_image):
    """Test prompted mode input validation."""
    from transformation_portal.spatial_ai.segmentation.contracts import SegmentationInput

    # Test with point prompts
    seg_input = SegmentationInput(
        image=test_image, gamma=1.0, mode="points", prompts={"points": [[100, 100], [200, 200]], "labels": [1, 1]}
    )

    assert seg_input.mode == "points"
    assert "points" in seg_input.prompts
    assert len(seg_input.prompts["points"]) == 2


def test_sam2_video_mode_not_implemented(sam2_backend):
    """Test that video mode method exists (implemented in Phase 4A)."""
    # Video mode was implemented in Phase 4A
    # Verify the method exists and is callable
    assert hasattr(sam2_backend, "_segment_video")
    assert callable(sam2_backend._segment_video)

    # Full video mode testing requires:
    # - decord package
    # - valid video file
    # - video-specific fixtures
    # These are tested in test_sam2_video_integration.py


def test_sam2_checkpoint_path():
    """Test checkpoint path resolution."""
    from pathlib import Path

    from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

    backend = SAM2Backend(model_size="large", device="cpu")
    checkpoint = Path(backend.checkpoint_path)

    # Should resolve to checkpoints/ directory
    assert "checkpoints" in str(checkpoint)
    assert checkpoint.name == "sam2.1_hiera_large.pt"


@pytest.mark.slow
def test_sam2_device_selection():
    """Test device selection logic."""
    from pathlib import Path

    from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

    # Skip if checkpoint not available
    checkpoint_path = Path("checkpoints/sam2.1_hiera_large.pt")
    if not checkpoint_path.exists():
        pytest.skip(f"Checkpoint not found: {checkpoint_path}")

    # CPU should always work - use large model which we know has checkpoint
    backend = SAM2Backend(model_size="large", device="cpu")
    assert backend.device == "cpu"

    # MPS/CUDA will depend on hardware availability
    # (not tested in CI)
