"""Tests for SAM2 segmentation backend (Phase 3 - Direct Checkpoint Loading)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# Module-level availability check
try:
    import sam2  # noqa: F401

    HAS_SAM2 = True
except ImportError:
    HAS_SAM2 = False

pytestmark = pytest.mark.skipif(not HAS_SAM2, reason="SAM2 package not installed (optional dependency)")


def test_sam2_backend_import():
    """Test SAM2Backend can be imported."""
    from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

    assert SAM2Backend is not None


def test_sam2_backend_invalid_model_size():
    """Test invalid model size raises ValueError."""
    from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

    with pytest.raises(ValueError, match="Invalid model_size"):
        SAM2Backend(model_size="invalid", device="cpu")


def test_sam2_video_mode_not_implemented():
    """Test video mode raises NotImplementedError (Phase 4)."""
    from transformation_portal.spatial_ai.segmentation.contracts import SegmentationInput
    from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

    checkpoint_path = "checkpoints/sam2_hiera_large.pt"
    if not Path(checkpoint_path).exists():
        pytest.skip(f"Checkpoint not found: {checkpoint_path}")

    backend = SAM2Backend(model_size="large", checkpoint_path=checkpoint_path, device="cpu")

    test_image = np.random.rand(512, 512, 3).astype(np.float32)
    seg_input = SegmentationInput(image=test_image, gamma=1.0, mode="video")

    with pytest.raises(NotImplementedError, match="Video tracking.*Phase 4"):
        backend._segment_video(seg_input)
