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


def test_sam2_video_mode_requires_video_path():
    """Test video mode requires video_path parameter."""
    from transformation_portal.spatial_ai.segmentation.contracts import SegmentationInput

    # Video mode without video_path should fail validation
    with pytest.raises(ValueError, match="requires video_path"):
        SegmentationInput(
            image=None, gamma=1.0, mode="video", prompts={"frame_idx": 0, "object_id": 1, "points": [[10, 20]], "labels": [1]}
        )
