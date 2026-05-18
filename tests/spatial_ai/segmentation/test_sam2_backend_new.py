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

pytestmark = [pytest.mark.ml, pytest.mark.skipif(not HAS_SAM2, reason="SAM2 package not installed (optional dependency)")]


def test_sam2_backend_import():
    """Test SAM2Backend can be imported."""
    from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

    assert SAM2Backend is not None


def test_sam2_backend_init_requires_checkpoint(tmp_path):
    """Test SAM2Backend requires checkpoint file."""
    from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

    missing_checkpoint = tmp_path / "missing_sam2_checkpoint.pt"

    # An explicitly missing checkpoint should raise FileNotFoundError.
    with pytest.raises(FileNotFoundError, match="SAM2 checkpoint not found"):
        SAM2Backend(model_size="base", checkpoint_path=str(missing_checkpoint), device="cpu")


def test_sam2_backend_init_with_checkpoint():
    """Test SAM2Backend initializes with valid checkpoint."""
    from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

    checkpoint_path = "checkpoints/sam2.1_hiera_large.pt"
    if not Path(checkpoint_path).exists():
        pytest.skip(f"Checkpoint not found: {checkpoint_path}")

    backend = SAM2Backend(model_size="large", checkpoint_path=checkpoint_path, device="cpu")
    assert backend.model_size == "large"
    assert backend.device == "cpu"
    assert Path(backend.checkpoint_path).exists()


def test_sam2_backend_invalid_model_size():
    """Test invalid model size raises ValueError."""
    from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

    with pytest.raises(ValueError, match="Invalid model_size"):
        SAM2Backend(model_size="invalid", device="cpu")


def test_sam2_video_mode_not_implemented():
    """Test video mode raises NotImplementedError (Phase 4)."""
    from transformation_portal.spatial_ai.segmentation.contracts import SegmentationInput
    from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

    checkpoint_path = "checkpoints/sam2.1_hiera_large.pt"
    if not Path(checkpoint_path).exists():
        pytest.skip(f"Checkpoint not found: {checkpoint_path}")

    backend = SAM2Backend(model_size="large", checkpoint_path=checkpoint_path, device="cpu")

    # Video mode was implemented in Phase 4A
    # Verify the method exists and is callable
    assert hasattr(backend, "_segment_video")
    assert callable(backend._segment_video)

    # Full video mode testing requires decord and valid video files
    # These are tested in test_sam2_video_integration.py


def test_sam2_checkpoint_path_resolution():
    """Test checkpoint path resolution logic."""
    from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

    # Test default checkpoint path for large model
    expected_default = Path("checkpoints/sam2.1_hiera_large.pt")

    # If checkpoint exists, test initialization
    if expected_default.exists():
        backend = SAM2Backend(model_size="large", device="cpu")
        assert Path(backend.checkpoint_path) == expected_default
