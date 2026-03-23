"""Tests for Phase 2.3 Reconstruction MVP.

Integration tests for multi-view reconstruction orchestration.
Tests core contracts without requiring full ML stack.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from transformation_portal.core.geometry import CoreCameraParams, MultiViewReconstructionRequest
from transformation_portal.core.geometry.multiview_request import CameraValidationError

pytestmark = pytest.mark.unit


class TestReconstructionMVPContracts:
    """Contract tests for reconstruction MVP (no ML required)."""

    def _make_cameras(self, count: int, sources: list[str] | None = None) -> list:
        """Create test cameras with specified sources."""
        if sources is None:
            sources = ["explicit"] * count
        return [
            CoreCameraParams(
                fx=800.0, fy=800.0, cx=512.0, cy=384.0,
                width=1024, height=768, source=src
            )
            for src in sources
        ]

    def _make_images(self, count: int) -> list:
        """Create test image arrays."""
        return [
            np.ones((768, 1024, 3), dtype=np.float32) * 0.5
            for _ in range(count)
        ]

    # --- View Count Tests ---

    def test_reject_zero_views(self):
        """Zero views rejected."""
        with pytest.raises(ValueError, match="at least 2 views"):
            MultiViewReconstructionRequest(
                cameras=[],
                images=[],
                tier="apex_research",
            )

    def test_reject_one_view(self):
        """Single view rejected."""
        cameras = self._make_cameras(1)
        images = self._make_images(1)

        with pytest.raises(ValueError, match="at least 2 views"):
            MultiViewReconstructionRequest(
                cameras=cameras,
                images=images,
                tier="apex_research",
            )

    def test_accept_two_views(self):
        """Two views accepted (minimum for multi-view)."""
        cameras = self._make_cameras(2)
        images = self._make_images(2)

        request = MultiViewReconstructionRequest(
            cameras=cameras,
            images=images,
            tier="apex_research",
        )

        assert request.num_views == 2

    def test_accept_many_views(self):
        """Many views accepted."""
        cameras = self._make_cameras(10)
        images = self._make_images(10)

        request = MultiViewReconstructionRequest(
            cameras=cameras,
            images=images,
            tier="apex_research",
        )

        assert request.num_views == 10

    # --- Camera/Image Alignment Tests ---

    def test_reject_camera_image_mismatch(self):
        """Camera count must equal image count."""
        cameras = self._make_cameras(3)
        images = self._make_images(2)

        with pytest.raises(ValueError, match="Camera count.*must match"):
            MultiViewReconstructionRequest(
                cameras=cameras,
                images=images,
                tier="apex_research",
            )

    # --- Camera Source Policy Tests ---

    def test_reject_all_synthetic(self):
        """All synthetic cameras rejected by default."""
        cameras = self._make_cameras(2, sources=["synthetic", "synthetic"])
        images = self._make_images(2)

        with pytest.raises(CameraValidationError, match="verified cameras"):
            MultiViewReconstructionRequest(
                cameras=cameras,
                images=images,
                tier="apex_research",
            )

    def test_reject_mixed_with_synthetic(self):
        """Mixed cameras with any synthetic rejected by default."""
        cameras = self._make_cameras(3, sources=["explicit", "exif", "synthetic"])
        images = self._make_images(3)

        with pytest.raises(CameraValidationError, match="Synthetic cameras found"):
            MultiViewReconstructionRequest(
                cameras=cameras,
                images=images,
                tier="apex_research",
            )

    def test_accept_all_explicit(self):
        """All explicit cameras accepted."""
        cameras = self._make_cameras(3, sources=["explicit", "explicit", "explicit"])
        images = self._make_images(3)

        request = MultiViewReconstructionRequest(
            cameras=cameras,
            images=images,
            tier="apex_research",
        )

        assert request.all_cameras_verified
        assert not request.has_synthetic_cameras

    def test_accept_all_exif(self):
        """All EXIF cameras accepted."""
        cameras = self._make_cameras(3, sources=["exif", "exif", "exif"])
        images = self._make_images(3)

        request = MultiViewReconstructionRequest(
            cameras=cameras,
            images=images,
            tier="apex_research",
        )

        assert request.all_cameras_verified

    def test_accept_mixed_explicit_exif(self):
        """Mixed explicit/exif cameras accepted."""
        cameras = self._make_cameras(3, sources=["explicit", "exif", "explicit"])
        images = self._make_images(3)

        request = MultiViewReconstructionRequest(
            cameras=cameras,
            images=images,
            tier="apex_research",
        )

        assert request.all_cameras_verified

    def test_accept_synthetic_with_override(self):
        """Synthetic cameras accepted with override flag."""
        cameras = self._make_cameras(2, sources=["synthetic", "synthetic"])
        images = self._make_images(2)

        request = MultiViewReconstructionRequest(
            cameras=cameras,
            images=images,
            tier="apex_research",
            allow_synthetic_cameras=True,
        )

        assert request.has_synthetic_cameras
        assert not request.all_cameras_verified

    # --- Tier Enforcement Tests ---

    def test_reject_standard_tier(self):
        """Standard tier rejected."""
        cameras = self._make_cameras(2)
        images = self._make_images(2)

        with pytest.raises(ValueError, match="requires research tier"):
            MultiViewReconstructionRequest(
                cameras=cameras,
                images=images,
                tier="standard",
            )

    def test_reject_invalid_tier(self):
        """Invalid tier rejected."""
        cameras = self._make_cameras(2)
        images = self._make_images(2)

        with pytest.raises(ValueError, match="requires research tier"):
            MultiViewReconstructionRequest(
                cameras=cameras,
                images=images,
                tier="commercial",
            )

    def test_accept_apex_research(self):
        """apex_research tier accepted."""
        cameras = self._make_cameras(2)
        images = self._make_images(2)

        request = MultiViewReconstructionRequest(
            cameras=cameras,
            images=images,
            tier="apex_research",
        )

        assert request.tier == "apex_research"

    def test_accept_apex_research_ultra(self):
        """apex_research_ultra tier accepted."""
        cameras = self._make_cameras(2)
        images = self._make_images(2)

        request = MultiViewReconstructionRequest(
            cameras=cameras,
            images=images,
            tier="apex_research_ultra",
        )

        assert request.tier == "apex_research_ultra"

    def test_accept_experimental(self):
        """experimental tier accepted."""
        cameras = self._make_cameras(2)
        images = self._make_images(2)

        request = MultiViewReconstructionRequest(
            cameras=cameras,
            images=images,
            tier="experimental",
        )

        assert request.tier == "experimental"

    # --- Gamma Enforcement Tests ---

    def test_reject_srgb_gamma(self):
        """sRGB gamma rejected."""
        cameras = self._make_cameras(2)
        images = self._make_images(2)

        with pytest.raises(ValueError, match="gamma=1.0"):
            MultiViewReconstructionRequest(
                cameras=cameras,
                images=images,
                tier="apex_research",
                gamma=2.2,
            )

    def test_accept_linear_gamma(self):
        """Linear gamma accepted."""
        cameras = self._make_cameras(2)
        images = self._make_images(2)

        request = MultiViewReconstructionRequest(
            cameras=cameras,
            images=images,
            tier="apex_research",
            gamma=1.0,
        )

        assert abs(request.gamma - 1.0) < 1e-6


class TestReconstructionMVPMetadata:
    """Tests for metadata generation."""

    def _make_cameras(self, count: int) -> list:
        return [
            CoreCameraParams(
                fx=800, fy=800, cx=512, cy=384,
                width=1024, height=768, source="explicit"
            )
            for _ in range(count)
        ]

    def _make_images(self, count: int) -> list:
        return [np.ones((768, 1024, 3), dtype=np.float32) for _ in range(count)]

    def test_camera_source_summary(self):
        """Camera source summary tracks all sources."""
        cameras = [
            CoreCameraParams(fx=800, fy=800, cx=512, cy=384, width=1024, height=768, source="explicit"),
            CoreCameraParams(fx=800, fy=800, cx=512, cy=384, width=1024, height=768, source="exif"),
            CoreCameraParams(fx=800, fy=800, cx=512, cy=384, width=1024, height=768, source="exif"),
        ]
        images = self._make_images(3)

        request = MultiViewReconstructionRequest(
            cameras=cameras,
            images=images,
            tier="apex_research",
        )

        summary = request.get_camera_source_summary()
        assert summary == {"explicit": 1, "exif": 2}

    def test_metadata_dict_completeness(self):
        """Metadata dict contains all expected fields."""
        cameras = self._make_cameras(3)
        images = self._make_images(3)
        depth_maps = [np.ones((768, 1024), dtype=np.float32) for _ in range(3)]
        masks = [np.ones((768, 1024), dtype=bool) for _ in range(3)]

        request = MultiViewReconstructionRequest(
            cameras=cameras,
            images=images,
            depth_maps=depth_maps,
            masks=masks,
            tier="apex_research",
            optimization_seed=42,
        )

        metadata = request.to_metadata_dict()

        # Required fields
        assert metadata["num_views"] == 3
        assert metadata["tier"] == "apex_research"
        assert metadata["gamma"] == 1.0
        assert metadata["has_depth_priors"] is True
        assert metadata["has_segmentation"] is True
        assert metadata["has_materials"] is False
        assert metadata["all_cameras_verified"] is True
        assert metadata["allow_synthetic_cameras"] is False
        assert metadata["optimization_seed"] == 42
        assert "camera_source_summary" in metadata


class TestReconstructionMVPPreset:
    """Tests for reconstruction MVP preset."""

    def test_preset_file_exists(self):
        """MVP preset file exists."""
        preset_path = Path(__file__).parent.parent.parent.parent / "config" / "presets" / "experimental" / "spatial_ai_reconstruction_mvp.yaml"

        # Handle case where test is run from different locations
        if not preset_path.exists():
            preset_path = Path("config/presets/experimental/spatial_ai_reconstruction_mvp.yaml")

        assert preset_path.exists(), f"MVP preset not found at {preset_path}"

    def test_preset_structure(self):
        """MVP preset has required structure."""
        import yaml

        preset_path = Path(__file__).parent.parent.parent.parent / "config" / "presets" / "experimental" / "spatial_ai_reconstruction_mvp.yaml"

        if not preset_path.exists():
            preset_path = Path("config/presets/experimental/spatial_ai_reconstruction_mvp.yaml")

        with open(preset_path) as f:
            preset = yaml.safe_load(f)

        # Required fields
        assert preset["tier"] == "apex_research"
        assert preset["license_restriction"] == "research_only"
        assert "pipeline" in preset
        assert preset["pipeline"]["reconstruction"]["enabled"] is True
        assert preset["pipeline"]["reconstruction"]["export_format"] == "ply"
        assert preset["pipeline"]["reconstruction"]["require_verified_cameras"] is True
        assert preset["pipeline"]["reconstruction"]["allow_synthetic_cameras"] is False
