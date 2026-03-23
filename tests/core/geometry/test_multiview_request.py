"""Tests for MultiViewReconstructionRequest contract (Phase 2.3 MVP).

Tests camera validation, tier enforcement, and view count requirements
for the neutral multi-view reconstruction request contract.
"""

from pathlib import Path

import numpy as np
import pytest

from transformation_portal.core.geometry import CoreCameraParams, MultiViewReconstructionRequest
from transformation_portal.core.geometry.multiview_request import CameraValidationError

pytestmark = pytest.mark.unit


class TestCoreCameraParams:
    """Tests for CoreCameraParams contract."""

    def test_valid_explicit_camera(self):
        """Explicit cameras are valid and verified."""
        camera = CoreCameraParams(
            fx=800.0, fy=800.0, cx=512.0, cy=384.0,
            width=1024, height=768, source="explicit"
        )
        assert camera.is_verified
        assert camera.source == "explicit"

    def test_valid_exif_camera(self):
        """EXIF cameras are valid and verified."""
        camera = CoreCameraParams(
            fx=800.0, fy=800.0, cx=512.0, cy=384.0,
            width=1024, height=768, source="exif"
        )
        assert camera.is_verified
        assert camera.source == "exif"

    def test_valid_synthetic_camera(self):
        """Synthetic cameras are valid but not verified."""
        camera = CoreCameraParams(
            fx=800.0, fy=800.0, cx=512.0, cy=384.0,
            width=1024, height=768, source="synthetic"
        )
        assert not camera.is_verified
        assert camera.source == "synthetic"

    def test_invalid_source_rejected(self):
        """Invalid camera source raises ValueError."""
        with pytest.raises(ValueError, match="source must be"):
            CoreCameraParams(
                fx=800.0, fy=800.0, cx=512.0, cy=384.0,
                width=1024, height=768, source="unknown"
            )

    def test_invalid_focal_length_rejected(self):
        """Negative focal length raises ValueError."""
        with pytest.raises(ValueError, match="Focal lengths must be positive"):
            CoreCameraParams(
                fx=-800.0, fy=800.0, cx=512.0, cy=384.0,
                width=1024, height=768
            )

    def test_invalid_dimensions_rejected(self):
        """Non-positive dimensions raise ValueError."""
        with pytest.raises(ValueError, match="dimensions must be positive"):
            CoreCameraParams(
                fx=800.0, fy=800.0, cx=512.0, cy=384.0,
                width=0, height=768
            )

    def test_intrinsics_tuple(self):
        """to_intrinsics_tuple returns correct values."""
        camera = CoreCameraParams(
            fx=800.0, fy=750.0, cx=512.0, cy=384.0,
            width=1024, height=768
        )
        assert camera.to_intrinsics_tuple() == (800.0, 750.0, 512.0, 384.0)


class TestMultiViewReconstructionRequestValidation:
    """Tests for request contract validation."""

    def _make_cameras(self, count: int, source: str = "explicit") -> list:
        """Create test cameras."""
        return [
            CoreCameraParams(
                fx=800.0, fy=800.0, cx=512.0, cy=384.0,
                width=1024, height=768, source=source
            )
            for _ in range(count)
        ]

    def _make_images(self, count: int) -> list:
        """Create test image arrays."""
        return [
            np.ones((768, 1024, 3), dtype=np.float32) * 0.5
            for _ in range(count)
        ]

    def test_reject_single_view(self):
        """Single-view input is rejected."""
        cameras = self._make_cameras(1)
        images = self._make_images(1)

        with pytest.raises(ValueError, match="at least 2 views"):
            MultiViewReconstructionRequest(
                cameras=cameras,
                images=images,
                tier="apex_research",
            )

    def test_accept_two_views(self):
        """Two-view input is accepted."""
        cameras = self._make_cameras(2)
        images = self._make_images(2)

        request = MultiViewReconstructionRequest(
            cameras=cameras,
            images=images,
            tier="apex_research",
        )

        assert request.num_views == 2

    def test_reject_camera_image_mismatch(self):
        """Camera count must match image count."""
        cameras = self._make_cameras(3)
        images = self._make_images(2)

        with pytest.raises(ValueError, match="Camera count.*must match"):
            MultiViewReconstructionRequest(
                cameras=cameras,
                images=images,
                tier="apex_research",
            )

    def test_reject_synthetic_cameras_by_default(self):
        """Synthetic cameras are rejected by default."""
        cameras = self._make_cameras(2, source="synthetic")
        images = self._make_images(2)

        with pytest.raises(CameraValidationError, match="verified cameras"):
            MultiViewReconstructionRequest(
                cameras=cameras,
                images=images,
                tier="apex_research",
            )

    def test_accept_synthetic_with_override(self):
        """Synthetic cameras accepted with explicit override."""
        cameras = self._make_cameras(2, source="synthetic")
        images = self._make_images(2)

        request = MultiViewReconstructionRequest(
            cameras=cameras,
            images=images,
            tier="apex_research",
            allow_synthetic_cameras=True,
        )

        assert request.has_synthetic_cameras
        assert not request.all_cameras_verified

    def test_accept_explicit_cameras(self):
        """Explicit cameras are accepted."""
        cameras = self._make_cameras(2, source="explicit")
        images = self._make_images(2)

        request = MultiViewReconstructionRequest(
            cameras=cameras,
            images=images,
            tier="apex_research",
        )

        assert request.all_cameras_verified

    def test_accept_exif_cameras(self):
        """EXIF cameras are accepted."""
        cameras = self._make_cameras(2, source="exif")
        images = self._make_images(2)

        request = MultiViewReconstructionRequest(
            cameras=cameras,
            images=images,
            tier="apex_research",
        )

        assert request.all_cameras_verified

    def test_reject_mixed_with_synthetic(self):
        """Mixed cameras with any synthetic are rejected by default."""
        cameras = [
            CoreCameraParams(fx=800, fy=800, cx=512, cy=384, width=1024, height=768, source="explicit"),
            CoreCameraParams(fx=800, fy=800, cx=512, cy=384, width=1024, height=768, source="synthetic"),
        ]
        images = self._make_images(2)

        with pytest.raises(CameraValidationError, match="Synthetic cameras found"):
            MultiViewReconstructionRequest(
                cameras=cameras,
                images=images,
                tier="apex_research",
            )


class TestMultiViewReconstructionRequestTierEnforcement:
    """Tests for tier restriction enforcement."""

    def _make_cameras(self, count: int) -> list:
        return [
            CoreCameraParams(fx=800, fy=800, cx=512, cy=384, width=1024, height=768, source="explicit")
            for _ in range(count)
        ]

    def _make_images(self, count: int) -> list:
        return [np.ones((768, 1024, 3), dtype=np.float32) for _ in range(count)]

    def test_reject_standard_tier(self):
        """Standard tier is rejected (research license required)."""
        cameras = self._make_cameras(2)
        images = self._make_images(2)

        with pytest.raises(ValueError, match="requires research tier"):
            MultiViewReconstructionRequest(
                cameras=cameras,
                images=images,
                tier="standard",
            )

    def test_accept_apex_research_tier(self):
        """apex_research tier is accepted."""
        cameras = self._make_cameras(2)
        images = self._make_images(2)

        request = MultiViewReconstructionRequest(
            cameras=cameras,
            images=images,
            tier="apex_research",
        )

        assert request.tier == "apex_research"

    def test_accept_apex_research_ultra_tier(self):
        """apex_research_ultra tier is accepted."""
        cameras = self._make_cameras(2)
        images = self._make_images(2)

        request = MultiViewReconstructionRequest(
            cameras=cameras,
            images=images,
            tier="apex_research_ultra",
        )

        assert request.tier == "apex_research_ultra"

    def test_accept_experimental_tier(self):
        """experimental tier is accepted."""
        cameras = self._make_cameras(2)
        images = self._make_images(2)

        request = MultiViewReconstructionRequest(
            cameras=cameras,
            images=images,
            tier="experimental",
        )

        assert request.tier == "experimental"


class TestMultiViewReconstructionRequestGammaEnforcement:
    """Tests for gamma=1.0 enforcement."""

    def _make_cameras(self, count: int) -> list:
        return [
            CoreCameraParams(fx=800, fy=800, cx=512, cy=384, width=1024, height=768, source="explicit")
            for _ in range(count)
        ]

    def _make_images(self, count: int) -> list:
        return [np.ones((768, 1024, 3), dtype=np.float32) for _ in range(count)]

    def test_reject_non_linear_gamma(self):
        """Non-linear gamma (e.g., 2.2) is rejected."""
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
        """Linear gamma (1.0) is accepted."""
        cameras = self._make_cameras(2)
        images = self._make_images(2)

        request = MultiViewReconstructionRequest(
            cameras=cameras,
            images=images,
            tier="apex_research",
            gamma=1.0,
        )

        assert request.gamma == 1.0


class TestMultiViewReconstructionRequestInputValidation:
    """Tests for image/path input validation."""

    def _make_cameras(self, count: int) -> list:
        return [
            CoreCameraParams(fx=800, fy=800, cx=512, cy=384, width=1024, height=768, source="explicit")
            for _ in range(count)
        ]

    def test_reject_neither_images_nor_paths(self):
        """Must provide either images or image_paths."""
        cameras = self._make_cameras(2)

        with pytest.raises(ValueError, match="Either image_paths or images"):
            MultiViewReconstructionRequest(
                cameras=cameras,
                tier="apex_research",
            )

    def test_reject_both_images_and_paths(self):
        """Cannot provide both images and image_paths."""
        cameras = self._make_cameras(2)
        images = [np.ones((768, 1024, 3), dtype=np.float32) for _ in range(2)]
        paths = [Path("view1.png"), Path("view2.png")]

        with pytest.raises(ValueError, match="not both"):
            MultiViewReconstructionRequest(
                cameras=cameras,
                images=images,
                image_paths=paths,
                tier="apex_research",
            )

    def test_accept_image_paths(self):
        """image_paths input is accepted."""
        cameras = self._make_cameras(2)
        paths = [Path("view1.png"), Path("view2.png")]

        request = MultiViewReconstructionRequest(
            cameras=cameras,
            image_paths=paths,
            tier="apex_research",
        )

        assert request.num_views == 2
        assert request.image_paths is not None

    def test_accept_image_arrays(self):
        """Image array input is accepted."""
        cameras = self._make_cameras(2)
        images = [np.ones((768, 1024, 3), dtype=np.float32) for _ in range(2)]

        request = MultiViewReconstructionRequest(
            cameras=cameras,
            images=images,
            tier="apex_research",
        )

        assert request.num_views == 2
        assert request.images is not None

    def test_reject_wrong_image_dtype(self):
        """Images must be float32."""
        cameras = self._make_cameras(2)
        images = [np.ones((768, 1024, 3), dtype=np.uint8) for _ in range(2)]

        with pytest.raises(ValueError, match="must be float32"):
            MultiViewReconstructionRequest(
                cameras=cameras,
                images=images,
                tier="apex_research",
            )

    def test_reject_wrong_image_shape(self):
        """Images must be (H, W, 3)."""
        cameras = self._make_cameras(2)
        images = [np.ones((768, 1024), dtype=np.float32) for _ in range(2)]  # Missing channels

        with pytest.raises(ValueError, match="must be.*H, W, 3"):
            MultiViewReconstructionRequest(
                cameras=cameras,
                images=images,
                tier="apex_research",
            )


class TestMultiViewReconstructionRequestMetadata:
    """Tests for metadata and summary methods."""

    def _make_cameras(self, count: int, source: str = "explicit") -> list:
        return [
            CoreCameraParams(fx=800, fy=800, cx=512, cy=384, width=1024, height=768, source=source)
            for _ in range(count)
        ]

    def _make_images(self, count: int) -> list:
        return [np.ones((768, 1024, 3), dtype=np.float32) for _ in range(count)]

    def test_camera_source_summary(self):
        """Camera source summary reports correct counts."""
        cameras = [
            CoreCameraParams(fx=800, fy=800, cx=512, cy=384, width=1024, height=768, source="explicit"),
            CoreCameraParams(fx=800, fy=800, cx=512, cy=384, width=1024, height=768, source="explicit"),
            CoreCameraParams(fx=800, fy=800, cx=512, cy=384, width=1024, height=768, source="exif"),
        ]
        images = self._make_images(3)

        request = MultiViewReconstructionRequest(
            cameras=cameras,
            images=images,
            tier="apex_research",
        )

        summary = request.get_camera_source_summary()
        assert summary == {"explicit": 2, "exif": 1}

    def test_to_metadata_dict(self):
        """Metadata dict contains expected fields."""
        cameras = self._make_cameras(3)
        images = self._make_images(3)
        depth_maps = [np.ones((768, 1024), dtype=np.float32) for _ in range(3)]

        request = MultiViewReconstructionRequest(
            cameras=cameras,
            images=images,
            depth_maps=depth_maps,
            tier="apex_research",
            optimization_seed=42,
        )

        metadata = request.to_metadata_dict()

        assert metadata["num_views"] == 3
        assert metadata["tier"] == "apex_research"
        assert metadata["gamma"] == 1.0
        assert metadata["has_depth_priors"] is True
        assert metadata["has_segmentation"] is False
        assert metadata["has_materials"] is False
        assert metadata["all_cameras_verified"] is True
        assert metadata["optimization_seed"] == 42
        assert metadata["camera_source_summary"] == {"explicit": 3}
