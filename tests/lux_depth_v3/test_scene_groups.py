from pathlib import Path

import pytest

from transformation_portal.lux_depth_v3.scene_groups import (
    CameraParams,
    SceneGroup,
    build_scene_groups,
    generate_synthetic_camera,
)

pytestmark = pytest.mark.unit


def test_scene_groups_default_behavior():
    images = [Path("foo/a.jpg"), Path("bar/a.png")]
    groups = build_scene_groups(images, dataset_root=Path("."), grouping_mode="single")

    assert len(groups) == 2

    assert groups[0].images == (Path("foo/a.jpg"),)
    assert groups[1].images == (Path("bar/a.png"),)

    assert len(groups[0].scene_id) == 12
    assert len(groups[1].scene_id) == 12
    assert groups[0].scene_id != groups[1].scene_id


def test_scene_groups_parent_dir_mode_groups_deterministically():
    images = [
        Path("scene_b/img2.png"),
        Path("scene_a/img2.png"),
        Path("scene_a/img1.png"),
        Path("scene_b/img1.png"),
    ]

    groups = build_scene_groups(images, dataset_root=Path("."), grouping_mode="parent_dir")

    assert len(groups) == 2
    assert groups[0].images == (Path("scene_a/img1.png"), Path("scene_a/img2.png"))
    assert groups[1].images == (Path("scene_b/img1.png"), Path("scene_b/img2.png"))
    assert len(groups[0].scene_id) == 12
    assert len(groups[1].scene_id) == 12
    assert groups[0].scene_id != groups[1].scene_id


# ============================================================================
# ADR-042 Phase B Tests: Camera Parameters and Reconstruction Eligibility
# ============================================================================


class TestCameraParams:
    """Tests for CameraParams dataclass (ADR-042 Phase B)."""

    def test_camera_params_creation(self):
        """Valid camera params should be created successfully."""
        cam = CameraParams(
            image_path=Path("test.jpg"),
            fx=1000.0,
            fy=1000.0,
            cx=320.0,
            cy=240.0,
            width=640,
            height=480,
            source="explicit",
        )
        assert cam.fx == 1000.0
        assert cam.fy == 1000.0
        assert cam.width == 640
        assert cam.height == 480
        assert cam.source == "explicit"

    def test_camera_params_rejects_invalid_focal_length(self):
        """Camera params should reject non-positive focal lengths."""
        with pytest.raises(ValueError, match="Focal lengths must be positive"):
            CameraParams(
                image_path=Path("test.jpg"),
                fx=0.0,
                fy=1000.0,
                cx=320.0,
                cy=240.0,
                width=640,
                height=480,
            )

    def test_camera_params_rejects_invalid_dimensions(self):
        """Camera params should reject non-positive image dimensions."""
        with pytest.raises(ValueError, match="Image dimensions must be positive"):
            CameraParams(
                image_path=Path("test.jpg"),
                fx=1000.0,
                fy=1000.0,
                cx=320.0,
                cy=240.0,
                width=0,
                height=480,
            )

    def test_camera_params_rejects_invalid_source(self):
        """Camera params should reject invalid source values."""
        with pytest.raises(ValueError, match="Camera source must be"):
            CameraParams(
                image_path=Path("test.jpg"),
                fx=1000.0,
                fy=1000.0,
                cx=320.0,
                cy=240.0,
                width=640,
                height=480,
                source="invalid",
            )


class TestSceneGroupPhaseBContract:
    """Tests for SceneGroup Phase B contract (cameras and reconstruction eligibility)."""

    def test_scene_group_with_cameras(self):
        """SceneGroup should accept optional cameras aligned with images."""
        images = (Path("a.jpg"), Path("b.jpg"))
        cameras = tuple(
            CameraParams(
                image_path=img,
                fx=1000.0,
                fy=1000.0,
                cx=320.0,
                cy=240.0,
                width=640,
                height=480,
            )
            for img in images
        )

        group = SceneGroup(scene_id="abc123456789", images=images, cameras=cameras)

        assert group.has_cameras is True
        assert len(group.cameras) == 2
        assert group.num_images == 2

    def test_scene_group_without_cameras(self):
        """SceneGroup should work without cameras (Phase A backward compat)."""
        group = SceneGroup(
            scene_id="abc123456789",
            images=(Path("a.jpg"),),
            cameras=None,
        )

        assert group.has_cameras is False
        assert group.cameras is None

    def test_scene_group_rejects_misaligned_cameras(self):
        """SceneGroup should reject cameras that don't align with images."""
        images = (Path("a.jpg"), Path("b.jpg"))
        cameras = (
            CameraParams(
                image_path=Path("a.jpg"),
                fx=1000.0,
                fy=1000.0,
                cx=320.0,
                cy=240.0,
                width=640,
                height=480,
            ),
        )  # Only 1 camera for 2 images

        with pytest.raises(ValueError, match="cameras must align with images"):
            SceneGroup(scene_id="abc123456789", images=images, cameras=cameras)


class TestReconstructionEligibility:
    """Tests for reconstruction eligibility logic (ADR-042)."""

    def test_single_image_not_eligible(self):
        """Single-image scenes are not eligible for reconstruction."""
        group = SceneGroup(
            scene_id="abc123456789",
            images=(Path("a.jpg"),),
            cameras=(
                CameraParams(
                    image_path=Path("a.jpg"),
                    fx=1000.0,
                    fy=1000.0,
                    cx=320.0,
                    cy=240.0,
                    width=640,
                    height=480,
                ),
            ),
        )

        assert group.is_reconstruction_eligible() is False

    def test_multi_image_without_cameras_not_eligible(self):
        """Multi-image scenes without cameras are not eligible."""
        group = SceneGroup(
            scene_id="abc123456789",
            images=(Path("a.jpg"), Path("b.jpg")),
            cameras=None,
        )

        assert group.is_reconstruction_eligible() is False

    def test_multi_image_with_cameras_is_eligible(self):
        """Multi-image scenes with aligned cameras are eligible."""
        images = (Path("a.jpg"), Path("b.jpg"))
        cameras = tuple(
            CameraParams(
                image_path=img,
                fx=1000.0,
                fy=1000.0,
                cx=320.0,
                cy=240.0,
                width=640,
                height=480,
            )
            for img in images
        )

        group = SceneGroup(scene_id="abc123456789", images=images, cameras=cameras)

        assert group.is_reconstruction_eligible() is True


class TestGenerateSyntheticCamera:
    """Tests for synthetic camera generation fallback."""

    def test_generates_valid_camera(self):
        """generate_synthetic_camera should produce valid CameraParams."""
        cam = generate_synthetic_camera(
            image_path=Path("test.jpg"),
            width=640,
            height=480,
            fov_degrees=60.0,
        )

        assert cam.source == "synthetic"
        assert cam.width == 640
        assert cam.height == 480
        assert cam.fx > 0
        assert cam.fy > 0
        # Centered principal point
        assert cam.cx == 320.0
        assert cam.cy == 240.0

    def test_focal_length_computation(self):
        """Focal length should be computed correctly from FOV."""
        import math

        cam = generate_synthetic_camera(
            image_path=Path("test.jpg"),
            width=1000,
            height=1000,
            fov_degrees=90.0,  # tan(45°) = 1, so fx = 1000 / 2 = 500
        )

        # For 90° FOV: fx = width / (2 * tan(45°)) = width / 2
        expected_fx = 1000 / (2 * math.tan(math.radians(45)))
        assert abs(cam.fx - expected_fx) < 0.01


class TestBuildSceneGroupsWithCameras:
    """Tests for build_scene_groups with camera parameter support."""

    def test_cameras_passed_to_single_groups(self):
        """Cameras should be correctly associated with single-image groups."""
        images = [Path("a.jpg"), Path("b.jpg")]
        cameras = [
            CameraParams(
                image_path=Path("a.jpg"),
                fx=1000.0,
                fy=1000.0,
                cx=320.0,
                cy=240.0,
                width=640,
                height=480,
            ),
            CameraParams(
                image_path=Path("b.jpg"),
                fx=1200.0,
                fy=1200.0,
                cx=400.0,
                cy=300.0,
                width=800,
                height=600,
            ),
        ]

        groups = build_scene_groups(
            images, dataset_root=Path("."), grouping_mode="single", cameras=cameras
        )

        assert len(groups) == 2
        assert groups[0].cameras is not None
        assert len(groups[0].cameras) == 1
        assert groups[0].cameras[0].fx == 1000.0

        assert groups[1].cameras is not None
        assert len(groups[1].cameras) == 1
        assert groups[1].cameras[0].fx == 1200.0

    def test_rejects_misaligned_cameras_at_build(self):
        """build_scene_groups should reject misaligned cameras."""
        images = [Path("a.jpg"), Path("b.jpg")]
        cameras = [
            CameraParams(
                image_path=Path("a.jpg"),
                fx=1000.0,
                fy=1000.0,
                cx=320.0,
                cy=240.0,
                width=640,
                height=480,
            ),
        ]  # Only 1 camera for 2 images

        with pytest.raises(ValueError, match="cameras must align with images"):
            build_scene_groups(
                images, dataset_root=Path("."), grouping_mode="single", cameras=cameras
            )
