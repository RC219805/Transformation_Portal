"""Tests for reconstruction contracts (Phase 2.3)."""

import numpy as np
import pytest

from transformation_portal.spatial_ai.reconstruction.contracts import (
    CameraParams,
    GaussianSplat,
    LicenseRestrictionError,
    ReconstructionInput,
    Scene3D,
)

pytestmark = pytest.mark.unit


class TestCameraParams:
    """Test CameraParams contract."""

    def test_valid_camera(self):
        """Test valid camera parameters."""
        intrinsics = np.array([[525.0, 0, 320.0], [0, 525.0, 240.0], [0, 0, 1.0]], dtype=np.float32)

        extrinsics = np.eye(4, dtype=np.float32)

        cam = CameraParams(intrinsics=intrinsics, extrinsics=extrinsics, width=640, height=480)

        assert cam.width == 640
        assert cam.height == 480
        assert cam.intrinsics.shape == (3, 3)
        assert cam.extrinsics.shape == (4, 4)

    def test_intrinsics_validation_shape(self):
        """Test intrinsics shape validation."""
        intrinsics = np.eye(2, dtype=np.float32)  # Wrong shape
        extrinsics = np.eye(4, dtype=np.float32)

        with pytest.raises(ValueError, match="Intrinsics must be \\(3, 3\\)"):
            CameraParams(intrinsics=intrinsics, extrinsics=extrinsics, width=640, height=480)

    def test_intrinsics_validation_dtype(self):
        """Test intrinsics dtype validation."""
        intrinsics = np.eye(3, dtype=np.float64)  # Wrong dtype
        extrinsics = np.eye(4, dtype=np.float32)

        with pytest.raises(ValueError, match="Intrinsics must be float32"):
            CameraParams(intrinsics=intrinsics, extrinsics=extrinsics, width=640, height=480)

    def test_intrinsics_validation_homogeneous(self):
        """Test intrinsics homogeneous coordinate validation."""
        intrinsics = np.eye(3, dtype=np.float32)
        intrinsics[2, 2] = 0.0  # Invalid
        extrinsics = np.eye(4, dtype=np.float32)

        with pytest.raises(ValueError, match="Intrinsics\\[2,2\\] must be 1.0"):
            CameraParams(intrinsics=intrinsics, extrinsics=extrinsics, width=640, height=480)

    def test_extrinsics_validation_shape(self):
        """Test extrinsics shape validation."""
        intrinsics = np.eye(3, dtype=np.float32)
        extrinsics = np.eye(3, dtype=np.float32)  # Wrong shape

        with pytest.raises(ValueError, match="Extrinsics must be \\(4, 4\\)"):
            CameraParams(intrinsics=intrinsics, extrinsics=extrinsics, width=640, height=480)

    def test_extrinsics_validation_bottom_row(self):
        """Test extrinsics bottom row validation."""
        intrinsics = np.eye(3, dtype=np.float32)
        extrinsics = np.eye(4, dtype=np.float32)
        extrinsics[3, :] = [1, 0, 0, 1]  # Invalid bottom row

        with pytest.raises(ValueError, match="Extrinsics bottom row must be"):
            CameraParams(intrinsics=intrinsics, extrinsics=extrinsics, width=640, height=480)

    def test_negative_dimensions(self):
        """Test negative dimension validation."""
        intrinsics = np.eye(3, dtype=np.float32)
        extrinsics = np.eye(4, dtype=np.float32)

        with pytest.raises(ValueError, match="Image dimensions must be positive"):
            CameraParams(intrinsics=intrinsics, extrinsics=extrinsics, width=-1, height=480)

    def test_distortion_coefficients(self):
        """Test optional distortion coefficients."""
        intrinsics = np.eye(3, dtype=np.float32)
        extrinsics = np.eye(4, dtype=np.float32)
        distortion = np.array([0.1, -0.2, 0.0, 0.0, 0.05], dtype=np.float32)

        cam = CameraParams(intrinsics=intrinsics, extrinsics=extrinsics, width=640, height=480, distortion=distortion)

        assert cam.distortion is not None
        assert len(cam.distortion) == 5

    def test_invalid_distortion_count(self):
        """Test invalid distortion coefficient count."""
        intrinsics = np.eye(3, dtype=np.float32)
        extrinsics = np.eye(4, dtype=np.float32)
        distortion = np.array([0.1, -0.2], dtype=np.float32)  # Only 2 coefficients

        with pytest.raises(ValueError, match="Distortion must have 4, 5, or 8 coefficients"):
            CameraParams(intrinsics=intrinsics, extrinsics=extrinsics, width=640, height=480, distortion=distortion)


class TestGaussianSplat:
    """Test GaussianSplat contract."""

    def test_valid_splat(self):
        """Test valid Gaussian splat."""
        N = 100
        positions = np.random.randn(N, 3).astype(np.float32)
        colors = np.random.rand(N, 3).astype(np.float32)
        scales = np.random.rand(N, 3).astype(np.float32) * 0.1
        rotations = np.zeros((N, 4), dtype=np.float32)
        rotations[:, 0] = 1.0  # Identity quaternions
        opacities = np.random.rand(N, 1).astype(np.float32)

        splat = GaussianSplat(positions=positions, colors=colors, scales=scales, rotations=rotations, opacities=opacities)

        assert splat.num_gaussians == N

    def test_positions_validation_shape(self):
        """Test positions shape validation."""
        positions = np.random.randn(100, 2).astype(np.float32)  # Wrong shape
        colors = np.random.rand(100, 3).astype(np.float32)
        scales = np.ones((100, 3), dtype=np.float32)
        rotations = np.zeros((100, 4), dtype=np.float32)
        rotations[:, 0] = 1.0
        opacities = np.ones((100, 1), dtype=np.float32)

        with pytest.raises(ValueError, match="Positions must be \\(N, 3\\)"):
            GaussianSplat(positions=positions, colors=colors, scales=scales, rotations=rotations, opacities=opacities)

    def test_colors_range_validation(self):
        """Test colors value range validation."""
        N = 100
        positions = np.zeros((N, 3), dtype=np.float32)
        colors = np.random.rand(N, 3).astype(np.float32) * 2.0  # Out of range
        scales = np.ones((N, 3), dtype=np.float32)
        rotations = np.zeros((N, 4), dtype=np.float32)
        rotations[:, 0] = 1.0
        opacities = np.ones((N, 1), dtype=np.float32)

        with pytest.raises(ValueError, match="Colors must be in \\[0, 1\\]"):
            GaussianSplat(positions=positions, colors=colors, scales=scales, rotations=rotations, opacities=opacities)

    def test_scales_positive_validation(self):
        """Test scales positive value validation."""
        N = 100
        positions = np.zeros((N, 3), dtype=np.float32)
        colors = np.ones((N, 3), dtype=np.float32) * 0.5
        scales = np.ones((N, 3), dtype=np.float32)
        scales[0, 0] = -0.1  # Negative scale
        rotations = np.zeros((N, 4), dtype=np.float32)
        rotations[:, 0] = 1.0
        opacities = np.ones((N, 1), dtype=np.float32)

        with pytest.raises(ValueError, match="Scales must be positive"):
            GaussianSplat(positions=positions, colors=colors, scales=scales, rotations=rotations, opacities=opacities)

    def test_rotations_normalization_validation(self):
        """Test rotation quaternion normalization."""
        N = 100
        positions = np.zeros((N, 3), dtype=np.float32)
        colors = np.ones((N, 3), dtype=np.float32) * 0.5
        scales = np.ones((N, 3), dtype=np.float32)
        rotations = np.ones((N, 4), dtype=np.float32)  # Not normalized
        opacities = np.ones((N, 1), dtype=np.float32)

        with pytest.raises(ValueError, match="Rotation quaternions must be normalized"):
            GaussianSplat(positions=positions, colors=colors, scales=scales, rotations=rotations, opacities=opacities)

    def test_opacities_range_validation(self):
        """Test opacities value range validation."""
        N = 100
        positions = np.zeros((N, 3), dtype=np.float32)
        colors = np.ones((N, 3), dtype=np.float32) * 0.5
        scales = np.ones((N, 3), dtype=np.float32)
        rotations = np.zeros((N, 4), dtype=np.float32)
        rotations[:, 0] = 1.0
        opacities = np.ones((N, 1), dtype=np.float32) * 1.5  # Out of range

        with pytest.raises(ValueError, match="Opacities must be in \\[0, 1\\]"):
            GaussianSplat(positions=positions, colors=colors, scales=scales, rotations=rotations, opacities=opacities)

    def test_sh_coefficients_validation(self):
        """Test spherical harmonics coefficients validation."""
        N = 100
        positions = np.zeros((N, 3), dtype=np.float32)
        colors = np.ones((N, 3), dtype=np.float32) * 0.5
        scales = np.ones((N, 3), dtype=np.float32)
        rotations = np.zeros((N, 4), dtype=np.float32)
        rotations[:, 0] = 1.0
        opacities = np.ones((N, 1), dtype=np.float32)
        sh_coefficients = np.random.randn(N, 16, 3).astype(np.float32)  # 3rd order SH

        splat = GaussianSplat(
            positions=positions,
            colors=colors,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
            sh_coefficients=sh_coefficients,
        )

        assert splat.sh_coefficients is not None
        assert splat.sh_coefficients.shape == (N, 16, 3)


class TestReconstructionInput:
    """Test ReconstructionInput contract."""

    def test_valid_input(self):
        """Test valid reconstruction input."""
        images = [np.random.rand(480, 640, 3).astype(np.float32) for _ in range(3)]
        intrinsics = np.eye(3, dtype=np.float32)
        extrinsics = np.eye(4, dtype=np.float32)
        cameras = [CameraParams(intrinsics, extrinsics, 640, 480) for _ in range(3)]

        reconstruction_input = ReconstructionInput(images=images, gamma=1.0, cameras=cameras, tier="apex_research")

        assert reconstruction_input.num_views == 3

    def test_gamma_enforcement(self):
        """Test gamma=1.0 enforcement (SpatialCaptureV1 contract)."""
        images = [np.random.rand(480, 640, 3).astype(np.float32)]
        intrinsics = np.eye(3, dtype=np.float32)
        extrinsics = np.eye(4, dtype=np.float32)
        cameras = [CameraParams(intrinsics, extrinsics, 640, 480)]

        with pytest.raises(ValueError, match="Reconstruction requires gamma=1.0"):
            ReconstructionInput(images=images, gamma=2.2, cameras=cameras, tier="apex_research")

    def test_tier_restriction(self):
        """Test tier restriction enforcement (Inria license)."""
        images = [np.random.rand(480, 640, 3).astype(np.float32) for _ in range(2)]
        intrinsics = np.eye(3, dtype=np.float32)
        extrinsics = np.eye(4, dtype=np.float32)
        cameras = [CameraParams(intrinsics, extrinsics, 640, 480) for _ in range(2)]

        with pytest.raises(LicenseRestrictionError, match="3D Gaussian Splatting requires research tier"):
            ReconstructionInput(images=images, gamma=1.0, cameras=cameras, tier="commercial")

    def test_minimum_views_validation(self):
        """Test minimum views validation."""
        images = [np.random.rand(480, 640, 3).astype(np.float32)]  # Only 1 view
        intrinsics = np.eye(3, dtype=np.float32)
        extrinsics = np.eye(4, dtype=np.float32)
        cameras = [CameraParams(intrinsics, extrinsics, 640, 480)]

        with pytest.raises(ValueError, match="Reconstruction requires at least 2 views"):
            ReconstructionInput(images=images, gamma=1.0, cameras=cameras, tier="apex_research")

    def test_camera_count_mismatch(self):
        """Test camera count mismatch validation."""
        images = [np.random.rand(480, 640, 3).astype(np.float32) for _ in range(3)]
        intrinsics = np.eye(3, dtype=np.float32)
        extrinsics = np.eye(4, dtype=np.float32)
        cameras = [CameraParams(intrinsics, extrinsics, 640, 480) for _ in range(2)]  # Mismatch

        with pytest.raises(ValueError, match="Number of cameras .* must match number of images"):
            ReconstructionInput(images=images, gamma=1.0, cameras=cameras, tier="apex_research")

    def test_depth_maps_validation(self):
        """Test depth maps validation."""
        images = [np.random.rand(480, 640, 3).astype(np.float32) for _ in range(2)]
        depth_maps = [np.random.rand(480, 640).astype(np.float32) for _ in range(2)]
        intrinsics = np.eye(3, dtype=np.float32)
        extrinsics = np.eye(4, dtype=np.float32)
        cameras = [CameraParams(intrinsics, extrinsics, 640, 480) for _ in range(2)]

        reconstruction_input = ReconstructionInput(
            images=images, gamma=1.0, cameras=cameras, depth_maps=depth_maps, tier="apex_research"
        )

        assert reconstruction_input.depth_maps is not None

    def test_masks_validation(self):
        """Test segmentation masks validation."""
        images = [np.random.rand(480, 640, 3).astype(np.float32) for _ in range(2)]
        masks = [np.random.rand(480, 640) > 0.5 for _ in range(2)]  # Boolean masks
        intrinsics = np.eye(3, dtype=np.float32)
        extrinsics = np.eye(4, dtype=np.float32)
        cameras = [CameraParams(intrinsics, extrinsics, 640, 480) for _ in range(2)]

        reconstruction_input = ReconstructionInput(
            images=images, gamma=1.0, cameras=cameras, masks=masks, tier="apex_research"
        )

        assert reconstruction_input.masks is not None


class TestScene3D:
    """Test Scene3D contract."""

    def test_valid_scene(self):
        """Test valid 3D scene."""
        N = 100
        positions = np.random.randn(N, 3).astype(np.float32)
        colors = np.random.rand(N, 3).astype(np.float32)
        scales = np.ones((N, 3), dtype=np.float32) * 0.1
        rotations = np.zeros((N, 4), dtype=np.float32)
        rotations[:, 0] = 1.0
        opacities = np.ones((N, 1), dtype=np.float32) * 0.5

        splats = GaussianSplat(positions=positions, colors=colors, scales=scales, rotations=rotations, opacities=opacities)

        intrinsics = np.eye(3, dtype=np.float32)
        extrinsics = np.eye(4, dtype=np.float32)
        cameras = [CameraParams(intrinsics, extrinsics, 640, 480) for _ in range(2)]

        scene = Scene3D(splats=splats, cameras=cameras, rmse=0.015, iteration=30000, convergence="converged")

        assert scene.is_converged
        assert scene.quality_score > 85  # Good quality

    def test_quality_score_excellent(self):
        """Test quality score for excellent RMSE."""
        N = 10
        splats = GaussianSplat(
            positions=np.zeros((N, 3), dtype=np.float32),
            colors=np.ones((N, 3), dtype=np.float32) * 0.5,
            scales=np.ones((N, 3), dtype=np.float32),
            rotations=np.tile([1, 0, 0, 0], (N, 1)).astype(np.float32),
            opacities=np.ones((N, 1), dtype=np.float32),
        )

        intrinsics = np.eye(3, dtype=np.float32)
        extrinsics = np.eye(4, dtype=np.float32)
        cameras = [CameraParams(intrinsics, extrinsics, 640, 480) for _ in range(2)]

        scene = Scene3D(splats=splats, cameras=cameras, rmse=0.005, iteration=30000, convergence="converged")

        assert scene.quality_score >= 95  # Excellent

    def test_quality_score_poor(self):
        """Test quality score for poor RMSE."""
        N = 10
        splats = GaussianSplat(
            positions=np.zeros((N, 3), dtype=np.float32),
            colors=np.ones((N, 3), dtype=np.float32) * 0.5,
            scales=np.ones((N, 3), dtype=np.float32),
            rotations=np.tile([1, 0, 0, 0], (N, 1)).astype(np.float32),
            opacities=np.ones((N, 1), dtype=np.float32),
        )

        intrinsics = np.eye(3, dtype=np.float32)
        extrinsics = np.eye(4, dtype=np.float32)
        cameras = [CameraParams(intrinsics, extrinsics, 640, 480) for _ in range(2)]

        scene = Scene3D(splats=splats, cameras=cameras, rmse=0.1, iteration=30000, convergence="diverged")

        assert scene.quality_score < 70  # Poor
        assert not scene.is_converged

    def test_negative_rmse_validation(self):
        """Test negative RMSE validation."""
        N = 10
        splats = GaussianSplat(
            positions=np.zeros((N, 3), dtype=np.float32),
            colors=np.ones((N, 3), dtype=np.float32) * 0.5,
            scales=np.ones((N, 3), dtype=np.float32),
            rotations=np.tile([1, 0, 0, 0], (N, 1)).astype(np.float32),
            opacities=np.ones((N, 1), dtype=np.float32),
        )

        intrinsics = np.eye(3, dtype=np.float32)
        extrinsics = np.eye(4, dtype=np.float32)
        cameras = [CameraParams(intrinsics, extrinsics, 640, 480) for _ in range(2)]

        with pytest.raises(ValueError, match="RMSE must be non-negative"):
            Scene3D(splats=splats, cameras=cameras, rmse=-0.01, iteration=30000, convergence="converged")

    def test_minimum_cameras_validation(self):
        """Test minimum cameras validation."""
        N = 10
        splats = GaussianSplat(
            positions=np.zeros((N, 3), dtype=np.float32),
            colors=np.ones((N, 3), dtype=np.float32) * 0.5,
            scales=np.ones((N, 3), dtype=np.float32),
            rotations=np.tile([1, 0, 0, 0], (N, 1)).astype(np.float32),
            opacities=np.ones((N, 1), dtype=np.float32),
        )

        intrinsics = np.eye(3, dtype=np.float32)
        extrinsics = np.eye(4, dtype=np.float32)
        cameras = [CameraParams(intrinsics, extrinsics, 640, 480)]  # Only 1 camera

        with pytest.raises(ValueError, match="Scene requires at least 2 camera views"):
            Scene3D(splats=splats, cameras=cameras, rmse=0.01, iteration=30000, convergence="converged")
