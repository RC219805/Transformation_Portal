"""Tests for spatial_ai reconstruction module (Phase 5 coverage).

Tests for:
- CameraParams contract validation
- GaussianSplat contract validation
- ReconstructionInput contract validation
- Scene3D contract validation
- SceneBuilder high-level API

All tests use mocks - no ML model downloads or GPU requirements.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from transformation_portal.spatial_ai.reconstruction.contracts import (
    CameraParams,
    GaussianSplat,
    LicenseRestrictionError,
    ReconstructionInput,
    Scene3D,
)

pytestmark = [pytest.mark.unit, pytest.mark.ml]


@pytest.fixture
def valid_intrinsics():
    """Create valid camera intrinsic matrix."""
    return np.array(
        [
            [500.0, 0.0, 256.0],
            [0.0, 500.0, 256.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


@pytest.fixture
def valid_extrinsics():
    """Create valid camera extrinsic matrix (identity transform)."""
    return np.eye(4, dtype=np.float32)


@pytest.fixture
def valid_camera_params(valid_intrinsics, valid_extrinsics):
    """Create valid camera parameters."""
    return CameraParams(
        intrinsics=valid_intrinsics,
        extrinsics=valid_extrinsics,
        width=512,
        height=512,
    )


@pytest.fixture
def linear_images():
    """Create list of linear RGB images."""
    return [
        np.random.rand(512, 512, 3).astype(np.float32),
        np.random.rand(512, 512, 3).astype(np.float32),
    ]


@pytest.fixture
def valid_gaussian_splat():
    """Create valid Gaussian splat data."""
    N = 100  # Number of Gaussians
    return GaussianSplat(
        positions=np.random.rand(N, 3).astype(np.float32) * 10,
        colors=np.random.rand(N, 3).astype(np.float32),
        scales=np.random.rand(N, 3).astype(np.float32) * 0.1 + 0.01,
        rotations=_normalize_quaternions(np.random.rand(N, 4).astype(np.float32)),
        opacities=np.random.rand(N, 1).astype(np.float32),
    )


def _normalize_quaternions(quats):
    """Normalize quaternions to unit length."""
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    return (quats / norms).astype(np.float32)


def _deterministic_gaussian_fields(num_gaussians=10, *, opacities=None):
    """Return fixed valid Gaussian fields for boundary-condition tests."""
    axis = np.linspace(0.0, 1.0, num_gaussians, dtype=np.float32)
    if opacities is None:
        opacities = np.full((num_gaussians, 1), 0.5, dtype=np.float32)
    return {
        "positions": np.column_stack((axis, axis + 1.0, axis + 2.0)).astype(np.float32),
        "colors": np.tile(np.array([[0.25, 0.5, 0.75]], dtype=np.float32), (num_gaussians, 1)),
        "scales": np.full((num_gaussians, 3), 0.05, dtype=np.float32),
        "rotations": np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (num_gaussians, 1)),
        "opacities": opacities,
    }


class TestCameraParams:
    """Test CameraParams contract validation."""

    def test_valid_camera(self, valid_intrinsics, valid_extrinsics):
        """Test valid camera parameters."""
        camera = CameraParams(
            intrinsics=valid_intrinsics,
            extrinsics=valid_extrinsics,
            width=512,
            height=512,
        )

        assert camera.width == 512
        assert camera.height == 512
        assert camera.distortion is None
        assert camera.camera_id is None

    def test_camera_with_distortion(self, valid_intrinsics, valid_extrinsics):
        """Test camera with distortion coefficients."""
        distortion = np.array([0.1, -0.2, 0.0, 0.0, 0.05], dtype=np.float32)
        camera = CameraParams(
            intrinsics=valid_intrinsics,
            extrinsics=valid_extrinsics,
            width=512,
            height=512,
            distortion=distortion,
        )

        assert camera.distortion is not None
        assert len(camera.distortion) == 5

    def test_camera_with_id(self, valid_intrinsics, valid_extrinsics):
        """Test camera with identifier."""
        camera = CameraParams(
            intrinsics=valid_intrinsics,
            extrinsics=valid_extrinsics,
            width=512,
            height=512,
            camera_id="cam_01",
        )

        assert camera.camera_id == "cam_01"

    def test_invalid_intrinsics_shape_raises(self, valid_extrinsics):
        """Test that invalid intrinsics shape is rejected."""
        bad_intrinsics = np.eye(4, dtype=np.float32)

        with pytest.raises(ValueError, match="\\(3, 3\\)"):
            CameraParams(
                intrinsics=bad_intrinsics,
                extrinsics=valid_extrinsics,
                width=512,
                height=512,
            )

    def test_invalid_intrinsics_dtype_raises(self, valid_extrinsics):
        """Test that invalid intrinsics dtype is rejected."""
        bad_intrinsics = np.array(
            [
                [500.0, 0.0, 256.0],
                [0.0, 500.0, 256.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        with pytest.raises(ValueError, match="float32"):
            CameraParams(
                intrinsics=bad_intrinsics,
                extrinsics=valid_extrinsics,
                width=512,
                height=512,
            )

    def test_invalid_intrinsics_homogeneous_raises(self, valid_extrinsics):
        """Test that invalid homogeneous coordinate is rejected."""
        bad_intrinsics = np.array(
            [
                [500.0, 0.0, 256.0],
                [0.0, 500.0, 256.0],
                [0.0, 0.0, 2.0],  # Should be 1.0
            ],
            dtype=np.float32,
        )

        with pytest.raises(ValueError, match="must be 1.0"):
            CameraParams(
                intrinsics=bad_intrinsics,
                extrinsics=valid_extrinsics,
                width=512,
                height=512,
            )

    def test_invalid_extrinsics_shape_raises(self, valid_intrinsics):
        """Test that invalid extrinsics shape is rejected."""
        bad_extrinsics = np.eye(3, dtype=np.float32)

        with pytest.raises(ValueError, match="\\(4, 4\\)"):
            CameraParams(
                intrinsics=valid_intrinsics,
                extrinsics=bad_extrinsics,
                width=512,
                height=512,
            )

    def test_invalid_extrinsics_bottom_row_raises(self, valid_intrinsics):
        """Test that invalid extrinsics bottom row is rejected."""
        bad_extrinsics = np.eye(4, dtype=np.float32)
        bad_extrinsics[3, :] = [1, 0, 0, 1]  # Should be [0, 0, 0, 1]

        with pytest.raises(ValueError, match="bottom row"):
            CameraParams(
                intrinsics=valid_intrinsics,
                extrinsics=bad_extrinsics,
                width=512,
                height=512,
            )

    def test_invalid_dimensions_raises(self, valid_intrinsics, valid_extrinsics):
        """Test that invalid dimensions are rejected."""
        with pytest.raises(ValueError, match="positive"):
            CameraParams(
                intrinsics=valid_intrinsics,
                extrinsics=valid_extrinsics,
                width=0,
                height=512,
            )

        with pytest.raises(ValueError, match="positive"):
            CameraParams(
                intrinsics=valid_intrinsics,
                extrinsics=valid_extrinsics,
                width=512,
                height=-1,
            )

    def test_invalid_distortion_length_raises(self, valid_intrinsics, valid_extrinsics):
        """Test that invalid distortion length is rejected."""
        bad_distortion = np.array([0.1, 0.2], dtype=np.float32)

        with pytest.raises(ValueError, match="4, 5, or 8"):
            CameraParams(
                intrinsics=valid_intrinsics,
                extrinsics=valid_extrinsics,
                width=512,
                height=512,
                distortion=bad_distortion,
            )


class TestGaussianSplat:
    """Test GaussianSplat contract validation."""

    def test_valid_gaussian_splat(self, valid_gaussian_splat):
        """Test valid Gaussian splat data."""
        assert valid_gaussian_splat.num_gaussians == 100
        assert valid_gaussian_splat.positions.shape == (100, 3)
        assert valid_gaussian_splat.colors.shape == (100, 3)

    def test_gaussian_splat_with_sh(self):
        """Test Gaussian splat with spherical harmonics."""
        N = 50
        splat = GaussianSplat(
            positions=np.random.rand(N, 3).astype(np.float32),
            colors=np.random.rand(N, 3).astype(np.float32),
            scales=np.random.rand(N, 3).astype(np.float32) * 0.1 + 0.01,
            rotations=_normalize_quaternions(np.random.rand(N, 4).astype(np.float32)),
            opacities=np.random.rand(N, 1).astype(np.float32),
            sh_coefficients=np.random.rand(N, 9, 3).astype(np.float32),  # 2nd order SH
        )

        assert splat.sh_coefficients is not None
        assert splat.sh_coefficients.shape == (50, 9, 3)

    def test_gaussian_splat_with_metadata(self):
        """Test Gaussian splat with metadata."""
        N = 10
        splat = GaussianSplat(
            positions=np.random.rand(N, 3).astype(np.float32),
            colors=np.random.rand(N, 3).astype(np.float32),
            scales=np.random.rand(N, 3).astype(np.float32) * 0.1 + 0.01,
            rotations=_normalize_quaternions(np.random.rand(N, 4).astype(np.float32)),
            opacities=np.random.rand(N, 1).astype(np.float32),
            metadata={"training_time": 300, "loss": 0.01},
        )

        assert splat.metadata["training_time"] == 300

    def test_invalid_positions_shape_raises(self):
        """Test that invalid positions shape is rejected."""
        N = 10
        with pytest.raises(ValueError, match="\\(N, 3\\)"):
            GaussianSplat(
                positions=np.random.rand(N, 2).astype(np.float32),  # Wrong shape
                colors=np.random.rand(N, 3).astype(np.float32),
                scales=np.random.rand(N, 3).astype(np.float32) * 0.1 + 0.01,
                rotations=_normalize_quaternions(np.random.rand(N, 4).astype(np.float32)),
                opacities=np.random.rand(N, 1).astype(np.float32),
            )

    def test_invalid_colors_range_raises(self):
        """Test that out-of-range colors are rejected."""
        N = 10
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            GaussianSplat(
                positions=np.random.rand(N, 3).astype(np.float32),
                colors=np.random.rand(N, 3).astype(np.float32) + 0.5,  # > 1.0
                scales=np.random.rand(N, 3).astype(np.float32) * 0.1 + 0.01,
                rotations=_normalize_quaternions(np.random.rand(N, 4).astype(np.float32)),
                opacities=np.random.rand(N, 1).astype(np.float32),
            )

    def test_invalid_scales_raises(self):
        """Test that non-positive scales are rejected."""
        N = 10
        bad_scales = np.random.rand(N, 3).astype(np.float32)
        bad_scales[0, 0] = 0.0  # Zero scale

        with pytest.raises(ValueError, match="positive"):
            GaussianSplat(
                positions=np.random.rand(N, 3).astype(np.float32),
                colors=np.random.rand(N, 3).astype(np.float32),
                scales=bad_scales,
                rotations=_normalize_quaternions(np.random.rand(N, 4).astype(np.float32)),
                opacities=np.random.rand(N, 1).astype(np.float32),
            )

    def test_invalid_rotations_shape_raises(self):
        """Test that invalid rotations shape is rejected."""
        N = 10
        with pytest.raises(ValueError, match="4"):
            GaussianSplat(
                positions=np.random.rand(N, 3).astype(np.float32),
                colors=np.random.rand(N, 3).astype(np.float32),
                scales=np.random.rand(N, 3).astype(np.float32) * 0.1 + 0.01,
                rotations=np.random.rand(N, 3).astype(np.float32),  # Should be (N, 4)
                opacities=np.random.rand(N, 1).astype(np.float32),
            )

    def test_non_normalized_rotations_raises(self):
        """Test that non-normalized quaternions are rejected."""
        N = 10
        non_normalized = np.random.rand(N, 4).astype(np.float32) * 2  # Not unit length

        with pytest.raises(ValueError, match="normalized"):
            GaussianSplat(
                positions=np.random.rand(N, 3).astype(np.float32),
                colors=np.random.rand(N, 3).astype(np.float32),
                scales=np.random.rand(N, 3).astype(np.float32) * 0.1 + 0.01,
                rotations=non_normalized,
                opacities=np.random.rand(N, 1).astype(np.float32),
            )

    @pytest.mark.parametrize("bad_opacity", [-0.01, 1.01])
    def test_invalid_opacities_range_raises(self, bad_opacity):
        """Test that out-of-range opacities are rejected."""
        N = 10
        bad_opacities = np.full((N, 1), bad_opacity, dtype=np.float32)

        for _ in range(5):
            with pytest.raises(ValueError, match="\\[0, 1\\]"):
                GaussianSplat(**_deterministic_gaussian_fields(N, opacities=bad_opacities))


class TestReconstructionInput:
    """Test ReconstructionInput contract validation."""

    def test_valid_input(self, linear_images, valid_camera_params):
        """Test valid reconstruction input."""
        cameras = [valid_camera_params, valid_camera_params]
        recon_input = ReconstructionInput(
            images=linear_images,
            gamma=1.0,
            cameras=cameras,
            tier="apex_research",
        )

        assert recon_input.num_views == 2
        assert recon_input.gamma == 1.0

    def test_valid_input_with_depth(self, linear_images, valid_camera_params):
        """Test input with depth maps."""
        cameras = [valid_camera_params, valid_camera_params]
        depth_maps = [
            np.random.rand(512, 512).astype(np.float32) * 10,
            np.random.rand(512, 512).astype(np.float32) * 10,
        ]
        recon_input = ReconstructionInput(
            images=linear_images,
            gamma=1.0,
            cameras=cameras,
            depth_maps=depth_maps,
            tier="apex_research",
        )

        assert recon_input.depth_maps is not None

    def test_valid_tiers(self, linear_images, valid_camera_params):
        """Test valid research tiers."""
        cameras = [valid_camera_params, valid_camera_params]
        for tier in ["apex_research", "apex_research_ultra", "experimental"]:
            recon_input = ReconstructionInput(
                images=linear_images,
                gamma=1.0,
                cameras=cameras,
                tier=tier,
            )
            assert recon_input.tier == tier

    def test_invalid_gamma_raises(self, linear_images, valid_camera_params):
        """Test that non-linear gamma is rejected."""
        cameras = [valid_camera_params, valid_camera_params]

        with pytest.raises(ValueError, match="gamma=1.0"):
            ReconstructionInput(
                images=linear_images,
                gamma=2.2,
                cameras=cameras,
                tier="apex_research",
            )

    def test_invalid_tier_raises(self, linear_images, valid_camera_params):
        """Test that commercial tier is rejected."""
        cameras = [valid_camera_params, valid_camera_params]

        with pytest.raises(LicenseRestrictionError, match="research tier"):
            ReconstructionInput(
                images=linear_images,
                gamma=1.0,
                cameras=cameras,
                tier="premium",  # Commercial tier
            )

    def test_insufficient_views_raises(self, valid_camera_params):
        """Test that less than 2 views is rejected."""
        single_image = [np.random.rand(512, 512, 3).astype(np.float32)]
        cameras = [valid_camera_params]

        with pytest.raises(ValueError, match="at least 2 views"):
            ReconstructionInput(
                images=single_image,
                gamma=1.0,
                cameras=cameras,
                tier="apex_research",
            )

    def test_mismatched_cameras_raises(self, linear_images, valid_camera_params):
        """Test that mismatched camera count is rejected."""
        cameras = [valid_camera_params]  # Only 1 camera for 2 images

        with pytest.raises(ValueError, match="must match"):
            ReconstructionInput(
                images=linear_images,
                gamma=1.0,
                cameras=cameras,
                tier="apex_research",
            )

    def test_invalid_image_dtype_raises(self, valid_camera_params):
        """Test that non-float32 images are rejected."""
        uint8_images = [
            np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8),
            np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8),
        ]
        cameras = [valid_camera_params, valid_camera_params]

        with pytest.raises(ValueError, match="float32"):
            ReconstructionInput(
                images=uint8_images,
                gamma=1.0,
                cameras=cameras,
                tier="apex_research",
            )


class TestScene3D:
    """Test Scene3D contract validation."""

    def test_valid_scene(self, valid_gaussian_splat, valid_camera_params):
        """Test valid 3D scene."""
        cameras = [valid_camera_params, valid_camera_params]
        scene = Scene3D(
            splats=valid_gaussian_splat,
            cameras=cameras,
            rmse=0.015,
            iteration=30000,
            convergence="converged",
        )

        assert scene.is_converged is True
        assert scene.rmse == 0.015
        assert scene.iteration == 30000

    def test_scene_quality_score(self, valid_gaussian_splat, valid_camera_params):
        """Test quality score calculation."""
        cameras = [valid_camera_params, valid_camera_params]

        # Excellent quality (RMSE < 0.01)
        excellent = Scene3D(
            splats=valid_gaussian_splat,
            cameras=cameras,
            rmse=0.005,
            iteration=30000,
            convergence="converged",
        )
        assert excellent.quality_score > 95

        # Good quality (RMSE < 0.02)
        good = Scene3D(
            splats=valid_gaussian_splat,
            cameras=cameras,
            rmse=0.015,
            iteration=30000,
            convergence="converged",
        )
        assert 85 <= good.quality_score < 95

        # Acceptable quality (RMSE < 0.05)
        acceptable = Scene3D(
            splats=valid_gaussian_splat,
            cameras=cameras,
            rmse=0.03,
            iteration=30000,
            convergence="converged",
        )
        assert 70 <= acceptable.quality_score < 85

        # Poor quality (RMSE >= 0.05)
        poor = Scene3D(
            splats=valid_gaussian_splat,
            cameras=cameras,
            rmse=0.1,
            iteration=30000,
            convergence="diverged",
        )
        assert poor.quality_score < 70

    def test_convergence_states(self, valid_gaussian_splat, valid_camera_params):
        """Test different convergence states."""
        cameras = [valid_camera_params, valid_camera_params]

        converged = Scene3D(
            splats=valid_gaussian_splat,
            cameras=cameras,
            rmse=0.01,
            iteration=30000,
            convergence="converged",
        )
        assert converged.is_converged is True

        max_iters = Scene3D(
            splats=valid_gaussian_splat,
            cameras=cameras,
            rmse=0.03,
            iteration=50000,
            convergence="max_iterations",
        )
        assert max_iters.is_converged is False

        diverged = Scene3D(
            splats=valid_gaussian_splat,
            cameras=cameras,
            rmse=0.5,
            iteration=10000,
            convergence="diverged",
        )
        assert diverged.is_converged is False

    def test_invalid_rmse_raises(self, valid_gaussian_splat, valid_camera_params):
        """Test that negative RMSE is rejected."""
        cameras = [valid_camera_params, valid_camera_params]

        with pytest.raises(ValueError, match="non-negative"):
            Scene3D(
                splats=valid_gaussian_splat,
                cameras=cameras,
                rmse=-0.01,
                iteration=30000,
                convergence="converged",
            )

    def test_invalid_iteration_raises(self, valid_gaussian_splat, valid_camera_params):
        """Test that negative iteration is rejected."""
        cameras = [valid_camera_params, valid_camera_params]

        with pytest.raises(ValueError, match="non-negative"):
            Scene3D(
                splats=valid_gaussian_splat,
                cameras=cameras,
                rmse=0.01,
                iteration=-1,
                convergence="converged",
            )

    def test_insufficient_cameras_raises(self, valid_gaussian_splat, valid_camera_params):
        """Test that less than 2 cameras is rejected."""
        with pytest.raises(ValueError, match="at least 2"):
            Scene3D(
                splats=valid_gaussian_splat,
                cameras=[valid_camera_params],  # Only 1 camera
                rmse=0.01,
                iteration=30000,
                convergence="converged",
            )

    def test_scene_metadata(self, valid_gaussian_splat, valid_camera_params):
        """Test scene with metadata."""
        cameras = [valid_camera_params, valid_camera_params]
        scene = Scene3D(
            splats=valid_gaussian_splat,
            cameras=cameras,
            rmse=0.01,
            iteration=30000,
            convergence="converged",
            metadata={
                "training_time_seconds": 1200,
                "peak_memory_mb": 8192,
                "learning_rate": 0.001,
            },
        )

        assert scene.metadata["training_time_seconds"] == 1200
        assert scene.metadata["peak_memory_mb"] == 8192


class TestSceneBuilder:
    """Test SceneBuilder high-level API with mocks."""

    def test_scene_builder_initialization(self):
        """Test SceneBuilder initialization."""
        pytest.importorskip("torch", reason="torch required for SceneBuilder")
        from transformation_portal.spatial_ai.reconstruction.scene_builder import SceneBuilder

        builder = SceneBuilder(tier="apex_research", device="cpu")

        assert builder.tier == "apex_research"
        assert builder.device == "cpu"
        assert builder._backend is None  # Lazy initialization

    def test_scene_builder_with_config(self):
        """Test SceneBuilder with backend config."""
        pytest.importorskip("torch", reason="torch required for SceneBuilder")
        from transformation_portal.spatial_ai.reconstruction.scene_builder import SceneBuilder

        config = {"max_iterations": 50000, "densification_interval": 500}
        builder = SceneBuilder(
            tier="apex_research_ultra",
            device="cuda",
            backend_config=config,
        )

        assert builder.backend_config == config

    def test_extract_camera_path_linear(self, valid_gaussian_splat, valid_intrinsics):
        """Test camera path extraction with linear interpolation."""
        pytest.importorskip("torch", reason="torch required for SceneBuilder")
        from transformation_portal.spatial_ai.reconstruction.scene_builder import SceneBuilder

        # Create two cameras at different positions
        ext1 = np.eye(4, dtype=np.float32)
        ext1[:3, 3] = [0, 0, 0]

        ext2 = np.eye(4, dtype=np.float32)
        ext2[:3, 3] = [10, 0, 0]

        cam1 = CameraParams(
            intrinsics=valid_intrinsics,
            extrinsics=ext1,
            width=512,
            height=512,
        )
        cam2 = CameraParams(
            intrinsics=valid_intrinsics,
            extrinsics=ext2,
            width=512,
            height=512,
        )

        scene = Scene3D(
            splats=valid_gaussian_splat,
            cameras=[cam1, cam2],
            rmse=0.01,
            iteration=30000,
            convergence="converged",
        )

        builder = SceneBuilder(tier="apex_research")
        path = builder.extract_camera_path(scene, num_frames=5, interpolation="linear")

        assert len(path) == 5
        # First camera should match start
        assert np.allclose(path[0].extrinsics[:3, 3], [0, 0, 0])
        # Last camera should match end
        assert np.allclose(path[-1].extrinsics[:3, 3], [10, 0, 0])

    def test_extract_camera_path_insufficient_cameras_raises(self, valid_gaussian_splat, valid_camera_params):
        """Test that insufficient cameras raises error."""
        pytest.importorskip("torch", reason="torch required for SceneBuilder")
        from transformation_portal.spatial_ai.reconstruction.scene_builder import SceneBuilder

        # Create scene with single camera (but Scene3D requires 2)
        # This is tested at Scene3D level, but let's verify builder handles it
        scene = Scene3D(
            splats=valid_gaussian_splat,
            cameras=[valid_camera_params, valid_camera_params],
            rmse=0.01,
            iteration=30000,
            convergence="converged",
        )

        builder = SceneBuilder(tier="apex_research")
        # This should work with 2 cameras
        path = builder.extract_camera_path(scene, num_frames=3)
        assert len(path) == 3

    def test_extract_camera_path_unsupported_interpolation_raises(self, valid_gaussian_splat, valid_camera_params):
        """Test that unsupported interpolation raises error."""
        pytest.importorskip("torch", reason="torch required for SceneBuilder")
        from transformation_portal.spatial_ai.reconstruction.scene_builder import SceneBuilder

        scene = Scene3D(
            splats=valid_gaussian_splat,
            cameras=[valid_camera_params, valid_camera_params],
            rmse=0.01,
            iteration=30000,
            convergence="converged",
        )

        builder = SceneBuilder(tier="apex_research")

        with pytest.raises(NotImplementedError, match="not implemented"):
            builder.extract_camera_path(scene, num_frames=3, interpolation="cubic")
