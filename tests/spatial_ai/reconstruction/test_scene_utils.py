"""Tests for SceneBuilder, MeshExporter, and GeometricValidator."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch", reason="torch is required for SceneBuilder reconstruction tests")
pytestmark = pytest.mark.ml

from transformation_portal.spatial_ai.reconstruction import (
    CameraParams,
    GaussianSplat,
    GeometricValidator,
    MeshExporter,
    Scene3D,
    SceneBuilder,
)

H, W = 48, 64
ITERS = 3


@pytest.fixture(scope="module")
def intrinsics():
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = K[1, 1] = 100.0
    K[0, 2] = W / 2.0
    K[1, 2] = H / 2.0
    return K


@pytest.fixture(scope="module")
def cameras(intrinsics):
    return [CameraParams(intrinsics, np.eye(4, dtype=np.float32), W, H) for _ in range(2)]


@pytest.fixture(scope="module")
def images():
    rng = np.random.default_rng(0)
    return [rng.random((H, W, 3), dtype=np.float32) for _ in range(2)]


@pytest.fixture(scope="module")
def builder():
    # Avoid MPS worker aborts under xdist on Apple Silicon; these tests exercise
    # reconstruction behavior, not accelerator selection.
    return SceneBuilder(tier="apex_research", device="cpu")


@pytest.fixture(scope="module")
def scene(builder, images, cameras):
    # Build ONCE for the entire module; this is the main PR-time win.
    return builder.build_from_arrays(images=images, cameras=cameras, iterations=ITERS)


@pytest.fixture(scope="module")
def scene_with_depth(builder, images, cameras):
    rng = np.random.default_rng(1)
    depth_maps = [rng.random((H, W), dtype=np.float32) for _ in range(2)]
    return builder.build_from_arrays(images=images, cameras=cameras, depth_maps=depth_maps, iterations=ITERS)


class TestSceneBuilder:
    """Test SceneBuilder."""

    def test_initialization(self, builder):
        """Test scene builder initialization."""
        assert builder.tier == "apex_research"

    def test_build_from_arrays(self, scene):
        """Test building scene from numpy arrays."""
        assert scene is not None
        assert scene.splats.num_gaussians > 0

    def test_build_with_depth_maps(self, scene_with_depth):
        """Test building scene with depth maps."""
        assert scene_with_depth.metadata["use_depth_prior"]

    def test_render_novel_view(self, builder, scene, intrinsics):
        """Test novel view rendering."""
        novel_camera = CameraParams(intrinsics, np.eye(4, dtype=np.float32), W, H)
        rendered = builder.render_novel_view(scene, novel_camera)

        assert rendered.shape == (H, W, 3)

    def test_extract_camera_path(self, builder, scene):
        """Test camera path extraction."""
        path = builder.extract_camera_path(scene, num_frames=50)

        assert len(path) == 50

    def test_extract_camera_path_preserves_rotation_orthogonality(self, builder, intrinsics):
        """Test that SLERP interpolation preserves rotation matrix orthogonality.

        This validates the fix for the TODO that was using naive linear interpolation
        of 4x4 extrinsic matrices (which produces sheared rotations).
        """
        # Test scene parameters (minimal valid scene for path extraction)
        TEST_RMSE = 0.01
        TEST_ITERATION = 1000
        TEST_CONVERGENCE = "converged"

        # Create two cameras with 90° rotation difference around Z-axis
        # cam0: identity rotation at origin
        extrinsics0 = np.eye(4, dtype=np.float32)
        extrinsics0[:3, 3] = [0.0, 0.0, 0.0]  # origin

        # cam1: 90° rotation around Z-axis (camera-up convention) + translation
        # This uses the standard right-hand coordinate system where Z points up/forward
        # Rotation matrix R = [[cos(θ), -sin(θ), 0], [sin(θ), cos(θ), 0], [0, 0, 1]]
        # For θ=90°: cos(90°)=0, sin(90°)=1
        extrinsics1 = np.eye(4, dtype=np.float32)
        extrinsics1[:3, :3] = np.array(
            [
                [0, -1, 0],  # x' = -y
                [1, 0, 0],  # y' = x
                [0, 0, 1],  # z' = z (Z-axis unchanged)
            ],
            dtype=np.float32,
        )
        extrinsics1[:3, 3] = [1.0, 1.0, 0.0]  # translated

        cameras = [
            CameraParams(intrinsics.copy(), extrinsics0, W, H),
            CameraParams(intrinsics.copy(), extrinsics1, W, H),
        ]

        # Create minimal valid scene for path extraction test
        N = 10
        splats = GaussianSplat(
            positions=np.zeros((N, 3), dtype=np.float32),
            colors=np.ones((N, 3), dtype=np.float32) * 0.5,
            scales=np.ones((N, 3), dtype=np.float32),
            rotations=np.tile([1, 0, 0, 0], (N, 1)).astype(np.float32),
            opacities=np.ones((N, 1), dtype=np.float32),
        )
        scene = Scene3D(
            splats=splats,
            cameras=cameras,
            rmse=TEST_RMSE,
            iteration=TEST_ITERATION,
            convergence=TEST_CONVERGENCE,
        )

        # Extract path
        path = builder.extract_camera_path(scene, num_frames=11)

        assert len(path) == 11

        # Verify all interpolated rotation matrices are orthogonal (R @ R.T = I)
        for i, cam in enumerate(path):
            R = cam.extrinsics[:3, :3]
            RRT = R @ R.T
            np.testing.assert_allclose(
                RRT, np.eye(3, dtype=np.float32), atol=1e-5, err_msg=f"Rotation at frame {i} is not orthogonal (SLERP failed)"
            )
            # Also verify determinant is +1 (not -1, which would be a reflection)
            det = np.linalg.det(R)
            np.testing.assert_allclose(
                det, 1.0, atol=1e-5, err_msg=f"Rotation at frame {i} has det={det}, expected +1 (proper rotation)"
            )

        # Verify boundary conditions: first and last match original cameras
        np.testing.assert_allclose(path[0].extrinsics, extrinsics0, atol=1e-5)
        np.testing.assert_allclose(path[-1].extrinsics, extrinsics1, atol=1e-5)

        # Verify midpoint (t=0.5) has 45° rotation (between 0° and 90°)
        # At t=0.5, the rotation should be 45° around Z-axis
        mid_R = path[5].extrinsics[:3, :3]
        expected_cos = np.cos(np.pi / 4)  # cos(45°) ≈ 0.707
        expected_sin = np.sin(np.pi / 4)  # sin(45°) ≈ 0.707
        expected_mid_R = np.array(
            [
                [expected_cos, -expected_sin, 0],
                [expected_sin, expected_cos, 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(mid_R, expected_mid_R, atol=1e-4)

        # Verify translation is linearly interpolated
        mid_t = path[5].extrinsics[:3, 3]
        expected_mid_t = np.array([0.5, 0.5, 0.0], dtype=np.float32)
        np.testing.assert_allclose(mid_t, expected_mid_t, atol=1e-5)

    def test_extract_camera_path_spline_passes_through_keyframes(self, builder, intrinsics):
        """Spline interpolation must pass through every keyframe camera.

        With three or more cameras, the natural cubic spline used for
        translation must hit each input camera's position at the corresponding
        uniformly-spaced parameter ``t``, and SLERP must reproduce each
        keyframe rotation at that ``t``.
        """
        TEST_RMSE = 0.01
        TEST_ITERATION = 1000
        TEST_CONVERGENCE = "converged"

        # Three keyframes with progressive Z-axis rotations (0°, 45°, 90°)
        # and translations along a curve in the XY plane.
        angles = [0.0, np.pi / 4, np.pi / 2]
        positions = [
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([1.0, 2.0, 0.0], dtype=np.float32),
            np.array([2.0, 1.0, 0.0], dtype=np.float32),
        ]
        cameras = []
        for angle, pos in zip(angles, positions):
            extrinsics = np.eye(4, dtype=np.float32)
            extrinsics[:3, :3] = np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
            extrinsics[:3, 3] = pos
            cameras.append(CameraParams(intrinsics.copy(), extrinsics, W, H))

        N = 10
        splats = GaussianSplat(
            positions=np.zeros((N, 3), dtype=np.float32),
            colors=np.ones((N, 3), dtype=np.float32) * 0.5,
            scales=np.ones((N, 3), dtype=np.float32),
            rotations=np.tile([1, 0, 0, 0], (N, 1)).astype(np.float32),
            opacities=np.ones((N, 1), dtype=np.float32),
        )
        scene = Scene3D(
            splats=splats,
            cameras=cameras,
            rmse=TEST_RMSE,
            iteration=TEST_ITERATION,
            convergence=TEST_CONVERGENCE,
        )

        # Use 5 frames so keyframes land exactly at indices 0, 2, 4 (t=0, 0.5, 1).
        path = builder.extract_camera_path(scene, num_frames=5, interpolation="spline")
        assert len(path) == 5

        # Every interpolated rotation must remain a proper rotation matrix.
        for i, cam in enumerate(path):
            R = cam.extrinsics[:3, :3]
            np.testing.assert_allclose(
                R @ R.T,
                np.eye(3, dtype=np.float32),
                atol=1e-5,
                err_msg=f"Frame {i}: spline rotation lost orthogonality",
            )

        # Path must pass through every keyframe camera at its uniform t.
        for keyframe_idx, frame_idx in enumerate((0, 2, 4)):
            np.testing.assert_allclose(
                path[frame_idx].extrinsics,
                cameras[keyframe_idx].extrinsics,
                atol=1e-4,
                err_msg=f"Spline path missed keyframe {keyframe_idx} at frame {frame_idx}",
            )

    def test_extract_camera_path_unknown_interpolation_rejected(self, builder, intrinsics):
        """Unknown interpolation modes must raise NotImplementedError."""
        extrinsics0 = np.eye(4, dtype=np.float32)
        extrinsics1 = np.eye(4, dtype=np.float32)
        extrinsics1[:3, 3] = [1.0, 0.0, 0.0]
        cameras = [
            CameraParams(intrinsics.copy(), extrinsics0, W, H),
            CameraParams(intrinsics.copy(), extrinsics1, W, H),
        ]
        N = 5
        splats = GaussianSplat(
            positions=np.zeros((N, 3), dtype=np.float32),
            colors=np.ones((N, 3), dtype=np.float32) * 0.5,
            scales=np.ones((N, 3), dtype=np.float32),
            rotations=np.tile([1, 0, 0, 0], (N, 1)).astype(np.float32),
            opacities=np.ones((N, 1), dtype=np.float32),
        )
        scene = Scene3D(splats=splats, cameras=cameras, rmse=0.01, iteration=1000, convergence="converged")

        with pytest.raises(NotImplementedError):
            builder.extract_camera_path(scene, num_frames=4, interpolation="bezier")


class TestMeshExporter:
    """Test MeshExporter."""

    def test_initialization(self):
        """Test mesh exporter initialization."""
        exporter = MeshExporter()
        assert exporter is not None

    def test_export_ply_binary(self):
        """Test PLY export (binary)."""
        exporter = MeshExporter()

        # Create scene
        N = 100
        splats = GaussianSplat(
            positions=np.random.randn(N, 3).astype(np.float32),
            colors=np.random.rand(N, 3).astype(np.float32),
            scales=np.ones((N, 3), dtype=np.float32) * 0.1,
            rotations=np.tile([1, 0, 0, 0], (N, 1)).astype(np.float32),
            opacities=np.ones((N, 1), dtype=np.float32) * 0.5,
        )

        intrinsics = np.eye(3, dtype=np.float32)
        cameras = [CameraParams(intrinsics, np.eye(4, dtype=np.float32), 640, 480) for _ in range(2)]
        scene = Scene3D(splats=splats, cameras=cameras, rmse=0.015, iteration=30000, convergence="converged")

        # Export
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.ply"
            exporter.export_ply(scene, output_path, binary=True)

            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_export_ply_ascii(self):
        """Test PLY export (ASCII)."""
        exporter = MeshExporter()

        N = 10
        splats = GaussianSplat(
            positions=np.zeros((N, 3), dtype=np.float32),
            colors=np.ones((N, 3), dtype=np.float32) * 0.5,
            scales=np.ones((N, 3), dtype=np.float32),
            rotations=np.tile([1, 0, 0, 0], (N, 1)).astype(np.float32),
            opacities=np.ones((N, 1), dtype=np.float32),
        )

        intrinsics = np.eye(3, dtype=np.float32)
        cameras = [CameraParams(intrinsics, np.eye(4, dtype=np.float32), 640, 480) for _ in range(2)]
        scene = Scene3D(splats=splats, cameras=cameras, rmse=0.01, iteration=1000, convergence="converged")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.ply"
            exporter.export_ply(scene, output_path, binary=False)

            assert output_path.exists()

            # Check ASCII format
            content = output_path.read_text()
            assert "format ascii 1.0" in content
            assert "element vertex 10" in content

    def test_export_obj(self):
        """Test OBJ export."""
        exporter = MeshExporter()

        N = 50
        splats = GaussianSplat(
            positions=np.random.randn(N, 3).astype(np.float32),
            colors=np.random.rand(N, 3).astype(np.float32),
            scales=np.ones((N, 3), dtype=np.float32),
            rotations=np.tile([1, 0, 0, 0], (N, 1)).astype(np.float32),
            opacities=np.ones((N, 1), dtype=np.float32),
        )

        intrinsics = np.eye(3, dtype=np.float32)
        cameras = [CameraParams(intrinsics, np.eye(4, dtype=np.float32), 640, 480) for _ in range(2)]
        scene = Scene3D(splats=splats, cameras=cameras, rmse=0.02, iteration=5000, convergence="converged")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.obj"
            exporter.export_obj(scene, output_path, vertex_colors=True)

            assert output_path.exists()
            assert output_path.with_suffix(".mtl").exists()

    def test_export_cameras(self):
        """Test camera export to JSON."""
        exporter = MeshExporter()

        intrinsics = np.eye(3, dtype=np.float32)
        cameras = [
            CameraParams(intrinsics, np.eye(4, dtype=np.float32), 640, 480, camera_id="cam_001"),
            CameraParams(intrinsics, np.eye(4, dtype=np.float32), 640, 480, camera_id="cam_002"),
        ]

        splats = GaussianSplat(
            positions=np.zeros((10, 3), dtype=np.float32),
            colors=np.ones((10, 3), dtype=np.float32) * 0.5,
            scales=np.ones((10, 3), dtype=np.float32),
            rotations=np.tile([1, 0, 0, 0], (10, 1)).astype(np.float32),
            opacities=np.ones((10, 1), dtype=np.float32),
        )

        scene = Scene3D(splats=splats, cameras=cameras, rmse=0.01, iteration=1000, convergence="converged")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "cameras.json"
            exporter.export_cameras(scene, output_path)

            assert output_path.exists()

            # Parse JSON
            import json

            data = json.loads(output_path.read_text())
            assert data["num_cameras"] == 2
            assert len(data["cameras"]) == 2


class TestGeometricValidator:
    """Test GeometricValidator."""

    def test_initialization(self):
        """Test validator initialization."""
        validator = GeometricValidator()
        assert validator is not None

    def test_compute_rmse(self):
        """Test RMSE computation."""
        validator = GeometricValidator()

        # Create scene
        N = 100
        splats = GaussianSplat(
            positions=np.random.randn(N, 3).astype(np.float32),
            colors=np.random.rand(N, 3).astype(np.float32),
            scales=np.ones((N, 3), dtype=np.float32),
            rotations=np.tile([1, 0, 0, 0], (N, 1)).astype(np.float32),
            opacities=np.ones((N, 1), dtype=np.float32),
        )

        intrinsics = np.eye(3, dtype=np.float32)
        cameras = [CameraParams(intrinsics, np.eye(4, dtype=np.float32), 320, 240) for _ in range(2)]
        scene = Scene3D(splats=splats, cameras=cameras, rmse=0.015, iteration=1000, convergence="converged")

        # Reference images
        reference_images = [np.random.rand(240, 320, 3).astype(np.float32) for _ in range(2)]

        rmse = validator.compute_rmse(scene, reference_images)

        assert isinstance(rmse, float)
        assert rmse >= 0

    def test_compute_coverage(self):
        """Test coverage statistics."""
        validator = GeometricValidator()

        N = 100
        splats = GaussianSplat(
            positions=np.random.randn(N, 3).astype(np.float32) * 5,
            colors=np.random.rand(N, 3).astype(np.float32),
            scales=np.ones((N, 3), dtype=np.float32),
            rotations=np.tile([1, 0, 0, 0], (N, 1)).astype(np.float32),
            opacities=np.ones((N, 1), dtype=np.float32),
        )

        intrinsics = np.eye(3, dtype=np.float32)
        intrinsics[0, 0] = intrinsics[1, 1] = 262.5
        intrinsics[0, 2] = 160.0
        intrinsics[1, 2] = 120.0

        cameras = [CameraParams(intrinsics, np.eye(4, dtype=np.float32), 320, 240) for _ in range(2)]
        scene = Scene3D(splats=splats, cameras=cameras, rmse=0.01, iteration=1000, convergence="converged")

        coverage = validator.compute_coverage(scene)

        assert "mean_points_per_view" in coverage
        assert "min_points_per_view" in coverage
        assert "max_points_per_view" in coverage
        assert "coverage_std" in coverage

    def test_validate_scene(self):
        """Test comprehensive scene validation."""
        validator = GeometricValidator()

        N = 100
        splats = GaussianSplat(
            positions=np.random.randn(N, 3).astype(np.float32),
            colors=np.random.rand(N, 3).astype(np.float32),
            scales=np.ones((N, 3), dtype=np.float32),
            rotations=np.tile([1, 0, 0, 0], (N, 1)).astype(np.float32),
            opacities=np.ones((N, 1), dtype=np.float32),
        )

        intrinsics = np.eye(3, dtype=np.float32)
        cameras = [CameraParams(intrinsics, np.eye(4, dtype=np.float32), 320, 240) for _ in range(2)]
        scene = Scene3D(splats=splats, cameras=cameras, rmse=0.015, iteration=1000, convergence="converged")

        results = validator.validate_scene(scene)

        assert "rmse" in results
        assert "rmse_pass" in results
        assert "coverage" in results
        assert "quality_grade" in results

        # Check quality grade
        assert results["quality_grade"] in ["A", "B", "C", "D", "F"]
        assert isinstance(results["rmse_pass"], bool)

    def test_quality_grading(self):
        """Test quality grade assignment."""
        validator = GeometricValidator()

        N = 10
        splats = GaussianSplat(
            positions=np.zeros((N, 3), dtype=np.float32),
            colors=np.ones((N, 3), dtype=np.float32) * 0.5,
            scales=np.ones((N, 3), dtype=np.float32),
            rotations=np.tile([1, 0, 0, 0], (N, 1)).astype(np.float32),
            opacities=np.ones((N, 1), dtype=np.float32),
        )

        intrinsics = np.eye(3, dtype=np.float32)
        cameras = [CameraParams(intrinsics, np.eye(4, dtype=np.float32), 320, 240) for _ in range(2)]

        # Test excellent grade
        scene_a = Scene3D(splats=splats, cameras=cameras, rmse=0.005, iteration=1000, convergence="converged")
        results_a = validator.validate_scene(scene_a)
        assert results_a["quality_grade"] == "A"

        # Test good grade
        scene_b = Scene3D(splats=splats, cameras=cameras, rmse=0.015, iteration=1000, convergence="converged")
        results_b = validator.validate_scene(scene_b)
        assert results_b["quality_grade"] == "B"

        # Test acceptable grade
        scene_c = Scene3D(splats=splats, cameras=cameras, rmse=0.03, iteration=1000, convergence="converged")
        results_c = validator.validate_scene(scene_c)
        assert results_c["quality_grade"] == "C"

        # Test poor grade
        scene_d = Scene3D(splats=splats, cameras=cameras, rmse=0.08, iteration=1000, convergence="diverged")
        results_d = validator.validate_scene(scene_d)
        assert results_d["quality_grade"] == "D"
