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
    return SceneBuilder(tier="apex_research")


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
