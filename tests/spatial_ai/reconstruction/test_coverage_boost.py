"""Additional tests to improve coverage for Phase 2.3."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("torch", reason="torch is required for reconstruction coverage tests")
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


def _dummy_scene(cameras):
    """Create a lightweight Scene3D without running reconstruction."""
    splats = GaussianSplat(
        positions=np.zeros((1, 3), dtype=np.float32),
        colors=np.ones((1, 3), dtype=np.float32) * 0.5,
        scales=np.ones((1, 3), dtype=np.float32) * 0.1,
        rotations=np.array([[1, 0, 0, 0]], dtype=np.float32),
        opacities=np.ones((1, 1), dtype=np.float32),
    )
    return Scene3D(splats=splats, cameras=cameras, rmse=0.0, iteration=0, convergence="mock")


class TestSceneBuilderFileLoading:
    """Test SceneBuilder file loading capabilities."""

    def test_load_images_from_files(self):
        """Test loading images from disk."""
        from PIL import Image

        builder = SceneBuilder(tier="apex_research")

        # Create test images
        with tempfile.TemporaryDirectory() as tmpdir:
            image_paths = []
            for i in range(2):
                img = Image.new("RGB", (W, H), color=(i * 50, i * 50, i * 50))
                path = Path(tmpdir) / f"test_{i}.png"
                img.save(path)
                image_paths.append(path)

            intrinsics = np.eye(3, dtype=np.float32)
            cameras = [CameraParams(intrinsics, np.eye(4, dtype=np.float32), W, H) for _ in range(2)]

            dummy = _dummy_scene(cameras)
            # Patch out heavy reconstruction; keep file I/O + preprocessing under test.
            with patch.object(builder.backend, "reconstruct", return_value=dummy) as m:
                scene = builder.build_from_images(image_paths=image_paths, cameras=cameras, iterations=999)

            assert scene is not None
            m.assert_called_once()
            # Validate that images passed into reconstruction input are RGB float32 arrays.
            reconstruction_input = m.call_args.args[0]
            imgs = reconstruction_input.images
            assert len(imgs) == 2
            assert imgs[0].shape == (H, W, 3)
            assert imgs[0].dtype == np.float32

    def test_load_nonexistent_file(self):
        """Test error handling for missing files."""
        builder = SceneBuilder(tier="apex_research")

        fake_paths = [Path("/nonexistent/image1.png"), Path("/nonexistent/image2.png")]
        intrinsics = np.eye(3, dtype=np.float32)
        cameras = [CameraParams(intrinsics, np.eye(4, dtype=np.float32), W, H) for _ in range(2)]

        with pytest.raises(FileNotFoundError):
            builder.build_from_images(image_paths=fake_paths, cameras=cameras)

    def test_load_grayscale_image(self):
        """Test loading grayscale images (auto-converted to RGB)."""
        from PIL import Image

        builder = SceneBuilder(tier="apex_research")

        with tempfile.TemporaryDirectory() as tmpdir:
            image_paths = []
            for i in range(2):
                img = Image.new("L", (W, H), color=128)  # Grayscale
                path = Path(tmpdir) / f"gray_{i}.png"
                img.save(path)
                image_paths.append(path)

            intrinsics = np.eye(3, dtype=np.float32)
            cameras = [CameraParams(intrinsics, np.eye(4, dtype=np.float32), W, H) for _ in range(2)]

            dummy = _dummy_scene(cameras)
            with patch.object(builder.backend, "reconstruct", return_value=dummy) as m:
                scene = builder.build_from_images(image_paths=image_paths, cameras=cameras, iterations=999)

            assert scene is not None
            m.assert_called_once()
            reconstruction_input = m.call_args.args[0]
            imgs = reconstruction_input.images
            assert imgs[0].shape == (H, W, 3)  # grayscale -> RGB

    def test_load_rgba_image(self):
        """Test loading RGBA images (alpha dropped)."""
        from PIL import Image

        builder = SceneBuilder(tier="apex_research")

        with tempfile.TemporaryDirectory() as tmpdir:
            image_paths = []
            for i in range(2):
                img = Image.new("RGBA", (W, H), color=(100, 150, 200, 255))
                path = Path(tmpdir) / f"rgba_{i}.png"
                img.save(path)
                image_paths.append(path)

            intrinsics = np.eye(3, dtype=np.float32)
            cameras = [CameraParams(intrinsics, np.eye(4, dtype=np.float32), W, H) for _ in range(2)]

            dummy = _dummy_scene(cameras)
            with patch.object(builder.backend, "reconstruct", return_value=dummy) as m:
                scene = builder.build_from_images(image_paths=image_paths, cameras=cameras, iterations=999)

            assert scene is not None
            m.assert_called_once()
            reconstruction_input = m.call_args.args[0]
            imgs = reconstruction_input.images
            assert imgs[0].shape == (H, W, 3)  # alpha dropped


class TestGeometricValidatorAdvanced:
    """Test GeometricValidator advanced features."""

    def test_compute_reprojection_error(self):
        """Test reprojection error computation."""
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
        intrinsics[0, 0] = intrinsics[1, 1] = 525.0
        intrinsics[0, 2] = 320.0
        intrinsics[1, 2] = 240.0

        cameras = [CameraParams(intrinsics, np.eye(4, dtype=np.float32), 640, 480) for _ in range(2)]
        scene = Scene3D(splats=splats, cameras=cameras, rmse=0.01, iteration=1000, convergence="converged")

        error = validator.compute_reprojection_error(scene, view_idx=0)

        assert isinstance(error, float)
        assert error >= 0

    def test_compute_reprojection_error_with_reference(self):
        """Test reprojection error with reference 2D points."""
        validator = GeometricValidator()

        N = 10
        splats = GaussianSplat(
            positions=np.random.randn(N, 3).astype(np.float32),
            colors=np.random.rand(N, 3).astype(np.float32),
            scales=np.ones((N, 3), dtype=np.float32),
            rotations=np.tile([1, 0, 0, 0], (N, 1)).astype(np.float32),
            opacities=np.ones((N, 1), dtype=np.float32),
        )

        intrinsics = np.eye(3, dtype=np.float32)
        cameras = [CameraParams(intrinsics, np.eye(4, dtype=np.float32), 640, 480) for _ in range(2)]
        scene = Scene3D(splats=splats, cameras=cameras, rmse=0.01, iteration=1000, convergence="converged")

        points_2d = np.random.rand(N, 2).astype(np.float32) * 640

        error = validator.compute_reprojection_error(scene, view_idx=0, points_2d=points_2d)

        assert isinstance(error, float)

    def test_compute_depth_consistency(self):
        """Test depth consistency computation."""
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
        intrinsics[0, 0] = intrinsics[1, 1] = 525.0
        intrinsics[0, 2] = 320.0
        intrinsics[1, 2] = 240.0

        cameras = [CameraParams(intrinsics, np.eye(4, dtype=np.float32), 640, 480) for _ in range(2)]
        scene = Scene3D(splats=splats, cameras=cameras, rmse=0.01, iteration=1000, convergence="converged")

        depth_maps = [np.random.rand(480, 640).astype(np.float32) * 10 for _ in range(2)]

        consistency = validator.compute_depth_consistency(scene, depth_maps)

        assert isinstance(consistency, float)
        assert 0 <= consistency <= 1

    def test_validate_scene_with_depth_maps(self):
        """Test scene validation with depth maps."""
        validator = GeometricValidator()

        N = 50
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

        depth_maps = [np.random.rand(240, 320).astype(np.float32) for _ in range(2)]

        results = validator.validate_scene(scene, depth_maps=depth_maps)

        assert "depth_consistency" in results
        assert results["depth_consistency"] is not None

    def test_invalid_view_index(self):
        """Test reprojection error with invalid view index."""
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
        scene = Scene3D(splats=splats, cameras=cameras, rmse=0.01, iteration=1000, convergence="converged")

        with pytest.raises(ValueError, match="Invalid view index"):
            validator.compute_reprojection_error(scene, view_idx=5)


class TestMeshExporterEdgeCases:
    """Test MeshExporter edge cases."""

    def test_export_ply_without_attributes(self):
        """Test PLY export without Gaussian attributes."""
        exporter = MeshExporter()

        N = 20
        splats = GaussianSplat(
            positions=np.random.randn(N, 3).astype(np.float32),
            colors=np.random.rand(N, 3).astype(np.float32),
            scales=np.ones((N, 3), dtype=np.float32),
            rotations=np.tile([1, 0, 0, 0], (N, 1)).astype(np.float32),
            opacities=np.ones((N, 1), dtype=np.float32),
        )

        intrinsics = np.eye(3, dtype=np.float32)
        cameras = [CameraParams(intrinsics, np.eye(4, dtype=np.float32), 640, 480) for _ in range(2)]
        scene = Scene3D(splats=splats, cameras=cameras, rmse=0.01, iteration=1000, convergence="converged")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "minimal.ply"
            exporter.export_ply(scene, output_path, include_attributes=False, binary=True)

            assert output_path.exists()

    def test_export_obj_with_subsampling(self):
        """Test OBJ export with subsampling."""
        exporter = MeshExporter()

        N = 100
        splats = GaussianSplat(
            positions=np.random.randn(N, 3).astype(np.float32),
            colors=np.random.rand(N, 3).astype(np.float32),
            scales=np.ones((N, 3), dtype=np.float32),
            rotations=np.tile([1, 0, 0, 0], (N, 1)).astype(np.float32),
            opacities=np.ones((N, 1), dtype=np.float32),
        )

        intrinsics = np.eye(3, dtype=np.float32)
        cameras = [CameraParams(intrinsics, np.eye(4, dtype=np.float32), 640, 480) for _ in range(2)]
        scene = Scene3D(splats=splats, cameras=cameras, rmse=0.01, iteration=1000, convergence="converged")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subsampled.obj"
            exporter.export_obj(scene, output_path, subsample_factor=2)

            assert output_path.exists()

    def test_export_obj_without_colors(self):
        """Test OBJ export without vertex colors."""
        exporter = MeshExporter()

        N = 30
        splats = GaussianSplat(
            positions=np.random.randn(N, 3).astype(np.float32),
            colors=np.random.rand(N, 3).astype(np.float32),
            scales=np.ones((N, 3), dtype=np.float32),
            rotations=np.tile([1, 0, 0, 0], (N, 1)).astype(np.float32),
            opacities=np.ones((N, 1), dtype=np.float32),
        )

        intrinsics = np.eye(3, dtype=np.float32)
        cameras = [CameraParams(intrinsics, np.eye(4, dtype=np.float32), 640, 480) for _ in range(2)]
        scene = Scene3D(splats=splats, cameras=cameras, rmse=0.01, iteration=1000, convergence="converged")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "no_colors.obj"
            exporter.export_obj(scene, output_path, vertex_colors=False)

            assert output_path.exists()
            assert not output_path.with_suffix(".mtl").exists()


class TestSceneBuilderCameraPath:
    """Test camera path extraction edge cases."""

    def test_camera_path_interpolation(self):
        """Test camera path interpolation between views."""
        builder = SceneBuilder(tier="apex_research")

        N = 10
        splats = GaussianSplat(
            positions=np.zeros((N, 3), dtype=np.float32),
            colors=np.ones((N, 3), dtype=np.float32) * 0.5,
            scales=np.ones((N, 3), dtype=np.float32),
            rotations=np.tile([1, 0, 0, 0], (N, 1)).astype(np.float32),
            opacities=np.ones((N, 1), dtype=np.float32),
        )

        intrinsics = np.eye(3, dtype=np.float32)
        cameras = [
            CameraParams(intrinsics, np.eye(4, dtype=np.float32), 320, 240),
            CameraParams(intrinsics, np.eye(4, dtype=np.float32), 320, 240),
        ]
        scene = Scene3D(splats=splats, cameras=cameras, rmse=0.01, iteration=1000, convergence="converged")

        # Test with different number of frames
        path = builder.extract_camera_path(scene, num_frames=10)
        assert len(path) == 10
