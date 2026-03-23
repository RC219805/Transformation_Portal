"""Tests for PLY export functionality (Phase 2.3 MVP)."""

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from transformation_portal.spatial_ai.reconstruction.contracts import (
    CameraParams,
    GaussianSplat,
    Scene3D,
)
from transformation_portal.spatial_ai.reconstruction.export_ply import (
    PLYExportConfig,
    PLYExporter,
    export_scene_to_ply,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def sample_splats():
    """Create sample Gaussian splats for testing."""
    N = 10
    positions = np.random.rand(N, 3).astype(np.float32)
    colors = np.random.rand(N, 3).astype(np.float32)
    scales = np.ones((N, 3), dtype=np.float32) * 0.01
    rotations = np.zeros((N, 4), dtype=np.float32)
    rotations[:, 0] = 1.0  # Identity quaternions
    opacities = np.ones((N, 1), dtype=np.float32) * 0.5

    return GaussianSplat(
        positions=positions,
        colors=colors,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        metadata={"test": True},
    )


@pytest.fixture
def sample_cameras():
    """Create sample camera parameters."""
    intrinsics = np.eye(3, dtype=np.float32)
    extrinsics = np.eye(4, dtype=np.float32)

    return [
        CameraParams(intrinsics=intrinsics, extrinsics=extrinsics, width=640, height=480),
        CameraParams(intrinsics=intrinsics, extrinsics=extrinsics, width=640, height=480),
    ]


@pytest.fixture
def sample_scene(sample_splats, sample_cameras):
    """Create sample Scene3D for testing."""
    return Scene3D(
        splats=sample_splats,
        cameras=sample_cameras,
        rmse=0.015,
        iteration=1000,
        convergence="converged",
        metadata={
            "backend": "gaussian_splatting",
            "tier": "apex_research",
            "device": "cpu",
            "num_views": 2,
            "requested_iterations": 1000,
            "elapsed_seconds": 5.5,
            "use_depth_prior": True,
            "use_segmentation": False,
            "use_pbr_textures": False,
        },
    )


class TestPLYExporter:
    """Tests for PLYExporter class."""

    def test_export_binary_creates_file(self, sample_scene, tmp_path):
        """Binary PLY export creates valid file."""
        output_path = tmp_path / "test.ply"
        exporter = PLYExporter(PLYExportConfig(binary=True))

        result = exporter.export(sample_scene, output_path, write_sidecar=False)

        assert result == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_export_ascii_creates_file(self, sample_scene, tmp_path):
        """ASCII PLY export creates valid file."""
        output_path = tmp_path / "test.ply"
        exporter = PLYExporter(PLYExportConfig(binary=False))

        result = exporter.export(sample_scene, output_path, write_sidecar=False)

        assert result == output_path
        assert output_path.exists()

    def test_export_creates_sidecar(self, sample_scene, tmp_path):
        """Export writes provenance sidecar JSON."""
        output_path = tmp_path / "test.ply"
        sidecar_path = output_path.with_suffix(".provenance.json")
        exporter = PLYExporter()

        exporter.export(sample_scene, output_path, write_sidecar=True)

        assert sidecar_path.exists()

        with open(sidecar_path) as f:
            provenance = json.load(f)

        assert provenance["schema_version"] == "1.0.0"
        assert provenance["backend"] == "gaussian_splatting"
        assert provenance["tier"] == "apex_research"
        assert provenance["num_gaussians"] == 10
        assert provenance["rmse"] == 0.015
        assert provenance["convergence"] == "converged"
        assert "output_hash_sha256" in provenance

    def test_export_sidecar_includes_additional_metadata(self, sample_scene, tmp_path):
        """Sidecar includes additional metadata when provided."""
        output_path = tmp_path / "test.ply"
        sidecar_path = output_path.with_suffix(".provenance.json")
        exporter = PLYExporter()

        additional = {
            "num_views": 3,
            "tier": "apex_research",
            "camera_source_summary": {"explicit": 3},
        }
        exporter.export(sample_scene, output_path, write_sidecar=True, additional_metadata=additional)

        with open(sidecar_path) as f:
            provenance = json.load(f)

        assert "request_metadata" in provenance
        assert provenance["request_metadata"]["num_views"] == 3

    def test_binary_ply_header_format(self, sample_scene, tmp_path):
        """Binary PLY has correct header format."""
        output_path = tmp_path / "test.ply"
        exporter = PLYExporter(PLYExportConfig(binary=True, include_attributes=True))

        exporter.export(sample_scene, output_path, write_sidecar=False)

        with open(output_path, "rb") as f:
            header_bytes = b""
            while True:
                line = f.readline()
                header_bytes += line
                if b"end_header" in line:
                    break

        header = header_bytes.decode("ascii")
        assert "ply" in header
        assert "format binary_little_endian 1.0" in header
        assert "element vertex 10" in header
        assert "property float x" in header
        assert "property float y" in header
        assert "property float z" in header
        assert "property uchar red" in header
        assert "property float scale_x" in header
        assert "property float rot_w" in header
        assert "property float opacity" in header

    def test_ascii_ply_header_format(self, sample_scene, tmp_path):
        """ASCII PLY has correct header format."""
        output_path = tmp_path / "test.ply"
        exporter = PLYExporter(PLYExportConfig(binary=False, include_attributes=True))

        exporter.export(sample_scene, output_path, write_sidecar=False)

        with open(output_path) as f:
            header = f.read()

        assert "ply" in header
        assert "format ascii 1.0" in header
        assert "element vertex 10" in header

    def test_export_without_attributes(self, sample_scene, tmp_path):
        """Export without attributes excludes scale/rotation/opacity."""
        output_path = tmp_path / "test.ply"
        exporter = PLYExporter(PLYExportConfig(binary=False, include_attributes=False))

        exporter.export(sample_scene, output_path, write_sidecar=False)

        with open(output_path) as f:
            header = f.read()

        assert "property float scale_x" not in header
        assert "property float rot_w" not in header
        assert "property float opacity" not in header

    def test_export_creates_parent_directories(self, sample_scene, tmp_path):
        """Export creates parent directories if needed."""
        output_path = tmp_path / "nested" / "deeply" / "test.ply"
        exporter = PLYExporter()

        exporter.export(sample_scene, output_path, write_sidecar=False)

        assert output_path.exists()


class TestExportSceneToPLY:
    """Tests for convenience export function."""

    def test_export_scene_to_ply(self, sample_scene, tmp_path):
        """Convenience function exports scene correctly."""
        output_path = tmp_path / "scene.ply"

        result = export_scene_to_ply(
            sample_scene,
            output_path,
            binary=True,
            include_attributes=True,
            write_sidecar=True,
        )

        assert result == output_path
        assert output_path.exists()
        assert output_path.with_suffix(".provenance.json").exists()


class TestPLYExportConfig:
    """Tests for PLYExportConfig."""

    def test_default_config(self):
        """Default config has expected values."""
        config = PLYExportConfig()

        assert config.binary is True
        assert config.include_attributes is True
        assert config.include_sh is False
        assert config.color_scale == 255

    def test_custom_config(self):
        """Custom config overrides defaults."""
        config = PLYExportConfig(binary=False, include_attributes=False)

        assert config.binary is False
        assert config.include_attributes is False
