"""Tests for DepthArtifact contract.

These tests validate the core depth artifact contract that serves as
universal currency across all pipeline stages.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.unit]

from transformation_portal.lux_depth_v3.contracts import (
    CameraIntrinsics,
    DepthArtifact,
    DepthArtifactWriter,
    DepthProvenance,
    LicenseTier,
)


class TestLicenseTier:
    """Test LicenseTier enum."""

    def test_commercial_tier_value(self):
        """Test COMMERCIAL tier has correct value."""
        assert LicenseTier.COMMERCIAL.value == "commercial"

    def test_non_commercial_tier_value(self):
        """Test NON_COMMERCIAL tier has correct value."""
        assert LicenseTier.NON_COMMERCIAL.value == "non_commercial"

    def test_experimental_tier_value(self):
        """Test EXPERIMENTAL tier has correct value."""
        assert LicenseTier.EXPERIMENTAL.value == "experimental"


class TestCameraIntrinsics:
    """Test CameraIntrinsics dataclass."""

    def test_create_intrinsics(self):
        """Test creating camera intrinsics."""
        intrinsics = CameraIntrinsics(
            fx=1000.0,
            fy=1000.0,
            cx=960.0,
            cy=540.0,
            width=1920,
            height=1080,
            source="manual",
        )
        assert intrinsics.fx == 1000.0
        assert intrinsics.fy == 1000.0
        assert intrinsics.cx == 960.0
        assert intrinsics.cy == 540.0
        assert intrinsics.width == 1920
        assert intrinsics.height == 1080
        assert intrinsics.source == "manual"

    def test_intrinsics_to_dict(self):
        """Test serialization to dictionary."""
        intrinsics = CameraIntrinsics(
            fx=500.0,
            fy=500.0,
            cx=320.0,
            cy=240.0,
            width=640,
            height=480,
        )
        data = intrinsics.to_dict()
        assert data["fx"] == 500.0
        assert data["width"] == 640
        assert data["source"] == "estimated"

    def test_intrinsics_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "fx": 800.0,
            "fy": 800.0,
            "cx": 512.0,
            "cy": 384.0,
            "width": 1024,
            "height": 768,
            "source": "exif",
        }
        intrinsics = CameraIntrinsics.from_dict(data)
        assert intrinsics.fx == 800.0
        assert intrinsics.source == "exif"

    def test_estimate_from_image(self):
        """Test intrinsics estimation from image dimensions."""
        intrinsics = CameraIntrinsics.estimate_from_image(
            width=1920,
            height=1080,
            fov_degrees=60.0,
        )
        assert intrinsics.width == 1920
        assert intrinsics.height == 1080
        assert intrinsics.cx == 960.0
        assert intrinsics.cy == 540.0
        assert intrinsics.source == "estimated"
        # Check that focal length is reasonable for 60° FOV
        # fx = width / (2 * tan(30°)) ≈ 1663 for 1920px width
        assert intrinsics.fx > 1000  # Focal length should be reasonable for typical FOV


class TestDepthProvenance:
    """Test DepthProvenance dataclass."""

    def test_create_provenance(self):
        """Test creating depth provenance."""
        provenance = DepthProvenance(
            model_id="depth-anything/DA3-Large",
            license_tier=LicenseTier.COMMERCIAL,
            checkpoint_sha256="abc123def456",
            preset="architectural_interior",
            device="mps",
        )
        assert provenance.model_id == "depth-anything/DA3-Large"
        assert provenance.license_tier == LicenseTier.COMMERCIAL
        assert provenance.checkpoint_sha256 == "abc123def456"

    def test_provenance_to_dict(self):
        """Test provenance serialization."""
        provenance = DepthProvenance(
            model_id="test-model",
            license_tier=LicenseTier.NON_COMMERCIAL,
            device="cuda",
        )
        data = provenance.to_dict()
        assert data["model_id"] == "test-model"
        assert data["license_tier"] == "non_commercial"
        assert data["device"] == "cuda"

    def test_provenance_from_dict(self):
        """Test provenance deserialization."""
        data = {
            "model_id": "test-model",
            "license_tier": "experimental",
            "checkpoint_sha256": None,
            "preset": None,
            "device": "cpu",
            "runtime_version": "2.0.0",
            "timestamp_utc": "2026-02-02T00:00:00+00:00",
            "request_id": None,
            "downgrade_events": [],
        }
        provenance = DepthProvenance.from_dict(data)
        assert provenance.license_tier == LicenseTier.EXPERIMENTAL
        assert provenance.device == "cpu"

    def test_provenance_downgrade_events(self):
        """Test provenance with downgrade events."""
        provenance = DepthProvenance(
            model_id="test",
            license_tier=LicenseTier.COMMERCIAL,
            downgrade_events=("model_not_found", "used_fallback"),
        )
        assert len(provenance.downgrade_events) == 2
        data = provenance.to_dict()
        assert data["downgrade_events"] == ["model_not_found", "used_fallback"]


class TestDepthArtifact:
    """Test DepthArtifact dataclass."""

    @pytest.fixture
    def sample_depth_map(self):
        """Create sample depth map."""
        return np.random.rand(480, 640).astype(np.float32)

    @pytest.fixture
    def sample_provenance(self):
        """Create sample provenance."""
        return DepthProvenance(
            model_id="test-model",
            license_tier=LicenseTier.COMMERCIAL,
        )

    def test_create_artifact(self, sample_depth_map, sample_provenance):
        """Test creating depth artifact."""
        artifact = DepthArtifact(
            depth_map=sample_depth_map,
            provenance=sample_provenance,
        )
        assert artifact.shape == (480, 640)
        assert artifact.provenance.model_id == "test-model"

    def test_artifact_with_metric_depth(self, sample_depth_map, sample_provenance):
        """Test artifact with metric depth."""
        metric_map = np.random.rand(480, 640).astype(np.float32) * 10  # 0-10 meters
        artifact = DepthArtifact(
            depth_map=sample_depth_map,
            provenance=sample_provenance,
            metric_map_m=metric_map,
        )
        assert artifact.has_metric_depth
        assert artifact.metric_map_m is not None

    def test_artifact_with_confidence(self, sample_depth_map, sample_provenance):
        """Test artifact with confidence map."""
        confidence = np.random.rand(480, 640).astype(np.float32)
        artifact = DepthArtifact(
            depth_map=sample_depth_map,
            provenance=sample_provenance,
            confidence=confidence,
        )
        assert artifact.has_confidence

    def test_artifact_with_intrinsics(self, sample_depth_map, sample_provenance):
        """Test artifact with camera intrinsics."""
        intrinsics = CameraIntrinsics.estimate_from_image(640, 480)
        artifact = DepthArtifact(
            depth_map=sample_depth_map,
            provenance=sample_provenance,
            intrinsics=intrinsics,
        )
        assert artifact.has_intrinsics

    def test_artifact_commercial_safe(self, sample_depth_map):
        """Test commercial safety check."""
        commercial_prov = DepthProvenance(
            model_id="test",
            license_tier=LicenseTier.COMMERCIAL,
        )
        artifact = DepthArtifact(
            depth_map=sample_depth_map,
            provenance=commercial_prov,
        )
        assert artifact.is_commercial_safe

        non_commercial_prov = DepthProvenance(
            model_id="test",
            license_tier=LicenseTier.NON_COMMERCIAL,
        )
        artifact2 = DepthArtifact(
            depth_map=sample_depth_map,
            provenance=non_commercial_prov,
        )
        assert not artifact2.is_commercial_safe

    def test_artifact_compute_stats(self, sample_provenance):
        """Test depth statistics computation."""
        depth_map = np.array([[0.1, 0.5], [0.3, 0.9]], dtype=np.float32)
        artifact = DepthArtifact(
            depth_map=depth_map,
            provenance=sample_provenance,
        )
        stats = artifact.compute_stats()
        assert stats["finite_pct"] == 100.0
        assert stats["min"] == pytest.approx(0.1)
        assert stats["max"] == pytest.approx(0.9)
        assert "median" in stats
        assert "p5" in stats
        assert "p95" in stats

    def test_artifact_to_sidecar_dict(self, sample_depth_map, sample_provenance):
        """Test sidecar JSON generation."""
        artifact = DepthArtifact(
            depth_map=sample_depth_map,
            provenance=sample_provenance,
        )
        sidecar = artifact.to_sidecar_dict()
        assert sidecar["schema_version"] == "1.0.0"
        assert sidecar["shape"] == [480, 640]
        assert sidecar["provenance"]["model_id"] == "test-model"
        # Verify it's JSON serializable
        json_str = json.dumps(sidecar)
        assert len(json_str) > 0

    def test_artifact_content_hash(self, sample_provenance):
        """Test content-addressable hash."""
        depth_map = np.ones((100, 100), dtype=np.float32)
        artifact = DepthArtifact(
            depth_map=depth_map,
            provenance=sample_provenance,
        )
        hash1 = artifact.compute_content_hash()
        assert len(hash1) == 16

        # Same content should produce same hash
        artifact2 = DepthArtifact(
            depth_map=np.ones((100, 100), dtype=np.float32),
            provenance=sample_provenance,
        )
        hash2 = artifact2.compute_content_hash()
        assert hash1 == hash2

        # Different content should produce different hash
        artifact3 = DepthArtifact(
            depth_map=np.zeros((100, 100), dtype=np.float32),
            provenance=sample_provenance,
        )
        hash3 = artifact3.compute_content_hash()
        assert hash1 != hash3

    def test_artifact_validation_non_2d(self, sample_provenance):
        """Test validation rejects non-2D depth maps."""
        with pytest.raises(ValueError, match="must be 2D"):
            DepthArtifact(
                depth_map=np.zeros((10, 10, 3), dtype=np.float32),
                provenance=sample_provenance,
            )

    def test_artifact_validation_shape_mismatch(self, sample_provenance):
        """Test validation rejects mismatched shapes."""
        depth = np.zeros((100, 100), dtype=np.float32)
        metric = np.zeros((50, 50), dtype=np.float32)
        with pytest.raises(ValueError, match="must match depth_map shape"):
            DepthArtifact(
                depth_map=depth,
                provenance=sample_provenance,
                metric_map_m=metric,
            )


class TestDepthArtifactWriter:
    """Test DepthArtifactWriter."""

    @pytest.fixture
    def sample_artifact(self):
        """Create sample artifact for testing."""
        depth_map = np.random.rand(240, 320).astype(np.float32)
        provenance = DepthProvenance(
            model_id="test-model",
            license_tier=LicenseTier.COMMERCIAL,
        )
        return DepthArtifact(depth_map=depth_map, provenance=provenance)

    def test_write_artifact(self, sample_artifact):
        """Test writing artifact to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = DepthArtifactWriter(output_dir=Path(tmpdir))
            paths = writer.write(sample_artifact, stem="test_output")

            # Check all expected files exist
            assert "depth_float" in paths
            assert "depth_preview" in paths
            assert "sidecar" in paths

            assert paths["depth_float"].exists()
            assert paths["depth_preview"].exists()
            assert paths["sidecar"].exists()

            # Verify depth can be reloaded
            loaded = np.load(paths["depth_float"])
            np.testing.assert_array_equal(loaded, sample_artifact.depth_map)

            # Verify sidecar is valid JSON
            with open(paths["sidecar"]) as f:
                sidecar = json.load(f)
            assert sidecar["schema_version"] == "1.0.0"

    def test_write_artifact_with_metric(self):
        """Test writing artifact with metric depth."""
        depth_map = np.random.rand(100, 100).astype(np.float32)
        metric_map = np.random.rand(100, 100).astype(np.float32) * 5
        provenance = DepthProvenance(
            model_id="metric-model",
            license_tier=LicenseTier.EXPERIMENTAL,
        )
        artifact = DepthArtifact(
            depth_map=depth_map,
            provenance=provenance,
            metric_map_m=metric_map,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = DepthArtifactWriter(output_dir=Path(tmpdir))
            paths = writer.write(artifact, stem="metric_test")

            assert "depth_metric" in paths
            assert paths["depth_metric"].exists()

            loaded_metric = np.load(paths["depth_metric"])
            np.testing.assert_array_equal(loaded_metric, metric_map)

    def test_write_artifact_selective(self):
        """Test writing with selective outputs."""
        depth_map = np.random.rand(50, 50).astype(np.float32)
        provenance = DepthProvenance(
            model_id="test",
            license_tier=LicenseTier.COMMERCIAL,
        )
        artifact = DepthArtifact(depth_map=depth_map, provenance=provenance)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Only save float depth, no preview or sidecar
            writer = DepthArtifactWriter(
                output_dir=Path(tmpdir),
                save_float=True,
                save_preview=False,
                save_sidecar=False,
            )
            paths = writer.write(artifact, stem="minimal")

            assert "depth_float" in paths
            assert "depth_preview" not in paths
            assert "sidecar" not in paths


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
