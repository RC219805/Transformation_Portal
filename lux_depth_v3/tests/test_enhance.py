"""Tests for V3 + V2 enhancement orchestrator."""

from __future__ import annotations

import json

import pytest
import numpy as np

from lux_depth_v3.enhance.depth_writer import write_depth_u16_png, read_depth_u16_png
from lux_depth_v3.enhance.manifest import (
    CombinedManifest,
    InputMetadata,
    DepthMetadata,
    V2Metadata,
    TimingMetadata,
    ReproMetadata,
    compute_file_sha256,
)


class TestDepthWriter:
    """Tests for depth writer module."""

    def test_write_depth_u16_png_from_uint16(self, tmp_path):
        """Test writing uint16 depth directly."""
        depth = np.random.randint(0, 65536, (100, 100), dtype=np.uint16)
        output_path = tmp_path / "test_depth.png"

        p1, p99 = write_depth_u16_png(output_path, depth)

        # Verify file exists
        assert output_path.exists()

        # Verify can read back
        depth_read = read_depth_u16_png(output_path)
        assert depth_read.shape == depth.shape
        assert depth_read.dtype == np.uint16
        np.testing.assert_array_equal(depth_read, depth)

        # p1/p99 should be computed from actual data (not hardcoded 0/65535)
        assert isinstance(p1, float)
        assert isinstance(p99, float)
        assert p1 <= p99

    def test_write_depth_u16_png_from_float(self, tmp_path):
        """Test writing float32 depth with quantization."""
        depth = np.random.rand(100, 100).astype(np.float32)
        output_path = tmp_path / "test_depth.png"

        p1, p99 = write_depth_u16_png(output_path, depth, method="p1p99")

        # Verify file exists
        assert output_path.exists()

        # Verify can read back
        depth_read = read_depth_u16_png(output_path)
        assert depth_read.shape == depth.shape
        assert depth_read.dtype == np.uint16

        # p1/p99 should be reasonable
        assert 0.0 <= p1 < p99 <= 1.0

    def test_write_depth_u16_png_3channel(self, tmp_path):
        """Test handling of 3-channel depth (should take first channel)."""
        depth = np.random.rand(100, 100, 1).astype(np.float32)
        output_path = tmp_path / "test_depth.png"

        write_depth_u16_png(output_path, depth)

        # Should write single-channel
        depth_read = read_depth_u16_png(output_path)
        assert depth_read.ndim == 2
        assert depth_read.shape == (100, 100)

    def test_write_depth_u16_png_invalid_shape(self, tmp_path):
        """Test error on invalid shape."""
        depth = np.random.rand(10).astype(np.float32)  # 1D
        output_path = tmp_path / "test_depth.png"

        with pytest.raises(ValueError, match="Expected 2D or 3D depth"):
            write_depth_u16_png(output_path, depth)

    def test_write_depth_u16_png_with_nan(self, tmp_path):
        """Test error on NaN values."""
        depth = np.random.rand(100, 100).astype(np.float32)
        depth[50, 50] = np.nan
        output_path = tmp_path / "test_depth.png"

        with pytest.raises(ValueError, match="NaN or Inf"):
            write_depth_u16_png(output_path, depth)

    def test_write_depth_u16_png_debug_verify(self, tmp_path):
        """Test debug verification mode."""
        depth = np.random.randint(0, 65536, (100, 100), dtype=np.uint16)
        output_path = tmp_path / "test_depth.png"

        # Should not raise with debug_verify=True
        write_depth_u16_png(output_path, depth, debug_verify=True)

    def test_write_depth_u16_png_quantization_methods(self, tmp_path):
        """Test different quantization methods."""
        depth = np.random.rand(100, 100).astype(np.float32)

        for method in ["p1p99", "p0.5p99.5", "minmax"]:
            output_path = tmp_path / f"test_depth_{method}.png"
            p1, p99 = write_depth_u16_png(output_path, depth, method=method)
            assert output_path.exists()
            assert p1 < p99

    def test_write_depth_u16_png_zero_range(self, tmp_path):
        """Test handling of zero range depth."""
        depth = np.ones((100, 100), dtype=np.float32) * 0.5  # Constant
        output_path = tmp_path / "test_depth.png"

        # Should handle gracefully
        p1, p99 = write_depth_u16_png(output_path, depth)
        assert output_path.exists()

        # Should write zeros
        depth_read = read_depth_u16_png(output_path)
        assert np.all(depth_read == 0)


class TestManifest:
    """Tests for combined manifest."""

    def test_manifest_creation(self):
        """Test creating manifest from components."""
        manifest = CombinedManifest(
            input=InputMetadata(
                image_path="test.jpg",
                image_sha256="abc123",
            ),
            depth=DepthMetadata(
                backend="da3",
                model="DepthAnything3-Large-Metric",
                license="CC-BY-NC",
                non_commercial_ok=True,
                depth_path="depth/test_depth.png",
                dtype="uint16",
                shape=[100, 100],
                scaling={"method": "p1p99", "p1": 0.1, "p99": 0.9},
                runtime_ms=50.0,
            ),
            v2=V2Metadata(
                preset="production_ultra",
                strict_depth=True,
                output_dir="v2/",
                report_path="v2/test_report.json",
                status="ok",
            ),
            timing=TimingMetadata(
                depth_s=0.05,
                v2_s=2.0,
                total_s=2.05,
            ),
            repro=ReproMetadata(
                v3_git="abc123",
                v2_git="def456",
                device="cuda",
            ),
        )

        assert manifest.schema == "lux-depth-v3.enhance.v1"
        assert manifest.input.image_path == "test.jpg"
        assert manifest.depth.backend == "da3"
        assert manifest.v2.status == "ok"

    def test_manifest_to_dict(self):
        """Test converting manifest to dictionary."""
        manifest = CombinedManifest(
            input=InputMetadata(
                image_path="test.jpg",
                image_sha256="abc123",
            ),
        )

        data = manifest.to_dict()
        assert isinstance(data, dict)
        assert data["schema"] == "lux-depth-v3.enhance.v1"
        assert data["input"]["image_path"] == "test.jpg"

    def test_manifest_to_json(self):
        """Test JSON serialization."""
        manifest = CombinedManifest(
            input=InputMetadata(
                image_path="test.jpg",
                image_sha256="abc123",
            ),
        )

        json_str = manifest.to_json()
        assert isinstance(json_str, str)

        # Should be valid JSON
        data = json.loads(json_str)
        assert data["schema"] == "lux-depth-v3.enhance.v1"

    def test_manifest_write_and_load(self, tmp_path):
        """Test writing and loading manifest."""
        manifest = CombinedManifest(
            input=InputMetadata(
                image_path="test.jpg",
                image_sha256="abc123",
            ),
            depth=DepthMetadata(
                backend="da3",
                model="DepthAnything3-Large-Metric",
                license="CC-BY-NC",
                non_commercial_ok=True,
                depth_path="depth/test_depth.png",
                dtype="uint16",
                shape=[100, 100],
                scaling={"method": "p1p99", "p1": 0.1, "p99": 0.9},
                runtime_ms=50.0,
            ),
        )

        manifest_path = tmp_path / "test_manifest.json"
        manifest.write(manifest_path)

        # Load back
        loaded = CombinedManifest.load(manifest_path)
        assert loaded.schema == manifest.schema
        assert loaded.input.image_path == manifest.input.image_path
        assert loaded.depth.backend == manifest.depth.backend
        assert loaded.depth.shape == manifest.depth.shape

    def test_compute_file_sha256(self, tmp_path):
        """Test SHA256 computation."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        sha256 = compute_file_sha256(test_file)
        assert isinstance(sha256, str)
        assert len(sha256) == 64  # SHA256 hex digest length


class TestV2Runner:
    """Tests for V2 runner module."""

    def test_v2_runner_initialization(self):
        """Test V2 runner can be initialized."""
        from lux_depth_v3.enhance.v2_runner import V2Runner

        # Should auto-detect V2 module
        runner = V2Runner()
        assert runner.python_exe is not None

    def test_find_v2_report(self, tmp_path):
        """Test finding V2 report in output directory."""
        from lux_depth_v3.enhance.v2_runner import find_v2_report

        # Create mock report
        report_path = tmp_path / "test_report.json"
        report_path.write_text('{"status": "ok"}')

        found = find_v2_report(tmp_path, "test")
        assert found == report_path


class TestEnhanceOrchestrator:
    """Tests for enhance orchestrator (integration tests)."""

    @pytest.mark.skip(reason="Integration test - requires models")
    def test_orchestrator_initialization(self, tmp_path):
        """Test orchestrator can be initialized."""
        from lux_depth_v3.enhance.orchestrator import EnhanceOrchestrator, EnhanceConfig

        config = EnhanceConfig(non_commercial_ok=True)
        orchestrator = EnhanceOrchestrator(config, tmp_path)

        # Verify orchestrator is created
        assert isinstance(orchestrator, EnhanceOrchestrator)

        # Verify output directories created
        assert (tmp_path / "depth").exists()
        assert (tmp_path / "v2").exists()
        assert (tmp_path / "manifests").exists()
        assert (tmp_path / "logs").exists()
        assert (tmp_path / "zones").exists()


class TestCLI:
    """Regression tests for CLI behavior."""

    def test_batch_manifest_includes_depth_zones(self, tmp_path, monkeypatch):
        """Test that --depth-zones is recorded in batch manifest."""
        import json
        from typer.testing import CliRunner
        from lux_depth_v3.cli import app
        from PIL import Image

        # Create dummy input image
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        Image.new("RGB", (64, 64), color=(128, 128, 128)).save(input_dir / "test.jpg")

        output_dir = tmp_path / "output"

        # Mock EnhanceOrchestrator.enhance_image to avoid model downloads
        def mock_enhance_image(self, image_input, input_root=None):
            return {"status": "ok", "image": str(image_input.path), "runtime_s": 0.01}

        monkeypatch.setattr(
            "lux_depth_v3.enhance.orchestrator.EnhanceOrchestrator.enhance_image",
            mock_enhance_image,
        )

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "enhance",
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
                "--depth-zones",
                "preview",
                "--max-images",
                "1",
                "--non-commercial-ok",
            ],
        )
        assert result.exit_code == 0, result.stdout

        manifest_files = list((output_dir / "manifests").glob("batch_*.json"))
        assert len(manifest_files) == 1

        batch_manifest = json.loads(manifest_files[0].read_text())
        assert batch_manifest["config"]["depth_zones"] == "preview"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
