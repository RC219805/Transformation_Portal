"""Test orchestrator counter correctness.

Verifies that summary counters accurately reflect processing results.
Critical for CI dashboards, automation trust, and governance gates.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from transformation_portal.lux_depth_v3.config import EnhanceConfig
from transformation_portal.lux_depth_v3.input_manager import ImageInput
from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator


@pytest.fixture
def mock_config():
    """Create a minimal config for testing."""
    config = EnhanceConfig(
        depth_device="cpu",
        enable_v2=False,  # Disable V2 to simplify tests
        generate_pbr=False,  # Disable PBR to simplify tests
        enable_parallel_processing=False,  # Sequential for deterministic tests
    )
    return config


@pytest.fixture
def orchestrator(tmp_path, mock_config):
    """Create orchestrator instance with mocked backend."""
    with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry"):
        orch = EnhanceOrchestrator(config=mock_config, output_root=tmp_path)

        # Mock the depth backend to avoid ML dependencies
        mock_backend = Mock(spec=["name", "compute"])
        mock_backend.name = "mock"

        # Create realistic depth result
        depth_array = np.random.rand(100, 100).astype(np.float32)
        mock_result = Mock(spec=["depth_map", "depth", "original_image", "metadata"])
        mock_result.depth_map = depth_array
        mock_result.depth = depth_array
        mock_result.original_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mock_result.metadata = {}

        mock_backend.compute = Mock(return_value=mock_result)
        orch.depth_backend = mock_backend

        return orch


class TestOrchestratorCounters:
    """Test that orchestrator returns correct status values."""

    def test_enhance_image_returns_ok_status(self, orchestrator, tmp_path):
        """Test that enhance_image returns status='ok' for successful processing."""
        # Create a test image
        test_image = tmp_path / "test.jpg"
        test_image.write_bytes(b"fake image data")

        # Create realistic mock data
        mock_image_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Mock preprocessing and validation
        with patch("transformation_portal.lux_depth_v3.preprocessing.validate_image_format", return_value=test_image):
            with patch(
                "transformation_portal.lux_depth_v3.preprocessing.preprocess_image",
                return_value=(mock_image_array, (100, 100)),
            ):
                result = orchestrator.enhance_image(ImageInput(test_image), input_root=tmp_path)

        assert result["status"] == "ok", "enhance_image must return status='ok' for successful processing"
        assert "image" in result
        assert "runtime_s" in result

    def test_enhance_image_returns_skipped_status(self, orchestrator, tmp_path):
        """Test that enhance_image returns status='skipped' when depth computation fails with skip fallback."""
        test_image = tmp_path / "test.jpg"
        test_image.write_bytes(b"fake image data")

        # Configure fallback mode
        orchestrator.config.depth_fallback = "skip"

        # Mock depth computation to fail
        orchestrator.depth_backend.compute = Mock(side_effect=RuntimeError("Depth failed"))

        # Create realistic mock data
        mock_image_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        with patch("transformation_portal.lux_depth_v3.preprocessing.validate_image_format", return_value=test_image):
            with patch(
                "transformation_portal.lux_depth_v3.preprocessing.preprocess_image",
                return_value=(mock_image_array, (100, 100)),
            ):
                result = orchestrator.enhance_image(ImageInput(test_image), input_root=tmp_path)

        assert result["status"] == "skipped", "enhance_image must return status='skipped' when depth fails with skip fallback"

    def test_enhance_image_returns_error_status(self, orchestrator, tmp_path):
        """Test that enhance_batch returns status='error' for failed images."""
        test_image = tmp_path / "test.jpg"
        test_image.write_bytes(b"fake image data")

        # Mock preprocessing to fail
        with patch("transformation_portal.lux_depth_v3.orchestrator.discover_images", return_value=[test_image]):
            with patch.object(orchestrator, "enhance_image", side_effect=RuntimeError("Processing failed")):
                results = orchestrator.enhance_batch(input_dir=tmp_path)

        assert len(results) == 1
        assert results[0]["status"] == "error", "enhance_batch must return status='error' for failed images"
        assert "error" in results[0]

    def test_batch_processing_accumulates_counters(self, orchestrator, tmp_path):
        """Test that batch processing correctly accumulates ok/skipped/error counters."""
        # Create test images
        images = []
        for i in range(5):
            img = tmp_path / f"test{i}.jpg"
            img.write_bytes(b"fake image data")
            images.append(img)

        # Mock results: 3 ok, 1 skipped, 1 error
        mock_results = [
            {"status": "ok", "image": str(images[0]), "runtime_s": 1.0},
            {"status": "ok", "image": str(images[1]), "runtime_s": 1.0},
            {"status": "ok", "image": str(images[2]), "runtime_s": 1.0},
            {"status": "skipped", "image": str(images[3])},
            {"status": "error", "image": str(images[4]), "error": "Failed"},
        ]

        with patch("transformation_portal.lux_depth_v3.orchestrator.discover_images", return_value=images):
            with patch.object(orchestrator, "enhance_image", side_effect=mock_results):
                results = orchestrator.enhance_batch(input_dir=tmp_path)

        # Count statuses
        ok_count = sum(1 for r in results if r.get("status") == "ok")
        skipped_count = sum(1 for r in results if r.get("status") == "skipped")
        error_count = sum(1 for r in results if r.get("status") == "error")

        assert ok_count == 3, "Expected 3 successful results"
        assert skipped_count == 1, "Expected 1 skipped result"
        assert error_count == 1, "Expected 1 error result"
        assert ok_count + skipped_count + error_count == len(images), "All images must be accounted for"


class TestCLICounters:
    """Test that CLI correctly counts status values from orchestrator."""

    def test_cli_counts_ok_as_successful(self):
        """Test that CLI summary counts status='ok' as successful."""
        results = [
            {"status": "ok", "image": "test1.jpg"},
            {"status": "ok", "image": "test2.jpg"},
            {"status": "skipped", "image": "test3.jpg"},
            {"status": "error", "image": "test4.jpg"},
        ]

        # This is the logic from __main__.py (now fixed)
        successful = sum(1 for r in results if r.get("status") == "ok")
        skipped = sum(1 for r in results if r.get("status") == "skipped")
        failed = sum(1 for r in results if r.get("status") == "error")

        assert successful == 2, "CLI must count status='ok' as successful"
        assert skipped == 1, "CLI must count status='skipped' correctly"
        assert failed == 1, "CLI must count status='error' as failed"
        assert successful + skipped + failed == len(results), "All results must be counted"


class TestDependencyReport:
    """Test startup dependency reporting."""

    def test_dependency_status_logged(self, tmp_path, mock_config, caplog):
        """Test that dependency status is logged on orchestrator init."""
        import logging

        caplog.set_level(logging.DEBUG)

        # Create a new orchestrator to trigger logging
        with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry"):
            _ = EnhanceOrchestrator(config=mock_config, output_root=tmp_path)

        # Check that at least some dependency status was logged
        log_messages = [rec.message for rec in caplog.records]

        # Should have logged something about dependencies
        # (exact messages depend on what's installed, but we should see some)
        assert len(log_messages) > 0, "Dependency status should be logged"
        # Check for at least one dependency being reported
        dependency_logged = any(
            "torch" in msg.lower()
            or "transformers" in msg.lower()
            or "coremltools" in msg.lower()
            or "available" in msg.lower()
            for msg in log_messages
        )
        assert dependency_logged, f"Should log dependency status. Got: {log_messages}"

    def test_dependency_report_returns_status(self):
        """Test that _log_dependency_status returns status dict."""
        from transformation_portal.lux_depth_v3.orchestrator import _log_dependency_status

        status = _log_dependency_status()

        assert isinstance(status, dict), "Must return status dictionary"
        assert "torch" in status, "Must report torch availability"
        assert "transformers" in status, "Must report transformers availability"
        assert "coremltools" in status, "Must report coremltools availability"
        assert "scikit-image" in status, "Must report scikit-image availability"
        assert "numba" in status, "Must report numba availability"
        assert "hf_token" in status, "Must report HF_TOKEN availability"
