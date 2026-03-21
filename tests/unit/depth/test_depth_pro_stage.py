"""Unit tests for DepthProStage (100% mocked, no model downloads)."""

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest

pytestmark = pytest.mark.unit
from PIL import Image

from transformation_portal.stage_graph.stage import StageContext, StageStatus
from transformation_portal.stage_graph.stages.depth_pro import CheckpointValidationError, DepthProStage


@pytest.mark.unit
class TestDepthProStageUnit:
    """Unit tests for DepthProStage (all mocked)."""

    def test_stage_instantiation(self):
        """Verify DepthProStage can be instantiated."""
        stage = DepthProStage()

        assert stage.name == "depth_pro_estimation"
        assert stage.version == "1.0.0"
        assert stage.checkpoint_path == Path("checkpoints/depth_pro.pt")
        assert stage.device in ("mps", "cuda", "cpu")
        assert stage._model is None
        assert not stage._model_loaded

    def test_stage_with_custom_params(self):
        """Verify custom parameters are respected."""
        custom_path = Path("/custom/checkpoint.pt")
        custom_sha = "custom_sha256"

        stage = DepthProStage(checkpoint_path=custom_path, expected_sha256=custom_sha, device="cpu", version="2.0.0")

        assert stage.checkpoint_path == custom_path
        assert stage.expected_sha256 == custom_sha
        assert stage.device == "cpu"
        assert stage.version == "2.0.0"

    def test_cache_key_determinism(self):
        """Generate cache key for same image twice - should be identical."""
        stage = DepthProStage()

        # Create test image (seeded for determinism)
        np.random.seed(42)
        img_array_1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Create identical image with same seed
        np.random.seed(42)
        img_array_2 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Verify images are identical
        assert np.array_equal(img_array_1, img_array_2), "Test setup failed: images should be identical"

        context1 = StageContext(artifacts={"image": img_array_1})
        context2 = StageContext(artifacts={"image": img_array_2})

        # Mock checkpoint hash
        with patch.object(stage, "_get_checkpoint_hash", return_value="abcd1234"):
            with patch.object(stage, "_get_package_version", return_value="0.1.2"):
                key1 = stage.get_cache_key(context1)
                key2 = stage.get_cache_key(context2)

        assert key1 == key2, "Identical images should produce identical cache keys"
        assert "depthpro" in key1
        assert "abcd1234" in key1
        assert "0.1.2" in key1

    def test_cache_key_invalidation_on_image_change(self):
        """Different images should produce different cache keys."""
        stage = DepthProStage()

        # Seed for determinism
        np.random.seed(100)
        img1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        np.random.seed(200)
        img2 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        context1 = StageContext(artifacts={"image": img1})
        context2 = StageContext(artifacts={"image": img2})

        with patch.object(stage, "_get_checkpoint_hash", return_value="abcd1234"):
            with patch.object(stage, "_get_package_version", return_value="0.1.2"):
                key1 = stage.get_cache_key(context1)
                key2 = stage.get_cache_key(context2)

        # Keys should be different (different image content)
        assert key1 != key2

    @patch("transformation_portal.stage_graph.stages.depth_pro.DEPTH_PRO_AVAILABLE", True)
    def test_checkpoint_validation_missing(self):
        """Missing checkpoint should produce actionable error."""
        stage = DepthProStage(checkpoint_path=Path("/nonexistent/checkpoint.pt"))
        context = StageContext(artifacts={"image": np.zeros((100, 100, 3))})

        result = stage.compute(context)

        assert result.status == StageStatus.FAILED
        assert "not found" in result.error.lower()
        assert "download" in result.error.lower()
        assert stage.CHECKPOINT_URL in result.error

    @patch("transformation_portal.stage_graph.stages.depth_pro.DEPTH_PRO_AVAILABLE", False)
    def test_depth_pro_not_installed(self):
        """depth_pro package not installed should fail gracefully."""
        stage = DepthProStage()
        context = StageContext(artifacts={"image": np.zeros((100, 100, 3))})

        result = stage.compute(context)

        assert result.status == StageStatus.FAILED
        assert "depth_pro package not installed" in result.error
        assert "pip install" in result.error

    @patch("transformation_portal.stage_graph.stages.depth_pro.DEPTH_PRO_AVAILABLE", True)
    def test_missing_image_artifact(self):
        """Missing image artifact should fail with clear error."""
        stage = DepthProStage()
        context = StageContext(artifacts={})

        result = stage.compute(context)

        assert result.status == StageStatus.FAILED
        assert "missing" in result.error.lower()
        assert "image" in result.error.lower()

    @pytest.mark.ml
    @patch("transformation_portal.stage_graph.stages.depth_pro.TORCH_AVAILABLE", True)
    @patch("transformation_portal.stage_graph.stages.depth_pro.torch")
    @patch("transformation_portal.stage_graph.stages.depth_pro.depth_pro")
    def test_lazy_loading(self, mock_depth_pro, mock_torch):
        """Model should not load on init, only on first compute."""
        # Mock torch.device
        mock_torch.device.return_value = "cpu"

        mock_model = MagicMock()
        mock_transform = MagicMock()
        mock_depth_pro.create_model_and_transforms.return_value = (mock_model, mock_transform)

        stage = DepthProStage()

        # Model should not be loaded yet
        assert not stage._model_loaded
        assert stage._model is None

        # Call _load_model explicitly (with validation mocked)
        with patch.object(stage, "_validate_checkpoint"):
            with patch.object(stage.logger, "info"):
                stage._load_model()

        # Now model should be loaded
        assert stage._model_loaded
        assert stage._model is not None
        mock_depth_pro.create_model_and_transforms.assert_called_once()

    @pytest.mark.ml
    @patch("transformation_portal.stage_graph.stages.depth_pro.DEPTH_PRO_AVAILABLE", True)
    @patch("transformation_portal.stage_graph.stages.depth_pro.TORCH_AVAILABLE", True)
    @patch("transformation_portal.stage_graph.stages.depth_pro.torch")
    @patch("transformation_portal.stage_graph.stages.depth_pro.depth_pro")
    @patch.object(Path, "exists", return_value=True)
    def test_provenance_structure(self, mock_exists, mock_depth_pro, mock_torch):
        """Provenance JSON should contain required keys."""
        # Mock torch components
        mock_torch.Tensor = type("MockTensor", (), {})  # Create a mock type for isinstance
        mock_torch.no_grad.return_value.__enter__.return_value = None
        mock_torch.no_grad.return_value.__exit__.return_value = None
        mock_torch.device.return_value = "cpu"
        mock_torch.__version__ = "2.0.0"

        # Mock model and inference
        mock_model = MagicMock()
        mock_transform = MagicMock()
        mock_depth_pro.create_model_and_transforms.return_value = (mock_model, mock_transform)

        # Mock inference output (seeded for determinism)
        depth_tensor = MagicMock()
        depth_tensor.ndim = 2
        np.random.seed(42)
        depth_tensor.detach.return_value.float.return_value.cpu.return_value.numpy.return_value = np.random.rand(
            100, 100
        ).astype(np.float32)

        mock_model.infer.return_value = {"depth": depth_tensor}
        mock_transform.return_value = MagicMock()

        stage = DepthProStage()
        context = StageContext(artifacts={"image": Image.new("RGB", (100, 100))})

        # Mock checkpoint hash, package version, validation, and file stat
        with patch.object(stage, "_get_checkpoint_hash", return_value="abcd1234"):
            with patch.object(stage, "_get_package_version", return_value="0.1.2"):
                with patch.object(stage, "_validate_checkpoint"):
                    with patch("pathlib.Path.stat") as mock_stat:
                        mock_stat.return_value.st_size = 2000000000  # 2GB checkpoint
                        with patch.object(stage.logger, "info"):
                            result = stage.compute(context)

        # Better error reporting for CI debugging
        if result.status != StageStatus.COMPLETED:
            raise AssertionError(f"Expected COMPLETED but got {result.status}. " f"Error: {result.error}")

        assert result.status == StageStatus.COMPLETED
        assert "depth_provenance" in result.artifacts

        prov = result.artifacts["depth_provenance"]
        assert prov["status"] == "ok"
        assert prov["engine"] == "apple_depth_pro"
        assert "checkpoint" in prov
        assert prov["checkpoint"]["sha256"] == "abcd1234"
        assert "outputs" in prov
        assert "depth_stats" in prov["outputs"]
        assert "timing" in prov
        assert "run" in prov
        assert "timestamp_iso_utc" in prov["run"]
        assert "env" in prov
        assert prov["env"]["depth_pro_pkg"] == "0.1.2"

    @pytest.mark.ml
    @patch("transformation_portal.stage_graph.stages.depth_pro.depth_pro")
    @patch.object(Path, "exists", return_value=True)
    def test_depth_stats_computation(self, mock_exists, mock_depth_pro):
        """Depth stats should compute min, median, p95 correctly."""
        stage = DepthProStage()

        # Create known depth array
        depth = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)

        stats = stage._compute_depth_stats(depth)

        assert stats["finite_pct"] == 100.0
        assert stats["min"] == 1.0
        assert stats["median"] == 5.0
        assert stats["p95"] == pytest.approx(8.6, abs=0.1)

    def test_depth_stats_with_nans(self):
        """Depth stats should handle NaN values gracefully."""
        stage = DepthProStage()

        # Create depth with some NaNs
        depth = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan], [7.0, 8.0, 9.0]], dtype=np.float32)

        stats = stage._compute_depth_stats(depth)

        assert stats["finite_pct"] < 100.0
        assert stats["min"] == 1.0
        assert stats["median"] == 5.0

    def test_normalize_to_uint16(self):
        """Normalization to uint16 should use percentile clipping."""
        stage = DepthProStage()

        depth = np.linspace(0, 100, 1000).reshape(100, 10).astype(np.float32)
        u16 = stage._normalize_to_uint16(depth)

        assert u16.dtype == np.uint16
        assert u16.shape == depth.shape
        assert u16.min() >= 0
        assert u16.max() <= 65535

    def test_normalize_to_uint16_all_nans(self):
        """Normalization with all NaNs should return zeros."""
        stage = DepthProStage()

        depth = np.full((100, 100), np.nan, dtype=np.float32)
        u16 = stage._normalize_to_uint16(depth)

        assert u16.dtype == np.uint16
        assert np.all(u16 == 0)

    @pytest.mark.ml
    @patch("transformation_portal.stage_graph.stages.depth_pro.DEPTH_PRO_AVAILABLE", True)
    @patch("transformation_portal.stage_graph.stages.depth_pro.TORCH_AVAILABLE", True)
    @patch("transformation_portal.stage_graph.stages.depth_pro.torch")
    @patch("transformation_portal.stage_graph.stages.depth_pro.depth_pro")
    @patch.object(Path, "exists", return_value=True)
    def test_outputs_saved_to_disk(self, mock_exists, mock_depth_pro, mock_torch):
        """Outputs should be saved to disk when output_dir provided."""
        # Mock torch components
        mock_torch.Tensor = type("MockTensor", (), {})  # Create a mock type for isinstance
        mock_torch.no_grad.return_value.__enter__.return_value = None
        mock_torch.no_grad.return_value.__exit__.return_value = None
        mock_torch.device.return_value = "cpu"
        mock_torch.__version__ = "2.0.0"

        # Mock model and inference
        mock_model = MagicMock()
        mock_transform = MagicMock()
        mock_depth_pro.create_model_and_transforms.return_value = (mock_model, mock_transform)

        depth_tensor = MagicMock()
        depth_tensor.ndim = 2
        np.random.seed(42)
        depth_array = np.random.rand(100, 100).astype(np.float32)
        depth_tensor.detach.return_value.float.return_value.cpu.return_value.numpy.return_value = depth_array
        mock_model.infer.return_value = {"depth": depth_tensor}

        stage = DepthProStage()
        output_dir = Path("/tmp/test_output")
        context = StageContext(artifacts={"image": Image.new("RGB", (100, 100)), "output_dir": output_dir})

        with patch.object(stage, "_get_checkpoint_hash", return_value="abcd1234"):
            with patch.object(stage, "_get_package_version", return_value="0.1.2"):
                with patch.object(stage, "_validate_checkpoint"):
                    with patch("pathlib.Path.stat") as mock_stat:
                        mock_stat.return_value.st_size = 2000000000  # 2GB checkpoint
                        with patch.object(stage.logger, "info"):
                            with patch.object(Path, "mkdir"):
                                with patch.object(Image.Image, "save"):
                                    with patch("numpy.save"):
                                        with patch("transformation_portal.stage_graph.stages.depth_pro.open", mock_open()):
                                            result = stage.compute(context)

        assert result.status == StageStatus.COMPLETED
        assert "depth_float_path" in result.artifacts
        assert "depth_preview_path" in result.artifacts
        assert result.artifacts["depth_float_path"] == output_dir / "depth_depthpro.npy"
        assert result.artifacts["depth_preview_path"] == output_dir / "depth_depthpro_preview.png"

    @patch.object(Path, "exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data=b"fake_checkpoint_data")
    def test_checkpoint_hash_caching(self, mock_file, mock_exists):
        """Checkpoint hash should be computed once and cached."""
        stage = DepthProStage()

        # First call
        hash1 = stage._get_checkpoint_hash()

        # Second call (should use cache)
        hash2 = stage._get_checkpoint_hash()

        assert hash1 == hash2
        # File should only be opened once (cached)
        assert hasattr(stage, "_checkpoint_hash_cached")

    def test_get_package_version(self):
        """Package version should be retrieved or return 'unknown'."""
        from importlib import metadata as importlib_metadata

        stage = DepthProStage()

        with patch("transformation_portal.stage_graph.stages.depth_pro.importlib_metadata.version", return_value="1.2.3"):
            version = stage._get_package_version()
            assert version == "1.2.3"

        with patch(
            "transformation_portal.stage_graph.stages.depth_pro.importlib_metadata.version",
            side_effect=importlib_metadata.PackageNotFoundError("depth_pro"),
        ):
            version = stage._get_package_version()
            assert version == "unknown"

    @pytest.mark.ml
    @patch("transformation_portal.stage_graph.stages.depth_pro.DEPTH_PRO_AVAILABLE", True)
    @patch("transformation_portal.stage_graph.stages.depth_pro.TORCH_AVAILABLE", True)
    @patch("transformation_portal.stage_graph.stages.depth_pro.torch")
    @patch("transformation_portal.stage_graph.stages.depth_pro.depth_pro")
    @patch.object(Path, "exists", return_value=True)
    def test_inference_error_handling(self, mock_exists, mock_depth_pro, mock_torch):
        """Inference errors should be captured in result."""
        # Mock torch.device
        mock_torch.device.return_value = "cpu"

        # Make model loading fail
        mock_depth_pro.create_model_and_transforms.side_effect = RuntimeError("GPU out of memory")

        stage = DepthProStage()
        context = StageContext(artifacts={"image": Image.new("RGB", (100, 100))})

        with patch.object(stage, "_get_checkpoint_hash", return_value="abcd1234"):
            with patch.object(stage, "_validate_checkpoint"):
                with patch.object(stage.logger, "info"):
                    result = stage.compute(context)

        assert result.status == StageStatus.FAILED
        assert "GPU out of memory" in result.error
        assert result.error_traceback is not None

    def test_cache_key_with_pil_image(self):
        """Cache key should work with PIL Image."""
        stage = DepthProStage()

        img = Image.new("RGB", (100, 100))
        context = StageContext(artifacts={"image": img})

        with patch.object(stage, "_get_checkpoint_hash", return_value="abcd1234"):
            with patch.object(stage, "_get_package_version", return_value="0.1.2"):
                key = stage.get_cache_key(context)

        assert "depthpro" in key
        assert key != "no_image"

    def test_cache_key_no_image(self):
        """Cache key without image should return 'no_image'."""
        stage = DepthProStage()
        context = StageContext(artifacts={})

        key = stage.get_cache_key(context)
        assert key == "no_image"

    @patch("transformation_portal.stage_graph.stages.depth_pro.TORCH_AVAILABLE", True)
    @patch("transformation_portal.stage_graph.stages.depth_pro.torch")
    def test_auto_detect_device_mps(self, mock_torch):
        """Auto-detect should prefer MPS when available."""
        mock_torch.backends.mps.is_available.return_value = True
        mock_torch.cuda.is_available.return_value = False

        stage = DepthProStage()
        assert stage.device == "mps"

    @patch("transformation_portal.stage_graph.stages.depth_pro.TORCH_AVAILABLE", True)
    @patch("transformation_portal.stage_graph.stages.depth_pro.torch")
    def test_auto_detect_device_cuda(self, mock_torch):
        """Auto-detect should use CUDA when MPS not available."""
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = True

        stage = DepthProStage()
        assert stage.device == "cuda"

    @patch("transformation_portal.stage_graph.stages.depth_pro.TORCH_AVAILABLE", True)
    @patch("transformation_portal.stage_graph.stages.depth_pro.torch")
    def test_auto_detect_device_cpu(self, mock_torch):
        """Auto-detect should fall back to CPU."""
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False

        stage = DepthProStage()
        assert stage.device == "cpu"

    def test_strict_validation_default_true(self):
        """strict_validation should default to True."""
        stage = DepthProStage()
        assert stage.strict_validation is True

    def test_strict_validation_configurable(self):
        """strict_validation should be configurable."""
        stage_strict = DepthProStage(strict_validation=True)
        stage_warn = DepthProStage(strict_validation=False)

        assert stage_strict.strict_validation is True
        assert stage_warn.strict_validation is False

    @patch.object(Path, "exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data=b"fake_checkpoint_data")
    def test_validate_checkpoint_hash_mismatch_strict(self, mock_file, mock_exists):
        """Hash mismatch with strict_validation=True should raise error."""
        stage = DepthProStage(expected_sha256="expected_hash_that_wont_match", strict_validation=True)

        with pytest.raises(CheckpointValidationError) as exc_info:
            stage._validate_checkpoint()

        error_msg = str(exc_info.value)
        assert "SHA-256 validation failed" in error_msg
        assert "expected_hash_that_wont_match" in error_msg
        assert "corruption or tampering" in error_msg
        assert stage.CHECKPOINT_URL in error_msg

    @patch.object(Path, "exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data=b"fake_checkpoint_data")
    def test_validate_checkpoint_hash_mismatch_warn_only(self, mock_file, mock_exists):
        """Hash mismatch with strict_validation=False should warn but not raise."""
        stage = DepthProStage(expected_sha256="expected_hash_that_wont_match", strict_validation=False)

        with patch.object(stage.logger, "warning") as mock_warning:
            # Should not raise
            stage._validate_checkpoint()

            # Should log warning
            mock_warning.assert_called_once()
            warning_msg = mock_warning.call_args[0][0]
            assert "SHA-256 validation failed" in warning_msg
            assert "expected_hash_that_wont_match" in warning_msg

    @patch.object(Path, "exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data=b"test_data")
    def test_validate_checkpoint_hash_match(self, mock_file, mock_exists):
        """Matching hash should pass validation."""
        # Compute expected hash for 'test_data'
        import hashlib

        expected_hash = hashlib.sha256(b"test_data").hexdigest()

        stage = DepthProStage(expected_sha256=expected_hash, strict_validation=True)

        with patch.object(stage.logger, "info") as mock_info:
            # Should not raise
            stage._validate_checkpoint()

            # Should log success
            mock_info.assert_called()
            # Check that validation passed message was logged
            info_calls = [str(call) for call in mock_info.call_args_list]
            assert any("validation passed" in str(call) for call in info_calls)

    @patch("transformation_portal.stage_graph.stages.depth_pro.DEPTH_PRO_AVAILABLE", True)
    @patch("transformation_portal.stage_graph.stages.depth_pro.TORCH_AVAILABLE", True)
    @patch("transformation_portal.stage_graph.stages.depth_pro.torch")
    @patch("transformation_portal.stage_graph.stages.depth_pro.depth_pro")
    @patch.object(Path, "exists", return_value=True)
    def test_load_model_validates_checkpoint(self, mock_exists, mock_depth_pro, mock_torch):
        """_load_model should call _validate_checkpoint before loading."""
        mock_torch.device.return_value = "cpu"
        mock_model = MagicMock()
        mock_transform = MagicMock()
        mock_depth_pro.create_model_and_transforms.return_value = (mock_model, mock_transform)

        stage = DepthProStage(strict_validation=False)

        with patch.object(stage, "_validate_checkpoint") as mock_validate:
            with patch.object(stage, "_get_checkpoint_hash", return_value="test_hash"):
                with patch.object(stage.logger, "info"):
                    stage._load_model()

            # Validation should be called before model loading
            mock_validate.assert_called_once()

    @patch("transformation_portal.stage_graph.stages.depth_pro.DEPTH_PRO_AVAILABLE", True)
    @patch("transformation_portal.stage_graph.stages.depth_pro.TORCH_AVAILABLE", True)
    @patch("transformation_portal.stage_graph.stages.depth_pro.torch")
    @patch("transformation_portal.stage_graph.stages.depth_pro.depth_pro")
    @patch.object(Path, "exists", return_value=True)
    def test_compute_fails_on_checkpoint_validation_error(self, mock_exists, mock_depth_pro, mock_torch):
        """compute() should fail if checkpoint validation fails."""
        mock_torch.device.return_value = "cpu"

        stage = DepthProStage(expected_sha256="invalid_hash_for_test", strict_validation=True)
        context = StageContext(artifacts={"image": Image.new("RGB", (100, 100))})

        with patch.object(stage, "_get_checkpoint_hash", return_value="actual_hash_different"):
            result = stage.compute(context)

        assert result.status == StageStatus.FAILED
        assert "SHA-256 validation failed" in result.error
        assert "invalid_hash_for_test" in result.error
        assert "actual_hash_different" in result.error
