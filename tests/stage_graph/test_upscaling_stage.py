"""Unit tests for UpscalingStage (StageGraph integration with Phase 4 backend)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from transformation_portal.stage_graph.stage import StageContext, StageStatus
from transformation_portal.stage_graph.stages.upscaling import UpscalingStage


@pytest.mark.unit
class TestUpscalingStageUnit:
    """Unit tests for UpscalingStage."""

    def test_stage_instantiation(self):
        """Verify UpscalingStage can be instantiated with default params."""
        stage = UpscalingStage()

        assert stage.name == "upscaling"
        assert stage.version == "1.0.0"
        assert stage.scale_factor == 2.0
        assert stage.backend == "bicubic"
        assert stage.allow_fallback is True
        assert stage._upscaler is None
        assert stage._registry is not None

    def test_stage_with_custom_params(self):
        """Verify custom parameters are respected."""
        stage = UpscalingStage(
            scale_factor=4.0,
            backend="realesrgan",
            allow_fallback=False,
            version="2.0.0",
        )

        assert stage.scale_factor == 4.0
        assert stage.backend == "realesrgan"
        assert stage.allow_fallback is False
        assert stage.version == "2.0.0"

    def test_get_dependencies(self):
        """UpscalingStage declares 'enhancement' dependency for backward compatibility."""
        stage = UpscalingStage()
        deps = stage.get_dependencies()

        assert isinstance(deps, list)
        assert len(deps) == 1
        assert "enhancement" in deps

    def test_cache_key_determinism(self):
        """Same image should produce same cache key."""
        stage = UpscalingStage(backend="bicubic", scale_factor=2.0)

        # Create identical images
        np.random.seed(42)
        img1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        np.random.seed(42)
        img2 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        context1 = StageContext(artifacts={"image": img1})
        context2 = StageContext(artifacts={"image": img2})

        key1 = stage.get_cache_key(context1)
        key2 = stage.get_cache_key(context2)

        assert key1 == key2
        assert "upscale" in key1
        assert "bicubic_2.0" in key1

    def test_cache_key_invalidation_on_image_change(self):
        """Different images should produce different cache keys."""
        stage = UpscalingStage(backend="bicubic", scale_factor=2.0)

        np.random.seed(100)
        img1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        np.random.seed(200)
        img2 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        context1 = StageContext(artifacts={"image": img1})
        context2 = StageContext(artifacts={"image": img2})

        key1 = stage.get_cache_key(context1)
        key2 = stage.get_cache_key(context2)

        assert key1 != key2

    def test_cache_key_invalidation_on_backend_change(self):
        """Different backends should produce different cache keys."""
        stage1 = UpscalingStage(backend="bicubic", scale_factor=2.0)
        stage2 = UpscalingStage(backend="realesrgan", scale_factor=2.0)

        np.random.seed(42)
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        context = StageContext(artifacts={"image": img})

        key1 = stage1.get_cache_key(context)
        key2 = stage2.get_cache_key(context)

        assert key1 != key2
        assert "bicubic" in key1
        assert "realesrgan" in key2

    def test_cache_key_no_image(self):
        """Cache key without image should return 'no_image'."""
        stage = UpscalingStage()
        context = StageContext(artifacts={})

        key = stage.get_cache_key(context)
        assert key == "no_image"

    def test_cache_key_tries_multiple_artifact_names(self):
        """Cache key should try enhanced_image, image, depth_map."""
        stage = UpscalingStage()

        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Test with 'enhanced_image'
        context1 = StageContext(artifacts={"enhanced_image": img})
        key1 = stage.get_cache_key(context1)
        assert key1 != "no_image"

        # Test with 'image'
        context2 = StageContext(artifacts={"image": img})
        key2 = stage.get_cache_key(context2)
        assert key2 != "no_image"

        # Test with 'depth_map'
        context3 = StageContext(artifacts={"depth_map": img})
        key3 = stage.get_cache_key(context3)
        assert key3 != "no_image"

        # All should produce same key (same image content)
        assert key1 == key2 == key3

    def test_missing_image_artifact(self):
        """Missing image should fail with clear error."""
        stage = UpscalingStage()
        context = StageContext(artifacts={})

        result = stage.compute(context)

        assert result.status == StageStatus.FAILED
        assert "missing" in result.error.lower()
        assert "image" in result.error.lower()

    def test_invalid_image_type(self):
        """Non-numpy image should fail with clear error."""
        stage = UpscalingStage()
        context = StageContext(artifacts={"image": "not_an_array"})

        result = stage.compute(context)

        assert result.status == StageStatus.FAILED
        assert "invalid image type" in result.error.lower()
        assert "numpy array" in result.error.lower()

    def test_skip_when_scale_factor_is_one(self):
        """Scale factor of 1.0 should skip upscaling."""
        stage = UpscalingStage(scale_factor=1.0)
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        context = StageContext(artifacts={"image": img})

        result = stage.compute(context)

        assert result.status == StageStatus.SKIPPED
        assert "upscaled_image" in result.artifacts
        assert np.array_equal(result.artifacts["upscaled_image"], img)
        assert result.artifacts["upscale_metadata"]["skipped"] is True
        assert result.artifacts["upscale_metadata"]["scale_factor"] == 1.0

    def test_bicubic_backend_uint8(self):
        """Test bicubic upscaling with uint8 images."""
        stage = UpscalingStage(backend="bicubic", scale_factor=2.0)
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        context = StageContext(artifacts={"image": img}, device="cpu")

        result = stage.compute(context)

        assert result.status == StageStatus.COMPLETED
        assert "upscaled_image" in result.artifacts
        upscaled = result.artifacts["upscaled_image"]
        assert upscaled.shape == (200, 200, 3)
        assert upscaled.dtype == np.uint8

        # Check metadata
        metadata = result.artifacts["upscale_metadata"]
        assert metadata["backend_requested"] == "bicubic"
        assert metadata["backend_used"] == "bicubic"
        assert metadata["scale_factor"] == 2.0
        assert metadata["input_dtype"] == "uint8"
        assert metadata["output_dtype"] == "uint8"

    def test_bicubic_backend_float32(self):
        """Test bicubic upscaling with float32 images (Phase 3 compatibility)."""
        stage = UpscalingStage(backend="bicubic", scale_factor=2.0)
        img = np.random.rand(100, 100, 3).astype(np.float32)
        context = StageContext(artifacts={"image": img}, device="cpu")

        result = stage.compute(context)

        assert result.status == StageStatus.COMPLETED
        upscaled = result.artifacts["upscaled_image"]
        assert upscaled.shape == (200, 200, 3)
        assert upscaled.dtype == np.float32
        assert upscaled.min() >= 0.0
        assert upscaled.max() <= 1.0

        # Check metadata
        metadata = result.artifacts["upscale_metadata"]
        assert metadata["input_dtype"] == "float32"
        assert metadata["output_dtype"] == "float32"

    def test_grayscale_depth_map_upscaling(self):
        """Test upscaling grayscale depth maps (H, W) → (H*scale, W*scale)."""
        stage = UpscalingStage(backend="bicubic", scale_factor=2.0)
        depth_map = np.random.rand(100, 100).astype(np.float32)
        context = StageContext(artifacts={"depth_map": depth_map}, device="cpu")

        result = stage.compute(context)

        assert result.status == StageStatus.COMPLETED
        upscaled = result.artifacts["upscaled_image"]

        # Should be grayscale output
        assert upscaled.ndim == 2
        assert upscaled.shape == (200, 200)
        assert upscaled.dtype == np.float32

        # Check metadata
        metadata = result.artifacts["upscale_metadata"]
        assert metadata["was_grayscale"] is True

    def test_enhanced_image_priority(self):
        """enhanced_image should take priority over image."""
        stage = UpscalingStage(backend="bicubic", scale_factor=2.0)

        img1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Both present, enhanced_image should be used
        context = StageContext(artifacts={"image": img1, "enhanced_image": img2}, device="cpu")

        result = stage.compute(context)

        assert result.status == StageStatus.COMPLETED
        # Verify enhanced_image was used (check via cache key or metadata)
        metadata = result.artifacts["upscale_metadata"]
        assert metadata["input_shape"] == img2.shape

    def test_fallback_to_bicubic_on_unknown_backend(self):
        """Unknown backend with allow_fallback=True should use bicubic."""
        stage = UpscalingStage(backend="unknown_backend", allow_fallback=True, scale_factor=2.0)
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        context = StageContext(artifacts={"image": img}, device="cpu")

        result = stage.compute(context)

        assert result.status == StageStatus.COMPLETED
        metadata = result.artifacts["upscale_metadata"]
        assert metadata["backend_requested"] == "unknown_backend"
        assert metadata["backend_used"] == "bicubic"  # Fallback occurred

    def test_no_fallback_raises_on_unknown_backend(self):
        """Unknown backend with allow_fallback=False should fail during load."""
        stage = UpscalingStage(backend="unknown_backend", allow_fallback=False, scale_factor=2.0)
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        context = StageContext(artifacts={"image": img}, device="cpu")

        # The error occurs during _load_upscaler, which is caught by Stage.execute()
        # When called via compute() directly, the exception propagates
        # This is expected behavior - the stage's execute() wrapper handles this
        with pytest.raises(ValueError, match="Unknown upscaler backend"):
            stage.compute(context)

    def test_no_fallback_handled_by_execute(self):
        """Unknown backend with allow_fallback=False handled gracefully by execute()."""
        stage = UpscalingStage(backend="unknown_backend", allow_fallback=False, scale_factor=2.0)
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        context = StageContext(artifacts={"image": img}, device="cpu", cache_enabled=False)

        # execute() wrapper catches exceptions and returns StageResult with FAILED status
        result = stage.execute(context)

        assert result.status == StageStatus.FAILED
        assert "unknown upscaler backend" in result.error.lower()
        assert result.error_traceback is not None

    def test_lazy_loading_upscaler(self):
        """Upscaler should not load on init, only on first compute."""
        stage = UpscalingStage(backend="bicubic", scale_factor=2.0)

        # Initially None
        assert stage._upscaler is None

        # After compute, should be loaded
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        context = StageContext(artifacts={"image": img}, device="cpu")
        result = stage.compute(context)

        assert result.status == StageStatus.COMPLETED
        assert stage._upscaler is not None
        assert stage._upscaler.name == "bicubic"

    def test_device_propagation(self):
        """Device from context should be passed to backend."""
        stage = UpscalingStage(backend="bicubic", scale_factor=2.0)
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Test with different devices
        for device in ["cpu", "cuda", "mps"]:
            stage._upscaler = None  # Reset for each device
            context = StageContext(artifacts={"image": img}, device=device)

            with patch.object(stage._registry, "get") as mock_get:
                mock_backend = MagicMock()
                mock_backend.name = "bicubic"
                mock_backend.upscale.return_value = np.zeros((200, 200, 3), dtype=np.uint8)
                mock_get.return_value = mock_backend

                _ = stage.compute(context)  # noqa: F841 (result used for side-effect verification)

                # Verify device was passed to registry
                mock_get.assert_called_once()
                call_kwargs = mock_get.call_args[1]
                assert call_kwargs["device"] == device

    def test_timing_metadata(self):
        """Result should include timing metadata."""
        stage = UpscalingStage(backend="bicubic", scale_factor=2.0)
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        context = StageContext(artifacts={"image": img}, device="cpu")

        result = stage.compute(context)

        assert result.status == StageStatus.COMPLETED
        assert result.duration_ms > 0
        assert "upscale_ms" in result.metadata
        assert result.metadata["upscale_ms"] > 0

    def test_shape_metadata(self):
        """Metadata should include input and output shapes."""
        stage = UpscalingStage(backend="bicubic", scale_factor=2.0)
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        context = StageContext(artifacts={"image": img}, device="cpu")

        result = stage.compute(context)

        assert result.status == StageStatus.COMPLETED
        metadata = result.artifacts["upscale_metadata"]
        assert metadata["input_shape"] == (100, 100, 3)
        assert metadata["output_shape"] == (200, 200, 3)

    @patch("transformation_portal.upscaling.registry.logger")
    def test_backend_loading_logs(self, mock_logger):
        """Backend loading should log useful information."""
        stage = UpscalingStage(backend="bicubic", scale_factor=2.0)
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        context = StageContext(artifacts={"image": img}, device="cpu")

        # Mock stage logger
        with patch.object(stage.logger, "info") as mock_info:
            result = stage.compute(context)

            assert result.status == StageStatus.COMPLETED
            # Check that backend loading was logged
            mock_info.assert_called()
            call_args = [str(call) for call in mock_info.call_args_list]
            assert any("bicubic" in str(call) for call in call_args)

    def test_upscaling_error_handling(self):
        """Upscaling errors should be captured in result."""
        stage = UpscalingStage(backend="bicubic", scale_factor=2.0)
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        context = StageContext(artifacts={"image": img}, device="cpu")

        # Force upscaler to raise error
        with patch.object(stage._registry, "get") as mock_get:
            mock_backend = MagicMock()
            mock_backend.name = "bicubic"
            mock_backend.upscale.side_effect = RuntimeError("Upscaling error")
            mock_get.return_value = mock_backend

            result = stage.compute(context)

            assert result.status == StageStatus.FAILED
            assert "upscaling failed" in result.error.lower()
            assert "upscaling error" in result.error.lower()

    def test_default_backend_alias(self):
        """'default' backend should resolve to bicubic."""
        stage = UpscalingStage(backend="default", scale_factor=2.0)
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        context = StageContext(artifacts={"image": img}, device="cpu")

        result = stage.compute(context)

        assert result.status == StageStatus.COMPLETED
        metadata = result.artifacts["upscale_metadata"]
        assert metadata["backend_used"] == "bicubic"


@pytest.mark.integration
class TestUpscalingStageIntegration:
    """Integration tests for UpscalingStage with real backends."""

    def test_end_to_end_bicubic_uint8(self):
        """End-to-end test with bicubic backend (uint8)."""
        stage = UpscalingStage(backend="bicubic", scale_factor=2.0)

        # Create realistic test image
        img = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
        context = StageContext(artifacts={"image": img}, device="cpu")

        result = stage.compute(context)

        assert result.status == StageStatus.COMPLETED
        upscaled = result.artifacts["upscaled_image"]
        assert upscaled.shape == (512, 512, 3)
        assert upscaled.dtype == np.uint8

    def test_end_to_end_bicubic_float32(self):
        """End-to-end test with bicubic backend (float32)."""
        stage = UpscalingStage(backend="bicubic", scale_factor=2.0)

        # Create realistic depth map
        img = np.random.rand(256, 256, 3).astype(np.float32)
        context = StageContext(artifacts={"image": img}, device="cpu")

        result = stage.compute(context)

        assert result.status == StageStatus.COMPLETED
        upscaled = result.artifacts["upscaled_image"]
        assert upscaled.shape == (512, 512, 3)
        assert upscaled.dtype == np.float32
        assert upscaled.min() >= 0.0
        assert upscaled.max() <= 1.0

    def test_end_to_end_depth_map_upscaling(self):
        """End-to-end test with grayscale depth map."""
        stage = UpscalingStage(backend="bicubic", scale_factor=2.0)

        # Create realistic depth map (grayscale)
        depth = np.random.rand(256, 256).astype(np.float32)
        context = StageContext(artifacts={"depth_map": depth}, device="cpu")

        result = stage.compute(context)

        assert result.status == StageStatus.COMPLETED
        upscaled = result.artifacts["upscaled_image"]
        assert upscaled.ndim == 2  # Still grayscale
        assert upscaled.shape == (512, 512)
        assert upscaled.dtype == np.float32

    def test_fractional_scale_factor(self):
        """Test with fractional scale factor (e.g., 1.5x)."""
        stage = UpscalingStage(backend="bicubic", scale_factor=1.5)

        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        context = StageContext(artifacts={"image": img}, device="cpu")

        result = stage.compute(context)

        assert result.status == StageStatus.COMPLETED
        upscaled = result.artifacts["upscaled_image"]

        # OpenCV resize uses rounding for fractional scales
        expected_h = int(round(100 * 1.5))
        expected_w = int(round(100 * 1.5))
        assert upscaled.shape == (expected_h, expected_w, 3)

    def test_max_scale_factor(self):
        """Test with maximum scale factor (4.0)."""
        stage = UpscalingStage(backend="bicubic", scale_factor=4.0)

        img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        context = StageContext(artifacts={"image": img}, device="cpu")

        result = stage.compute(context)

        assert result.status == StageStatus.COMPLETED
        upscaled = result.artifacts["upscaled_image"]
        assert upscaled.shape == (200, 200, 3)
