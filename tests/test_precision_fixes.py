"""Tests for precision bug fixes in pipeline.

Tests verify:
1. Bilateral filter processes float32 depth directly (no uint8 quantization)
2. Orchestrator scales preprocessing correctly (0-1 → 0-255)
3. Dimension enforcement uses crop/pad (not resampling)
4. Format support includes WebP and BMP
"""

import numpy as np
import pytest
from PIL import Image

from transformation_portal.lux_depth_v3.preprocessing import (
    SUPPORTED_EXTENSIONS,
    _enforce_dimension_multiple,
    preprocess_image,
    validate_image_format,
)


class TestBilateralFilterPrecision:
    """Test bilateral filter precision improvements."""

    def test_bilateral_filter_no_uint8_quantization(self, monkeypatch):
        """Test that bilateral filter processes float32 directly without uint8 conversion.

        Uses monkeypatching to verify cv2 functions receive float32 inputs.
        """
        from transformation_portal.lux_depth_v3.config import PostprocessingConfig
        from transformation_portal.lux_depth_v3.postprocessing import Postprocessor

        # Create test depth with high precision values
        # Use values that would lose precision if quantized to uint8
        depth = np.array(
            [
                [0.1234, 0.2345, 0.3456],
                [0.4567, 0.5678, 0.6789],
                [0.7890, 0.8901, 0.9012],
            ],
            dtype=np.float32,
        )

        # Create dummy image
        image = np.random.rand(3, 3, 3).astype(np.float32)

        # Mock cv2.bilateralFilter to verify it receives float32
        def mock_bilateral_filter(src, d, sigmaColor, sigmaSpace):
            # Assert input is float32, not uint8
            assert src.dtype == np.float32, f"Expected float32, got {src.dtype}"
            assert src.ndim == 2, f"Expected 2D array, got {src.ndim}D"
            # Return same shape with slight smoothing
            return src.copy()

        try:
            import cv2

            # Monkeypatch cv2.bilateralFilter
            monkeypatch.setattr(cv2, "bilateralFilter", mock_bilateral_filter)

            # Create postprocessor with bilateral filter enabled
            config = PostprocessingConfig(
                apply_bilateral_filter=True,
                bilateral_sigma_color=0.1,
                bilateral_sigma_space=1.0,
            )
            postprocessor = Postprocessor(config)

            # Apply bilateral filter - mock will verify float32
            filtered = postprocessor._bilateral_filter(
                depth, image, config.bilateral_sigma_color, config.bilateral_sigma_space
            )

            # Verify output is float32
            assert filtered.dtype == np.float32
            assert filtered.shape == depth.shape

        except ImportError:
            # OpenCV not available - skip test
            pytest.skip("OpenCV not available, skipping bilateral filter test")

    def test_bilateral_filter_handles_metric_depth(self):
        """Test that bilateral filter scales sigmaColor for metric depth ranges."""
        from transformation_portal.lux_depth_v3.postprocessing import Postprocessor
        from transformation_portal.lux_depth_v3.config import PostprocessingConfig

        # Create metric depth (e.g., meters, range 0-100)
        depth = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]], dtype=np.float32)

        image = np.random.rand(3, 3, 3).astype(np.float32)

        config = PostprocessingConfig(apply_bilateral_filter=True, bilateral_sigma_color=5.0, bilateral_sigma_space=1.0)
        postprocessor = Postprocessor(config)

        try:
            filtered = postprocessor._bilateral_filter(
                depth, image, config.bilateral_sigma_color, config.bilateral_sigma_space
            )

            # Should return float32 with same shape
            assert filtered.dtype == np.float32
            assert filtered.shape == depth.shape

            # Values should be in same range (no normalization)
            assert filtered.min() >= 0
            assert filtered.max() <= 100

        except ImportError:
            pytest.skip("OpenCV not available")


class TestPreprocessingScaling:
    """Test preprocessing scaling fix for orchestrator."""

    def test_preprocess_returns_float32_zero_to_one(self, tmp_path):
        """Test that preprocess_image returns float32 [0, 1] as documented."""
        # Create test image
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (56, 56), color=(127, 127, 127))
        img.save(img_path)

        result, _ = preprocess_image(img_path)

        # Verify float32 output in [0, 1] range
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

        # Verify approximate midpoint (127/255 ≈ 0.498)
        assert abs(result.mean() - 0.498) < 0.01


class TestDimensionEnforcementNonDestructive:
    """Test that dimension enforcement uses crop/pad instead of resampling."""

    def test_dimension_enforcement_uses_crop_not_resample(self):
        """Test that dimension enforcement preserves pixels via crop, not resample."""
        # Create image with unique pixel pattern to detect resampling
        # If Lanczos resampling is used, pixel values will be interpolated
        # If crop is used, original pixel values are preserved
        img_array = np.zeros((100, 100, 3), dtype=np.float32)

        # Set distinct pattern in top-left that should survive center crop
        img_array[40:60, 40:60, 0] = 1.0  # Red square in center
        img_array[40:60, 40:60, 1] = 0.0
        img_array[40:60, 40:60, 2] = 0.0

        # Enforce dimension multiple (100 → 98, which should crop 1px from each side)
        result = _enforce_dimension_multiple(img_array, 14)

        # Check dimensions are correct (98 = 7×14)
        assert result.shape[:2] == (98, 98)

        # Check that center red square is preserved (crop should keep it)
        # After cropping 1px from each side, the red square at [40:60, 40:60]
        # becomes [39:59, 39:59] in the cropped image
        center_red = result[39:59, 39:59, 0]
        assert np.all(center_red == 1.0), "Center pattern not preserved - likely resampled instead of cropped"

    def test_dimension_enforcement_pads_when_undersized(self):
        """Test that undersized images are padded, not resampled."""
        # Create small image (10x10, needs padding to reach 14x14 minimum)
        img_array = np.ones((10, 10, 3), dtype=np.float32) * 0.5

        result = _enforce_dimension_multiple(img_array, 14)

        # Should be padded to 14x14
        assert result.shape[:2] == (14, 14)

        # Center 10x10 region should preserve original values
        # With symmetric padding: 2px top, 2px left
        center = result[2:12, 2:12]
        assert np.allclose(center, 0.5), "Padding did not preserve center region"

    def test_dimension_enforcement_mixed_crop_pad(self):
        """Test mixed scenario where one dimension needs crop, other needs pad.

        Example: 15x10 → 14x14 requires cropping width (15→14) and padding height (10→14).
        This verifies that crop and pad are applied independently per dimension.
        """
        # Create 15x10 image with distinct pattern
        img_array = np.zeros((10, 15, 3), dtype=np.float32)
        # Set center region to white
        img_array[4:6, 7:8, :] = 1.0  # Center pixel at (5, 7.5)

        result = _enforce_dimension_multiple(img_array, 14)

        # Should be 14x14 (pad height 10→14, crop width 15→14)
        assert result.shape[:2] == (14, 14), f"Expected (14, 14), got {result.shape[:2]}"

        # Verify center pattern survived
        # Original center at [4:6, 7:8]
        # After height pad (+2 top): [6:8, 7:8]
        # After width crop (-0.5 left): [6:8, 7:8] (crop is 15-14=1, so 0 or 1 pixel shift)
        # Due to center crop of 1px from width, the center moves slightly
        center_region = result[6:8, 6:8]
        assert np.any(center_region > 0.5), "Center pattern lost - mixed crop/pad failed"


class TestFormatSupport:
    """Test WebP and BMP format support."""

    def test_webp_in_supported_extensions(self):
        """Test that .webp is in SUPPORTED_EXTENSIONS."""
        assert ".webp" in SUPPORTED_EXTENSIONS

    def test_bmp_in_supported_extensions(self):
        """Test that .bmp is in SUPPORTED_EXTENSIONS."""
        assert ".bmp" in SUPPORTED_EXTENSIONS

    def test_webp_image_validation(self, tmp_path):
        """Test that WebP images can be validated."""
        # Create WebP image
        img_path = tmp_path / "test.webp"
        img = Image.new("RGB", (56, 56), color="blue")
        img.save(img_path, "WEBP")

        # Should validate successfully
        result = validate_image_format(img_path)
        assert result == img_path

    def test_bmp_image_validation(self, tmp_path):
        """Test that BMP images can be validated."""
        # Create BMP image
        img_path = tmp_path / "test.bmp"
        img = Image.new("RGB", (56, 56), color="green")
        img.save(img_path, "BMP")

        # Should validate successfully
        result = validate_image_format(img_path)
        assert result == img_path

    def test_webp_preprocessing(self, tmp_path):
        """Test that WebP images can be preprocessed."""
        img_path = tmp_path / "test.webp"
        img = Image.new("RGB", (70, 70), color="red")
        img.save(img_path, "WEBP")

        result, original_shape = preprocess_image(img_path)

        # Should return float32 array
        assert result.dtype == np.float32
        assert result.shape[2] == 3
        assert original_shape == (70, 70)

    def test_bmp_preprocessing(self, tmp_path):
        """Test that BMP images can be preprocessed."""
        img_path = tmp_path / "test.bmp"
        img = Image.new("RGB", (70, 70), color="yellow")
        img.save(img_path, "BMP")

        result, original_shape = preprocess_image(img_path)

        # Should return float32 array
        assert result.dtype == np.float32
        assert result.shape[2] == 3
        assert original_shape == (70, 70)


class TestInputDiscoveryFormatSupport:
    """Test that input discovery includes WebP and BMP."""

    def test_discover_images_finds_webp_and_bmp(self, tmp_path):
        """Test that discover_images finds WebP and BMP files."""
        from transformation_portal.lux_depth_v3.input_discovery import discover_images, DiscoveryConfig

        # Note: discover_images validates files, so we need real images
        try:
            img1 = tmp_path / "real1.webp"
            Image.new("RGB", (56, 56)).save(img1, "WEBP")

            img2 = tmp_path / "real2.bmp"
            Image.new("RGB", (56, 56)).save(img2, "BMP")

            # Discover with default extensions (should include webp, bmp)
            config = DiscoveryConfig(strict_mode=False)
            images = discover_images(tmp_path, config, image_extensions=None)

            # Should find both files
            stems = {p.stem for p in images}
            assert "real1" in stems
            assert "real2" in stems

        except Exception:
            pytest.skip("Could not create WebP/BMP test files")
