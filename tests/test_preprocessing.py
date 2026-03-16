"""Tests for preprocessing module.

Tests image validation, format conversion, dimension enforcement,
and normalization for depth inference.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Pytest markers
pytestmark = [
    pytest.mark.unit,
]

from transformation_portal.lux_depth_v3.preprocessing import (
    DIMENSION_MULTIPLE,
    SUPPORTED_EXTENSIONS,
    preprocess_image,
    validate_image_format,
)


class TestValidateImageFormat:
    """Test image format validation."""

    def test_valid_image_passes(self, tmp_path):
        """Test that valid image passes validation."""
        # Create valid PNG
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (64, 64), color="red")
        img.save(img_path)

        result = validate_image_format(img_path)

        assert result == img_path

    def test_nonexistent_file_raises_filenotfounderror(self, tmp_path):
        """Test that missing file raises FileNotFoundError."""
        fake_path = tmp_path / "nonexistent.jpg"

        with pytest.raises(FileNotFoundError, match="not found"):
            validate_image_format(fake_path)

    def test_unsupported_extension_raises_valueerror(self, tmp_path):
        """Test that unsupported format raises ValueError."""
        # Use .gif which is actually unsupported (we now support .bmp and .webp)
        bad_path = tmp_path / "test.gif"
        bad_path.write_text("fake gif")

        with pytest.raises(ValueError, match="Unsupported image format"):
            validate_image_format(bad_path)

    def test_corrupt_image_raises_valueerror(self, tmp_path):
        """Test that corrupt image raises ValueError."""
        corrupt_path = tmp_path / "corrupt.jpg"
        corrupt_path.write_bytes(b"not a valid image")

        with pytest.raises(ValueError, match="corrupt or invalid"):
            validate_image_format(corrupt_path)

    @pytest.mark.parametrize("ext", [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp", ".bmp"])
    def test_all_supported_extensions(self, tmp_path, ext):
        """Test that all supported extensions are accepted."""
        img_path = tmp_path / f"test{ext}"
        img = Image.new("RGB", (32, 32))
        # Map extensions to PIL format names
        fmt_map = {
            ".jpg": "JPEG",
            ".jpeg": "JPEG",
            ".png": "PNG",
            ".tiff": "TIFF",
            ".tif": "TIFF",
            ".webp": "WEBP",
            ".bmp": "BMP",
        }
        img.save(img_path, fmt_map[ext])

        result = validate_image_format(img_path)

        assert result == img_path


class TestPreprocessImage:
    """Test image preprocessing and normalization."""

    def test_uint8_rgb_to_float32(self, tmp_path):
        """Test conversion from uint8 RGB to float32 [0, 1]."""
        # Create uint8 RGB image
        img_path = tmp_path / "rgb.png"
        img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        img.save(img_path)

        result, original_shape = preprocess_image(img_path)

        # Check dtype and range
        assert result.dtype == np.float32
        assert result.ndim == 3
        assert result.shape[2] == 3
        assert result.min() >= 0.0
        assert result.max() <= 1.0

        # Check approximate value (128/255 ≈ 0.5)
        assert np.abs(result.mean() - 0.5) < 0.01

    def test_grayscale_converted_to_rgb(self, tmp_path):
        """Test that grayscale images are converted to 3-channel RGB."""
        img_path = tmp_path / "gray.png"
        img = Image.new("L", (50, 50), color=100)
        img.save(img_path)

        result, original_shape = preprocess_image(img_path)

        # Should be 3-channel
        assert result.shape[2] == 3

        # All channels should be identical (grayscale)
        assert np.allclose(result[:, :, 0], result[:, :, 1])
        assert np.allclose(result[:, :, 1], result[:, :, 2])

    def test_rgba_converted_to_rgb(self, tmp_path):
        """Test that RGBA images drop alpha channel."""
        img_path = tmp_path / "rgba.png"
        img = Image.new("RGBA", (50, 50), color=(255, 0, 0, 128))
        img.save(img_path)

        result, original_shape = preprocess_image(img_path)

        # Should be 3-channel (alpha dropped)
        assert result.shape[2] == 3

    def test_dimensions_enforced_to_multiple_of_14(self):
        """Test that dimensions are enforced to multiples of 14."""
        # Create image with non-compliant dimensions
        test_cases = [
            ((100, 100), (98, 98)),  # 100 → 98 (7×14)
            ((50, 70), (42, 70)),  # 50 → 42 (3×14), 70 → 70 (5×14)
            ((15, 15), (14, 14)),  # 15 → 14 (1×14)
            ((7, 7), (14, 14)),  # 7 → 14 (minimum)
        ]

        for input_size, expected_size in test_cases:
            img_array = np.random.rand(*input_size, 3).astype(np.float32)

            result, original_shape = preprocess_image(img_array)

            # Check dimensions are multiples of 14
            h, w = result.shape[:2]
            assert h % DIMENSION_MULTIPLE == 0, f"Height {h} not multiple of {DIMENSION_MULTIPLE}"
            assert w % DIMENSION_MULTIPLE == 0, f"Width {w} not multiple of {DIMENSION_MULTIPLE}"

            # Check expected size
            assert result.shape[:2] == expected_size

    def test_original_shape_preserved(self, tmp_path):
        """Test that original shape is returned correctly."""
        # Create image
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (120, 80))  # W=120, H=80
        img.save(img_path)

        result, original_shape = preprocess_image(img_path)

        # Original shape should be (H, W)
        assert original_shape == (80, 120)

    def test_target_size_resizes_long_edge(self, tmp_path):
        """Test that target_size resizes long edge while maintaining aspect."""
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (200, 100))  # W=200 (long), H=100
        img.save(img_path)

        result, original_shape = preprocess_image(img_path, target_size=112)

        # Long edge should be close to 112 (after 14-alignment)
        # 112 is already multiple of 14 (8×14)
        h, w = result.shape[:2]
        assert max(h, w) <= 112

        # Aspect ratio should be approximately preserved
        aspect_original = 200 / 100
        aspect_result = w / h
        assert abs(aspect_original - aspect_result) < 0.2

    def test_numpy_array_input_uint8(self):
        """Test preprocessing from numpy uint8 array."""
        img_array = np.random.randint(0, 256, (56, 56, 3), dtype=np.uint8)

        result, original_shape = preprocess_image(img_array)

        assert result.dtype == np.float32
        assert result.shape[:2] == (56, 56)
        assert original_shape == (56, 56)

    def test_numpy_array_input_float32(self):
        """Test preprocessing from numpy float32 array."""
        img_array = np.random.rand(70, 70, 3).astype(np.float32)

        result, original_shape = preprocess_image(img_array)

        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_numpy_grayscale_to_rgb(self):
        """Test preprocessing from numpy grayscale (H, W)."""
        img_array = np.random.rand(56, 56).astype(np.float32)

        result, original_shape = preprocess_image(img_array)

        # Should be 3-channel
        assert result.shape[2] == 3

        # All channels should be identical
        assert np.allclose(result[:, :, 0], result[:, :, 1])

    def test_invalid_array_shape_raises_error(self):
        """Test that invalid array shapes raise ValueError."""
        # 4D array (invalid)
        bad_array = np.random.rand(10, 10, 3, 1).astype(np.float32)

        with pytest.raises(ValueError, match="Unsupported array shape"):
            preprocess_image(bad_array)

    def test_invalid_type_raises_error(self):
        """Test that invalid input types raise TypeError."""
        with pytest.raises(TypeError, match="must be np.ndarray, Path, or str"):
            preprocess_image(123)

    def test_minimum_dimension_enforced(self):
        """Test that minimum dimension is enforced (14)."""
        # Very small image
        img_array = np.random.rand(5, 5, 3).astype(np.float32)

        result, original_shape = preprocess_image(img_array)

        # Should be at least 14×14
        assert result.shape[0] >= DIMENSION_MULTIPLE
        assert result.shape[1] >= DIMENSION_MULTIPLE


class TestDimensionEnforcement:
    """Test dimension enforcement edge cases."""

    @pytest.mark.parametrize(
        "input_dim,expected_dim",
        [
            (14, 14),  # Already compliant
            (28, 28),  # Already compliant
            (42, 42),  # Already compliant
            (15, 14),  # Round down
            (27, 14),  # Round down
            (29, 28),  # Round down
            (100, 98),  # Round down
            (1, 14),  # Clamp to minimum
            (7, 14),  # Clamp to minimum
        ],
    )
    def test_dimension_rounding(self, input_dim, expected_dim):
        """Test dimension rounding behavior."""
        img_array = np.random.rand(input_dim, input_dim, 3).astype(np.float32)

        result, _ = preprocess_image(img_array)

        h, w = result.shape[:2]
        assert h == expected_dim
        assert w == expected_dim
        assert h % DIMENSION_MULTIPLE == 0
        assert w % DIMENSION_MULTIPLE == 0


class TestEndToEnd:
    """End-to-end preprocessing tests."""

    def test_full_pipeline_from_file(self, tmp_path):
        """Test complete preprocessing pipeline from file."""
        # Create test image
        img_path = tmp_path / "test.jpg"
        img = Image.new("RGB", (100, 75), color=(200, 100, 50))
        img.save(img_path)

        # Preprocess
        result, original_shape = preprocess_image(img_path, target_size=None)

        # Validate all requirements
        assert result.dtype == np.float32
        assert result.ndim == 3
        assert result.shape[2] == 3
        assert result.min() >= 0.0
        assert result.max() <= 1.0
        assert result.shape[0] % DIMENSION_MULTIPLE == 0
        assert result.shape[1] % DIMENSION_MULTIPLE == 0
        assert original_shape == (75, 100)  # H, W


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
