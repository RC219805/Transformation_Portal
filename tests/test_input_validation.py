"""Tests for input validation module.

Tests cover:
- Image file validation
- PIL Image validation
- NumPy array validation
- Batch validation
- Edge cases and error handling
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from transformation_portal.utils.input_validation import (
    BatchValidator,
    ImageValidationError,
    ImageValidator,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    is_valid_image,
    require_valid_image,
    validate_image,
    validate_image_strict,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def small_rgb_image():
    """Create a small valid RGB image."""
    return Image.new('RGB', (100, 100), color='red')


@pytest.fixture
def large_rgb_image():
    """Create a larger valid RGB image."""
    return Image.new('RGB', (1920, 1080), color='blue')


@pytest.fixture
def rgba_image():
    """Create an RGBA image with alpha channel."""
    return Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))


@pytest.fixture
def grayscale_image():
    """Create a grayscale image."""
    return Image.new('L', (100, 100), color=128)


@pytest.fixture
def temp_image_file(small_rgb_image):
    """Create a temporary image file."""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        small_rgb_image.save(f.name, 'JPEG')
        yield Path(f.name)
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def temp_png_file(small_rgb_image):
    """Create a temporary PNG file."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        small_rgb_image.save(f.name, 'PNG')
        yield Path(f.name)
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def temp_large_image_file(large_rgb_image):
    """Create a temporary large image file."""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        large_rgb_image.save(f.name, 'JPEG')
        yield Path(f.name)
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def rgb_array():
    """Create an RGB numpy array."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def float_array():
    """Create a float32 numpy array in [0, 1] range."""
    return np.random.rand(100, 100, 3).astype(np.float32)


@pytest.fixture
def grayscale_array():
    """Create a grayscale numpy array."""
    return np.random.randint(0, 255, (100, 100), dtype=np.uint8)


# =============================================================================
# Test ValidationResult
# =============================================================================


class TestValidationResult:
    """Tests for ValidationResult class."""

    def test_empty_result_is_valid(self):
        """Empty result should be valid."""
        result = ValidationResult()
        assert result.is_valid
        assert len(result.issues) == 0

    def test_add_error_invalidates(self):
        """Adding an error should invalidate the result."""
        result = ValidationResult()
        result.add_error("TEST", "Test error")
        assert not result.is_valid
        assert len(result.errors) == 1

    def test_add_warning_keeps_valid(self):
        """Adding a warning should keep result valid."""
        result = ValidationResult()
        result.add_warning("TEST", "Test warning")
        assert result.is_valid
        assert len(result.warnings) == 1

    def test_raise_if_invalid(self):
        """raise_if_invalid should raise on invalid result."""
        result = ValidationResult()
        result.add_error("TEST", "Test error")
        with pytest.raises(ImageValidationError):
            result.raise_if_invalid()

    def test_raise_if_invalid_passes_on_valid(self):
        """raise_if_invalid should not raise on valid result."""
        result = ValidationResult()
        result.add_warning("TEST", "Test warning")
        result.raise_if_invalid()  # Should not raise


# =============================================================================
# Test ImageValidator with Files
# =============================================================================


class TestImageValidatorFiles:
    """Tests for ImageValidator with file inputs."""

    def test_valid_jpeg(self, temp_image_file):
        """Test validation of valid JPEG file."""
        validator = ImageValidator()
        result = validator.validate(temp_image_file)
        assert result.is_valid

    def test_valid_png(self, temp_png_file):
        """Test validation of valid PNG file."""
        validator = ImageValidator()
        result = validator.validate(temp_png_file)
        assert result.is_valid

    def test_nonexistent_file(self):
        """Test validation of non-existent file."""
        validator = ImageValidator()
        result = validator.validate(Path("/nonexistent/file.jpg"))
        assert not result.is_valid
        assert any(e.code == "FILE_NOT_FOUND" for e in result.errors)

    def test_unsupported_format(self, small_rgb_image):
        """Test validation with unsupported format."""
        with tempfile.NamedTemporaryFile(suffix='.bmp', delete=False) as f:
            small_rgb_image.save(f.name, 'BMP')
            path = Path(f.name)

        try:
            validator = ImageValidator(allowed_formats={'JPEG', 'PNG'})
            result = validator.validate(path)
            assert not result.is_valid
            assert any(e.code == "UNSUPPORTED_FORMAT" for e in result.errors)
        finally:
            path.unlink(missing_ok=True)

    def test_file_too_large(self):
        """Test validation of file exceeding size limit."""
        # Create a moderately large image
        large_img = Image.new('RGB', (3000, 3000), color='white')

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            large_img.save(f.name, 'PNG')
            path = Path(f.name)

        try:
            # Set very low file size limit
            validator = ImageValidator(max_file_size_mb=0.001)
            result = validator.validate(path)
            assert not result.is_valid
            assert any(e.code == "FILE_TOO_LARGE" for e in result.errors)
        finally:
            path.unlink(missing_ok=True)


# =============================================================================
# Test ImageValidator with PIL Images
# =============================================================================


class TestImageValidatorPIL:
    """Tests for ImageValidator with PIL Image inputs."""

    def test_valid_rgb_image(self, small_rgb_image):
        """Test validation of valid RGB PIL Image."""
        validator = ImageValidator()
        result = validator.validate(small_rgb_image)
        assert result.is_valid

    def test_valid_rgba_image(self, rgba_image):
        """Test validation of valid RGBA PIL Image."""
        validator = ImageValidator()
        result = validator.validate(rgba_image)
        assert result.is_valid

    def test_valid_grayscale_image(self, grayscale_image):
        """Test validation of grayscale PIL Image."""
        validator = ImageValidator()
        result = validator.validate(grayscale_image)
        assert result.is_valid

    def test_image_too_small(self):
        """Test validation of image below minimum size."""
        small_img = Image.new('RGB', (50, 50))
        validator = ImageValidator(min_width=100, min_height=100)
        result = validator.validate(small_img)
        assert not result.is_valid
        assert any("TOO_SMALL" in e.code for e in result.errors)

    def test_image_too_large(self):
        """Test validation of image above maximum size."""
        # Don't actually create a huge image, just check logic
        validator = ImageValidator(max_width=100, max_height=100)
        large_img = Image.new('RGB', (200, 200))
        result = validator.validate(large_img)
        assert not result.is_valid
        assert any("TOO_LARGE" in e.code for e in result.errors)

    def test_require_rgb_fails_grayscale(self, grayscale_image):
        """Test that require_rgb fails for grayscale images."""
        validator = ImageValidator(require_rgb=True)
        result = validator.validate(grayscale_image)
        assert not result.is_valid
        assert any(e.code == "RGB_REQUIRED" for e in result.errors)

    def test_unusual_aspect_ratio_warning(self):
        """Test warning for unusual aspect ratio."""
        wide_img = Image.new('RGB', (1000, 100))
        validator = ImageValidator()
        result = validator.validate(wide_img)
        assert result.is_valid  # Still valid, just warning
        assert any(w.code == "UNUSUAL_ASPECT_RATIO" for w in result.warnings)


# =============================================================================
# Test ImageValidator with NumPy Arrays
# =============================================================================


class TestImageValidatorNumPy:
    """Tests for ImageValidator with numpy array inputs."""

    def test_valid_uint8_rgb(self, rgb_array):
        """Test validation of valid uint8 RGB array."""
        validator = ImageValidator()
        result = validator.validate(rgb_array)
        assert result.is_valid

    def test_valid_float32_rgb(self, float_array):
        """Test validation of valid float32 RGB array."""
        validator = ImageValidator()
        result = validator.validate(float_array)
        assert result.is_valid

    def test_valid_grayscale(self, grayscale_array):
        """Test validation of grayscale array."""
        validator = ImageValidator()
        result = validator.validate(grayscale_array)
        assert result.is_valid

    def test_invalid_dimensions(self):
        """Test validation of array with wrong dimensions."""
        arr_1d = np.array([1, 2, 3])
        validator = ImageValidator()
        result = validator.validate(arr_1d)
        assert not result.is_valid
        assert any(e.code == "INVALID_ARRAY_DIMENSIONS" for e in result.errors)

    def test_invalid_channels(self):
        """Test validation of array with wrong number of channels."""
        arr_5ch = np.random.randint(0, 255, (100, 100, 5), dtype=np.uint8)
        validator = ImageValidator()
        result = validator.validate(arr_5ch)
        assert not result.is_valid
        assert any(e.code == "INVALID_CHANNEL_COUNT" for e in result.errors)

    def test_nan_values(self):
        """Test validation of array with NaN values."""
        arr = np.random.rand(100, 100, 3).astype(np.float32)
        arr[50, 50, 0] = np.nan
        validator = ImageValidator()
        result = validator.validate(arr)
        assert not result.is_valid
        assert any(e.code == "CONTAINS_NAN" for e in result.errors)

    def test_inf_values(self):
        """Test validation of array with infinite values."""
        arr = np.random.rand(100, 100, 3).astype(np.float32)
        arr[50, 50, 0] = np.inf
        validator = ImageValidator()
        result = validator.validate(arr)
        assert not result.is_valid
        assert any(e.code == "CONTAINS_INF" for e in result.errors)

    def test_float_range_warning(self):
        """Test warning for float values outside [0, 1]."""
        arr = np.random.rand(100, 100, 3).astype(np.float32) * 2  # 0-2 range
        validator = ImageValidator()
        result = validator.validate(arr)
        assert result.is_valid  # Still valid, just warning
        assert any(w.code == "FLOAT_RANGE_WARNING" for w in result.warnings)

    def test_require_rgb_fails_2d(self, grayscale_array):
        """Test require_rgb fails for 2D array."""
        validator = ImageValidator(require_rgb=True)
        result = validator.validate(grayscale_array)
        assert not result.is_valid
        assert any(e.code == "RGB_REQUIRED" for e in result.errors)


# =============================================================================
# Test BatchValidator
# =============================================================================


class TestBatchValidator:
    """Tests for BatchValidator."""

    def test_batch_all_valid(self, temp_image_file, temp_png_file):
        """Test batch validation with all valid images."""
        validator = BatchValidator()
        results = validator.validate_batch([temp_image_file, temp_png_file])

        assert len(results) == 2
        assert validator.all_valid
        assert len(validator.valid_paths) == 2
        assert len(validator.invalid_paths) == 0

    def test_batch_with_invalid(self, temp_image_file):
        """Test batch validation with some invalid paths."""
        validator = BatchValidator()
        results = validator.validate_batch([
            temp_image_file,
            Path("/nonexistent/file.jpg"),
        ])

        assert len(results) == 2
        assert not validator.all_valid
        assert len(validator.valid_paths) == 1
        assert len(validator.invalid_paths) == 1

    def test_batch_stop_on_error(self, temp_image_file):
        """Test batch validation stopping on first error."""
        validator = BatchValidator()
        results = validator.validate_batch(
            [
                Path("/nonexistent1.jpg"),
                Path("/nonexistent2.jpg"),
                temp_image_file,
            ],
            stop_on_first_error=True,
        )

        # Should stop after first invalid
        assert len(results) == 1

    def test_batch_summary(self, temp_image_file, temp_png_file):
        """Test batch validation summary."""
        validator = BatchValidator()
        validator.validate_batch([temp_image_file, temp_png_file])

        summary = validator.summary()
        assert summary["total"] == 2
        assert summary["valid"] == 2
        assert summary["invalid"] == 0
        assert summary["success_rate"] == 1.0


# =============================================================================
# Test Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_validate_image(self, temp_image_file):
        """Test validate_image function."""
        result = validate_image(temp_image_file)
        assert result.is_valid

    def test_validate_image_strict(self, temp_large_image_file):
        """Test validate_image_strict function."""
        result = validate_image_strict(temp_large_image_file)
        assert result.is_valid

    def test_validate_image_strict_fails_small(self):
        """Test validate_image_strict fails for small images."""
        small_img = Image.new('RGB', (100, 100))
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            small_img.save(f.name, 'JPEG')
            path = Path(f.name)

        try:
            result = validate_image_strict(path, min_size=512)
            assert not result.is_valid
        finally:
            path.unlink(missing_ok=True)

    def test_is_valid_image_true(self, temp_image_file):
        """Test is_valid_image returns True for valid image."""
        assert is_valid_image(temp_image_file)

    def test_is_valid_image_false(self):
        """Test is_valid_image returns False for invalid path."""
        assert not is_valid_image(Path("/nonexistent.jpg"))

    def test_require_valid_image_passes(self, temp_image_file):
        """Test require_valid_image passes for valid image."""
        require_valid_image(temp_image_file)  # Should not raise

    def test_require_valid_image_raises(self):
        """Test require_valid_image raises for invalid image."""
        with pytest.raises(ImageValidationError):
            require_valid_image(Path("/nonexistent.jpg"))


# =============================================================================
# Test ValidationIssue
# =============================================================================


class TestValidationIssue:
    """Tests for ValidationIssue class."""

    def test_issue_string_representation(self):
        """Test ValidationIssue string representation."""
        issue = ValidationIssue(
            code="TEST_ERROR",
            message="Test error message",
            severity=ValidationSeverity.ERROR,
            suggestion="Fix it",
        )
        str_repr = str(issue)
        assert "TEST_ERROR" in str_repr
        assert "Test error message" in str_repr
        assert "Fix it" in str_repr

    def test_issue_without_suggestion(self):
        """Test ValidationIssue without suggestion."""
        issue = ValidationIssue(
            code="TEST",
            message="Test message",
            severity=ValidationSeverity.WARNING,
        )
        str_repr = str(issue)
        assert "TEST" in str_repr
        assert "Test message" in str_repr


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_invalid_input_type(self):
        """Test validation with invalid input type."""
        validator = ImageValidator()
        result = validator.validate("not a path object")  # String, not Path
        # String should be converted to Path
        assert not result.is_valid  # File won't exist

    def test_validation_with_context(self, small_rgb_image):
        """Test validation with context string."""
        validator = ImageValidator()
        result = validator.validate(small_rgb_image, context="Test Image")
        assert result.is_valid

    def test_empty_image(self):
        """Test validation of empty (0x0) image is handled."""
        validator = ImageValidator(min_width=1, min_height=1)
        # PIL doesn't allow 0x0 images, but check dimension validation
        tiny = Image.new('RGB', (1, 1))
        result = validator.validate(tiny)
        assert result.is_valid

    def test_custom_formats(self, small_rgb_image):
        """Test validation with custom format set."""
        with tempfile.NamedTemporaryFile(suffix='.tiff', delete=False) as f:
            small_rgb_image.save(f.name, 'TIFF')
            path = Path(f.name)

        try:
            # Only allow TIFF
            validator = ImageValidator(allowed_formats={'TIFF'})
            result = validator.validate(path)
            assert result.is_valid
        finally:
            path.unlink(missing_ok=True)
