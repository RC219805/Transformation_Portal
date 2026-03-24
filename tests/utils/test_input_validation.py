"""Tests for input_validation module.

Provides comprehensive coverage of image validation functionality
including file validation, PIL image validation, and numpy array validation.

Coverage Target: 70% of input_validation.py (195 statements)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytestmark = [
    pytest.mark.unit,
]


class TestValidationSeverity:
    """Tests for ValidationSeverity enum."""

    def test_severity_values(self):
        """Test that severity enum has expected values."""
        from transformation_portal.utils.input_validation import ValidationSeverity

        assert ValidationSeverity.INFO.value == "info"
        assert ValidationSeverity.WARNING.value == "warning"
        assert ValidationSeverity.ERROR.value == "error"


class TestValidationIssue:
    """Tests for ValidationIssue dataclass."""

    def test_basic_issue(self):
        """Test creating a basic validation issue."""
        from transformation_portal.utils.input_validation import (
            ValidationIssue,
            ValidationSeverity,
        )

        issue = ValidationIssue(
            code="TEST_CODE",
            message="Test message",
            severity=ValidationSeverity.ERROR,
        )

        assert issue.code == "TEST_CODE"
        assert issue.message == "Test message"
        assert issue.severity == ValidationSeverity.ERROR
        assert issue.suggestion is None
        assert issue.details == {}

    def test_issue_with_suggestion(self):
        """Test issue with suggestion."""
        from transformation_portal.utils.input_validation import (
            ValidationIssue,
            ValidationSeverity,
        )

        issue = ValidationIssue(
            code="TEST_CODE",
            message="Test message",
            severity=ValidationSeverity.WARNING,
            suggestion="Try this instead",
        )

        assert issue.suggestion == "Try this instead"

    def test_issue_str_representation(self):
        """Test string representation of issue."""
        from transformation_portal.utils.input_validation import (
            ValidationIssue,
            ValidationSeverity,
        )

        issue = ValidationIssue(
            code="WIDTH_TOO_SMALL",
            message="Image too narrow",
            severity=ValidationSeverity.ERROR,
            suggestion="Resize image",
        )

        result = str(issue)
        assert "[ERROR]" in result
        assert "WIDTH_TOO_SMALL" in result
        assert "Image too narrow" in result
        assert "Resize image" in result


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_default_result_is_valid(self):
        """Test that default result is valid."""
        from transformation_portal.utils.input_validation import ValidationResult

        result = ValidationResult()

        assert result.is_valid is True
        assert result.issues == []

    def test_add_error_makes_invalid(self):
        """Test that adding error makes result invalid."""
        from transformation_portal.utils.input_validation import ValidationResult

        result = ValidationResult()
        result.add_error("TEST_ERROR", "Test error message")

        assert result.is_valid is False
        assert len(result.issues) == 1
        assert len(result.errors) == 1

    def test_add_warning_keeps_valid(self):
        """Test that adding warning keeps result valid."""
        from transformation_portal.utils.input_validation import ValidationResult

        result = ValidationResult()
        result.add_warning("TEST_WARNING", "Test warning message")

        assert result.is_valid is True
        assert len(result.issues) == 1
        assert len(result.warnings) == 1
        assert len(result.errors) == 0

    def test_errors_property(self):
        """Test errors property filters correctly."""
        from transformation_portal.utils.input_validation import ValidationResult

        result = ValidationResult()
        result.add_error("ERROR_1", "Error 1")
        result.add_warning("WARNING_1", "Warning 1")
        result.add_error("ERROR_2", "Error 2")

        errors = result.errors
        assert len(errors) == 2
        assert all(e.code.startswith("ERROR_") for e in errors)

    def test_warnings_property(self):
        """Test warnings property filters correctly."""
        from transformation_portal.utils.input_validation import ValidationResult

        result = ValidationResult()
        result.add_warning("WARNING_1", "Warning 1")
        result.add_error("ERROR_1", "Error 1")
        result.add_warning("WARNING_2", "Warning 2")

        warnings = result.warnings
        assert len(warnings) == 2
        assert all(w.code.startswith("WARNING_") for w in warnings)

    def test_raise_if_invalid(self):
        """Test raise_if_invalid method."""
        from transformation_portal.utils.input_validation import (
            ImageValidationError,
            ValidationResult,
        )

        result = ValidationResult()
        result.add_error("TEST_ERROR", "Test error")

        with pytest.raises(ImageValidationError) as exc_info:
            result.raise_if_invalid()

        assert "TEST_ERROR" in str(exc_info.value)

    def test_raise_if_invalid_no_raise_when_valid(self):
        """Test raise_if_invalid doesn't raise when valid."""
        from transformation_portal.utils.input_validation import ValidationResult

        result = ValidationResult()
        result.add_warning("TEST_WARNING", "Warning only")

        # Should not raise
        result.raise_if_invalid()


class TestImageValidator:
    """Tests for ImageValidator class."""

    @pytest.fixture
    def validator(self):
        """Create default ImageValidator."""
        from transformation_portal.utils.input_validation import ImageValidator

        return ImageValidator()

    @pytest.fixture
    def strict_validator(self):
        """Create strict ImageValidator."""
        from transformation_portal.utils.input_validation import ImageValidator

        return ImageValidator(
            min_width=512,
            max_width=4096,
            min_height=512,
            max_height=4096,
            require_rgb=True,
            allowed_formats={"JPEG", "PNG", "TIFF"},
        )

    def test_default_formats(self, validator):
        """Test default allowed formats."""
        assert "JPEG" in validator.allowed_formats
        assert "PNG" in validator.allowed_formats
        assert "TIFF" in validator.allowed_formats
        assert "WEBP" in validator.allowed_formats

    def test_validate_file_not_found(self, validator):
        """Test validation of nonexistent file."""
        result = validator.validate(Path("/nonexistent/image.jpg"))

        assert result.is_valid is False
        assert any(e.code == "FILE_NOT_FOUND" for e in result.errors)

    def test_validate_valid_pil_image(self, validator):
        """Test validation of valid PIL image."""
        img = Image.new("RGB", (100, 100), color="red")

        result = validator.validate(img)

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_valid_numpy_array(self, validator):
        """Test validation of valid numpy array."""
        arr = np.zeros((100, 100, 3), dtype=np.uint8)

        result = validator.validate(arr)

        assert result.is_valid is True

    def test_validate_invalid_input_type(self, validator):
        """Test validation of invalid input type."""
        # Strings are treated as file paths, so use a different invalid type
        result = validator.validate(12345)  # Integer is invalid

        assert result.is_valid is False
        assert any(e.code == "INVALID_INPUT_TYPE" for e in result.errors)

    def test_validate_width_too_small(self):
        """Test detection of image too narrow."""
        from transformation_portal.utils.input_validation import ImageValidator

        validator = ImageValidator(min_width=200)
        img = Image.new("RGB", (100, 300))

        result = validator.validate(img)

        assert result.is_valid is False
        assert any(e.code == "WIDTH_TOO_SMALL" for e in result.errors)

    def test_validate_width_too_large(self):
        """Test detection of image too wide."""
        from transformation_portal.utils.input_validation import ImageValidator

        validator = ImageValidator(max_width=500)
        img = Image.new("RGB", (1000, 300))

        result = validator.validate(img)

        assert result.is_valid is False
        assert any(e.code == "WIDTH_TOO_LARGE" for e in result.errors)

    def test_validate_height_too_small(self):
        """Test detection of image too short."""
        from transformation_portal.utils.input_validation import ImageValidator

        validator = ImageValidator(min_height=200)
        img = Image.new("RGB", (300, 100))

        result = validator.validate(img)

        assert result.is_valid is False
        assert any(e.code == "HEIGHT_TOO_SMALL" for e in result.errors)

    def test_validate_height_too_large(self):
        """Test detection of image too tall."""
        from transformation_portal.utils.input_validation import ImageValidator

        validator = ImageValidator(max_height=500)
        img = Image.new("RGB", (300, 1000))

        result = validator.validate(img)

        assert result.is_valid is False
        assert any(e.code == "HEIGHT_TOO_LARGE" for e in result.errors)

    def test_validate_unsupported_color_mode(self):
        """Test detection of unsupported color mode."""
        from transformation_portal.utils.input_validation import ImageValidator

        validator = ImageValidator(allowed_modes={"RGB"})
        img = Image.new("RGBA", (100, 100))

        result = validator.validate(img)

        assert result.is_valid is False
        assert any(e.code == "UNSUPPORTED_COLOR_MODE" for e in result.errors)

    def test_validate_rgb_required_grayscale(self):
        """Test detection when RGB required but grayscale provided."""
        from transformation_portal.utils.input_validation import ImageValidator

        validator = ImageValidator(require_rgb=True)
        img = Image.new("L", (100, 100))

        result = validator.validate(img)

        assert result.is_valid is False
        assert any(e.code == "RGB_REQUIRED" for e in result.errors)

    def test_validate_unusual_aspect_ratio_warning(self):
        """Test warning for unusual aspect ratio."""
        from transformation_portal.utils.input_validation import ImageValidator

        validator = ImageValidator()
        img = Image.new("RGB", (1000, 100))  # 10:1 ratio

        result = validator.validate(img)

        # Should be valid but have warning
        assert result.is_valid is True
        assert any(w.code == "UNUSUAL_ASPECT_RATIO" for w in result.warnings)

    def test_validate_file_with_context(self, validator, temp_workspace):
        """Test validation with context string."""
        img_path = temp_workspace["input_dir"] / "test.jpg"
        img = Image.new("RGB", (100, 100))
        img.save(img_path)

        result = validator.validate(img_path, context="test_context")

        # Context should be in error messages if any
        # Valid image, so just check it doesn't crash
        assert result.is_valid is True


class TestImageValidatorNumpyArrays:
    """Tests for ImageValidator numpy array validation."""

    @pytest.fixture
    def validator(self):
        """Create default ImageValidator."""
        from transformation_portal.utils.input_validation import ImageValidator

        return ImageValidator()

    def test_validate_2d_grayscale(self, validator):
        """Test validation of 2D grayscale array."""
        arr = np.zeros((100, 100), dtype=np.uint8)

        result = validator.validate(arr)

        assert result.is_valid is True

    def test_validate_3d_rgb(self, validator):
        """Test validation of 3D RGB array."""
        arr = np.zeros((100, 100, 3), dtype=np.uint8)

        result = validator.validate(arr)

        assert result.is_valid is True

    def test_validate_3d_rgba(self, validator):
        """Test validation of 3D RGBA array."""
        arr = np.zeros((100, 100, 4), dtype=np.uint8)

        result = validator.validate(arr)

        assert result.is_valid is True

    def test_validate_invalid_dimensions(self, validator):
        """Test detection of invalid array dimensions."""
        arr = np.zeros((10,), dtype=np.uint8)

        result = validator.validate(arr)

        assert result.is_valid is False
        assert any(e.code == "INVALID_ARRAY_DIMENSIONS" for e in result.errors)

    def test_validate_4d_array_invalid(self, validator):
        """Test detection of 4D array as invalid."""
        arr = np.zeros((10, 10, 3, 1), dtype=np.uint8)

        result = validator.validate(arr)

        assert result.is_valid is False
        assert any(e.code == "INVALID_ARRAY_DIMENSIONS" for e in result.errors)

    def test_validate_invalid_channel_count(self, validator):
        """Test detection of invalid channel count."""
        arr = np.zeros((100, 100, 2), dtype=np.uint8)

        result = validator.validate(arr)

        assert result.is_valid is False
        assert any(e.code == "INVALID_CHANNEL_COUNT" for e in result.errors)

    def test_validate_float_array_out_of_range(self, validator):
        """Test warning for float array outside [0, 1]."""
        arr = np.ones((100, 100, 3), dtype=np.float32) * 2.0

        result = validator.validate(arr)

        # Should have warning
        assert any(w.code == "FLOAT_RANGE_WARNING" for w in result.warnings)

    def test_validate_unusual_dtype_warning(self, validator):
        """Test warning for unusual dtype."""
        arr = np.zeros((100, 100), dtype=np.int32)

        result = validator.validate(arr)

        assert any(w.code == "UNUSUAL_DTYPE" for w in result.warnings)

    def test_validate_nan_values(self, validator):
        """Test detection of NaN values."""
        arr = np.ones((100, 100), dtype=np.float32)
        arr[50, 50] = np.nan

        result = validator.validate(arr)

        assert result.is_valid is False
        assert any(e.code == "CONTAINS_NAN" for e in result.errors)

    def test_validate_inf_values(self, validator):
        """Test detection of infinite values."""
        arr = np.ones((100, 100), dtype=np.float32)
        arr[50, 50] = np.inf

        result = validator.validate(arr)

        assert result.is_valid is False
        assert any(e.code == "CONTAINS_INF" for e in result.errors)

    def test_validate_rgb_required_grayscale_array(self):
        """Test RGB required with grayscale array."""
        from transformation_portal.utils.input_validation import ImageValidator

        validator = ImageValidator(require_rgb=True)
        arr = np.zeros((100, 100), dtype=np.uint8)

        result = validator.validate(arr)

        assert result.is_valid is False
        assert any(e.code == "RGB_REQUIRED" for e in result.errors)

    def test_validate_rgb_required_single_channel_3d(self):
        """Test RGB required with single channel 3D array."""
        from transformation_portal.utils.input_validation import ImageValidator

        validator = ImageValidator(require_rgb=True)
        arr = np.zeros((100, 100, 1), dtype=np.uint8)

        result = validator.validate(arr)

        assert result.is_valid is False
        assert any(e.code == "RGB_REQUIRED" for e in result.errors)


class TestImageValidatorFileValidation:
    """Tests for ImageValidator file validation."""

    @pytest.fixture
    def validator(self):
        """Create default ImageValidator."""
        from transformation_portal.utils.input_validation import ImageValidator

        return ImageValidator(check_corruption=True)

    def test_validate_valid_jpeg(self, validator, temp_workspace):
        """Test validation of valid JPEG file."""
        img_path = temp_workspace["input_dir"] / "test.jpg"
        img = Image.new("RGB", (200, 200), color="blue")
        img.save(img_path, "JPEG")

        result = validator.validate(img_path)

        assert result.is_valid is True

    def test_validate_valid_png(self, validator, temp_workspace):
        """Test validation of valid PNG file."""
        img_path = temp_workspace["input_dir"] / "test.png"
        img = Image.new("RGBA", (200, 200), color=(0, 128, 255, 200))
        img.save(img_path, "PNG")

        result = validator.validate(img_path)

        assert result.is_valid is True

    def test_validate_file_too_large(self, temp_workspace):
        """Test detection of file too large."""
        from transformation_portal.utils.input_validation import ImageValidator

        # Create a validator with tiny max file size (0.0001 MB = ~100 bytes)
        validator = ImageValidator(max_file_size_mb=0.0001)

        img_path = temp_workspace["input_dir"] / "large.png"
        # Create uncompressed image that will exceed limit
        img = Image.new("RGB", (500, 500), color="red")
        img.save(img_path, "PNG", compress_level=0)  # No compression

        result = validator.validate(img_path)

        assert result.is_valid is False
        assert any(e.code == "FILE_TOO_LARGE" for e in result.errors)

    def test_validate_corrupted_image(self, validator, temp_workspace):
        """Test detection of corrupted image file."""
        img_path = temp_workspace["input_dir"] / "corrupted.jpg"
        img_path.write_bytes(b"not a valid image")

        result = validator.validate(img_path)

        assert result.is_valid is False
        assert any(e.code == "UNRECOGNIZED_FORMAT" for e in result.errors)

    def test_validate_unsupported_format(self, temp_workspace):
        """Test detection of unsupported format."""
        from transformation_portal.utils.input_validation import ImageValidator

        validator = ImageValidator(allowed_formats={"JPEG"})

        img_path = temp_workspace["input_dir"] / "test.png"
        img = Image.new("RGB", (100, 100))
        img.save(img_path, "PNG")

        result = validator.validate(img_path)

        assert result.is_valid is False
        assert any(e.code == "UNSUPPORTED_FORMAT" for e in result.errors)


class TestBatchValidator:
    """Tests for BatchValidator class."""

    @pytest.fixture
    def batch_validator(self):
        """Create BatchValidator instance."""
        from transformation_portal.utils.input_validation import BatchValidator

        return BatchValidator()

    def test_validate_empty_batch(self, batch_validator):
        """Test validation of empty batch."""
        results = batch_validator.validate_batch([])

        assert results == {}
        assert batch_validator.all_valid is True

    def test_validate_batch_all_valid(self, batch_validator, temp_workspace):
        """Test batch validation with all valid images."""
        paths = []
        for i in range(3):
            path = temp_workspace["input_dir"] / f"valid_{i}.png"
            img = Image.new("RGB", (100, 100))
            img.save(path)
            paths.append(path)

        results = batch_validator.validate_batch(paths)

        assert len(results) == 3
        assert batch_validator.all_valid is True
        assert len(batch_validator.valid_paths) == 3
        assert len(batch_validator.invalid_paths) == 0

    def test_validate_batch_some_invalid(self, batch_validator, temp_workspace):
        """Test batch validation with some invalid images."""
        # Create valid image
        valid_path = temp_workspace["input_dir"] / "valid.png"
        Image.new("RGB", (100, 100)).save(valid_path)

        # Invalid path (nonexistent)
        invalid_path = temp_workspace["input_dir"] / "nonexistent.png"

        results = batch_validator.validate_batch([valid_path, invalid_path])

        assert len(results) == 2
        assert batch_validator.all_valid is False
        assert len(batch_validator.valid_paths) == 1
        assert len(batch_validator.invalid_paths) == 1

    def test_validate_batch_stop_on_first_error(self, batch_validator, temp_workspace):
        """Test batch validation stops on first error."""
        invalid_path = temp_workspace["input_dir"] / "nonexistent.png"
        valid_path = temp_workspace["input_dir"] / "valid.png"
        Image.new("RGB", (100, 100)).save(valid_path)

        results = batch_validator.validate_batch(
            [invalid_path, valid_path],
            stop_on_first_error=True,
        )

        # Should only have processed first file
        assert len(results) == 1
        assert str(invalid_path) in results

    def test_batch_summary(self, batch_validator, temp_workspace):
        """Test batch summary generation."""
        # Create some valid and invalid paths
        valid_path = temp_workspace["input_dir"] / "valid.png"
        Image.new("RGB", (100, 100)).save(valid_path)
        invalid_path = temp_workspace["input_dir"] / "nonexistent.png"

        batch_validator.validate_batch([valid_path, invalid_path])
        summary = batch_validator.summary()

        assert summary["total"] == 2
        assert summary["valid"] == 1
        assert summary["invalid"] == 1
        assert summary["success_rate"] == 0.5


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_validate_image_function(self, temp_workspace):
        """Test validate_image convenience function."""
        from transformation_portal.utils.input_validation import validate_image

        img_path = temp_workspace["input_dir"] / "test.png"
        Image.new("RGB", (100, 100)).save(img_path)

        result = validate_image(img_path)

        assert result.is_valid is True

    def test_validate_image_strict_function(self, temp_workspace):
        """Test validate_image_strict convenience function."""
        from transformation_portal.utils.input_validation import validate_image_strict

        img_path = temp_workspace["input_dir"] / "test.jpg"
        Image.new("RGB", (1024, 1024)).save(img_path, "JPEG")

        result = validate_image_strict(img_path)

        assert result.is_valid is True

    def test_validate_image_strict_too_small(self, temp_workspace):
        """Test validate_image_strict with small image."""
        from transformation_portal.utils.input_validation import validate_image_strict

        img_path = temp_workspace["input_dir"] / "small.jpg"
        Image.new("RGB", (100, 100)).save(img_path, "JPEG")

        result = validate_image_strict(img_path, min_size=512)

        assert result.is_valid is False

    def test_require_valid_image_raises(self, temp_workspace):
        """Test require_valid_image raises on invalid."""
        from transformation_portal.utils.input_validation import (
            ImageValidationError,
            require_valid_image,
        )

        invalid_path = temp_workspace["input_dir"] / "nonexistent.png"

        with pytest.raises(ImageValidationError):
            require_valid_image(invalid_path)

    def test_require_valid_image_no_raise(self, temp_workspace):
        """Test require_valid_image doesn't raise on valid."""
        from transformation_portal.utils.input_validation import require_valid_image

        img_path = temp_workspace["input_dir"] / "valid.png"
        Image.new("RGB", (100, 100)).save(img_path)

        # Should not raise
        require_valid_image(img_path)

    def test_is_valid_image_true(self, temp_workspace):
        """Test is_valid_image returns True for valid image."""
        from transformation_portal.utils.input_validation import is_valid_image

        img_path = temp_workspace["input_dir"] / "valid.png"
        Image.new("RGB", (100, 100)).save(img_path)

        assert is_valid_image(img_path) is True

    def test_is_valid_image_false(self, temp_workspace):
        """Test is_valid_image returns False for invalid image."""
        from transformation_portal.utils.input_validation import is_valid_image

        invalid_path = temp_workspace["input_dir"] / "nonexistent.png"

        assert is_valid_image(invalid_path) is False


class TestVeryHighResolutionWarning:
    """Tests for very high resolution warning."""

    def test_very_high_resolution_warning(self):
        """Test warning for very high resolution image."""
        from transformation_portal.utils.input_validation import ImageValidator

        validator = ImageValidator()

        # Create a PIL image that exceeds 50 megapixels (e.g., 8000x7000 = 56MP)
        # Using a smaller size that still triggers the warning
        img = Image.new("RGB", (8000, 7000), color="blue")

        result = validator.validate(img)

        assert result.is_valid is True
        assert any(w.code == "VERY_HIGH_RESOLUTION" for w in result.warnings)


class TestImageValidationError:
    """Tests for ImageValidationError exception."""

    def test_exception_inherits_from_processing_error(self):
        """Test that ImageValidationError inherits from ProcessingError."""
        from transformation_portal.utils.error_handling import ProcessingError
        from transformation_portal.utils.input_validation import ImageValidationError

        assert issubclass(ImageValidationError, ProcessingError)

    def test_exception_message(self):
        """Test exception message."""
        from transformation_portal.utils.input_validation import ImageValidationError

        exc = ImageValidationError("Test validation failed")

        assert "Test validation failed" in str(exc)
