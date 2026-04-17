"""Tests for input validation utilities.

This module tests the input validation utilities from utils/input_validation.py
which provides comprehensive validation for:
- Image files (format, dimensions, color space, corruption)
- File paths and permissions
- Pipeline configurations
- Context data

Covers:
- ValidationSeverity enum
- ValidationIssue dataclass
- ValidationResult class
- ImageValidator class
- BatchValidator class
- Convenience functions (validate_image, validate_image_strict, etc.)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytestmark = [pytest.mark.unit, pytest.mark.security]


# =============================================================================
# Test ValidationSeverity enum
# =============================================================================


@pytest.mark.security
class TestValidationSeverity:
    """Tests for ValidationSeverity enum."""

    def test_severity_values(self) -> None:
        """ValidationSeverity has expected values."""
        from transformation_portal.utils.input_validation import ValidationSeverity

        assert ValidationSeverity.INFO.value == "info"
        assert ValidationSeverity.WARNING.value == "warning"
        assert ValidationSeverity.ERROR.value == "error"

    def test_severity_comparison(self) -> None:
        """ValidationSeverity values can be used in comparisons."""
        from transformation_portal.utils.input_validation import ValidationSeverity

        # Enum members are distinct
        assert ValidationSeverity.ERROR != ValidationSeverity.WARNING
        assert ValidationSeverity.WARNING != ValidationSeverity.INFO


# =============================================================================
# Test ValidationIssue dataclass
# =============================================================================


@pytest.mark.security
class TestValidationIssue:
    """Tests for ValidationIssue dataclass."""

    def test_create_issue(self) -> None:
        """ValidationIssue can be created with required fields."""
        from transformation_portal.utils.input_validation import (
            ValidationIssue,
            ValidationSeverity,
        )

        issue = ValidationIssue(
            code="TEST_ERROR",
            message="Test error message",
            severity=ValidationSeverity.ERROR,
        )

        assert issue.code == "TEST_ERROR"
        assert issue.message == "Test error message"
        assert issue.severity == ValidationSeverity.ERROR
        assert issue.suggestion is None
        assert issue.details == {}

    def test_create_issue_with_optional_fields(self) -> None:
        """ValidationIssue can include suggestion and details."""
        from transformation_portal.utils.input_validation import (
            ValidationIssue,
            ValidationSeverity,
        )

        issue = ValidationIssue(
            code="TEST_WARNING",
            message="Warning message",
            severity=ValidationSeverity.WARNING,
            suggestion="Try this instead",
            details={"key": "value"},
        )

        assert issue.suggestion == "Try this instead"
        assert issue.details == {"key": "value"}

    def test_issue_str_representation(self) -> None:
        """ValidationIssue str includes code, message, and severity."""
        from transformation_portal.utils.input_validation import (
            ValidationIssue,
            ValidationSeverity,
        )

        issue = ValidationIssue(
            code="FILE_ERROR",
            message="File not found",
            severity=ValidationSeverity.ERROR,
        )

        result = str(issue)

        assert "[ERROR]" in result
        assert "FILE_ERROR" in result
        assert "File not found" in result

    def test_issue_str_with_suggestion(self) -> None:
        """ValidationIssue str includes suggestion when present."""
        from transformation_portal.utils.input_validation import (
            ValidationIssue,
            ValidationSeverity,
        )

        issue = ValidationIssue(
            code="SIZE_WARNING",
            message="File is large",
            severity=ValidationSeverity.WARNING,
            suggestion="Consider compression",
        )

        result = str(issue)

        assert "Consider compression" in result


# =============================================================================
# Test ValidationResult class
# =============================================================================


@pytest.mark.security
class TestValidationResult:
    """Tests for ValidationResult class."""

    def test_create_result_default_valid(self) -> None:
        """ValidationResult is valid by default."""
        from transformation_portal.utils.input_validation import ValidationResult

        result = ValidationResult()

        assert result.is_valid is True
        assert result.issues == []

    def test_add_issue(self) -> None:
        """ValidationResult.add_issue adds issue to list."""
        from transformation_portal.utils.input_validation import (
            ValidationIssue,
            ValidationResult,
            ValidationSeverity,
        )

        result = ValidationResult()
        issue = ValidationIssue(
            code="TEST",
            message="Test",
            severity=ValidationSeverity.INFO,
        )

        result.add_issue(issue)

        assert len(result.issues) == 1
        assert result.issues[0] is issue

    def test_add_error_sets_invalid(self) -> None:
        """Adding error-level issue sets is_valid to False."""
        from transformation_portal.utils.input_validation import (
            ValidationIssue,
            ValidationResult,
            ValidationSeverity,
        )

        result = ValidationResult()

        error_issue = ValidationIssue(
            code="ERROR",
            message="Error",
            severity=ValidationSeverity.ERROR,
        )
        result.add_issue(error_issue)

        assert result.is_valid is False

    def test_add_warning_keeps_valid(self) -> None:
        """Adding warning-level issue keeps is_valid True."""
        from transformation_portal.utils.input_validation import (
            ValidationIssue,
            ValidationResult,
            ValidationSeverity,
        )

        result = ValidationResult()

        warning_issue = ValidationIssue(
            code="WARNING",
            message="Warning",
            severity=ValidationSeverity.WARNING,
        )
        result.add_issue(warning_issue)

        assert result.is_valid is True
        assert len(result.issues) == 1

    def test_add_error_convenience(self) -> None:
        """ValidationResult.add_error is convenience method."""
        from transformation_portal.utils.input_validation import ValidationResult

        result = ValidationResult()
        result.add_error("CODE", "Message", suggestion="Fix it")

        assert result.is_valid is False
        assert len(result.issues) == 1
        assert result.issues[0].code == "CODE"
        assert result.issues[0].suggestion == "Fix it"

    def test_add_warning_convenience(self) -> None:
        """ValidationResult.add_warning is convenience method."""
        from transformation_portal.utils.input_validation import ValidationResult

        result = ValidationResult()
        result.add_warning("WARN_CODE", "Warning message")

        assert result.is_valid is True
        assert len(result.warnings) == 1

    def test_errors_property(self) -> None:
        """ValidationResult.errors returns only error-level issues."""
        from transformation_portal.utils.input_validation import ValidationResult

        result = ValidationResult()
        result.add_error("ERROR1", "Error 1")
        result.add_warning("WARN1", "Warning 1")
        result.add_error("ERROR2", "Error 2")

        assert len(result.errors) == 2
        assert all(e.code.startswith("ERROR") for e in result.errors)

    def test_warnings_property(self) -> None:
        """ValidationResult.warnings returns only warning-level issues."""
        from transformation_portal.utils.input_validation import ValidationResult

        result = ValidationResult()
        result.add_error("ERROR1", "Error 1")
        result.add_warning("WARN1", "Warning 1")
        result.add_warning("WARN2", "Warning 2")

        assert len(result.warnings) == 2
        assert all(w.code.startswith("WARN") for w in result.warnings)

    def test_raise_if_invalid_does_not_raise_when_valid(self) -> None:
        """ValidationResult.raise_if_invalid does nothing when valid."""
        from transformation_portal.utils.input_validation import ValidationResult

        result = ValidationResult()
        result.add_warning("WARN", "Just a warning")

        # Should not raise
        result.raise_if_invalid()

    def test_raise_if_invalid_raises_when_invalid(self) -> None:
        """ValidationResult.raise_if_invalid raises ImageValidationError when invalid."""
        from transformation_portal.utils.input_validation import (
            ImageValidationError,
            ValidationResult,
        )

        result = ValidationResult()
        result.add_error("ERROR1", "First error")
        result.add_error("ERROR2", "Second error")

        with pytest.raises(ImageValidationError) as exc_info:
            result.raise_if_invalid()

        assert "2 error(s)" in str(exc_info.value)


# =============================================================================
# Test ImageValidator class
# =============================================================================


@pytest.mark.security
class TestImageValidator:
    """Tests for ImageValidator class."""

    def test_create_validator_defaults(self) -> None:
        """ImageValidator creates with default settings."""
        from transformation_portal.utils.input_validation import ImageValidator

        validator = ImageValidator()

        assert validator.min_width == 64
        assert validator.max_width == 16384
        assert validator.min_height == 64
        assert validator.max_height == 16384
        assert validator.max_file_size_mb == 500

    def test_create_validator_custom_settings(self) -> None:
        """ImageValidator accepts custom settings."""
        from transformation_portal.utils.input_validation import ImageValidator

        validator = ImageValidator(
            min_width=128,
            max_width=4096,
            min_height=128,
            max_height=4096,
            max_file_size_mb=100,
            require_rgb=True,
        )

        assert validator.min_width == 128
        assert validator.max_width == 4096
        assert validator.require_rgb is True

    def test_validate_file_not_found(self, tmp_path: Path) -> None:
        """Validation fails for non-existent file."""
        from transformation_portal.utils.input_validation import ImageValidator

        validator = ImageValidator()
        result = validator.validate(tmp_path / "nonexistent.jpg")

        assert result.is_valid is False
        assert any(e.code == "FILE_NOT_FOUND" for e in result.errors)

    def test_validate_file_too_large(self, tmp_path: Path) -> None:
        """Validation fails for file exceeding size limit."""
        from transformation_portal.utils.input_validation import ImageValidator

        # Create a valid but large-ish image (we'll use a small limit)
        img = Image.new("RGB", (100, 100), color="red")
        img_path = tmp_path / "large.png"
        img.save(img_path)

        validator = ImageValidator(max_file_size_mb=0.0001)  # Very small limit
        result = validator.validate(img_path)

        assert result.is_valid is False
        assert any(e.code == "FILE_TOO_LARGE" for e in result.errors)

    def test_validate_unrecognized_format(self, tmp_path: Path) -> None:
        """Validation fails for unrecognized image format."""
        from transformation_portal.utils.input_validation import ImageValidator

        fake_file = tmp_path / "fake.xyz"
        fake_file.write_bytes(b"not an image")

        validator = ImageValidator()
        result = validator.validate(fake_file)

        assert result.is_valid is False
        assert any(e.code == "UNRECOGNIZED_FORMAT" for e in result.errors)

    def test_validate_valid_image_file(self, tmp_path: Path) -> None:
        """Validation passes for valid image file."""
        from transformation_portal.utils.input_validation import ImageValidator

        img = Image.new("RGB", (512, 512), color="blue")
        img_path = tmp_path / "valid.png"
        img.save(img_path)

        validator = ImageValidator()
        result = validator.validate(img_path)

        assert result.is_valid is True

    def test_validate_width_too_small(self, tmp_path: Path) -> None:
        """Validation fails for image below min width."""
        from transformation_portal.utils.input_validation import ImageValidator

        img = Image.new("RGB", (32, 512), color="blue")
        img_path = tmp_path / "narrow.png"
        img.save(img_path)

        validator = ImageValidator(min_width=64)
        result = validator.validate(img_path)

        assert result.is_valid is False
        assert any(e.code == "WIDTH_TOO_SMALL" for e in result.errors)

    def test_validate_width_too_large(self, tmp_path: Path) -> None:
        """Validation fails for image above max width."""
        from transformation_portal.utils.input_validation import ImageValidator

        img = Image.new("RGB", (200, 100), color="blue")
        img_path = tmp_path / "wide.png"
        img.save(img_path)

        validator = ImageValidator(max_width=150)
        result = validator.validate(img_path)

        assert result.is_valid is False
        assert any(e.code == "WIDTH_TOO_LARGE" for e in result.errors)

    def test_validate_height_constraints(self, tmp_path: Path) -> None:
        """Validation enforces height constraints."""
        from transformation_portal.utils.input_validation import ImageValidator

        # Too small
        small_img = Image.new("RGB", (512, 32), color="blue")
        small_path = tmp_path / "short.png"
        small_img.save(small_path)

        validator = ImageValidator(min_height=64)
        result = validator.validate(small_path)

        assert any(e.code == "HEIGHT_TOO_SMALL" for e in result.errors)

        # Too large
        tall_img = Image.new("RGB", (100, 200), color="blue")
        tall_path = tmp_path / "tall.png"
        tall_img.save(tall_path)

        validator = ImageValidator(max_height=150)
        result = validator.validate(tall_path)

        assert any(e.code == "HEIGHT_TOO_LARGE" for e in result.errors)

    def test_validate_require_rgb(self, tmp_path: Path) -> None:
        """Validation enforces RGB requirement when set."""
        from transformation_portal.utils.input_validation import ImageValidator

        # Grayscale image
        gray_img = Image.new("L", (256, 256), color=128)
        gray_path = tmp_path / "gray.png"
        gray_img.save(gray_path)

        validator = ImageValidator(require_rgb=True)
        result = validator.validate(gray_path)

        assert result.is_valid is False
        assert any(e.code == "RGB_REQUIRED" for e in result.errors)

    def test_validate_allowed_formats(self, tmp_path: Path) -> None:
        """Validation enforces allowed formats."""
        from transformation_portal.utils.input_validation import ImageValidator

        img = Image.new("RGB", (256, 256), color="red")
        bmp_path = tmp_path / "test.bmp"
        img.save(bmp_path, format="BMP")

        validator = ImageValidator(allowed_formats={"JPEG", "PNG"})
        result = validator.validate(bmp_path)

        assert result.is_valid is False
        assert any(e.code == "UNSUPPORTED_FORMAT" for e in result.errors)

    def test_validate_pil_image_directly(self) -> None:
        """Validation works on PIL Image objects."""
        from transformation_portal.utils.input_validation import ImageValidator

        img = Image.new("RGB", (512, 512), color="green")

        validator = ImageValidator()
        result = validator.validate(img)

        assert result.is_valid is True

    def test_validate_numpy_array(self) -> None:
        """Validation works on numpy arrays."""
        from transformation_portal.utils.input_validation import ImageValidator

        arr = np.zeros((256, 256, 3), dtype=np.uint8)

        validator = ImageValidator()
        result = validator.validate(arr)

        assert result.is_valid is True

    def test_validate_numpy_grayscale(self) -> None:
        """Validation works on grayscale numpy arrays."""
        from transformation_portal.utils.input_validation import ImageValidator

        arr = np.zeros((256, 256), dtype=np.uint8)

        validator = ImageValidator(require_rgb=False)
        result = validator.validate(arr)

        assert result.is_valid is True

    def test_validate_numpy_invalid_dimensions(self) -> None:
        """Validation fails for arrays with invalid dimensions."""
        from transformation_portal.utils.input_validation import ImageValidator

        arr = np.zeros((256,), dtype=np.uint8)  # 1D array

        validator = ImageValidator()
        result = validator.validate(arr)

        assert result.is_valid is False
        assert any(e.code == "INVALID_ARRAY_DIMENSIONS" for e in result.errors)

    def test_validate_numpy_invalid_channels(self) -> None:
        """Validation fails for arrays with wrong channel count."""
        from transformation_portal.utils.input_validation import ImageValidator

        arr = np.zeros((256, 256, 5), dtype=np.uint8)  # 5 channels

        validator = ImageValidator()
        result = validator.validate(arr)

        assert result.is_valid is False
        assert any(e.code == "INVALID_CHANNEL_COUNT" for e in result.errors)

    def test_validate_numpy_contains_nan(self) -> None:
        """Validation fails for arrays containing NaN."""
        from transformation_portal.utils.input_validation import ImageValidator

        arr = np.zeros((256, 256, 3), dtype=np.float32)
        arr[100, 100, 0] = np.nan

        validator = ImageValidator()
        result = validator.validate(arr)

        assert result.is_valid is False
        assert any(e.code == "CONTAINS_NAN" for e in result.errors)

    def test_validate_numpy_contains_inf(self) -> None:
        """Validation fails for arrays containing infinity."""
        from transformation_portal.utils.input_validation import ImageValidator

        arr = np.zeros((256, 256, 3), dtype=np.float32)
        arr[100, 100, 0] = np.inf

        validator = ImageValidator()
        result = validator.validate(arr)

        assert result.is_valid is False
        assert any(e.code == "CONTAINS_INF" for e in result.errors)

    def test_validate_invalid_input_type(self) -> None:
        """Validation fails for unsupported input types."""
        from transformation_portal.utils.input_validation import ImageValidator

        validator = ImageValidator()
        result = validator.validate("not a path object")  # type: ignore

        # String that's not a valid path
        assert result.is_valid is False

    def test_validate_with_context(self, tmp_path: Path) -> None:
        """Validation includes context in error messages."""
        from transformation_portal.utils.input_validation import ImageValidator

        validator = ImageValidator()
        result = validator.validate(tmp_path / "missing.jpg", context="input_image")

        assert result.is_valid is False
        # Context should appear in error message
        assert any("[input_image]" in str(e) for e in result.errors)

    def test_warns_unusual_aspect_ratio(self) -> None:
        """Validation warns on unusual aspect ratios."""
        from transformation_portal.utils.input_validation import ImageValidator

        # Very wide image
        wide_img = Image.new("RGB", (1000, 100), color="blue")

        validator = ImageValidator()
        result = validator.validate(wide_img)

        assert any(w.code == "UNUSUAL_ASPECT_RATIO" for w in result.warnings)

    def test_warns_float_range(self) -> None:
        """Validation warns when float array outside [0,1] range."""
        from transformation_portal.utils.input_validation import ImageValidator

        arr = np.ones((256, 256, 3), dtype=np.float32) * 2.0  # Values at 2.0

        validator = ImageValidator()
        result = validator.validate(arr)

        assert any(w.code == "FLOAT_RANGE_WARNING" for w in result.warnings)


# =============================================================================
# Test BatchValidator class
# =============================================================================


@pytest.mark.security
class TestBatchValidator:
    """Tests for BatchValidator class."""

    def test_batch_validate_all_valid(self, tmp_path: Path) -> None:
        """Batch validation succeeds for all valid images."""
        from transformation_portal.utils.input_validation import BatchValidator

        # Create valid images
        for i in range(3):
            img = Image.new("RGB", (256, 256), color="red")
            img.save(tmp_path / f"image{i}.png")

        paths = [tmp_path / f"image{i}.png" for i in range(3)]

        validator = BatchValidator()
        results = validator.validate_batch(paths)

        assert validator.all_valid is True
        assert len(validator.valid_paths) == 3
        assert len(validator.invalid_paths) == 0

    def test_batch_validate_mixed(self, tmp_path: Path) -> None:
        """Batch validation correctly identifies invalid images."""
        from transformation_portal.utils.input_validation import BatchValidator

        # Create one valid image
        valid_img = Image.new("RGB", (256, 256), color="blue")
        valid_img.save(tmp_path / "valid.png")

        # Create one invalid file
        invalid_path = tmp_path / "invalid.txt"
        invalid_path.write_text("not an image")

        paths = [tmp_path / "valid.png", tmp_path / "invalid.txt"]

        validator = BatchValidator()
        results = validator.validate_batch(paths)

        assert validator.all_valid is False
        assert len(validator.valid_paths) == 1
        assert len(validator.invalid_paths) == 1

    def test_batch_validate_stop_on_first_error(self, tmp_path: Path) -> None:
        """Batch validation can stop on first error."""
        from transformation_portal.utils.input_validation import BatchValidator

        # Create invalid files first
        for i in range(5):
            (tmp_path / f"invalid{i}.txt").write_text("not an image")

        paths = [tmp_path / f"invalid{i}.txt" for i in range(5)]

        validator = BatchValidator()
        results = validator.validate_batch(paths, stop_on_first_error=True)

        # Should stop after first error
        assert len(results) == 1

    def test_batch_summary(self, tmp_path: Path) -> None:
        """Batch validator provides summary statistics."""
        from transformation_portal.utils.input_validation import BatchValidator

        # Create mix of valid and invalid
        for i in range(2):
            img = Image.new("RGB", (256, 256), color="green")
            img.save(tmp_path / f"valid{i}.png")

        (tmp_path / "invalid.txt").write_text("not an image")

        paths = [
            tmp_path / "valid0.png",
            tmp_path / "valid1.png",
            tmp_path / "invalid.txt",
        ]

        validator = BatchValidator()
        validator.validate_batch(paths)

        summary = validator.summary()

        assert summary["total"] == 3
        assert summary["valid"] == 2
        assert summary["invalid"] == 1
        assert 0.6 < summary["success_rate"] < 0.7


# =============================================================================
# Test convenience functions
# =============================================================================


@pytest.mark.security
class TestConvenienceFunctions:
    """Tests for convenience validation functions."""

    def test_validate_image(self, tmp_path: Path) -> None:
        """validate_image function works."""
        from transformation_portal.utils.input_validation import validate_image

        img = Image.new("RGB", (256, 256), color="purple")
        img_path = tmp_path / "test.png"
        img.save(img_path)

        result = validate_image(img_path)

        assert result.is_valid is True

    def test_validate_image_strict(self, tmp_path: Path) -> None:
        """validate_image_strict applies stricter settings."""
        from transformation_portal.utils.input_validation import validate_image_strict

        # Image below strict minimum (512)
        small_img = Image.new("RGB", (256, 256), color="orange")
        small_path = tmp_path / "small.png"
        small_img.save(small_path)

        result = validate_image_strict(small_path)

        assert result.is_valid is False

    def test_validate_image_strict_passes_large(self, tmp_path: Path) -> None:
        """validate_image_strict passes for large enough images."""
        from transformation_portal.utils.input_validation import validate_image_strict

        large_img = Image.new("RGB", (1024, 1024), color="cyan")
        large_path = tmp_path / "large.png"
        large_img.save(large_path)

        result = validate_image_strict(large_path)

        assert result.is_valid is True

    def test_require_valid_image_passes(self, tmp_path: Path) -> None:
        """require_valid_image doesn't raise for valid image."""
        from transformation_portal.utils.input_validation import require_valid_image

        img = Image.new("RGB", (256, 256), color="yellow")
        img_path = tmp_path / "valid.png"
        img.save(img_path)

        # Should not raise
        require_valid_image(img_path)

    def test_require_valid_image_raises(self, tmp_path: Path) -> None:
        """require_valid_image raises for invalid image."""
        from transformation_portal.utils.input_validation import (
            ImageValidationError,
            require_valid_image,
        )

        invalid_path = tmp_path / "invalid.txt"
        invalid_path.write_text("not an image")

        with pytest.raises(ImageValidationError):
            require_valid_image(invalid_path)

    def test_is_valid_image_returns_bool(self, tmp_path: Path) -> None:
        """is_valid_image returns boolean."""
        from transformation_portal.utils.input_validation import is_valid_image

        img = Image.new("RGB", (256, 256), color="magenta")
        img_path = tmp_path / "test.png"
        img.save(img_path)

        assert is_valid_image(img_path) is True

        invalid_path = tmp_path / "invalid.xyz"
        invalid_path.write_bytes(b"not an image")

        assert is_valid_image(invalid_path) is False


# =============================================================================
# Test edge cases and boundary conditions
# =============================================================================


@pytest.mark.security
class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_validate_zero_dimension_image(self) -> None:
        """Validation handles zero-dimension arrays."""
        from transformation_portal.utils.input_validation import ImageValidator

        arr = np.zeros((0, 0, 3), dtype=np.uint8)

        validator = ImageValidator()
        result = validator.validate(arr)

        # Should fail dimension checks
        assert result.is_valid is False

    def test_validate_single_pixel_image(self) -> None:
        """Validation handles single pixel images."""
        from transformation_portal.utils.input_validation import ImageValidator

        img = Image.new("RGB", (1, 1), color="white")

        validator = ImageValidator(min_width=1, min_height=1)
        result = validator.validate(img)

        # Should pass with relaxed constraints
        assert result.is_valid is True

    def test_validate_exact_boundary_dimensions(self) -> None:
        """Validation handles exact boundary dimensions."""
        from transformation_portal.utils.input_validation import ImageValidator

        # Exactly at min width
        img = Image.new("RGB", (64, 100), color="white")

        validator = ImageValidator(min_width=64, min_height=64)
        result = validator.validate(img)

        assert result.is_valid is True

        # Exactly at max width
        img2 = Image.new("RGB", (16384, 100), color="white")

        validator2 = ImageValidator(max_width=16384, min_height=64)
        result2 = validator2.validate(img2)

        assert result2.is_valid is True

    def test_validate_rgba_image(self, tmp_path: Path) -> None:
        """Validation handles RGBA images correctly."""
        from transformation_portal.utils.input_validation import ImageValidator

        img = Image.new("RGBA", (256, 256), color=(255, 0, 0, 128))
        img_path = tmp_path / "rgba.png"
        img.save(img_path)

        validator = ImageValidator(require_rgb=True)
        result = validator.validate(img_path)

        # RGBA should satisfy require_rgb
        assert result.is_valid is True

    def test_validate_16bit_image(self, tmp_path: Path) -> None:
        """Validation handles 16-bit images."""
        from transformation_portal.utils.input_validation import ImageValidator

        arr = np.zeros((256, 256), dtype=np.uint16)
        arr[128, 128] = 65535

        validator = ImageValidator(require_rgb=False)
        result = validator.validate(arr)

        assert result.is_valid is True

    def test_validate_float64_array(self) -> None:
        """Validation handles float64 arrays."""
        from transformation_portal.utils.input_validation import ImageValidator

        arr = np.zeros((256, 256, 3), dtype=np.float64)

        validator = ImageValidator()
        result = validator.validate(arr)

        # Should pass (within 0-1 range)
        assert result.is_valid is True

    def test_validate_unreadable_file(self, tmp_path: Path) -> None:
        """Validation handles unreadable files."""
        from transformation_portal.utils.input_validation import ImageValidator

        # Create a file
        test_file = tmp_path / "unreadable.png"
        test_file.write_bytes(b"fake content")

        # Make it unreadable (Unix only)
        if os.name != "nt":
            os.chmod(test_file, 0o000)

            try:
                validator = ImageValidator()
                result = validator.validate(test_file)

                assert result.is_valid is False
                assert any(e.code == "FILE_NOT_READABLE" for e in result.errors)
            finally:
                # Restore permissions for cleanup
                os.chmod(test_file, 0o644)

    def test_validate_dict_input_fails(self) -> None:
        """Validation fails gracefully for dict input."""
        from transformation_portal.utils.input_validation import ImageValidator

        validator = ImageValidator()
        result = validator.validate({"not": "valid"})  # type: ignore

        assert result.is_valid is False
        assert any(e.code == "INVALID_INPUT_TYPE" for e in result.errors)

    def test_validate_none_input_fails(self) -> None:
        """Validation fails gracefully for None input."""
        from transformation_portal.utils.input_validation import ImageValidator

        validator = ImageValidator()
        result = validator.validate(None)  # type: ignore

        assert result.is_valid is False
