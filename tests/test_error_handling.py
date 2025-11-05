#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for error handling utilities."""
import pytest

# Use proper package imports (assumes package is installed or PYTHONPATH is set)
# For development: pip install -e . or set PYTHONPATH to include src/
from transformation_portal.utils.error_handling import (
    ProcessingError,
    FileValidationError,
    DependencyError,
    ConfigurationError,
    validate_file_path,
    validate_directory,
    check_dependency,
    safe_execute,
    validate_range,
    batch_with_error_handling,
    get_error_summary,
)


class TestFileValidation:
    """Tests for file path validation."""
    
    def test_validate_existing_file(self, tmp_path):
        """Test validation of existing file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        
        validated = validate_file_path(test_file, must_exist=True)
        assert validated.exists()
        assert validated.is_absolute()
    
    def test_validate_nonexistent_file_required(self, tmp_path):
        """Test validation fails for missing file when required."""
        test_file = tmp_path / "missing.txt"
        
        with pytest.raises(FileValidationError, match="File not found"):
            validate_file_path(test_file, must_exist=True)
    
    def test_validate_nonexistent_file_optional(self, tmp_path):
        """Test validation succeeds for missing file when optional."""
        test_file = tmp_path / "missing.txt"
        validated = validate_file_path(test_file, must_exist=False)
        assert validated.is_absolute()
    
    def test_validate_file_extension(self, tmp_path):
        """Test file extension validation."""
        jpg_file = tmp_path / "image.jpg"
        jpg_file.write_text("fake image")
        
        # Valid extension
        validated = validate_file_path(
            jpg_file,
            extensions=['.jpg', '.png']
        )
        assert validated.exists()
        
        # Invalid extension
        txt_file = tmp_path / "doc.txt"
        txt_file.write_text("text")
        
        with pytest.raises(FileValidationError, match="Invalid file extension"):
            validate_file_path(txt_file, extensions=['.jpg', '.png'])
    
    def test_validate_invalid_path(self):
        """Test validation of invalid path."""
        with pytest.raises(FileValidationError, match="Invalid path format"):
            validate_file_path(None)


class TestDirectoryValidation:
    """Tests for directory validation."""
    
    def test_validate_existing_directory(self, tmp_path):
        """Test validation of existing directory."""
        validated = validate_directory(tmp_path)
        assert validated.is_dir()
        assert validated.is_absolute()
    
    def test_validate_nonexistent_directory_create(self, tmp_path):
        """Test directory creation."""
        new_dir = tmp_path / "new_directory"
        assert not new_dir.exists()
        
        validated = validate_directory(new_dir, create=True)
        assert validated.exists()
        assert validated.is_dir()
    
    def test_validate_nonexistent_directory_no_create(self, tmp_path):
        """Test validation fails without create flag."""
        new_dir = tmp_path / "missing_directory"
        
        with pytest.raises(FileValidationError, match="Directory not found"):
            validate_directory(new_dir, create=False)
    
    def test_validate_directory_writable(self, tmp_path):
        """Test writable check."""
        # Should succeed for writable temp directory
        validated = validate_directory(tmp_path, writable=True)
        assert validated.is_dir()
    
    def test_validate_path_not_directory(self, tmp_path):
        """Test validation fails for file path."""
        test_file = tmp_path / "file.txt"
        test_file.write_text("content")
        
        with pytest.raises(FileValidationError, match="not a directory"):
            validate_directory(test_file)


class TestDependencyChecking:
    """Tests for dependency checking."""
    
    def test_check_available_dependency(self):
        """Test checking for available module."""
        # sys should always be available
        result = check_dependency('sys')
        assert result is True
    
    def test_check_missing_dependency(self):
        """Test checking for missing module."""
        with pytest.raises(DependencyError, match="not installed"):
            check_dependency('nonexistent_module_xyz123')
    
    def test_check_dependency_with_version(self):
        """Test version checking (if packaging available)."""
        try:
            # Try to check Python version
            result = check_dependency('sys', min_version='3.0.0')
            assert result is True
        except DependencyError:
            # OK if packaging not available or version too old
            pass


class TestSafeExecute:
    """Tests for safe execution wrapper."""
    
    def test_safe_execute_success(self):
        """Test successful execution."""
        def successful_func(x):
            return x * 2
        
        result = safe_execute(successful_func, 5)
        assert result == 10
    
    def test_safe_execute_with_exception(self):
        """Test execution with exception returns default."""
        def failing_func():
            raise ValueError("test error")
        
        result = safe_execute(
            failing_func,
            default=42,
            log_errors=False  # Suppress logging in test
        )
        assert result == 42
    
    def test_safe_execute_with_kwargs(self):
        """Test execution with keyword arguments."""
        def func_with_kwargs(a, b=10):
            return a + b
        
        result = safe_execute(func_with_kwargs, 5, b=20)
        assert result == 25


class TestValidateRange:
    """Tests for range validation."""
    
    def test_validate_within_range(self):
        """Test value within valid range."""
        value = validate_range(0.5, min_value=0.0, max_value=1.0)
        assert value == 0.5
    
    def test_validate_below_minimum(self):
        """Test value below minimum."""
        with pytest.raises(ConfigurationError, match="must be >= 0.0"):
            validate_range(-0.1, min_value=0.0, max_value=1.0)
    
    def test_validate_above_maximum(self):
        """Test value above maximum."""
        with pytest.raises(ConfigurationError, match="must be <= 1.0"):
            validate_range(1.5, min_value=0.0, max_value=1.0)
    
    def test_validate_no_bounds(self):
        """Test validation with no bounds."""
        value = validate_range(1000)
        assert value == 1000
    
    def test_validate_custom_name(self):
        """Test error message includes parameter name."""
        with pytest.raises(ConfigurationError, match="strength"):
            validate_range(2.0, min_value=0.0, max_value=1.0, name="strength")


class TestBatchErrorHandling:
    """Tests for batch processing with error handling."""
    
    def test_batch_all_success(self):
        """Test batch processing with all successes."""
        items = [1, 2, 3, 4, 5]
        results = batch_with_error_handling(
            items,
            lambda x: x * 2,
            skip_errors=True
        )
        assert results == [2, 4, 6, 8, 10]
    
    def test_batch_skip_errors(self):
        """Test batch processing skips errors."""
        def process_func(x):
            if x == 3:
                raise ValueError("bad value")
            return x * 2
        
        items = [1, 2, 3, 4, 5]
        results = batch_with_error_handling(
            items,
            process_func,
            skip_errors=True
        )
        # Should skip item 3
        assert results == [2, 4, 8, 10]
    
    def test_batch_dont_skip_errors(self):
        """Test batch processing fails on error."""
        def process_func(x):
            if x == 3:
                raise ValueError("bad value")
            return x * 2
        
        items = [1, 2, 3, 4, 5]
        with pytest.raises(ProcessingError):
            batch_with_error_handling(
                items,
                process_func,
                skip_errors=False
            )
    
    def test_batch_error_limit(self):
        """Test batch processing respects error limit."""
        def process_func(x):
            if x in [2, 3, 4]:
                raise ValueError("bad value")
            return x * 2
        
        items = [1, 2, 3, 4, 5]
        with pytest.raises(ProcessingError, match="Error limit"):
            batch_with_error_handling(
                items,
                process_func,
                skip_errors=True,
                error_limit=2  # Should fail when hitting 3rd error
            )


class TestErrorSummary:
    """Tests for error summary generation."""
    
    def test_error_summary_empty(self):
        """Test summary with no errors."""
        summary = get_error_summary([])
        assert "No errors" in summary
    
    def test_error_summary_single_type(self):
        """Test summary with single error type."""
        errors = [ValueError("1"), ValueError("2"), ValueError("3")]
        summary = get_error_summary(errors)
        assert "Total errors: 3" in summary
        assert "ValueError: 3" in summary
    
    def test_error_summary_multiple_types(self):
        """Test summary with multiple error types."""
        errors = [
            ValueError("1"),
            ValueError("2"),
            IOError("3"),  # Note: In Python 3, IOError is an alias for OSError
            TypeError("4")
        ]
        summary = get_error_summary(errors)
        assert "Total errors: 4" in summary
        assert "ValueError: 2" in summary
        # In Python 3, IOError is an alias for OSError, so the type name is "OSError"
        assert "OSError: 1" in summary
        assert "TypeError: 1" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
