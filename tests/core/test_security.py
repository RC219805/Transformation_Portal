"""Tests for core security module."""

import pytest
from pathlib import Path
import tempfile

from transformation_portal.core.security import (
    InputValidator,
    ValidationError,
    PathValidator,
    safe_resolve_path,
    is_safe_path,
    sanitize_filename,
    validate_input_file,
)


def test_input_validator_missing_file():
    """Test validation of missing file."""
    validator = InputValidator()
    
    result = validator.validate_file(Path("/nonexistent/file.jpg"), strict=False)
    assert not result.valid
    assert len(result.errors) > 0


def test_input_validator_invalid_extension():
    """Test validation of invalid extension."""
    validator = InputValidator(allowed_extensions=(".jpg", ".png"))
    
    with tempfile.NamedTemporaryFile(suffix=".txt") as f:
        result = validator.validate_file(Path(f.name), strict=False)
        assert not result.valid


def test_input_validator_size_limit():
    """Test file size limit validation."""
    validator = InputValidator(max_size_mb=0.001)  # 1KB limit
    
    with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
        # Write 10KB
        f.write(b"x" * 10240)
        f.flush()
        
        result = validator.validate_file(Path(f.name), strict=False)
        assert not result.valid


def test_input_validator_valid_file():
    """Test validation of valid file."""
    validator = InputValidator()
    
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        # Write JPEG header
        f.write(b"\xFF\xD8\xFF")
        f.write(b"x" * 100)
        f.flush()
        temp_path = Path(f.name)
    
    try:
        result = validator.validate_file(temp_path, strict=False)
        assert result.valid
        assert result.file_type == "jpeg"
    finally:
        temp_path.unlink()


def test_path_validator():
    """Test path validator."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        validator = PathValidator(allowed_roots=[root])
        
        # Valid path
        valid_path = root / "subdir" / "file.txt"
        assert validator.validate(valid_path, must_exist=False)
        
        # Path outside root
        outside_path = Path("/tmp/outside.txt")
        assert not validator.validate(outside_path, must_exist=False)


def test_safe_resolve_path():
    """Test safe path resolution."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir).resolve()  # Resolve root to handle symlinks (e.g., /var -> /private/var on macOS)
        
        # Valid path
        safe_path = root / "subdir" / "file.txt"
        resolved = safe_resolve_path(safe_path, root)
        assert resolved.is_relative_to(root)
        
        # Attempt traversal (should fail)
        with pytest.raises(ValueError):
            traversal_path = root / ".." / "outside.txt"
            safe_resolve_path(traversal_path, root)


def test_is_safe_path():
    """Test safe path checking."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        
        safe = root / "file.txt"
        assert is_safe_path(safe, allowed_roots=[root])
        
        unsafe = Path("/tmp/outside.txt")
        assert not is_safe_path(unsafe, allowed_roots=[root])


def test_sanitize_filename():
    """Test filename sanitization."""
    # Dangerous patterns
    assert ".." not in sanitize_filename("../etc/passwd")
    assert "/" not in sanitize_filename("/etc/passwd")
    assert "~" not in sanitize_filename("~/file.txt")
    
    # Special characters
    result = sanitize_filename("file name!@#$.txt")
    assert " " not in result or "_" in result
    
    # Long filename
    long_name = "a" * 300 + ".txt"
    result = sanitize_filename(long_name)
    assert len(result) <= 255


def test_validate_input_file():
    """Test input file validation convenience function."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(b"\xFF\xD8\xFF")
        f.write(b"x" * 100)
        f.flush()
        temp_path = Path(f.name)
    
    try:
        assert validate_input_file(temp_path, strict=True)
    finally:
        temp_path.unlink()


def test_validate_input_file_strict():
    """Test strict validation."""
    with pytest.raises(ValidationError):
        validate_input_file(Path("/nonexistent.jpg"), strict=True)
