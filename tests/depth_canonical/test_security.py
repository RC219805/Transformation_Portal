"""Tests for security validation utilities."""

import tempfile
from pathlib import Path

import pytest

from transformation_portal.depth_canonical.security import validate_image_extension, validate_path


def test_validate_path_accepts_safe_path():
    """Test validate_path accepts paths within base directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir).resolve()
        safe_path = base_dir / "subdir" / "file.txt"

        result = validate_path(safe_path, base_dir)

        assert result.is_absolute()
        # Should be under base_dir (both resolved to handle symlinks)
        result.relative_to(base_dir)  # Will raise if not under base_dir


def test_validate_path_rejects_traversal_attempt():
    """Test validate_path rejects path traversal attempts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir) / "restricted"
        base_dir.mkdir()

        # Try to escape base_dir
        evil_path = base_dir / ".." / ".." / "etc" / "passwd"

        with pytest.raises(ValueError, match="Path traversal detected"):
            validate_path(evil_path, base_dir)


def test_validate_path_rejects_absolute_path_outside_base():
    """Test validate_path rejects absolute paths outside base directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir) / "base"
        base_dir.mkdir()

        outside_path = Path("/tmp/outside.txt")

        with pytest.raises(ValueError, match="Path traversal detected"):
            validate_path(outside_path, base_dir)


def test_validate_image_extension_accepts_valid_extensions():
    """Test validate_image_extension accepts allowed extensions."""
    allowed = (".jpg", ".jpeg", ".png", ".tiff", ".tif")

    # All valid extensions should pass
    validate_image_extension(Path("image.jpg"), allowed)
    validate_image_extension(Path("image.jpeg"), allowed)
    validate_image_extension(Path("image.png"), allowed)
    validate_image_extension(Path("image.tiff"), allowed)
    validate_image_extension(Path("image.tif"), allowed)

    # Case insensitive
    validate_image_extension(Path("image.JPG"), allowed)
    validate_image_extension(Path("image.PNG"), allowed)


def test_validate_image_extension_rejects_invalid_extensions():
    """Test validate_image_extension rejects disallowed extensions."""
    allowed = (".jpg", ".jpeg", ".png")

    with pytest.raises(ValueError, match="Invalid file extension"):
        validate_image_extension(Path("image.gif"), allowed)

    with pytest.raises(ValueError, match="Invalid file extension"):
        validate_image_extension(Path("image.bmp"), allowed)

    with pytest.raises(ValueError, match="Invalid file extension"):
        validate_image_extension(Path("script.py"), allowed)


def test_validate_image_extension_case_insensitive():
    """Test validate_image_extension is case insensitive."""
    allowed = (".jpg", ".png")

    # Uppercase should work
    validate_image_extension(Path("IMAGE.JPG"), allowed)
    validate_image_extension(Path("IMAGE.PNG"), allowed)

    # Mixed case should work
    validate_image_extension(Path("image.JpG"), allowed)
    validate_image_extension(Path("image.PnG"), allowed)
