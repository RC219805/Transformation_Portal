"""Tests for security utilities."""

from __future__ import annotations

import pytest
from pathlib import Path

from lux_depth_v3.enhance.security import (
    sanitize_file_stem,
    validate_extra_args,
    validate_device_spec,
    validate_quantization_method,
    validate_depth_fallback,
    validate_git_repository,
)


class TestSanitizeFileStem:
    """Tests for file stem sanitization."""

    def test_simple_stem(self):
        """Test sanitization of simple alphanumeric stem."""
        assert sanitize_file_stem("test123") == "test123"

    def test_path_traversal_prevention(self):
        """Test that path traversal attempts are blocked."""
        # Should replace path separators
        assert "/" not in sanitize_file_stem("../../../etc/passwd")
        assert "\\" not in sanitize_file_stem("..\\..\\..\\windows\\system32")

    def test_hidden_file_prevention(self):
        """Test that hidden files are prevented."""
        # Should strip leading dots
        assert not sanitize_file_stem(".hidden").startswith(".")

    def test_special_characters_removed(self):
        """Test that special characters are sanitized."""
        # Should replace special chars with underscore
        result = sanitize_file_stem("test@#$%file")
        assert "@" not in result
        assert "#" not in result
        assert "$" not in result

    def test_double_dots_removed(self):
        """Test that double dots are collapsed."""
        result = sanitize_file_stem("test..file")
        assert ".." not in result

    def test_length_limit(self):
        """Test that overly long stems are truncated."""
        long_stem = "a" * 300
        result = sanitize_file_stem(long_stem, max_length=200)
        assert len(result) == 200

    def test_empty_stem_raises(self):
        """Test that empty stems raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            sanitize_file_stem("")

    def test_only_dots_raises(self):
        """Test that stems with only dots raise ValueError."""
        with pytest.raises(ValueError, match="empty after sanitization"):
            sanitize_file_stem("...")


class TestValidateExtraArgs:
    """Tests for extra args validation."""

    def test_allowed_args(self):
        """Test that allowed args pass validation."""
        # Should not raise
        validate_extra_args(["--verbose"])
        validate_extra_args(["--quiet"])
        validate_extra_args(["--debug"])

    def test_disallowed_args(self):
        """Test that disallowed args raise ValueError."""
        with pytest.raises(ValueError, match="Disallowed"):
            validate_extra_args(["--malicious"])

    def test_injection_attempt(self):
        """Test that injection attempts are blocked."""
        # Argument values are not allowed (strict exact match)
        with pytest.raises(ValueError, match="Disallowed V2 extra argument"):
            validate_extra_args(["--config=/etc/passwd"])

    def test_empty_list(self):
        """Test that empty list is valid."""
        # Should not raise
        validate_extra_args([])

    def test_none(self):
        """Test that None is valid."""
        # Should not raise
        validate_extra_args(None)


class TestValidateDeviceSpec:
    """Tests for device specification validation."""

    def test_allowed_devices(self):
        """Test that allowed devices pass validation."""
        for device in ["auto", "cpu", "cuda", "mps"]:
            assert validate_device_spec(device) == device

    def test_cuda_indexed(self):
        """Test that cuda:N pattern works for all N in 0-9."""
        for i in range(10):
            assert validate_device_spec(f"cuda:{i}") == f"cuda:{i}"

    def test_invalid_device(self):
        """Test that invalid devices raise ValueError."""
        with pytest.raises(ValueError, match="Invalid device"):
            validate_device_spec("tpu")

    def test_cuda_double_digit_fails(self):
        """Test that cuda:NN (double digit) fails."""
        with pytest.raises(ValueError):
            validate_device_spec("cuda:10")


class TestValidateQuantizationMethod:
    """Tests for quantization method validation."""

    def test_allowed_methods(self):
        """Test that allowed methods pass validation."""
        for method in ["p1p99", "p0.5p99.5", "minmax"]:
            assert validate_quantization_method(method) == method

    def test_invalid_method(self):
        """Test that invalid methods raise ValueError."""
        with pytest.raises(ValueError, match="Invalid quantization"):
            validate_quantization_method("unknown")


class TestValidateDepthFallback:
    """Tests for depth fallback validation."""

    def test_allowed_fallbacks(self):
        """Test that allowed fallbacks pass validation."""
        for fallback in ["fail", "skip", "v2-auto"]:
            assert validate_depth_fallback(fallback) == fallback

    def test_invalid_fallback(self):
        """Test that invalid fallbacks raise ValueError."""
        with pytest.raises(ValueError, match="Invalid depth fallback"):
            validate_depth_fallback("unknown")


class TestValidateGitRepository:
    """Tests for git repository validation."""

    def test_non_git_directory(self, tmp_path):
        """Test that non-git directories return None."""
        assert validate_git_repository(tmp_path) is None

    def test_git_directory(self, tmp_path):
        """Test that git directories are validated."""
        # Create fake .git directory
        git_dir = tmp_path / ".git"
        git_dir.mkdir()

        result = validate_git_repository(tmp_path)
        assert result is not None
        assert result.is_absolute()

    def test_symlink_resolution(self, tmp_path):
        """Test that symlinks are resolved."""
        # Create real directory with .git
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        (real_dir / ".git").mkdir()

        # Create symlink
        link = tmp_path / "link"
        link.symlink_to(real_dir)

        result = validate_git_repository(link)
        # Should resolve to real directory
        assert result == real_dir.resolve()

    def test_nonexistent_path(self):
        """Test that nonexistent paths return None."""
        fake_path = Path("/nonexistent/fake/path")
        assert validate_git_repository(fake_path) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
