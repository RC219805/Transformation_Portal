"""Tests for lux_depth_v3 security module.

Tests security validation functions including:
- validate_preset_name: Preset name validation
- sanitize_file_stem: File stem sanitization
- Other security validation functions
"""

import pytest

from transformation_portal.lux_depth_v3.security import (
    sanitize_file_stem,
    sanitize_path_component_nonlossy,
    validate_depth_fallback,
    validate_device_spec,
    validate_preset_name,
    validate_quantization_method,
)


class TestValidatePresetName:
    """Tests for validate_preset_name function."""

    def test_valid_simple_name(self):
        """Test valid simple preset names are accepted."""
        assert validate_preset_name("default") == "default"
        assert validate_preset_name("custom") == "custom"
        assert validate_preset_name("my_preset") == "my_preset"

    def test_valid_with_hyphen(self):
        """Test valid preset names with hyphens."""
        assert validate_preset_name("my-preset") == "my-preset"
        assert validate_preset_name("apex-v2") == "apex-v2"

    def test_valid_with_dot(self):
        """Test valid preset names with dots."""
        assert validate_preset_name("preset.v2") == "preset.v2"
        assert validate_preset_name("config.1.0") == "config.1.0"

    def test_valid_with_numbers(self):
        """Test valid preset names with numbers."""
        assert validate_preset_name("preset123") == "preset123"
        assert validate_preset_name("v2_config") == "v2_config"

    def test_rejects_empty(self):
        """Test that empty preset name raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_preset_name("")

    def test_rejects_path_traversal_dotdot(self):
        """Test that path traversal with .. is rejected."""
        with pytest.raises(ValueError, match="invalid characters"):
            validate_preset_name("../etc/passwd")
        with pytest.raises(ValueError, match="invalid characters"):
            validate_preset_name("..preset")
        with pytest.raises(ValueError, match="invalid characters"):
            validate_preset_name("preset..")

    def test_rejects_forward_slash(self):
        """Test that forward slashes are rejected."""
        with pytest.raises(ValueError, match="invalid characters"):
            validate_preset_name("preset/config")
        with pytest.raises(ValueError, match="invalid characters"):
            validate_preset_name("/root/config")

    def test_rejects_backslash(self):
        """Test that backslashes are rejected."""
        with pytest.raises(ValueError, match="invalid characters"):
            validate_preset_name("preset\\config")
        with pytest.raises(ValueError, match="invalid characters"):
            validate_preset_name("\\\\unc\\path")

    def test_rejects_control_characters(self):
        """Test that control characters are rejected."""
        with pytest.raises(ValueError, match="control characters"):
            validate_preset_name("preset\x00name")
        with pytest.raises(ValueError, match="control characters"):
            validate_preset_name("preset\n")
        with pytest.raises(ValueError, match="control characters"):
            validate_preset_name("\tpreset")

    def test_rejects_special_characters(self):
        """Test that special characters are rejected."""
        with pytest.raises(ValueError, match="invalid characters"):
            validate_preset_name("preset@name")
        with pytest.raises(ValueError, match="invalid characters"):
            validate_preset_name("preset!name")
        with pytest.raises(ValueError, match="invalid characters"):
            validate_preset_name("preset#name")
        with pytest.raises(ValueError, match="invalid characters"):
            validate_preset_name("preset name")  # space

    def test_rejects_log_injection_newlines(self):
        """Test that newline characters (log injection) are rejected."""
        # Newlines could allow log injection attacks
        with pytest.raises(ValueError, match="control characters"):
            validate_preset_name("preset\nmalicious")
        with pytest.raises(ValueError, match="control characters"):
            validate_preset_name("preset\r\nmalicious")


class TestSanitizeFileStem:
    """Tests for sanitize_file_stem function."""

    def test_valid_stem_unchanged(self):
        """Test that valid stems are returned unchanged."""
        assert sanitize_file_stem("my_image") == "my_image"
        assert sanitize_file_stem("test123") == "test123"

    def test_empty_returns_unnamed(self):
        """Test that empty stem returns 'unnamed'."""
        assert sanitize_file_stem("") == "unnamed"

    def test_path_traversal_sanitized(self):
        """Test that path traversal sequences are sanitized."""
        result = sanitize_file_stem("../../../etc/passwd")
        assert ".." not in result
        assert "/" not in result
        # Should contain sanitized parts
        assert "etc" in result or "passwd" in result

    def test_forbidden_characters_replaced(self):
        """Test that forbidden characters are replaced."""
        result = sanitize_file_stem("file<>with|bad:chars")
        assert "<" not in result
        assert ">" not in result
        assert "|" not in result
        assert ":" not in result

    def test_max_length_enforced(self):
        """Test that max_length parameter is enforced."""
        long_stem = "a" * 500
        result = sanitize_file_stem(long_stem, max_length=100)
        assert len(result) <= 100

    def test_leading_dot_sanitized(self):
        """Test that leading dots are sanitized."""
        result = sanitize_file_stem(".hidden")
        assert not result.startswith(".")

    def test_leading_dash_sanitized(self):
        """Test that leading dashes are sanitized."""
        result = sanitize_file_stem("-flag")
        assert not result.startswith("-")


class TestSanitizePathComponentNonlossy:
    """Tests for sanitize_path_component_nonlossy function."""

    def test_valid_path_unchanged(self):
        """Test that valid paths are returned unchanged."""
        assert sanitize_path_component_nonlossy("valid_path") == "valid_path"

    def test_preserves_nested_structure(self):
        """Test that nested paths are flattened with delimiter."""
        result = sanitize_path_component_nonlossy("dir1/dir2/file")
        assert "__" in result
        assert "dir1" in result
        assert "dir2" in result
        assert "file" in result

    def test_path_traversal_removed(self):
        """Test that path traversal sequences are removed."""
        result = sanitize_path_component_nonlossy("../../../etc/passwd")
        assert ".." not in result
        assert "etc" in result
        assert "passwd" in result

    def test_backslash_normalized(self):
        """Test that backslashes are normalized."""
        result = sanitize_path_component_nonlossy("dir\\with\\backslash")
        assert "\\" not in result
        assert "dir" in result

    def test_empty_returns_unnamed(self):
        """Test that empty input returns 'unnamed'."""
        assert sanitize_path_component_nonlossy("") == "unnamed"


class TestValidateDeviceSpec:
    """Tests for validate_device_spec function."""

    def test_cpu_accepted(self):
        """Test that CPU device spec is accepted."""
        assert validate_device_spec("cpu") == "cpu"
        assert validate_device_spec("CPU") == "cpu"

    def test_cuda_accepted(self):
        """Test that CUDA device specs are accepted."""
        assert validate_device_spec("cuda") == "cuda"
        assert validate_device_spec("cuda:0") == "cuda:0"
        assert validate_device_spec("cuda:1") == "cuda:1"

    def test_mps_accepted(self):
        """Test that MPS device spec is accepted."""
        assert validate_device_spec("mps") == "mps"

    def test_auto_accepted(self):
        """Test that auto device spec is accepted."""
        assert validate_device_spec("auto") == "auto"

    def test_rejects_invalid(self):
        """Test that invalid device specs are rejected."""
        with pytest.raises(ValueError, match="Invalid device"):
            validate_device_spec("invalid")
        with pytest.raises(ValueError, match="Invalid device"):
            validate_device_spec("gpu")

    def test_rejects_empty(self):
        """Test that empty device spec raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_device_spec("")


class TestValidateQuantizationMethod:
    """Tests for validate_quantization_method function."""

    def test_valid_methods_accepted(self):
        """Test that valid quantization methods are accepted."""
        assert validate_quantization_method("none") == "none"
        assert validate_quantization_method("int8") == "int8"
        assert validate_quantization_method("fp16") == "fp16"
        assert validate_quantization_method("fp32") == "fp32"
        assert validate_quantization_method("auto") == "auto"

    def test_case_insensitive(self):
        """Test that method validation is case insensitive."""
        assert validate_quantization_method("FP16") == "fp16"
        assert validate_quantization_method("NONE") == "none"

    def test_rejects_invalid(self):
        """Test that invalid methods are rejected."""
        with pytest.raises(ValueError, match="Invalid quantization"):
            validate_quantization_method("invalid")

    def test_rejects_empty(self):
        """Test that empty method raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_quantization_method("")


class TestValidateDepthFallback:
    """Tests for validate_depth_fallback function."""

    def test_none_accepted(self):
        """Test that None fallback is accepted."""
        assert validate_depth_fallback(None) is None

    def test_valid_fallbacks_accepted(self):
        """Test that valid fallback strategies are accepted."""
        assert validate_depth_fallback("fail") == "fail"
        assert validate_depth_fallback("skip") == "skip"
        assert validate_depth_fallback("v2-auto") == "v2-auto"

    def test_case_insensitive(self):
        """Test that fallback validation is case insensitive."""
        assert validate_depth_fallback("FAIL") == "fail"
        assert validate_depth_fallback("Skip") == "skip"

    def test_rejects_invalid(self):
        """Test that invalid fallback strategies are rejected."""
        with pytest.raises(ValueError, match="Invalid depth fallback"):
            validate_depth_fallback("invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
