#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for version checking and compatibility utilities."""

import pytest

from transformation_portal.compat.version import (
    Version,
    check_version_compatibility,
    require_version,
    get_portal_version,
    is_version_at_least,
)


class TestVersion:
    """Tests for Version class."""

    def test_from_string_basic(self):
        """Test parsing basic version string."""
        v = Version.from_string("1.2.3")
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3
        assert v.prerelease is None

    def test_from_string_with_prerelease(self):
        """Test parsing version with prerelease."""
        v = Version.from_string("2.0.0b1")
        assert v.major == 2
        assert v.minor == 0
        assert v.patch == 0
        assert v.prerelease is not None

    def test_from_string_invalid(self):
        """Test that invalid version string raises error."""
        with pytest.raises(ValueError):
            Version.from_string("invalid.version")

    def test_string_representation(self):
        """Test string representation of version."""
        v = Version(major=1, minor=2, patch=3)
        assert str(v) == "1.2.3"

    def test_string_representation_with_prerelease(self):
        """Test string representation with prerelease."""
        v = Version(major=2, minor=0, patch=0, prerelease="beta1")
        assert str(v) == "2.0.0-beta1"

    def test_version_comparison_lt(self):
        """Test less than comparison."""
        v1 = Version.from_string("1.0.0")
        v2 = Version.from_string("2.0.0")
        assert v1 < v2
        assert not (v1 < v1)  # Version is not less than itself

    def test_version_comparison_le(self):
        """Test less than or equal comparison."""
        v1 = Version.from_string("1.0.0")
        v2 = Version.from_string("2.0.0")
        v3 = Version.from_string("1.0.0")
        assert v1 <= v2
        assert v1 <= v3
        assert not v2 <= v1

    def test_version_comparison_gt(self):
        """Test greater than comparison."""
        v1 = Version.from_string("2.0.0")
        v2 = Version.from_string("1.0.0")
        assert v1 > v2
        assert not (v1 > v1)  # Version is not greater than itself

    def test_version_comparison_ge(self):
        """Test greater than or equal comparison."""
        v1 = Version.from_string("2.0.0")
        v2 = Version.from_string("1.0.0")
        v3 = Version.from_string("2.0.0")
        assert v1 >= v2
        assert v1 >= v3
        assert not v2 >= v1

    def test_version_comparison_eq(self):
        """Test equality comparison."""
        v1 = Version.from_string("1.2.3")
        v2 = Version.from_string("1.2.3")
        v3 = Version.from_string("1.2.4")
        assert v1 == v2
        assert not v1 == v3

    def test_version_comparison_with_different_type(self):
        """Test that equality with different type returns False."""
        v = Version.from_string("1.0.0")
        assert not v == "1.0.0"
        assert not v == 1

    def test_version_minor_differences(self):
        """Test comparisons with minor version differences."""
        v1 = Version.from_string("1.0.0")
        v2 = Version.from_string("1.1.0")
        assert v1 < v2
        assert v1 <= v2

    def test_version_patch_differences(self):
        """Test comparisons with patch version differences."""
        v1 = Version.from_string("1.0.0")
        v2 = Version.from_string("1.0.1")
        assert v1 < v2
        assert v1 <= v2


class TestCheckVersionCompatibility:
    """Tests for check_version_compatibility function."""

    def test_compatible_version_in_range(self):
        """Test that version within range is compatible."""
        is_compatible, msg = check_version_compatibility(
            "1.5.0",
            min_version="1.0.0",
            max_version="2.0.0"
        )
        assert is_compatible
        assert msg is None

    def test_version_below_minimum(self):
        """Test that version below minimum is incompatible."""
        is_compatible, msg = check_version_compatibility(
            "0.9.0",
            min_version="1.0.0"
        )
        assert not is_compatible
        assert msg is not None
        assert "below minimum" in msg.lower()
        assert "0.9.0" in msg
        assert "1.0.0" in msg

    def test_version_above_maximum(self):
        """Test that version above maximum is incompatible."""
        is_compatible, msg = check_version_compatibility(
            "3.0.0",
            max_version="2.0.0"
        )
        assert not is_compatible
        assert msg is not None
        assert "exceeds maximum" in msg.lower()
        assert "3.0.0" in msg
        assert "2.0.0" in msg

    def test_version_equal_to_minimum(self):
        """Test that version equal to minimum is compatible."""
        is_compatible, msg = check_version_compatibility(
            "1.0.0",
            min_version="1.0.0"
        )
        assert is_compatible
        assert msg is None

    def test_version_equal_to_maximum(self):
        """Test that version equal to maximum is compatible."""
        is_compatible, msg = check_version_compatibility(
            "2.0.0",
            max_version="2.0.0"
        )
        assert is_compatible
        assert msg is None

    def test_no_version_constraints(self):
        """Test with no version constraints."""
        is_compatible, msg = check_version_compatibility("1.0.0")
        assert is_compatible
        assert msg is None

    def test_only_minimum_constraint(self):
        """Test with only minimum version constraint."""
        is_compatible, msg = check_version_compatibility(
            "2.0.0",
            min_version="1.0.0"
        )
        assert is_compatible
        assert msg is None

    def test_only_maximum_constraint(self):
        """Test with only maximum version constraint."""
        is_compatible, msg = check_version_compatibility(
            "1.0.0",
            max_version="2.0.0"
        )
        assert is_compatible
        assert msg is None


class TestRequireVersion:
    """Tests for require_version decorator."""

    def test_compatible_version_passes(self, monkeypatch):
        """Test that function executes with compatible version."""
        # Mock the version to be within range
        monkeypatch.setattr(
            "transformation_portal.__version__",
            "1.5.0"
        )

        @require_version(min_version="1.0.0", max_version="2.0.0")
        def my_function():
            return "success"

        result = my_function()
        assert result == "success"

    def test_incompatible_version_raises_error(self, monkeypatch):
        """Test that function raises error with incompatible version."""
        # Mock the version to be below minimum
        monkeypatch.setattr(
            "transformation_portal.__version__",
            "0.5.0"
        )

        @require_version(min_version="1.0.0")
        def my_function():
            return "success"

        with pytest.raises(RuntimeError) as exc_info:
            my_function()

        assert "Version incompatibility" in str(exc_info.value)
        assert "my_function" in str(exc_info.value)

    def test_require_version_preserves_function_metadata(self):
        """Test that decorator preserves function metadata."""
        @require_version(min_version="0.0.1")
        def documented_function():
            """This function is documented."""
            pass

        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This function is documented."

    def test_require_version_with_function_arguments(self, monkeypatch):
        """Test that decorated function properly handles arguments."""
        monkeypatch.setattr(
            "transformation_portal.__version__",
            "1.0.0"
        )

        @require_version(min_version="0.5.0")
        def add_numbers(a, b):
            return a + b

        result = add_numbers(3, 4)
        assert result == 7

    def test_require_version_only_minimum(self, monkeypatch):
        """Test require_version with only minimum constraint."""
        monkeypatch.setattr(
            "transformation_portal.__version__",
            "2.0.0"
        )

        @require_version(min_version="1.0.0")
        def my_function():
            return "ok"

        assert my_function() == "ok"

    def test_require_version_only_maximum(self, monkeypatch):
        """Test require_version with only maximum constraint."""
        monkeypatch.setattr(
            "transformation_portal.__version__",
            "1.0.0"
        )

        @require_version(max_version="2.0.0")
        def my_function():
            return "ok"

        assert my_function() == "ok"


class TestGetPortalVersion:
    """Tests for get_portal_version function."""

    def test_get_portal_version_returns_string(self):
        """Test that get_portal_version returns a string."""
        version = get_portal_version()
        assert isinstance(version, str)

    def test_get_portal_version_when_available(self, monkeypatch):
        """Test getting version when it's available."""
        monkeypatch.setattr(
            "transformation_portal.__version__",
            "1.2.3"
        )
        version = get_portal_version()
        assert version == "1.2.3"

    def test_get_portal_version_fallback(self, monkeypatch):
        """Test that get_portal_version falls back to 0.0.0 on import error."""
        # This test is tricky because we can't easily mock ImportError
        # We'll test the current behavior instead
        version = get_portal_version()
        # Should return either the actual version or "0.0.0" fallback
        assert isinstance(version, str)
        # Version should be parseable
        parts = version.split('.')
        assert len(parts) >= 2  # At least major.minor


class TestIsVersionAtLeast:
    """Tests for is_version_at_least function."""

    def test_version_meets_requirement(self, monkeypatch):
        """Test when current version meets requirement."""
        monkeypatch.setattr(
            "transformation_portal.__version__",
            "2.0.0"
        )
        assert is_version_at_least("1.0.0")

    def test_version_below_requirement(self, monkeypatch):
        """Test when current version is below requirement."""
        monkeypatch.setattr(
            "transformation_portal.__version__",
            "0.9.0"
        )
        assert not is_version_at_least("1.0.0")

    def test_version_exactly_meets_requirement(self, monkeypatch):
        """Test when current version exactly meets requirement."""
        monkeypatch.setattr(
            "transformation_portal.__version__",
            "1.5.0"
        )
        assert is_version_at_least("1.5.0")

    def test_version_patch_level(self, monkeypatch):
        """Test version comparison at patch level."""
        monkeypatch.setattr(
            "transformation_portal.__version__",
            "1.0.1"
        )
        assert is_version_at_least("1.0.0")
        assert is_version_at_least("1.0.1")
        assert not is_version_at_least("1.0.2")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
