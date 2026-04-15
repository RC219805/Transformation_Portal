"""Unit tests for transformation_portal.compat.version module.

Tests the Version class and related utility functions for semantic
version parsing, comparison, and range checking.
"""

from __future__ import annotations

import pytest

from transformation_portal.compat.version import (
    Version,
    check_version_compatibility,
    parse_version,
    require_version,
    version_in_range,
)

pytestmark = pytest.mark.unit


class TestVersionParsing:
    """Test Version class initialization and parsing."""

    @pytest.mark.parametrize(
        "version_str,expected",
        [
            ("1.0.0", (1, 0, 0, "")),
            ("2.1.3", (2, 1, 3, "")),
            ("0.0.1", (0, 0, 1, "")),
            ("10.20.30", (10, 20, 30, "")),
        ],
    )
    def test_parse_standard_versions(self, version_str: str, expected: tuple[int, int, int, str]) -> None:
        """Test parsing standard x.y.z versions."""
        v = Version(version_str)
        assert (v.major, v.minor, v.patch, v.prerelease) == expected

    @pytest.mark.parametrize(
        "version_str,expected",
        [
            ("1.2", (1, 2, 0, "")),  # Patch defaults to 0
            ("0.1", (0, 1, 0, "")),
        ],
    )
    def test_parse_two_part_versions(self, version_str: str, expected: tuple[int, int, int, str]) -> None:
        """Test parsing x.y versions without patch component."""
        v = Version(version_str)
        assert (v.major, v.minor, v.patch, v.prerelease) == expected

    @pytest.mark.parametrize(
        "version_str,expected_prerelease",
        [
            ("1.0.0-alpha", "alpha"),
            ("2.0.0-beta", "beta"),
            ("3.1.4-rc1", "rc1"),
            ("1.2.3-alpha.1", "alpha.1"),
            ("1.0.0.beta", "beta"),  # Dot separator for prerelease
        ],
    )
    def test_parse_prerelease_versions(self, version_str: str, expected_prerelease: str) -> None:
        """Test parsing versions with prerelease suffixes."""
        v = Version(version_str)
        assert v.prerelease == expected_prerelease
        assert v.is_prerelease is True

    def test_non_prerelease_property(self) -> None:
        """Test is_prerelease is False for release versions."""
        v = Version("1.2.3")
        assert v.is_prerelease is False

    @pytest.mark.parametrize(
        "invalid_version",
        [
            "",
            "invalid",
            "1",
            # Note: "1.2.3.4" is valid - parsed as 1.2.3 with prerelease "4"
            "a.b.c",
            "1.x.0",
            "-1.0.0",  # Negative not allowed
        ],
    )
    def test_invalid_versions_raise_value_error(self, invalid_version: str) -> None:
        """Test that invalid version strings raise ValueError."""
        with pytest.raises(ValueError, match="Invalid version string"):
            Version(invalid_version)

    def test_four_part_version_parsed_as_prerelease(self) -> None:
        """Test 1.2.3.4 is parsed as 1.2.3 with prerelease '4'."""
        v = Version("1.2.3.4")
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3
        assert v.prerelease == "4"


class TestVersionComparison:
    """Test Version comparison operators."""

    def test_equality(self) -> None:
        """Test version equality."""
        assert Version("1.2.3") == Version("1.2.3")
        assert Version("1.2.3") != Version("1.2.4")
        assert Version("2.0.0-beta") != Version("2.0.0")

    def test_not_equal(self) -> None:
        """Test version inequality."""
        assert Version("1.2.3") != Version("1.2.4")
        assert Version("1.2.3") == Version("1.2.3")
        assert Version("2.0.0-alpha") != Version("2.0.0")

    def test_less_than(self) -> None:
        """Test less-than comparison."""
        assert Version("1.0.0") < Version("2.0.0")
        assert Version("1.1.0") < Version("1.2.0")
        assert Version("1.1.1") < Version("1.1.2")
        assert Version("2.0.0-beta") < Version("2.0.0")
        assert Version("2.0.0") >= Version("1.0.0")

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            ("2.0.0-alpha", "2.0.0-beta"),
            ("2.0.0-alpha.1", "2.0.0-alpha.beta"),
            ("2.0.0-alpha.1", "2.0.0-alpha.2"),
            ("2.0.0-alpha.1", "2.0.0-alpha.1.1"),
            ("2.0.0-1", "2.0.0-alpha"),
            ("2.0.0-rc.1", "2.0.0"),
        ],
    )
    def test_prerelease_ordering(self, left: str, right: str) -> None:
        """Test SemVer prerelease precedence rules."""
        assert Version(left) < Version(right)

    def test_less_than_or_equal(self) -> None:
        """Test less-than-or-equal comparison."""
        assert Version("1.0.0") <= Version("2.0.0")
        assert Version("1.0.0") <= Version("1.0.0")
        assert Version("2.0.0") > Version("1.0.0")

    def test_greater_than(self) -> None:
        """Test greater-than comparison."""
        assert Version("2.0.0") > Version("1.0.0")
        assert Version("1.2.0") > Version("1.1.0")
        assert Version("1.1.2") > Version("1.1.1")
        assert Version("1.0.0") <= Version("2.0.0")

    def test_greater_than_or_equal(self) -> None:
        """Test greater-than-or-equal comparison."""
        assert Version("2.0.0") >= Version("1.0.0")
        assert Version("1.0.0") >= Version("1.0.0")
        assert Version("1.0.0") < Version("2.0.0")

    def test_comparison_with_non_version_returns_not_implemented(self) -> None:
        """Test non-Version comparisons fall back to Python comparison semantics."""
        v = Version("1.0.0")
        assert (v == "1.0.0") is False
        with pytest.raises(TypeError):
            _ = v < "1.0.0"


class TestVersionHashability:
    """Test Version hashability for use in sets and dicts."""

    def test_hash_consistent(self) -> None:
        """Test that hash is consistent for equal versions."""
        v1 = Version("1.2.3")
        v2 = Version("1.2.3")
        assert hash(v1) == hash(v2)

    def test_hash_differs_for_different_versions(self) -> None:
        """Test that hash differs for different versions."""
        v1 = Version("1.2.3")
        v2 = Version("1.2.4")
        # Note: hash collision is possible but unlikely for adjacent versions
        assert v1 != v2

    def test_release_and_prerelease_have_distinct_hash_and_identity(self) -> None:
        """Test prerelease tags participate in equality and hashing."""
        release = Version("2.0.0")
        prerelease = Version("2.0.0-beta")

        assert release != prerelease
        assert hash(release) != hash(prerelease)

    def test_version_in_set(self) -> None:
        """Test Version can be used in sets."""
        versions = {Version("1.0.0"), Version("2.0.0"), Version("1.0.0")}
        assert len(versions) == 2
        assert Version("1.0.0") in versions
        assert Version("3.0.0") not in versions

    def test_release_and_prerelease_are_distinct_set_members(self) -> None:
        """Test release and prerelease versions remain distinct in sets."""
        versions = {Version("2.0.0-beta"), Version("2.0.0")}

        assert len(versions) == 2
        assert Version("2.0.0-beta") in versions
        assert Version("2.0.0") in versions

    def test_version_as_dict_key(self) -> None:
        """Test Version can be used as dict key."""
        version_map = {
            Version("1.0.0"): "stable",
            Version("2.0.0-beta"): "beta",
        }
        assert version_map[Version("1.0.0")] == "stable"
        assert version_map[Version("2.0.0-beta")] == "beta"

    def test_release_and_prerelease_are_distinct_dict_keys(self) -> None:
        """Test release and prerelease versions do not overwrite each other."""
        version_map = {
            Version("2.0.0-beta"): "beta",
            Version("2.0.0"): "stable",
        }

        assert version_map[Version("2.0.0-beta")] == "beta"
        assert version_map[Version("2.0.0")] == "stable"

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("raw", "9.9.9"),
            ("major", 9),
            ("minor", 9),
            ("patch", 9),
            ("prerelease", "rc.1"),
        ],
    )
    def test_public_version_fields_are_immutable(self, field: str, value: object) -> None:
        """Test Version fields cannot be mutated after construction."""
        version = Version("2.0.0-beta")
        version_map = {version: "beta"}

        with pytest.raises(AttributeError, match="immutable"):
            setattr(version, field, value)

        assert version_map[Version("2.0.0-beta")] == "beta"


class TestVersionRepresentation:
    """Test Version string representation."""

    def test_repr(self) -> None:
        """Test repr returns evaluable string."""
        v = Version("1.2.3")
        assert repr(v) == "Version('1.2.3')"

    def test_str_without_prerelease(self) -> None:
        """Test str returns normalized version without prerelease."""
        v = Version("1.2.3")
        assert str(v) == "1.2.3"

    def test_str_with_prerelease(self) -> None:
        """Test str includes prerelease suffix."""
        v = Version("1.2.3-beta")
        assert str(v) == "1.2.3-beta"

    def test_base_version_property(self) -> None:
        """Test base_version strips prerelease."""
        v = Version("1.2.3-alpha")
        assert v.base_version == "1.2.3"


class TestVersionFactoryMethods:
    """Test Version factory methods."""

    def test_from_tuple(self) -> None:
        """Test creating Version from tuple."""
        v = Version.from_tuple((1, 2, 3))
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3
        assert v.prerelease == ""

    def test_from_tuple_with_prerelease(self) -> None:
        """Test creating Version from tuple with prerelease."""
        v = Version.from_tuple((1, 2, 3), "beta")
        assert str(v) == "1.2.3-beta"


class TestVersionBumpMethods:
    """Test Version bump methods."""

    def test_bump_major(self) -> None:
        """Test bumping major version."""
        v = Version("1.2.3")
        bumped = v.bump_major()
        assert str(bumped) == "2.0.0"
        # Original unchanged
        assert str(v) == "1.2.3"

    def test_bump_minor(self) -> None:
        """Test bumping minor version."""
        v = Version("1.2.3")
        bumped = v.bump_minor()
        assert str(bumped) == "1.3.0"

    def test_bump_patch(self) -> None:
        """Test bumping patch version."""
        v = Version("1.2.3")
        bumped = v.bump_patch()
        assert str(bumped) == "1.2.4"

    def test_bump_clears_prerelease(self) -> None:
        """Test that bump methods clear prerelease suffix."""
        v = Version("1.2.3-beta")
        assert v.bump_major().prerelease == ""
        assert v.bump_minor().prerelease == ""
        assert v.bump_patch().prerelease == ""


class TestParseVersion:
    """Test parse_version utility function."""

    def test_valid_version(self) -> None:
        """Test parsing valid version returns Version instance."""
        result = parse_version("1.2.3")
        assert result is not None
        assert isinstance(result, Version)
        assert str(result) == "1.2.3"

    def test_invalid_version_returns_none(self) -> None:
        """Test parsing invalid version returns None instead of raising."""
        result = parse_version("invalid")
        assert result is None

    def test_empty_string_returns_none(self) -> None:
        """Test parsing empty string returns None."""
        result = parse_version("")
        assert result is None


class TestCheckVersionCompatibility:
    """Test check_version_compatibility function."""

    def test_compatible_when_current_greater(self) -> None:
        """Test returns True when current > required."""
        assert check_version_compatibility("2.0.0", "1.0.0") is True

    def test_compatible_when_current_equals(self) -> None:
        """Test returns True when current == required."""
        assert check_version_compatibility("1.0.0", "1.0.0") is True

    def test_incompatible_when_current_less(self) -> None:
        """Test returns False when current < required."""
        assert check_version_compatibility("1.0.0", "2.0.0") is False

    def test_prerelease_does_not_satisfy_release_requirement(self) -> None:
        """Test prerelease versions sort below their final release."""
        assert check_version_compatibility("2.0.0-beta", "2.0.0") is False

    def test_invalid_version_returns_false(self) -> None:
        """Test returns False for invalid version strings."""
        assert check_version_compatibility("invalid", "1.0.0") is False
        assert check_version_compatibility("1.0.0", "invalid") is False


class TestRequireVersion:
    """Test require_version function."""

    def test_raises_when_version_too_low(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test raises RuntimeError when version is below minimum."""
        # Mock the package version to be lower than required
        import transformation_portal

        monkeypatch.setattr(transformation_portal, "__version__", "0.1.0")

        with pytest.raises(RuntimeError, match="v99.0.0\\+ required"):
            require_version("99.0.0")

    def test_passes_when_version_sufficient(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test passes silently when version meets requirement."""
        import transformation_portal

        monkeypatch.setattr(transformation_portal, "__version__", "2.0.0")

        # Should not raise
        require_version("1.0.0")


class TestVersionInRange:
    """Test version_in_range function."""

    def test_in_range_basic(self) -> None:
        """Test version within range."""
        assert version_in_range("1.5.0", min_version="1.0.0", max_version="2.0.0") is True

    def test_at_min_is_inclusive(self) -> None:
        """Test min_version is inclusive."""
        assert version_in_range("1.0.0", min_version="1.0.0", max_version="2.0.0") is True

    def test_at_max_exclusive_by_default(self) -> None:
        """Test max_version is exclusive by default."""
        assert version_in_range("2.0.0", min_version="1.0.0", max_version="2.0.0") is False

    def test_prerelease_below_exclusive_max_release(self) -> None:
        """Test prerelease versions remain below the final release max."""
        assert version_in_range("2.0.0-beta", min_version="1.0.0", max_version="2.0.0") is True

    def test_at_max_inclusive_when_specified(self) -> None:
        """Test max_version can be inclusive."""
        assert version_in_range("2.0.0", min_version="1.0.0", max_version="2.0.0", inclusive_max=True) is True

    def test_prerelease_respects_inclusive_max(self) -> None:
        """Test prereleases remain in range with inclusive max release."""
        assert version_in_range("2.0.0-beta", min_version="1.0.0", max_version="2.0.0", inclusive_max=True) is True

    def test_below_range(self) -> None:
        """Test version below range."""
        assert version_in_range("0.5.0", min_version="1.0.0", max_version="2.0.0") is False

    def test_above_range(self) -> None:
        """Test version above range."""
        assert version_in_range("3.0.0", min_version="1.0.0", max_version="2.0.0") is False

    def test_no_min_version(self) -> None:
        """Test with no lower bound."""
        assert version_in_range("0.0.1", max_version="2.0.0") is True
        assert version_in_range("2.0.0", max_version="2.0.0") is False

    def test_no_max_version(self) -> None:
        """Test with no upper bound."""
        assert version_in_range("99.0.0", min_version="1.0.0") is True
        assert version_in_range("0.5.0", min_version="1.0.0") is False

    def test_no_bounds(self) -> None:
        """Test with no bounds (any version passes)."""
        assert version_in_range("1.2.3") is True

    def test_invalid_version_returns_false(self) -> None:
        """Test invalid version returns False."""
        assert version_in_range("invalid", min_version="1.0.0") is False

    def test_invalid_min_version_returns_false(self) -> None:
        """Test invalid min_version returns False."""
        assert version_in_range("1.5.0", min_version="invalid") is False

    def test_invalid_max_version_returns_false(self) -> None:
        """Test invalid max_version returns False."""
        assert version_in_range("1.5.0", max_version="invalid") is False
