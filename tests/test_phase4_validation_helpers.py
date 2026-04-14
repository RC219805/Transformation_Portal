"""Unit tests for Phase 4 validation helpers module."""

from __future__ import annotations

import pytest

from tp.phase4.validation_helpers import (
    SHA256_HEX_RE,
    build_path_index,
    ensure_sha256_hex,
    is_valid_sha256_hex,
    require_contract_version,
    require_sorted_relative_paths,
    require_unique_relative_paths,
    string_or_none,
)


class TestSHA256HexValidation:
    """Tests for SHA256 hex validation functions."""

    def test_valid_sha256_hex(self) -> None:
        """Valid lowercase 64-char hex should pass."""
        valid_hash = "a" * 64
        assert is_valid_sha256_hex(valid_hash) is True

    def test_valid_sha256_hex_mixed(self) -> None:
        """Valid hex with digits should pass."""
        valid_hash = "abcdef0123456789" * 4
        assert is_valid_sha256_hex(valid_hash) is True

    def test_invalid_sha256_uppercase(self) -> None:
        """Uppercase hex should fail."""
        invalid_hash = "A" * 64
        assert is_valid_sha256_hex(invalid_hash) is False

    def test_invalid_sha256_wrong_length(self) -> None:
        """Wrong length should fail."""
        assert is_valid_sha256_hex("a" * 63) is False
        assert is_valid_sha256_hex("a" * 65) is False

    def test_invalid_sha256_non_hex(self) -> None:
        """Non-hex characters should fail."""
        invalid_hash = "g" * 64
        assert is_valid_sha256_hex(invalid_hash) is False

    def test_invalid_sha256_not_string(self) -> None:
        """Non-string types should fail."""
        assert is_valid_sha256_hex(None) is False
        assert is_valid_sha256_hex(12345) is False
        assert is_valid_sha256_hex([]) is False

    def test_ensure_sha256_hex_valid(self) -> None:
        """ensure_sha256_hex should return valid hash."""
        valid_hash = "a" * 64
        result = ensure_sha256_hex(valid_hash, label="test", error_cls=ValueError)
        assert result == valid_hash

    def test_ensure_sha256_hex_invalid(self) -> None:
        """ensure_sha256_hex should raise on invalid hash."""
        with pytest.raises(ValueError) as exc_info:
            ensure_sha256_hex("invalid", label="test_field", error_cls=ValueError)
        assert "test_field" in str(exc_info.value)
        assert "sha256 hex digest" in str(exc_info.value)


class TestRelativePathValidation:
    """Tests for relative path validation functions."""

    def test_require_unique_relative_paths_valid(self) -> None:
        """Unique paths should pass."""
        records = [
            {"relative_path": "a/file.txt"},
            {"relative_path": "b/file.txt"},
            {"relative_path": "c/file.txt"},
        ]
        require_unique_relative_paths(records, label="test", error_cls=ValueError)

    def test_require_unique_relative_paths_duplicate(self) -> None:
        """Duplicate paths should fail."""
        records = [
            {"relative_path": "a/file.txt"},
            {"relative_path": "a/file.txt"},
        ]
        with pytest.raises(ValueError) as exc_info:
            require_unique_relative_paths(records, label="test", error_cls=ValueError)
        assert "duplicate" in str(exc_info.value).lower()

    def test_require_unique_relative_paths_missing(self) -> None:
        """Missing relative_path should fail."""
        records = [{"other_field": "value"}]
        with pytest.raises(ValueError) as exc_info:
            require_unique_relative_paths(records, label="test", error_cls=ValueError)
        assert "missing relative_path" in str(exc_info.value)

    def test_require_sorted_relative_paths_valid(self) -> None:
        """Sorted paths should pass."""
        records = [
            {"relative_path": "a/file.txt"},
            {"relative_path": "b/file.txt"},
            {"relative_path": "c/file.txt"},
        ]
        require_sorted_relative_paths(records, label="test", error_cls=ValueError)

    def test_require_sorted_relative_paths_unsorted(self) -> None:
        """Unsorted paths should fail."""
        records = [
            {"relative_path": "c/file.txt"},
            {"relative_path": "a/file.txt"},
        ]
        with pytest.raises(ValueError) as exc_info:
            require_sorted_relative_paths(records, label="test", error_cls=ValueError)
        assert "sorted" in str(exc_info.value).lower()


class TestPathIndex:
    """Tests for build_path_index function."""

    def test_build_path_index_valid(self) -> None:
        """Valid records should produce correct index."""
        records = [
            {"relative_path": "a/file.txt", "data": "a"},
            {"relative_path": "b/file.txt", "data": "b"},
        ]
        index = build_path_index(records, label="test", error_cls=ValueError)
        assert len(index) == 2
        assert index["a/file.txt"]["data"] == "a"
        assert index["b/file.txt"]["data"] == "b"

    def test_build_path_index_missing_path(self) -> None:
        """Missing relative_path should fail."""
        records = [{"other_field": "value"}]
        with pytest.raises(ValueError) as exc_info:
            build_path_index(records, label="test", error_cls=ValueError)
        assert "missing relative_path" in str(exc_info.value)


class TestContractVersion:
    """Tests for require_contract_version function."""

    def test_require_contract_version_match(self) -> None:
        """Matching version should pass."""
        result = require_contract_version(
            "tp.meta.capture.v1",
            expected="tp.meta.capture.v1",
            label="test",
            error_cls=ValueError,
        )
        assert result == "tp.meta.capture.v1"

    def test_require_contract_version_mismatch(self) -> None:
        """Mismatched version should fail."""
        with pytest.raises(ValueError) as exc_info:
            require_contract_version(
                "tp.meta.capture.v2",
                expected="tp.meta.capture.v1",
                label="test_contract",
                error_cls=ValueError,
            )
        assert "mismatch" in str(exc_info.value)
        assert "tp.meta.capture.v1" in str(exc_info.value)


class TestStringOrNone:
    """Tests for string_or_none function."""

    def test_string_value(self) -> None:
        """String input should return the string."""
        assert string_or_none("test") == "test"
        assert string_or_none("") == ""

    def test_non_string_value(self) -> None:
        """Non-string input should return None."""
        assert string_or_none(None) is None
        assert string_or_none(123) is None
        assert string_or_none([]) is None
        assert string_or_none({}) is None


class TestSHA256Regex:
    """Tests for the SHA256_HEX_RE regex pattern."""

    def test_regex_matches_valid(self) -> None:
        """Regex should match valid SHA256 hex."""
        assert SHA256_HEX_RE.fullmatch("a" * 64) is not None
        assert SHA256_HEX_RE.fullmatch("0123456789abcdef" * 4) is not None

    def test_regex_rejects_invalid(self) -> None:
        """Regex should reject invalid inputs."""
        assert SHA256_HEX_RE.fullmatch("A" * 64) is None  # uppercase
        assert SHA256_HEX_RE.fullmatch("a" * 63) is None  # too short
        assert SHA256_HEX_RE.fullmatch("a" * 65) is None  # too long
        assert SHA256_HEX_RE.fullmatch("g" * 64) is None  # non-hex
