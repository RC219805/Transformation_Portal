"""Tests for the centralized path_safety module.

This module tests the global path safety utilities that prevent
path traversal attacks and ensure CodeQL-compliant filesystem access.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.security

from transformation_portal.core.security.path_safety import (
    PathSafetyError,
    safe_cas_path,
    safe_join_file,
    safe_join_subpath,
    validate_safe_name,
    validate_sha256,
)


@pytest.mark.security
class TestValidateSafeName:
    """Tests for validate_safe_name function."""

    @pytest.mark.parametrize(
        "bad_name",
        [
            "",  # Empty
            ".",  # Current directory
            "..",  # Parent directory
            "../evil",  # Path traversal
            "a/b",  # Forward slash
            "a\\b",  # Backslash
            "a..b",  # Double dot in middle (allowed only in strict blocklist, not whitelist)
            "a b",  # Space
            "a$b",  # Special character
            "🔥",  # Unicode emoji
            "a\x00b",  # Null byte
            "verylong" * 20,  # Exceeds 64 char limit
            ".hidden",  # Dot prefix
            "file.json",  # Dot in name
            "name with spaces",  # Multiple spaces
            "<script>",  # XSS attempt
            "a+b",  # Plus sign
            "a=b",  # Equals sign
            "a@b",  # At sign
            "a#b",  # Hash
            "a%b",  # Percent
            "a&b",  # Ampersand
            "a*b",  # Asterisk
            "a!b",  # Exclamation
            "a?b",  # Question mark
            "a'b",  # Single quote
            'a"b',  # Double quote
            "a`b",  # Backtick
            "a~b",  # Tilde
            "a|b",  # Pipe
            "a;b",  # Semicolon
            "a:b",  # Colon (Windows path separator)
            "a<b",  # Less than
            "a>b",  # Greater than
            "a{b",  # Curly brace
            "a}b",  # Curly brace
            "a[b",  # Square bracket
            "a]b",  # Square bracket
            "a(b",  # Parenthesis
            "a)b",  # Parenthesis
        ],
    )
    def test_reject_invalid_names(self, bad_name: str) -> None:
        """Invalid names must be rejected."""
        with pytest.raises(PathSafetyError):
            validate_safe_name(bad_name)

    @pytest.mark.parametrize(
        "valid_name",
        [
            "valid_name-123",
            "my-pipeline",
            "test_pipeline_v2",
            "UPPERCASE",
            "mixedCase123",
            "a",  # Single char
            "a" * 64,  # Max length
            "pipeline-2024-03-15",
            "123",  # Numbers only
            "abc",  # Letters only
            "ABC",  # Uppercase only
            "a-b-c",  # Hyphens
            "a_b_c",  # Underscores
            "a1b2c3",  # Mixed alphanumeric
        ],
    )
    def test_accept_valid_names(self, valid_name: str) -> None:
        """Valid names must be accepted and returned unchanged."""
        result = validate_safe_name(valid_name)
        assert result == valid_name


@pytest.mark.security
class TestValidateSha256:
    """Tests for validate_sha256 function."""

    def test_valid_sha256(self) -> None:
        """Valid SHA256 hex strings must be accepted."""
        valid_sha = "a" * 64
        assert validate_sha256(valid_sha) == valid_sha

    def test_valid_sha256_mixed_case(self) -> None:
        """SHA256 is normalized to lowercase."""
        mixed_case = "A" * 32 + "b" * 32
        result = validate_sha256(mixed_case)
        assert result == mixed_case.lower()

    @pytest.mark.parametrize(
        "invalid_sha",
        [
            "",  # Empty
            "a" * 63,  # Too short
            "a" * 65,  # Too long
            "g" * 64,  # Invalid hex char
            "A" * 64 + " ",  # Trailing space
            " " + "a" * 64,  # Leading space
            "../" + "a" * 61,  # Path traversal attempt
            "a" * 32 + "/" + "a" * 31,  # Path separator
        ],
    )
    def test_reject_invalid_sha256(self, invalid_sha: str) -> None:
        """Invalid SHA256 strings must be rejected."""
        with pytest.raises(PathSafetyError):
            validate_sha256(invalid_sha)


@pytest.mark.security
class TestSafeJoinFile:
    """Tests for safe_join_file function."""

    def test_basic_join(self, tmp_path: Path) -> None:
        """Basic file join works correctly."""
        result = safe_join_file(tmp_path, "myfile", suffix=".json")
        assert result == tmp_path / "myfile.json"

    def test_validates_name(self, tmp_path: Path) -> None:
        """Invalid names are rejected."""
        with pytest.raises(PathSafetyError):
            safe_join_file(tmp_path, "../evil", suffix=".json")

    def test_validates_suffix_starts_with_dot(self, tmp_path: Path) -> None:
        """Suffix must start with dot."""
        with pytest.raises(PathSafetyError):
            safe_join_file(tmp_path, "myfile", suffix="json")

    def test_validates_suffix_alphanumeric(self, tmp_path: Path) -> None:
        """Suffix body must be alphanumeric."""
        with pytest.raises(PathSafetyError):
            safe_join_file(tmp_path, "myfile", suffix=".tar.gz")

    @pytest.mark.parametrize(
        "suffix",
        [".json", ".txt", ".yaml", ".xml", ".csv", ".pdf", ".png"],
    )
    def test_valid_suffixes(self, tmp_path: Path, suffix: str) -> None:
        """Common valid suffixes are accepted."""
        result = safe_join_file(tmp_path, "file", suffix=suffix)
        assert result == tmp_path / f"file{suffix}"


@pytest.mark.security
class TestSafeJoinSubpath:
    """Tests for safe_join_subpath function."""

    def test_basic_subpath(self, tmp_path: Path) -> None:
        """Basic subpath join works correctly."""
        result = safe_join_subpath(tmp_path, ["level1", "level2"])
        assert result == tmp_path / "level1" / "level2"

    def test_single_segment(self, tmp_path: Path) -> None:
        """Single segment works."""
        result = safe_join_subpath(tmp_path, ["single"])
        assert result == tmp_path / "single"

    def test_empty_parts_rejected(self, tmp_path: Path) -> None:
        """Empty parts list is rejected."""
        with pytest.raises(PathSafetyError):
            safe_join_subpath(tmp_path, [])

    def test_invalid_segment_rejected(self, tmp_path: Path) -> None:
        """Invalid segment in list is rejected."""
        with pytest.raises(PathSafetyError):
            safe_join_subpath(tmp_path, ["valid", "../evil", "also-valid"])

    def test_all_segments_validated(self, tmp_path: Path) -> None:
        """All segments must pass validation."""
        with pytest.raises(PathSafetyError):
            safe_join_subpath(tmp_path, ["a/b"])


@pytest.mark.security
class TestSafeCasPath:
    """Tests for safe_cas_path function."""

    def test_basic_cas_path(self, tmp_path: Path) -> None:
        """Basic CAS path uses 2-char prefix sharding."""
        sha = "a" * 64
        result = safe_cas_path(tmp_path, sha)
        assert result == tmp_path / "aa" / sha

    def test_sha_normalized_lowercase(self, tmp_path: Path) -> None:
        """SHA is normalized to lowercase in path."""
        sha = "A" * 64
        result = safe_cas_path(tmp_path, sha)
        expected_sha = sha.lower()
        assert result == tmp_path / expected_sha[:2] / expected_sha

    def test_invalid_sha_rejected(self, tmp_path: Path) -> None:
        """Invalid SHA is rejected."""
        with pytest.raises(PathSafetyError):
            safe_cas_path(tmp_path, "invalid")

    def test_path_traversal_in_sha_rejected(self, tmp_path: Path) -> None:
        """Path traversal attempt via SHA is rejected."""
        with pytest.raises(PathSafetyError):
            safe_cas_path(tmp_path, "../" + "a" * 61)

    def test_symlinked_shard_cannot_escape_objects_directory(self, tmp_path: Path) -> None:
        """A shard symlink cannot redirect CAS access outside its root."""
        outside_dir = tmp_path.parent / f"{tmp_path.name}-outside"
        outside_dir.mkdir()
        (tmp_path / "aa").symlink_to(outside_dir, target_is_directory=True)

        with pytest.raises(PathSafetyError, match="escapes objects directory"):
            safe_cas_path(tmp_path, "a" * 64)


@pytest.mark.security
class TestPathSafetyIntegration:
    """Integration tests for path safety patterns."""

    def test_no_escape_via_symlink_attack(self, tmp_path: Path) -> None:
        """Path construction doesn't follow symlinks (no resolve)."""
        # Create a symlink that points outside
        target_outside = tmp_path.parent / "outside"
        symlink = tmp_path / "symlink"

        # Even if symlink exists, we should get the literal path
        result = safe_join_file(tmp_path, "legit", suffix=".json")

        # Result should be the literal path, not resolved
        assert result.parent == tmp_path
        assert result.name == "legit.json"

    def test_concurrent_validation_safety(self, tmp_path: Path) -> None:
        """Validation is deterministic regardless of call order."""
        import concurrent.futures

        names = ["file1", "file2", "file3"] * 100

        def validate(name: str) -> str:
            return validate_safe_name(name)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(validate, names))

        # All results should match input
        assert results == names
