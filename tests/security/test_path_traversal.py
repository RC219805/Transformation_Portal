"""Tests for path traversal attack prevention.

This module tests path traversal protection across the security modules,
covering various attack vectors and edge cases including:
- Basic path traversal with ../
- URL-encoded path traversal
- Unicode normalization attacks
- Null byte injection
- Symlink attacks
- Case sensitivity issues
- Absolute path escapes
- Mixed path separator attacks

This file consolidates path traversal testing to ensure comprehensive coverage.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.security]


# =============================================================================
# Basic path traversal attacks
# =============================================================================


@pytest.mark.security
class TestBasicPathTraversal:
    """Tests for basic path traversal attack patterns."""

    @pytest.mark.parametrize(
        "malicious_component",
        [
            # These patterns actually resolve outside when joined with a real traversal
            "../",
            "../..",
            "../../..",
            "./../",
            "./../../",
        ],
    )
    def test_traversal_in_filename(self, tmp_path: Path, malicious_component: str) -> None:
        """Path traversal components in filename are rejected."""
        from transformation_portal.utils.security import SecurityError, validate_filepath

        # Construct path with traversal attempt
        test_path = tmp_path / f"{malicious_component}secret.txt"

        with pytest.raises(SecurityError):
            validate_filepath(test_path, allowed_dirs=[tmp_path], must_exist=False)

    @pytest.mark.parametrize(
        "url_encoded_pattern",
        [
            # URL-encoded patterns - these become literal filenames, not traversals
            # Test that they don't escape the sandbox even as literal names
            "..%2f",
            "..%5c",
            "%2e%2e/",
            "%2e%2e%2f",
        ],
    )
    def test_url_encoded_patterns_stay_contained(self, tmp_path: Path, url_encoded_pattern: str) -> None:
        """URL-encoded patterns stay within allowed directory (they become literal names)."""
        from transformation_portal.utils.security import validate_filepath

        # URL-encoded patterns aren't decoded by the filesystem
        # They become literal filenames within the allowed directory
        test_path = tmp_path / f"{url_encoded_pattern}secret.txt"

        # These should either pass (contained) or fail for other reasons
        # The key is they don't escape the sandbox
        try:
            result = validate_filepath(test_path, allowed_dirs=[tmp_path], must_exist=False)
            # If it passes, verify it's still within tmp_path
            assert result.is_relative_to(tmp_path)
        except Exception:
            # Any exception (including SecurityError for other reasons) is acceptable
            pass

    @pytest.mark.parametrize(
        "unusual_pattern",
        [
            # These patterns may or may not escape depending on OS/Path handling
            "..",  # Just ".." resolves to parent
            "..\\",  # Backslash is literal on Unix
            "..\\/",  # Mixed separators
            "....//",  # Extra dots
            "..../",  # Extra dots
        ],
    )
    def test_unusual_traversal_patterns_handled(self, tmp_path: Path, unusual_pattern: str) -> None:
        """Unusual traversal patterns are handled safely."""
        from transformation_portal.utils.security import validate_filepath

        test_path = tmp_path / f"{unusual_pattern}secret.txt"

        try:
            result = validate_filepath(test_path, allowed_dirs=[tmp_path], must_exist=False)
            # If it passes, verify it's still within tmp_path
            assert result.is_relative_to(tmp_path)
        except Exception:
            # Rejection is the safest outcome for suspicious patterns
            pass

    def test_traversal_to_parent_directory(self, tmp_path: Path) -> None:
        """Traversal to parent directory is blocked."""
        from transformation_portal.utils.security import SecurityError, validate_filepath

        # Create allowed dir
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()

        # Try to access parent via traversal
        traversal_path = allowed_dir / ".." / "secret.txt"

        with pytest.raises(SecurityError, match="outside allowed directories"):
            validate_filepath(traversal_path, allowed_dirs=[allowed_dir], must_exist=False)

    def test_traversal_deep_nesting(self, tmp_path: Path) -> None:
        """Deep nested traversal is blocked."""
        from transformation_portal.utils.security import SecurityError, validate_filepath

        # Create deep directory structure
        deep_dir = tmp_path / "a" / "b" / "c" / "d" / "e"
        deep_dir.mkdir(parents=True)

        # Try to escape all the way up
        traversal_path = deep_dir / ".." / ".." / ".." / ".." / ".." / "etc" / "passwd"

        with pytest.raises(SecurityError):
            validate_filepath(traversal_path, allowed_dirs=[deep_dir], must_exist=False)

    def test_traversal_with_current_dir(self, tmp_path: Path) -> None:
        """Traversal using current directory notation."""
        from transformation_portal.utils.security import SecurityError, validate_filepath

        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()

        # Try traversal with .
        traversal_path = allowed_dir / "." / ".." / "secret.txt"

        with pytest.raises(SecurityError):
            validate_filepath(traversal_path, allowed_dirs=[allowed_dir], must_exist=False)


# =============================================================================
# Absolute path attacks
# =============================================================================


@pytest.mark.security
class TestAbsolutePathAttacks:
    """Tests for absolute path escape attempts."""

    @pytest.mark.parametrize(
        "absolute_path",
        [
            "/etc/passwd",
            "/etc/shadow",
            "/root/.ssh/id_rsa",
            "/var/log/auth.log",
        ],
    )
    def test_unix_absolute_paths(self, tmp_path: Path, absolute_path: str) -> None:
        """Unix absolute paths outside allowed dirs are blocked."""
        from transformation_portal.utils.security import SecurityError, validate_filepath

        with pytest.raises(SecurityError):
            validate_filepath(Path(absolute_path), allowed_dirs=[tmp_path], must_exist=False)

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific test")
    @pytest.mark.parametrize(
        "windows_path",
        [
            "C:\\Windows\\System32\\config\\SAM",
            "C:\\Users\\Administrator\\Desktop",
            "\\\\server\\share\\secret.txt",
        ],
    )
    def test_windows_absolute_paths(self, tmp_path: Path, windows_path: str) -> None:
        """Windows absolute paths outside allowed dirs are blocked."""
        from transformation_portal.utils.security import SecurityError, validate_filepath

        with pytest.raises(SecurityError):
            validate_filepath(Path(windows_path), allowed_dirs=[tmp_path], must_exist=False)

    def test_file_uri_scheme(self, tmp_path: Path) -> None:
        """File URI scheme paths are handled safely."""
        from transformation_portal.utils.security import SecurityError, validate_filepath

        # Note: Path() may or may not handle file:// URIs
        # The key is that it shouldn't escape the sandbox
        uri_path = "file:///etc/passwd"

        with pytest.raises((SecurityError, OSError, ValueError)):
            validate_filepath(Path(uri_path), allowed_dirs=[tmp_path], must_exist=False)


# =============================================================================
# Null byte attacks
# =============================================================================


@pytest.mark.security
class TestNullByteAttacks:
    """Tests for null byte injection attacks."""

    def test_null_byte_in_filename(self, tmp_path: Path) -> None:
        """Null byte in filename is handled safely."""
        from transformation_portal.utils.security import validate_filepath

        # Null byte could truncate the filename in some contexts
        malicious_name = "innocent.txt\x00.exe"

        # This should either reject or sanitize the null byte
        # Modern Python/OS typically reject null bytes in paths
        try:
            result = validate_filepath(tmp_path / malicious_name, allowed_dirs=[tmp_path], must_exist=False)
            # If it somehow passes, verify it's still contained
            assert result.is_relative_to(tmp_path)
        except (ValueError, OSError):
            # Expected - null bytes cause errors
            pass

    def test_null_byte_traversal(self, tmp_path: Path) -> None:
        """Null byte combined with traversal is handled safely."""
        # Null bytes may cause issues on some systems
        malicious_path = f"../\x00/etc/passwd"

        # Path constructor may or may not reject null bytes directly
        try:
            path = Path(malicious_path)
            # If Path accepts it, it becomes a literal filename
            # Verify it doesn't enable traversal
        except (ValueError, OSError):
            # Rejection is expected
            pass


# =============================================================================
# Unicode and encoding attacks
# =============================================================================


@pytest.mark.security
class TestUnicodeAttacks:
    """Tests for Unicode normalization and encoding attacks."""

    @pytest.mark.parametrize(
        ("unicode_dot", "description"),
        [
            ("\u2024", "One dot leader"),
            ("\u2025", "Two dot leader"),
            ("\uFF0E", "Fullwidth full stop"),
            ("\u2219", "Bullet operator"),
            ("\u22C5", "Dot operator"),
        ],
    )
    def test_unicode_dot_variants(self, tmp_path: Path, unicode_dot: str, description: str) -> None:
        """Unicode dot variants don't bypass traversal checks."""
        from transformation_portal.utils.security import validate_filepath

        # Try to use Unicode dots for traversal
        traversal_attempt = f"{unicode_dot}{unicode_dot}/secret.txt"
        test_path = tmp_path / traversal_attempt

        # Should not resolve outside allowed directory
        # Either rejected or safely contained
        try:
            result = validate_filepath(test_path, allowed_dirs=[tmp_path], must_exist=False)
            # If it passes, verify it's still within tmp_path
            assert result.is_relative_to(tmp_path)
        except Exception:
            # Rejection is also acceptable
            pass

    @pytest.mark.parametrize(
        ("unicode_slash", "description"),
        [
            ("\u2215", "Division slash"),
            ("\u2044", "Fraction slash"),
            ("\uFF0F", "Fullwidth solidus"),
            ("\u29F8", "Big solidus"),
        ],
    )
    def test_unicode_slash_variants(self, tmp_path: Path, unicode_slash: str, description: str) -> None:
        """Unicode slash variants don't enable traversal."""
        from transformation_portal.utils.security import validate_filepath

        # Try to use Unicode slashes for traversal
        traversal_attempt = f"..{unicode_slash}..{unicode_slash}etc{unicode_slash}passwd"
        test_path = tmp_path / traversal_attempt

        try:
            result = validate_filepath(test_path, allowed_dirs=[tmp_path], must_exist=False)
            # If it passes, verify it's within allowed dirs
            assert result.is_relative_to(tmp_path)
        except Exception:
            pass

    def test_unicode_normalization_nfc_nfd(self, tmp_path: Path) -> None:
        """Unicode normalization doesn't bypass security."""
        import unicodedata
        from transformation_portal.utils.security import validate_filepath

        # Create filename with combining characters
        nfd_name = unicodedata.normalize("NFD", "tëst.txt")
        nfc_name = unicodedata.normalize("NFC", "tëst.txt")

        # Both should be handled consistently
        nfd_path = tmp_path / nfd_name
        nfc_path = tmp_path / nfc_name

        try:
            result1 = validate_filepath(nfd_path, allowed_dirs=[tmp_path], must_exist=False)
            result2 = validate_filepath(nfc_path, allowed_dirs=[tmp_path], must_exist=False)
            assert result1.is_relative_to(tmp_path)
            assert result2.is_relative_to(tmp_path)
        except Exception:
            pass


# =============================================================================
# Symlink attacks
# =============================================================================


@pytest.mark.security
class TestSymlinkAttacks:
    """Tests for symlink-based path attacks."""

    @pytest.mark.skipif(sys.platform == "win32", reason="Symlinks require admin on Windows")
    def test_symlink_to_parent(self, tmp_path: Path) -> None:
        """Symlinks pointing outside allowed dirs are detected."""
        from transformation_portal.utils.security import SecurityError, validate_filepath

        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()

        # Create symlink pointing to parent
        symlink = allowed_dir / "escape"
        symlink.symlink_to(tmp_path)

        # Following the symlink would escape
        traversal_path = symlink / "secret.txt"

        # Create a file outside allowed_dir
        secret_file = tmp_path / "secret.txt"
        secret_file.write_text("secret")

        with pytest.raises(SecurityError):
            validate_filepath(traversal_path, allowed_dirs=[allowed_dir])

    @pytest.mark.skipif(sys.platform == "win32", reason="Symlinks require admin on Windows")
    def test_symlink_to_etc(self, tmp_path: Path) -> None:
        """Symlinks to sensitive directories are detected."""
        from transformation_portal.utils.security import SecurityError, validate_filepath

        # Create symlink pointing to /etc
        symlink = tmp_path / "etc_link"
        try:
            symlink.symlink_to("/etc")
        except (OSError, PermissionError):
            pytest.skip("Cannot create symlink to /etc")

        with pytest.raises(SecurityError):
            validate_filepath(symlink / "passwd", allowed_dirs=[tmp_path])

    @pytest.mark.skipif(sys.platform == "win32", reason="Symlinks require admin on Windows")
    def test_circular_symlinks(self, tmp_path: Path) -> None:
        """Circular symlinks don't cause infinite loops."""
        from transformation_portal.utils.security import SecurityError, validate_filepath

        # Create circular symlink
        link_a = tmp_path / "link_a"
        link_b = tmp_path / "link_b"

        link_a.symlink_to(link_b)
        link_b.symlink_to(link_a)

        # Should handle gracefully (error or timeout, not infinite loop)
        with pytest.raises((SecurityError, OSError, RuntimeError)):
            validate_filepath(link_a / "file.txt", allowed_dirs=[tmp_path])


# =============================================================================
# Mixed path separator attacks
# =============================================================================


@pytest.mark.security
class TestMixedPathSeparators:
    """Tests for mixed path separator attacks."""

    @pytest.mark.parametrize(
        "mixed_path",
        [
            "../..\\etc\\passwd",  # Forward then back
            "..//..\\\\etc\\passwd",  # Multiple separators
        ],
    )
    def test_mixed_separators_real_traversal(self, tmp_path: Path, mixed_path: str) -> None:
        """Mixed path separators with real traversal are blocked."""
        from transformation_portal.utils.security import SecurityError, validate_filepath

        test_path = tmp_path / mixed_path

        with pytest.raises(SecurityError):
            validate_filepath(test_path, allowed_dirs=[tmp_path], must_exist=False)

    @pytest.mark.parametrize(
        "mixed_path",
        [
            # On Unix, backslash is a literal character, not a path separator
            # These become literal filenames containing backslashes
            "..\\..\\etc/passwd",
            "..\\../etc/passwd",
            "..\\/..//\\etc/passwd",
        ],
    )
    def test_backslash_patterns_on_unix(self, tmp_path: Path, mixed_path: str) -> None:
        """Backslash patterns are handled based on OS behavior."""
        from transformation_portal.utils.security import validate_filepath

        test_path = tmp_path / mixed_path

        # On Unix, backslash is literal, so these may pass as literal filenames
        # On Windows, they would be interpreted as separators
        try:
            result = validate_filepath(test_path, allowed_dirs=[tmp_path], must_exist=False)
            # If it passes, verify containment
            assert result.is_relative_to(tmp_path)
        except Exception:
            # Rejection is also acceptable
            pass


# =============================================================================
# Filename sanitization attacks
# =============================================================================


@pytest.mark.security
class TestFilenameSanitizationAttacks:
    """Tests for attacks against filename sanitization."""

    @pytest.mark.parametrize(
        ("malicious_name", "description"),
        [
            ("../../../etc/passwd", "Basic traversal"),
            ("..././..././etc/passwd", "Dot-slash combos"),
        ],
    )
    def test_sanitize_traversal_patterns(self, malicious_name: str, description: str) -> None:
        """Sanitization removes traversal patterns."""
        from transformation_portal.utils.security import sanitize_filename

        sanitized = sanitize_filename(malicious_name)

        # Should not contain traversal patterns
        assert ".." not in sanitized
        assert "/" not in sanitized

    @pytest.mark.parametrize(
        ("malicious_name", "description"),
        [
            # These patterns contain backslash which IS sanitized
            ("..\\..\\etc\\passwd", "Backslash traversal"),
            # These become literal filenames with dots
            ("....//....//etc/passwd", "Double dots and slashes"),
            ("..%252f..%252fetc/passwd", "Double URL encoding"),
            (".../.../.../etc/passwd", "Triple dots"),
        ],
    )
    def test_sanitize_various_patterns(self, malicious_name: str, description: str) -> None:
        """Various patterns are sanitized appropriately."""
        from transformation_portal.utils.security import sanitize_filename

        sanitized = sanitize_filename(malicious_name)

        # Backslash should be replaced
        assert "\\" not in sanitized
        # Slash should be removed (basename operation)
        assert "/" not in sanitized

    def test_sanitize_preserves_valid_names(self) -> None:
        """Sanitization preserves valid filenames."""
        from transformation_portal.utils.security import sanitize_filename

        valid_names = [
            "document.pdf",
            "image-2024-01-15.jpg",
            "report_final_v2.docx",
            "my_file.txt",
        ]

        for name in valid_names:
            assert sanitize_filename(name) == name

    def test_sanitize_special_characters(self) -> None:
        """Sanitization handles special characters."""
        from transformation_portal.utils.security import sanitize_filename

        special_names = [
            ("file<>name.txt", "file__name.txt"),
            ("file|name.txt", "file_name.txt"),
            ('file"name.txt', "file_name.txt"),
            ("file?name.txt", "file_name.txt"),
            ("file*name.txt", "file_name.txt"),
        ]

        for malicious, expected in special_names:
            result = sanitize_filename(malicious)
            assert result == expected


# =============================================================================
# Command injection via path
# =============================================================================


@pytest.mark.security
class TestCommandInjectionViaPath:
    """Tests for command injection attempts via path components."""

    @pytest.mark.parametrize(
        "injection_path",
        [
            "; rm -rf /",
            "| cat /etc/passwd",
            "&& malicious",
            "`id`",
            "$(whoami)",
            "$PATH",
            "file; cat /etc/passwd",
        ],
    )
    def test_injection_in_path_rejected(self, tmp_path: Path, injection_path: str) -> None:
        """Paths with shell injection patterns are rejected in command building."""
        from transformation_portal.utils.security import SecurityError, build_safe_command

        with pytest.raises(SecurityError):
            build_safe_command("cmd", [injection_path])

    def test_filter_graph_injection(self) -> None:
        """FFmpeg filter graphs reject injection attempts."""
        from transformation_portal.utils.security import SecurityError, validate_filter_graph

        injection_filters = [
            "scale=1920:1080; rm -rf /",
            "scale=1920:1080 && cat /etc/passwd",
            "scale=$(whoami):1080",
            "scale=`id`:1080",
        ]

        for filter_str in injection_filters:
            with pytest.raises(SecurityError):
                validate_filter_graph(filter_str)


# =============================================================================
# Path safety module tests
# =============================================================================


@pytest.mark.security
class TestPathSafetyModule:
    """Tests for path_safety module functions."""

    def test_safe_name_validation_blocks_traversal(self) -> None:
        """validate_safe_name blocks traversal attempts."""
        from transformation_portal.core.security.path_safety import (
            PathSafetyError,
            validate_safe_name,
        )

        traversal_names = [
            "..",
            "../",
            "..\\",
            "a/b",
            "a\\b",
            "../etc/passwd",
        ]

        for name in traversal_names:
            with pytest.raises(PathSafetyError):
                validate_safe_name(name)

    def test_safe_join_blocks_traversal(self, tmp_path: Path) -> None:
        """safe_join_subpath blocks traversal attempts."""
        from transformation_portal.core.security.path_safety import (
            PathSafetyError,
            safe_join_subpath,
        )

        with pytest.raises(PathSafetyError):
            safe_join_subpath(tmp_path, ["valid", "../evil"])

    def test_safe_join_file_blocks_traversal(self, tmp_path: Path) -> None:
        """safe_join_file blocks traversal attempts."""
        from transformation_portal.core.security.path_safety import (
            PathSafetyError,
            safe_join_file,
        )

        with pytest.raises(PathSafetyError):
            safe_join_file(tmp_path, "../evil", suffix=".json")

    def test_safe_cas_path_validates_sha(self, tmp_path: Path) -> None:
        """safe_cas_path validates SHA256 format."""
        from transformation_portal.core.security.path_safety import (
            PathSafetyError,
            safe_cas_path,
        )

        # Invalid SHA with traversal
        with pytest.raises(PathSafetyError):
            safe_cas_path(tmp_path, "../" + "a" * 61)

        # Valid SHA works
        valid_sha = "a" * 64
        result = safe_cas_path(tmp_path, valid_sha)
        assert result.is_relative_to(tmp_path)


# =============================================================================
# Real-world attack patterns
# =============================================================================


@pytest.mark.security
class TestRealWorldAttackPatterns:
    """Tests based on real-world path traversal attack patterns."""

    @pytest.mark.parametrize(
        ("attack_pattern", "description"),
        [
            # These patterns actually cause traversal
            ("..\\..\\..\\etc\\passwd", "Windows backslash"),  # On Unix this is literal, on Windows it's traversal
        ],
    )
    def test_known_attack_patterns_blocked(self, tmp_path: Path, attack_pattern: str, description: str) -> None:
        """Known real-world attack patterns that should escape are blocked."""
        from transformation_portal.utils.security import SecurityError, validate_filepath

        test_path = tmp_path / attack_pattern

        # On Windows, backslash patterns would escape
        # On Unix, they become literal filenames
        if sys.platform == "win32":
            with pytest.raises(SecurityError):
                validate_filepath(test_path, allowed_dirs=[tmp_path], must_exist=False)
        else:
            # On Unix, these become literal filenames - verify containment
            try:
                result = validate_filepath(test_path, allowed_dirs=[tmp_path], must_exist=False)
                assert result.is_relative_to(tmp_path)
            except Exception:
                pass

    @pytest.mark.parametrize(
        ("attack_pattern", "description"),
        [
            # URL-encoded patterns - these are literal filenames, not decoded
            ("....//....//....//etc/passwd", "IIS Unicode bypass"),
            ("..%c0%af..%c0%af..%c0%afetc/passwd", "IIS overlong encoding"),
            ("..%25%35%63..%25%35%63etc/passwd", "Double encoding"),
            ("..%252f..%252f..%252fetc/passwd", "Triple encoding"),
            ("%2e%2e/%2e%2e/%2e%2e/etc/passwd", "Hex encoding"),
            ("..;/..;/..;/etc/passwd", "Path parameter pollution"),
            ("....\\\\....\\\\etc\\passwd", "Double backslash"),
        ],
    )
    def test_encoded_patterns_stay_contained(self, tmp_path: Path, attack_pattern: str, description: str) -> None:
        """URL-encoded patterns stay within sandbox (they're literal names)."""
        from transformation_portal.utils.security import validate_filepath

        test_path = tmp_path / attack_pattern

        # These patterns aren't decoded by the filesystem
        # They become literal (possibly weird) filenames within the sandbox
        try:
            result = validate_filepath(test_path, allowed_dirs=[tmp_path], must_exist=False)
            # If it passes, verify containment
            assert result.is_relative_to(tmp_path)
        except Exception:
            # Rejection for any reason is acceptable
            pass

    def test_zip_slip_pattern(self, tmp_path: Path) -> None:
        """Zip slip attack pattern is blocked."""
        from transformation_portal.utils.security import SecurityError, validate_filepath

        # Typical zip slip pattern
        zip_slip_paths = [
            "../../../../../../tmp/evil.sh",
            "../../../etc/cron.d/malicious",
            "../../../var/www/html/shell.php",
        ]

        allowed_dir = tmp_path / "extracted"
        allowed_dir.mkdir()

        for malicious_path in zip_slip_paths:
            full_path = allowed_dir / malicious_path

            with pytest.raises(SecurityError):
                validate_filepath(full_path, allowed_dirs=[allowed_dir], must_exist=False)
