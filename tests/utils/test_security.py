"""
Comprehensive tests for security utilities module.

Tests CWE-22 (Path Traversal), CWE-78 (Command Injection), and CWE-400 (Resource Exhaustion)
prevention functions in transformation_portal.utils.security.

These tests validate all security-critical functions that protect against:
- Path traversal attacks via directory traversal sequences (../)
- Command injection via shell metacharacters (;|&`$)
- Resource exhaustion via timeout enforcement
- Unsafe filenames that could cause filesystem exploits

References:
    - ADR-002: docs/architecture/adr/ADR-002-security-input-validation.md
    - SECURITY.md
    - IMPROVEMENT_OPPORTUNITIES.md (SEC-001 testing gap)
"""

# pylint: disable=redefined-outer-name  # pytest fixtures use other fixtures as params

from __future__ import annotations

import os
import signal
import warnings
from pathlib import Path

import pytest

from transformation_portal.utils.security import (
    CONFIG_EXTENSIONS,
    DANGEROUS_SHELL_CHARS,
    IMAGE_EXTENSIONS,
    MAX_CONFIG_SIZE,
    MAX_IMAGE_SIZE,
    MAX_VIDEO_SIZE,
    VIDEO_EXTENSIONS,
    SecurityError,
)
from transformation_portal.utils.security import TimeoutError as SecurityTimeoutError
from transformation_portal.utils.security import (
    build_ffmpeg_command,
    build_safe_command,
    sanitize_filename,
    timeout,
    validate_config_path,
    validate_file_path,
    validate_filepath,
    validate_filter_graph,
    validate_image_path,
    validate_video_path,
)

# Module-level pytest marker - all tests in this module are unit tests (ADR-044)
pytestmark = [
    pytest.mark.unit,
]

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace with sample files for testing."""
    # Create directory structure
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Create sample files
    (data_dir / "test.jpg").write_bytes(b"\xff\xd8\xff" + b"x" * 100)  # JPEG header
    (data_dir / "test.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 100)
    (data_dir / "test.tiff").write_bytes(b"II*\x00" + b"x" * 100)
    (data_dir / "test.mp4").write_bytes(b"\x00\x00\x00\x20ftyp" + b"x" * 100)
    (data_dir / "test.yaml").write_text("key: value\n")
    (data_dir / "test.json").write_text('{"key": "value"}\n')
    (data_dir / "large_file.txt").write_bytes(b"x" * 1000)

    # Create subdirectory
    subdir = data_dir / "subdir"
    subdir.mkdir()
    (subdir / "nested.txt").write_text("nested content")

    return tmp_path


@pytest.fixture
def allowed_dirs(temp_workspace: Path) -> list[Path]:
    """Return list of allowed directories for testing."""
    return [temp_workspace / "data"]


# =============================================================================
# Test: validate_filepath - Path Traversal Prevention (CWE-22)
# =============================================================================


class TestValidateFilepath:
    """Tests for validate_filepath function (CWE-22 Path Traversal)."""

    def test_valid_path_within_allowed_directory(self, temp_workspace: Path, allowed_dirs: list[Path]) -> None:
        """Test that valid paths within allowed directories pass validation."""
        test_file = temp_workspace / "data" / "test.jpg"
        result = validate_filepath(test_file, allowed_dirs)
        assert result == test_file.resolve()

    def test_valid_path_in_subdirectory(self, temp_workspace: Path, allowed_dirs: list[Path]) -> None:
        """Test that paths in subdirectories of allowed dirs pass validation."""
        nested_file = temp_workspace / "data" / "subdir" / "nested.txt"
        result = validate_filepath(nested_file, allowed_dirs)
        assert result == nested_file.resolve()

    def test_rejects_path_outside_allowed_directories(self, temp_workspace: Path) -> None:
        """Test that paths outside allowed directories are rejected."""
        # Create a file outside allowed dir
        outside_file = temp_workspace / "outside.txt"
        outside_file.write_text("content")

        allowed_dirs = [temp_workspace / "data"]

        with pytest.raises(SecurityError, match="outside allowed directories"):
            validate_filepath(outside_file, allowed_dirs)

    def test_rejects_path_traversal_attack_dotdot(self, temp_workspace: Path, allowed_dirs: list[Path]) -> None:
        """Test rejection of classic ../ path traversal attack (CWE-22)."""
        # Craft a path that tries to escape allowed directory
        attack_path = temp_workspace / "data" / ".." / "outside.txt"

        # Create the target file
        (temp_workspace / "outside.txt").write_text("sensitive data")

        with pytest.raises(SecurityError, match="outside allowed directories"):
            validate_filepath(attack_path, allowed_dirs)

    def test_rejects_deep_path_traversal(self, temp_workspace: Path, allowed_dirs: list[Path]) -> None:
        """Test rejection of deep path traversal with multiple ../ segments."""
        # Create a real file outside the allowed "data" directory but still within temp_workspace
        sensitive_dir = temp_workspace / "secret" / "deep"
        sensitive_dir.mkdir(parents=True, exist_ok=True)
        sensitive_file = sensitive_dir / "etc_passwd"
        sensitive_file.write_text("sensitive data")

        # Create the intermediate directory structure so resolve(strict=True) can succeed
        nested_dir = temp_workspace / "data" / "subdir" / "nested"
        nested_dir.mkdir(parents=True, exist_ok=True)

        # Craft a path that starts inside the allowed dir tree and uses multiple ../ segments
        # to escape into the sibling "secret/deep" directory.
        attack_path = temp_workspace / "data" / "subdir" / "nested" / ".." / ".." / ".." / "secret" / "deep" / "etc_passwd"

        with pytest.raises(SecurityError, match="outside allowed directories"):
            validate_filepath(attack_path, allowed_dirs)

    def test_accepts_string_input(self, temp_workspace: Path, allowed_dirs: list[Path]) -> None:
        """Test that string paths are accepted and converted to Path."""
        test_file = str(temp_workspace / "data" / "test.jpg")
        result = validate_filepath(test_file, allowed_dirs)
        assert isinstance(result, Path)

    def test_nonexistent_file_with_must_exist_true(self, temp_workspace: Path, allowed_dirs: list[Path]) -> None:
        """Test that nonexistent files raise error when must_exist=True."""
        nonexistent = temp_workspace / "data" / "nonexistent.jpg"

        with pytest.raises(SecurityError, match="Cannot resolve path"):
            validate_filepath(nonexistent, allowed_dirs, must_exist=True)

    def test_nonexistent_file_with_must_exist_false(self, temp_workspace: Path, allowed_dirs: list[Path]) -> None:
        """Test that nonexistent files pass when must_exist=False."""
        nonexistent = temp_workspace / "data" / "new_output.jpg"
        result = validate_filepath(nonexistent, allowed_dirs, must_exist=False)
        assert result.parent == (temp_workspace / "data").resolve()

    def test_file_size_limit_enforcement(self, temp_workspace: Path, allowed_dirs: list[Path]) -> None:
        """Test that file size limits are enforced."""
        large_file = temp_workspace / "data" / "large_file.txt"
        # File has 1000 bytes, set limit to 500
        with pytest.raises(SecurityError, match="exceeds size limit"):
            validate_filepath(large_file, allowed_dirs, max_file_size=500)

    def test_file_size_within_limit(self, temp_workspace: Path, allowed_dirs: list[Path]) -> None:
        """Test that files within size limit pass validation."""
        large_file = temp_workspace / "data" / "large_file.txt"
        # File has 1000 bytes, set limit to 2000
        result = validate_filepath(large_file, allowed_dirs, max_file_size=2000)
        assert result == large_file.resolve()

    def test_extension_whitelist_pass(self, temp_workspace: Path, allowed_dirs: list[Path]) -> None:
        """Test that allowed extensions pass validation."""
        test_file = temp_workspace / "data" / "test.jpg"
        result = validate_filepath(test_file, allowed_dirs, allowed_extensions=[".jpg", ".png"])
        assert result == test_file.resolve()

    def test_extension_whitelist_reject(self, temp_workspace: Path, allowed_dirs: list[Path]) -> None:
        """Test that disallowed extensions are rejected."""
        test_file = temp_workspace / "data" / "test.jpg"
        with pytest.raises(SecurityError, match="not in whitelist"):
            validate_filepath(test_file, allowed_dirs, allowed_extensions=[".png", ".gif"])

    def test_extension_case_insensitive(self, temp_workspace: Path, allowed_dirs: list[Path]) -> None:
        """Test that extension matching is case-insensitive."""
        test_file = temp_workspace / "data" / "test.jpg"
        result = validate_filepath(test_file, allowed_dirs, allowed_extensions=[".JPG", ".PNG"])
        assert result == test_file.resolve()

    def test_multiple_allowed_directories(self, temp_workspace: Path) -> None:
        """Test validation with multiple allowed directories."""
        extra_dir = temp_workspace / "extra"
        extra_dir.mkdir()
        (extra_dir / "file.txt").write_text("content")

        allowed = [temp_workspace / "data", extra_dir]

        # Both directories should work
        result1 = validate_filepath(temp_workspace / "data" / "test.jpg", allowed)
        result2 = validate_filepath(extra_dir / "file.txt", allowed)

        assert result1.is_file()
        assert result2.is_file()


# =============================================================================
# Test: Type-specific Path Validators
# =============================================================================


class TestTypeSpecificValidators:
    """Tests for validate_image_path, validate_video_path, validate_config_path."""

    def test_validate_image_path_accepts_image_extensions(self, temp_workspace: Path, allowed_dirs: list[Path]) -> None:
        """Test that validate_image_path accepts standard image extensions."""
        test_file = temp_workspace / "data" / "test.jpg"
        result = validate_image_path(test_file, allowed_dirs)
        assert result == test_file.resolve()

    def test_validate_image_path_rejects_non_image(self, temp_workspace: Path, allowed_dirs: list[Path]) -> None:
        """Test that validate_image_path rejects non-image extensions."""
        test_file = temp_workspace / "data" / "test.mp4"
        with pytest.raises(SecurityError, match="not in whitelist"):
            validate_image_path(test_file, allowed_dirs)

    def test_validate_video_path_accepts_video_extensions(self, temp_workspace: Path, allowed_dirs: list[Path]) -> None:
        """Test that validate_video_path accepts standard video extensions."""
        test_file = temp_workspace / "data" / "test.mp4"
        result = validate_video_path(test_file, allowed_dirs)
        assert result == test_file.resolve()

    def test_validate_video_path_rejects_non_video(self, temp_workspace: Path, allowed_dirs: list[Path]) -> None:
        """Test that validate_video_path rejects non-video extensions."""
        test_file = temp_workspace / "data" / "test.jpg"
        with pytest.raises(SecurityError, match="not in whitelist"):
            validate_video_path(test_file, allowed_dirs)

    def test_validate_config_path_accepts_config_extensions(self, temp_workspace: Path, allowed_dirs: list[Path]) -> None:
        """Test that validate_config_path accepts standard config extensions."""
        test_file = temp_workspace / "data" / "test.yaml"
        result = validate_config_path(test_file, allowed_dirs)
        assert result == test_file.resolve()

    def test_validate_config_path_rejects_non_config(self, temp_workspace: Path, allowed_dirs: list[Path]) -> None:
        """Test that validate_config_path rejects non-config extensions."""
        test_file = temp_workspace / "data" / "test.jpg"
        with pytest.raises(SecurityError, match="not in whitelist"):
            validate_config_path(test_file, allowed_dirs)

    def test_extension_constants_are_correct(self) -> None:
        """Test that extension constant sets contain expected values."""
        assert ".jpg" in IMAGE_EXTENSIONS
        assert ".jpeg" in IMAGE_EXTENSIONS
        assert ".png" in IMAGE_EXTENSIONS
        assert ".tif" in IMAGE_EXTENSIONS
        assert ".tiff" in IMAGE_EXTENSIONS

        assert ".mp4" in VIDEO_EXTENSIONS
        assert ".mov" in VIDEO_EXTENSIONS
        assert ".avi" in VIDEO_EXTENSIONS

        assert ".yaml" in CONFIG_EXTENSIONS
        assert ".yml" in CONFIG_EXTENSIONS
        assert ".json" in CONFIG_EXTENSIONS

    def test_size_limit_constants(self) -> None:
        """Test that size limit constants are sensible."""
        assert MAX_IMAGE_SIZE == 100 * 1024 * 1024  # 100MB
        assert MAX_VIDEO_SIZE == 10 * 1024 * 1024 * 1024  # 10GB
        assert MAX_CONFIG_SIZE == 10 * 1024 * 1024  # 10MB


# =============================================================================
# Test: sanitize_filename - Filename Safety
# =============================================================================


class TestSanitizeFilename:
    """Tests for sanitize_filename function."""

    def test_normal_filename_unchanged(self) -> None:
        """Test that normal filenames pass through unchanged."""
        assert sanitize_filename("test.jpg") == "test.jpg"
        assert sanitize_filename("my_image.png") == "my_image.png"
        assert sanitize_filename("document-v2.pdf") == "document-v2.pdf"

    def test_removes_path_components(self) -> None:
        """Test that path components are stripped from filename."""
        assert sanitize_filename("/etc/passwd") == "passwd"
        assert sanitize_filename("../../../etc/passwd") == "passwd"
        assert sanitize_filename("folder/subfolder/file.txt") == "file.txt"

    def test_replaces_dangerous_characters(self) -> None:
        """Test replacement of dangerous characters."""
        # Test each dangerous character
        assert "<" not in sanitize_filename("file<script>.jpg")
        assert ">" not in sanitize_filename("file>redirect.jpg")
        assert ":" not in sanitize_filename("file:alternate.jpg")
        assert '"' not in sanitize_filename('file"quoted.jpg')
        assert "|" not in sanitize_filename("file|pipe.jpg")
        assert "?" not in sanitize_filename("file?query.jpg")
        assert "*" not in sanitize_filename("file*glob.jpg")
        assert "/" not in sanitize_filename("file/slash.jpg")
        assert "\\" not in sanitize_filename("file\\backslash.jpg")
        assert ";" not in sanitize_filename("file;cmd.jpg")

    def test_preserves_extension(self) -> None:
        """Test that file extension is preserved after sanitization."""
        result = sanitize_filename("bad<>:file.jpg")
        assert result.endswith(".jpg")

    def test_strips_leading_trailing_dots(self) -> None:
        """Test that leading/trailing dots and spaces are removed."""
        assert sanitize_filename("...filename.txt") == "filename.txt"
        assert sanitize_filename("filename.txt...") == "filename.txt"
        assert sanitize_filename("  filename.txt  ") == "filename.txt"

    def test_empty_becomes_unnamed(self) -> None:
        """Test that empty/invalid filenames become 'unnamed'."""
        assert sanitize_filename("") == "unnamed"
        assert sanitize_filename("...") == "unnamed"
        assert sanitize_filename("   ") == "unnamed"

    def test_truncates_long_filenames(self) -> None:
        """Test filename truncation to max_length."""
        long_name = "a" * 300 + ".jpg"
        result = sanitize_filename(long_name, max_length=255)
        assert len(result) <= 255
        assert result.endswith(".jpg")

    def test_truncation_preserves_extension(self) -> None:
        """Test that truncation preserves the file extension."""
        long_name = "a" * 300 + ".longextension"
        result = sanitize_filename(long_name, max_length=50)
        assert len(result) <= 50
        assert result.endswith(".longextension")

    def test_unicode_filename(self) -> None:
        """Test handling of Unicode filenames."""
        # Unicode should pass through (not in dangerous chars)
        result = sanitize_filename("文件名.jpg")
        assert result == "文件名.jpg"

    def test_custom_max_length(self) -> None:
        """Test custom max_length parameter."""
        result = sanitize_filename("verylongfilename.txt", max_length=10)
        assert len(result) <= 10


# =============================================================================
# Test: build_safe_command - Command Injection Prevention (CWE-78)
# =============================================================================


class TestBuildSafeCommand:
    """Tests for build_safe_command function (CWE-78 Command Injection)."""

    def test_basic_command_construction(self) -> None:
        """Test basic command list construction."""
        cmd = build_safe_command("ffmpeg", ["-i", "input.mp4", "output.mp4"])
        assert cmd == ["ffmpeg", "-i", "input.mp4", "output.mp4"]

    def test_empty_args(self) -> None:
        """Test command with no arguments."""
        cmd = build_safe_command("ls", [])
        assert cmd == ["ls"]

    def test_rejects_semicolon_in_argument(self) -> None:
        """Test rejection of semicolon (command chain) in arguments."""
        with pytest.raises(SecurityError, match="dangerous characters"):
            build_safe_command("ffmpeg", ["-i", "input.mp4; rm -rf /"])

    def test_rejects_ampersand_in_argument(self) -> None:
        """Test rejection of ampersand (background/chain) in arguments."""
        with pytest.raises(SecurityError, match="dangerous characters"):
            build_safe_command("ffmpeg", ["-i", "input.mp4 && rm -rf /"])

    def test_rejects_pipe_in_argument(self) -> None:
        """Test rejection of pipe character in arguments."""
        with pytest.raises(SecurityError, match="dangerous characters"):
            build_safe_command("ffmpeg", ["-i", "input.mp4 | cat > stolen"])

    def test_rejects_backtick_in_argument(self) -> None:
        """Test rejection of backtick (command substitution) in arguments."""
        with pytest.raises(SecurityError, match="dangerous characters"):
            build_safe_command("ffmpeg", ["-i", "`whoami`.mp4"])

    def test_rejects_dollar_in_argument(self) -> None:
        """Test rejection of dollar sign (variable/subshell) in arguments."""
        with pytest.raises(SecurityError, match="dangerous characters"):
            build_safe_command("ffmpeg", ["-i", "$(whoami).mp4"])

    def test_rejects_parentheses_in_argument(self) -> None:
        """Test rejection of parentheses in arguments."""
        with pytest.raises(SecurityError, match="dangerous characters"):
            build_safe_command("ffmpeg", ["-i", "(rm -rf /).mp4"])

    def test_rejects_angle_brackets_in_argument(self) -> None:
        """Test rejection of angle brackets (redirection) in arguments."""
        with pytest.raises(SecurityError, match="dangerous characters"):
            build_safe_command("ffmpeg", ["-i", "input.mp4 > /dev/null"])

    def test_rejects_dangerous_executable(self) -> None:
        """Test rejection of dangerous characters in executable name."""
        with pytest.raises(SecurityError, match="Executable.*dangerous"):
            build_safe_command("ffmpeg; rm -rf /", ["-i", "input.mp4"])

    def test_all_dangerous_chars_blocked(self) -> None:
        """Test that all DANGEROUS_SHELL_CHARS are blocked."""
        for char in DANGEROUS_SHELL_CHARS:
            with pytest.raises(SecurityError):
                build_safe_command("cmd", [f"arg{char}injection"])

    def test_custom_dangerous_chars(self) -> None:
        """Test custom dangerous character set."""
        # Normally safe character becomes dangerous with custom set
        custom_chars = {"@"}
        with pytest.raises(SecurityError, match="dangerous characters"):
            build_safe_command("cmd", ["user@domain"], dangerous_chars=custom_chars)

    def test_accepts_safe_special_chars(self) -> None:
        """Test that non-dangerous special characters are allowed."""
        # These are safe when not using shell=True
        cmd = build_safe_command(
            "ffmpeg",
            ["-i", "file with spaces.mp4", "-filter:v", "scale=1920:1080"],
        )
        assert "file with spaces.mp4" in cmd

    def test_converts_args_to_strings(self) -> None:
        """Test that non-string arguments are converted to strings."""
        cmd = build_safe_command("python", ["-c", 123])
        assert cmd == ["python", "-c", "123"]


# =============================================================================
# Test: build_ffmpeg_command - FFmpeg Pipeline Safety
# =============================================================================


class TestBuildFfmpegCommand:
    """Tests for build_ffmpeg_command function."""

    def test_basic_ffmpeg_command(self, temp_workspace: Path) -> None:
        """Test basic FFmpeg command construction."""
        input_file = temp_workspace / "data" / "test.mp4"
        output_file = temp_workspace / "data" / "output.mp4"

        cmd = build_ffmpeg_command(
            input_file,
            output_file,
            allowed_dirs=[temp_workspace / "data"],
        )

        assert cmd[0] == "ffmpeg"
        assert "-i" in cmd
        assert str(input_file) in cmd
        assert str(output_file) in cmd

    def test_ffmpeg_with_filters(self, temp_workspace: Path) -> None:
        """Test FFmpeg command with video filters."""
        input_file = temp_workspace / "data" / "test.mp4"
        output_file = temp_workspace / "data" / "output.mp4"

        cmd = build_ffmpeg_command(
            input_file,
            output_file,
            filters=["scale=1920:1080", "fps=30"],
            allowed_dirs=[temp_workspace / "data"],
        )

        assert "-vf" in cmd
        # Filters should be joined with commas
        vf_index = cmd.index("-vf")
        assert cmd[vf_index + 1] == "scale=1920:1080,fps=30"

    def test_ffmpeg_rejects_dangerous_filters(self, temp_workspace: Path) -> None:
        """Test rejection of dangerous characters in filters."""
        input_file = temp_workspace / "data" / "test.mp4"
        output_file = temp_workspace / "data" / "output.mp4"

        with pytest.raises(SecurityError, match="dangerous characters"):
            build_ffmpeg_command(
                input_file,
                output_file,
                filters=["scale=1920; rm -rf /"],
                allowed_dirs=[temp_workspace / "data"],
            )

    def test_ffmpeg_codec_option(self, temp_workspace: Path) -> None:
        """Test FFmpeg command with custom codec."""
        input_file = temp_workspace / "data" / "test.mp4"
        output_file = temp_workspace / "data" / "output.mp4"

        cmd = build_ffmpeg_command(
            input_file,
            output_file,
            codec="libx265",
            allowed_dirs=[temp_workspace / "data"],
        )

        assert "-c:v" in cmd
        cv_index = cmd.index("-c:v")
        assert cmd[cv_index + 1] == "libx265"

    def test_ffmpeg_additional_args(self, temp_workspace: Path) -> None:
        """Test FFmpeg command with additional arguments."""
        input_file = temp_workspace / "data" / "test.mp4"
        output_file = temp_workspace / "data" / "output.mp4"

        cmd = build_ffmpeg_command(
            input_file,
            output_file,
            additional_args=["-crf", "23", "-preset", "fast"],
            allowed_dirs=[temp_workspace / "data"],
        )

        assert "-crf" in cmd
        assert "23" in cmd
        assert "-preset" in cmd
        assert "fast" in cmd

    def test_ffmpeg_path_validation_default(self, temp_workspace: Path) -> None:
        """Test that path validation is enabled by default."""
        # Try to access file outside allowed directory
        input_file = temp_workspace / "outside.txt"
        input_file.write_text("content")
        output_file = temp_workspace / "data" / "output.mp4"

        with pytest.raises(SecurityError, match="outside allowed directories"):
            build_ffmpeg_command(
                input_file,
                output_file,
                allowed_dirs=[temp_workspace / "data"],
            )

    def test_ffmpeg_path_validation_disabled(self, temp_workspace: Path) -> None:
        """Test FFmpeg command with path validation disabled."""
        # Path outside allowed directory
        input_file = temp_workspace / "outside.txt"
        input_file.write_text("content")
        output_file = temp_workspace / "output.mp4"

        # Should succeed when validation is disabled
        cmd = build_ffmpeg_command(
            input_file,
            output_file,
            validate_paths=False,
        )

        assert str(input_file) in cmd
        assert str(output_file) in cmd

    def test_ffmpeg_default_allowed_dirs_cwd(self, temp_workspace: Path) -> None:
        """Test that allowed_dirs defaults to cwd if not specified."""
        # Change to temp directory for this test
        old_cwd = os.getcwd()
        try:
            os.chdir(temp_workspace / "data")

            input_file = Path("test.mp4")
            output_file = Path("output.mp4")

            cmd = build_ffmpeg_command(input_file, output_file)
            assert "ffmpeg" in cmd
        finally:
            os.chdir(old_cwd)


# =============================================================================
# Test: validate_filter_graph - Filter Graph Safety
# =============================================================================


class TestValidateFilterGraph:
    """Tests for validate_filter_graph function."""

    def test_valid_filter_graph(self) -> None:
        """Test that valid filter graphs pass validation."""
        assert validate_filter_graph("scale=1920:1080") == "scale=1920:1080"
        assert validate_filter_graph("fps=30") == "fps=30"
        assert validate_filter_graph("scale=1920:1080,fps=30") == "scale=1920:1080,fps=30"

    def test_complex_filter_graph(self) -> None:
        """Test complex but valid filter graph."""
        complex_filter = "[0:v]scale=1920:1080[scaled];[scaled]fps=30[out]"
        # This contains semicolons which are dangerous
        with pytest.raises(SecurityError, match="dangerous characters"):
            validate_filter_graph(complex_filter)

    def test_rejects_semicolon(self) -> None:
        """Test rejection of semicolon in filter graph."""
        with pytest.raises(SecurityError, match="dangerous characters"):
            validate_filter_graph("scale=1920; rm -rf /")

    def test_rejects_all_dangerous_chars(self) -> None:
        """Test rejection of all dangerous characters in filter graph."""
        for char in DANGEROUS_SHELL_CHARS:
            with pytest.raises(SecurityError):
                validate_filter_graph(f"scale=1920{char}1080")

    def test_shows_which_chars_found(self) -> None:
        """Test that error message shows which dangerous chars were found."""
        try:
            validate_filter_graph("scale=1920;1080")
        except SecurityError as e:
            assert ";" in str(e)


# =============================================================================
# Test: timeout - Resource Exhaustion Prevention (CWE-400)
# =============================================================================


class TestTimeout:
    """Tests for timeout context manager (CWE-400 Resource Exhaustion)."""

    @pytest.mark.skipif(
        not hasattr(signal, "SIGALRM"),
        reason="SIGALRM not available on this platform (Windows)",
    )
    def test_timeout_not_exceeded(self) -> None:
        """Test that operations completing within timeout succeed."""
        with timeout(5):
            result = sum(range(100))
        assert result == 4950  # Sum of 0-99

    @pytest.mark.skipif(
        not hasattr(signal, "SIGALRM"),
        reason="SIGALRM not available on this platform (Windows)",
    )
    def test_timeout_exceeded_raises_error(self) -> None:
        """Test that exceeding timeout raises TimeoutError."""
        import time

        with pytest.raises(SecurityTimeoutError, match="exceeded.*timeout"):
            with timeout(1):
                time.sleep(3)

    @pytest.mark.skipif(
        hasattr(signal, "SIGALRM"),
        reason="Test for Windows behavior",
    )
    def test_timeout_raises_on_windows(self) -> None:
        """Test that timeout raises NotImplementedError on Windows."""
        with pytest.raises(NotImplementedError, match="Unix-only"):
            with timeout(5):
                pass

    @pytest.mark.skipif(
        not hasattr(signal, "SIGALRM"),
        reason="SIGALRM not available on this platform (Windows)",
    )
    def test_timeout_restores_signal_handler(self) -> None:
        """Test that timeout properly restores original signal handler."""
        original_handler = signal.getsignal(signal.SIGALRM)

        with timeout(5):
            pass

        restored_handler = signal.getsignal(signal.SIGALRM)
        assert restored_handler == original_handler

    @pytest.mark.skipif(
        not hasattr(signal, "SIGALRM"),
        reason="SIGALRM not available on this platform (Windows)",
    )
    def test_timeout_cancels_alarm_on_success(self) -> None:
        """Test that alarm is cancelled when operation succeeds."""
        with timeout(10):
            # Quick operation
            pass

        # After a successful operation, no alarm should remain scheduled.
        # signal.alarm(0) cancels any pending alarm and returns the remaining
        # seconds; this should be 0 if the timeout context cleared it.
        remaining = signal.alarm(0)
        assert remaining == 0


# =============================================================================
# Test: Deprecated Functions
# =============================================================================


class TestDeprecatedFunctions:
    """Tests for deprecated function aliases."""

    def test_validate_file_path_deprecation_warning(self, temp_workspace: Path, allowed_dirs: list[Path]) -> None:
        """Test that validate_file_path emits deprecation warning."""
        test_file = temp_workspace / "data" / "test.jpg"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_file_path(test_file, allowed_dirs)

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()

    def test_validate_file_path_still_works(self, temp_workspace: Path, allowed_dirs: list[Path]) -> None:
        """Test that deprecated validate_file_path still functions correctly."""
        test_file = temp_workspace / "data" / "test.jpg"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = validate_file_path(test_file, allowed_dirs)

        assert result == test_file.resolve()


# =============================================================================
# Test: SecurityError Exception
# =============================================================================


class TestSecurityError:
    """Tests for SecurityError exception class."""

    def test_security_error_inherits_from_exception(self) -> None:
        """Test that SecurityError inherits from Exception."""
        assert issubclass(SecurityError, Exception)

    def test_security_error_message(self) -> None:
        """Test SecurityError with custom message."""
        error = SecurityError("Path traversal detected")
        assert str(error) == "Path traversal detected"

    def test_security_error_can_be_caught(self) -> None:
        """Test that SecurityError can be caught and handled."""
        caught = False
        try:
            raise SecurityError("Test error")
        except SecurityError as e:
            caught = True
            assert "Test error" in str(e)
        assert caught


# =============================================================================
# Test: Constants
# =============================================================================


class TestSecurityConstants:
    """Tests for security module constants."""

    def test_dangerous_shell_chars_completeness(self) -> None:
        """Test that DANGEROUS_SHELL_CHARS contains key injection vectors."""
        required_chars = {";", "&", "|", "`", "$", "(", ")", "<", ">"}
        assert required_chars.issubset(DANGEROUS_SHELL_CHARS)

    def test_dangerous_shell_chars_is_set(self) -> None:
        """Test that DANGEROUS_SHELL_CHARS is a set for O(1) lookup."""
        assert isinstance(DANGEROUS_SHELL_CHARS, set)


# =============================================================================
# Test: Edge Cases and Integration
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_symlink_resolution(self, temp_workspace: Path) -> None:
        """Test that symlinks are resolved and validated."""
        # Create a symlink pointing outside allowed directory
        target = temp_workspace / "outside.txt"
        target.write_text("sensitive data")

        link = temp_workspace / "data" / "link_to_outside"
        try:
            link.symlink_to(target)
        except OSError:
            pytest.skip("Symlinks not supported on this platform")

        allowed_dirs = [temp_workspace / "data"]

        # Symlink should resolve to outside and be rejected
        with pytest.raises(SecurityError, match="outside allowed directories"):
            validate_filepath(link, allowed_dirs)

    def test_symlink_within_allowed_directory(self, temp_workspace: Path) -> None:
        """Test that symlinks within allowed directories work."""
        target = temp_workspace / "data" / "test.jpg"
        link = temp_workspace / "data" / "link_to_test.jpg"
        try:
            link.symlink_to(target)
        except OSError:
            pytest.skip("Symlinks not supported on this platform")

        allowed_dirs = [temp_workspace / "data"]
        result = validate_filepath(link, allowed_dirs)

        # Should resolve to the target
        assert result == target.resolve()

    def test_directory_instead_of_file(self, temp_workspace: Path, allowed_dirs: list[Path]) -> None:
        """Test behavior when directory is passed instead of file."""
        directory = temp_workspace / "data" / "subdir"
        # Should not raise for existence, but may fail size check
        result = validate_filepath(directory, allowed_dirs)
        assert result == directory.resolve()

    def test_empty_allowed_dirs(self, temp_workspace: Path) -> None:
        """Test behavior with empty allowed_dirs list."""
        test_file = temp_workspace / "data" / "test.jpg"

        with pytest.raises(SecurityError, match="outside allowed directories"):
            validate_filepath(test_file, [])

    def test_relative_path_resolution(self, temp_workspace: Path) -> None:
        """Test that relative paths are properly resolved."""
        old_cwd = os.getcwd()
        try:
            os.chdir(temp_workspace / "data")

            # Use relative path
            result = validate_filepath(Path("test.jpg"), [temp_workspace / "data"])

            assert result.is_absolute()
            assert result.exists()
        finally:
            os.chdir(old_cwd)

    def test_very_long_path(self, temp_workspace: Path) -> None:
        """Test handling of very long paths."""
        # Create deeply nested directory (respecting filesystem limits)
        deep_path = temp_workspace / "data"
        for i in range(10):
            deep_path = deep_path / f"level_{i}"
        deep_path.mkdir(parents=True)
        (deep_path / "file.txt").write_text("content")

        allowed_dirs = [temp_workspace / "data"]
        result = validate_filepath(deep_path / "file.txt", allowed_dirs)
        assert result.exists()

    def test_special_characters_in_filename(self, temp_workspace: Path, allowed_dirs: list[Path]) -> None:
        """Test files with special (but safe) characters in name."""
        special_file = temp_workspace / "data" / "test file (1).jpg"
        special_file.write_bytes(b"\xff\xd8\xff")

        result = validate_filepath(special_file, allowed_dirs)
        assert result == special_file.resolve()
