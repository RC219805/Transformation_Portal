"""Tests for security utilities.

This module tests the security utilities from utils/security.py
which provides validation functions to prevent common security vulnerabilities:
- Path traversal (CWE-22)
- Command injection (CWE-78)
- Resource exhaustion (CWE-400)

Covers:
- validate_filepath with various attack vectors
- validate_image_path, validate_video_path, validate_config_path
- sanitize_filename
- build_safe_command
- build_ffmpeg_command
- validate_filter_graph
- timeout context manager
- SecurityError exception
"""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.security]


# =============================================================================
# Test SecurityError exception
# =============================================================================


@pytest.mark.security
class TestSecurityError:
    """Tests for SecurityError exception class."""

    def test_security_error_is_exception(self) -> None:
        """SecurityError is a proper exception."""
        from transformation_portal.utils.security import SecurityError

        assert issubclass(SecurityError, Exception)

    def test_security_error_message(self) -> None:
        """SecurityError stores message correctly."""
        from transformation_portal.utils.security import SecurityError

        error = SecurityError("Test security violation")
        assert str(error) == "Test security violation"

    def test_security_error_can_be_raised(self) -> None:
        """SecurityError can be raised and caught."""
        from transformation_portal.utils.security import SecurityError

        with pytest.raises(SecurityError) as exc_info:
            raise SecurityError("Path traversal detected")

        assert "Path traversal detected" in str(exc_info.value)


# =============================================================================
# Test validate_filepath
# =============================================================================


@pytest.mark.security
class TestValidateFilepath:
    """Tests for validate_filepath function."""

    def test_validate_valid_filepath(self, tmp_path: Path) -> None:
        """Valid filepath within allowed directory passes."""
        from transformation_portal.utils.security import validate_filepath

        test_file = tmp_path / "valid_file.txt"
        test_file.write_text("content")

        result = validate_filepath(test_file, allowed_dirs=[tmp_path])

        assert result.is_absolute()
        assert result.exists()

    def test_validate_filepath_string_input(self, tmp_path: Path) -> None:
        """String input is converted to Path."""
        from transformation_portal.utils.security import validate_filepath

        test_file = tmp_path / "string_test.txt"
        test_file.write_text("content")

        result = validate_filepath(str(test_file), allowed_dirs=[tmp_path])

        assert isinstance(result, Path)
        assert result.exists()

    def test_validate_filepath_rejects_traversal_dotdot(self, tmp_path: Path) -> None:
        """Path traversal with .. is rejected."""
        from transformation_portal.utils.security import SecurityError, validate_filepath

        test_file = tmp_path / "legit.txt"
        test_file.write_text("content")

        with pytest.raises(SecurityError):
            validate_filepath(tmp_path / ".." / "etc" / "passwd", allowed_dirs=[tmp_path])

    def test_validate_filepath_rejects_absolute_escape(self, tmp_path: Path) -> None:
        """Absolute path outside allowed directory is rejected."""
        from transformation_portal.utils.security import SecurityError, validate_filepath

        with pytest.raises(SecurityError):
            validate_filepath(Path("/etc/passwd"), allowed_dirs=[tmp_path], must_exist=False)

    def test_validate_filepath_file_not_found(self, tmp_path: Path) -> None:
        """Non-existent file raises SecurityError when must_exist=True."""
        from transformation_portal.utils.security import SecurityError, validate_filepath

        nonexistent = tmp_path / "does_not_exist.txt"

        with pytest.raises(SecurityError, match="(does not exist|Cannot resolve)"):
            validate_filepath(nonexistent, allowed_dirs=[tmp_path], must_exist=True)

    def test_validate_filepath_must_exist_false(self, tmp_path: Path) -> None:
        """Non-existent file allowed when must_exist=False."""
        from transformation_portal.utils.security import validate_filepath

        nonexistent = tmp_path / "new_file.txt"

        result = validate_filepath(nonexistent, allowed_dirs=[tmp_path], must_exist=False)

        assert not result.exists()

    def test_validate_filepath_file_size_limit(self, tmp_path: Path) -> None:
        """File exceeding size limit is rejected."""
        from transformation_portal.utils.security import SecurityError, validate_filepath

        large_file = tmp_path / "large.bin"
        large_file.write_bytes(b"x" * 1000)

        with pytest.raises(SecurityError, match="exceeds size limit"):
            validate_filepath(large_file, allowed_dirs=[tmp_path], max_file_size=500)

    def test_validate_filepath_within_size_limit(self, tmp_path: Path) -> None:
        """File within size limit passes."""
        from transformation_portal.utils.security import validate_filepath

        small_file = tmp_path / "small.bin"
        small_file.write_bytes(b"x" * 100)

        result = validate_filepath(small_file, allowed_dirs=[tmp_path], max_file_size=500)

        assert result.exists()

    def test_validate_filepath_extension_whitelist(self, tmp_path: Path) -> None:
        """File with allowed extension passes."""
        from transformation_portal.utils.security import validate_filepath

        jpg_file = tmp_path / "image.jpg"
        jpg_file.write_bytes(b"fake jpg")

        result = validate_filepath(jpg_file, allowed_dirs=[tmp_path], allowed_extensions=[".jpg", ".png"])

        assert result.suffix == ".jpg"

    def test_validate_filepath_extension_rejected(self, tmp_path: Path) -> None:
        """File with disallowed extension is rejected."""
        from transformation_portal.utils.security import SecurityError, validate_filepath

        exe_file = tmp_path / "program.exe"
        exe_file.write_bytes(b"fake exe")

        with pytest.raises(SecurityError, match="extension.*not in whitelist"):
            validate_filepath(exe_file, allowed_dirs=[tmp_path], allowed_extensions=[".jpg", ".png"])

    def test_validate_filepath_extension_case_insensitive(self, tmp_path: Path) -> None:
        """Extension check is case-insensitive."""
        from transformation_portal.utils.security import validate_filepath

        jpg_file = tmp_path / "image.JPG"
        jpg_file.write_bytes(b"fake jpg")

        result = validate_filepath(jpg_file, allowed_dirs=[tmp_path], allowed_extensions=[".jpg"])

        assert result.exists()

    def test_validate_filepath_multiple_allowed_dirs(self, tmp_path: Path) -> None:
        """File in any allowed directory passes."""
        from transformation_portal.utils.security import validate_filepath

        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()

        file_in_dir2 = dir2 / "file.txt"
        file_in_dir2.write_text("content")

        result = validate_filepath(file_in_dir2, allowed_dirs=[dir1, dir2])

        assert result.exists()

    @pytest.mark.parametrize(
        "malicious_path",
        [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/passwd",
            "C:\\Windows\\System32",
            "....//....//etc/passwd",
            "./../.../etc/passwd",
        ],
    )
    def test_validate_filepath_traversal_patterns(self, tmp_path: Path, malicious_path: str) -> None:
        """Various path traversal patterns are rejected."""
        from transformation_portal.utils.security import SecurityError, validate_filepath

        with pytest.raises(SecurityError):
            validate_filepath(Path(malicious_path), allowed_dirs=[tmp_path], must_exist=False)


# =============================================================================
# Test specialized path validators
# =============================================================================


@pytest.mark.security
class TestSpecializedPathValidators:
    """Tests for validate_image_path, validate_video_path, validate_config_path."""

    def test_validate_image_path_valid(self, tmp_path: Path) -> None:
        """Valid image path passes validation."""
        from transformation_portal.utils.security import validate_image_path

        image = tmp_path / "test.jpg"
        image.write_bytes(b"fake jpg data")

        result = validate_image_path(image, allowed_dirs=[tmp_path])

        assert result.suffix == ".jpg"

    def test_validate_image_path_wrong_extension(self, tmp_path: Path) -> None:
        """Non-image extension is rejected."""
        from transformation_portal.utils.security import SecurityError, validate_image_path

        text_file = tmp_path / "document.txt"
        text_file.write_text("not an image")

        with pytest.raises(SecurityError, match="extension.*not in whitelist"):
            validate_image_path(text_file, allowed_dirs=[tmp_path])

    @pytest.mark.parametrize("ext", [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"])
    def test_validate_image_path_all_extensions(self, tmp_path: Path, ext: str) -> None:
        """All supported image extensions pass."""
        from transformation_portal.utils.security import validate_image_path

        image = tmp_path / f"test{ext}"
        image.write_bytes(b"fake image")

        result = validate_image_path(image, allowed_dirs=[tmp_path])

        assert result.exists()

    def test_validate_video_path_valid(self, tmp_path: Path) -> None:
        """Valid video path passes validation."""
        from transformation_portal.utils.security import validate_video_path

        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake video data")

        result = validate_video_path(video, allowed_dirs=[tmp_path])

        assert result.suffix == ".mp4"

    @pytest.mark.parametrize("ext", [".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"])
    def test_validate_video_path_all_extensions(self, tmp_path: Path, ext: str) -> None:
        """All supported video extensions pass."""
        from transformation_portal.utils.security import validate_video_path

        video = tmp_path / f"test{ext}"
        video.write_bytes(b"fake video")

        result = validate_video_path(video, allowed_dirs=[tmp_path])

        assert result.exists()

    def test_validate_config_path_valid(self, tmp_path: Path) -> None:
        """Valid config path passes validation."""
        from transformation_portal.utils.security import validate_config_path

        config = tmp_path / "config.yaml"
        config.write_text("key: value")

        result = validate_config_path(config, allowed_dirs=[tmp_path])

        assert result.suffix == ".yaml"

    @pytest.mark.parametrize("ext", [".yaml", ".yml", ".json", ".toml"])
    def test_validate_config_path_all_extensions(self, tmp_path: Path, ext: str) -> None:
        """All supported config extensions pass."""
        from transformation_portal.utils.security import validate_config_path

        config = tmp_path / f"config{ext}"
        config.write_text("{}")

        result = validate_config_path(config, allowed_dirs=[tmp_path])

        assert result.exists()


# =============================================================================
# Test sanitize_filename
# =============================================================================


@pytest.mark.security
class TestSanitizeFilename:
    """Tests for sanitize_filename function."""

    def test_sanitize_simple_filename(self) -> None:
        """Simple filename passes through unchanged."""
        from transformation_portal.utils.security import sanitize_filename

        assert sanitize_filename("simple.txt") == "simple.txt"

    def test_sanitize_removes_path_components(self) -> None:
        """Path traversal components are removed."""
        from transformation_portal.utils.security import sanitize_filename

        result = sanitize_filename("../../../etc/passwd")
        assert ".." not in result
        assert "/" not in result

    @pytest.mark.parametrize(
        ("input_name", "expected_result"),
        [
            ("file<script>.jpg", "file_script_.jpg"),
            ("file>name.txt", "file_name.txt"),
            ('file"quote.txt', "file_quote.txt"),
            ("file|pipe.txt", "file_pipe.txt"),
            ("file?query.txt", "file_query.txt"),
            ("file*glob.txt", "file_glob.txt"),
            ("file:colon.txt", "file_colon.txt"),
            ("file\\backslash.txt", "file_backslash.txt"),
            ("file;semicolon.txt", "file_semicolon.txt"),
        ],
    )
    def test_sanitize_replaces_dangerous_chars(self, input_name: str, expected_result: str) -> None:
        """Dangerous characters are replaced with underscore."""
        from transformation_portal.utils.security import sanitize_filename

        result = sanitize_filename(input_name)
        assert result == expected_result

    def test_sanitize_strips_dots_and_spaces(self) -> None:
        """Leading/trailing dots and spaces are stripped."""
        from transformation_portal.utils.security import sanitize_filename

        assert sanitize_filename("...file.txt...") == "file.txt"
        assert sanitize_filename("  file.txt  ") == "file.txt"
        assert sanitize_filename(". file .") == "file"

    def test_sanitize_empty_becomes_unnamed(self) -> None:
        """Empty filename becomes 'unnamed'."""
        from transformation_portal.utils.security import sanitize_filename

        assert sanitize_filename("") == "unnamed"
        assert sanitize_filename("...") == "unnamed"
        assert sanitize_filename("   ") == "unnamed"

    def test_sanitize_truncates_long_filename(self) -> None:
        """Long filenames are truncated."""
        from transformation_portal.utils.security import sanitize_filename

        long_name = "a" * 300 + ".txt"
        result = sanitize_filename(long_name)

        assert len(result) <= 255
        assert result.endswith(".txt")

    def test_sanitize_preserves_extension_on_truncate(self) -> None:
        """Extension is preserved when truncating."""
        from transformation_portal.utils.security import sanitize_filename

        long_name = "x" * 260 + ".jpeg"
        result = sanitize_filename(long_name, max_length=50)

        assert len(result) == 50
        assert result.endswith(".jpeg")

    def test_sanitize_custom_max_length(self) -> None:
        """Custom max_length is respected."""
        from transformation_portal.utils.security import sanitize_filename

        result = sanitize_filename("verylongfilename.txt", max_length=10)

        assert len(result) <= 10


# =============================================================================
# Test build_safe_command
# =============================================================================


@pytest.mark.security
class TestBuildSafeCommand:
    """Tests for build_safe_command function."""

    def test_build_basic_command(self) -> None:
        """Basic command is built correctly."""
        from transformation_portal.utils.security import build_safe_command

        result = build_safe_command("ls", ["-la", "/tmp"])

        assert result == ["ls", "-la", "/tmp"]

    def test_build_empty_args(self) -> None:
        """Command with no args is built correctly."""
        from transformation_portal.utils.security import build_safe_command

        result = build_safe_command("pwd", [])

        assert result == ["pwd"]

    @pytest.mark.parametrize(
        "dangerous_char",
        [";", "&", "|", "`", "$", "(", ")", "<", ">"],
    )
    def test_build_rejects_dangerous_char_in_arg(self, dangerous_char: str) -> None:
        """Arguments with shell metacharacters are rejected."""
        from transformation_portal.utils.security import SecurityError, build_safe_command

        with pytest.raises(SecurityError, match="dangerous characters"):
            build_safe_command("cmd", [f"arg{dangerous_char}value"])

    @pytest.mark.parametrize(
        "injection_attempt",
        [
            "; rm -rf /",
            "| cat /etc/passwd",
            "&& malicious",
            "$(whoami)",
            "`id`",
            "file > /etc/passwd",
            "file < /dev/zero",
        ],
    )
    def test_build_rejects_injection_attempts(self, injection_attempt: str) -> None:
        """Command injection attempts are rejected."""
        from transformation_portal.utils.security import SecurityError, build_safe_command

        with pytest.raises(SecurityError):
            build_safe_command("cmd", [injection_attempt])

    def test_build_rejects_dangerous_executable(self) -> None:
        """Dangerous characters in executable are rejected."""
        from transformation_portal.utils.security import SecurityError, build_safe_command

        with pytest.raises(SecurityError, match="Executable.*dangerous"):
            build_safe_command("rm; cat /etc/passwd", [])

    def test_build_with_custom_dangerous_chars(self) -> None:
        """Custom dangerous chars set can be used."""
        from transformation_portal.utils.security import SecurityError, build_safe_command

        # Default would reject this
        with pytest.raises(SecurityError):
            build_safe_command("cmd", ["file$var"])

        # Custom set without $ allows it
        result = build_safe_command("cmd", ["file$var"], dangerous_chars={";", "&"})
        assert result == ["cmd", "file$var"]

    def test_build_converts_args_to_strings(self) -> None:
        """Non-string arguments are converted to strings."""
        from transformation_portal.utils.security import build_safe_command

        result = build_safe_command("cmd", [123, 45.6, "str"])

        assert result == ["cmd", "123", "45.6", "str"]


# =============================================================================
# Test build_ffmpeg_command
# =============================================================================


@pytest.mark.security
class TestBuildFfmpegCommand:
    """Tests for build_ffmpeg_command function."""

    def test_build_basic_ffmpeg(self, tmp_path: Path) -> None:
        """Basic FFmpeg command is built correctly."""
        from transformation_portal.utils.security import build_ffmpeg_command

        input_file = tmp_path / "input.mp4"
        output_file = tmp_path / "output.mp4"
        input_file.write_bytes(b"fake video")

        result = build_ffmpeg_command(input_file, output_file, allowed_dirs=[tmp_path])

        assert result[0] == "ffmpeg"
        assert "-i" in result
        assert str(input_file) in result
        assert str(output_file) in result

    def test_build_ffmpeg_with_filters(self, tmp_path: Path) -> None:
        """FFmpeg command with filters is built correctly."""
        from transformation_portal.utils.security import build_ffmpeg_command

        input_file = tmp_path / "input.mp4"
        output_file = tmp_path / "output.mp4"
        input_file.write_bytes(b"fake video")

        result = build_ffmpeg_command(
            input_file,
            output_file,
            filters=["scale=1920:1080", "fps=30"],
            allowed_dirs=[tmp_path],
        )

        assert "-vf" in result
        filter_idx = result.index("-vf")
        assert result[filter_idx + 1] == "scale=1920:1080,fps=30"

    def test_build_ffmpeg_with_codec(self, tmp_path: Path) -> None:
        """FFmpeg command respects codec setting."""
        from transformation_portal.utils.security import build_ffmpeg_command

        input_file = tmp_path / "input.mp4"
        output_file = tmp_path / "output.mp4"
        input_file.write_bytes(b"fake video")

        result = build_ffmpeg_command(
            input_file,
            output_file,
            codec="libx265",
            allowed_dirs=[tmp_path],
        )

        assert "-c:v" in result
        codec_idx = result.index("-c:v")
        assert result[codec_idx + 1] == "libx265"

    def test_build_ffmpeg_rejects_dangerous_filter(self, tmp_path: Path) -> None:
        """FFmpeg command rejects dangerous filter strings."""
        from transformation_portal.utils.security import SecurityError, build_ffmpeg_command

        input_file = tmp_path / "input.mp4"
        output_file = tmp_path / "output.mp4"
        input_file.write_bytes(b"fake video")

        with pytest.raises(SecurityError, match="dangerous characters"):
            build_ffmpeg_command(
                input_file,
                output_file,
                filters=["scale=1920; rm -rf /"],
                allowed_dirs=[tmp_path],
            )

    def test_build_ffmpeg_with_additional_args(self, tmp_path: Path) -> None:
        """FFmpeg command includes additional arguments."""
        from transformation_portal.utils.security import build_ffmpeg_command

        input_file = tmp_path / "input.mp4"
        output_file = tmp_path / "output.mp4"
        input_file.write_bytes(b"fake video")

        result = build_ffmpeg_command(
            input_file,
            output_file,
            additional_args=["-crf", "23", "-preset", "medium"],
            allowed_dirs=[tmp_path],
        )

        assert "-crf" in result
        assert "23" in result
        assert "-preset" in result
        assert "medium" in result

    def test_build_ffmpeg_skip_path_validation(self, tmp_path: Path) -> None:
        """FFmpeg command can skip path validation."""
        from transformation_portal.utils.security import build_ffmpeg_command

        # Non-existent files
        result = build_ffmpeg_command(
            Path("nonexistent_input.mp4"),
            Path("nonexistent_output.mp4"),
            validate_paths=False,
        )

        assert result[0] == "ffmpeg"


# =============================================================================
# Test validate_filter_graph
# =============================================================================


@pytest.mark.security
class TestValidateFilterGraph:
    """Tests for validate_filter_graph function."""

    def test_validate_simple_filter(self) -> None:
        """Simple filter graph passes validation."""
        from transformation_portal.utils.security import validate_filter_graph

        result = validate_filter_graph("scale=1920:1080")

        assert result == "scale=1920:1080"

    def test_validate_complex_filter(self) -> None:
        """Complex filter graphs with semicolons are rejected by security policy."""
        from transformation_portal.utils.security import SecurityError, validate_filter_graph

        # FFmpeg uses ; to chain filter graphs, but our security policy rejects
        # all shell metacharacters including ; for maximum safety
        filter_graph = "split[main][tmp];[tmp]crop=iw:ih/2:0:0,vflip[flip];[main][flip]overlay=0:H/2"

        with pytest.raises(SecurityError):
            validate_filter_graph(filter_graph)

    @pytest.mark.parametrize(
        "dangerous_filter",
        [
            "scale=1920; rm -rf /",
            "scale=1920 && cat /etc/passwd",
            "scale=$(whoami):1080",
            "scale=`id`:1080",
            "scale=1920|cat",
            "scale=1920>file",
            "scale=1920<file",
        ],
    )
    def test_validate_rejects_dangerous_filters(self, dangerous_filter: str) -> None:
        """Dangerous filter strings are rejected."""
        from transformation_portal.utils.security import SecurityError, validate_filter_graph

        with pytest.raises(SecurityError, match="dangerous characters"):
            validate_filter_graph(dangerous_filter)

    def test_validate_filter_shows_found_chars(self) -> None:
        """Error message shows which dangerous characters were found."""
        from transformation_portal.utils.security import SecurityError, validate_filter_graph

        with pytest.raises(SecurityError) as exc_info:
            validate_filter_graph("scale; rm -rf /")

        assert ";" in str(exc_info.value)


# =============================================================================
# Test timeout context manager
# =============================================================================


@pytest.mark.security
class TestTimeout:
    """Tests for timeout context manager."""

    @pytest.mark.skipif(sys.platform == "win32", reason="SIGALRM not available on Windows")
    def test_timeout_completes_within_time(self) -> None:
        """Operation completing within timeout succeeds."""
        from transformation_portal.utils.security import timeout

        with timeout(5):
            result = 1 + 1

        assert result == 2

    @pytest.mark.skipif(sys.platform == "win32", reason="SIGALRM not available on Windows")
    def test_timeout_raises_on_exceed(self) -> None:
        """Operation exceeding timeout raises TimeoutError."""
        from transformation_portal.utils.security import TimeoutError, timeout
        import time

        with pytest.raises(TimeoutError, match="exceeded.*timeout"):
            with timeout(1):
                time.sleep(5)

    @pytest.mark.skipif(sys.platform != "win32", reason="Only relevant on Windows")
    def test_timeout_raises_not_implemented_on_windows(self) -> None:
        """timeout() raises NotImplementedError on Windows."""
        from transformation_portal.utils.security import timeout

        with pytest.raises(NotImplementedError, match="Unix-only"):
            with timeout(5):
                pass

    @pytest.mark.skipif(sys.platform == "win32", reason="SIGALRM not available on Windows")
    def test_timeout_restores_signal_handler(self) -> None:
        """timeout() restores original signal handler after use."""
        from transformation_portal.utils.security import timeout

        original_handler = signal.signal(signal.SIGALRM, signal.SIG_DFL)
        signal.signal(signal.SIGALRM, original_handler)

        with timeout(5):
            pass

        current_handler = signal.signal(signal.SIGALRM, signal.SIG_DFL)
        signal.signal(signal.SIGALRM, current_handler)

        assert current_handler == original_handler


# =============================================================================
# Test deprecated alias
# =============================================================================


@pytest.mark.security
class TestDeprecatedAlias:
    """Tests for deprecated function aliases."""

    def test_validate_file_path_alias_exists(self) -> None:
        """Deprecated validate_file_path alias is available."""
        from transformation_portal.utils.security import validate_file_path

        assert callable(validate_file_path)

    def test_validate_file_path_alias_works(self, tmp_path: Path) -> None:
        """Deprecated alias still works for compatibility."""
        from transformation_portal.utils.security import validate_file_path

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        # Should work but may emit deprecation warning
        result = validate_file_path(test_file, allowed_dirs=[tmp_path])

        assert result.exists()


# =============================================================================
# Test constants
# =============================================================================


@pytest.mark.security
class TestSecurityConstants:
    """Tests for security module constants."""

    def test_image_extensions_defined(self) -> None:
        """IMAGE_EXTENSIONS constant is defined."""
        from transformation_portal.utils.security import IMAGE_EXTENSIONS

        assert ".jpg" in IMAGE_EXTENSIONS
        assert ".png" in IMAGE_EXTENSIONS
        assert ".tiff" in IMAGE_EXTENSIONS

    def test_video_extensions_defined(self) -> None:
        """VIDEO_EXTENSIONS constant is defined."""
        from transformation_portal.utils.security import VIDEO_EXTENSIONS

        assert ".mp4" in VIDEO_EXTENSIONS
        assert ".mov" in VIDEO_EXTENSIONS

    def test_config_extensions_defined(self) -> None:
        """CONFIG_EXTENSIONS constant is defined."""
        from transformation_portal.utils.security import CONFIG_EXTENSIONS

        assert ".yaml" in CONFIG_EXTENSIONS
        assert ".json" in CONFIG_EXTENSIONS

    def test_dangerous_shell_chars_defined(self) -> None:
        """DANGEROUS_SHELL_CHARS constant is defined."""
        from transformation_portal.utils.security import DANGEROUS_SHELL_CHARS

        assert ";" in DANGEROUS_SHELL_CHARS
        assert "|" in DANGEROUS_SHELL_CHARS
        assert "&" in DANGEROUS_SHELL_CHARS

    def test_max_size_constants(self) -> None:
        """Size limit constants are defined with reasonable values."""
        from transformation_portal.utils.security import MAX_CONFIG_SIZE, MAX_IMAGE_SIZE, MAX_VIDEO_SIZE

        assert MAX_IMAGE_SIZE == 100 * 1024 * 1024  # 100MB
        assert MAX_VIDEO_SIZE == 10 * 1024 * 1024 * 1024  # 10GB
        assert MAX_CONFIG_SIZE == 10 * 1024 * 1024  # 10MB
