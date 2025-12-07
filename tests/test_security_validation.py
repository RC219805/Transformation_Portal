"""
Tests for security validation utilities.

Validates path traversal protection, command injection prevention,
and FFmpeg security features.
"""

import pytest
from pathlib import Path

from transformation_portal.utils.security import (
    SecurityError,
    validate_filepath,
    validate_image_path,
    validate_video_path,
    sanitize_filename,
    build_safe_command,
    build_ffmpeg_command,
    validate_filter_graph,
    timeout,
    TimeoutError,
    MAX_IMAGE_SIZE,
    MAX_VIDEO_SIZE,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
)


class TestValidateFilepath:
    """Test validate_filepath function."""
    
    def test_valid_path_within_allowed_dir(self, tmp_path):
        """Test that valid path within allowed directory is accepted."""
        test_file = tmp_path / "test.jpg"
        test_file.touch()
        
        result = validate_filepath(test_file, [tmp_path])
        assert result.exists()
        assert result.is_relative_to(tmp_path)
    
    def test_path_traversal_blocked(self, tmp_path):
        """Test that path traversal attempts are blocked."""
        # Create a file outside allowed directory
        outside_dir = tmp_path.parent / "outside"
        outside_dir.mkdir(exist_ok=True)
        outside_file = outside_dir / "sensitive.txt"
        outside_file.write_text("secret")
        
        # Try to access it via path traversal
        traversal_path = tmp_path / ".." / "outside" / "sensitive.txt"
        
        with pytest.raises(SecurityError, match="outside allowed directories"):
            validate_filepath(traversal_path, [tmp_path])
    
    def test_absolute_path_outside_allowed(self, tmp_path):
        """Test that absolute paths outside allowed dirs are blocked."""
        # Try to access /etc/passwd
        passwd_path = Path("/etc/passwd")
        
        if passwd_path.exists():
            with pytest.raises(SecurityError, match="outside allowed directories"):
                validate_filepath(passwd_path, [tmp_path])
    
    def test_nonexistent_file_rejected_by_default(self, tmp_path):
        """Test that non-existent files are rejected when must_exist=True."""
        nonexistent = tmp_path / "does_not_exist.jpg"
        
        with pytest.raises(SecurityError, match="Cannot resolve path"):
            validate_filepath(nonexistent, [tmp_path], must_exist=True)
    
    def test_nonexistent_file_allowed_with_flag(self, tmp_path):
        """Test that non-existent files are allowed when must_exist=False."""
        nonexistent = tmp_path / "future_output.jpg"
        
        result = validate_filepath(nonexistent, [tmp_path], must_exist=False)
        assert result.parent.is_relative_to(tmp_path)
    
    def test_file_size_limit_enforced(self, tmp_path):
        """Test that file size limits are enforced."""
        large_file = tmp_path / "large.bin"
        large_file.write_bytes(b"x" * (10 * 1024 * 1024))  # 10MB
        
        # Should pass with 20MB limit
        validate_filepath(large_file, [tmp_path], max_file_size=20 * 1024 * 1024)
        
        # Should fail with 5MB limit
        with pytest.raises(SecurityError, match="exceeds size limit"):
            validate_filepath(large_file, [tmp_path], max_file_size=5 * 1024 * 1024)
    
    def test_extension_whitelist_enforced(self, tmp_path):
        """Test that extension whitelist is enforced."""
        jpg_file = tmp_path / "image.jpg"
        jpg_file.touch()
        
        # Should pass with .jpg in whitelist
        validate_filepath(jpg_file, [tmp_path], allowed_extensions=['.jpg', '.png'])
        
        # Should fail with only .png in whitelist
        with pytest.raises(SecurityError, match="not in whitelist"):
            validate_filepath(jpg_file, [tmp_path], allowed_extensions=['.png'])


class TestValidateImagePath:
    """Test validate_image_path convenience function."""
    
    def test_image_extensions_accepted(self, tmp_path):
        """Test that common image extensions are accepted."""
        for ext in ['.jpg', '.png', '.tif', '.tiff']:
            img_file = tmp_path / f"image{ext}"
            img_file.touch()
            result = validate_image_path(img_file, [tmp_path])
            assert result.exists()
    
    def test_non_image_extension_rejected(self, tmp_path):
        """Test that non-image extensions are rejected."""
        txt_file = tmp_path / "notanimage.txt"
        txt_file.touch()
        
        with pytest.raises(SecurityError, match="not in whitelist"):
            validate_image_path(txt_file, [tmp_path])


class TestValidateVideoPath:
    """Test validate_video_path convenience function."""
    
    def test_video_extensions_accepted(self, tmp_path):
        """Test that common video extensions are accepted."""
        for ext in ['.mp4', '.mov', '.avi']:
            vid_file = tmp_path / f"video{ext}"
            vid_file.touch()
            result = validate_video_path(vid_file, [tmp_path])
            assert result.exists()


class TestSanitizeFilename:
    """Test sanitize_filename function."""
    
    def test_remove_path_components(self):
        """Test that path components are removed."""
        result = sanitize_filename("../../../etc/passwd")
        assert "/" not in result
        assert ".." not in result
    
    def test_replace_dangerous_characters(self):
        """Test that dangerous characters are replaced."""
        result = sanitize_filename("file<script>.jpg")
        assert "<" not in result
        assert ">" not in result
        assert result == "file_script_.jpg"
    
    def test_preserve_extension(self):
        """Test that file extensions are preserved."""
        result = sanitize_filename("my file!@#.jpg")
        assert result.endswith(".jpg")
    
    def test_truncate_long_filenames(self):
        """Test that overly long filenames are truncated."""
        long_name = "a" * 300 + ".jpg"
        result = sanitize_filename(long_name, max_length=255)
        assert len(result) <= 255
        assert result.endswith(".jpg")


class TestBuildSafeCommand:
    """Test build_safe_command function."""
    
    def test_simple_command(self):
        """Test building a simple safe command."""
        cmd = build_safe_command("ls", ["-la", "/tmp"])
        assert cmd == ["ls", "-la", "/tmp"]
    
    def test_dangerous_characters_blocked(self):
        """Test that dangerous characters in arguments are blocked."""
        with pytest.raises(SecurityError, match="dangerous characters"):
            build_safe_command("cat", ["/etc/passwd; rm -rf /"])
        
        with pytest.raises(SecurityError, match="dangerous characters"):
            build_safe_command("echo", ["$(whoami)"])


class TestBuildFFmpegCommand:
    """Test build_ffmpeg_command function."""
    
    def test_simple_ffmpeg_command(self, tmp_path):
        """Test building a simple FFmpeg command."""
        input_file = tmp_path / "input.mp4"
        output_file = tmp_path / "output.mp4"
        input_file.touch()
        
        cmd = build_ffmpeg_command(
            input_file,
            output_file,
            filters=["scale=1920:1080"],
            validate_paths=False
        )
        
        assert cmd[0] == "ffmpeg"
        assert "-i" in cmd
        assert str(input_file) in cmd
        assert "-vf" in cmd
        assert "scale=1920:1080" in cmd
        assert str(output_file) in cmd
    
    def test_multiple_filters(self, tmp_path):
        """Test FFmpeg command with multiple filters."""
        input_file = tmp_path / "input.mp4"
        output_file = tmp_path / "output.mp4"
        input_file.touch()
        
        cmd = build_ffmpeg_command(
            input_file,
            output_file,
            filters=["scale=1920:1080", "fps=30"],
            validate_paths=False
        )
        
        # Filters should be comma-separated in -vf argument
        vf_index = cmd.index("-vf")
        filter_string = cmd[vf_index + 1]
        assert "scale=1920:1080,fps=30" == filter_string
    
    def test_dangerous_filter_blocked(self, tmp_path):
        """Test that dangerous filter strings are blocked."""
        input_file = tmp_path / "input.mp4"
        output_file = tmp_path / "output.mp4"
        input_file.touch()
        
        with pytest.raises(SecurityError, match="dangerous characters"):
            build_ffmpeg_command(
                input_file,
                output_file,
                filters=["scale=1920:1080; rm -rf /"],
                validate_paths=False
            )
    
    def test_additional_args(self, tmp_path):
        """Test FFmpeg command with additional arguments."""
        input_file = tmp_path / "input.mp4"
        output_file = tmp_path / "output.mp4"
        input_file.touch()
        
        cmd = build_ffmpeg_command(
            input_file,
            output_file,
            additional_args=["-preset", "fast"],
            validate_paths=False
        )
        
        assert "-preset" in cmd
        assert "fast" in cmd


class TestValidateFilterGraph:
    """Test validate_filter_graph function."""
    
    def test_safe_filter_accepted(self):
        """Test that safe filter graphs are accepted."""
        safe_filters = [
            "scale=1920:1080",
            "fps=30",
            "crop=1920:1080:0:0",
            "overlay=10:10",
        ]
        
        for filter_str in safe_filters:
            result = validate_filter_graph(filter_str)
            assert result == filter_str
    
    def test_dangerous_filter_rejected(self):
        """Test that dangerous filter strings are rejected."""
        dangerous_filters = [
            "scale=1920:1080; rm -rf /",
            "fps=30 | cat /etc/passwd",
            "crop=1920:1080 && echo pwned",
            "overlay=$(whoami)",
        ]
        
        for filter_str in dangerous_filters:
            with pytest.raises(SecurityError, match="dangerous characters"):
                validate_filter_graph(filter_str)


class TestTimeout:
    """Test timeout context manager."""
    
    @pytest.mark.skipif(
        not hasattr(__import__('signal'), 'SIGALRM'),
        reason="timeout() requires SIGALRM (Unix-only)"
    )
    def test_timeout_successful_operation(self):
        """Test that fast operations complete successfully."""
        import time
        
        with timeout(2):
            time.sleep(0.1)  # Fast operation
    
    @pytest.mark.skipif(
        not hasattr(__import__('signal'), 'SIGALRM'),
        reason="timeout() requires SIGALRM (Unix-only)"
    )
    def test_timeout_slow_operation(self):
        """Test that slow operations raise TimeoutError."""
        import time
        
        with pytest.raises(TimeoutError, match="exceeded"):
            with timeout(1):
                time.sleep(3)  # Slow operation
    
    def test_timeout_windows_raises_not_implemented(self):
        """Test that timeout raises NotImplementedError on Windows."""
        import signal
        
        if not hasattr(signal, 'SIGALRM'):
            with pytest.raises(NotImplementedError, match="Unix-only"):
                with timeout(1):
                    pass


class TestSecurityConstants:
    """Test that security constants are defined."""
    
    def test_file_size_limits_defined(self):
        """Test that file size limit constants are defined."""
        assert MAX_IMAGE_SIZE > 0
        assert MAX_VIDEO_SIZE > 0
        assert MAX_VIDEO_SIZE > MAX_IMAGE_SIZE  # Videos should have larger limit
    
    def test_extension_sets_defined(self):
        """Test that extension whitelist sets are defined."""
        assert len(IMAGE_EXTENSIONS) > 0
        assert len(VIDEO_EXTENSIONS) > 0
        assert '.jpg' in IMAGE_EXTENSIONS
        assert '.mp4' in VIDEO_EXTENSIONS


class TestSecurityIntegration:
    """Integration tests combining multiple security features."""
    
    def test_full_ffmpeg_workflow(self, tmp_path):
        """Test complete FFmpeg workflow with all security checks."""
        # Create test files
        input_file = tmp_path / "input.mp4"
        output_file = tmp_path / "output.mp4"
        input_file.touch()
        
        # Build safe command
        cmd = build_ffmpeg_command(
            input_file,
            output_file,
            filters=["scale=1920:1080"],
            codec="libx264",
            additional_args=["-preset", "fast"],
            validate_paths=True,
            allowed_dirs=[tmp_path]
        )
        
        # Verify command structure
        assert isinstance(cmd, list)
        assert cmd[0] == "ffmpeg"
        assert not any(";" in str(arg) for arg in cmd)  # No shell metacharacters
        
        # Command would be safe to run with subprocess.run(cmd, check=True)
        # (not actually running ffmpeg in test)
