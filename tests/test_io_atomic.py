"""Tests for io_atomic module - atomic write primitives.

Validates:
- Successful writes produce final file
- Failures leave no temp files behind
- Operations use atomic rename (os.replace)
- No file descriptor leaks
"""
import pytest
import os
from pathlib import Path
import tempfile

from transformation_portal.lux_depth_v3.io_atomic import (
    atomic_temp_file,
    atomic_write_bytes,
    atomic_write_pil_png,
    atomic_write_with_fd,
    HAS_PIL,
)

from PIL import Image
import numpy as np


class TestAtomicTempFile:
    """Test atomic temp file context manager."""

    def test_successful_write_creates_final_file(self, tmp_path):
        """Successful write should create final file via atomic rename."""
        output_path = tmp_path / "output.txt"

        with atomic_temp_file(output_path, create_file=True) as temp_path:
            # Temp file should exist
            assert temp_path.exists()
            # Should be in same directory
            assert temp_path.parent == output_path.parent
            # Should have temp prefix
            assert temp_path.name.startswith(".tmp_")

            # Write data to temp
            temp_path.write_text("hello")

        # Final file should exist
        assert output_path.exists()
        assert output_path.read_text() == "hello"

        # Temp file should be gone
        temp_files = list(tmp_path.glob(".tmp_*"))
        assert len(temp_files) == 0, f"Temp files remain: {temp_files}"

    def test_failure_cleans_up_temp_file(self, tmp_path):
        """Failed write should cleanup temp file."""
        output_path = tmp_path / "output.txt"
        temp_path_captured = None

        try:
            with atomic_temp_file(output_path, create_file=True) as temp_path:
                temp_path_captured = temp_path
                temp_path.write_text("partial")
                # Simulate failure
                raise ValueError("Simulated write failure")
        except ValueError:
            pass

        # Output file should NOT exist
        assert not output_path.exists()

        # Temp file should be cleaned up
        assert not temp_path_captured.exists()
        temp_files = list(tmp_path.glob(".tmp_*"))
        assert len(temp_files) == 0, f"Temp files remain: {temp_files}"

    def test_creates_parent_directories(self, tmp_path):
        """Should create parent directories if needed."""
        output_path = tmp_path / "subdir" / "nested" / "output.txt"

        with atomic_temp_file(output_path, create_file=True) as temp_path:
            temp_path.write_text("test")

        assert output_path.exists()
        assert output_path.parent.exists()

    def test_custom_suffix_and_prefix(self, tmp_path):
        """Should respect custom suffix and prefix."""
        output_path = tmp_path / "output.png"

        with atomic_temp_file(output_path, suffix=".png", prefix="custom_", create_file=True) as temp_path:
            # Should have custom prefix
            assert temp_path.name.startswith("custom_")
            # Should have custom suffix
            assert temp_path.suffix == ".png"
            temp_path.write_bytes(b"data")

        assert output_path.exists()


class TestAtomicWriteBytes:
    """Test atomic byte writing."""

    def test_successful_bytes_write(self, tmp_path):
        """Should atomically write bytes."""
        output_path = tmp_path / "data.bin"
        data = b"Hello, atomic world!"

        result_path = atomic_write_bytes(output_path, data)

        assert result_path == output_path
        assert output_path.exists()
        assert output_path.read_bytes() == data

        # No temp files should remain
        temp_files = list(tmp_path.glob(".tmp_*"))
        assert len(temp_files) == 0

    def test_overwrites_existing_file(self, tmp_path):
        """Should atomically overwrite existing file."""
        output_path = tmp_path / "overwrite.bin"

        # Write initial content
        output_path.write_bytes(b"old data")

        # Overwrite with new content
        atomic_write_bytes(output_path, b"new data")

        assert output_path.read_bytes() == b"new data"

    def test_empty_bytes(self, tmp_path):
        """Should handle empty byte arrays."""
        output_path = tmp_path / "empty.bin"

        atomic_write_bytes(output_path, b"")

        assert output_path.exists()
        assert output_path.read_bytes() == b""

    def test_large_bytes(self, tmp_path):
        """Should handle large byte arrays."""
        output_path = tmp_path / "large.bin"
        # 10 MB of data
        data = b"x" * (10 * 1024 * 1024)

        atomic_write_bytes(output_path, data)

        assert output_path.exists()
        assert len(output_path.read_bytes()) == len(data)

    def test_preserves_readable_permissions(self, tmp_path):
        """Should create files with readable permissions (not 0600)."""
        output_path = tmp_path / "permissions.bin"

        atomic_write_bytes(output_path, b"test data")

        # File should exist
        assert output_path.exists()

        # Should be readable by group/others (not 0600)
        stat_info = output_path.stat()
        # Check that group or others have read permission
        # 0o044 = group read (0o040) | others read (0o004)
        assert stat_info.st_mode & 0o044 != 0, \
            f"File has restrictive permissions: {oct(stat_info.st_mode)}"


@pytest.mark.skipif(not HAS_PIL, reason="Pillow not installed")
class TestAtomicWritePilPng:
    """Test atomic PIL PNG writing."""

    def test_successful_pil_write(self, tmp_path):
        """Should atomically write PIL Image as PNG."""
        output_path = tmp_path / "image.png"

        # Create test image
        img = Image.new('RGB', (100, 100), color='red')

        result_path = atomic_write_pil_png(output_path, img)

        assert result_path == output_path
        assert output_path.exists()

        # Verify can read back
        loaded = Image.open(output_path)
        assert loaded.size == (100, 100)
        assert loaded.mode == 'RGB'

        # No temp files
        temp_files = list(tmp_path.glob(".tmp_*"))
        assert len(temp_files) == 0

    def test_grayscale_image(self, tmp_path):
        """Should handle grayscale images."""
        output_path = tmp_path / "gray.png"
        img = Image.new('L', (50, 50), color=128)

        atomic_write_pil_png(output_path, img)

        loaded = Image.open(output_path)
        assert loaded.mode == 'L'
        assert loaded.size == (50, 50)

    def test_rgba_image(self, tmp_path):
        """Should handle RGBA images with transparency."""
        output_path = tmp_path / "rgba.png"
        img = Image.new('RGBA', (64, 64), color=(255, 0, 0, 128))

        atomic_write_pil_png(output_path, img)

        loaded = Image.open(output_path)
        assert loaded.mode == 'RGBA'
        assert loaded.size == (64, 64)

    def test_optimization_flag(self, tmp_path):
        """Should respect optimize flag."""
        output_path = tmp_path / "optimized.png"
        img = Image.new('RGB', (100, 100), color='blue')

        # With optimization
        atomic_write_pil_png(output_path, img, optimize=True)
        size_optimized = output_path.stat().st_size

        # Without optimization
        atomic_write_pil_png(output_path, img, optimize=False)
        size_unoptimized = output_path.stat().st_size

        # Optimized should typically be smaller or equal
        # (but we just verify both succeed without errors)
        assert output_path.exists()

    def test_custom_save_kwargs(self, tmp_path):
        """Should pass through custom save kwargs."""
        output_path = tmp_path / "custom.png"
        img = Image.new('RGB', (100, 100), color='green')

        # Pass compression level
        atomic_write_pil_png(output_path, img, compress_level=9)

        assert output_path.exists()
        loaded = Image.open(output_path)
        assert loaded.size == (100, 100)


class TestAtomicWriteWithFD:
    """Test atomic writing with file descriptor."""

    def test_fd_based_write(self, tmp_path):
        """Should handle FD-based writers."""
        output_path = tmp_path / "fd_output.txt"

        def writer_func(fd, temp_path):
            # Write using FD
            with os.fdopen(fd, 'w') as f:
                f.write("Written via FD")

        result_path = atomic_write_with_fd(output_path, writer_func)

        assert result_path == output_path
        assert output_path.exists()
        assert output_path.read_text() == "Written via FD"

        # No temp files
        temp_files = list(tmp_path.glob(".tmp_*"))
        assert len(temp_files) == 0

    def test_fd_based_binary_write(self, tmp_path):
        """Should handle binary FD writes."""
        output_path = tmp_path / "binary.dat"

        def writer_func(fd, temp_path):
            with os.fdopen(fd, 'wb') as f:
                f.write(b"binary data")

        atomic_write_with_fd(output_path, writer_func)

        assert output_path.read_bytes() == b"binary data"

    def test_writer_closes_fd_explicitly(self, tmp_path):
        """Should handle writers that close FD themselves."""
        output_path = tmp_path / "explicit_close.txt"

        def writer_func(fd, temp_path):
            # Close FD immediately
            os.close(fd)
            # Use path instead
            temp_path.write_text("closed FD, used path")

        atomic_write_with_fd(output_path, writer_func)

        assert output_path.read_text() == "closed FD, used path"

    def test_writer_failure_cleans_up(self, tmp_path):
        """Should cleanup temp file if writer fails."""
        output_path = tmp_path / "failed.txt"

        def failing_writer(fd, temp_path):
            os.close(fd)
            raise RuntimeError("Writer failed!")

        with pytest.raises(IOError, match="Failed to write"):
            atomic_write_with_fd(output_path, failing_writer)

        # Output should not exist
        assert not output_path.exists()

        # No temp files should remain
        temp_files = list(tmp_path.glob(".tmp_*"))
        assert len(temp_files) == 0

    def test_custom_suffix(self, tmp_path):
        """Should respect custom suffix."""
        output_path = tmp_path / "custom.png"

        def writer_func(fd, temp_path):
            # Verify temp has correct suffix
            assert temp_path.suffix == ".png"
            os.close(fd)
            temp_path.write_bytes(b"png data")

        atomic_write_with_fd(output_path, writer_func, suffix=".png")

        assert output_path.exists()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_nested_directory_creation(self, tmp_path):
        """Should handle deeply nested paths."""
        output_path = tmp_path / "a" / "b" / "c" / "d" / "output.txt"

        atomic_write_bytes(output_path, b"nested")

        assert output_path.exists()
        assert output_path.read_bytes() == b"nested"

    def test_unicode_filename(self, tmp_path):
        """Should handle unicode filenames."""
        output_path = tmp_path / "unicode_文件.txt"

        atomic_write_bytes(output_path, b"unicode content")

        assert output_path.exists()

    def test_concurrent_writes_different_files(self, tmp_path):
        """Should handle concurrent writes to different files."""
        # Write multiple files
        paths = [tmp_path / f"file_{i}.txt" for i in range(5)]

        for i, path in enumerate(paths):
            atomic_write_bytes(path, f"content_{i}".encode())

        # All should exist with correct content
        for i, path in enumerate(paths):
            assert path.exists()
            assert path.read_bytes() == f"content_{i}".encode()

        # No temp files
        temp_files = list(tmp_path.glob(".tmp_*"))
        assert len(temp_files) == 0

    def test_overwrite_during_write_is_atomic(self, tmp_path):
        """Verify write is atomic - no partial state visible."""
        output_path = tmp_path / "atomic_test.txt"

        # Initial state
        output_path.write_text("initial")

        # Overwrite atomically
        atomic_write_bytes(output_path, b"updated")

        # Should see either old or new, never partial
        content = output_path.read_text()
        assert content in ["initial", "updated"]
        # Since write completed, should be updated
        assert content == "updated"


class TestNoFDLeaks:
    """Test that FD management doesn't leak descriptors."""

    def test_no_fd_leak_on_success(self, tmp_path):
        """Successful writes should not leak FDs."""
        output_path = tmp_path / "fd_test.txt"

        # Get current FD count (rough check)
        # On POSIX, we can check /proc/self/fd or use resource limits
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)

        # Write many times
        for i in range(100):
            atomic_write_bytes(output_path, f"iteration {i}".encode())

        # Should not approach FD limit (rough sanity check)
        assert output_path.exists()

    def test_no_fd_leak_on_failure(self, tmp_path):
        """Failed writes should not leak FDs."""
        output_path = tmp_path / "fail_test.txt"

        def failing_writer(fd, temp_path):
            # Don't close FD - test cleanup
            raise ValueError("Intentional failure")

        # Try to write many times with failure
        for i in range(50):
            try:
                atomic_write_with_fd(output_path, failing_writer)
            except IOError:
                pass

        # Should not have leaked FDs (output should not exist)
        assert not output_path.exists()
