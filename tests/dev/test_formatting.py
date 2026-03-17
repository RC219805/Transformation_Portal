"""Tests for the zero-diff formatting pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import mock

import pytest


class TestFormatFile:
    """Tests for format_file function."""

    def test_format_file_nonexistent_raises(self):
        """format_file should raise FileNotFoundError for non-existent files."""
        from transformation_portal.dev.formatting import format_file

        with pytest.raises(FileNotFoundError):
            format_file(Path("/nonexistent/file.py"))

    def test_format_file_skips_non_python(self):
        """format_file should skip non-Python files."""
        from transformation_portal.dev.formatting import format_file

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"some content")
            f.flush()
            path = Path(f.name)

        try:
            result = format_file(path)
            assert result is True
        finally:
            path.unlink()

    @mock.patch("transformation_portal.dev.formatting._tool_available")
    def test_format_file_handles_missing_tools(self, mock_available):
        """format_file should handle missing formatting tools gracefully."""
        from transformation_portal.dev.formatting import format_file

        mock_available.return_value = False

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"x=1")
            f.flush()
            path = Path(f.name)

        try:
            result = format_file(path)
            assert result is True  # Should succeed even without tools
        finally:
            path.unlink()


class TestWriteFormatted:
    """Tests for write_formatted function."""

    def test_write_formatted_creates_parent_dirs(self):
        """write_formatted should create parent directories."""
        from transformation_portal.dev.formatting import write_formatted

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "nested" / "file.py"

            write_formatted(path, "x = 1\n")

            assert path.exists()
            assert path.parent.exists()

    def test_write_formatted_writes_content(self):
        """write_formatted should write the specified content."""
        from transformation_portal.dev.formatting import write_formatted

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.py"
            content = "def hello():\n    pass\n"

            write_formatted(path, content)

            assert path.exists()
            # Content should be written (may be reformatted)
            written = path.read_text()
            assert "def hello" in written


class TestCheckFormatting:
    """Tests for check_formatting function."""

    def test_check_formatting_nonexistent_raises(self):
        """check_formatting should raise FileNotFoundError for non-existent files."""
        from transformation_portal.dev.formatting import check_formatting

        with pytest.raises(FileNotFoundError):
            check_formatting(Path("/nonexistent/file.py"))

    def test_check_formatting_non_python_returns_true(self):
        """check_formatting should return True for non-Python files."""
        from transformation_portal.dev.formatting import check_formatting

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"content")
            f.flush()
            path = Path(f.name)

        try:
            result = check_formatting(path)
            assert result is True
        finally:
            path.unlink()


class TestFormattedFileWriter:
    """Tests for FormattedFileWriter context manager."""

    def test_context_manager_writes_on_exit(self):
        """FormattedFileWriter should write content on successful exit."""
        from transformation_portal.dev.formatting import FormattedFileWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.py"

            with FormattedFileWriter(path) as writer:
                writer.write("def hello():\n")
                writer.write("    pass\n")

            assert path.exists()
            content = path.read_text()
            assert "def hello" in content

    def test_context_manager_handles_exception(self):
        """FormattedFileWriter should not write on exception."""
        from transformation_portal.dev.formatting import FormattedFileWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.py"

            try:
                with FormattedFileWriter(path) as writer:
                    writer.write("content")
                    raise ValueError("Test error")
            except ValueError:
                pass

            # File should not exist after exception
            assert not path.exists()


class TestFormatDirectory:
    """Tests for format_directory function."""

    def test_format_directory_not_a_directory_raises(self):
        """format_directory should raise for non-directories."""
        from transformation_portal.dev.formatting import format_directory

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = Path(f.name)

        try:
            with pytest.raises(NotADirectoryError):
                format_directory(path)
        finally:
            path.unlink()

    def test_format_directory_returns_counts(self):
        """format_directory should return formatted and failed counts."""
        from transformation_portal.dev.formatting import format_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create some Python files
            (tmppath / "file1.py").write_text("x=1\n")
            (tmppath / "file2.py").write_text("y=2\n")

            formatted, failed = format_directory(tmppath)

            assert formatted >= 0
            assert failed >= 0
            assert formatted + failed == 2


class TestConstants:
    """Tests for module constants."""

    def test_black_line_length_matches_pyproject(self):
        """BLACK_LINE_LENGTH should match pyproject.toml."""
        from transformation_portal.dev.formatting import BLACK_LINE_LENGTH

        # The repo standard is 127
        assert BLACK_LINE_LENGTH == 127
