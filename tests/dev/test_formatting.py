"""Tests for the zero-diff formatting pipeline."""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path
from unittest import mock

import pytest

pytestmark = pytest.mark.unit


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

    def test_write_formatted_preserves_existing_mode(self, tmp_path):
        """An executable script must stay executable after a rewrite."""
        from transformation_portal.dev import formatting

        path = tmp_path / "tool.py"
        path.write_text("print('hi')\n")
        path.chmod(0o755)

        with mock.patch.object(formatting, "format_file", return_value=True):
            formatting.write_formatted(path, "print('hello')\n")

        assert stat.S_IMODE(path.stat().st_mode) == 0o755
        assert path.read_text() == "print('hello')\n"

    def test_write_formatted_new_file_honours_umask(self, tmp_path):
        """A new file gets 0o666 & ~umask, not mkstemp's private 0o600."""
        from transformation_portal.dev import formatting

        path = tmp_path / "fresh.py"
        old_mask = os.umask(0o027)
        try:
            with mock.patch.object(formatting, "format_file", return_value=True):
                formatting.write_formatted(path, "x = 1\n")
        finally:
            os.umask(old_mask)

        assert stat.S_IMODE(path.stat().st_mode) == 0o640

    def test_write_formatted_raises_and_keeps_destination_on_formatter_failure(self, tmp_path):
        """A formatter failure must not publish the unformatted content."""
        from transformation_portal.dev import formatting

        path = tmp_path / "module.py"
        path.write_text("x = 1\n")

        with mock.patch.object(formatting, "format_file", return_value=False):
            with pytest.raises(RuntimeError, match="Formatting failed"):
                formatting.write_formatted(path, "x=2\n")

        assert path.read_text() == "x = 1\n"
        assert sorted(p.name for p in tmp_path.iterdir()) == ["module.py"]

    def test_write_canonical_raises_on_formatter_failure(self, tmp_path):
        """write_canonical shares the fail-closed contract of write_formatted."""
        from transformation_portal.dev import formatting

        path = tmp_path / "gen.py"
        with mock.patch.object(formatting, "format_file", return_value=False):
            with pytest.raises(RuntimeError, match="Formatting failed"):
                formatting.write_canonical(path, "x = {'b': 1, 'a': 2}\n")

        assert not path.exists()
        assert list(tmp_path.iterdir()) == []


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

    def test_format_directory_skips_hidden_and_vendored_trees(self, tmp_path):
        """Virtualenvs, hidden dirs and vendored trees must never be rewritten."""
        from transformation_portal.dev import formatting

        kept = [tmp_path / "src" / "a.py", tmp_path / "b.py"]
        skipped = [
            tmp_path / ".venv" / "lib" / "site-packages" / "dep" / "mod.py",
            tmp_path / "venv" / "lib" / "mod.py",
            tmp_path / ".git" / "hooks" / "hook.py",
            tmp_path / "node_modules" / "pkg" / "gen.py",
            tmp_path / "src" / "__pycache__" / "a.py",
            tmp_path / "build" / "lib" / "a.py",
            tmp_path / "src" / ".hidden.py",
        ]
        for f in kept + skipped:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("x=1\n")

        seen: list[Path] = []

        def fake_format(path, *, quiet=True):
            seen.append(path)
            return True

        with mock.patch.object(formatting, "format_file", side_effect=fake_format):
            assert formatting.format_directory(tmp_path) == (2, 0)

        assert sorted(seen) == sorted(kept)

    def test_format_directory_allows_hidden_root(self, tmp_path):
        """Only components below the formatted directory are filtered."""
        from transformation_portal.dev import formatting

        root = tmp_path / ".config_scripts"
        root.mkdir()
        (root / "a.py").write_text("x=1\n")

        with mock.patch.object(formatting, "format_file", return_value=True):
            assert formatting.format_directory(root) == (1, 0)


class TestFormatFileSubprocesses:
    """format_file runs each formatter once per file."""

    def test_black_runs_once_per_file(self, tmp_path):
        from transformation_portal.dev import formatting

        path = tmp_path / "m.py"
        path.write_text("x=1\n")

        with (
            mock.patch.object(formatting, "_tool_available", return_value=True),
            mock.patch.object(formatting.subprocess, "run") as run,
        ):
            assert formatting.format_file(path) is True

        tools = [call.args[0][0] for call in run.call_args_list]
        assert tools == ["isort", "black"]


class TestConstants:
    """Tests for module constants."""

    def test_black_line_length_matches_pyproject(self):
        """BLACK_LINE_LENGTH should match pyproject.toml."""
        from transformation_portal.dev.formatting import BLACK_LINE_LENGTH

        # The repo standard is 127
        assert BLACK_LINE_LENGTH == 127
