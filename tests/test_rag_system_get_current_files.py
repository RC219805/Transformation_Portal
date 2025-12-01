"""
Tests for RAGSystem._get_current_files() method.

Tests cache validation logic to ensure:
1. Only configured index_directories are scanned
2. Include patterns correctly match files
3. Exclude patterns properly filter out files
4. Default behavior when index_directories is empty
5. ValueError handling from relative_to() for files outside repo_root
"""

# pylint: disable=redefined-outer-name  # pytest fixtures pattern

import sys
import tempfile
from pathlib import Path

import pytest

# Add agents directory to path
agents_path = Path(__file__).parent.parent / '.github' / 'agents'
sys.path.insert(0, str(agents_path))

# pylint: disable=wrong-import-position
from rag_system.phase1_integration import RAGConfig, RAGSystem  # noqa: E402
# pylint: enable=wrong-import-position


@pytest.fixture
def temp_repo_with_structure():
    """Create a temporary repository with a known file structure for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = Path(tmpdir)

        # Create directory structure similar to real repo
        # docs/ directory
        docs_dir = repo_root / "docs"
        docs_dir.mkdir()
        (docs_dir / "readme.md").write_text("# Readme\nDocumentation here.")
        (docs_dir / "guide.txt").write_text("Guide content")
        (docs_dir / "notes.py").write_text("# Python in docs")

        # src/ directory (not in default index_directories)
        src_dir = repo_root / "src"
        src_dir.mkdir()
        (src_dir / "main.py").write_text("def main(): pass")
        (src_dir / "utils.py").write_text("def util(): pass")

        # tests/ directory
        tests_dir = repo_root / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_main.py").write_text("def test_main(): pass")
        (tests_dir / "test_utils.py").write_text("def test_utils(): pass")

        # config/ directory
        config_dir = repo_root / "config"
        config_dir.mkdir()
        (config_dir / "settings.yaml").write_text("key: value")
        (config_dir / "options.json").write_text('{"option": true}')

        # Root level files
        (repo_root / "README.md").write_text("# Root readme")
        (repo_root / "setup.py").write_text("# Setup script")

        # Excluded directories
        cache_dir = repo_root / ".rag_cache"
        cache_dir.mkdir()
        (cache_dir / "cached.py").write_text("# Should be excluded")

        pycache_dir = repo_root / "__pycache__"
        pycache_dir.mkdir()
        (pycache_dir / "module.pyc").write_text("# Should be excluded")

        deprecated_dir = repo_root / "deprecated"
        deprecated_dir.mkdir()
        (deprecated_dir / "old_code.py").write_text("# Should be excluded")

        yield repo_root


class TestGetCurrentFilesIndexDirectories:
    """Test that _get_current_files respects index_directories configuration."""

    def test_scans_only_configured_directories(self, temp_repo_with_structure):
        """Test that only configured index_directories are scanned."""
        config = RAGConfig(
            repo_root=str(temp_repo_with_structure),
            index_directories=["docs/", "config/"],
            include_patterns=["*.py", "*.md", "*.yaml", "*.json", "*.txt"],
            exclude_patterns=[],
        )
        rag = RAGSystem(config)

        files = rag._get_current_files()

        # Should find files in docs/ and config/
        assert "docs/readme.md" in files
        assert "docs/guide.txt" in files
        assert "docs/notes.py" in files
        assert "config/settings.yaml" in files
        assert "config/options.json" in files

        # Should NOT find files in src/ or tests/ (not in index_directories)
        assert "src/main.py" not in files
        assert "src/utils.py" not in files
        assert "tests/test_main.py" not in files
        assert "tests/test_utils.py" not in files

        # Should NOT find root level files
        assert "README.md" not in files
        assert "setup.py" not in files

    def test_empty_index_directories_scans_repo_root(self, temp_repo_with_structure):
        """Test that empty index_directories defaults to scanning repo_root."""
        config = RAGConfig(
            repo_root=str(temp_repo_with_structure),
            index_directories=[],
            include_patterns=["*.py", "*.md"],
            exclude_patterns=["deprecated/*", ".rag_cache/*", "__pycache__/*"],
        )
        rag = RAGSystem(config)

        files = rag._get_current_files()

        # Should find files throughout the repo (except excluded)
        assert "docs/readme.md" in files
        assert "docs/notes.py" in files
        assert "src/main.py" in files
        assert "tests/test_main.py" in files
        assert "README.md" in files
        assert "setup.py" in files

    def test_nonexistent_index_directory_is_skipped(self, temp_repo_with_structure):
        """Test that non-existent directories in index_directories are skipped."""
        config = RAGConfig(
            repo_root=str(temp_repo_with_structure),
            index_directories=["docs/", "nonexistent/", "config/"],
            include_patterns=["*.py", "*.md", "*.yaml"],
            exclude_patterns=[],
        )
        rag = RAGSystem(config)

        files = rag._get_current_files()

        # Should still find files from existing directories
        assert "docs/readme.md" in files
        assert "config/settings.yaml" in files
        # No error should be raised


class TestGetCurrentFilesIncludePatterns:
    """Test that _get_current_files respects include_patterns."""

    def test_include_patterns_match_correctly(self, temp_repo_with_structure):
        """Test that include patterns correctly match files."""
        config = RAGConfig(
            repo_root=str(temp_repo_with_structure),
            index_directories=["docs/"],
            include_patterns=["*.md"],  # Only markdown files
            exclude_patterns=[],
        )
        rag = RAGSystem(config)

        files = rag._get_current_files()

        # Should find markdown files
        assert "docs/readme.md" in files

        # Should NOT find non-markdown files
        assert "docs/guide.txt" not in files
        assert "docs/notes.py" not in files

    def test_multiple_include_patterns(self, temp_repo_with_structure):
        """Test multiple include patterns work correctly."""
        config = RAGConfig(
            repo_root=str(temp_repo_with_structure),
            index_directories=["docs/", "config/"],
            include_patterns=["*.md", "*.yaml"],
            exclude_patterns=[],
        )
        rag = RAGSystem(config)

        files = rag._get_current_files()

        # Should find matching files
        assert "docs/readme.md" in files
        assert "config/settings.yaml" in files

        # Should NOT find non-matching files
        assert "docs/guide.txt" not in files
        assert "docs/notes.py" not in files
        assert "config/options.json" not in files


class TestGetCurrentFilesExcludePatterns:
    """Test that _get_current_files respects exclude_patterns."""

    def test_exclude_patterns_filter_files(self, temp_repo_with_structure):
        """Test that exclude patterns properly filter out files."""
        config = RAGConfig(
            repo_root=str(temp_repo_with_structure),
            index_directories=[],  # Scan entire repo
            include_patterns=["*.py"],
            exclude_patterns=["deprecated/*", ".rag_cache/*", "__pycache__/*"],
        )
        rag = RAGSystem(config)

        files = rag._get_current_files()

        # Should find regular Python files
        assert "docs/notes.py" in files
        assert "src/main.py" in files
        assert "setup.py" in files

        # Should NOT find files in excluded directories
        assert "deprecated/old_code.py" not in files
        assert ".rag_cache/cached.py" not in files
        # Note: __pycache__/*.pyc won't match *.py anyway

    def test_exclude_pattern_with_specific_file(self, temp_repo_with_structure):
        """Test exclude pattern can target specific files."""
        config = RAGConfig(
            repo_root=str(temp_repo_with_structure),
            index_directories=["docs/"],
            include_patterns=["*.md", "*.txt"],
            exclude_patterns=["docs/guide.txt"],
        )
        rag = RAGSystem(config)

        files = rag._get_current_files()

        # Should find readme.md
        assert "docs/readme.md" in files

        # Should NOT find excluded file
        assert "docs/guide.txt" not in files


class TestGetCurrentFilesEdgeCases:
    """Test edge cases for _get_current_files."""

    def test_handles_nested_directories(self, temp_repo_with_structure):
        """Test that nested directories are scanned correctly."""
        # Create nested structure
        nested_dir = temp_repo_with_structure / "docs" / "api" / "v1"
        nested_dir.mkdir(parents=True)
        (nested_dir / "spec.md").write_text("# API Spec")

        config = RAGConfig(
            repo_root=str(temp_repo_with_structure),
            index_directories=["docs/"],
            include_patterns=["*.md"],
            exclude_patterns=[],
        )
        rag = RAGSystem(config)

        files = rag._get_current_files()

        # Should find nested files
        assert "docs/api/v1/spec.md" in files
        assert "docs/readme.md" in files

    def test_handles_symlinks_gracefully(self, temp_repo_with_structure):
        """Test that symlinks don't cause issues."""
        # Create a symlink (if platform supports it)
        try:
            link_path = temp_repo_with_structure / "docs" / "link.md"
            target_path = temp_repo_with_structure / "README.md"
            link_path.symlink_to(target_path)
        except (OSError, NotImplementedError):
            pytest.skip("Platform doesn't support symlinks")

        config = RAGConfig(
            repo_root=str(temp_repo_with_structure),
            index_directories=["docs/"],
            include_patterns=["*.md"],
            exclude_patterns=[],
        )
        rag = RAGSystem(config)

        # Should not raise an error
        files = rag._get_current_files()

        # Regular files should still be found
        assert "docs/readme.md" in files

    def test_empty_directory_returns_empty_dict(self):
        """Test that empty directory returns empty dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RAGConfig(
                repo_root=tmpdir,
                index_directories=["nonexistent/"],
                include_patterns=["*.py"],
                exclude_patterns=[],
            )
            rag = RAGSystem(config)

            files = rag._get_current_files()

            assert files == {}

    def test_returns_path_objects_as_values(self, temp_repo_with_structure):
        """Test that returned dict values are Path objects."""
        config = RAGConfig(
            repo_root=str(temp_repo_with_structure),
            index_directories=["docs/"],
            include_patterns=["*.md"],
            exclude_patterns=[],
        )
        rag = RAGSystem(config)

        files = rag._get_current_files()

        # All values should be Path objects
        for rel_path, full_path in files.items():
            assert isinstance(rel_path, str)
            assert isinstance(full_path, Path)
            assert full_path.exists()

    def test_relative_paths_are_consistent(self, temp_repo_with_structure):
        """Test that relative paths are normalized correctly."""
        config = RAGConfig(
            repo_root=str(temp_repo_with_structure),
            index_directories=["docs/"],  # With trailing slash
            include_patterns=["*.md"],
            exclude_patterns=[],
        )
        rag = RAGSystem(config)

        files = rag._get_current_files()

        # Paths should be relative to repo_root without leading slashes
        for rel_path in files.keys():
            assert not rel_path.startswith("/")
            assert not rel_path.startswith("./")


class TestGetCurrentFilesIntegration:
    """Integration tests for _get_current_files with cache validation."""

    def test_matches_indexer_behavior(self, temp_repo_with_structure):
        """Test that _get_current_files matches what would be indexed."""
        index_dirs = ["docs/", "tests/"]
        include_patterns = ["*.py", "*.md"]
        exclude_patterns = ["deprecated/*"]

        config = RAGConfig(
            repo_root=str(temp_repo_with_structure),
            index_directories=index_dirs,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )
        rag = RAGSystem(config)

        files = rag._get_current_files()

        # Should match expected files
        expected_files = {
            "docs/readme.md",
            "docs/notes.py",
            "tests/test_main.py",
            "tests/test_utils.py",
        }

        assert set(files.keys()) == expected_files

    def test_consistency_across_multiple_calls(self, temp_repo_with_structure):
        """Test that multiple calls return consistent results."""
        config = RAGConfig(
            repo_root=str(temp_repo_with_structure),
            index_directories=["docs/"],
            include_patterns=["*.md"],
            exclude_patterns=[],
        )
        rag = RAGSystem(config)

        files1 = rag._get_current_files()
        files2 = rag._get_current_files()

        assert files1.keys() == files2.keys()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
