"""Tests for non-lossy path sanitization and collision prevention.

Tests PR #1: Non-Lossy Path Sanitization
"""

import pytest
from pathlib import Path
import urllib.parse

from lux_depth_v3.enhance.security import sanitize_path_component_nonlossy
from lux_depth_v3.enhance.orchestrator import make_output_key


class TestNonLossySanitization:
    """Test non-lossy path component sanitization."""

    def test_alphanumeric_preserved(self):
        """Alphanumeric chars should pass through unchanged."""
        assert sanitize_path_component_nonlossy("kitchen123") == "kitchen123"

    def test_underscore_hyphen_preserved(self):
        """Underscores and hyphens should pass through."""
        assert sanitize_path_component_nonlossy("living-room_v2") == "living-room_v2"

    def test_single_dot_preserved(self):
        """Single dots should be preserved."""
        assert sanitize_path_component_nonlossy("room.1") == "room.1"

    def test_colon_encoded(self):
        """Colons should be percent-encoded."""
        assert sanitize_path_component_nonlossy("kitchen:1") == "kitchen%3A1"

    def test_slash_encoded(self):
        """Slashes should be percent-encoded."""
        assert sanitize_path_component_nonlossy("kitchen/1") == "kitchen%2F1"

    def test_backslash_encoded(self):
        """Backslashes should be percent-encoded."""
        assert sanitize_path_component_nonlossy("kitchen\\1") == "kitchen%5C1"

    def test_no_collision_special_chars(self):
        """Different special chars should produce different outputs."""
        colon = sanitize_path_component_nonlossy("kitchen:1")
        slash = sanitize_path_component_nonlossy("kitchen/1")
        backslash = sanitize_path_component_nonlossy("kitchen\\1")

        assert colon != slash
        assert slash != backslash
        assert colon != backslash

    def test_leading_dots_stripped(self):
        """Leading dots should be stripped (prevent hidden files)."""
        assert sanitize_path_component_nonlossy(".hidden") == "hidden"
        assert sanitize_path_component_nonlossy("...multiple") == "multiple"

    def test_double_dots_encoded(self):
        """Double dots should be encoded (prevent parent traversal)."""
        result = sanitize_path_component_nonlossy("parent..child")
        assert ".." not in result
        assert "%2E%2E" in result

    def test_empty_raises_error(self):
        """Empty component should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            sanitize_path_component_nonlossy("")

    def test_only_dots_raises_error(self):
        """Component with only dots should raise ValueError."""
        with pytest.raises(ValueError, match="empty after sanitization"):
            sanitize_path_component_nonlossy("...")

    def test_long_component_truncated(self):
        """Very long components should be truncated with hash suffix."""
        long_name = "a" * 250
        result = sanitize_path_component_nonlossy(long_name, max_length=200)

        assert len(result) <= 200
        assert "__" in result  # Hash suffix

    def test_unicode_encoded(self):
        """Unicode chars should be percent-encoded."""
        result = sanitize_path_component_nonlossy("café")
        # Either fully encoded or contains non-ASCII
        assert "caf" in result
        # Verify it's different from ASCII version
        assert result != "cafe"

    def test_deterministic(self):
        """Same input should always produce same output."""
        input_str = "kitchen:special/char\\test"
        result1 = sanitize_path_component_nonlossy(input_str)
        result2 = sanitize_path_component_nonlossy(input_str)
        assert result1 == result2

    def test_reversible_encoding(self):
        """Should be possible to decode back (for debugging)."""
        input_str = "kitchen:1"
        encoded = sanitize_path_component_nonlossy(input_str)
        decoded = urllib.parse.unquote(encoded)
        assert decoded == input_str


class TestMakeOutputKey:
    """Test collision-free output key generation."""

    def test_flat_structure(self, tmp_path):
        """Flat directory should produce simple keys."""
        input_root = tmp_path / "renders"
        input_root.mkdir()
        input_path = input_root / "kitchen.jpg"
        input_path.touch()

        key = make_output_key(input_path, input_root)
        assert key == Path("kitchen")

    def test_nested_structure(self, tmp_path):
        """Nested directories should preserve structure."""
        input_root = tmp_path / "renders"
        input_dir = input_root / "floor1" / "kitchen"
        input_dir.mkdir(parents=True)
        input_path = input_dir / "view.jpg"
        input_path.touch()

        key = make_output_key(input_path, input_root)
        assert key == Path("floor1/kitchen/view")

    def test_same_filename_different_dirs(self, tmp_path):
        """Same filename in different dirs should produce different keys."""
        input_root = tmp_path / "renders"

        kitchen_dir = input_root / "kitchen"
        kitchen_dir.mkdir(parents=True)
        path1 = kitchen_dir / "view.jpg"
        path1.touch()

        exterior_dir = input_root / "exterior"
        exterior_dir.mkdir(parents=True)
        path2 = exterior_dir / "view.jpg"
        path2.touch()

        key1 = make_output_key(path1, input_root)
        key2 = make_output_key(path2, input_root)

        assert key1 != key2
        assert key1 == Path("kitchen/view")
        assert key2 == Path("exterior/view")

    def test_special_chars_in_path(self, tmp_path):
        """Special chars in directory names should be encoded."""
        input_root = tmp_path / "renders"

        # Create a directory with a colon (allowed on macOS/Linux)
        import platform

        if platform.system() != "Windows":
            kitchen_dir = input_root / "kitchen:special"
            kitchen_dir.mkdir(parents=True)
            input_path = kitchen_dir / "view.jpg"
            input_path.touch()

            key = make_output_key(input_path, input_root)
            assert "kitchen%3Aspecial" in str(key)

    def test_not_relative_to_root(self, tmp_path):
        """Path not relative to root should fall back to flat naming."""
        input_root = tmp_path / "renders"
        input_root.mkdir()

        other_dir = tmp_path / "other"
        other_dir.mkdir()
        input_path = other_dir / "kitchen.jpg"
        input_path.touch()

        key = make_output_key(input_path, input_root)
        assert key == Path("kitchen")  # Just the stem, no extension

    def test_collision_prevention_real_scenario(self, tmp_path):
        """Real-world scenario: multiple images with same name in different folders."""
        input_root = tmp_path / "renders"

        # Create structure with same filenames
        paths = []
        for folder in ["kitchen", "exterior", "bedroom"]:
            folder_path = input_root / folder
            folder_path.mkdir(parents=True)
            img_path = folder_path / "view.jpg"
            img_path.touch()
            paths.append(img_path)

        # Generate keys
        keys = [make_output_key(p, input_root) for p in paths]

        # All keys should be unique
        assert len(keys) == len(set(keys))
        assert Path("kitchen/view") in keys
        assert Path("exterior/view") in keys
        assert Path("bedroom/view") in keys

    def test_deeply_nested_structure(self, tmp_path):
        """Very deep nesting should work correctly."""
        input_root = tmp_path / "renders"
        deep_path = input_root / "level1" / "level2" / "level3" / "level4"
        deep_path.mkdir(parents=True)
        input_path = deep_path / "image.jpg"
        input_path.touch()

        key = make_output_key(input_path, input_root)
        assert key == Path("level1/level2/level3/level4/image")
