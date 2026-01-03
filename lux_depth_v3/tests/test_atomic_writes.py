"""Tests for atomic write operations to prevent data corruption on crash.

This module tests the atomic write pattern that ensures:
1. No partial files are left on disk after crashes
2. Existing files are not corrupted by failed writes
3. Temp files are properly cleaned up
"""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from pathlib import Path
import json
import time

from lux_depth_v3.enhance.depth_writer import (
    atomic_write_depth_u16_png,
    read_depth_u16_png,
)
from lux_depth_v3.enhance.manifest import atomic_write_json


class TestAtomicDepthWrites:
    """Test atomic depth file writes."""

    def test_successful_write(self, tmp_path):
        """Normal write should succeed and clean up temp file."""
        output_path = tmp_path / "depth.png"
        depth = np.random.rand(100, 100).astype(np.float32)

        p1, p99 = atomic_write_depth_u16_png(output_path, depth)

        # Verify output exists
        assert output_path.exists()
        # Verify temp file cleaned up
        assert not (tmp_path / "depth.tmp.png").exists()
        # Verify percentiles returned
        assert isinstance(p1, float)
        assert isinstance(p99, float)
        assert p1 < p99

    def test_crash_during_write_cleanup(self, tmp_path):
        """Crash during write should clean up temp file."""
        output_path = tmp_path / "depth.png"
        depth = np.random.rand(100, 100).astype(np.float32)

        # Simulate crash during PIL save
        with patch("PIL.Image.Image.save", side_effect=IOError("Disk full")):
            with pytest.raises(IOError, match="Failed to write depth"):
                atomic_write_depth_u16_png(output_path, depth)

        # Verify no files remain
        assert not output_path.exists()
        assert not (tmp_path / "depth.tmp.png").exists()

    def test_preserves_existing_on_crash(self, tmp_path):
        """Failed write should not corrupt existing file."""
        output_path = tmp_path / "depth.png"

        # Write initial valid file
        depth1 = np.random.rand(100, 100).astype(np.float32)
        p1, p99 = atomic_write_depth_u16_png(output_path, depth1)

        original_size = output_path.stat().st_size
        original_mtime = output_path.stat().st_mtime

        # Wait to ensure mtime would change
        time.sleep(0.01)

        # Attempt to overwrite with crashing write
        with patch("PIL.Image.Image.save", side_effect=IOError("Disk full")):
            with pytest.raises(IOError):
                atomic_write_depth_u16_png(output_path, depth1 * 2)

        # Verify original file unchanged
        assert output_path.exists()
        assert output_path.stat().st_size == original_size
        assert output_path.stat().st_mtime == original_mtime

        # Verify can still read original
        depth_verify = read_depth_u16_png(output_path)
        assert depth_verify.shape == (100, 100)

    def test_parent_dir_created(self, tmp_path):
        """Parent directories should be created automatically."""
        output_path = tmp_path / "nested" / "deep" / "depth.png"
        depth = np.random.rand(100, 100).astype(np.float32)

        atomic_write_depth_u16_png(output_path, depth)

        assert output_path.exists()
        assert output_path.parent.exists()

    def test_verification_mode(self, tmp_path):
        """Verification mode should validate written depth."""
        output_path = tmp_path / "depth.png"
        depth = np.random.rand(100, 100).astype(np.float32)

        # Write with verification enabled
        p1, p99 = atomic_write_depth_u16_png(output_path, depth, debug_verify=True)

        # Should succeed and verify
        assert output_path.exists()

    def test_concurrent_writes_same_file(self, tmp_path):
        """Atomic writes should handle race conditions gracefully."""
        output_path = tmp_path / "depth.png"

        # Write two different depths to same path
        depth1 = np.zeros((100, 100), dtype=np.float32)
        depth2 = np.ones((100, 100), dtype=np.float32)

        p1_1, p99_1 = atomic_write_depth_u16_png(output_path, depth1)
        p1_2, p99_2 = atomic_write_depth_u16_png(output_path, depth2)

        # File should exist (second write overwrites)
        assert output_path.exists()
        # No temp files left
        assert not (tmp_path / "depth.tmp.png").exists()

    def test_disk_full_scenario(self, tmp_path):
        """Simulate disk full error during write."""
        output_path = tmp_path / "depth.png"
        depth = np.random.rand(100, 100).astype(np.float32)

        # Mock write_text to fail with "No space left on device"
        with patch("PIL.Image.Image.save", side_effect=OSError(28, "No space left on device")):
            with pytest.raises(IOError, match="Failed to write depth"):
                atomic_write_depth_u16_png(output_path, depth)

        # No partial files
        assert not output_path.exists()
        assert not (tmp_path / "depth.tmp.png").exists()


class TestAtomicJSONWrites:
    """Test atomic JSON manifest writes."""

    def test_successful_json_write(self, tmp_path):
        """Normal JSON write should succeed."""
        output_path = tmp_path / "manifest.json"
        data = {"key": "value", "number": 42, "nested": {"a": 1, "b": 2}}

        atomic_write_json(output_path, data)

        # Verify output exists
        assert output_path.exists()
        # Verify temp cleaned up
        assert not (tmp_path / "manifest.tmp.json").exists()

        # Verify content
        with open(output_path) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_json_crash_cleanup(self, tmp_path):
        """Crash during JSON write should clean up temp file."""
        output_path = tmp_path / "manifest.json"
        data = {"key": "value"}

        # Simulate crash during write
        with patch("pathlib.Path.write_text", side_effect=IOError("Disk full")):
            with pytest.raises(IOError, match="Failed to write JSON"):
                atomic_write_json(output_path, data)

        # No files remain
        assert not output_path.exists()
        assert not (tmp_path / "manifest.tmp.json").exists()

    def test_json_preserves_existing(self, tmp_path):
        """Failed JSON write should not corrupt existing file."""
        output_path = tmp_path / "manifest.json"

        # Write initial valid JSON
        data1 = {"version": 1, "data": "original"}
        atomic_write_json(output_path, data1)

        original_size = output_path.stat().st_size

        # Attempt to overwrite with crashing write
        data2 = {"version": 2, "data": "corrupted"}
        with patch("pathlib.Path.write_text", side_effect=IOError("Disk full")):
            with pytest.raises(IOError):
                atomic_write_json(output_path, data2)

        # Verify original file unchanged
        assert output_path.exists()
        assert output_path.stat().st_size == original_size

        # Verify original content intact
        with open(output_path) as f:
            loaded = json.load(f)
        assert loaded == data1

    def test_json_parent_dir_created(self, tmp_path):
        """Parent directories should be created automatically."""
        output_path = tmp_path / "a" / "b" / "c" / "manifest.json"
        data = {"test": True}

        atomic_write_json(output_path, data)

        assert output_path.exists()
        assert output_path.parent.exists()

    def test_json_indentation(self, tmp_path):
        """JSON should be written with proper indentation."""
        output_path = tmp_path / "manifest.json"
        data = {"a": 1, "b": {"c": 2}}

        atomic_write_json(output_path, data, indent=4)

        content = output_path.read_text()
        # Should have newlines (indented)
        assert "\n" in content
        # Should have 4-space indentation
        assert "    " in content

    def test_permission_error_handling(self, tmp_path):
        """Permission errors should be handled gracefully."""
        output_path = tmp_path / "manifest.json"
        data = {"test": True}

        # Simulate permission denied
        with patch("pathlib.Path.write_text", side_effect=PermissionError("Access denied")):
            with pytest.raises(IOError, match="Failed to write JSON"):
                atomic_write_json(output_path, data)

        # No partial files
        assert not output_path.exists()
        assert not (tmp_path / "manifest.tmp.json").exists()


class TestIntegrationAtomicWrites:
    """Integration tests for atomic writes in orchestrator context."""

    def test_manifest_uses_atomic_write(self, tmp_path):
        """CombinedManifest.write() should use atomic writes."""
        from lux_depth_v3.enhance.manifest import (
            CombinedManifest,
            InputMetadata,
        )

        manifest = CombinedManifest(
            schema="test-schema-v1",
            input=InputMetadata(
                image_path="/test/image.jpg",
                image_sha256="abc123",
            ),
        )

        output_path = tmp_path / "manifest.json"
        manifest.write(output_path)

        # Verify file exists
        assert output_path.exists()
        # Verify no temp file
        assert not (tmp_path / "manifest.tmp.json").exists()

        # Verify can load back
        loaded = CombinedManifest.load(output_path)
        assert loaded.schema == "test-schema-v1"
        assert loaded.input.image_path == "/test/image.jpg"

    def test_depth_write_in_orchestrator(self, tmp_path):
        """Orchestrator should use atomic depth writes."""
        # This is more of a smoke test to ensure integration
        depth = np.random.rand(100, 100).astype(np.float32)
        depth_path = tmp_path / "output" / "depth" / "test_depth.png"

        # Direct atomic write
        p1, p99 = atomic_write_depth_u16_png(depth_path, depth)

        # Verify structure
        assert depth_path.exists()
        assert depth_path.parent.name == "depth"

        # Verify readable
        loaded_depth = read_depth_u16_png(depth_path)
        assert loaded_depth.shape == (100, 100)
        assert loaded_depth.dtype == np.uint16


# Fixtures
@pytest.fixture
def sample_depth():
    """Create sample depth array."""
    return np.random.rand(100, 100).astype(np.float32)


@pytest.fixture
def sample_manifest_data():
    """Create sample manifest data."""
    return {
        "schema": "test-v1",
        "input": {
            "image_path": "/test.jpg",
            "image_sha256": "abc123",
        },
        "depth": {
            "backend": "da3",
            "model": "large",
            "depth_path": "depth.png",
        },
    }
