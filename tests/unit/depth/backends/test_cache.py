"""Unit tests for enhanced depth cache (npz + json sidecar)."""

import json

import numpy as np
import pytest

from transformation_portal.depth.backends import DepthCacheWriter, DepthResult

pytestmark = pytest.mark.unit


@pytest.mark.unit
class TestDepthCacheWriter:
    """Test enhanced depth cache with metadata sidecar."""

    @pytest.fixture
    def cache_dir(self, tmp_path):
        """Temporary cache directory."""
        cache_path = tmp_path / ".depth_cache"
        cache_path.mkdir(parents=True)
        return cache_path

    @pytest.fixture
    def cache_writer(self, cache_dir):
        """Cache writer instance."""
        return DepthCacheWriter(cache_dir)

    def test_write_read_relative_depth(self, cache_writer):
        """Legacy format: relative depth uses .npy only."""
        result = DepthResult(
            depth_map=np.random.rand(100, 100).astype(np.float32),
            original_image=np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
            metadata={"test": "metadata"},
            depth_units="relative",
        )

        cache_key = "test_relative_depth"
        cache_path = cache_writer.write(cache_key, result)

        assert cache_path.suffix == ".npy"
        assert cache_path.exists()

        # Read back
        loaded = cache_writer.read(cache_key)

        assert loaded is not None
        assert loaded.depth_units == "relative"
        assert np.allclose(loaded.depth_map, result.depth_map)

    def test_write_read_metric_depth(self, cache_writer):
        """Enhanced format: metric depth uses .npz + .json."""
        result = DepthResult(
            depth_map=np.random.rand(100, 100).astype(np.float32) * 10,  # 0-10 meters
            original_image=np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
            metadata={"engine": "depth_pro", "checkpoint": "test"},
            depth_units="meters",
            focal_length_px=525.0,
            field_of_view_deg=65.0,
            backend_id="depth_pro",
            device="mps",
            dtype="float32",
            input_size=(100, 100),
        )

        cache_key = "test_metric_depth"
        cache_path = cache_writer.write(cache_key, result)

        assert cache_path.suffix == ".npz"
        assert cache_path.exists()

        # Check sidecar exists
        json_path = cache_path.with_suffix(".json")
        assert json_path.exists()

        # Read back
        loaded = cache_writer.read(cache_key)

        assert loaded is not None
        assert loaded.depth_units == "meters"
        assert loaded.is_metric is True
        assert np.allclose(loaded.depth_map, result.depth_map)
        assert loaded.focal_length_px == 525.0
        assert loaded.field_of_view_deg == 65.0
        assert loaded.backend_id == "depth_pro"
        assert loaded.device == "mps"

    def test_read_legacy_npy_cache(self, cache_dir, cache_writer):
        """Should read legacy .npy caches (backward compatibility)."""
        # Simulate legacy cache (just .npy file)
        legacy_key = "legacy_depth"
        legacy_depth = np.random.rand(50, 50).astype(np.float32)
        legacy_path = cache_dir / f"{legacy_key}.npy"
        np.save(legacy_path, legacy_depth)

        # Read using new cache writer
        loaded = cache_writer.read(legacy_key)

        assert loaded is not None
        assert loaded.depth_units == "relative"  # Legacy assumed relative
        assert np.allclose(loaded.depth_map, legacy_depth)
        assert "legacy" in " ".join(loaded.warnings).lower()

    def test_metadata_sidecar_contents(self, cache_writer, cache_dir):
        """Verify metadata sidecar JSON structure."""
        result = DepthResult(
            depth_map=np.random.rand(100, 100).astype(np.float32) * 5,
            original_image=np.zeros((100, 100, 3), dtype=np.uint8),
            metadata={"provenance": "test"},
            depth_units="meters",
            focal_length_px=600.0,
            field_of_view_deg=70.0,
            backend_id="depth_pro",
            device="cuda",
            dtype="float32",
            input_size=(100, 100),
            warnings=["test warning"],
        )

        cache_key = "test_sidecar"
        cache_writer.write(cache_key, result)

        json_path = cache_dir / f"{cache_key}.json"
        with open(json_path) as f:
            metadata = json.load(f)

        assert metadata["cache_version"] == "2.0"
        assert metadata["format"] == "enhanced"
        assert metadata["depth_units"] == "meters"
        assert metadata["backend_id"] == "depth_pro"
        assert metadata["device"] == "cuda"
        assert metadata["focal_length_px"] == 600.0
        assert metadata["field_of_view_deg"] == 70.0
        assert metadata["input_size"] == [100, 100]
        assert "timestamp_utc" in metadata
        assert "test warning" in metadata["warnings"]

    def test_cache_exists(self, cache_writer):
        """exists() should detect both formats."""
        result_rel = DepthResult(
            depth_map=np.zeros((10, 10), dtype=np.float32),
            original_image=np.zeros((10, 10, 3), dtype=np.uint8),
            metadata={},
            depth_units="relative",
        )

        result_metric = DepthResult(
            depth_map=np.zeros((10, 10), dtype=np.float32),
            original_image=np.zeros((10, 10, 3), dtype=np.uint8),
            metadata={},
            depth_units="meters",
        )

        cache_writer.write("key_relative", result_rel)
        cache_writer.write("key_metric", result_metric)

        assert cache_writer.exists("key_relative")
        assert cache_writer.exists("key_metric")
        assert not cache_writer.exists("nonexistent")

    def test_cache_delete(self, cache_writer, cache_dir):
        """delete() should remove all associated files."""
        result = DepthResult(
            depth_map=np.zeros((10, 10), dtype=np.float32),
            original_image=np.zeros((10, 10, 3), dtype=np.uint8),
            metadata={},
            depth_units="meters",
            focal_length_px=500.0,
        )

        cache_key = "to_delete"
        cache_writer.write(cache_key, result)

        assert cache_writer.exists(cache_key)
        assert (cache_dir / f"{cache_key}.npz").exists()
        assert (cache_dir / f"{cache_key}.json").exists()

        # Delete
        deleted = cache_writer.delete(cache_key)

        assert deleted is True
        assert not cache_writer.exists(cache_key)
        assert not (cache_dir / f"{cache_key}.npz").exists()
        assert not (cache_dir / f"{cache_key}.json").exists()

    def test_read_nonexistent_key(self, cache_writer):
        """read() should return None for nonexistent keys."""
        loaded = cache_writer.read("nonexistent_key")
        assert loaded is None

    def test_enhanced_format_prefers_npz(self, cache_writer, cache_dir):
        """When both .npz and .npy exist, prefer .npz (enhanced)."""
        cache_key = "both_formats"

        # Write both formats manually
        legacy_depth = np.random.rand(10, 10).astype(np.float32)
        np.save(cache_dir / f"{cache_key}.npy", legacy_depth)

        enhanced_depth = np.random.rand(10, 10).astype(np.float32) * 10
        np.savez_compressed(
            cache_dir / f"{cache_key}.npz",
            depth=enhanced_depth,
            focal_length_px=np.array([500.0]),
        )

        with open(cache_dir / f"{cache_key}.json", "w") as f:
            json.dump({"depth_units": "meters", "cache_version": "2.0"}, f)

        # Should read enhanced format
        loaded = cache_writer.read(cache_key)

        assert loaded is not None
        assert loaded.depth_units == "meters"
        assert np.allclose(loaded.depth_map, enhanced_depth)
