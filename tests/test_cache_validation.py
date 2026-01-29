"""
Tests for cache validation hardening (DepthCache).

Tests DepthCache validation logic for disk cache integrity.
"""

import pickle
from pathlib import Path

import numpy as np
import pytest

from transformation_portal.depth.utils.cache import DepthCache


class TestDepthCacheValidation:
    """Tests for DepthCache validation hardening."""

    def test_validate_disk_entry_success(self, tmp_path):
        """Test successful validation of disk cache entry."""
        cache = DepthCache(enable_disk_cache=True, cache_dir=tmp_path)

        # Create a valid cache file
        cache_file = tmp_path / "test_key.pkl"
        test_data = {"depth": np.random.rand(100, 100).astype(np.float16)}
        with open(cache_file, "wb") as f:
            pickle.dump(test_data, f)

        assert cache._validate_disk_entry(cache_file) is True

    def test_validate_disk_entry_missing_file(self, tmp_path):
        """Test validation fails for missing file."""
        cache = DepthCache(enable_disk_cache=True, cache_dir=tmp_path)

        missing_file = tmp_path / "nonexistent.pkl"
        assert cache._validate_disk_entry(missing_file) is False

    def test_validate_disk_entry_too_small(self, tmp_path):
        """Test validation fails for file that's too small."""
        cache = DepthCache(enable_disk_cache=True, cache_dir=tmp_path)

        # Create tiny file
        cache_file = tmp_path / "tiny.pkl"
        cache_file.write_bytes(b"x" * 50)  # Less than 100 bytes

        assert cache._validate_disk_entry(cache_file) is False

    def test_validate_disk_entry_corrupted_pickle(self, tmp_path):
        """Test validation fails for corrupted pickle file."""
        cache = DepthCache(enable_disk_cache=True, cache_dir=tmp_path)

        # Create invalid pickle
        cache_file = tmp_path / "corrupted.pkl"
        cache_file.write_bytes(b"not a valid pickle" * 20)

        assert cache._validate_disk_entry(cache_file) is False

    def test_validate_disk_entry_wrong_structure(self, tmp_path):
        """Test validation fails for wrong data structure."""
        cache = DepthCache(enable_disk_cache=True, cache_dir=tmp_path)

        # Create valid pickle but wrong structure
        cache_file = tmp_path / "wrong_structure.pkl"
        with open(cache_file, "wb") as f:
            pickle.dump(["not", "a", "dict"], f)

        assert cache._validate_disk_entry(cache_file) is False

    def test_validate_disk_entry_missing_depth_key(self, tmp_path):
        """Test validation fails for dict missing 'depth' key."""
        cache = DepthCache(enable_disk_cache=True, cache_dir=tmp_path)

        # Create dict without required keys
        cache_file = tmp_path / "missing_key.pkl"
        with open(cache_file, "wb") as f:
            pickle.dump({"other_key": "value"}, f)

        assert cache._validate_disk_entry(cache_file) is False

    def test_load_from_disk_with_validation(self, tmp_path):
        """Test that _load_from_disk validates before loading."""
        cache = DepthCache(enable_disk_cache=True, cache_dir=tmp_path)

        # Create valid entry
        key = "valid_key"
        cache_file = tmp_path / f"{key}.pkl"
        test_depth = np.random.rand(100, 100).astype(np.float16)
        with open(cache_file, "wb") as f:
            pickle.dump({"depth": test_depth}, f)

        result = cache._load_from_disk(key)
        assert result is not None
        assert "depth" in result

    def test_load_from_disk_removes_corrupted(self, tmp_path):
        """Test that corrupted files are removed during load."""
        cache = DepthCache(enable_disk_cache=True, cache_dir=tmp_path)

        # Create corrupted entry
        key = "corrupted_key"
        cache_file = tmp_path / f"{key}.pkl"
        cache_file.write_bytes(b"corrupted data" * 10)

        assert cache_file.exists()
        result = cache._load_from_disk(key)

        # Should return None and delete the corrupted file
        assert result is None
        assert not cache_file.exists()

    def test_disk_cache_size_limit(self, tmp_path):
        """Test that size limit is enforced."""
        cache = DepthCache(enable_disk_cache=True, cache_dir=tmp_path)

        # Create file larger than limit (simulated by setting small limit)
        cache_file = tmp_path / "large.pkl"
        large_data = {"depth": np.random.rand(1000, 1000).astype(np.float16)}
        with open(cache_file, "wb") as f:
            pickle.dump(large_data, f)

        # Temporarily lower the limit for testing
        original_limit = cache.MAX_DISK_CACHE_SIZE
        cache.MAX_DISK_CACHE_SIZE = 100  # Very small limit

        try:
            assert cache._validate_disk_entry(cache_file) is False
        finally:
            cache.MAX_DISK_CACHE_SIZE = original_limit
