"""Tests for incremental cache module"""

import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from utils.incremental_cache import (
    CacheConfig,
    CacheEntry,
    IncrementalCache,
    CachedPipeline,
    hash_file,
    hash_array,
    hash_dict
)


class TestCacheConfig:
    """Test CacheConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = CacheConfig()
        assert config.cache_dir == Path(".cache/transformation_portal")
        assert config.max_size_gb == 10.0
        assert config.max_age_days == 30.0
        assert config.compression is True
        assert config.verbose is False
        
    def test_custom_config(self):
        """Test custom configuration"""
        cache_dir = Path("/tmp/custom_cache")
        config = CacheConfig(
            cache_dir=cache_dir,
            max_size_gb=5.0,
            max_age_days=7.0,
            verbose=True
        )
        assert config.cache_dir == cache_dir
        assert config.max_size_gb == 5.0
        assert config.max_age_days == 7.0
        assert config.verbose is True


class TestCacheEntry:
    """Test CacheEntry dataclass"""
    
    def test_cache_entry_creation(self):
        """Test cache entry creation"""
        entry = CacheEntry(
            key="test_key",
            path=Path("/tmp/test.pkl"),
            size_bytes=1024,
            created_at=time.time(),
            last_accessed=time.time()
        )
        assert entry.key == "test_key"
        assert entry.size_bytes == 1024
        
    def test_expiration_check(self):
        """Test expiration check"""
        old_time = time.time() - (31 * 86400)
        entry = CacheEntry(
            key="test",
            path=Path("/tmp/test.pkl"),
            size_bytes=100,
            created_at=old_time,
            last_accessed=old_time
        )
        assert entry.is_expired(30.0)
        
        recent_entry = CacheEntry(
            key="test",
            path=Path("/tmp/test.pkl"),
            size_bytes=100,
            created_at=time.time(),
            last_accessed=time.time()
        )
        assert not recent_entry.is_expired(30.0)


class TestIncrementalCache:
    """Test IncrementalCache class"""
    
    @pytest.fixture
    def temp_cache(self):
        """Create temporary cache directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheConfig(
                cache_dir=Path(tmpdir),
                max_size_gb=1.0,
                verbose=False
            )
            yield IncrementalCache(config)
            
    def test_cache_initialization(self, temp_cache):
        """Test cache initialization"""
        assert temp_cache.config.cache_dir.exists()
        assert len(temp_cache.entries) == 0
        
    def test_simple_cache_put_get(self, temp_cache):
        """Test simple put and get operations"""
        result = {"value": 42}
        inputs = {"param1": "test", "param2": 123}
        
        temp_cache.put("test_namespace", inputs, result)
        
        cached = temp_cache.get("test_namespace", inputs)
        assert cached == result
        
    def test_cache_miss(self, temp_cache):
        """Test cache miss"""
        inputs = {"param": "nonexistent"}
        cached = temp_cache.get("test_namespace", inputs)
        assert cached is None
        
    def test_get_or_compute_cache_hit(self, temp_cache):
        """Test get_or_compute with cache hit"""
        inputs = {"x": 10}
        
        compute_fn = lambda: {"result": 42}
        
        result1 = temp_cache.get_or_compute(
            "test",
            compute_fn,
            inputs
        )
        assert result1 == {"result": 42}
        
        call_count = 0
        def counting_compute():
            nonlocal call_count
            call_count += 1
            return {"result": 42}
            
        result2 = temp_cache.get_or_compute(
            "test",
            counting_compute,
            inputs
        )
        assert result2 == {"result": 42}
        assert call_count == 0
        
    def test_get_or_compute_cache_miss(self, temp_cache):
        """Test get_or_compute with cache miss"""
        inputs = {"x": 10}
        
        compute_calls = []
        def compute_fn():
            compute_calls.append(1)
            return {"result": 42}
            
        result = temp_cache.get_or_compute(
            "test",
            compute_fn,
            inputs
        )
        
        assert result == {"result": 42}
        assert len(compute_calls) == 1
        
    def test_force_recompute(self, temp_cache):
        """Test forced recomputation"""
        inputs = {"x": 10}
        
        result1 = temp_cache.get_or_compute(
            "test",
            lambda: {"value": 1},
            inputs
        )
        assert result1 == {"value": 1}
        
        result2 = temp_cache.get_or_compute(
            "test",
            lambda: {"value": 2},
            inputs,
            force=True
        )
        assert result2 == {"value": 2}
        
    def test_cache_invalidation(self, temp_cache):
        """Test cache invalidation"""
        inputs = {"x": 10}
        
        temp_cache.put("test", inputs, {"result": 42})
        key = temp_cache._compute_key("test", inputs)
        
        assert temp_cache.get("test", inputs) is not None
        
        temp_cache.invalidate(key)
        
        assert temp_cache.get("test", inputs) is None
        
    def test_namespace_invalidation(self, temp_cache):
        """Test namespace invalidation"""
        temp_cache.put("namespace1", {"x": 1}, {"result": 1})
        temp_cache.put("namespace1", {"x": 2}, {"result": 2})
        temp_cache.put("namespace2", {"x": 3}, {"result": 3})
        
        temp_cache.invalidate_namespace("namespace1")
        
        assert temp_cache.get("namespace1", {"x": 1}) is None
        assert temp_cache.get("namespace1", {"x": 2}) is None
        assert temp_cache.get("namespace2", {"x": 3}) is not None
        
    def test_dependency_tracking(self, temp_cache):
        """Test dependency tracking and invalidation"""
        temp_cache.put(
            "derived",
            {"x": 1},
            {"result": 42},
            dependencies={"base_key"}
        )
        
        temp_cache.invalidate_dependencies("base_key")
        
        assert temp_cache.get("derived", {"x": 1}) is None
        
    def test_cache_clear(self, temp_cache):
        """Test cache clearing"""
        temp_cache.put("test1", {"x": 1}, {"result": 1})
        temp_cache.put("test2", {"x": 2}, {"result": 2})
        
        assert len(temp_cache.entries) == 2
        
        temp_cache.clear()
        
        assert len(temp_cache.entries) == 0
        
    def test_cache_statistics(self, temp_cache):
        """Test cache statistics"""
        temp_cache.put("test1", {"x": 1}, {"result": 1})
        temp_cache.put("test2", {"x": 2}, {"result": 2})
        
        stats = temp_cache.get_stats()
        
        assert stats['total_entries'] == 2
        assert stats['total_size_gb'] > 0
        assert 'test1' in stats['namespaces'] or 'test2' in stats['namespaces']
        
    def test_numpy_array_caching(self, temp_cache):
        """Test caching numpy arrays"""
        array = np.random.randn(100, 100)
        inputs = {"shape": array.shape}
        
        temp_cache.put("arrays", inputs, array)
        
        cached_array = temp_cache.get("arrays", inputs)
        
        assert cached_array is not None
        np.testing.assert_array_equal(cached_array, array)
        
    def test_content_based_hashing(self, temp_cache):
        """Test content-based hashing for cache keys"""
        inputs1 = {"x": 10, "y": 20}
        inputs2 = {"y": 20, "x": 10}
        
        key1 = temp_cache._compute_key("test", inputs1)
        key2 = temp_cache._compute_key("test", inputs2)
        
        assert key1 == key2
        
    def test_lru_eviction(self, temp_cache):
        """Test LRU eviction policy"""
        temp_cache.config.max_size_gb = 0.001
        
        for i in range(10):
            large_data = np.random.randn(1000, 1000)
            temp_cache.put(f"large_{i}", {"idx": i}, large_data)
            
        stats = temp_cache.get_stats()
        assert stats['total_entries'] < 10


class TestHashingFunctions:
    """Test hashing utility functions"""
    
    def test_hash_array(self):
        """Test array hashing"""
        array1 = np.array([1, 2, 3, 4, 5])
        array2 = np.array([1, 2, 3, 4, 5])
        array3 = np.array([1, 2, 3, 4, 6])
        
        hash1 = hash_array(array1)
        hash2 = hash_array(array2)
        hash3 = hash_array(array3)
        
        assert hash1 == hash2
        assert hash1 != hash3
        
    def test_hash_dict(self):
        """Test dictionary hashing"""
        dict1 = {"a": 1, "b": 2, "c": 3}
        dict2 = {"c": 3, "b": 2, "a": 1}
        dict3 = {"a": 1, "b": 2, "c": 4}
        
        hash1 = hash_dict(dict1)
        hash2 = hash_dict(dict2)
        hash3 = hash_dict(dict3)
        
        assert hash1 == hash2
        assert hash1 != hash3
        
    def test_hash_file(self):
        """Test file hashing"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            temp_path = Path(f.name)
            
        try:
            hash1 = hash_file(temp_path)
            hash2 = hash_file(temp_path)
            assert hash1 == hash2
        finally:
            temp_path.unlink()


class TestCachedPipeline:
    """Test CachedPipeline base class"""
    
    def test_cached_pipeline_initialization(self):
        """Test cached pipeline initialization"""
        pipeline = CachedPipeline()
        assert pipeline.cache is not None
        
    def test_custom_cache(self):
        """Test pipeline with custom cache"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheConfig(cache_dir=Path(tmpdir))
            cache = IncrementalCache(config)
            pipeline = CachedPipeline(cache)
            assert pipeline.cache.config.cache_dir == Path(tmpdir)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
