"""Tests for core artifacts module."""

import pytest
from pathlib import Path
import tempfile

from transformation_portal.core.artifacts import (
    CacheManager,
    ContentAddressedCache,
    ArtifactStorage,
    StorageBackend,
)


def test_content_addressed_cache():
    """Test content-addressed cache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = ContentAddressedCache(Path(tmpdir), max_size_gb=0.001)
        
        # Compute key
        key = cache.compute_key("input.jpg", preset="test")
        assert isinstance(key, str)
        assert len(key) == 64  # SHA256 hex
        
        # Cache miss
        assert cache.get(key) is None
        
        # Add to cache
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            f.flush()
            temp_file = Path(f.name)
        
        try:
            cache.put(key, temp_file)
            
            # Cache hit
            cached = cache.get(key)
            assert cached is not None
            assert cached.exists()
        finally:
            if temp_file.exists():
                temp_file.unlink()


def test_cache_stats():
    """Test cache statistics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = ContentAddressedCache(Path(tmpdir))
        
        stats = cache.get_stats()
        assert stats.total_entries == 0
        assert stats.hit_count == 0
        assert stats.miss_count == 0


def test_cache_eviction():
    """Test cache eviction on size limit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Very small cache (1KB)
        cache = ContentAddressedCache(Path(tmpdir), max_size_gb=0.000001)
        
        # Add multiple files
        for i in range(5):
            with tempfile.NamedTemporaryFile(delete=False, dir=tmpdir) as f:
                f.write(b"x" * 1024)  # 1KB each
                f.flush()
                temp_file = Path(f.name)
            
            key = cache.compute_key(f"file{i}")
            cache.put(key, temp_file)
        
        # Should have evicted old entries
        stats = cache.get_stats()
        assert stats.total_entries < 5


def test_cache_manager():
    """Test cache manager."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CacheManager(Path(tmpdir))
        
        # Test get_or_compute
        def compute_fn():
            with tempfile.NamedTemporaryFile(delete=False, dir=tmpdir) as f:
                f.write(b"computed")
                f.flush()
                return Path(f.name)
        
        key = "test_key"
        
        # First call should compute
        result1 = manager.get_or_compute(key, compute_fn)
        assert result1 is not None
        
        # Second call should use cache
        result2 = manager.get_or_compute(key, compute_fn)
        assert result2 is not None


def test_artifact_storage():
    """Test artifact storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        primary = Path(tmpdir) / "primary"
        storage = ArtifactStorage(primary_path=primary)
        
        # Create test file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test artifact")
            f.flush()
            source = Path(f.name)
        
        try:
            # Store artifact
            dest = storage.store(source, "test/artifact.bin")
            assert dest.exists()
            assert dest.is_relative_to(primary)
            
            # Retrieve artifact
            retrieved = storage.retrieve("test/artifact.bin")
            assert retrieved is not None
            assert retrieved == dest
        finally:
            if source.exists():
                source.unlink()


def test_artifact_storage_with_external():
    """Test artifact storage with external backend."""
    with tempfile.TemporaryDirectory() as tmpdir:
        primary = Path(tmpdir) / "primary"
        external = Path(tmpdir) / "external"
        
        storage = ArtifactStorage(
            primary_path=primary,
            external_path=external,
            auto_migrate_threshold_mb=0.001  # 1KB threshold
        )
        
        # Small file -> primary
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"small")
            f.flush()
            small_file = Path(f.name)
        
        try:
            dest_small = storage.store(small_file, "small.bin")
            assert dest_small.is_relative_to(primary)
        finally:
            if small_file.exists():
                small_file.unlink()
        
        # Large file -> external
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"x" * 10240)  # 10KB
            f.flush()
            large_file = Path(f.name)
        
        try:
            dest_large = storage.store(large_file, "large.bin")
            assert dest_large.is_relative_to(external)
        finally:
            if large_file.exists():
                large_file.unlink()


def test_artifact_migration():
    """Test artifact migration between backends."""
    with tempfile.TemporaryDirectory() as tmpdir:
        primary = Path(tmpdir) / "primary"
        external = Path(tmpdir) / "external"
        
        storage = ArtifactStorage(
            primary_path=primary,
            external_path=external
        )
        
        # Store in primary
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test")
            f.flush()
            source = Path(f.name)
        
        try:
            storage.store(source, "test.bin", backend=StorageBackend.LOCAL)
            
            # Migrate to external
            new_path = storage.migrate("test.bin", StorageBackend.EXTERNAL)
            assert new_path is not None
            assert new_path.is_relative_to(external)
            
            # Original should be gone
            assert not (primary / "test.bin").exists()
        finally:
            if source.exists():
                source.unlink()
