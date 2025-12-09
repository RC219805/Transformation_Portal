"""
Unified caching system for all pipelines.

Provides content-addressed caching with automatic invalidation.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, Dict
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry metadata."""
    key: str
    path: Path
    created_at: float
    last_accessed: float
    size_bytes: int
    metadata: Dict[str, Any]


@dataclass
class CacheStats:
    """Cache statistics."""
    total_entries: int
    total_size_mb: float
    hit_count: int
    miss_count: int
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hit_count + self.miss_count
        if total == 0:
            return 0.0
        return float(self.hit_count) / float(total)


class ContentAddressedCache:
    """
    Content-addressed cache for pipeline outputs.
    
    Uses content hashing to automatically detect when cached results
    are still valid based on input content and configuration.
    """
    
    def __init__(self, cache_dir: Path, max_size_gb: float = 10.0):
        """
        Initialize cache.
        
        Args:
            cache_dir: Cache directory
            max_size_gb: Maximum cache size in GB
        """
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._entries: Dict[str, CacheEntry] = {}
        self._hit_count = 0
        self._miss_count = 0
        
        self._load_index()
    
    def _load_index(self):
        """Load cache index from disk."""
        index_path = self.cache_dir / "index.json"
        if not index_path.exists():
            return
        
        try:
            with open(index_path) as f:
                data = json.load(f)
            
            for key, entry_data in data.get("entries", {}).items():
                self._entries[key] = CacheEntry(
                    key=key,
                    path=Path(entry_data["path"]),
                    created_at=entry_data["created_at"],
                    last_accessed=entry_data["last_accessed"],
                    size_bytes=entry_data["size_bytes"],
                    metadata=entry_data.get("metadata", {})
                )
        except Exception as e:
            logger.warning(f"Failed to load cache index: {e}")
    
    def _save_index(self):
        """Save cache index to disk."""
        index_path = self.cache_dir / "index.json"
        
        data = {
            "entries": {
                key: {
                    "path": str(entry.path),
                    "created_at": entry.created_at,
                    "last_accessed": entry.last_accessed,
                    "size_bytes": entry.size_bytes,
                    "metadata": entry.metadata,
                }
                for key, entry in self._entries.items()
            }
        }
        
        try:
            with open(index_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache index: {e}")
    
    def compute_key(self, *args, **kwargs) -> str:
        """
        Compute cache key from arguments.
        
        Args:
            *args: Positional arguments to hash
            **kwargs: Keyword arguments to hash
            
        Returns:
            Cache key (hex string)
        """
        hasher = hashlib.sha256()
        
        # Hash positional args
        for arg in args:
            hasher.update(str(arg).encode())
        
        # Hash keyword args (sorted for consistency)
        for key in sorted(kwargs.keys()):
            hasher.update(key.encode())
            hasher.update(str(kwargs[key]).encode())
        
        return hasher.hexdigest()
    
    def get(self, key: str) -> Optional[Path]:
        """
        Get cached file path by key.
        
        Args:
            key: Cache key
            
        Returns:
            Path to cached file or None if not found
        """
        if key not in self._entries:
            self._miss_count += 1
            return None
        
        entry = self._entries[key]
        
        # Check if file still exists
        if not entry.path.exists():
            logger.warning(f"Cache entry exists but file missing: {key}")
            del self._entries[key]
            self._miss_count += 1
            return None
        
        # Update access time
        entry.last_accessed = time.time()
        self._hit_count += 1
        
        return entry.path
    
    def put(self, key: str, path: Path, metadata: Optional[Dict[str, Any]] = None):
        """
        Add file to cache.
        
        Args:
            key: Cache key
            path: Path to file to cache
            metadata: Optional metadata
        """
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        # Get file size
        size_bytes = path.stat().st_size
        
        # Create entry
        entry = CacheEntry(
            key=key,
            path=path,
            created_at=time.time(),
            last_accessed=time.time(),
            size_bytes=size_bytes,
            metadata=metadata or {}
        )
        
        self._entries[key] = entry
        self._save_index()
        
        # Evict if over size limit
        self._evict_if_needed()
    
    def _evict_if_needed(self):
        """Evict old entries if cache is over size limit."""
        total_size = sum(entry.size_bytes for entry in self._entries.values())
        
        if total_size <= self.max_size_bytes:
            return
        
        # Sort by last accessed (oldest first)
        sorted_entries = sorted(
            self._entries.items(),
            key=lambda x: x[1].last_accessed
        )
        
        # Remove oldest entries until under limit
        for key, entry in sorted_entries:
            if total_size <= self.max_size_bytes:
                break
            
            # Delete file
            try:
                entry.path.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete cache file {entry.path}: {e}")
            
            # Remove entry
            del self._entries[key]
            total_size -= entry.size_bytes
        
        self._save_index()
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        total_size = sum(entry.size_bytes for entry in self._entries.values())
        
        return CacheStats(
            total_entries=len(self._entries),
            total_size_mb=total_size / (1024 * 1024),
            hit_count=self._hit_count,
            miss_count=self._miss_count
        )
    
    def clear(self):
        """Clear entire cache."""
        for entry in self._entries.values():
            try:
                entry.path.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete cache file {entry.path}: {e}")
        
        self._entries.clear()
        self._hit_count = 0
        self._miss_count = 0
        self._save_index()


class CacheManager:
    """
    High-level cache manager.
    
    Provides a simple interface for caching pipeline results.
    """
    
    def __init__(self, cache_dir: Path, max_size_gb: float = 10.0):
        """Initialize cache manager."""
        self.cache = ContentAddressedCache(cache_dir, max_size_gb)
    
    def get_or_compute(self, key: str, compute_fn, *args, **kwargs):
        """
        Get cached result or compute if not available.
        
        Args:
            key: Cache key
            compute_fn: Function to compute result if not cached
            *args: Arguments to pass to compute_fn
            **kwargs: Keyword arguments to pass to compute_fn
            
        Returns:
            Cached or computed result
        """
        cached_path = self.cache.get(key)
        
        if cached_path is not None:
            logger.debug(f"Cache hit for key: {key}")
            return cached_path
        
        logger.debug(f"Cache miss for key: {key}, computing...")
        result = compute_fn(*args, **kwargs)
        
        # If result is a path, cache it
        if isinstance(result, (str, Path)):
            self.cache.put(key, Path(result))
        
        return result
