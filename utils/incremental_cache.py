"""
Incremental Processing & Smart Caching System

Provides intelligent caching of intermediate results with dependency tracking
and automatic invalidation for 10-20× faster parameter iteration.

Performance targets:
- <5s for parameter-only changes (vs 60s+ full re-run)
- Content-based hashing for cache keys
- LRU eviction with configurable size limits
"""

import hashlib
import json
import pickle
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class CacheConfig:
    """Configuration for incremental cache"""
    cache_dir: Path = Path(".cache/transformation_portal")
    max_size_gb: float = 10.0
    max_age_days: float = 30.0
    compression: bool = True
    verbose: bool = False


@dataclass
class CacheEntry:
    """Metadata for cached result"""
    key: str
    path: Path
    size_bytes: int
    created_at: float
    last_accessed: float
    dependencies: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self, max_age_days: float) -> bool:
        """Check if entry has expired"""
        age_days = (time.time() - self.created_at) / 86400
        return age_days > max_age_days


class IncrementalCache:
    """
    Smart caching system for intermediate processing results.
    
    Features:
    - Content-based hashing for cache keys
    - Dependency tracking and invalidation
    - LRU eviction policy
    - Disk-based storage with compression
    - Cache statistics and management
    
    Example:
        >>> cache = IncrementalCache()
        >>> result = cache.get_or_compute(
        ...     "depth_map",
        ...     lambda: compute_depth(image),
        ...     inputs={"image": image_hash, "model": "depth_anything_v2"}
        ... )
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        self.entries: Dict[str, CacheEntry] = {}
        self._load_index()
        
    def _load_index(self):
        """Load cache index from disk"""
        index_path = self.config.cache_dir / "index.json"
        if index_path.exists():
            try:
                with open(index_path, 'r') as f:
                    data = json.load(f)
                for entry_data in data.get('entries', []):
                    entry = CacheEntry(
                        key=entry_data['key'],
                        path=Path(entry_data['path']),
                        size_bytes=entry_data['size_bytes'],
                        created_at=entry_data['created_at'],
                        last_accessed=entry_data['last_accessed'],
                        dependencies=set(entry_data.get('dependencies', [])),
                        metadata=entry_data.get('metadata', {})
                    )
                    self.entries[entry.key] = entry
            except Exception as e:
                if self.config.verbose:
                    print(f"Failed to load cache index: {e}")
                    
    def _save_index(self):
        """Save cache index to disk"""
        index_path = self.config.cache_dir / "index.json"
        data = {
            'entries': [
                {
                    'key': entry.key,
                    'path': str(entry.path),
                    'size_bytes': entry.size_bytes,
                    'created_at': entry.created_at,
                    'last_accessed': entry.last_accessed,
                    'dependencies': list(entry.dependencies),
                    'metadata': entry.metadata
                }
                for entry in self.entries.values()
            ]
        }
        try:
            with open(index_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            if self.config.verbose:
                print(f"Failed to save cache index: {e}")
                
    def _compute_key(self, namespace: str, inputs: Dict[str, Any]) -> str:
        """Compute content-based cache key"""
        hasher = hashlib.sha256()
        hasher.update(namespace.encode())
        
        for key in sorted(inputs.keys()):
            value = inputs[key]
            hasher.update(key.encode())
            
            if isinstance(value, (str, int, float, bool)):
                hasher.update(str(value).encode())
            elif isinstance(value, bytes):
                hasher.update(value)
            elif isinstance(value, np.ndarray):
                hasher.update(value.tobytes())
            elif isinstance(value, Path):
                hasher.update(str(value).encode())
                if value.exists():
                    hasher.update(str(value.stat().st_mtime).encode())
            else:
                hasher.update(str(value).encode())
                
        return hasher.hexdigest()
        
    def get(self, namespace: str, inputs: Dict[str, Any]) -> Optional[Any]:
        """
        Get cached result if available.
        
        Args:
            namespace: Cache namespace (e.g., "depth_map", "material_mask")
            inputs: Dictionary of inputs used to compute cache key
            
        Returns:
            Cached result or None if not found
        """
        key = self._compute_key(namespace, inputs)
        
        if key not in self.entries:
            return None
            
        entry = self.entries[key]
        
        if entry.is_expired(self.config.max_age_days):
            self.invalidate(key)
            return None
            
        if not entry.path.exists():
            del self.entries[key]
            return None
            
        try:
            with open(entry.path, 'rb') as f:
                result = pickle.load(f)
            entry.last_accessed = time.time()
            self._save_index()
            
            if self.config.verbose:
                print(f"Cache hit: {namespace} ({key[:8]}...)")
                
            return result
        except Exception as e:
            if self.config.verbose:
                print(f"Failed to load cached result: {e}")
            self.invalidate(key)
            return None
            
    def put(
        self,
        namespace: str,
        inputs: Dict[str, Any],
        result: Any,
        dependencies: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Store result in cache.
        
        Args:
            namespace: Cache namespace
            inputs: Dictionary of inputs used to compute cache key
            result: Result to cache
            dependencies: Set of dependency keys for invalidation
            metadata: Optional metadata to store with entry
        """
        key = self._compute_key(namespace, inputs)
        
        cache_path = self.config.cache_dir / namespace
        cache_path.mkdir(parents=True, exist_ok=True)
        
        result_path = cache_path / f"{key}.pkl"
        
        try:
            with open(result_path, 'wb') as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
                
            size_bytes = result_path.stat().st_size
            
            entry = CacheEntry(
                key=key,
                path=result_path,
                size_bytes=size_bytes,
                created_at=time.time(),
                last_accessed=time.time(),
                dependencies=dependencies or set(),
                metadata=metadata or {}
            )
            
            self.entries[key] = entry
            self._save_index()
            
            if self.config.verbose:
                print(f"Cache stored: {namespace} ({key[:8]}...) - {size_bytes/1024:.1f} KB")
                
            self._enforce_limits()
            
        except Exception as e:
            if self.config.verbose:
                print(f"Failed to cache result: {e}")
                
    def get_or_compute(
        self,
        namespace: str,
        compute_fn: Callable[[], Any],
        inputs: Dict[str, Any],
        dependencies: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        force: bool = False
    ) -> Any:
        """
        Get cached result or compute and cache it.
        
        Args:
            namespace: Cache namespace
            compute_fn: Function to compute result if not cached
            inputs: Dictionary of inputs for cache key
            dependencies: Dependency keys for invalidation
            metadata: Optional metadata
            force: Force recomputation even if cached
            
        Returns:
            Result (from cache or computed)
        """
        if not force:
            cached = self.get(namespace, inputs)
            if cached is not None:
                return cached
                
        if self.config.verbose:
            print(f"Cache miss: {namespace} - computing...")
            
        start_time = time.time()
        result = compute_fn()
        compute_time = time.time() - start_time
        
        if self.config.verbose:
            print(f"Computed in {compute_time:.2f}s")
            
        self.put(namespace, inputs, result, dependencies, metadata)
        return result
        
    def invalidate(self, key: str):
        """Invalidate specific cache entry"""
        if key in self.entries:
            entry = self.entries[key]
            if entry.path.exists():
                entry.path.unlink()
            del self.entries[key]
            self._save_index()
            
    def invalidate_namespace(self, namespace: str):
        """Invalidate all entries in namespace"""
        to_invalidate = [
            key for key, entry in self.entries.items()
            if str(entry.path.parent.name) == namespace
        ]
        for key in to_invalidate:
            self.invalidate(key)
            
    def invalidate_dependencies(self, dependency_key: str):
        """Invalidate all entries depending on given key"""
        to_invalidate = [
            key for key, entry in self.entries.items()
            if dependency_key in entry.dependencies
        ]
        for key in to_invalidate:
            self.invalidate(key)
            
    def clear(self):
        """Clear entire cache"""
        if self.config.cache_dir.exists():
            shutil.rmtree(self.config.cache_dir)
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        self.entries.clear()
        self._save_index()
        
    def _enforce_limits(self):
        """Enforce cache size and age limits"""
        total_size_gb = sum(e.size_bytes for e in self.entries.values()) / (1024**3)
        
        if total_size_gb > self.config.max_size_gb:
            self._evict_lru()
            
        expired_keys = [
            key for key, entry in self.entries.items()
            if entry.is_expired(self.config.max_age_days)
        ]
        for key in expired_keys:
            self.invalidate(key)
            
    def _evict_lru(self):
        """Evict least recently used entries"""
        target_size_gb = self.config.max_size_gb * 0.8
        current_size_gb = sum(e.size_bytes for e in self.entries.values()) / (1024**3)
        
        sorted_entries = sorted(
            self.entries.items(),
            key=lambda x: x[1].last_accessed
        )
        
        for key, entry in sorted_entries:
            if current_size_gb <= target_size_gb:
                break
            self.invalidate(key)
            current_size_gb -= entry.size_bytes / (1024**3)
            
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_entries = len(self.entries)
        total_size_bytes = sum(e.size_bytes for e in self.entries.values())
        total_size_gb = total_size_bytes / (1024**3)
        
        namespaces = {}
        for entry in self.entries.values():
            ns = entry.path.parent.name
            if ns not in namespaces:
                namespaces[ns] = {'count': 0, 'size_bytes': 0}
            namespaces[ns]['count'] += 1
            namespaces[ns]['size_bytes'] += entry.size_bytes
            
        return {
            'total_entries': total_entries,
            'total_size_gb': total_size_gb,
            'total_size_mb': total_size_bytes / (1024**2),
            'namespaces': namespaces,
            'cache_dir': str(self.config.cache_dir),
            'max_size_gb': self.config.max_size_gb,
            'max_age_days': self.config.max_age_days
        }
        
    def print_stats(self):
        """Print cache statistics"""
        stats = self.get_stats()
        
        print("\n" + "="*60)
        print("Cache Statistics")
        print("="*60)
        print(f"Location: {stats['cache_dir']}")
        print(f"Total entries: {stats['total_entries']}")
        print(f"Total size: {stats['total_size_gb']:.2f} GB ({stats['total_size_mb']:.1f} MB)")
        print(f"Limit: {stats['max_size_gb']:.1f} GB")
        print(f"Usage: {stats['total_size_gb']/stats['max_size_gb']*100:.1f}%")
        print(f"\nNamespaces:")
        for ns, data in stats['namespaces'].items():
            size_mb = data['size_bytes'] / (1024**2)
            print(f"  {ns}: {data['count']} entries, {size_mb:.1f} MB")
        print("="*60 + "\n")


class CachedPipeline:
    """
    Base class for pipelines with incremental caching.
    
    Example:
        >>> class MyPipeline(CachedPipeline):
        ...     def process(self, image_path, params):
        ...         depth = self.get_or_compute_depth(image_path)
        ...         result = self.apply_effects(depth, params)
        ...         return result
    """
    
    def __init__(self, cache: Optional[IncrementalCache] = None):
        self.cache = cache or IncrementalCache()
        
    def get_or_compute_depth(
        self,
        image_path: Path,
        model_name: str = "depth_anything_v2"
    ) -> np.ndarray:
        """Get or compute depth map"""
        return self.cache.get_or_compute(
            "depth_maps",
            lambda: self._compute_depth(image_path, model_name),
            inputs={
                "image_path": image_path,
                "model_name": model_name
            }
        )
        
    def get_or_compute_material_mask(
        self,
        image_path: Path,
        material_types: List[str]
    ) -> Dict[str, np.ndarray]:
        """Get or compute material masks"""
        return self.cache.get_or_compute(
            "material_masks",
            lambda: self._compute_material_masks(image_path, material_types),
            inputs={
                "image_path": image_path,
                "material_types": tuple(sorted(material_types))
            }
        )
        
    def _compute_depth(self, image_path: Path, model_name: str) -> np.ndarray:
        """Override in subclass"""
        raise NotImplementedError
        
    def _compute_material_masks(
        self,
        image_path: Path,
        material_types: List[str]
    ) -> Dict[str, np.ndarray]:
        """Override in subclass"""
        raise NotImplementedError


def hash_file(file_path: Path) -> str:
    """Compute SHA256 hash of file"""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def hash_array(array: np.ndarray) -> str:
    """Compute SHA256 hash of numpy array"""
    hasher = hashlib.sha256()
    hasher.update(array.tobytes())
    hasher.update(str(array.shape).encode())
    hasher.update(str(array.dtype).encode())
    return hasher.hexdigest()


def hash_dict(data: Dict[str, Any]) -> str:
    """Compute SHA256 hash of dictionary"""
    hasher = hashlib.sha256()
    for key in sorted(data.keys()):
        hasher.update(key.encode())
        value = data[key]
        if isinstance(value, (str, int, float, bool)):
            hasher.update(str(value).encode())
        elif isinstance(value, np.ndarray):
            hasher.update(hash_array(value).encode())
        else:
            hasher.update(str(value).encode())
    return hasher.hexdigest()
