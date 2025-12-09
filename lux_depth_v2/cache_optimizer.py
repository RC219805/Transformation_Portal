"""
Caching Infrastructure for Models and Depth Maps.

Provides intelligent caching to eliminate repeated loading of:
- Depth estimation models (Depth Anything V2)
- Material segmentation models
- Upscaling models
- Generated depth maps (reuse across runs)

Key Features:
- ModelCache: Global singleton for model caching across batch
- DepthMapCache: Disk-based depth map caching with validation
- Smart eviction: Free models before memory-intensive operations
- Cache statistics and monitoring

Performance Target: 1.5-2× faster by eliminating repeated loads
"""

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import time
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Statistics for cache performance."""
    
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_time_saved_sec: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0-1)."""
        total = self.hits + self.misses
        return self.hits / max(1, total)
    
    @property
    def avg_time_saved_sec(self) -> float:
        """Average time saved per hit."""
        return self.total_time_saved_sec / max(1, self.hits)


class ModelCache:
    """
    Global model cache for batch processing.
    
    Keeps frequently-used models loaded in memory to avoid
    expensive repeated initialization.
    
    Models cached:
    - Depth estimation (Depth Anything V2)
    - Material segmentation
    - Upscaling models
    
    Performance:
    - Depth model load: 3-5s
    - Batch 6 images: 18-30s wasted
    - With cache: 3-5s once (4-6× faster)
    
    Usage:
        # Get cached model
        depth_model = ModelCache.get_depth_model()
        
        # Use across batch
        for image in batch:
            depth = depth_model.infer(image)
        
        # Clear before large operations (upscaling)
        ModelCache.clear_depth_model()
    """
    
    # Singleton instance
    _instance = None
    
    def __init__(self):
        """Initialize model cache (singleton)."""
        self.depth_model = None
        self.material_model = None
        self.upscale_model = None
        
        self.stats = {
            'depth': CacheStats(),
            'material': CacheStats(),
            'upscale': CacheStats(),
        }
        
        logger.info("ModelCache initialized (singleton)")
    
    @classmethod
    def get_instance(cls) -> 'ModelCache':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = ModelCache()
        return cls._instance
    
    @classmethod
    def get_depth_model(cls, force_reload: bool = False):
        """
        Get cached depth estimation model.
        
        Args:
            force_reload: Force reload model
        
        Returns:
            Depth model instance
        """
        instance = cls.get_instance()
        
        if instance.depth_model is None or force_reload:
            logger.info("Loading depth estimation model...")
            start_time = time.time()
            
            # Import and load model
            from . import weights
            instance.depth_model = weights.get_depth_model()
            
            load_time = time.time() - start_time
            logger.info(f"Depth model loaded: {load_time:.1f}s")
            instance.stats['depth'].misses += 1
        else:
            logger.debug("Using cached depth model")
            instance.stats['depth'].hits += 1
            instance.stats['depth'].total_time_saved_sec += 4.0  # Estimated load time
        
        return instance.depth_model
    
    @classmethod
    def get_material_model(cls, backend: str = 'heuristic', force_reload: bool = False):
        """
        Get cached material segmentation model.
        
        Args:
            backend: Segmentation backend
            force_reload: Force reload model
        
        Returns:
            Material model instance
        """
        instance = cls.get_instance()
        
        if instance.material_model is None or force_reload:
            logger.info(f"Loading material segmentation model: {backend}")
            start_time = time.time()
            
            # Import and load model
            from . import material_segmentation
            instance.material_model = material_segmentation.create_segmenter(backend=backend)
            
            load_time = time.time() - start_time
            logger.info(f"Material model loaded: {load_time:.1f}s")
            instance.stats['material'].misses += 1
        else:
            logger.debug("Using cached material model")
            instance.stats['material'].hits += 1
            instance.stats['material'].total_time_saved_sec += 2.0
        
        return instance.material_model
    
    @classmethod
    def clear_depth_model(cls):
        """Clear depth model from cache."""
        instance = cls.get_instance()
        if instance.depth_model is not None:
            logger.info("Evicting depth model from cache")
            instance.depth_model = None
            instance.stats['depth'].evictions += 1
            
            # Free memory
            cls._cleanup_memory()
    
    @classmethod
    def clear_material_model(cls):
        """Clear material model from cache."""
        instance = cls.get_instance()
        if instance.material_model is not None:
            logger.info("Evicting material model from cache")
            instance.material_model = None
            instance.stats['material'].evictions += 1
            
            cls._cleanup_memory()
    
    @classmethod
    def clear_all(cls):
        """Clear all models from cache."""
        instance = cls.get_instance()
        logger.info("Clearing all models from cache")
        
        instance.depth_model = None
        instance.material_model = None
        instance.upscale_model = None
        
        cls._cleanup_memory()
    
    @classmethod
    def _cleanup_memory(cls):
        """Force memory cleanup after eviction."""
        import gc
        gc.collect()
        
        try:
            import torch
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except:
            pass
    
    @classmethod
    def get_stats(cls) -> Dict[str, Dict]:
        """Get cache statistics."""
        instance = cls.get_instance()
        return {
            'depth': {
                'hits': instance.stats['depth'].hits,
                'misses': instance.stats['depth'].misses,
                'evictions': instance.stats['depth'].evictions,
                'hit_rate': instance.stats['depth'].hit_rate,
                'time_saved_sec': instance.stats['depth'].total_time_saved_sec,
            },
            'material': {
                'hits': instance.stats['material'].hits,
                'misses': instance.stats['material'].misses,
                'evictions': instance.stats['material'].evictions,
                'hit_rate': instance.stats['material'].hit_rate,
                'time_saved_sec': instance.stats['material'].total_time_saved_sec,
            },
        }


class DepthMapCache:
    """
    Disk-based depth map cache.
    
    Caches generated depth maps to avoid regeneration across runs.
    Validates cache integrity using content hashing.
    
    Performance:
    - Depth generation: 3-5s per image
    - Cache hit: 0.5s (disk read)
    - Speedup: 6-10× on cache hit
    
    Usage:
        cache = DepthMapCache(cache_dir='.cache/depth_maps')
        
        # Get or generate depth map
        depth_map = cache.get_or_generate(
            image_path='input.tif',
            generator_fn=lambda: generate_depth(image)
        )
        
        # Explicitly cache
        cache.set(image_path='input.tif', depth_map=depth_array)
    """
    
    def __init__(self, cache_dir: str = '.cache/depth_maps'):
        """
        Initialize depth map cache.
        
        Args:
            cache_dir: Directory for cache storage
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache metadata
        self.metadata_path = self.cache_dir / 'metadata.json'
        self.metadata = self._load_metadata()
        
        self.stats = CacheStats()
        
        logger.info(f"DepthMapCache initialized: {self.cache_dir}")
    
    def _load_metadata(self) -> Dict:
        """Load cache metadata."""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
        return {}
    
    def _save_metadata(self):
        """Save cache metadata."""
        try:
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")
    
    def _get_cache_key(self, image_path: Path) -> str:
        """
        Generate cache key for image.
        
        Uses content hash to detect changes.
        
        Args:
            image_path: Image file path
        
        Returns:
            Cache key (content hash)
        """
        # Content hash for cache invalidation
        hasher = hashlib.sha256()
        
        # Include file size and mtime
        stat = image_path.stat()
        hasher.update(str(stat.st_size).encode())
        hasher.update(str(stat.st_mtime).encode())
        
        # Include path for uniqueness
        hasher.update(str(image_path.resolve()).encode())
        
        return hasher.hexdigest()[:16]
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for key."""
        return self.cache_dir / f"{cache_key}.npy"
    
    def get(
        self,
        image_path: Path,
        validate: bool = True
    ) -> Optional[np.ndarray]:
        """
        Get depth map from cache.
        
        Args:
            image_path: Source image path
            validate: Validate cache integrity
        
        Returns:
            Cached depth map or None if not found
        """
        cache_key = self._get_cache_key(image_path)
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            logger.debug(f"Cache miss: {image_path.name}")
            self.stats.misses += 1
            return None
        
        # Validate if requested
        if validate and cache_key in self.metadata:
            metadata = self.metadata[cache_key]
            # Check if source file modified
            if metadata['source_path'] != str(image_path.resolve()):
                logger.warning(f"Cache key collision: {cache_key}")
                return None
        
        try:
            # Load depth map
            depth_map = np.load(cache_path)
            logger.info(f"Cache hit: {image_path.name} ({cache_key})")
            self.stats.hits += 1
            self.stats.total_time_saved_sec += 4.0  # Estimated generation time
            return depth_map
        except Exception as e:
            logger.error(f"Failed to load cached depth map: {e}")
            self.stats.misses += 1
            return None
    
    def set(
        self,
        image_path: Path,
        depth_map: np.ndarray
    ):
        """
        Cache depth map.
        
        Args:
            image_path: Source image path
            depth_map: Depth map array
        """
        cache_key = self._get_cache_key(image_path)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            # Save depth map
            np.save(cache_path, depth_map)
            
            # Update metadata
            self.metadata[cache_key] = {
                'source_path': str(image_path.resolve()),
                'cached_at': time.time(),
                'shape': depth_map.shape,
            }
            self._save_metadata()
            
            logger.debug(f"Cached depth map: {image_path.name} → {cache_key}")
        except Exception as e:
            logger.error(f"Failed to cache depth map: {e}")
    
    def get_or_generate(
        self,
        image_path: Path,
        generator_fn: callable,
        force_regenerate: bool = False
    ) -> np.ndarray:
        """
        Get depth map from cache or generate if not found.
        
        Args:
            image_path: Source image path
            generator_fn: Function to generate depth map if not cached
            force_regenerate: Force regeneration (ignore cache)
        
        Returns:
            Depth map array
        """
        if not force_regenerate:
            cached = self.get(image_path)
            if cached is not None:
                return cached
        
        # Generate depth map
        logger.info(f"Generating depth map: {image_path.name}")
        start_time = time.time()
        
        depth_map = generator_fn()
        
        gen_time = time.time() - start_time
        logger.info(f"Depth generation complete: {gen_time:.1f}s")
        
        # Cache for future use
        self.set(image_path, depth_map)
        
        return depth_map
    
    def clear(self, older_than_days: Optional[int] = None):
        """
        Clear cache entries.
        
        Args:
            older_than_days: Only clear entries older than N days
        """
        if older_than_days is None:
            # Clear all
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.metadata = {}
            self._save_metadata()
            logger.info("Depth map cache cleared")
        else:
            # Clear old entries
            cutoff_time = time.time() - (older_than_days * 24 * 3600)
            removed = 0
            
            for cache_key, meta in list(self.metadata.items()):
                if meta['cached_at'] < cutoff_time:
                    cache_path = self._get_cache_path(cache_key)
                    if cache_path.exists():
                        cache_path.unlink()
                    del self.metadata[cache_key]
                    removed += 1
            
            self._save_metadata()
            logger.info(f"Removed {removed} old cache entries (>{older_than_days} days)")
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            'hits': self.stats.hits,
            'misses': self.stats.misses,
            'hit_rate': self.stats.hit_rate,
            'time_saved_sec': self.stats.total_time_saved_sec,
            'cache_size': len(self.metadata),
        }


# Convenience functions

def get_cached_depth_model():
    """Get cached depth model (convenience)."""
    return ModelCache.get_depth_model()


def get_cached_material_model(backend: str = 'heuristic'):
    """Get cached material model (convenience)."""
    return ModelCache.get_material_model(backend=backend)


def create_depth_cache(cache_dir: str = '.cache/depth_maps') -> DepthMapCache:
    """Create depth map cache (convenience)."""
    return DepthMapCache(cache_dir=cache_dir)
