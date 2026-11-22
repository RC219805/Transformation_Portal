"""GPU Model Cache for persistent model storage.

This module provides per-process model caching to avoid:
- Repeated model loads
- VRAM thrashing
- Slow startup times

Models are cached by key and reused across node executions
within the same worker process.

Example:
    >>> cache = ModelCache()
    >>>
    >>> # First call loads the model
    >>> model = cache.get_or_load("llava-large", lambda: load_llava("large"))
    >>>
    >>> # Second call returns cached model
    >>> model2 = cache.get_or_load("llava-large", lambda: load_llava("large"))
    >>>
    >>> assert model is model2  # Same object, no reload!
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, Optional, TypeVar

logger = logging.getLogger(__name__)


T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """Entry in the model cache.

    Attributes:
        model: Cached model instance
        key: Cache key
        loaded_at: Timestamp when loaded
        access_count: Number of accesses
        last_accessed: Last access timestamp
        size_bytes: Estimated memory size (if known)
    """

    model: T
    key: str
    loaded_at: float
    access_count: int = 0
    last_accessed: float = 0
    size_bytes: Optional[int] = None

    def touch(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = time.time()


class ModelCacheError(RuntimeError):
    """Raised for model cache errors."""


class _InFlightLoad:
    """A load in progress for one key; waiters block on ``done``."""

    __slots__ = ("done", "model", "error")

    def __init__(self) -> None:
        self.done = threading.Event()
        self.model: Any = None
        self.error: Optional[BaseException] = None


class ModelCache:
    """Per-process persistent model cache.

    Caches loaded models to avoid repeated loading and VRAM churn.
    Thread-safe for concurrent access.

    Example:
        >>> cache = ModelCache(max_entries=10)
        >>>
        >>> def load_fn():
        ...     return load_heavy_model()
        >>>
        >>> # First access loads
        >>> model = cache.get_or_load("my-model", load_fn)
        >>>
        >>> # Subsequent accesses return cached
        >>> model2 = cache.get_or_load("my-model", load_fn)
        >>> assert model is model2
    """

    def __init__(
        self,
        *,
        max_entries: Optional[int] = None,
        max_memory_bytes: Optional[int] = None,
    ) -> None:
        """Initialize model cache.

        Args:
            max_entries: Maximum number of cached models (None = unlimited)
            max_memory_bytes: Maximum total memory (None = unlimited)
        """
        self._cache: Dict[str, CacheEntry] = {}
        self._inflight: Dict[str, _InFlightLoad] = {}
        self._lock = threading.RLock()
        self._max_entries = max_entries
        self._max_memory = max_memory_bytes
        self._load_count = 0
        self._hit_count = 0
        self._miss_count = 0

    def get(self, key: str) -> Optional[Any]:
        """Get a cached model by key.

        Args:
            key: Cache key

        Returns:
            Cached model or None if not found
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None:
                entry.touch()
                self._hit_count += 1
                return entry.model
            return None

    def get_or_load(
        self,
        key: str,
        load_fn: Callable[[], T],
        *,
        size_bytes: Optional[int] = None,
    ) -> T:
        """Get cached model or load if not present.

        Thread-safe: only one thread will execute load_fn for a given key.

        Args:
            key: Cache key
            load_fn: Function to load the model (called if not cached)
            size_bytes: Estimated memory size of the model

        Returns:
            Cached or newly loaded model
        """
        with self._lock:
            # Check cache first
            entry = self._cache.get(key)
            if entry is not None:
                entry.touch()
                self._hit_count += 1
                logger.debug("Model cache hit: %s", key)
                return entry.model

            inflight = self._inflight.get(key)
            if inflight is None:
                # Cache miss - this thread owns the load for ``key``.
                inflight = _InFlightLoad()
                self._inflight[key] = inflight
                self._miss_count += 1
                owner = True
            else:
                # Another thread is already loading ``key``; share its result.
                self._hit_count += 1
                owner = False

        if not owner:
            inflight.done.wait()
            if inflight.error is not None:
                raise ModelCacheError(f"Failed to load model {key}: {inflight.error}") from inflight.error
            return inflight.model

        # Load outside the cache lock so other keys are not blocked; concurrent
        # callers for the same key wait on ``inflight`` instead of loading again.
        logger.info("Model cache miss, loading: %s", key)
        start_time = time.time()

        try:
            model = load_fn()
        except BaseException as e:
            with self._lock:
                self._inflight.pop(key, None)
            inflight.error = e
            inflight.done.set()
            if not isinstance(e, Exception):
                raise
            logger.error("Failed to load model %s: %s", key, e)
            raise ModelCacheError(f"Failed to load model {key}: {e}") from e

        load_time = time.time() - start_time
        logger.info("Loaded model %s in %.2fs", key, load_time)

        # Store in cache
        try:
            with self._lock:
                self._load_count += 1
                if self._make_room(size_bytes):
                    entry = CacheEntry(
                        model=model,
                        key=key,
                        loaded_at=time.time(),
                        size_bytes=size_bytes,
                    )
                    entry.touch()
                    self._cache[key] = entry
                else:
                    logger.warning("Model %s does not fit the cache limits; returning it uncached", key)
        finally:
            with self._lock:
                self._inflight.pop(key, None)
            inflight.model = model
            inflight.done.set()

        return model

    def _make_room(self, incoming_size: Optional[int]) -> bool:
        """Evict LRU entries so an entry of ``incoming_size`` fits the limits.

        Returns False when the entry cannot fit even in an empty cache
        (``max_entries <= 0`` or ``incoming_size > max_memory_bytes``); nothing
        is evicted in that case. Must be called with lock held.
        """
        incoming = incoming_size or 0
        if self._max_entries is not None and self._max_entries <= 0:
            return False
        if self._max_memory is not None and incoming > self._max_memory:
            return False
        self._maybe_evict(incoming)
        return True

    def _maybe_evict(self, incoming_size: int = 0) -> None:
        """Evict entries until one more entry of ``incoming_size`` fits.

        Uses LRU (Least Recently Used) eviction policy.
        Must be called with lock held.
        """
        while self._max_entries is not None and self._cache and len(self._cache) >= self._max_entries:
            self._evict(self._lru_key())

        if self._max_memory is not None:
            total_size = sum(e.size_bytes or 0 for e in self._cache.values())
            while self._cache and total_size + incoming_size > self._max_memory:
                lru_key = self._lru_key()
                total_size -= self._cache[lru_key].size_bytes or 0
                self._evict(lru_key)

    def _lru_key(self) -> str:
        """Key of the least recently used entry (cache must be non-empty)."""
        return min(self._cache.keys(), key=lambda k: self._cache[k].last_accessed)

    def _evict(self, key: str) -> None:
        """Evict a single entry.

        Must be called with lock held.
        """
        if key in self._cache:
            logger.info("Evicting model from cache: %s", key)
            del self._cache[key]

    def contains(self, key: str) -> bool:
        """Check if a key is in the cache."""
        with self._lock:
            return key in self._cache

    def remove(self, key: str) -> bool:
        """Remove a model from the cache.

        Args:
            key: Cache key

        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                logger.info("Removed model from cache: %s", key)
                return True
            return False

    def clear(self) -> int:
        """Clear all cached models.

        Returns:
            Number of models cleared
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info("Cleared %d models from cache", count)
            return count

    def keys(self) -> list[str]:
        """List all cached model keys."""
        with self._lock:
            return list(self._cache.keys())

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache metrics
        """
        with self._lock:
            total_accesses = self._hit_count + self._miss_count
            hit_rate = self._hit_count / total_accesses if total_accesses > 0 else 0

            return {
                "entries": len(self._cache),
                "max_entries": self._max_entries,
                "load_count": self._load_count,
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "hit_rate": hit_rate,
                "total_size_bytes": sum(e.size_bytes or 0 for e in self._cache.values()),
                "keys": list(self._cache.keys()),
            }

    def __len__(self) -> int:
        """Number of cached models."""
        with self._lock:
            return len(self._cache)

    def __contains__(self, key: str) -> bool:
        """Check if key is cached."""
        return self.contains(key)


# Global model cache instance
GLOBAL_MODEL_CACHE = ModelCache()


def get_model_cache() -> ModelCache:
    """Get the global model cache."""
    return GLOBAL_MODEL_CACHE


def cached_model(key: str):
    """Decorator for caching model loading functions.

    Example:
        >>> @cached_model("llava-large")
        ... def load_llava():
        ...     return LlavaModel.from_pretrained(...)
        >>>
        >>> model = load_llava()  # Loads on first call, cached after
    """

    def decorator(load_fn: Callable[[], T]) -> Callable[[], T]:
        def wrapper() -> T:
            return GLOBAL_MODEL_CACHE.get_or_load(key, load_fn)

        return wrapper

    return decorator
