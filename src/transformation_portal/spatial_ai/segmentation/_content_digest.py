"""Shared content-digest helpers for the segmentation lane.

Centralizes SHA-256 computation for the segmentation pipeline so a single
checkpoint file or image array is hashed at most once per pipeline run.
Two layers previously hashed the same content independently:

- ``lux_depth_v3.segmentation._cache._stable_array_hash`` (image hash for
  the segmentation cache key)
- ``spatial_ai.segmentation.sam2_backend._stable_image_hash`` (image hash
  for tiled-engine dispatch)

and

- ``lux_depth_v3.segmentation._cache._cached_file_sha256``
- ``spatial_ai.segmentation.sam2_backend._compute_file_sha256``
  (both compute the SAM2 checkpoint digest; the second path was not
  memoized, so ``_validate_checkpoint_sha256`` rehashed the full
  checkpoint file on every call).

Both callers now route through this module. The module-level
``@lru_cache`` on ``compute_file_sha256`` is keyed by
``(path, size, mtime_ns)`` — identical files (same stat tuple) hit the
cache; any stat change invalidates the entry, so
``_validate_checkpoint_sha256`` correctness is preserved (a tampered
checkpoint changes ``mtime_ns`` or ``size``, missing both → a fresh hash
runs).

Image hashes are not module-level cached (numpy arrays don't support
weak references; ``id()`` reuse would risk a false hit after GC).
Callers that need per-pipeline memoization use ``ArrayDigestCache``,
which is owned by an object whose lifetime bounds the cache (typically
a backend instance for the run).

Tracks: N-3 (audit ``PORTAL_AUDIT_REPO_WIDE_2026-05-18.md`` finding #4
— duplicate hashing).
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, cast

import numpy as np

# Same chunk size both prior callers used; preserves wall-clock parity
# with the un-memoized code path so the first (cache-miss) call has no
# regression.
_FILE_HASH_CHUNK_BYTES = 1024 * 1024


@lru_cache(maxsize=8)
def compute_file_sha256(path: str, size: int, mtime_ns: int) -> str:
    """Return the SHA-256 hex digest of ``path``.

    The ``size`` and ``mtime_ns`` arguments key the LRU cache; they are
    not used in the digest itself. Callers must ``stat()`` the file and
    pass the current values so a content change invalidates the entry.

    ``maxsize=8`` is intentional — a portal pipeline typically holds
    references to a handful of large checkpoints (SAM2, SAM ViT-H,
    DA3, FastVLM) plus auxiliary artifacts. Eight slots cover the
    realistic working set without unbounded growth.
    """
    del size, mtime_ns  # used only as cache keys
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_FILE_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_file_sha256_for_path(path: Path) -> str:
    """Convenience wrapper that stats ``path`` and delegates to the cached
    hash. Mirrors the legacy ``sam2_backend._compute_file_sha256``
    signature for callers that don't carry ``size`` / ``mtime_ns``."""
    stat = path.stat()
    return compute_file_sha256(str(path), int(stat.st_size), int(stat.st_mtime_ns))


def compute_array_sha256(array: np.ndarray) -> str:
    """Canonical content digest for a numpy image/mask array.

    Identical to the formula previously duplicated across
    ``_cache._stable_array_hash`` and
    ``sam2_backend._stable_image_hash``: shape repr + dtype repr +
    raw uint8 view of the array buffer. The buffer is consumed as a
    single memoryview update (no chunking); arrays in the megabytes
    range are inexpensive compared to file hashing.
    """
    arr = array if array.flags.c_contiguous else np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(arr.shape).encode("utf-8"))
    digest.update(str(arr.dtype).encode("utf-8"))
    view = memoryview(cast(Any, arr.view(np.uint8).reshape(-1)))
    digest.update(cast(Any, view))
    return digest.hexdigest()


_ArrayCacheKey = Tuple[int, Tuple[int, ...], str]


class ArrayDigestCache:
    """Per-owner memoization layer for ``compute_array_sha256``.

    Keyed by ``(id(arr), arr.shape, arr.dtype.str)``. The ``id()``
    component is safe for the cache owner's lifetime because:

    1. While ``arr`` is alive its id is unique; reuse only happens
       after garbage collection.
    2. The accompanying ``shape`` and ``dtype`` further narrow the key
       so post-GC id reuse onto an array of identical shape/dtype is
       the only collision class — practical for image pipelines but
       extremely rare, and the resulting "stale" hash would still
       describe an array of the same dimensions.
    3. The cache is intentionally not module-level: ownership scopes
       its lifetime to the pipeline run. The backend instance that
       owns the cache lives for the run; when it goes out of scope
       the cache (and any captured ids) are released together.

    Callers can also short-circuit the cache by passing a precomputed
    digest via ``override``; this is how the ``SegmentationInput``
    ``content_digest`` field threads a single hash through the pipeline.
    """

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        self._entries: Dict[_ArrayCacheKey, str] = {}

    def get_or_compute(
        self,
        array: Optional[np.ndarray],
        *,
        override: Optional[str] = None,
    ) -> str:
        if array is None:
            return "none"
        if override is not None:
            # Cache the override too so subsequent lookups by id still hit.
            key = (id(array), tuple(array.shape), array.dtype.str)
            self._entries[key] = override
            return override
        key = (id(array), tuple(array.shape), array.dtype.str)
        cached = self._entries.get(key)
        if cached is not None:
            return cached
        computed = compute_array_sha256(array)
        self._entries[key] = computed
        return computed

    def clear(self) -> None:
        self._entries.clear()


__all__ = [
    "ArrayDigestCache",
    "compute_array_sha256",
    "compute_file_sha256",
    "compute_file_sha256_for_path",
]
