"""Shared content-digest helpers for the segmentation lane.

Centralizes SHA-256 computation for the segmentation pipeline so a
single image array or checkpoint file is hashed at most once per
pipeline run **for informational uses** (e.g. building cache keys).

Two layers previously hashed the same content independently:

- ``lux_depth_v3.segmentation._cache._stable_array_hash`` (image hash
  for the segmentation cache key)
- ``spatial_ai.segmentation.sam2_backend._stable_image_hash`` (image
  hash for tiled-engine dispatch)

and (for files):

- ``lux_depth_v3.segmentation._cache._cached_file_sha256``
- ``spatial_ai.segmentation.sam2_backend._compute_file_sha256``

Both callers now route through this module. Image hashes use the
shared ``compute_array_sha256`` formula plus an optional per-owner
``ArrayDigestCache``; file hashes route through one of two helpers:

* ``compute_file_sha256`` — module-level ``@lru_cache``; keyed by the
  full stat identity ``(path, dev, ino, size, mtime_ns, ctime_ns)``.
  Used for cache-key construction where a stale digest only causes
  a benign cache miss.
* ``compute_file_sha256_uncached`` — always re-streams the file. Used
  by ``_validate_checkpoint_sha256`` so checkpoint integrity is
  verified against fresh bytes every call. Stat-tuple memoization
  cannot detect a same-size, mtime-restored, ctime-reset overwrite on
  filesystems where those fields are operator-controllable, so the
  integrity path opts out of memoization entirely. This is the
  "do not weaken ``_validate_checkpoint_sha256`` correctness" half of
  audit acceptance criterion N-3.

Tracks: N-3 (audit ``PORTAL_AUDIT_REPO_WIDE_2026-05-18.md`` finding
#4 — duplicate hashing).
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Tuple, cast

import numpy as np

# Same chunk size both prior callers used; preserves wall-clock parity
# with the un-memoized code path so the first (cache-miss) call has no
# regression.
_FILE_HASH_CHUNK_BYTES = 1024 * 1024

# Default ArrayDigestCache bound. Sized to hold a comfortable working
# set for a tiling pass over a single image (multiple tile views may
# alias different ids) without unbounded growth in long-lived backends.
_DEFAULT_ARRAY_DIGEST_CACHE_MAX = 64


def _stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_FILE_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=8)
def compute_file_sha256(
    path: str,
    dev: int,
    ino: int,
    size: int,
    mtime_ns: int,
    ctime_ns: int,
) -> str:
    """Return the SHA-256 hex digest of ``path``.

    The five stat fields key the LRU; they are not used in the digest
    itself. Callers must ``stat()`` the file and pass the current
    values so any tracked metadata change invalidates the entry.

    Note this is **not** a tamper-proof identity. A privileged or
    co-located attacker can rewrite a file with same-size content and
    restore mtime/ctime via ``os.utime`` / ``debugfs`` / equivalent on
    many filesystems. Callers that need integrity must use
    ``compute_file_sha256_uncached`` instead. This cached path is for
    informational uses (cache-key building) where a stale digest
    causes only a benign cache miss.

    ``maxsize=8`` covers the realistic working set — a portal pipeline
    typically holds references to a handful of large checkpoints
    (SAM2, SAM ViT-H, DA3, FastVLM) plus auxiliary artifacts — without
    unbounded growth.
    """
    del dev, ino, size, mtime_ns, ctime_ns  # used only as cache keys
    return _stream_sha256(Path(path))


def compute_file_sha256_for_path(path: Path) -> str:
    """Convenience wrapper for cached file hashing.

    Stats ``path`` and forwards the full identity tuple to
    ``compute_file_sha256``. **Not suitable for integrity validation**
    — see module docstring; use ``compute_file_sha256_uncached`` there.
    """
    stat = path.stat()
    return compute_file_sha256(
        str(path),
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


def compute_file_sha256_uncached(path: Path) -> str:
    """Compute the SHA-256 of ``path`` from fresh bytes, bypassing all
    memoization.

    Used by ``sam2_backend._validate_checkpoint_sha256`` so the
    integrity guarantee is independent of stat-tuple identity. A
    tampered checkpoint with same-size content and a restored
    mtime/ctime cannot be detected by stat-keyed memoization; reading
    the bytes is the only reliable detection.
    """
    return _stream_sha256(path)


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
    """Per-owner bounded LRU memoization for ``compute_array_sha256``.

    Keyed by ``(id(arr), arr.shape, arr.dtype.str)``. The ``id()``
    component is safe for the cache owner's lifetime because:

    1. While ``arr`` is alive its id is unique; reuse only happens
       after garbage collection.
    2. The accompanying ``shape`` and ``dtype`` further narrow the key
       so post-GC id reuse onto an array of identical shape/dtype is
       the only collision class — extremely rare in image pipelines.
    3. The cache is intentionally not module-level: ownership scopes
       its lifetime to the pipeline run, typically a backend instance.
    4. The cache is **bounded** (``maxsize`` entries, LRU eviction).
       Long-lived backends handling many requests cannot grow this
       dict without bound; the oldest seen id is evicted first.

    Callers can also short-circuit the cache by passing a precomputed
    digest via ``override``; this is how the ``SegmentationInput``
    ``content_digest`` field threads a single hash through the
    pipeline.
    """

    __slots__ = ("_entries", "_maxsize")

    def __init__(self, maxsize: int = _DEFAULT_ARRAY_DIGEST_CACHE_MAX) -> None:
        if maxsize <= 0:
            raise ValueError(f"ArrayDigestCache maxsize must be positive; got {maxsize!r}")
        self._entries: "OrderedDict[_ArrayCacheKey, str]" = OrderedDict()
        self._maxsize = maxsize

    def get_or_compute(
        self,
        array: Optional[np.ndarray],
        *,
        override: Optional[str] = None,
    ) -> str:
        if array is None:
            return "none"
        key = (id(array), tuple(array.shape), array.dtype.str)
        if override is not None:
            self._entries[key] = override
            self._entries.move_to_end(key)
            self._evict_if_needed()
            return override
        cached = self._entries.get(key)
        if cached is not None:
            self._entries.move_to_end(key)
            return cached
        computed = compute_array_sha256(array)
        self._entries[key] = computed
        self._evict_if_needed()
        return computed

    def _evict_if_needed(self) -> None:
        while len(self._entries) > self._maxsize:
            self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()


__all__ = [
    "ArrayDigestCache",
    "compute_array_sha256",
    "compute_file_sha256",
    "compute_file_sha256_for_path",
    "compute_file_sha256_uncached",
]
