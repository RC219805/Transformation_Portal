"""Managed SAM2 checkpoint validation and checksum cache helpers."""

from __future__ import annotations

import hashlib
import os
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Deque, Dict, Iterable, List, NamedTuple, Optional

from transformation_portal.portal import path_security


class Sam2CheckpointValidationError(ValueError):
    """Validation error raised by app-independent SAM2 checkpoint helpers."""

    def __init__(self, message: str, *, reason: str = "invalid_path_value") -> None:
        cleaned_message = str(message or "").strip() or "Invalid path value"
        self.reason = str(reason or "invalid_path_value").strip() or "invalid_path_value"
        super().__init__(cleaned_message)


@dataclass(frozen=True)
class ManagedSam2CheckpointValidationResult:
    normalized_path: Optional[str]
    reason: Optional[str] = None


@dataclass(frozen=True)
class _ManagedSam2ChecksumCacheEntry:
    digest: Optional[str]
    reason: Optional[str]


class _Sam2CacheKey(NamedTuple):
    """Cache key for SAM2 checkpoint trust results."""

    path: str
    size: int
    mtime_ns: int
    dev: int
    ino: int
    ctime_ns: int


_MANAGED_SAM2_CHECKSUM_CACHE_MAX_ENTRIES = 128

_MANAGED_SAM2_REASON_MESSAGES = {
    "checkpoint_file_too_large": "SAM2 checkpoint path exceeds checksum verification size limit",
    "invalid_path_value": "Invalid path value",
    "path_outside_allowed_roots": "Path outside allowed roots",
    "path_shorthand_traversal_disallowed": "Path shorthand traversal disallowed",
    "untrusted_checkpoint_path": "SAM2 checkpoint path is not trusted",
}


class _ManagedSam2BoundedChecksumCache(Dict[_Sam2CacheKey, _ManagedSam2ChecksumCacheEntry]):
    """Bounded FIFO cache for SAM2 checkpoint trust results."""

    def __init__(self, max_entries: int = _MANAGED_SAM2_CHECKSUM_CACHE_MAX_ENTRIES) -> None:
        super().__init__()
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self._max_entries = max_entries
        self._insertion_order: Deque[_Sam2CacheKey] = deque()

    def __setitem__(
        self,
        key: _Sam2CacheKey,
        value: _ManagedSam2ChecksumCacheEntry,
    ) -> None:
        if key not in self:
            self._insertion_order.append(key)
            if len(self._insertion_order) > self._max_entries:
                oldest = self._insertion_order.popleft()
                super().pop(oldest, None)
        super().__setitem__(key, value)

    def clear(self) -> None:
        super().clear()
        self._insertion_order.clear()


def _managed_sam2_reason_message(reason: str) -> str:
    """Return the canonical internal validation message for a SAM2 reason code."""

    return _MANAGED_SAM2_REASON_MESSAGES.get(reason, "Invalid path value")


def _managed_sam2_checksum_cache_key(path: Path) -> _Sam2CacheKey:
    """Build the checksum cache key for a trusted SAM2 checkpoint path."""

    stat_result = path.stat()
    return _Sam2CacheKey(
        path=str(path),
        size=stat_result.st_size,
        mtime_ns=stat_result.st_mtime_ns,
        dev=stat_result.st_dev,
        ino=stat_result.st_ino,
        ctime_ns=stat_result.st_ctime_ns,
    )


def _clear_managed_sam2_checksum_cache(
    checksum_cache: _ManagedSam2BoundedChecksumCache,
    checksum_cache_lock: threading.Lock,
) -> None:
    """Clear an in-process SAM2 checksum cache."""

    with checksum_cache_lock:
        checksum_cache.clear()


def _hash_file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest for a local file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cached_managed_sam2_checksum_result(
    path: Path,
    *,
    trusted_sha256: Iterable[str],
    checksum_max_bytes: int,
    checksum_cache: _ManagedSam2BoundedChecksumCache,
    checksum_cache_lock: threading.Lock,
    hash_file_sha256: Callable[[Path], str] = _hash_file_sha256,
) -> _ManagedSam2ChecksumCacheEntry:
    """Return the cached or newly computed trust result for an external SAM2 checkpoint."""

    try:
        cache_key = _managed_sam2_checksum_cache_key(path)
    except OSError as exc:
        raise Sam2CheckpointValidationError("Invalid path value", reason="invalid_path_value") from exc

    with checksum_cache_lock:
        cached = checksum_cache.get(cache_key)
    if cached is not None:
        return cached

    if cache_key.size > checksum_max_bytes:
        entry = _ManagedSam2ChecksumCacheEntry(digest=None, reason="checkpoint_file_too_large")
    else:
        digest = hash_file_sha256(path)
        reason = None if digest in trusted_sha256 else "untrusted_checkpoint_path"
        entry = _ManagedSam2ChecksumCacheEntry(digest=digest, reason=reason)

    with checksum_cache_lock:
        checksum_cache[cache_key] = entry
    return entry


def _resolve_managed_sam2_checkpoint_validation(
    path_value: str,
    *,
    allowed_input_roots: List[Path],
    trusted_roots: Iterable[Path],
    trusted_sha256: Iterable[str],
    checksum_max_bytes: int,
    checksum_cache: _ManagedSam2BoundedChecksumCache,
    checksum_cache_lock: threading.Lock,
    repo_root: Path | None = None,
    hash_file_sha256: Callable[[Path], str] = _hash_file_sha256,
) -> ManagedSam2CheckpointValidationResult:
    """Resolve a managed SAM2 checkpoint path and preserve the exact failure reason."""

    try:
        resolved = path_security._resolve_allowed_request_path(path_value, allowed_input_roots, repo_root=repo_root)
    except path_security.PathSecurityValidationError as exc:
        return ManagedSam2CheckpointValidationResult(normalized_path=None, reason=exc.reason)
    except (OSError, RuntimeError, ValueError):
        return ManagedSam2CheckpointValidationResult(normalized_path=None, reason="invalid_path_value")

    # Repo-controlled checkpoints remain valid even before the artifact exists locally.
    if any(path_security._path_is_within_root(resolved, Path(os.path.realpath(root))) for root in trusted_roots):
        return ManagedSam2CheckpointValidationResult(normalized_path=str(resolved), reason=None)

    try:
        safe_file = path_security._ensure_safe_regular_file_path(resolved, allowed_input_roots, repo_root=repo_root)
    except path_security.PathSecurityValidationError as exc:
        return ManagedSam2CheckpointValidationResult(normalized_path=None, reason=exc.reason)

    checksum_result = _cached_managed_sam2_checksum_result(
        safe_file,
        trusted_sha256=trusted_sha256,
        checksum_max_bytes=checksum_max_bytes,
        checksum_cache=checksum_cache,
        checksum_cache_lock=checksum_cache_lock,
        hash_file_sha256=hash_file_sha256,
    )
    if checksum_result.reason is not None:
        return ManagedSam2CheckpointValidationResult(normalized_path=None, reason=checksum_result.reason)
    return ManagedSam2CheckpointValidationResult(normalized_path=str(safe_file), reason=None)


def _validate_managed_sam2_checkpoint_path(
    path_value: str,
    *,
    allowed_input_roots: List[Path],
    trusted_roots: Iterable[Path],
    trusted_sha256: Iterable[str],
    checksum_max_bytes: int,
    checksum_cache: _ManagedSam2BoundedChecksumCache,
    checksum_cache_lock: threading.Lock,
    repo_root: Path | None = None,
    hash_file_sha256: Callable[[Path], str] = _hash_file_sha256,
) -> str:
    validation = _resolve_managed_sam2_checkpoint_validation(
        path_value,
        allowed_input_roots=allowed_input_roots,
        trusted_roots=trusted_roots,
        trusted_sha256=trusted_sha256,
        checksum_max_bytes=checksum_max_bytes,
        checksum_cache=checksum_cache,
        checksum_cache_lock=checksum_cache_lock,
        repo_root=repo_root,
        hash_file_sha256=hash_file_sha256,
    )
    if validation.reason is not None:
        raise Sam2CheckpointValidationError(
            _managed_sam2_reason_message(validation.reason),
            reason=validation.reason,
        )
    return str(validation.normalized_path or "")
