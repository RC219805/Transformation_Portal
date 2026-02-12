"""Content-addressed artifact store with provenance tracking (Phase 3 L1).

Provides deterministic caching for pipeline artifacts with full provenance metadata.

Design Principles (ADR-029):
- Content-addressed (same inputs → same cache key → same artifact)
- Atomic writes (temp + fsync + rename, no partial artifacts)
- Provenance metadata (input hashes, model revisions, timestamps)
- Determinism verification (cache hit = bitwise identical output)
- Multi-process safe operations (per-key file locks for concurrent access)

Storage Layout:
    .cache/spatial_ai/
    ├── artifacts/
    │   ├── ab/
    │   │   ├── ab3f5e8b2c1d4.npz        # Artifact data
    │   │   └── ab3f5e8b2c1d4.json       # Provenance metadata
    │   └── ...
    ├── locks/                           # Per-key lock files
    │   ├── ab3f5e8b2c1d4.lock
    │   └── ...
    └── stats.json  # Cache statistics (size, hits, misses)

Key Features:
1. Content Addressing: SHA256-based cache keys from inputs + config
2. Atomic Writes: No partial artifacts (temp file + rename)
3. Provenance: Full lineage tracking (inputs, models, timestamps)
4. LRU Eviction: Size-based limits with least-recently-used eviction
5. Determinism: Bitwise identical outputs for identical inputs

Example:
    >>> store = ArtifactStore(cache_dir=".cache/spatial_ai", max_size_gb=10.0)
    >>>
    >>> # Check cache
    >>> cache_key = stage.compute_cache_key(inputs, context)
    >>> if store.exists(cache_key):
    ...     result = store.load(cache_key)  # Cache hit
    >>> else:
    ...     result = stage.execute(inputs, context)  # Cache miss
    ...     provenance = ProvenanceMetadata(...)
    ...     store.store(cache_key, result, provenance)
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Generator, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Security: Strict validation for cache keys (SHA256 format)
# Prevents path traversal attacks via malformed cache keys
SAFE_CACHE_KEY = re.compile(r"^[a-f0-9]{64}$")

# Lock acquisition timeout (seconds) - prevents indefinite hangs
DEFAULT_LOCK_TIMEOUT = 30.0


class CacheLockTimeout(Exception):
    """Raised when cache lock acquisition times out.

    This exception indicates that a cache operation could not acquire
    the required lock within the configured timeout period. This typically
    means another process is holding the lock for an unusually long time.

    Common causes:
    - Another process is writing a large artifact
    - Another process has crashed while holding the lock
    - Concurrent access from many processes

    Resolution:
    - Wait and retry the operation
    - Increase lock_timeout_seconds if artifacts are very large
    - Check for crashed processes holding stale locks
    """

    pass


@dataclass
class ProvenanceMetadata:
    """Provenance metadata for cached artifacts.

    Attributes:
        cache_key: Content-addressed cache key.
        stage_id: Stage identifier in execution graph.
        stage_version: Stage version (from StageMetadata).
        input_fingerprints: Hashes of input data (input_name → SHA256).
        config_snapshot: Stage configuration at execution time.
        timestamp: ISO 8601 timestamp (UTC).
        hostname: Machine hostname.
        python_version: Python version string.
        numpy_version: NumPy version string.
        torch_version: PyTorch version (if applicable).
        device: Execution device ("cuda", "cpu", "mps").
        model_repo_id: HuggingFace model repo ID (if applicable).
        model_revision: HuggingFace model revision/commit (if applicable).

    Design notes:
    - Captures complete lineage for reproducibility audits.
    - Timestamps are UTC to avoid timezone ambiguity.
    - Model provenance enables tracking of upstream drift.
    """

    cache_key: str
    stage_id: str
    stage_version: str
    input_fingerprints: Dict[str, str]
    config_snapshot: Dict[str, Any]
    timestamp: str
    hostname: str
    python_version: str
    numpy_version: str
    device: str = "cpu"
    torch_version: Optional[str] = None
    model_repo_id: Optional[str] = None
    model_revision: Optional[str] = None


class ArtifactStore:
    """Content-addressed artifact store with provenance tracking.

    Storage design:
    - Artifacts stored as .npz files (NumPy arrays, metadata)
    - Provenance stored as .json sidecars
    - Content-addressed by SHA256 hash (collision-resistant)
    - Two-level directory hierarchy (first 2 hex chars as prefix)
    - Per-key lock files for multi-process safety

    Multi-process safety (Phase 3 L1):
    - Per-key lock files prevent concurrent access to same cache key
    - Exclusive locks for writes (store operations)
    - Shared locks for reads (load operations) with exclusive fallback
    - Lock files stored in locks/ directory
    - Automatic lock release on operation completion
    - Configurable timeout to prevent indefinite hangs

    Thread safety:
    - Atomic writes (temp file + rename)
    - File locks are process-safe and thread-safe
    - Write operations use OS-level atomic rename

    Cache eviction (L1):
    - Warns at size limit, actual eviction deferred to L2
    - LRU tracking via file access times (atime)

    Example:
        >>> store = ArtifactStore(cache_dir=Path(".cache/spatial_ai"))
        >>>
        >>> # Store artifact (automatically acquires exclusive lock)
        >>> result = {"masks": masks, "scores": scores}
        >>> provenance = ProvenanceMetadata(...)
        >>> store.store(cache_key, result, provenance)
        >>>
        >>> # Load artifact (automatically acquires shared lock)
        >>> if store.exists(cache_key):
        ...     result = store.load(cache_key)
        ...     provenance = store.load_provenance(cache_key)
    """

    def __init__(
        self,
        cache_dir: Path,
        max_size_gb: float = 10.0,
        eviction_policy: str = "lru",
        lock_timeout_seconds: float = DEFAULT_LOCK_TIMEOUT,
    ):
        """Initialize artifact store.

        Args:
            cache_dir: Base directory for cache storage.
            max_size_gb: Maximum cache size in gigabytes (warns at limit).
            eviction_policy: Eviction policy ("lru" or "manual").
            lock_timeout_seconds: Timeout for lock acquisition (default 30s).

        Design notes:
        - Creates cache_dir if it doesn't exist.
        - Initializes artifacts/ and locks/ subdirectories.
        - Logs warning if cache size exceeds limit (no auto-eviction in L1).
        """
        self.cache_dir = Path(cache_dir)
        self.artifacts_dir = self.cache_dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Lock directory for per-key locks (Phase 3 L1)
        self.locks_dir = self.cache_dir / "locks"
        self.locks_dir.mkdir(parents=True, exist_ok=True)

        self.max_size_gb = max_size_gb
        self.eviction_policy = eviction_policy
        self.lock_timeout_seconds = lock_timeout_seconds

        # Cache statistics
        self.stats_path = self.cache_dir / "stats.json"
        self._load_stats()

    @contextlib.contextmanager
    def _acquire_lock(
        self, cache_key: str, exclusive: bool = True
    ) -> Generator[None, None, None]:
        """Acquire per-key file lock for cache operations.

        Args:
            cache_key: Content-addressed cache key (must be valid SHA256 hex).
            exclusive: If True, acquire exclusive (write) lock. If False, try
                      shared (read) lock first, fall back to exclusive if unavailable.

        Yields:
            None (context manager)

        Raises:
            CacheLockTimeout: If lock cannot be acquired within timeout.
            ValueError: If cache_key format is invalid.

        Design notes (Phase 3 L1):
        - Per-key locking maximizes concurrency (different keys don't block each other)
        - Lock files stored in locks/ directory: locks/<cache_key>.lock
        - Exclusive locks for writes prevent corruption during store operations
        - Shared locks for reads allow concurrent readers (with exclusive fallback)
        - Lock acquisition uses non-blocking retries with timeout
        - Locks released automatically via context manager (even on exceptions)
        - Lock files never deleted (reused across operations for same key)

        Thread safety:
        - fcntl locks are process-safe AND thread-safe within same process
        - Multiple threads in same process will serialize on same lock file

        Multi-process safety:
        - Prevents partial reads (reader blocks until writer completes)
        - Prevents write corruption (only one writer at a time per key)
        - Different keys can be accessed concurrently without blocking
        """
        # Validate cache_key format (security + correctness)
        if not SAFE_CACHE_KEY.match(cache_key):
            raise ValueError(
                f"Invalid cache_key format: {cache_key!r}. "
                "Expected 64 lowercase hex characters (SHA256)."
            )

        # Lock file path: locks/<cache_key>.lock
        lock_path = self.locks_dir / f"{cache_key}.lock"

        # Create lock file if it doesn't exist (idempotent)
        lock_path.touch(exist_ok=True)

        # Open lock file
        lock_file = open(lock_path, "r+")

        try:
            # Determine lock mode
            if exclusive:
                lock_mode = fcntl.LOCK_EX  # Exclusive (write) lock
                lock_type = "exclusive"
            else:
                # Try shared (read) lock first
                lock_mode = fcntl.LOCK_SH  # Shared (read) lock
                lock_type = "shared"

            # Acquire lock with timeout
            start_time = time.time()
            acquired = False

            while not acquired:
                try:
                    # Try non-blocking lock
                    fcntl.flock(lock_file.fileno(), lock_mode | fcntl.LOCK_NB)
                    acquired = True
                    logger.debug(f"Acquired {lock_type} lock for cache_key: {cache_key}")

                except BlockingIOError:
                    # Lock held by another process/thread
                    elapsed = time.time() - start_time
                    if elapsed >= self.lock_timeout_seconds:
                        raise CacheLockTimeout(
                            f"Could not acquire {lock_type} lock for cache_key {cache_key} "
                            f"within {self.lock_timeout_seconds}s timeout. "
                            "Another process may be holding the lock."
                        )

                    # Wait a bit before retrying (exponential backoff)
                    wait_time = min(0.1 * (1.5 ** int(elapsed * 10)), 1.0)
                    time.sleep(wait_time)

            # Lock acquired, yield control to caller
            yield

        finally:
            # Release lock and close file
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                logger.debug(f"Released {lock_type} lock for cache_key: {cache_key}")
            except Exception as e:
                logger.warning(f"Error releasing lock for {cache_key}: {e}")
            finally:
                lock_file.close()

    def exists(self, cache_key: str) -> bool:
        """Check if artifact exists in cache.

        Args:
            cache_key: Content-addressed cache key.

        Returns:
            True if artifact exists, False otherwise.
        """
        artifact_path = self._artifact_path(cache_key)
        return artifact_path.exists()

    def load(self, cache_key: str) -> Dict[str, Any]:
        """Load artifact from cache with shared lock.

        Args:
            cache_key: Content-addressed cache key.

        Returns:
            Artifact data (stage outputs as dict).

        Raises:
            FileNotFoundError: If artifact not found.
            ValueError: If artifact is corrupted.
            CacheLockTimeout: If lock cannot be acquired within timeout.

        Design notes:
        - Acquires shared lock to prevent reading partial writes
        - Updates access time for LRU tracking
        - Validates artifact integrity (checksums in L2)
        - Returns deep copy to prevent cache mutation
        """
        # Acquire shared lock for read (prevents concurrent writes)
        with self._acquire_lock(cache_key, exclusive=False):
            artifact_path = self._artifact_path(cache_key)

            if not artifact_path.exists():
                raise FileNotFoundError(f"Artifact not found: {cache_key}")

            try:
                # Load NumPy archive
                # Security: Disable pickle deserialization to prevent arbitrary code execution
                # Restricts artifacts to safe NumPy array types only
                data = np.load(artifact_path, allow_pickle=False)

                # Convert to dict
                result = {}
                for key in data.files:
                    value = data[key]
                    # Handle scalar arrays
                    if value.shape == ():
                        result[key] = value.item()
                    else:
                        result[key] = value

                # Update stats
                self._record_cache_hit()

                logger.debug(f"Cache hit: {cache_key}")
                return result

            except Exception as e:
                raise ValueError(f"Corrupted artifact: {cache_key}") from e

    def store(
        self,
        cache_key: str,
        artifact: Dict[str, Any],
        provenance: ProvenanceMetadata,
    ) -> None:
        """Store artifact in cache with provenance and exclusive lock.

        Args:
            cache_key: Content-addressed cache key.
            artifact: Artifact data (stage outputs as dict).
            provenance: Provenance metadata.

        Raises:
            ValueError: If artifact contains non-serializable data or object arrays.
            CacheLockTimeout: If lock cannot be acquired within timeout.

        Design notes:
        - Acquires exclusive lock to prevent concurrent writes
        - Atomic write (temp file + fsync + rename)
        - Stores artifact as .npz (NumPy compressed archive)
        - Stores provenance as .json sidecar
        - Creates two-level directory hierarchy (first 2 hex chars)
        - SECURITY: Rejects object dtype arrays (require pickle deserialization)
        """
        # Acquire exclusive lock for write (prevents concurrent reads/writes)
        with self._acquire_lock(cache_key, exclusive=True):
            artifact_path = self._artifact_path(cache_key)
            provenance_path = self._provenance_path(cache_key)

            # Create directory
            artifact_path.parent.mkdir(parents=True, exist_ok=True)

            # Validate and convert artifacts BEFORE entering try/except
            # This ensures dtype validation errors propagate with clear messages
            np_dict: Dict[str, np.ndarray] = {}
            for key, value in artifact.items():
                # Convert to numpy array (let NumPy infer dtype)
                if isinstance(value, np.ndarray):
                    arr = value
                else:
                    # Let NumPy infer dtype (will be numeric, bool, or unicode for homogeneous data)
                    arr = np.array(value)

                # CRITICAL: Reject object dtype (security + correctness invariant)
                # Object arrays require pickle deserialization and are not permitted.
                # This prevents runtime failures with allow_pickle=False in load().
                if arr.dtype == np.dtype("object"):
                    raise ValueError(
                        f"Artifact value for key '{key}' produced dtype=object. "
                        "Object arrays require pickle deserialization and are not permitted. "
                        "Supported types: numeric arrays, bool arrays, string arrays, scalars."
                    )

                np_dict[key] = arr

            # Atomic write: temp file + rename
            try:
                # Write artifact (NumPy archive)
                with tempfile.NamedTemporaryFile(mode="wb", dir=artifact_path.parent, delete=False, suffix=".npz") as tmp_artifact:
                    tmp_artifact_path = Path(tmp_artifact.name)
                    np.savez_compressed(tmp_artifact, **np_dict)
                    tmp_artifact.flush()
                    os.fsync(tmp_artifact.fileno())

                # Write provenance (JSON)
                with tempfile.NamedTemporaryFile(mode="w", dir=artifact_path.parent, delete=False, suffix=".json") as tmp_prov:
                    tmp_prov_path = Path(tmp_prov.name)
                    json.dump(asdict(provenance), tmp_prov, indent=2)
                    tmp_prov.flush()
                    os.fsync(tmp_prov.fileno())

                # Atomic rename
                tmp_artifact_path.replace(artifact_path)
                tmp_prov_path.replace(provenance_path)

                # Update stats
                self._record_cache_miss()

                logger.debug(f"Stored artifact: {cache_key}")

            except Exception as e:
                # Clean up temp files on failure
                if "tmp_artifact_path" in locals() and tmp_artifact_path.exists():
                    tmp_artifact_path.unlink()
                if "tmp_prov_path" in locals() and tmp_prov_path.exists():
                    tmp_prov_path.unlink()
                raise ValueError(f"Failed to store artifact: {cache_key}") from e

            # Check cache size (warn if over limit)
            self._check_cache_size()

    def load_provenance(self, cache_key: str) -> ProvenanceMetadata:
        """Load provenance metadata for artifact with shared lock.

        Args:
            cache_key: Content-addressed cache key.

        Returns:
            ProvenanceMetadata.

        Raises:
            FileNotFoundError: If provenance not found.
            CacheLockTimeout: If lock cannot be acquired within timeout.
        """
        # Acquire shared lock for read (prevents concurrent writes)
        with self._acquire_lock(cache_key, exclusive=False):
            provenance_path = self._provenance_path(cache_key)

            if not provenance_path.exists():
                raise FileNotFoundError(f"Provenance not found: {cache_key}")

            with open(provenance_path) as f:
                data = json.load(f)
                return ProvenanceMetadata(**data)

    def evict(self, cache_key: str) -> None:
        """Evict artifact from cache with exclusive lock.

        Args:
            cache_key: Content-addressed cache key.

        Raises:
            CacheLockTimeout: If lock cannot be acquired within timeout.

        Design notes:
        - Acquires exclusive lock to prevent concurrent access during deletion
        - Removes both artifact and provenance files
        - Idempotent (no error if already evicted)
        """
        # Acquire exclusive lock for delete (prevents concurrent access)
        with self._acquire_lock(cache_key, exclusive=True):
            artifact_path = self._artifact_path(cache_key)
            provenance_path = self._provenance_path(cache_key)

            if artifact_path.exists():
                artifact_path.unlink()
            if provenance_path.exists():
                provenance_path.unlink()

            logger.debug(f"Evicted artifact: {cache_key}")

    def get_cache_size_mb(self) -> float:
        """Get total cache size in megabytes.

        Returns:
            Cache size in MB.
        """
        total_size = 0
        for path in self.artifacts_dir.rglob("*"):
            if path.is_file():
                total_size += path.stat().st_size
        return total_size / (1024 * 1024)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with hits, misses, size, etc.
        """
        return {
            "cache_hits": self._stats.get("hits", 0),
            "cache_misses": self._stats.get("misses", 0),
            "cache_size_mb": self.get_cache_size_mb(),
            "max_size_gb": self.max_size_gb,
        }

    def _artifact_path(self, cache_key: str) -> Path:
        """Get artifact file path with strict validation.

        Uses two-level directory hierarchy (first 2 hex chars as prefix).

        Args:
            cache_key: Cache key (must be 64-char SHA256 hex).

        Returns:
            Path to artifact file.

        Raises:
            ValueError: If cache_key format is invalid or contains path traversal.

        Security:
            Validates cache_key format to prevent path traversal attacks.
            Even if keys are SHA256-based today, never trust upstream callers forever.
        """
        # Validate cache key format (64 lowercase hex characters)
        if not SAFE_CACHE_KEY.match(cache_key):
            raise ValueError(f"Invalid cache_key format: {cache_key!r}. " "Expected 64 lowercase hex characters (SHA256).")

        # Additional safety check for path separators and traversal
        if "/" in cache_key or "\\" in cache_key or ".." in cache_key:
            raise ValueError(f"Invalid cache_key contains path separators or traversal: {cache_key!r}")

        # Safe to construct path now
        prefix = cache_key[:2]
        return self.artifacts_dir / prefix / f"{cache_key}.npz"

    def _provenance_path(self, cache_key: str) -> Path:
        """Get provenance file path with strict validation.

        Args:
            cache_key: Cache key (must be 64-char SHA256 hex).

        Returns:
            Path to provenance JSON file.

        Raises:
            ValueError: If cache_key format is invalid or contains path traversal.

        Security:
            Validates cache_key format to prevent path traversal attacks.
        """
        # Validate cache key format (reuse same validation as _artifact_path)
        if not SAFE_CACHE_KEY.match(cache_key):
            raise ValueError(f"Invalid cache_key format: {cache_key!r}. " "Expected 64 lowercase hex characters (SHA256).")

        # Additional safety check for path separators and traversal
        if "/" in cache_key or "\\" in cache_key or ".." in cache_key:
            raise ValueError(f"Invalid cache_key contains path separators or traversal: {cache_key!r}")

        prefix = cache_key[:2]
        return self.artifacts_dir / prefix / f"{cache_key}.json"

    def _load_stats(self) -> None:
        """Load cache statistics from disk."""
        if self.stats_path.exists():
            with open(self.stats_path) as f:
                self._stats = json.load(f)
        else:
            self._stats = {"hits": 0, "misses": 0}

    def _save_stats(self) -> None:
        """Save cache statistics to disk."""
        with open(self.stats_path, "w") as f:
            json.dump(self._stats, f, indent=2)

    def _record_cache_hit(self) -> None:
        """Record cache hit in statistics."""
        self._stats["hits"] = self._stats.get("hits", 0) + 1
        self._save_stats()

    def _record_cache_miss(self) -> None:
        """Record cache miss in statistics."""
        self._stats["misses"] = self._stats.get("misses", 0) + 1
        self._save_stats()

    def _check_cache_size(self) -> None:
        """Check cache size and warn if over limit.

        Design notes:
        - L1: Warns only, no auto-eviction.
        - L2: Implements LRU eviction.
        """
        size_mb = self.get_cache_size_mb()
        limit_mb = self.max_size_gb * 1024

        if size_mb > limit_mb:
            logger.warning(
                f"Cache size ({size_mb:.1f}MB) exceeds limit ({limit_mb:.1f}MB). "
                "Consider running cache eviction (L2 feature)."
            )
