"""Content-addressed artifact store with provenance tracking (Phase 3 L1).

Provides deterministic caching for pipeline artifacts with full provenance metadata.

Design Principles (ADR-029):
- Content-addressed (same inputs → same cache key → same artifact)
- Atomic writes (temp + fsync + rename, no partial artifacts)
- Provenance metadata (input hashes, model revisions, timestamps)
- Determinism verification (cache hit = bitwise identical output)
- Multi-process safe operations (per-key file locks for concurrent access)

Lock Ordering Invariant (Issue #925):
    If both per-key lock and stats lock are required in the same operation,
    ALWAYS acquire per-key lock(s) first, then stats lock.
    This prevents AB/BA deadlock hazards in future features.

Storage Layout:
    .cache/spatial_ai/
    ├── artifacts/
    │   ├── ab/
    │   │   ├── ab3f5e8b2c1d4.npz        # Artifact data
    │   │   ├── ab3f5e8b2c1d4.json       # Provenance metadata
    │   │   └── ab3f5e8b2c1d4.committed  # Commit marker (transactional)
    │   └── ...
    ├── locks/                           # Per-key lock files
    │   ├── ab3f5e8b2c1d4.lock
    │   └── ...
    ├── stats.lock                       # Global stats lock
    └── stats.json                       # Cache statistics (size, hits, misses)

Transactional Commit (Issue #929):
    Artifact + provenance are written atomically via temp+fsync+rename, then a
    ".committed" marker file is atomically created (also temp+fsync+rename).
    Readers only trust entries where the marker is present. This ensures
    all-or-nothing visibility: if the process crashes between writing
    artifact/provenance and creating the marker, the entry is treated as
    uncommitted and ignored (cleaned up by the scavenger).

    **Cache Entry Visibility Invariant:**
    A cache entry is visible (exists/loadable) iff ALL of:
    - `<key>.npz` exists (artifact payload)
    - `<key>.json` exists (provenance metadata)
    - `<key>.committed` exists (transactional marker)
    - No active writer holds the per-key exclusive lock

    **Scavenger Safety Invariant:**
    Scavenger cleanup is concurrent-safe iff:
    - Shared per-key locking protocol (scavenger acquires lock before delete)
    - Age threshold > maximum expected rename→marker latency
    - Re-check marker after lock acquisition (TOCTOU prevention)

    **Durability Contract (Two Tiers):**
    Tier 1 (durable_commits=False, default):
    - store() success means: atomic visibility (survives process crash)
    - Payload + marker fsynced to disk (file durability)
    - Directory metadata NOT fsynced (may lose entries on power loss)
    - Best for: ephemeral/recomputable cache workloads

    Tier 2 (durable_commits=True, opt-in):
    - store() success means: durable across power loss/OS crash
    - Payload + marker fsynced + directory fsynced (POSIX durability)
    - Adds ~1-10ms overhead per store() (filesystem-dependent)
    - Best for: expensive-to-regenerate entries on persistent volumes

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
import json
import logging
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import numpy as np

# Platform-specific imports (guarded for cross-platform safety)
try:
    import fcntl

    _HAVE_FCNTL = True
except ImportError:
    fcntl = None  # type: ignore
    _HAVE_FCNTL = False

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
    - Shared locks for reads (load operations)
    - Lock files stored in locks/ directory
    - Automatic lock release on operation completion
    - Configurable timeout to prevent indefinite hangs

    Platform requirements:
    - Requires POSIX-compliant OS (fcntl.flock)
    - Requires local filesystem for correctness (advisory locks)
    - Not supported on Windows
    - May behave incorrectly on some network filesystems (NFS, etc.)

    Thread safety:
    - Atomic writes (temp file + rename)
    - File locks are process-safe and thread-safe
    - Write operations use OS-level atomic rename

    Lock-free operations (advisory only):
    - exists() does not acquire locks (may return stale data)
    - Correctness guarantees only apply to load()/store()/evict()
    - Do not build critical logic on exists() alone - use load() for safety

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
        durable_commits: bool = False,
    ):
        """Initialize artifact store.

        Args:
            cache_dir: Base directory for cache storage.
            max_size_gb: Maximum cache size in gigabytes (warns at limit).
            eviction_policy: Eviction policy ("lru" or "manual").
            lock_timeout_seconds: Timeout for lock acquisition (default 30s).
            durable_commits: If True, fsync containing directory after commit
                marker creation to ensure durability across power loss.
                Default False (performance > durability for cache workloads).

        Design notes:
        - Creates cache_dir if it doesn't exist.
        - Initializes artifacts/ and locks/ subdirectories.
        - Logs warning if cache size exceeds limit (no auto-eviction in L1).

        Durability semantics:
        - durable_commits=False (default): Atomic visibility (survives process
          crash, may lose entries on OS/power loss). Recommended for caches.
        - durable_commits=True: Full crash durability (survives power loss).
          Adds directory fsync overhead. Use for quasi-storage workloads.
        """
        self.cache_dir = Path(cache_dir)
        self.artifacts_dir = self.cache_dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Lock directory for per-key locks (Phase 3 L1)
        self.locks_dir = self.cache_dir / "locks"
        self.locks_dir.mkdir(parents=True, exist_ok=True)

        # Stats lock for global stats.json (Issue #925)
        self.stats_lock_path = self.cache_dir / "stats.lock"

        self.max_size_gb = max_size_gb
        self.eviction_policy = eviction_policy
        self.lock_timeout_seconds = lock_timeout_seconds
        self.durable_commits = durable_commits
        self.eviction_policy = eviction_policy
        self.lock_timeout_seconds = lock_timeout_seconds
        self.durable_commits = durable_commits

        # Cache statistics
        self.stats_path = self.cache_dir / "stats.json"
        self._load_stats()

    def _fsync_directory(self, directory: Path) -> None:
        """Fsync a directory to ensure metadata durability (opt-in elite tier).

        Args:
            directory: Directory path to fsync.

        Design notes (crash durability):
        - Called only when durable_commits=True
        - Ensures directory metadata (rename operations) persists to disk
        - Required for durability across OS/power loss (not just process crash)
        - On POSIX: fsync() on directory fd flushes directory entries
        - On Windows: no-op (directory fsync not supported/needed)
        - Performance cost: adds ~1-10ms per store() depending on filesystem

        References:
        - PostgreSQL durable_rename(): fsync file + directory
        - Linux fsync(2): "For a directory, ensures entries are persistent"
        """
        if not self.durable_commits:
            return

        # Platform check: directory fsync only works on POSIX
        if not _HAVE_FCNTL:
            logger.debug("Skipping directory fsync (non-POSIX platform)")
            return

        try:
            # Open directory and fsync to flush metadata
            dir_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
                logger.debug(f"Fsynced directory: {directory}")
            finally:
                os.close(dir_fd)
        except OSError as e:
            # Log but don't fail - this is a durability optimization
            # Some filesystems/mounts may not support directory fsync
            logger.warning(f"Directory fsync failed for {directory}: {e}")

    @contextlib.contextmanager
    def _acquire_lock(
        self,
        cache_key: str,
        exclusive: bool = True,
        timeout: Optional[float] = None,
    ) -> Generator[None, None, None]:
        """Acquire per-key file lock for cache operations.

        Args:
            cache_key: Content-addressed cache key (must be valid SHA256 hex).
            exclusive: If True, acquire exclusive (write) lock. If False, acquire
                      shared (read) lock.
            timeout: Optional lock timeout in seconds. If None, uses self.lock_timeout_seconds.
                    Use small values (e.g., 0.1) for non-blocking scavenger.

        Yields:
            None (context manager)

        Raises:
            CacheLockTimeout: If lock cannot be acquired within timeout.
            ValueError: If cache_key format is invalid.

        Design notes (Phase 3 L1):
        - Per-key locking maximizes concurrency (different keys don't block each other)
        - Lock files stored in locks/ directory: locks/<cache_key>.lock
        - Exclusive locks for writes prevent corruption during store operations
        - Shared locks for reads allow concurrent readers
        - Lock acquisition uses non-blocking retries with timeout
        - Locks released automatically via context manager (even on exceptions)
        - Lock files never deleted (reused across operations for same key)

        Platform requirements:
        - Requires POSIX-compliant OS (fcntl.flock)
        - Requires local filesystem semantics for correctness
        - Not supported on Windows (flock not available)
        - May behave incorrectly on some network filesystems (e.g., NFS with certain mount options)

        Thread safety:
        - fcntl locks are process-safe AND thread-safe within same process
        - Multiple threads in same process will serialize on same lock file

        Multi-process safety:
        - Prevents partial reads (reader blocks until writer completes)
        - Prevents write corruption (only one writer at a time per key)
        - Different keys can be accessed concurrently without blocking
        """
        # Platform check: Require fcntl for multi-process locking
        if not _HAVE_FCNTL:
            raise RuntimeError(
                "Per-key file locking requires POSIX fcntl.flock support. "
                "This environment does not provide it (Windows or non-POSIX platform). "
                "ArtifactStore cannot guarantee multi-process safety without fcntl."
            )

        # Validate cache_key format (security + correctness)
        if not SAFE_CACHE_KEY.match(cache_key):
            raise ValueError(f"Invalid cache_key format: {cache_key!r}. " "Expected 64 lowercase hex characters (SHA256).")

        # Use provided timeout or fall back to instance timeout
        lock_timeout = timeout if timeout is not None else self.lock_timeout_seconds

        # Lock file path: locks/<cache_key>.lock
        lock_path = self.locks_dir / f"{cache_key}.lock"

        # Create lock file if it doesn't exist (idempotent)
        lock_path.touch(exist_ok=True)

        # Open lock file with context manager to ensure cleanup
        with open(lock_path, "r+") as lock_file:
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
                # Use monotonic clock to avoid system time changes (NTP corrections, VM time skew)
                start_time = time.monotonic()
                acquired = False
                attempt = 0

                while not acquired:
                    try:
                        # Try non-blocking lock
                        fcntl.flock(lock_file.fileno(), lock_mode | fcntl.LOCK_NB)
                        acquired = True
                        logger.debug(f"Acquired {lock_type} lock for cache_key: {cache_key}")

                    except BlockingIOError:
                        # Lock held by another process/thread
                        elapsed = time.monotonic() - start_time
                        if elapsed >= lock_timeout:
                            raise CacheLockTimeout(
                                f"Could not acquire {lock_type} lock for cache_key {cache_key} "
                                f"within {lock_timeout}s timeout. "
                                "Another process may be holding the lock."
                            )

                        # Exponential backoff with cap (100ms base, 2x multiplier, 1s max)
                        attempt += 1
                        wait_time = min(0.1 * (2**attempt), 1.0)
                        time.sleep(wait_time)

                # Lock acquired, yield control to caller
                yield

            finally:
                # Release lock (file close handled by context manager)
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                    logger.debug(f"Released {lock_type} lock for cache_key: {cache_key}")
                except Exception as e:
                    logger.warning(f"Error releasing lock for {cache_key}: {e}")

    @contextlib.contextmanager
    def _acquire_stats_lock(self) -> Generator[None, None, None]:
        """Acquire global lock for stats.json operations.

        Yields:
            None (context manager)

        Raises:
            RuntimeError: If fcntl is not available (non-POSIX platform).
            CacheLockTimeout: If lock cannot be acquired within timeout.

        Design notes (Issue #925):
        - Global lock for global shared state (stats.json)
        - Per-key locks cannot serialize stats access across different keys
        - Lock cost is negligible compared to I/O + NumPy operations
        - Prevents corrupted/lost increments under multi-process contention
        - Uses timeout to prevent indefinite hangs from wedged processes
        """
        # Platform check: Require fcntl for stats locking
        if not _HAVE_FCNTL:
            raise RuntimeError(
                "Stats locking requires POSIX fcntl.flock support. "
                "This environment does not provide it (Windows or non-POSIX platform)."
            )

        # Create stats lock file if it doesn't exist (idempotent)
        self.stats_lock_path.touch(exist_ok=True)

        # Open lock file with context manager to ensure cleanup
        with open(self.stats_lock_path, "r+") as lock_file:
            # Acquire exclusive lock with timeout (same as per-key locks)
            start_time = time.monotonic()
            attempt = 0

            while True:
                try:
                    # Try non-blocking lock
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    logger.debug("Acquired stats lock")
                    break
                except BlockingIOError:
                    # Lock held by another process
                    elapsed = time.monotonic() - start_time
                    if elapsed >= self.lock_timeout_seconds:
                        raise CacheLockTimeout(f"Failed to acquire stats lock after {elapsed:.1f}s")

                    # Exponential backoff with cap
                    backoff = min(0.1 * (2**attempt), 1.0)
                    time.sleep(backoff)
                    attempt += 1

            try:
                # Lock acquired, yield control to caller
                yield

            finally:
                # Release lock (file close handled by context manager)
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                    logger.debug("Released stats lock")
                except Exception as e:
                    logger.warning(f"Error releasing stats lock: {e}")

    def exists(self, cache_key: str) -> bool:
        """Check if committed artifact exists in cache (lock-free, advisory only).

        Args:
            cache_key: Content-addressed cache key.

        Returns:
            True if artifact exists and is committed, False otherwise.

        Warning:
            This method does not acquire locks and may return stale data
            due to concurrent operations. Do not build critical logic on
            exists() alone - use load() for correctness guarantees.

            Safe patterns:
            - if store.exists(key): result = store.load(key)  # OK (load locks)
            - if not store.exists(key): store.store(key, ...)  # OK (store locks)

            Unsafe patterns:
            - if store.exists(key): do_critical_action()  # WRONG (race condition)

        Design notes (Issue #929):
            Requires the .committed marker to be present. Uncommitted entries
            (artifact without marker) are treated as non-existent.

            Also checks for corruption: if marker exists but payload is missing,
            returns False (entry is unusable).
        """
        artifact_path = self._artifact_path(cache_key)
        provenance_path = self._provenance_path(cache_key)
        committed_path = self._committed_path(cache_key)

        # Entry is valid only if marker AND both payload files exist
        return artifact_path.exists() and provenance_path.exists() and committed_path.exists()

    def load(self, cache_key: str) -> Dict[str, Any]:
        """Load artifact from cache with shared lock.

        Args:
            cache_key: Content-addressed cache key.

        Returns:
            Artifact data (stage outputs as dict).

        Raises:
            FileNotFoundError: If artifact not found or not committed.
            ValueError: If artifact is corrupted.
            CacheLockTimeout: If lock cannot be acquired within timeout.

        Design notes:
        - Acquires shared lock to prevent reading partial writes
        - Requires .committed marker (Issue #929: transactional visibility)
        - Updates access time for LRU tracking
        - Validates artifact integrity (checksums in L2)
        - Returns deep copy to prevent cache mutation
        """
        # Acquire shared lock for read (prevents concurrent writes)
        with self._acquire_lock(cache_key, exclusive=False):
            artifact_path = self._artifact_path(cache_key)
            provenance_path = self._provenance_path(cache_key)
            committed_path = self._committed_path(cache_key)

            # Reader self-healing: if marker exists but payload is missing,
            # treat as corruption and raise FileNotFoundError.
            # This handles pre-fix corruption or manual tampering.
            marker_exists = committed_path.exists()
            artifact_exists = artifact_path.exists()
            provenance_exists = provenance_path.exists()

            # Case 1: No marker → cache miss (normal)
            if not marker_exists:
                raise FileNotFoundError(f"Artifact not found: {cache_key}")

            # Case 2: Marker exists but payload missing → corruption
            if not artifact_exists or not provenance_exists:
                logger.warning(
                    f"Cache corruption detected for {cache_key}: "
                    f"marker exists but payload missing "
                    f"(artifact={artifact_exists}, provenance={provenance_exists}). "
                    "Treating as cache miss."
                )
                raise FileNotFoundError(f"Artifact corrupted: {cache_key}")

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
        - Creates .committed marker for transactional visibility (Issue #929)
        - Creates two-level directory hierarchy (first 2 hex chars)
        - SECURITY: Rejects object dtype arrays (require pickle deserialization)
        """
        # Acquire exclusive lock for write (prevents concurrent reads/writes)
        with self._acquire_lock(cache_key, exclusive=True):
            artifact_path = self._artifact_path(cache_key)
            provenance_path = self._provenance_path(cache_key)
            committed_path = self._committed_path(cache_key)

            # Idempotency check: if already committed, skip write (Issue #929 correctness)
            # This prevents overwrite hazards when multiple processes compute same key.
            # If crash happens during overwrite, old .committed marker could point to
            # mismatched artifact/provenance pair.
            if committed_path.exists():
                if artifact_path.exists() and provenance_path.exists():
                    logger.debug(f"Store skipped (already committed): {cache_key}")
                    return
                else:
                    # Marker exists but files missing → corruption, invalidate and rebuild
                    logger.warning(f"Corrupted entry detected (marker without files): {cache_key}")
                    committed_path.unlink(missing_ok=True)

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
                with tempfile.NamedTemporaryFile(
                    mode="wb", dir=artifact_path.parent, delete=False, suffix=".npz"
                ) as tmp_artifact:
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

                # Atomic rename (artifact + provenance)
                tmp_artifact_path.replace(artifact_path)
                tmp_prov_path.replace(provenance_path)

                # Atomic commit marker (Issue #929: transactional visibility)
                # Only after both artifact and provenance are in place,
                # atomically create the .committed marker via temp+fsync+rename.
                # Fsync maintains crash consistency symmetry with artifact/provenance.
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    dir=artifact_path.parent,
                    delete=False,
                    suffix=".committed_tmp",
                ) as tmp_marker:
                    tmp_marker_path = Path(tmp_marker.name)
                    tmp_marker.flush()
                    os.fsync(tmp_marker.fileno())
                tmp_marker_path.replace(committed_path)

                # Optional: fsync containing directory for crash durability
                # (elite tier - ensures rename persists across power loss)
                self._fsync_directory(artifact_path.parent)

                # Update stats
                self._record_cache_miss()

                logger.debug(f"Stored artifact: {cache_key}")

            except Exception as e:
                # Clean up temp files AND uncommitted artifacts on failure
                # Without marker, these are by definition uncommitted junk
                if "tmp_artifact_path" in locals() and tmp_artifact_path.exists():
                    tmp_artifact_path.unlink()
                if "tmp_prov_path" in locals() and tmp_prov_path.exists():
                    tmp_prov_path.unlink()
                if "tmp_marker_path" in locals() and tmp_marker_path.exists():
                    tmp_marker_path.unlink()

                # Best-effort cleanup of renamed-but-uncommitted files
                artifact_path.unlink(missing_ok=True)
                provenance_path.unlink(missing_ok=True)

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
            FileNotFoundError: If provenance not found or not committed.
            CacheLockTimeout: If lock cannot be acquired within timeout.

        Design notes (Issue #929):
            Requires .committed marker for transactional visibility.
        """
        # Acquire shared lock for read (prevents concurrent writes)
        with self._acquire_lock(cache_key, exclusive=False):
            artifact_path = self._artifact_path(cache_key)
            provenance_path = self._provenance_path(cache_key)
            committed_path = self._committed_path(cache_key)

            # Reader self-healing: check for corruption (same logic as load())
            marker_exists = committed_path.exists()
            artifact_exists = artifact_path.exists()
            provenance_exists = provenance_path.exists()

            if not marker_exists:
                raise FileNotFoundError(f"Provenance not found: {cache_key}")

            if not artifact_exists or not provenance_exists:
                logger.warning(
                    f"Cache corruption detected for {cache_key}: "
                    f"marker exists but payload missing "
                    f"(artifact={artifact_exists}, provenance={provenance_exists}). "
                    "Treating as cache miss."
                )
                raise FileNotFoundError(f"Provenance corrupted: {cache_key}")

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
        - Removes committed marker, artifact, and provenance files
        - Idempotent (no error if already evicted)
        """
        # Acquire exclusive lock for delete (prevents concurrent access)
        with self._acquire_lock(cache_key, exclusive=True):
            artifact_path = self._artifact_path(cache_key)
            provenance_path = self._provenance_path(cache_key)
            committed_path = self._committed_path(cache_key)

            # Remove marker first (ensures readers see entry as uncommitted)
            if committed_path.exists():
                committed_path.unlink()
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

    def _committed_path(self, cache_key: str) -> Path:
        """Get commit marker file path with strict validation.

        Args:
            cache_key: Cache key (must be 64-char SHA256 hex).

        Returns:
            Path to commit marker file.

        Raises:
            ValueError: If cache_key format is invalid or contains path traversal.

        Design notes (Issue #929):
            The .committed marker signals that artifact + provenance were
            both written successfully. Readers only trust entries with this
            marker present.
        """
        # Validate cache key format (reuse same validation as _artifact_path)
        if not SAFE_CACHE_KEY.match(cache_key):
            raise ValueError(f"Invalid cache_key format: {cache_key!r}. " "Expected 64 lowercase hex characters (SHA256).")

        # Additional safety check for path separators and traversal
        if "/" in cache_key or "\\" in cache_key or ".." in cache_key:
            raise ValueError(f"Invalid cache_key contains path separators or traversal: {cache_key!r}")

        prefix = cache_key[:2]
        return self.artifacts_dir / prefix / f"{cache_key}.committed"

    def scavenge(self, max_temp_age_seconds: float = 300.0) -> Dict[str, int]:
        """Remove orphaned artifacts and stale temp files (Issue #929).

        Scans the artifacts directory for:
        1. Uncommitted entries: .npz or .json files without a matching
           .committed marker → removed (orphaned from a crashed store).
        2. Stale temp files: files with temp-like suffixes older than
           max_temp_age_seconds → removed.

        Args:
            max_temp_age_seconds: Maximum age (in seconds) for temp files
                before they are considered stale and removed. Default 300s (5 min).

        Returns:
            Cleanup report dict with keys:
            - orphaned_artifacts_removed: count of .npz files removed
            - orphaned_provenance_removed: count of .json files removed
            - stale_temp_files_removed: count of temp files removed

        Design notes (concurrent safety):
        - Acquires per-key locks before deleting uncommitted entries
        - Only deletes uncommitted files older than max_temp_age_seconds (age grace period)
        - Re-checks marker presence after acquiring lock (TOCTOU prevention)
        - Scavenger is best-effort: skips keys where lock can't be acquired
        - Safe to run concurrently with store/load operations:
          (a) Age check prevents deleting fresh in-flight writes
          (b) Lock check prevents deleting files held by active writers
          (c) Re-check after lock prevents race between check and delete
        - Logs each removal at DEBUG level for audit trail.
        """
        report = {
            "orphaned_artifacts_removed": 0,
            "orphaned_provenance_removed": 0,
            "stale_temp_files_removed": 0,
        }

        # Temp file suffixes produced by store() and _save_stats_atomic()
        _TEMP_SUFFIXES = (".npz", ".json", ".committed_tmp")

        # Track uncommitted entries by cache key (to process as units)
        uncommitted_keys: Dict[str, List[Path]] = {}
        now = time.time()

        for prefix_dir in self.artifacts_dir.iterdir():
            if not prefix_dir.is_dir():
                continue

            for entry in prefix_dir.iterdir():
                if not entry.is_file():
                    continue

                name = entry.name

                # --- Stale temp files ---
                # NamedTemporaryFile produces names like tmpXXXXXX.suffix
                if name.startswith("tmp") and any(name.endswith(s) for s in _TEMP_SUFFIXES):
                    try:
                        age = now - entry.stat().st_mtime
                        if age > max_temp_age_seconds:
                            entry.unlink(missing_ok=True)
                            report["stale_temp_files_removed"] += 1
                            logger.debug(f"Scavenger: removed stale temp file {entry}")
                    except OSError:
                        pass  # File may have been removed concurrently
                    continue

                # --- Collect uncommitted entries by key ---
                if name.endswith(".npz") and SAFE_CACHE_KEY.match(name[:-4]):
                    cache_key = name[:-4]
                    committed = prefix_dir / f"{cache_key}.committed"
                    if not committed.exists():
                        if cache_key not in uncommitted_keys:
                            uncommitted_keys[cache_key] = []
                        uncommitted_keys[cache_key].append(entry)
                    continue

                if name.endswith(".json") and SAFE_CACHE_KEY.match(name[:-5]):
                    cache_key = name[:-5]
                    committed = prefix_dir / f"{cache_key}.committed"
                    if not committed.exists():
                        if cache_key not in uncommitted_keys:
                            uncommitted_keys[cache_key] = []
                        uncommitted_keys[cache_key].append(entry)
                    continue

        # --- Process uncommitted entries with lock + age checks ---
        for cache_key, files in uncommitted_keys.items():
            # Age check: only scavenge old uncommitted files
            try:
                mtimes = [f.stat().st_mtime for f in files if f.exists()]
                if not mtimes:
                    continue  # Files deleted concurrently

                newest = max(mtimes)
                age = now - newest

                if age < max_temp_age_seconds:
                    logger.debug(f"Scavenger: skipping fresh uncommitted key {cache_key} (age={age:.1f}s)")
                    continue
            except OSError:
                continue  # Stat failed, skip this key

            # Lock check: try to acquire per-key lock (non-blocking)
            try:
                with self._acquire_lock(cache_key, exclusive=True, timeout=0.1):
                    # Re-check marker after lock acquisition (TOCTOU prevention)
                    committed_path = self._committed_path(cache_key)
                    if committed_path.exists():
                        continue  # Became committed while we waited for lock

                    # Safe to delete uncommitted files
                    artifact_path = self._artifact_path(cache_key)
                    prov_path = self._provenance_path(cache_key)

                    try:
                        if artifact_path.exists():
                            artifact_path.unlink()
                            report["orphaned_artifacts_removed"] += 1
                            logger.debug(f"Scavenger: removed orphaned artifact {artifact_path}")
                    except FileNotFoundError:
                        pass  # Deleted concurrently
                    except OSError as e:
                        logger.debug(f"Scavenger: failed to remove {artifact_path} ({e})")

                    try:
                        if prov_path.exists():
                            prov_path.unlink()
                            report["orphaned_provenance_removed"] += 1
                            logger.debug(f"Scavenger: removed orphaned provenance {prov_path}")
                    except FileNotFoundError:
                        pass
                    except OSError as e:
                        logger.debug(f"Scavenger: failed to remove {prov_path} ({e})")

            except CacheLockTimeout:
                # Writer holding lock, skip this key
                logger.debug(f"Scavenger: skipping locked key {cache_key}")
                continue

        logger.info(
            f"Scavenger complete: {report['orphaned_artifacts_removed']} orphaned artifacts, "
            f"{report['orphaned_provenance_removed']} orphaned provenance, "
            f"{report['stale_temp_files_removed']} stale temp files removed"
        )
        return report

    def _load_stats(self) -> None:
        """Load cache statistics from disk (with locking).

        Design notes (Issue #925):
        - Acquires stats lock to prevent concurrent access
        - Initializes to zero if stats.json doesn't exist
        """
        with self._acquire_stats_lock():
            if self.stats_path.exists():
                with open(self.stats_path) as f:
                    self._stats = json.load(f)
            else:
                self._stats = {"hits": 0, "misses": 0}

    def _save_stats_atomic(self) -> None:
        """Save cache statistics to disk atomically.

        Design notes (Issue #925):
        - Uses temp → fsync → rename pattern (same as artifacts)
        - Atomic write prevents corruption from crashes mid-write
        - Temp file cleaned up on failure (disk full, permissions, etc.)
        - Caller must hold stats lock
        """
        # Write to temp file in same directory (ensures same filesystem for atomic rename)
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=self.cache_dir,
            delete=False,
            suffix=".json",
            prefix=".stats_tmp_",
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
            json.dump(self._stats, tmp_file, indent=2)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())

        # Atomic rename with cleanup on failure
        try:
            tmp_path.replace(self.stats_path)
        except Exception:
            # Clean up temp file on failure (disk full, permissions, etc.)
            tmp_path.unlink(missing_ok=True)
            raise

    def _record_cache_hit(self) -> None:
        """Record cache hit in statistics (with locking and atomic write).

        Design notes (Issue #925):
        - Acquires global stats lock to prevent lost increments
        - Reloads stats from disk inside lock to ensure consistency
        - Uses atomic write to prevent corruption
        """
        with self._acquire_stats_lock():
            # Reload from disk to get latest counts
            if self.stats_path.exists():
                with open(self.stats_path) as f:
                    current_stats = json.load(f)
            else:
                current_stats = {"hits": 0, "misses": 0}

            # Increment
            current_stats["hits"] = current_stats.get("hits", 0) + 1

            # Save atomically
            self._stats = current_stats
            self._save_stats_atomic()

    def _record_cache_miss(self) -> None:
        """Record cache miss in statistics (with locking and atomic write).

        Design notes (Issue #925):
        - Acquires global stats lock to prevent lost increments
        - Reloads stats from disk inside lock to ensure consistency
        - Uses atomic write to prevent corruption
        """
        with self._acquire_stats_lock():
            # Reload from disk to get latest counts
            if self.stats_path.exists():
                with open(self.stats_path) as f:
                    current_stats = json.load(f)
            else:
                current_stats = {"hits": 0, "misses": 0}

            # Increment
            current_stats["misses"] = current_stats.get("misses", 0) + 1

            # Save atomically
            self._stats = current_stats
            self._save_stats_atomic()

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
