"""CAS-aware execution wrapper for stage graph.

This module provides the execution gate and wrapper that integrates
the CAS identity model with the stage graph for deterministic execution.

Key Components:
    - CASExecutor: Wraps stage execution with CAS lookup/store
    - execute_with_caching: Functional wrapper for single stage execution
    - FileLock integration for parallel safety

Design:
    Every stage execution is wrapped as:
        1. Compute CAS identity from (stage, inputs, config, env, platform)
        2. Check if artifact exists in CAS
        3. If cache hit: load and return
        4. If cache miss: execute stage, store result, return

Example:
    >>> executor = CASExecutor(artifact_store, cache_dir)
    >>> result = executor.execute(
    ...     stage=depth_stage,
    ...     inputs={"image": image_array},
    ...     config={"model": "DA3-Large"},
    ... )
    >>> # On second call with same inputs: instant cache hit
    >>> result2 = executor.execute(stage=depth_stage, inputs=inputs, config=config)
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Optional, Protocol, TypeVar, Union

from transformation_portal.core._cas_helpers import CASObjectMissingError
from transformation_portal.core._cas_helpers import atomic_write_json as _shared_atomic_write_json
from transformation_portal.core._cas_helpers import canonical_input_value as _shared_canonical_input_value
from transformation_portal.core._cas_helpers import compute_numpy_array_id as _shared_compute_numpy_array_id
from transformation_portal.core._cas_helpers import load_list_recursive as _shared_load_list_recursive
from transformation_portal.core._cas_helpers import load_serializable as _shared_load_serializable
from transformation_portal.core._cas_helpers import make_serializable as _shared_make_serializable
from transformation_portal.core._cas_helpers import sanitize_cas_id_for_filename as _shared_sanitize_cas_id_for_filename
from transformation_portal.core._cas_helpers import sanitize_key_for_filename as _shared_sanitize_key_for_filename
from transformation_portal.core._cas_helpers import serialize_list_recursive as _shared_serialize_list_recursive
from transformation_portal.core.execution_identity import (
    ArtifactMetadata,
    ExecutionIdentity,
    compute_cas_id,
    create_artifact_metadata,
    is_compatible,
    resolve_platform_lockfile,
    should_execute,
)
from transformation_portal.determinism.jcs import dumpb as jcs_dumpb
from transformation_portal.storage.cas_store import ArtifactStore

logger = logging.getLogger(__name__)

T = TypeVar("T")

# POSIX advisory locking; absent on platforms without ``fcntl``.
_FCNTL: Optional[ModuleType]
try:
    _FCNTL = importlib.import_module("fcntl")
except ImportError:  # pragma: no cover - exercised only where fcntl is missing
    _FCNTL = None


def _sanitize_cas_id_for_filename(cas_id: str) -> str:
    """Compatibility wrapper for shared CAS filename sanitization."""
    return _shared_sanitize_cas_id_for_filename(cas_id)


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Compatibility wrapper for shared atomic JSON writes."""
    _shared_atomic_write_json(path, data)


def _compute_numpy_array_id(arr: Any) -> str:
    """Compatibility wrapper for shared NumPy identity computation."""
    return _shared_compute_numpy_array_id(arr)


def _sanitize_key_for_filename(key: str) -> str:
    """Compatibility wrapper for shared artifact-key sanitization."""
    return _shared_sanitize_key_for_filename(key)


def make_serializable(
    outputs: dict[str, Any],
    artifact_store: Any,
    base_path: Path,
    cas_id: str,
) -> dict[str, Any]:
    """Compatibility wrapper for shared recursive serialization."""
    return _shared_make_serializable(outputs, artifact_store, base_path, cas_id)


def _serialize_list_recursive(
    items: list | tuple,
    artifact_store: Any,
    base_path: Path,
    cas_id: str,
) -> list:
    """Compatibility wrapper for shared recursive list serialization."""
    return _shared_serialize_list_recursive(items, artifact_store, base_path, cas_id)


def load_serializable(
    data: dict[str, Any],
    artifact_store: Any,
) -> dict[str, Any]:
    """Compatibility wrapper for shared recursive deserialization."""
    return _shared_load_serializable(data, artifact_store)


def _load_list_recursive(
    items: list,
    artifact_store: Any,
) -> list:
    """Compatibility wrapper for shared recursive list deserialization."""
    return _shared_load_list_recursive(items, artifact_store)


class ExecutableStage(Protocol):
    """Protocol for stages that can be executed with CAS caching."""

    @property
    def name(self) -> str:
        """Stage name for identification."""
        ...

    @property
    def version(self) -> str:
        """Stage version for cache invalidation."""
        ...

    def execute(self, inputs: dict[str, Any], config: Any) -> dict[str, Any]:
        """Execute stage and return outputs."""
        ...


@dataclass
class CacheResult:
    """Result of a CAS-aware execution.

    Attributes:
        outputs: Stage output artifacts/data
        cache_hit: True if result was loaded from cache
        execution_identity: CAS identity of this execution
        duration_ms: Execution time (load time if cache hit)
        artifact_metadata: Metadata of stored artifacts (if any)
    """

    outputs: dict[str, Any]
    cache_hit: bool
    execution_identity: ExecutionIdentity
    duration_ms: float
    artifact_metadata: Optional[ArtifactMetadata] = None


@dataclass
class ExecutorConfig:
    """Configuration for CAS executor.

    Attributes:
        enable_caching: If False, always execute (no cache lookup)
        verify_on_load: If True, verify artifact integrity on cache hit
        allow_cross_platform: If True, allow cross-platform CPU artifacts
        lock_timeout: Timeout for acquiring file locks (seconds)
        max_retries: Maximum retries for failed executions
        code_paths: Paths to include in code hash computation
        lockfile_path: Path to lockfile for deterministic builds (CI-required)
    """

    enable_caching: bool = True
    verify_on_load: bool = True
    allow_cross_platform: bool = False
    lock_timeout: float = 300.0  # 5 minutes
    max_retries: int = 0
    code_paths: list[str] = field(default_factory=lambda: ["src/"])
    lockfile_path: Optional[str] = None


class FileLock:
    """File-based lock for parallel safety across threads and processes.

    On POSIX the lock is an OS advisory lock (``fcntl.flock``) held on an open
    descriptor of the lock file. Ownership ends when the holder releases the
    lock or its process dies, and never because of the file's age or content:
    a waiter cannot take a lock that is still held, however long the holder
    works, and a file left behind by a dead holder blocks nobody.

    Where ``fcntl`` is unavailable the lock falls back to exclusive creation
    of the lock file. That keeps mutual exclusion but cannot detect a dead
    holder, so a file left by a crashed process has to be removed by hand.

    The lock is not reentrant and expects a local filesystem. A lock that is
    never released stays held until its process exits.

    Example:
        >>> with FileLock(Path("/tmp/my_lock")):
        ...     # Critical section
        ...     do_work()
    """

    def __init__(self, lock_path: Path, timeout: float = 300.0):
        """Initialize file lock.

        Args:
            lock_path: Path to lock file (will be created)
            timeout: Maximum time to wait for lock (seconds)
        """
        self.lock_path = Path(lock_path)
        self.timeout = timeout
        self._acquired = False
        self._descriptor: Optional[int] = None

    def acquire(self) -> bool:
        """Acquire the lock.

        Returns:
            True if lock acquired, False if timeout

        Note:
            Uses exponential backoff with jitter for contention handling.
            The timeout only bounds how long this caller waits; it never
            decides that a lock held by someone else has gone stale.
        """
        import random

        deadline = time.monotonic() + self.timeout
        wait_time = 0.01  # Start with 10ms

        while True:
            if self._try_acquire():
                return True

            remaining = deadline - time.monotonic()
            if not remaining > 0:
                return False

            # Wait with exponential backoff + jitter
            time.sleep(min(wait_time + random.uniform(0, wait_time * 0.1), remaining))
            wait_time = min(wait_time * 2, 1.0)  # Cap at 1 second

    def _try_acquire(self) -> bool:
        """Make one non-blocking attempt to take the lock."""
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        if _FCNTL is None:
            return self._try_create_exclusive_file()

        descriptor = os.open(self.lock_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            try:
                _FCNTL.flock(descriptor, _FCNTL.LOCK_EX | _FCNTL.LOCK_NB)
            except BlockingIOError:
                return False
            # A holder unlinks the file as it releases, so a descriptor opened
            # before that now locks an orphaned inode that excludes nobody.
            # Only a lock on the file currently at lock_path counts.
            if not self._is_current_lock_file(descriptor):
                return False
            self._descriptor = descriptor
            self._acquired = True
            return True
        finally:
            if self._descriptor != descriptor:
                os.close(descriptor)

    def _try_create_exclusive_file(self) -> bool:
        """Fallback without ``fcntl``: own the lock by creating its file."""
        try:
            descriptor = os.open(self.lock_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return False
        self._descriptor = descriptor
        self._acquired = True
        return True

    def _is_current_lock_file(self, descriptor: int) -> bool:
        """Return whether ``descriptor`` is the file currently at ``lock_path``."""
        try:
            current = os.lstat(self.lock_path)
        except FileNotFoundError:
            return False
        return os.path.samestat(os.fstat(descriptor), current)

    def release(self) -> None:
        """Release the lock if this instance holds it.

        The lock file is removed only while it is provably this holder's own,
        so releasing never deletes a lock that somebody else now holds.
        """
        if not self._acquired or self._descriptor is None:
            return
        descriptor, self._descriptor = self._descriptor, None
        self._acquired = False

        if _FCNTL is None:
            try:
                created_here = self._is_current_lock_file(descriptor)
            finally:
                # An open file cannot be unlinked on every platform.
                os.close(descriptor)
            if created_here:
                self.lock_path.unlink(missing_ok=True)
            return

        try:
            # Unlink before unlocking: while the lock is held nobody else can
            # own this path, and waiters re-check the path after locking.
            if self._is_current_lock_file(descriptor):
                self.lock_path.unlink(missing_ok=True)
        finally:
            try:
                _FCNTL.flock(descriptor, _FCNTL.LOCK_UN)
            finally:
                os.close(descriptor)

    def __enter__(self) -> "FileLock":
        if not self.acquire():
            raise TimeoutError(f"Could not acquire lock: {self.lock_path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


class CASExecutor:
    """CAS-aware executor for pipeline stages.

    Wraps stage execution with content-addressable caching:
    - Computes execution identity before running
    - Checks CAS for existing results
    - Stores new results in CAS with full provenance

    Example:
        >>> store = ArtifactStore(Path("/cache/cas"))
        >>> executor = CASExecutor(store, Path("/cache/results"))
        >>>
        >>> # First execution: computes and caches
        >>> result = executor.execute(
        ...     stage=my_stage,
        ...     inputs={"image": img},
        ...     config={"quality": "high"},
        ... )
        >>> assert not result.cache_hit
        >>>
        >>> # Second execution: instant cache hit
        >>> result2 = executor.execute(
        ...     stage=my_stage,
        ...     inputs={"image": img},
        ...     config={"quality": "high"},
        ... )
        >>> assert result2.cache_hit
    """

    def __init__(
        self,
        artifact_store: ArtifactStore,
        cache_dir: Path,
        config: Optional[ExecutorConfig] = None,
    ):
        """Initialize CAS executor.

        Args:
            artifact_store: CAS store for artifact storage
            cache_dir: Directory for result caching
            config: Executor configuration
        """
        self.artifact_store = artifact_store
        self.cache_dir = Path(cache_dir)
        self.config = config or ExecutorConfig()

        # Create directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.locks_dir = self.cache_dir / ".locks"
        self.locks_dir.mkdir(exist_ok=True)
        self.metadata_dir = self.cache_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)

        # Pre-compute code hash (reused across executions)
        from transformation_portal.core.execution_identity import compute_code_hash

        self._code_hash = compute_code_hash(self.config.code_paths)

        # Resolve lockfile path for CI determinism (ADR-032)
        # Use explicit path if provided, otherwise auto-resolve
        if self.config.lockfile_path:
            self._lockfile_path = str(self.config.lockfile_path)
        else:
            resolved = resolve_platform_lockfile()
            self._lockfile_path = str(resolved) if resolved else None

    def _get_lock(self, cas_id: str) -> FileLock:
        """Get file lock for a CAS ID."""
        safe_id = _sanitize_cas_id_for_filename(cas_id)
        lock_file = self.locks_dir / f"{safe_id}.lock"
        return FileLock(lock_file, timeout=self.config.lock_timeout)

    def _result_path(self, cas_id: str) -> Path:
        """Get result cache path for a CAS ID."""
        safe_id = _sanitize_cas_id_for_filename(cas_id)
        prefix = safe_id[:2]
        return self.cache_dir / "results" / prefix / f"{safe_id}.json"

    def _metadata_path(self, cas_id: str) -> Path:
        """Get metadata path for a CAS ID."""
        safe_id = _sanitize_cas_id_for_filename(cas_id)
        prefix = safe_id[:2]
        return self.metadata_dir / prefix / f"{safe_id}.meta.json"

    def _save_result(
        self,
        cas_id: str,
        outputs: dict[str, Any],
        metadata: ArtifactMetadata,
    ) -> None:
        """Save result and metadata to cache.

        Note: This method serializes outputs before saving.
        For pre-serialized outputs, use _save_result_serialized().
        """
        # Handle numpy arrays in outputs
        serializable_outputs = self._make_serializable(outputs, cas_id)
        self._save_result_serialized(cas_id, serializable_outputs, metadata)

    def _save_result_serialized(
        self,
        cas_id: str,
        serializable_outputs: dict[str, Any],
        metadata: ArtifactMetadata,
    ) -> None:
        """Save pre-serialized result and metadata to cache atomically.

        Uses atomic writes (temp-file + fsync + rename) to prevent
        partial writes visible to concurrent readers.

        Args:
            cas_id: CAS identity hash
            serializable_outputs: Already-serialized outputs (from _make_serializable)
            metadata: Artifact metadata
        """
        result_path = self._result_path(cas_id)
        metadata_path = self._metadata_path(cas_id)

        result_data = {
            "cas_id": cas_id,
            "outputs": serializable_outputs,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        # Write both files atomically
        _atomic_write_json(result_path, result_data)
        _atomic_write_json(metadata_path, metadata.to_dict())

        logger.debug("Saved result for %s", cas_id[:16])

    def _sanitize_key(self, key: str) -> str:
        """Sanitize an output key for safe use in filenames."""
        return _sanitize_key_for_filename(key)

    def _make_serializable(
        self,
        outputs: dict[str, Any],
        cas_id: str,
    ) -> dict[str, Any]:
        """Convert outputs to JSON-serializable format."""
        return make_serializable(outputs, self.artifact_store, self._result_path(cas_id).parent, cas_id)

    def _serialize_list(self, items: list | tuple, cas_id: str) -> list:
        """Serialize a list/tuple, handling nested numpy arrays and dicts."""
        return _serialize_list_recursive(items, self.artifact_store, self._result_path(cas_id).parent, cas_id)

    def _load_serializable(self, data: dict[str, Any]) -> dict[str, Any]:
        """Reconstruct outputs from serialized format."""
        return load_serializable(data, self.artifact_store)

    def _load_list(self, items: list) -> list:
        """Reconstruct a list, handling nested numpy arrays and dicts."""
        return _load_list_recursive(items, self.artifact_store)

    def _load_result(self, cas_id: str) -> Optional[dict[str, Any]]:
        """Load result from cache if available."""
        result_path = self._result_path(cas_id)
        if not result_path.exists():
            return None

        try:
            result_data = json.loads(result_path.read_text())
            return self._load_serializable(result_data.get("outputs", {}))
        except (json.JSONDecodeError, OSError, CASObjectMissingError) as e:
            logger.warning("Failed to load cached result for %s: %s", cas_id[:16], e)
            return None

    def _load_metadata(self, cas_id: str) -> Optional[ArtifactMetadata]:
        """Load metadata from cache if available."""
        metadata_path = self._metadata_path(cas_id)
        if not metadata_path.exists():
            return None

        try:
            data = json.loads(metadata_path.read_text())
            return ArtifactMetadata.from_dict(data)
        except (json.JSONDecodeError, OSError, KeyError) as e:
            logger.warning("Failed to load metadata for %s: %s", cas_id[:16], e)
            return None

    def _compute_input_ids(self, inputs: dict[str, Any]) -> list[str]:
        """Compute one SHA-256 content ID per input, in sorted key order.

        Every value is encoded with the typed canonical form the DAG executor
        uses, so an ID never depends on ``str(value)``: sequences and nested
        mappings are hashed element by element, equal text and bytes stay
        distinct, and a value without a byte identity (a ``Path``, an arbitrary
        object, an object array, a non-finite float) raises instead of sharing
        a cache entry with a different value.

        For NumPy arrays, includes dtype, shape, and data bytes to prevent
        false cache hits between arrays with same bytes but different semantics.

        The IDs do not carry the input names; ``_compute_input_identity`` binds
        them to their keys for the execution identity.

        Raises:
            TypeError: If a key is not a string or a value has no canonical form
            ValueError: If a float value is not finite
        """
        if any(type(key) is not str for key in inputs):
            raise TypeError("Stage input mappings require string keys")

        ids = []
        for key in sorted(inputs.keys()):
            canonical = _shared_canonical_input_value(inputs[key])
            if canonical[0] == "ndarray":
                # Shared NumPy identity, consistent across executors
                ids.append(canonical[1])
            else:
                ids.append(hashlib.sha256(jcs_dumpb(canonical)).hexdigest())

        return ids

    def _compute_input_identity(self, inputs: dict[str, Any]) -> str:
        """Compute one SHA-256 ID that binds every input name to its content ID.

        ``compute_cas_id`` sorts the IDs it is given, so bare content IDs
        describe an unordered bag of values: swapping two inputs, or renaming
        a key, would resolve to the same cached execution.
        """
        input_ids = self._compute_input_ids(inputs)
        bound = dict(zip(sorted(inputs.keys()), input_ids))
        return hashlib.sha256(jcs_dumpb({"inputs": bound})).hexdigest()

    def execute(
        self,
        stage: ExecutableStage,
        inputs: dict[str, Any],
        config: Any,
    ) -> CacheResult:
        """Execute stage with CAS-aware caching.

        Args:
            stage: Stage to execute
            inputs: Input artifacts/data
            config: Stage configuration

        Returns:
            CacheResult with outputs and cache status
        """
        start_time = time.time()

        # Compute input identity (name-bound, see _compute_input_identity)
        input_ids = [self._compute_input_identity(inputs)]

        # Compute execution identity (with lockfile for CI determinism)
        identity = compute_cas_id(
            stage_name=stage.name,
            input_ids=input_ids,
            config=config,
            stage_version=stage.version,
            code_hash=self._code_hash,
            lockfile_path=self._lockfile_path,
        )

        logger.debug(
            "Execution identity for %s: %s",
            stage.name,
            identity.cas_id[:16],
        )

        # Check cache if enabled
        if self.config.enable_caching:
            cached_result = self._load_result(identity.cas_id)
            if cached_result is not None:
                # Verify compatibility if metadata exists
                metadata = self._load_metadata(identity.cas_id)
                if metadata and not is_compatible(
                    metadata,
                    allow_cross_platform=self.config.allow_cross_platform,
                ):
                    logger.info(
                        "Cache hit for %s but incompatible platform, re-executing",
                        stage.name,
                    )
                else:
                    duration_ms = (time.time() - start_time) * 1000
                    logger.info(
                        "Cache hit for %s: %s (%.1fms)",
                        stage.name,
                        identity.cas_id[:16],
                        duration_ms,
                    )
                    return CacheResult(
                        outputs=cached_result,
                        cache_hit=True,
                        execution_identity=identity,
                        duration_ms=duration_ms,
                        artifact_metadata=metadata,
                    )

        # Cache miss - acquire lock and execute
        with self._get_lock(identity.cas_id):
            # Double-check cache (another process may have computed)
            if self.config.enable_caching:
                cached_result = self._load_result(identity.cas_id)
                if cached_result is not None:
                    duration_ms = (time.time() - start_time) * 1000
                    logger.info(
                        "Cache hit (after lock) for %s: %s",
                        stage.name,
                        identity.cas_id[:16],
                    )
                    return CacheResult(
                        outputs=cached_result,
                        cache_hit=True,
                        execution_identity=identity,
                        duration_ms=duration_ms,
                    )

            # Execute stage
            logger.info("Executing %s (cache miss)", stage.name)
            outputs = stage.execute(inputs, config)

            # Convert outputs to serializable form BEFORE hashing
            # This handles numpy arrays and other complex types
            serializable_outputs = self._make_serializable(outputs, identity.cas_id)

            # Compute metadata hash from serialized output structure
            # Note: For numpy arrays, actual content is stored in CAS with its own SHA-256;
            # this hash represents the output manifest structure for metadata purposes
            output_hash = hashlib.sha256(jcs_dumpb(serializable_outputs)).hexdigest()

            # Create artifact metadata
            metadata = create_artifact_metadata(
                artifact_id=output_hash,
                execution_identity=identity,
            )

            # Save result (uses already-serialized outputs)
            self._save_result_serialized(identity.cas_id, serializable_outputs, metadata)

            duration_ms = (time.time() - start_time) * 1000
            logger.info(
                "Executed %s: %s (%.1fms)",
                stage.name,
                identity.cas_id[:16],
                duration_ms,
            )

            return CacheResult(
                outputs=outputs,
                cache_hit=False,
                execution_identity=identity,
                duration_ms=duration_ms,
                artifact_metadata=metadata,
            )


def execute_with_caching(
    stage_fn: Callable[[dict[str, Any], Any], dict[str, Any]],
    stage_name: str,
    stage_version: str,
    inputs: dict[str, Any],
    config: Any,
    artifact_store: ArtifactStore,
    cache_dir: Path,
    **kwargs,
) -> CacheResult:
    """Functional wrapper for CAS-aware execution.

    Convenience function for one-off cached execution without
    creating an executor instance.

    Args:
        stage_fn: Function to execute (inputs, config) -> outputs
        stage_name: Name of the stage
        stage_version: Version of the stage
        inputs: Input artifacts/data
        config: Configuration
        artifact_store: CAS store
        cache_dir: Cache directory
        **kwargs: Additional ExecutorConfig options

    Returns:
        CacheResult with outputs and cache status
    """

    # Create a simple stage wrapper
    class FunctionStage:
        def __init__(self, fn, name, version):
            self._fn = fn
            self._name = name
            self._version = version

        @property
        def name(self) -> str:
            return self._name

        @property
        def version(self) -> str:
            return self._version

        def execute(self, inputs: dict[str, Any], config: Any) -> dict[str, Any]:
            return self._fn(inputs, config)

    stage = FunctionStage(stage_fn, stage_name, stage_version)
    config_obj = ExecutorConfig(**kwargs) if kwargs else None
    executor = CASExecutor(artifact_store, cache_dir, config_obj)

    return executor.execute(stage, inputs, config)
