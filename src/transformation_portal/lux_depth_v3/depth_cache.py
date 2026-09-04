"""Verified, execution-identity-bound depth cache.

The cache stores immutable NumPy objects separately from identity pointers. A
pointer is the commit record: it is published only after the exact serialized
``.npy`` bytes are durable, and readers accept an entry only after validating
both the closed pointer schema and the object through one open file handle.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import os
import secrets
import stat
import sys
import threading
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Generator, Iterable, Mapping, Optional, overload

import numpy as np

from ..core.execution_identity_v3 import MaterializedExecutionIdentityV3
from ..ingest.canonical_json import canonicalize_json
from .io_atomic import (
    _acquire_platform_file_lock,
    _publication_thread_lock,
    _release_platform_file_lock,
)

logger = logging.getLogger(__name__)


DEPTH_CACHE_SCHEMA = "tp.lux.depth-cache.v1"
DEPTH_CACHE_POINTER_SCHEMA = "tp.lux.depth-cache.pointer.v1"

_POINTER_MAX_BYTES = 64 * 1024
_HASH_CHUNK_BYTES = 1024 * 1024
_MAX_NPY_HEADER_BYTES = 10_000
_MAX_NPY_PREAMBLE_BYTES = 12
_ABSOLUTE_OBJECT_MAX_BYTES = 2 * 1024**3
_LOCK_SHARD_COUNT = 64
_SIZE_CHECK_INTERVAL = 16
_POINTER_MAX_DEPTH = 16
_POINTER_MAX_NODES = 4_096
_ARGUMENT_MISSING = object()
_QUOTA_STATE_NAME = ".quota-state.json"
_QUOTA_STATE_SCHEMA = "tp.lux.depth-cache.quota-state.v1"
_REMOVAL_QUARANTINE_PREFIX = ".remove-"
_QUOTA_STATE_KEYS = frozenset(
    {
        "schema",
        "phase",
        "authority_device",
        "authority_inode",
        "cache_device",
        "cache_inode",
        "namespace_device",
        "namespace_inode",
        "entries_device",
        "entries_inode",
        "objects_device",
        "objects_inode",
        "locks_device",
        "locks_inode",
        "max_size_bytes",
        "physical_size_bytes",
        "store_count",
        "reserved_add_bytes",
        "planned_remove_bytes",
    }
)
_HEX_DIGEST_LENGTH = 64
_POINTER_KEYS = frozenset(
    {
        "schema",
        "cache_schema",
        "cache_key",
        "execution_identity_sha256",
        "config_fingerprint_sha256",
        "input_content_sha256",
        "model_constituents",
        "materialized_weights_sha256",
        "dependency_lock_sha256",
        "npy_sha256",
        "byte_length",
        "shape",
        "dtype",
    }
)
_CONSTITUENT_KEYS = frozenset(
    {
        "constituent_ordinal",
        "backend_id",
        "model_canonical_key",
        "model_lock_revision",
        "materialized_weights_sha256",
    }
)


class _DuplicateJsonKeyError(ValueError):
    """A cache pointer contains an ambiguous duplicate JSON key."""


@dataclass(frozen=True)
class _CacheEntry:
    """One fully verified cache entry used by housekeeping."""

    pointer_path: Path
    object_path: Path
    pointer: Mapping[str, Any]
    pointer_stat: os.stat_result
    access_time_ns: int


@dataclass(frozen=True)
class _OpenDirectoryBinding:
    """One retained parent/name/descriptor binding during initialization."""

    path: Path
    parent_descriptor: int
    name: str
    descriptor: int
    descriptor_stat: os.stat_result
    created: bool


@dataclass(frozen=True)
class _QuotaState:
    """Durable cross-process size state protected by the cache authority lock."""

    phase: str
    max_size_bytes: int
    physical_size_bytes: int
    store_count: int
    reserved_add_bytes: int = 0
    planned_remove_bytes: int = 0


@dataclass(frozen=True)
class _VerifiedObject:
    """Metadata derived from one fully verified physical CAS object."""

    path: Path
    file_stat: os.stat_result
    npy_sha256: str
    byte_length: int
    shape: tuple[int, int]
    dtype: str


@dataclass(frozen=True)
class _Removal:
    """One identity-bound deletion selected by a complete eviction plan."""

    path: Path
    root: Path
    file_stat: os.stat_result


@dataclass(frozen=True)
class _ReconcilePlan:
    """A feasible set of removals computed before the first mutation."""

    removals: tuple[_Removal, ...]
    remaining_entries: tuple[_CacheEntry, ...]
    final_size_bytes: int


@dataclass(frozen=True)
class _CacheSnapshot:
    """Read-only view used to prove an eviction is feasible."""

    entries: tuple[_CacheEntry, ...]
    objects: Mapping[Path, _VerifiedObject]
    cleanup: tuple[_Removal, ...]
    physical_size_bytes: int


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _HEX_DIGEST_LENGTH
        and value == value.lower()
        and value != "0" * _HEX_DIGEST_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _json_object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _validate_json_structure_bounds(value: object) -> None:
    """Reject cache pointers whose container graph is too deep or too large."""

    pending: list[tuple[object, int]] = [(value, 1)]
    nodes = 0
    while pending:
        current, depth = pending.pop()
        nodes += 1
        if nodes > _POINTER_MAX_NODES:
            raise ValueError("cache pointer exceeds the structural node limit")
        if isinstance(current, dict):
            if depth > _POINTER_MAX_DEPTH:
                raise ValueError("cache pointer exceeds the structural depth limit")
            pending.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            if depth > _POINTER_MAX_DEPTH:
                raise ValueError("cache pointer exceeds the structural depth limit")
            pending.extend((item, depth + 1) for item in current)


def _cache_key_for_execution_identity(execution_identity_sha256: str) -> str:
    payload = canonicalize_json(
        {
            "cache_schema": DEPTH_CACHE_SCHEMA,
            "execution_identity_sha256": execution_identity_sha256,
        }
    )
    return hashlib.sha256(b"tp.execution.cache-key.v1\0" + payload).hexdigest()


def _canonicalize_top_level_alias(path: Path) -> Path:
    """Canonicalize only the standard macOS ``/tmp`` and ``/var`` aliases."""

    if sys.platform != "darwin" or not path.is_absolute() or len(path.parts) < 2:
        return path
    alias_name = path.parts[1]
    expected_target = {
        "tmp": Path("/private/tmp"),
        "var": Path("/private/var"),
    }.get(alias_name)
    if expected_target is None:
        return path
    try:
        canonical_top_level = (Path(path.anchor) / alias_name).resolve(strict=True)
    except (OSError, RuntimeError):
        return path
    if canonical_top_level != expected_target:
        return path
    return canonical_top_level.joinpath(*path.parts[2:])


def _is_real_depth_dtype(dtype: np.dtype[Any]) -> bool:
    return (
        not dtype.hasobject
        and dtype.fields is None
        and (np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.floating))
    )


def _validate_constituent(value: object) -> None:
    if not isinstance(value, dict) or frozenset(value) != _CONSTITUENT_KEYS:
        raise ValueError("cache pointer model constituent has an invalid schema")
    ordinal = value["constituent_ordinal"]
    if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 0:
        raise ValueError("cache pointer constituent ordinal is invalid")
    if not isinstance(value["backend_id"], str) or not value["backend_id"]:
        raise ValueError("cache pointer constituent backend is invalid")
    if not isinstance(value["model_canonical_key"], str) or not value["model_canonical_key"]:
        raise ValueError("cache pointer model key is invalid")
    revision = value["model_lock_revision"]
    if revision is not None and (not isinstance(revision, str) or not revision):
        raise ValueError("cache pointer model revision is invalid")
    if not _is_sha256(value["materialized_weights_sha256"]):
        raise ValueError("cache pointer weights digest is invalid")


def _constituent_projection(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_constituents = payload.get("model_constituents")
    if not isinstance(raw_constituents, list) or not raw_constituents:
        raise ValueError("materialized identity requires model_constituents")

    result: list[dict[str, Any]] = []
    for raw in raw_constituents:
        if not isinstance(raw, Mapping):
            raise ValueError("materialized identity constituent must be an object")
        projected = {
            "constituent_ordinal": raw.get("constituent_ordinal"),
            "backend_id": raw.get("backend_id"),
            "model_canonical_key": raw.get("model_canonical_key"),
            "model_lock_revision": raw.get("model_lock_revision"),
            "materialized_weights_sha256": raw.get("materialized_weights_sha256"),
        }
        _validate_constituent(projected)
        result.append(projected)
    return result


def _identity_projection(identity: MaterializedExecutionIdentityV3) -> dict[str, Any]:
    if not isinstance(identity, MaterializedExecutionIdentityV3):
        raise TypeError("depth cache requires MaterializedExecutionIdentityV3")
    if not identity.cacheable:
        raise ValueError("depth cache requires a cacheable materialized identity")
    payload = identity.to_payload()
    execution_identity_sha256 = identity.execution_identity_sha256
    if not _is_sha256(execution_identity_sha256):
        raise ValueError("materialized execution identity digest is invalid")
    config_fingerprint_sha256 = payload.get("config_fingerprint_sha256")
    dependency_lock_sha256 = payload.get("dependency_lock_sha256")
    input_content_sha256 = payload.get("input_content_sha256")
    materialized_weights_sha256 = payload.get("materialized_weights_sha256")
    if not _is_sha256(config_fingerprint_sha256):
        raise ValueError("materialized config fingerprint is invalid")
    if not _is_sha256(input_content_sha256):
        raise ValueError("materialized input content digest is invalid")
    if not _is_sha256(materialized_weights_sha256):
        raise ValueError("materialized weights digest is invalid")
    if not _is_sha256(dependency_lock_sha256):
        raise ValueError("materialized dependency lock digest is invalid")
    cache_key = identity.cache_key(DEPTH_CACHE_SCHEMA)
    if not _is_sha256(cache_key) or cache_key != _cache_key_for_execution_identity(execution_identity_sha256):
        raise ValueError("materialized execution identity returned an invalid cache key")
    return {
        "cache_key": cache_key,
        "execution_identity_sha256": execution_identity_sha256,
        "config_fingerprint_sha256": config_fingerprint_sha256,
        "input_content_sha256": input_content_sha256,
        "model_constituents": _constituent_projection(payload),
        "materialized_weights_sha256": materialized_weights_sha256,
        "dependency_lock_sha256": dependency_lock_sha256,
    }


class DepthCache:
    """Immutable-array CAS committed by verified identity pointer sidecars."""

    def __init__(self, cache_dir: Path, max_size_gb: float = 10.0):
        if not math.isfinite(max_size_gb) or max_size_gb < 0:
            raise ValueError("max_size_gb must be finite and non-negative")
        (
            base_dir,
            self.cache_dir,
            self._v1_dir,
            self._entries_dir,
            self._objects_dir,
            self._locks_dir,
            self._namespace_roots,
        ) = self._initialize_namespace(Path(cache_dir))
        self._lock_authority_dir = base_dir
        self.max_size_gb = max_size_gb
        self._fenced_shards_lock = threading.Lock()
        self._fenced_shards: set[tuple[Path, str, int, int]] = set()
        self._configured_max_size_bytes = self._max_size_bytes
        # Missing or prepared quota state is reconciled under the cache-wide
        # descriptor lock. A clean state is the durable fast path, so opening
        # a cache does not rehash the entire namespace on every process start.
        with self._locked_shards(range(_LOCK_SHARD_COUNT)):
            self._configure_quota_locked(self._configured_max_size_bytes)
        logger.debug("Depth cache initialized: %s (max %.3fGB)", self.cache_dir, max_size_gb)

    @property
    def _max_size_bytes(self) -> int:
        return int(self.max_size_gb * (1024**3))

    @classmethod
    def _initialize_namespace(
        cls,
        cache_dir: Path,
    ) -> tuple[Path, Path, Path, Path, Path, Path, tuple[tuple[Path, os.stat_result], ...]]:
        """Create and pin the hierarchy through retained directory descriptors."""

        if (
            os.name != "posix"
            or not hasattr(os, "O_DIRECTORY")
            or not hasattr(os, "O_NOFOLLOW")
            or os.link not in os.supports_dir_fd
            or os.link not in os.supports_follow_symlinks
        ):
            raise OSError("depth cache requires POSIX descriptor-relative directory safety")

        lexical_base_dir = Path(os.path.abspath(os.fspath(cache_dir)))
        base_dir = _canonicalize_top_level_alias(lexical_base_dir)
        root_path = Path(base_dir.anchor)
        descriptors: list[int] = []
        bindings: list[_OpenDirectoryBinding] = []
        created_bindings: list[_OpenDirectoryBinding] = []

        def open_child(
            parent_descriptor: int,
            parent_path: Path,
            name: str,
            *,
            fence_existing: bool = False,
        ) -> _OpenDirectoryBinding:
            cls._assert_directory_bindings(bindings)
            created = False
            created_stat: Optional[os.stat_result] = None
            registered = False
            try:
                descriptor = os.open(name, cls._directory_open_flags(), dir_fd=parent_descriptor)
            except FileNotFoundError:
                cls._assert_directory_bindings(bindings)
                try:
                    os.mkdir(name, 0o755, dir_fd=parent_descriptor)
                    created = True
                    created_stat = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
                except FileExistsError:
                    pass
                try:
                    cls._assert_directory_bindings(bindings)
                    descriptor = os.open(name, cls._directory_open_flags(), dir_fd=parent_descriptor)
                except BaseException:
                    if created and created_stat is not None:
                        try:
                            current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
                            if os.path.samestat(current, created_stat):
                                os.rmdir(name, dir_fd=parent_descriptor)
                                os.fsync(parent_descriptor)
                        except OSError:
                            pass
                    raise

            try:
                descriptors.append(descriptor)
                descriptor_stat = os.fstat(descriptor)
                path_stat = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
                if (
                    not stat.S_ISDIR(descriptor_stat.st_mode)
                    or not stat.S_ISDIR(path_stat.st_mode)
                    or not os.path.samestat(descriptor_stat, path_stat)
                ):
                    raise OSError("depth cache namespace directory binding is unsafe")
                binding = _OpenDirectoryBinding(
                    path=parent_path / name,
                    parent_descriptor=parent_descriptor,
                    name=name,
                    descriptor=descriptor,
                    descriptor_stat=descriptor_stat,
                    created=created,
                )
                bindings.append(binding)
                if created:
                    created_bindings.append(binding)
                registered = True
                if created or fence_existing:
                    # Cache-owned entries are fenced even when pre-existing so
                    # a retry repairs a killed mkdir-before-fsync attempt.
                    os.fsync(parent_descriptor)
                cls._assert_directory_bindings(bindings)
                return binding
            except BaseException:
                if created and not registered and created_stat is not None:
                    try:
                        current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
                        if os.path.samestat(current, created_stat):
                            os.rmdir(name, dir_fd=parent_descriptor)
                            os.fsync(parent_descriptor)
                    except OSError:
                        pass
                raise

        try:
            root_descriptor = os.open(root_path, cls._directory_open_flags())
            descriptors.append(root_descriptor)
            root_stat = os.fstat(root_descriptor)
            if not stat.S_ISDIR(root_stat.st_mode):
                raise OSError("depth cache filesystem root is not a directory")

            parent_descriptor = root_descriptor
            parent_path = root_path
            base_components = base_dir.parts[1:]
            for component in base_components:
                binding = open_child(
                    parent_descriptor,
                    parent_path,
                    component,
                    fence_existing=True,
                )
                parent_descriptor = binding.descriptor
                parent_path = binding.path

            cache_binding = open_child(parent_descriptor, base_dir, ".depth_cache", fence_existing=True)
            v1_binding = open_child(cache_binding.descriptor, cache_binding.path, "v1", fence_existing=True)
            entries_binding = open_child(v1_binding.descriptor, v1_binding.path, "entries", fence_existing=True)
            objects_binding = open_child(v1_binding.descriptor, v1_binding.path, "objects", fence_existing=True)
            locks_binding = open_child(v1_binding.descriptor, v1_binding.path, "locks", fence_existing=True)
            cls._assert_directory_bindings(bindings)

            namespace_roots = ((root_path, root_stat),) + tuple(
                (binding.path, binding.descriptor_stat) for binding in bindings
            )
            return (
                base_dir,
                cache_binding.path,
                v1_binding.path,
                entries_binding.path,
                objects_binding.path,
                locks_binding.path,
                namespace_roots,
            )
        except OSError as exc:
            # Roll back only entries created by this initializer and only
            # through their retained parent descriptors. Never follow a
            # substituted path during cleanup.
            for binding in reversed(created_bindings):
                try:
                    current = os.stat(binding.name, dir_fd=binding.parent_descriptor, follow_symlinks=False)
                    if os.path.samestat(current, binding.descriptor_stat):
                        os.rmdir(binding.name, dir_fd=binding.parent_descriptor)
                        os.fsync(binding.parent_descriptor)
                except OSError:
                    pass
            raise OSError(f"depth cache namespace root could not be securely initialized: {base_dir}") from exc
        finally:
            for descriptor in reversed(descriptors):
                os.close(descriptor)

    @staticmethod
    def _assert_directory_bindings(bindings: Iterable[_OpenDirectoryBinding]) -> None:
        for binding in bindings:
            descriptor_stat = os.fstat(binding.descriptor)
            path_stat = os.stat(binding.name, dir_fd=binding.parent_descriptor, follow_symlinks=False)
            if (
                not stat.S_ISDIR(descriptor_stat.st_mode)
                or not stat.S_ISDIR(path_stat.st_mode)
                or not os.path.samestat(descriptor_stat, binding.descriptor_stat)
                or not os.path.samestat(descriptor_stat, path_stat)
            ):
                raise OSError(f"depth cache namespace root was replaced: {binding.path}")

    def _validate_namespace_roots(self) -> None:
        for path, expected_stat in self._namespace_roots:
            current_stat = path.lstat()
            if not stat.S_ISDIR(current_stat.st_mode) or not os.path.samestat(current_stat, expected_stat):
                raise OSError(f"depth cache namespace root was replaced: {path}")

    def _namespace_roots_match_expected(self) -> bool:
        """Return whether every path still names this instance's pinned root."""

        try:
            self._validate_namespace_roots()
        except OSError:
            return False
        return True

    def _expected_root_stat(self, root: Path) -> os.stat_result:
        expected = next(
            (expected_stat for namespace_root, expected_stat in self._namespace_roots if namespace_root == root),
            None,
        )
        if expected is None:
            raise OSError("cache operation is outside the validated namespace")
        return expected

    def _descriptor_matches_expected_root(self, descriptor: int, root: Path) -> bool:
        """Return whether a retained descriptor is still bound at ``root``."""

        try:
            expected = self._expected_root_stat(root)
            descriptor_stat = os.fstat(descriptor)
            current = root.lstat()
        except OSError:
            return False
        return (
            stat.S_ISDIR(descriptor_stat.st_mode)
            and stat.S_ISDIR(current.st_mode)
            and os.path.samestat(descriptor_stat, expected)
            and os.path.samestat(descriptor_stat, current)
        )

    @staticmethod
    def _directory_open_flags() -> int:
        return os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)

    @contextmanager
    def _open_root_descriptor(self, root: Path) -> Generator[int, None, None]:
        self._validate_namespace_roots()
        expected = self._expected_root_stat(root)
        descriptor = os.open(root, self._directory_open_flags())
        try:
            if not os.path.samestat(os.fstat(descriptor), expected):
                raise OSError("cache namespace root was replaced before open")
            yield descriptor
            current = root.lstat()
            if not stat.S_ISDIR(current.st_mode) or not os.path.samestat(current, expected):
                raise OSError("cache namespace root was replaced during operation")
        finally:
            os.close(descriptor)

    @staticmethod
    def _assert_shard_binding(root_descriptor: int, shard: str, shard_descriptor: int) -> None:
        descriptor_stat = os.fstat(shard_descriptor)
        path_stat = os.stat(shard, dir_fd=root_descriptor, follow_symlinks=False)
        if (
            not stat.S_ISDIR(descriptor_stat.st_mode)
            or not stat.S_ISDIR(path_stat.st_mode)
            or not os.path.samestat(descriptor_stat, path_stat)
        ):
            raise OSError("cache namespace shard was replaced during operation")

    @contextmanager
    def _open_shard_descriptor(
        self,
        root: Path,
        shard: str,
        *,
        create: bool = False,
    ) -> Generator[tuple[int, int], None, None]:
        if len(shard) != 2 or any(character not in "0123456789abcdef" for character in shard):
            raise ValueError("invalid cache namespace shard")
        with self._open_root_descriptor(root) as root_descriptor:
            created_stat: Optional[os.stat_result] = None
            if create:
                try:
                    os.mkdir(shard, dir_fd=root_descriptor)
                    created_stat = os.stat(shard, dir_fd=root_descriptor, follow_symlinks=False)
                except FileExistsError:
                    pass
            try:
                shard_descriptor = os.open(shard, self._directory_open_flags(), dir_fd=root_descriptor)
            except BaseException:
                if created_stat is not None:
                    try:
                        current = os.stat(shard, dir_fd=root_descriptor, follow_symlinks=False)
                        if os.path.samestat(current, created_stat):
                            os.rmdir(shard, dir_fd=root_descriptor)
                            os.fsync(root_descriptor)
                    except OSError:
                        pass
                raise
            try:
                self._assert_shard_binding(root_descriptor, shard, shard_descriptor)
                shard_stat = os.fstat(shard_descriptor)
                fence_key = (root, shard, shard_stat.st_dev, shard_stat.st_ino)
                with self._fenced_shards_lock:
                    if fence_key not in self._fenced_shards:
                        try:
                            os.fsync(root_descriptor)
                        except BaseException:
                            if created_stat is not None:
                                try:
                                    current = os.stat(shard, dir_fd=root_descriptor, follow_symlinks=False)
                                    if os.path.samestat(current, created_stat):
                                        os.close(shard_descriptor)
                                        shard_descriptor = -1
                                        os.rmdir(shard, dir_fd=root_descriptor)
                                        os.fsync(root_descriptor)
                                except OSError:
                                    pass
                            raise
                        self._fenced_shards.add(fence_key)
                yield root_descriptor, shard_descriptor
                self._assert_shard_binding(root_descriptor, shard, shard_descriptor)
            finally:
                if shard_descriptor >= 0:
                    os.close(shard_descriptor)

    def _assert_global_lock_binding(self, descriptor: int) -> None:
        """Verify that ``descriptor`` still names the external lock authority."""

        self._validate_namespace_roots()
        expected = self._expected_root_stat(self._lock_authority_dir)
        descriptor_stat = os.fstat(descriptor)
        path_stat = self._lock_authority_dir.lstat()
        if (
            not stat.S_ISDIR(descriptor_stat.st_mode)
            or not stat.S_ISDIR(path_stat.st_mode)
            or not os.path.samestat(descriptor_stat, expected)
            or not os.path.samestat(descriptor_stat, path_stat)
        ):
            raise OSError("cache global lock directory was replaced")

    @contextmanager
    def _global_cache_lock(self) -> Generator[int, None, None]:
        """Serialize mutations on the caller-owned cache base directory."""

        with self._open_root_descriptor(self._lock_authority_dir) as descriptor:
            descriptor_stat = os.fstat(descriptor)
            # The helper only uses this path as a process-local key. Binding
            # the key to the physical directory identity makes lexical aliases
            # converge without creating a replaceable lock-file authority.
            thread_key = Path(f"/.tp-depth-cache-authority-{descriptor_stat.st_dev:x}-{descriptor_stat.st_ino:x}.lock")
            thread_lock = _publication_thread_lock(thread_key)
            with thread_lock:
                _acquire_platform_file_lock(descriptor)
                try:
                    self._assert_global_lock_binding(descriptor)
                    yield descriptor
                    self._assert_global_lock_binding(descriptor)
                finally:
                    _release_platform_file_lock(descriptor)

    @contextmanager
    def _fixed_shard_lock(self, index: int) -> Generator[int, None, None]:
        """Compatibility wrapper over the cache-wide directory-inode lease."""

        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < _LOCK_SHARD_COUNT:
            raise ValueError("invalid cache lock shard")
        with self._global_cache_lock() as descriptor:
            yield descriptor

    @staticmethod
    def _shard_index(cache_key: str) -> int:
        return int(cache_key[:8], 16) % _LOCK_SHARD_COUNT

    @contextmanager
    def _locked_shards(self, indices: Iterable[int]) -> Generator[None, None, None]:
        requested = tuple(sorted(set(indices)))
        if any(
            not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < _LOCK_SHARD_COUNT for index in requested
        ):
            raise ValueError("invalid cache lock shard")
        with self._global_cache_lock():
            yield

    def _entry_path(self, cache_key: str) -> Path:
        if not _is_sha256(cache_key):
            raise ValueError("invalid cache entry key")
        return self._entries_dir / cache_key[:2] / f"{cache_key}.json"

    def _object_path(self, npy_sha256: str) -> Path:
        if not _is_sha256(npy_sha256):
            raise ValueError("invalid cache object digest")
        return self._objects_dir / npy_sha256[:2] / f"{npy_sha256}.npy"

    def _namespace_paths(self, root: Path, suffix: str) -> list[Path]:
        paths: list[Path] = []
        with self._open_root_descriptor(root) as root_descriptor:
            shard_names = [entry.name for entry in os.scandir(root_descriptor)]
        for shard in shard_names:
            if len(shard) != 2 or any(character not in "0123456789abcdef" for character in shard):
                continue
            with self._open_shard_descriptor(root, shard) as (_, shard_descriptor):
                file_names = [
                    entry.name
                    for entry in os.scandir(shard_descriptor)
                    if entry.name.endswith(suffix) and not entry.is_dir(follow_symlinks=False)
                ]
            paths.extend(root / shard / name for name in file_names)
        return paths

    @staticmethod
    def _is_governed_temp_name(name: str, *, shard: str, destination_suffix: str) -> bool:
        """Return whether ``name`` is an exact cache-generated temp name."""

        marker = f"{destination_suffix}.tmp-"
        expected_length = 1 + _HEX_DIGEST_LENGTH + len(marker) + 32
        if len(name) != expected_length or not name.startswith("."):
            return False
        digest = name[1 : 1 + _HEX_DIGEST_LENGTH]
        if name[1 + _HEX_DIGEST_LENGTH : 1 + _HEX_DIGEST_LENGTH + len(marker)] != marker:
            return False
        token = name[-32:]
        return digest[:2] == shard and _is_sha256(digest) and all(character in "0123456789abcdef" for character in token)

    def _governed_temp_paths(self, root: Path, destination_suffix: str) -> list[Path]:
        """Enumerate regular orphan temps without following namespace links."""

        paths: list[Path] = []
        with self._open_root_descriptor(root) as root_descriptor:
            shard_names = [entry.name for entry in os.scandir(root_descriptor)]
        for shard in shard_names:
            if len(shard) != 2 or any(character not in "0123456789abcdef" for character in shard):
                continue
            with self._open_shard_descriptor(root, shard) as (_, shard_descriptor):
                file_names = [
                    entry.name
                    for entry in os.scandir(shard_descriptor)
                    if entry.is_file(follow_symlinks=False)
                    and self._is_governed_temp_name(
                        entry.name,
                        shard=shard,
                        destination_suffix=destination_suffix,
                    )
                ]
            paths.extend(root / shard / name for name in file_names)
        return paths

    @staticmethod
    def _is_governed_removal_name(name: str) -> bool:
        """Return whether ``name`` is an exact bounded removal quarantine."""

        expected_length = len(_REMOVAL_QUARANTINE_PREFIX) + 64 + 1 + 32
        if len(name) != expected_length or not name.startswith(_REMOVAL_QUARANTINE_PREFIX):
            return False
        identity_digest = name[len(_REMOVAL_QUARANTINE_PREFIX) : len(_REMOVAL_QUARANTINE_PREFIX) + 64]
        separator = name[len(_REMOVAL_QUARANTINE_PREFIX) + 64]
        nonce = name[-32:]
        return separator == "-" and all(character in "0123456789abcdef" for character in identity_digest + nonce)

    @staticmethod
    def _removal_identity_digest(file_stat: os.stat_result) -> str:
        identity = f"{file_stat.st_dev}:{file_stat.st_ino}".encode("ascii")
        return hashlib.sha256(identity).hexdigest()

    @classmethod
    def _removal_name_matches_stat(cls, name: str, file_stat: os.stat_result) -> bool:
        if not cls._is_governed_removal_name(name):
            return False
        encoded_digest = name[len(_REMOVAL_QUARANTINE_PREFIX) : len(_REMOVAL_QUARANTINE_PREFIX) + 64]
        return encoded_digest == cls._removal_identity_digest(file_stat)

    def _namespace_removal_paths(self, root: Path) -> list[Path]:
        """Enumerate exact removal quarantines beneath two-level data roots."""

        paths: list[Path] = []
        with self._open_root_descriptor(root) as root_descriptor:
            shard_names = [entry.name for entry in os.scandir(root_descriptor)]
        for shard in shard_names:
            if len(shard) != 2 or any(character not in "0123456789abcdef" for character in shard):
                continue
            with self._open_shard_descriptor(root, shard) as (_, shard_descriptor):
                file_names = [
                    entry.name
                    for entry in os.scandir(shard_descriptor)
                    if entry.is_file(follow_symlinks=False) and self._is_governed_removal_name(entry.name)
                ]
            paths.extend(root / shard / name for name in file_names)
        return paths

    def _root_removal_paths(self, root: Path) -> list[Path]:
        """Enumerate exact removal quarantines directly below ``root``."""

        with self._open_root_descriptor(root) as root_descriptor:
            names = [
                entry.name
                for entry in os.scandir(root_descriptor)
                if entry.is_file(follow_symlinks=False) and self._is_governed_removal_name(entry.name)
            ]
        return [root / name for name in names]

    def _root_paths(self, root: Path, suffix: str) -> list[Path]:
        with self._open_root_descriptor(root) as root_descriptor:
            names = [
                entry.name
                for entry in os.scandir(root_descriptor)
                if entry.name.endswith(suffix) and entry.is_file(follow_symlinks=False)
            ]
        return [root / name for name in names]

    @staticmethod
    def _namespace_address(path: Path, root: Path) -> tuple[str, str]:
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise OSError("cache file is outside its namespace root") from exc
        if len(relative.parts) != 2:
            raise OSError("cache file path has an invalid namespace depth")
        shard, name = relative.parts
        if len(shard) != 2 or any(character not in "0123456789abcdef" for character in shard):
            raise OSError("cache file path has an invalid namespace shard")
        if not name or name in {".", ".."} or "/" in name or "\\" in name:
            raise OSError("cache file path has an invalid basename")
        return shard, name

    @staticmethod
    def _root_address(path: Path, root: Path) -> str:
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise OSError("cache file is outside its namespace root") from exc
        if len(relative.parts) != 1:
            raise OSError("cache file path has an invalid namespace depth")
        name = relative.parts[0]
        if not name or name in {".", ".."} or "/" in name or "\\" in name:
            raise OSError("cache file path has an invalid basename")
        return name

    @contextmanager
    def _open_namespace_regular(
        self,
        path: Path,
        root: Path,
        *,
        allow_metadata_change: bool = False,
    ) -> Generator[tuple[BinaryIO, os.stat_result], None, None]:
        shard, name = self._namespace_address(path, root)
        with self._open_shard_descriptor(root, shard) as (_, shard_descriptor):
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
            descriptor = os.open(name, flags, dir_fd=shard_descriptor)
            try:
                descriptor_stat = os.fstat(descriptor)
                path_stat = os.stat(name, dir_fd=shard_descriptor, follow_symlinks=False)
                if (
                    not stat.S_ISREG(descriptor_stat.st_mode)
                    or descriptor_stat.st_nlink != 1
                    or not os.path.samestat(descriptor_stat, path_stat)
                ):
                    raise OSError("cache entry is not a regular, unaliased file")
                handle = os.fdopen(descriptor, "rb", closefd=True)
                descriptor = -1
                try:
                    yield handle, descriptor_stat
                    final_descriptor_stat = os.fstat(handle.fileno())
                    final_path_stat = os.stat(name, dir_fd=shard_descriptor, follow_symlinks=False)
                    if (
                        not stat.S_ISREG(final_descriptor_stat.st_mode)
                        or final_descriptor_stat.st_nlink != 1
                        or not os.path.samestat(final_descriptor_stat, descriptor_stat)
                        or not os.path.samestat(final_descriptor_stat, final_path_stat)
                        or final_descriptor_stat.st_size != descriptor_stat.st_size
                        or final_descriptor_stat.st_mtime_ns != descriptor_stat.st_mtime_ns
                        or (not allow_metadata_change and final_descriptor_stat.st_ctime_ns != descriptor_stat.st_ctime_ns)
                    ):
                        raise OSError("cache entry changed or was replaced while open")
                finally:
                    handle.close()
            finally:
                if descriptor >= 0:
                    os.close(descriptor)

    @contextmanager
    def _open_root_regular(
        self,
        path: Path,
        root: Path,
    ) -> Generator[tuple[BinaryIO, os.stat_result], None, None]:
        name = self._root_address(path, root)
        with self._open_root_descriptor(root) as root_descriptor:
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
            descriptor = os.open(name, flags, dir_fd=root_descriptor)
            try:
                descriptor_stat = os.fstat(descriptor)
                path_stat = os.stat(name, dir_fd=root_descriptor, follow_symlinks=False)
                if (
                    not stat.S_ISREG(descriptor_stat.st_mode)
                    or descriptor_stat.st_nlink != 1
                    or not os.path.samestat(descriptor_stat, path_stat)
                ):
                    raise OSError("legacy cache entry is not a regular, unaliased file")
                handle = os.fdopen(descriptor, "rb", closefd=True)
                descriptor = -1
                try:
                    yield handle, descriptor_stat
                    final_descriptor_stat = os.fstat(handle.fileno())
                    final_path_stat = os.stat(name, dir_fd=root_descriptor, follow_symlinks=False)
                    if (
                        not stat.S_ISREG(final_descriptor_stat.st_mode)
                        or final_descriptor_stat.st_nlink != 1
                        or not os.path.samestat(final_descriptor_stat, descriptor_stat)
                        or not os.path.samestat(final_descriptor_stat, final_path_stat)
                        or final_descriptor_stat.st_size != descriptor_stat.st_size
                        or final_descriptor_stat.st_mtime_ns != descriptor_stat.st_mtime_ns
                        or final_descriptor_stat.st_ctime_ns != descriptor_stat.st_ctime_ns
                    ):
                        raise OSError("legacy cache entry changed or was replaced while open")
                finally:
                    handle.close()
            finally:
                if descriptor >= 0:
                    os.close(descriptor)

    @staticmethod
    def _write_all(descriptor: int, data: bytes) -> None:
        remaining = memoryview(data)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("short write while publishing cache bytes")
            remaining = remaining[written:]

    def _quota_state_payload(
        self,
        *,
        phase: str,
        max_size_bytes: int,
        physical_size_bytes: int,
        store_count: int,
        reserved_add_bytes: int = 0,
        planned_remove_bytes: int = 0,
    ) -> bytes:
        if phase not in {"clean", "prepared"}:
            raise ValueError("invalid depth cache quota phase")
        values = (max_size_bytes, physical_size_bytes, store_count, reserved_add_bytes, planned_remove_bytes)
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values):
            raise ValueError("invalid depth cache quota state")
        authority_stat = self._expected_root_stat(self._lock_authority_dir)
        cache_stat = self._expected_root_stat(self.cache_dir)
        namespace_stat = self._expected_root_stat(self._v1_dir)
        entries_stat = self._expected_root_stat(self._entries_dir)
        objects_stat = self._expected_root_stat(self._objects_dir)
        locks_stat = self._expected_root_stat(self._locks_dir)
        return canonicalize_json(
            {
                "schema": _QUOTA_STATE_SCHEMA,
                "phase": phase,
                "authority_device": authority_stat.st_dev,
                "authority_inode": authority_stat.st_ino,
                "cache_device": cache_stat.st_dev,
                "cache_inode": cache_stat.st_ino,
                "namespace_device": namespace_stat.st_dev,
                "namespace_inode": namespace_stat.st_ino,
                "entries_device": entries_stat.st_dev,
                "entries_inode": entries_stat.st_ino,
                "objects_device": objects_stat.st_dev,
                "objects_inode": objects_stat.st_ino,
                "locks_device": locks_stat.st_dev,
                "locks_inode": locks_stat.st_ino,
                "max_size_bytes": max_size_bytes,
                "physical_size_bytes": physical_size_bytes,
                "store_count": store_count,
                "reserved_add_bytes": reserved_add_bytes,
                "planned_remove_bytes": planned_remove_bytes,
            }
        )

    def _write_quota_state_locked(
        self,
        *,
        phase: str,
        max_size_bytes: int,
        physical_size_bytes: int,
        store_count: int,
        reserved_add_bytes: int = 0,
        planned_remove_bytes: int = 0,
    ) -> None:
        data = self._quota_state_payload(
            phase=phase,
            max_size_bytes=max_size_bytes,
            physical_size_bytes=physical_size_bytes,
            store_count=store_count,
            reserved_add_bytes=reserved_add_bytes,
            planned_remove_bytes=planned_remove_bytes,
        )
        temp_name = f"{_QUOTA_STATE_NAME}.tmp-{secrets.token_hex(16)}"
        descriptor = -1
        rollback_descriptor = -1
        temp_stat: Optional[os.stat_result] = None
        published_stat: Optional[os.stat_result] = None
        committed = False
        try:
            with self._open_root_descriptor(self._locks_dir) as root_descriptor:
                # Retain an identity-bound cleanup capability even when the
                # locks root is replaced before the context can validate it.
                rollback_descriptor = os.dup(root_descriptor)
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
                descriptor = os.open(temp_name, flags, 0o644, dir_fd=root_descriptor)
                temp_stat = os.fstat(descriptor)
                if not stat.S_ISREG(temp_stat.st_mode) or temp_stat.st_nlink != 1:
                    raise OSError("unsafe depth cache quota state temporary path")
                os.fchmod(descriptor, 0o644)
                self._write_all(descriptor, data)
                os.fsync(descriptor)
                closing_descriptor = descriptor
                descriptor = -1
                os.close(closing_descriptor)
                published_stat = temp_stat
                os.replace(
                    temp_name,
                    _QUOTA_STATE_NAME,
                    src_dir_fd=root_descriptor,
                    dst_dir_fd=root_descriptor,
                )
                os.fsync(root_descriptor)
                destination_stat = os.stat(
                    _QUOTA_STATE_NAME,
                    dir_fd=root_descriptor,
                    follow_symlinks=False,
                )
                if not os.path.samestat(destination_stat, published_stat):
                    raise OSError("depth cache quota state changed during publication")
                self._validate_namespace_roots()
            committed = True
        finally:
            try:
                if descriptor >= 0:
                    closing_descriptor = descriptor
                    descriptor = -1
                    os.close(closing_descriptor)
            finally:
                if rollback_descriptor >= 0:
                    closing_rollback_descriptor = rollback_descriptor
                    rollback_descriptor = -1
                    try:
                        cleanup_changed = False
                        if temp_stat is not None:
                            cleanup_changed = self._unlink_matching_inode(
                                closing_rollback_descriptor,
                                temp_name,
                                temp_stat,
                            )
                        published_is_current = False
                        if published_stat is not None:
                            try:
                                current_published = os.stat(
                                    _QUOTA_STATE_NAME,
                                    dir_fd=closing_rollback_descriptor,
                                    follow_symlinks=False,
                                )
                            except OSError:
                                pass
                            else:
                                published_is_current = os.path.samestat(current_published, published_stat)
                        if (
                            not committed
                            and published_stat is not None
                            and (not published_is_current or not self._namespace_roots_match_expected())
                        ):
                            cleanup_changed = (
                                self._unlink_matching_inode(
                                    closing_rollback_descriptor,
                                    _QUOTA_STATE_NAME,
                                    published_stat,
                                )
                                or cleanup_changed
                            )
                        if cleanup_changed:
                            os.fsync(closing_rollback_descriptor)
                    finally:
                        os.close(closing_rollback_descriptor)

    def _cleanup_quota_temps_locked(self) -> None:
        """Remove exact abandoned quota temporaries without scanning data."""

        prefix = f"{_QUOTA_STATE_NAME}.tmp-"
        expected_length = len(prefix) + 32
        with self._open_root_descriptor(self._locks_dir) as root_descriptor:
            removed = False
            for entry in os.scandir(root_descriptor):
                name = entry.name
                token = name[len(prefix) :]
                is_quota_temp = (
                    len(name) == expected_length
                    and name.startswith(prefix)
                    and all(character in "0123456789abcdef" for character in token)
                )
                if not is_quota_temp and not self._is_governed_removal_name(name):
                    continue
                try:
                    file_stat = os.stat(name, dir_fd=root_descriptor, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                is_owned_removal = self._removal_name_matches_stat(name, file_stat)
                if stat.S_ISREG(file_stat.st_mode) and file_stat.st_nlink == 1 and (is_quota_temp or is_owned_removal):
                    removed = self._unlink_matching_inode(root_descriptor, name, file_stat) or removed
            if removed:
                os.fsync(root_descriptor)

    def _read_quota_state_locked(self) -> Optional[_QuotaState]:
        with self._open_root_descriptor(self._locks_dir) as root_descriptor:
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
            try:
                descriptor = os.open(_QUOTA_STATE_NAME, flags, dir_fd=root_descriptor)
            except FileNotFoundError:
                return None
            try:
                descriptor_stat = os.fstat(descriptor)
                path_stat = os.stat(_QUOTA_STATE_NAME, dir_fd=root_descriptor, follow_symlinks=False)
                if (
                    not stat.S_ISREG(descriptor_stat.st_mode)
                    or descriptor_stat.st_nlink != 1
                    or descriptor_stat.st_size <= 0
                    or descriptor_stat.st_size > 4096
                    or not os.path.samestat(descriptor_stat, path_stat)
                ):
                    raise OSError("unsafe depth cache quota state path")
                raw = os.read(descriptor, 4097)
                if len(raw) != descriptor_stat.st_size:
                    raise OSError("depth cache quota state changed while read")
                final_descriptor_stat = os.fstat(descriptor)
                final_path_stat = os.stat(_QUOTA_STATE_NAME, dir_fd=root_descriptor, follow_symlinks=False)
                if (
                    not stat.S_ISREG(final_descriptor_stat.st_mode)
                    or final_descriptor_stat.st_nlink != 1
                    or not os.path.samestat(final_descriptor_stat, descriptor_stat)
                    or not os.path.samestat(final_descriptor_stat, final_path_stat)
                    or final_descriptor_stat.st_size != descriptor_stat.st_size
                    or final_descriptor_stat.st_mtime_ns != descriptor_stat.st_mtime_ns
                    or final_descriptor_stat.st_ctime_ns != descriptor_stat.st_ctime_ns
                ):
                    raise OSError("depth cache quota state changed or was replaced while read")
            finally:
                os.close(descriptor)

        try:
            value = json.loads(raw, object_pairs_hook=_json_object_without_duplicates)
            _validate_json_structure_bounds(value)
            if not isinstance(value, dict) or frozenset(value) != _QUOTA_STATE_KEYS:
                return None
            if canonicalize_json(value) != raw or value["schema"] != _QUOTA_STATE_SCHEMA:
                return None
            if value["phase"] not in {"clean", "prepared"}:
                return None
            integer_fields = (
                "authority_device",
                "authority_inode",
                "cache_device",
                "cache_inode",
                "namespace_device",
                "namespace_inode",
                "entries_device",
                "entries_inode",
                "objects_device",
                "objects_inode",
                "locks_device",
                "locks_inode",
                "max_size_bytes",
                "physical_size_bytes",
                "store_count",
                "reserved_add_bytes",
                "planned_remove_bytes",
            )
            if any(
                not isinstance(value[field], int) or isinstance(value[field], bool) or value[field] < 0
                for field in integer_fields
            ):
                return None
            expected_root_fields = (
                ("authority_device", "authority_inode", self._lock_authority_dir),
                ("cache_device", "cache_inode", self.cache_dir),
                ("namespace_device", "namespace_inode", self._v1_dir),
                ("entries_device", "entries_inode", self._entries_dir),
                ("objects_device", "objects_inode", self._objects_dir),
                ("locks_device", "locks_inode", self._locks_dir),
            )
            for device_field, inode_field, root in expected_root_fields:
                expected_stat = self._expected_root_stat(root)
                if (value[device_field], value[inode_field]) != (
                    expected_stat.st_dev,
                    expected_stat.st_ino,
                ):
                    return None
            if value["phase"] == "clean" and (value["reserved_add_bytes"] != 0 or value["planned_remove_bytes"] != 0):
                return None
            return _QuotaState(
                phase=value["phase"],
                max_size_bytes=value["max_size_bytes"],
                physical_size_bytes=value["physical_size_bytes"],
                store_count=value["store_count"],
                reserved_add_bytes=value["reserved_add_bytes"],
                planned_remove_bytes=value["planned_remove_bytes"],
            )
        except (RecursionError, TypeError, ValueError, json.JSONDecodeError, _DuplicateJsonKeyError):
            return None

    def _commit_clean_quota_state_locked(
        self,
        *,
        max_size_bytes: int,
        physical_size_bytes: int,
        store_count: int,
    ) -> _QuotaState:
        self._write_quota_state_locked(
            phase="clean",
            max_size_bytes=max_size_bytes,
            physical_size_bytes=physical_size_bytes,
            store_count=store_count,
        )
        return _QuotaState(
            phase="clean",
            max_size_bytes=max_size_bytes,
            physical_size_bytes=physical_size_bytes,
            store_count=store_count,
        )

    def _recover_quota_locked(
        self,
        max_size_bytes: int,
        *,
        store_count: int = 0,
        rejected_pointer: Optional[tuple[Path, os.stat_result]] = None,
    ) -> _QuotaState:
        snapshot = self._scan_cache_locked()
        if rejected_pointer is not None:
            pointer_path, pointer_stat = rejected_pointer
            snapshot = self._snapshot_for_pointer_change(
                snapshot,
                pointer_path=pointer_path,
                expected_stat=pointer_stat,
            )
        plan = self._plan_reconcile(snapshot, target_bytes=max_size_bytes)
        self._write_quota_state_locked(
            phase="prepared",
            max_size_bytes=max_size_bytes,
            physical_size_bytes=snapshot.physical_size_bytes,
            store_count=store_count,
            planned_remove_bytes=snapshot.physical_size_bytes - plan.final_size_bytes,
        )
        self._apply_reconcile_plan_locked(plan)
        return self._commit_clean_quota_state_locked(
            max_size_bytes=max_size_bytes,
            physical_size_bytes=plan.final_size_bytes,
            store_count=store_count,
        )

    def _load_quota_state_locked(self) -> _QuotaState:
        self._cleanup_quota_temps_locked()
        state = self._read_quota_state_locked()
        if state is None:
            return self._recover_quota_locked(self._configured_max_size_bytes)
        if state.phase == "prepared":
            state = self._recover_quota_locked(state.max_size_bytes, store_count=state.store_count)
        if state.max_size_bytes != self._configured_max_size_bytes:
            raise ValueError("depth cache namespace maximum changed after this instance was configured")
        return state

    def _configure_quota_locked(
        self,
        max_size_bytes: int,
        *,
        expected_previous_max_size_bytes: Optional[int] = None,
    ) -> _QuotaState:
        self._cleanup_quota_temps_locked()
        state = self._read_quota_state_locked()
        if state is None:
            recovery_limit = max_size_bytes if expected_previous_max_size_bytes is None else expected_previous_max_size_bytes
            state = self._recover_quota_locked(recovery_limit)
        elif state.phase == "prepared":
            state = self._recover_quota_locked(state.max_size_bytes, store_count=state.store_count)
        if state.max_size_bytes == max_size_bytes:
            return state
        if expected_previous_max_size_bytes is None:
            raise ValueError("depth cache namespace is already configured with a different maximum")
        if state.max_size_bytes != expected_previous_max_size_bytes:
            raise ValueError("depth cache namespace maximum changed before the requested resize")
        if state.physical_size_bytes <= max_size_bytes:
            return self._commit_clean_quota_state_locked(
                max_size_bytes=max_size_bytes,
                physical_size_bytes=state.physical_size_bytes,
                store_count=state.store_count,
            )
        snapshot = self._scan_cache_locked()
        plan = self._plan_reconcile(snapshot, target_bytes=max_size_bytes)
        self._write_quota_state_locked(
            phase="prepared",
            max_size_bytes=max_size_bytes,
            physical_size_bytes=snapshot.physical_size_bytes,
            store_count=state.store_count,
            planned_remove_bytes=snapshot.physical_size_bytes - plan.final_size_bytes,
        )
        self._apply_reconcile_plan_locked(plan)
        return self._commit_clean_quota_state_locked(
            max_size_bytes=max_size_bytes,
            physical_size_bytes=plan.final_size_bytes,
            store_count=state.store_count,
        )

    @staticmethod
    def _unlink_matching_inode(directory_descriptor: int, name: str, expected: os.stat_result) -> bool:
        """Quarantine then unlink ``expected`` without removing a fixed-name replacement."""

        identity_digest = DepthCache._removal_identity_digest(expected)
        quarantine_name = f"{_REMOVAL_QUARANTINE_PREFIX}{identity_digest}-{secrets.token_hex(16)}"
        try:
            os.rename(
                name,
                quarantine_name,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
            )
        except FileNotFoundError:
            return False

        quarantine_descriptor = -1
        try:
            try:
                flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
                quarantine_descriptor = os.open(quarantine_name, flags, dir_fd=directory_descriptor)
                quarantined = os.fstat(quarantine_descriptor)
                quarantine_path_stat = os.stat(
                    quarantine_name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
            except OSError:
                return False
            if (
                not stat.S_ISREG(quarantined.st_mode)
                or quarantined.st_nlink != 1
                or not os.path.samestat(quarantined, quarantine_path_stat)
                or not os.path.samestat(quarantined, expected)
            ):
                # The atomic rename captured a replacement rather than the
                # selected inode. Best-effort restoration never overwrites a
                # newer fixed-name occupant; otherwise retain the quarantine.
                try:
                    os.link(
                        quarantine_name,
                        name,
                        src_dir_fd=directory_descriptor,
                        dst_dir_fd=directory_descriptor,
                        follow_symlinks=False,
                    )
                    os.unlink(quarantine_name, dir_fd=directory_descriptor)
                except OSError:
                    pass
                return False

            try:
                os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                # A new canonical occupant appeared after quarantine. Retain
                # the selected inode so reconciliation can account both files.
                return False
            try:
                os.unlink(quarantine_name, dir_fd=directory_descriptor)
            except OSError:
                return False
            if os.fstat(quarantine_descriptor).st_nlink != 0:
                return False
            try:
                os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
            except FileNotFoundError:
                return True
            return False
        finally:
            if quarantine_descriptor >= 0:
                os.close(quarantine_descriptor)

    def _atomic_write_namespace(self, path: Path, root: Path, data: bytes) -> None:
        shard, name = self._namespace_address(path, root)
        temp_name = f".{name}.tmp-{secrets.token_hex(16)}"
        descriptor = -1
        rollback_descriptor = -1
        temp_stat: Optional[os.stat_result] = None
        published_stat: Optional[os.stat_result] = None
        committed = False
        try:
            with self._open_shard_descriptor(root, shard, create=True) as (root_descriptor, shard_descriptor):
                # Keep a cleanup capability alive even if the context's final
                # ancestor-binding check fails after publication.
                rollback_descriptor = os.dup(shard_descriptor)
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
                descriptor = os.open(temp_name, flags, 0o644, dir_fd=shard_descriptor)
                temp_stat = os.fstat(descriptor)
                os.fchmod(descriptor, 0o644)
                self._write_all(descriptor, data)
                os.fsync(descriptor)
                closing_descriptor = descriptor
                descriptor = -1
                os.close(closing_descriptor)
                os.replace(
                    temp_name,
                    name,
                    src_dir_fd=shard_descriptor,
                    dst_dir_fd=shard_descriptor,
                )
                published_stat = temp_stat
                os.fsync(shard_descriptor)
                destination_stat = os.stat(name, dir_fd=shard_descriptor, follow_symlinks=False)
                if not os.path.samestat(destination_stat, published_stat):
                    raise OSError("cache destination changed during publication")
                self._assert_shard_binding(root_descriptor, shard, shard_descriptor)
                self._validate_namespace_roots()
            committed = True
        finally:
            try:
                if descriptor >= 0:
                    closing_descriptor = descriptor
                    descriptor = -1
                    os.close(closing_descriptor)
            finally:
                if rollback_descriptor >= 0:
                    closing_rollback_descriptor = rollback_descriptor
                    rollback_descriptor = -1
                    try:
                        cleanup_changed = False
                        if temp_stat is not None:
                            cleanup_changed = self._unlink_matching_inode(
                                closing_rollback_descriptor,
                                temp_name,
                                temp_stat,
                            )
                        if not committed and published_stat is not None:
                            cleanup_changed = (
                                self._unlink_matching_inode(
                                    closing_rollback_descriptor,
                                    name,
                                    published_stat,
                                )
                                or cleanup_changed
                            )
                        if cleanup_changed:
                            os.fsync(closing_rollback_descriptor)
                    finally:
                        os.close(closing_rollback_descriptor)

    def _unlink_namespace(
        self,
        path: Path,
        root: Path,
        *,
        expected_stat: Optional[os.stat_result] = None,
    ) -> None:
        shard, name = self._namespace_address(path, root)
        try:
            with self._open_shard_descriptor(root, shard) as (_, shard_descriptor):
                if expected_stat is None:
                    raise ValueError("cache removal requires an expected inode identity")
                if not self._unlink_matching_inode(shard_descriptor, name, expected_stat):
                    raise OSError("cache removal target disappeared or was replaced")
                os.fsync(shard_descriptor)
        except FileNotFoundError:
            return

    def _unlink_root_namespace(
        self,
        path: Path,
        root: Path,
        *,
        expected_stat: Optional[os.stat_result] = None,
    ) -> None:
        name = self._root_address(path, root)
        try:
            with self._open_root_descriptor(root) as root_descriptor:
                if expected_stat is None:
                    raise ValueError("cache removal requires an expected inode identity")
                if not self._unlink_matching_inode(root_descriptor, name, expected_stat):
                    raise OSError("cache removal target disappeared or was replaced")
                os.fsync(root_descriptor)
        except FileNotFoundError:
            return

    def _stat_namespace(self, path: Path, root: Path) -> os.stat_result:
        with self._open_namespace_regular(path, root) as (_, file_stat):
            return file_stat

    def _stat_root_namespace(self, path: Path, root: Path) -> os.stat_result:
        with self._open_root_regular(path, root) as (_, file_stat):
            return file_stat

    def _touch_namespace(self, path: Path, root: Path) -> None:
        self._set_namespace_access_time(path, root, time.time_ns())

    def _set_namespace_access_time(self, path: Path, root: Path, access_time_ns: int) -> None:
        with self._open_namespace_regular(path, root, allow_metadata_change=True) as (handle, file_stat):
            os.utime(handle.fileno(), ns=(access_time_ns, file_stat.st_mtime_ns))

    def _serialize_depth(self, depth: np.ndarray) -> tuple[np.ndarray, bytes, str]:
        array = np.asarray(depth)
        if array.ndim != 2 or array.size == 0 or not array.flags.c_contiguous or not _is_real_depth_dtype(array.dtype):
            raise ValueError("depth cache accepts only numeric 2-D C-contiguous arrays")
        if array.nbytes >= _ABSOLUTE_OBJECT_MAX_BYTES:
            raise ValueError("depth payload exceeds the absolute cache object limit")
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        serialized = buffer.getvalue()
        if len(serialized) > _ABSOLUTE_OBJECT_MAX_BYTES:
            raise ValueError("serialized depth payload exceeds the absolute cache object limit")
        return array, serialized, hashlib.sha256(serialized).hexdigest()

    def _read_pointer(self, path: Path) -> dict[str, Any]:
        with self._open_namespace_regular(path, self._entries_dir) as (handle, descriptor_stat):
            if descriptor_stat.st_size <= 0 or descriptor_stat.st_size > _POINTER_MAX_BYTES:
                raise ValueError("cache pointer byte length is outside the allowed bound")
            raw = handle.read(_POINTER_MAX_BYTES + 1)
            if len(raw) != descriptor_stat.st_size:
                raise ValueError("cache pointer changed while it was read")

        try:
            value = json.loads(raw, object_pairs_hook=_json_object_without_duplicates)
            _validate_json_structure_bounds(value)
            canonical = canonicalize_json(value)
        except RecursionError as exc:
            raise ValueError("cache pointer exceeds the structural depth limit") from exc
        if not isinstance(value, dict) or frozenset(value) != _POINTER_KEYS:
            raise ValueError("cache pointer has an invalid closed schema")
        if canonical != raw:
            raise ValueError("cache pointer is not canonical JSON")
        self._validate_pointer(value)
        if value["byte_length"] > _ABSOLUTE_OBJECT_MAX_BYTES:
            raise ValueError("cached NumPy object exceeds the absolute object limit")
        return value

    @staticmethod
    def _validate_pointer(pointer: dict[str, Any]) -> None:
        if pointer["schema"] != DEPTH_CACHE_POINTER_SCHEMA or pointer["cache_schema"] != DEPTH_CACHE_SCHEMA:
            raise ValueError("cache pointer schema is unsupported")
        for field_name in (
            "cache_key",
            "execution_identity_sha256",
            "config_fingerprint_sha256",
            "input_content_sha256",
            "materialized_weights_sha256",
            "dependency_lock_sha256",
            "npy_sha256",
        ):
            if not _is_sha256(pointer[field_name]):
                raise ValueError(f"cache pointer {field_name} is invalid")
        expected_key = _cache_key_for_execution_identity(pointer["execution_identity_sha256"])
        if pointer["cache_key"] != expected_key:
            raise ValueError("cache pointer key does not bind its execution identity")
        constituents = pointer["model_constituents"]
        if not isinstance(constituents, list) or not constituents:
            raise ValueError("cache pointer requires model constituents")
        for constituent in constituents:
            _validate_constituent(constituent)
        ordinals = [constituent["constituent_ordinal"] for constituent in constituents]
        if ordinals != sorted(set(ordinals)):
            raise ValueError("cache pointer model constituents are not uniquely ordered")
        byte_length = pointer["byte_length"]
        if not isinstance(byte_length, int) or isinstance(byte_length, bool) or byte_length <= 0:
            raise ValueError("cache pointer byte length is invalid")
        shape = pointer["shape"]
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0 for dimension in shape)
        ):
            raise ValueError("cache pointer shape is invalid")
        dtype = pointer["dtype"]
        if not isinstance(dtype, str) or not dtype:
            raise ValueError("cache pointer dtype is invalid")
        parsed_dtype = np.dtype(dtype)
        if not _is_real_depth_dtype(parsed_dtype):
            raise ValueError("cache pointer dtype is invalid")
        payload_bytes = shape[0] * shape[1] * parsed_dtype.itemsize
        header_bytes = byte_length - payload_bytes
        if payload_bytes <= 0 or header_bytes <= 0 or header_bytes > _MAX_NPY_HEADER_BYTES:
            raise ValueError("cache pointer NumPy payload size is inconsistent")

    @staticmethod
    def _read_actual_npy_header(handle: BinaryIO) -> tuple[tuple[int, int], np.dtype[Any], int]:
        handle.seek(0)
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                handle,
                max_header_size=_MAX_NPY_HEADER_BYTES,
            )
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(
                handle,
                max_header_size=_MAX_NPY_HEADER_BYTES,
            )
        else:
            raise ValueError(f"unsupported cached NumPy format version: {version!r}")

        header_end = handle.tell()
        if header_end <= 0 or header_end > _MAX_NPY_HEADER_BYTES + _MAX_NPY_PREAMBLE_BYTES:
            raise ValueError("cached NumPy header exceeds the supported bound")
        if (
            not isinstance(shape, tuple)
            or len(shape) != 2
            or any(not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0 for dimension in shape)
        ):
            raise ValueError("cached NumPy header shape is invalid")
        if fortran_order:
            raise ValueError("Fortran-ordered cached NumPy payloads are unsupported")
        parsed_dtype = np.dtype(dtype)
        if not _is_real_depth_dtype(parsed_dtype):
            raise ValueError("cached NumPy header dtype is invalid")
        return (shape[0], shape[1]), parsed_dtype, header_end

    @staticmethod
    def _assert_pointer_matches_object(pointer: Mapping[str, Any], record: _VerifiedObject) -> None:
        if pointer["npy_sha256"] != record.npy_sha256:
            raise ValueError("cache pointer digest does not match its CAS object")
        if pointer["byte_length"] != record.byte_length:
            raise ValueError("cached NumPy byte length does not match its pointer")
        if tuple(pointer["shape"]) != record.shape:
            raise ValueError("cached NumPy shape does not match its pointer")
        if pointer["dtype"] != record.dtype:
            raise ValueError("cached NumPy dtype does not match its pointer")

    def _inspect_verified_object(self, path: Path) -> tuple[np.ndarray, _VerifiedObject]:
        expected_digest = path.stem
        if not _is_sha256(expected_digest) or path.suffix != ".npy":
            raise ValueError("cache object path does not encode a valid digest")
        with self._open_namespace_regular(path, self._objects_dir) as (handle, initial_stat):
            if initial_stat.st_size <= 0 or initial_stat.st_size > _ABSOLUTE_OBJECT_MAX_BYTES:
                raise ValueError("cached NumPy byte length is outside the supported bound")
            digest = hashlib.sha256()
            while chunk := handle.read(_HASH_CHUNK_BYTES):
                digest.update(chunk)
            if digest.hexdigest() != expected_digest:
                raise ValueError("cached NumPy checksum does not match its CAS path")
            shape, parsed_dtype, header_end = self._read_actual_npy_header(handle)
            payload_bytes = shape[0] * shape[1] * parsed_dtype.itemsize
            if header_end + payload_bytes != initial_stat.st_size:
                raise ValueError("cached NumPy header and payload length are inconsistent")
            handle.seek(0)
            depth = np.load(handle, allow_pickle=False, max_header_size=_MAX_NPY_HEADER_BYTES)
            if not isinstance(depth, np.ndarray):
                raise ValueError("cached NumPy payload is not an array")
            if depth.ndim != 2 or depth.size == 0 or not depth.flags.c_contiguous or not _is_real_depth_dtype(depth.dtype):
                raise ValueError("cached NumPy payload is not a numeric 2-D C-contiguous depth array")
            if tuple(depth.shape) != shape or depth.dtype.str != parsed_dtype.str:
                raise ValueError("cached NumPy payload does not match its header")

            handle.seek(0)
            final_digest = hashlib.sha256()
            while chunk := handle.read(_HASH_CHUNK_BYTES):
                final_digest.update(chunk)
            final_stat = os.fstat(handle.fileno())
            if final_digest.hexdigest() != expected_digest or initial_stat.st_size != final_stat.st_size:
                raise ValueError("cached NumPy object changed during validation")
            record = _VerifiedObject(
                path=path,
                file_stat=final_stat,
                npy_sha256=expected_digest,
                byte_length=final_stat.st_size,
                shape=shape,
                dtype=parsed_dtype.str,
            )
            return depth, record

    def _read_verified_array(self, path: Path, pointer: Mapping[str, Any]) -> tuple[np.ndarray, os.stat_result]:
        depth, record = self._inspect_verified_object(path)
        self._assert_pointer_matches_object(pointer, record)
        return depth, record.file_stat

    @staticmethod
    def _pointer_matches_identity(pointer: Mapping[str, Any], projection: Mapping[str, Any]) -> bool:
        return all(pointer[field_name] == projection[field_name] for field_name in projection)

    @overload
    def get(self, identity: MaterializedExecutionIdentityV3) -> Optional[np.ndarray]: ...

    @overload
    def get(self, image_sha256: str, config_fingerprint: str) -> None: ...

    def get(
        self,
        *args: object,
        identity: object = _ARGUMENT_MISSING,
        image_sha256: object = _ARGUMENT_MISSING,
        config_fingerprint: object = _ARGUMENT_MISSING,
    ) -> Optional[np.ndarray]:
        """Return a verified hit, or a safe miss for the legacy two-key API.

        Historical callers supplied ``(image_sha256, config_fingerprint)``.
        Those values cannot prove model/runtime identity, so the compatibility
        adapter deliberately returns a miss instead of raising or consulting
        the v3 namespace.
        """

        if args:
            if len(args) > 2 or identity is not _ARGUMENT_MISSING or image_sha256 is not _ARGUMENT_MISSING:
                logger.warning("Depth cache bypassed unsupported execution identity arguments")
                return None
            identity = args[0]
            if len(args) == 2:
                if config_fingerprint is not _ARGUMENT_MISSING:
                    logger.warning("Depth cache bypassed ambiguous execution identity arguments")
                    return None
                config_fingerprint = args[1]
        if image_sha256 is not _ARGUMENT_MISSING:
            if identity is not _ARGUMENT_MISSING:
                logger.warning("Depth cache bypassed ambiguous execution identity arguments")
                return None
            identity = image_sha256
        if config_fingerprint is not _ARGUMENT_MISSING or not isinstance(identity, MaterializedExecutionIdentityV3):
            logger.warning("Depth cache bypassed a legacy or incomplete execution identity")
            return None

        try:
            self._validate_namespace_roots()
            projection = _identity_projection(identity)
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("Depth cache bypassed an invalid execution identity: %s", exc)
            return None
        cache_key = projection["cache_key"]
        pointer_path = self._entry_path(cache_key)

        try:
            with self._locked_shards((self._shard_index(cache_key),)):
                pointer_present = False
                pointer_stat: Optional[os.stat_result] = None
                try:
                    pointer_stat = self._stat_namespace(pointer_path, self._entries_dir)
                    pointer_present = True
                    pointer = self._read_pointer(pointer_path)
                    if not self._pointer_matches_identity(pointer, projection):
                        raise ValueError("cache pointer does not match the requested execution identity")
                    object_path = self._object_path(pointer["npy_sha256"])
                    depth, _ = self._read_verified_array(object_path, pointer)
                    self._touch_namespace(pointer_path, self._entries_dir)
                    self._touch_namespace(object_path, self._objects_dir)
                    logger.debug("Verified depth cache hit: %s", cache_key)
                    return depth
                except FileNotFoundError:
                    if pointer_present and pointer_stat is not None:
                        state = self._load_quota_state_locked()
                        self._recover_quota_locked(
                            state.max_size_bytes,
                            store_count=state.store_count,
                            rejected_pointer=(pointer_path, pointer_stat),
                        )
                    return None
                except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                    logger.warning("Rejected depth cache entry %s: %s", cache_key, exc)
                    state = self._load_quota_state_locked()
                    self._recover_quota_locked(
                        state.max_size_bytes,
                        store_count=state.store_count,
                        rejected_pointer=(pointer_path, pointer_stat) if pointer_stat is not None else None,
                    )
                    return None
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            logger.warning("Rejected depth cache entry %s: %s", cache_key, exc)
            return None

    def _pointer_for(
        self,
        projection: Mapping[str, Any],
        array: np.ndarray,
        serialized: bytes,
        npy_sha256: str,
    ) -> dict[str, Any]:
        return {
            "schema": DEPTH_CACHE_POINTER_SCHEMA,
            "cache_schema": DEPTH_CACHE_SCHEMA,
            **projection,
            "npy_sha256": npy_sha256,
            "byte_length": len(serialized),
            "shape": list(array.shape),
            "dtype": array.dtype.str,
        }

    def _publish_object(
        self,
        object_path: Path,
        serialized: bytes,
        npy_sha256: str,
        *,
        existing_record: Optional[_VerifiedObject] = None,
    ) -> bool:
        if existing_record is not None:
            current_stat = self._stat_namespace(object_path, self._objects_dir)
            if (
                existing_record.path != object_path
                or existing_record.npy_sha256 != npy_sha256
                or existing_record.byte_length != len(serialized)
                or not os.path.samestat(current_stat, existing_record.file_stat)
                or current_stat.st_size != existing_record.file_stat.st_size
                or current_stat.st_mtime_ns != existing_record.file_stat.st_mtime_ns
                or current_stat.st_ctime_ns != existing_record.file_stat.st_ctime_ns
            ):
                raise ValueError("verified cache object does not match publication bytes")
            return False

        def verify_existing() -> None:
            with self._open_namespace_regular(object_path, self._objects_dir) as (handle, descriptor_stat):
                if descriptor_stat.st_size != len(serialized):
                    raise ValueError("immutable cache object path contains different bytes")
                digest = hashlib.sha256()
                while chunk := handle.read(_HASH_CHUNK_BYTES):
                    digest.update(chunk)
                if digest.hexdigest() != npy_sha256:
                    raise ValueError("immutable cache object path contains different bytes")

        wrote_object = False
        try:
            verify_existing()
        except FileNotFoundError:
            self._atomic_write_namespace(object_path, self._objects_dir, serialized)
            verify_existing()
            wrote_object = True
        return wrote_object

    @overload
    def store(self, identity: MaterializedExecutionIdentityV3, depth: np.ndarray) -> bool: ...

    @overload
    def store(self, image_sha256: str, config_fingerprint: str, depth: np.ndarray) -> bool: ...

    def store(
        self,
        *args: object,
        identity: object = _ARGUMENT_MISSING,
        image_sha256: object = _ARGUMENT_MISSING,
        config_fingerprint: object = _ARGUMENT_MISSING,
        depth: object = _ARGUMENT_MISSING,
    ) -> bool:
        """Publish verified bytes, or reject the legacy three-argument API.

        ``store(image_sha256, config_fingerprint, depth)`` remains callable for
        compatibility but cannot authorize a v3 cache entry and therefore
        returns ``False`` without writing.
        """

        if args:
            if len(args) > 3 or identity is not _ARGUMENT_MISSING or image_sha256 is not _ARGUMENT_MISSING:
                logger.warning("Depth cache refused unsupported execution identity arguments")
                return False
            identity = args[0]
            if len(args) >= 2:
                if depth is not _ARGUMENT_MISSING:
                    logger.warning("Depth cache refused ambiguous execution identity arguments")
                    return False
                depth = args[1]
            if len(args) == 3:
                if config_fingerprint is not _ARGUMENT_MISSING:
                    logger.warning("Depth cache refused ambiguous execution identity arguments")
                    return False
                config_fingerprint = depth
                depth = args[2]
        if not (
            isinstance(identity, MaterializedExecutionIdentityV3)
            and isinstance(depth, np.ndarray)
            and config_fingerprint is _ARGUMENT_MISSING
            and image_sha256 is _ARGUMENT_MISSING
        ):
            logger.warning("Depth cache refused a legacy or incomplete execution identity")
            return False

        try:
            self._validate_namespace_roots()
            projection = _identity_projection(identity)
            array, serialized, npy_sha256 = self._serialize_depth(depth)
            pointer = self._pointer_for(projection, array, serialized, npy_sha256)
            self._validate_pointer(pointer)
            pointer_bytes = canonicalize_json(pointer)
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("Depth cache refused an invalid store: %s", exc)
            return False

        cache_key = projection["cache_key"]
        pointer_path = self._entry_path(cache_key)
        object_path = self._object_path(npy_sha256)
        try:
            with self._locked_shards(
                (
                    self._shard_index(cache_key),
                    self._shard_index(npy_sha256),
                )
            ):
                requested_limit = self._max_size_bytes
                if requested_limit != self._configured_max_size_bytes:
                    state = self._configure_quota_locked(
                        requested_limit,
                        expected_previous_max_size_bytes=self._configured_max_size_bytes,
                    )
                    self._configured_max_size_bytes = requested_limit
                else:
                    state = self._load_quota_state_locked()
                if len(serialized) + len(pointer_bytes) > state.max_size_bytes:
                    raise ValueError("cache entry exceeds the namespace physical size limit")

                force_scan = False
                pointer_present = False
                pointer_stat: Optional[os.stat_result] = None
                try:
                    pointer_stat = self._stat_namespace(pointer_path, self._entries_dir)
                    pointer_present = True
                    existing = self._read_pointer(pointer_path)
                    if not self._pointer_matches_identity(existing, projection):
                        raise ValueError("existing cache pointer has conflicting identity")
                    if existing["npy_sha256"] != npy_sha256:
                        logger.warning("Depth cache refused nondeterministic output for identity %s", cache_key)
                        return False
                    self._read_verified_array(self._object_path(npy_sha256), existing)
                    self._touch_namespace(pointer_path, self._entries_dir)
                    self._touch_namespace(object_path, self._objects_dir)
                    return True
                except FileNotFoundError:
                    force_scan = pointer_present
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    force_scan = True

                periodic_audit = (state.store_count + 1) % _SIZE_CHECK_INTERVAL == 0
                pressure_audit = state.physical_size_bytes + len(pointer_bytes) + len(serialized) > state.max_size_bytes
                plan = _ReconcilePlan(removals=(), remaining_entries=(), final_size_bytes=state.physical_size_bytes)
                existing_record: Optional[_VerifiedObject] = None
                baseline_size = state.physical_size_bytes
                if not (force_scan or periodic_audit or pressure_audit):
                    try:
                        _, existing_record = self._inspect_verified_object(object_path)
                        self._assert_pointer_matches_object(pointer, existing_record)
                    except FileNotFoundError:
                        existing_record = None
                    except (OSError, ValueError, TypeError):
                        # A changed or malformed canonical object invalidates
                        # the fast ledger path. Reconcile its actual bytes
                        # before reserving or publishing anything new.
                        force_scan = True
                        existing_record = None
                if force_scan or periodic_audit or pressure_audit:
                    snapshot = self._scan_cache_locked()
                    cleanup_paths = {(item.root, item.path) for item in snapshot.cleanup}
                    candidate_record = snapshot.objects.get(object_path)
                    if candidate_record is not None and (self._objects_dir, object_path) not in cleanup_paths:
                        self._assert_pointer_matches_object(pointer, candidate_record)
                        existing_record = candidate_record
                    publication_bytes = len(pointer_bytes) + (0 if existing_record is not None else len(serialized))
                    object_footprint = existing_record.byte_length if existing_record is not None else len(serialized)
                    if object_footprint + len(pointer_bytes) > state.max_size_bytes:
                        raise ValueError("cache entry cannot fit without removing its shared object")
                    plan = self._plan_reconcile(
                        snapshot,
                        target_bytes=state.max_size_bytes,
                        add_bytes=publication_bytes,
                        protected_objects=(object_path,) if existing_record is not None else (),
                        replacement_pointers=(
                            {pointer_path: pointer_stat} if pointer_present and pointer_stat is not None else None
                        ),
                    )
                    baseline_size = snapshot.physical_size_bytes
                else:
                    publication_bytes = len(pointer_bytes) + (0 if existing_record is not None else len(serialized))
                    object_footprint = existing_record.byte_length if existing_record is not None else len(serialized)
                    if object_footprint + len(pointer_bytes) > state.max_size_bytes:
                        raise ValueError("cache entry cannot fit without removing its shared object")
                    if state.physical_size_bytes + publication_bytes > state.max_size_bytes:
                        raise ValueError("cache quota state requires a recovery scan")

                self._write_quota_state_locked(
                    phase="prepared",
                    max_size_bytes=state.max_size_bytes,
                    physical_size_bytes=baseline_size,
                    store_count=state.store_count,
                    reserved_add_bytes=publication_bytes,
                    planned_remove_bytes=baseline_size - plan.final_size_bytes,
                )
                self._apply_reconcile_plan_locked(plan)
                wrote_object = self._publish_object(
                    object_path,
                    serialized,
                    npy_sha256,
                    existing_record=existing_record,
                )
                # The canonical pointer is the commit record and is always last.
                self._atomic_write_namespace(pointer_path, self._entries_dir, pointer_bytes)
                actual_add_bytes = len(pointer_bytes) + (len(serialized) if wrote_object else 0)
                final_size = plan.final_size_bytes + actual_add_bytes
                if final_size > state.max_size_bytes:
                    raise ValueError("cache publication exceeded its prepared physical reservation")
                self._commit_clean_quota_state_locked(
                    max_size_bytes=state.max_size_bytes,
                    physical_size_bytes=final_size,
                    store_count=state.store_count + 1,
                )
            logger.debug("Cached verified depth: %s (%.1fKB)", cache_key, len(serialized) / 1024)
            return True
        except (OSError, ValueError) as exc:
            logger.warning("Failed to cache verified depth %s: %s", cache_key, exc)
            return False

    def _pointer_paths(self) -> list[Path]:
        return self._namespace_paths(self._entries_dir, ".json")

    def _object_paths(self) -> list[Path]:
        return self._namespace_paths(self._objects_dir, ".npy")

    def _scan_cache_locked(self) -> _CacheSnapshot:
        """Inspect the namespace without deleting membership entries."""

        entries: list[_CacheEntry] = []
        cleanup: list[_Removal] = []
        referenced_objects: set[Path] = set()
        validated_objects: dict[Path, _VerifiedObject] = {}
        failed_objects: set[Path] = set()
        for pointer_path in self._pointer_paths():
            pointer_stat: Optional[os.stat_result] = None
            try:
                pointer_stat = self._stat_namespace(pointer_path, self._entries_dir)
                pointer = self._read_pointer(pointer_path)
                if pointer_path != self._entry_path(pointer["cache_key"]):
                    raise ValueError("cache pointer is stored under the wrong key")
                object_path = self._object_path(pointer["npy_sha256"])
                if object_path in failed_objects:
                    raise ValueError("cache object failed a prior physical validation")
                record = validated_objects.get(object_path)
                if record is None:
                    try:
                        _, record = self._inspect_verified_object(object_path)
                    except (OSError, ValueError, TypeError):
                        failed_objects.add(object_path)
                        raise
                    validated_objects[object_path] = record
                else:
                    current_stat = self._stat_namespace(object_path, self._objects_dir)
                    if (
                        not os.path.samestat(current_stat, record.file_stat)
                        or current_stat.st_nlink != 1
                        or current_stat.st_size != record.file_stat.st_size
                        or current_stat.st_mtime_ns != record.file_stat.st_mtime_ns
                        or current_stat.st_ctime_ns != record.file_stat.st_ctime_ns
                    ):
                        raise ValueError("shared cache object changed during housekeeping")
                self._assert_pointer_matches_object(pointer, record)
                self._set_namespace_access_time(pointer_path, self._entries_dir, pointer_stat.st_atime_ns)
                entries.append(
                    _CacheEntry(
                        pointer_path=pointer_path,
                        object_path=object_path,
                        pointer=pointer,
                        pointer_stat=pointer_stat,
                        access_time_ns=pointer_stat.st_atime_ns,
                    )
                )
                referenced_objects.add(object_path)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                if pointer_stat is None:
                    raise
                cleanup.append(_Removal(pointer_path, self._entries_dir, pointer_stat))

        for object_path in self._object_paths():
            if object_path not in referenced_objects:
                cleanup.append(_Removal(object_path, self._objects_dir, self._stat_namespace(object_path, self._objects_dir)))
        for temp_path in self._governed_temp_paths(self._entries_dir, ".json"):
            cleanup.append(_Removal(temp_path, self._entries_dir, self._stat_namespace(temp_path, self._entries_dir)))
        for temp_path in self._governed_temp_paths(self._objects_dir, ".npy"):
            cleanup.append(_Removal(temp_path, self._objects_dir, self._stat_namespace(temp_path, self._objects_dir)))
        for root in (self._entries_dir, self._objects_dir):
            for removal_path in self._namespace_removal_paths(root):
                removal_stat = self._stat_namespace(removal_path, root)
                if self._removal_name_matches_stat(removal_path.name, removal_stat):
                    cleanup.append(_Removal(removal_path, root, removal_stat))
        for legacy_path in self._root_paths(self.cache_dir, ".npy"):
            cleanup.append(_Removal(legacy_path, self.cache_dir, self._stat_root_namespace(legacy_path, self.cache_dir)))
        for removal_path in self._root_removal_paths(self.cache_dir):
            removal_stat = self._stat_root_namespace(removal_path, self.cache_dir)
            if self._removal_name_matches_stat(removal_path.name, removal_stat):
                cleanup.append(_Removal(removal_path, self.cache_dir, removal_stat))
        return _CacheSnapshot(
            entries=tuple(entries),
            objects=validated_objects,
            cleanup=tuple(cleanup),
            physical_size_bytes=self._physical_size_bytes(),
        )

    def _snapshot_for_pointer_change(
        self,
        snapshot: _CacheSnapshot,
        *,
        pointer_path: Path,
        expected_stat: os.stat_result,
        protected_objects: Iterable[Path] = (),
    ) -> _CacheSnapshot:
        """Convert one observed, rejected pointer into planned cleanup."""

        target = next((entry for entry in snapshot.entries if entry.pointer_path == pointer_path), None)
        if target is None:
            cleanup_target = next(
                (
                    removal
                    for removal in snapshot.cleanup
                    if removal.root == self._entries_dir and removal.path == pointer_path
                ),
                None,
            )
            if cleanup_target is not None and not os.path.samestat(cleanup_target.file_stat, expected_stat):
                raise OSError("cache pointer changed before rejection planning")
            return snapshot
        if not os.path.samestat(target.pointer_stat, expected_stat):
            raise OSError("cache pointer changed before reconciliation")

        remaining_entries = tuple(entry for entry in snapshot.entries if entry.pointer_path != pointer_path)
        cleanup = list(snapshot.cleanup)
        cleanup_keys = {(item.root, item.path) for item in cleanup}
        pointer_key = (self._entries_dir, target.pointer_path)
        if pointer_key not in cleanup_keys:
            cleanup.append(_Removal(target.pointer_path, self._entries_dir, target.pointer_stat))
            cleanup_keys.add(pointer_key)

        remaining_references = Counter(entry.object_path for entry in remaining_entries)
        protected = set(protected_objects)
        if target.object_path not in remaining_references and target.object_path not in protected:
            object_record = snapshot.objects.get(target.object_path)
            object_key = (self._objects_dir, target.object_path)
            if object_record is not None and object_key not in cleanup_keys:
                cleanup.append(_Removal(target.object_path, self._objects_dir, object_record.file_stat))

        return _CacheSnapshot(
            entries=remaining_entries,
            objects=snapshot.objects,
            cleanup=tuple(cleanup),
            physical_size_bytes=snapshot.physical_size_bytes,
        )

    def _physical_size_bytes(self) -> int:
        total = 0
        namespace_paths = [
            *((path, self._entries_dir) for path in self._pointer_paths()),
            *((path, self._objects_dir) for path in self._object_paths()),
            *((path, self._entries_dir) for path in self._governed_temp_paths(self._entries_dir, ".json")),
            *((path, self._objects_dir) for path in self._governed_temp_paths(self._objects_dir, ".npy")),
            *((path, self._entries_dir) for path in self._namespace_removal_paths(self._entries_dir)),
            *((path, self._objects_dir) for path in self._namespace_removal_paths(self._objects_dir)),
        ]
        for path, root in namespace_paths:
            total += self._stat_namespace(path, root).st_size
        for path in self._root_paths(self.cache_dir, ".npy"):
            total += self._stat_root_namespace(path, self.cache_dir).st_size
        for path in self._root_removal_paths(self.cache_dir):
            total += self._stat_root_namespace(path, self.cache_dir).st_size
        return total

    def _plan_reconcile(
        self,
        snapshot: _CacheSnapshot,
        *,
        target_bytes: int,
        add_bytes: int = 0,
        protected_objects: Iterable[Path] = (),
        replacement_pointers: Optional[Mapping[Path, os.stat_result]] = None,
    ) -> _ReconcilePlan:
        if target_bytes < 0 or add_bytes < 0:
            raise ValueError("cache quota cannot reserve a negative target")
        protected = set(protected_objects)
        removals = list(snapshot.cleanup)
        removal_keys = {(item.root, item.path) for item in removals}
        current_size = snapshot.physical_size_bytes - sum(item.file_stat.st_size for item in removals)
        entries = list(snapshot.entries)
        replacement_entries: list[_CacheEntry] = []
        for pointer_path, expected_stat in (replacement_pointers or {}).items():
            replacement = next((entry for entry in entries if entry.pointer_path == pointer_path), None)
            if replacement is None:
                cleanup_replacement = next(
                    (removal for removal in removals if removal.root == self._entries_dir and removal.path == pointer_path),
                    None,
                )
                if cleanup_replacement is not None and not os.path.samestat(
                    cleanup_replacement.file_stat,
                    expected_stat,
                ):
                    raise OSError("cache pointer changed before replacement planning")
                continue
            if not os.path.samestat(replacement.pointer_stat, expected_stat):
                raise OSError("cache pointer changed before replacement planning")
            replacement_entries.append(replacement)
        replacement_paths = {entry.pointer_path for entry in replacement_entries}
        entries = [entry for entry in entries if entry.pointer_path not in replacement_paths]
        current_size -= sum(entry.pointer_stat.st_size for entry in replacement_entries)
        remaining_references = Counter(entry.object_path for entry in entries)
        for replacement in replacement_entries:
            if replacement.object_path in remaining_references or replacement.object_path in protected:
                continue
            record = snapshot.objects[replacement.object_path]
            object_key = (self._objects_dir, replacement.object_path)
            if object_key not in removal_keys:
                removals.append(_Removal(replacement.object_path, self._objects_dir, record.file_stat))
                removal_keys.add(object_key)
                current_size -= record.file_stat.st_size
        evicted: set[Path] = set()
        for entry in sorted(entries, key=lambda item: (item.access_time_ns, item.pointer["cache_key"])):
            if current_size + add_bytes <= target_bytes:
                break
            pointer_key = (self._entries_dir, entry.pointer_path)
            if pointer_key not in removal_keys:
                removals.append(_Removal(entry.pointer_path, self._entries_dir, entry.pointer_stat))
                removal_keys.add(pointer_key)
                current_size -= entry.pointer_stat.st_size
            evicted.add(entry.pointer_path)
            remaining_references[entry.object_path] -= 1
            if remaining_references[entry.object_path] <= 0:
                del remaining_references[entry.object_path]
                if entry.object_path not in protected:
                    record = snapshot.objects[entry.object_path]
                    object_key = (self._objects_dir, entry.object_path)
                    if object_key not in removal_keys:
                        removals.append(_Removal(entry.object_path, self._objects_dir, record.file_stat))
                        removal_keys.add(object_key)
                        current_size -= record.file_stat.st_size
        if current_size + add_bytes > target_bytes:
            raise ValueError("cache quota cannot be satisfied without removing the protected object")
        return _ReconcilePlan(
            removals=tuple(removals),
            remaining_entries=tuple(entry for entry in entries if entry.pointer_path not in evicted),
            final_size_bytes=current_size,
        )

    def _apply_reconcile_plan_locked(self, plan: _ReconcilePlan) -> None:
        for removal in plan.removals:
            if removal.root == self.cache_dir:
                self._unlink_root_namespace(removal.path, removal.root, expected_stat=removal.file_stat)
            else:
                self._unlink_namespace(removal.path, removal.root, expected_stat=removal.file_stat)

    def _enforce_size_limit(self) -> None:
        with self._locked_shards(range(_LOCK_SHARD_COUNT)):
            requested_limit = self._max_size_bytes
            if requested_limit != self._configured_max_size_bytes:
                self._configure_quota_locked(
                    requested_limit,
                    expected_previous_max_size_bytes=self._configured_max_size_bytes,
                )
                self._configured_max_size_bytes = requested_limit
            state = self._load_quota_state_locked()
            self._recover_quota_locked(state.max_size_bytes, store_count=state.store_count)

    def clear(self) -> None:
        """Remove all entry pointers and immutable objects."""

        removed = 0
        try:
            with self._locked_shards(range(_LOCK_SHARD_COUNT)):
                state = self._load_quota_state_locked()
                snapshot = self._scan_cache_locked()
                plan = self._plan_reconcile(snapshot, target_bytes=0)
                self._write_quota_state_locked(
                    phase="prepared",
                    max_size_bytes=state.max_size_bytes,
                    physical_size_bytes=snapshot.physical_size_bytes,
                    store_count=state.store_count,
                    planned_remove_bytes=snapshot.physical_size_bytes,
                )
                self._apply_reconcile_plan_locked(plan)
                removed = len(plan.removals)
                self._commit_clean_quota_state_locked(
                    max_size_bytes=state.max_size_bytes,
                    physical_size_bytes=0,
                    store_count=state.store_count,
                )
            logger.info("Depth cache cleared: removed %d files", removed)
        except (OSError, ValueError) as exc:
            logger.warning("Depth cache clear failed: %s", exc)

    def stats(self) -> dict[str, Any]:
        """Return counts for verified entries and physical unique-object bytes."""

        try:
            with self._locked_shards(range(_LOCK_SHARD_COUNT)):
                requested_limit = self._max_size_bytes
                if requested_limit != self._configured_max_size_bytes:
                    self._configure_quota_locked(
                        requested_limit,
                        expected_previous_max_size_bytes=self._configured_max_size_bytes,
                    )
                    self._configured_max_size_bytes = requested_limit
                state = self._load_quota_state_locked()
                snapshot = self._scan_cache_locked()
                plan = self._plan_reconcile(snapshot, target_bytes=state.max_size_bytes)
                self._write_quota_state_locked(
                    phase="prepared",
                    max_size_bytes=state.max_size_bytes,
                    physical_size_bytes=snapshot.physical_size_bytes,
                    store_count=state.store_count,
                    planned_remove_bytes=snapshot.physical_size_bytes - plan.final_size_bytes,
                )
                self._apply_reconcile_plan_locked(plan)
                state = self._commit_clean_quota_state_locked(
                    max_size_bytes=state.max_size_bytes,
                    physical_size_bytes=plan.final_size_bytes,
                    store_count=state.store_count,
                )
                size_gb = state.physical_size_bytes / (1024**3)
            return {
                "entry_count": len(plan.remaining_entries),
                "size_gb": size_gb,
                "max_size_gb": state.max_size_bytes / (1024**3),
                "cache_dir": str(self.cache_dir),
            }
        except (OSError, ValueError) as exc:
            logger.debug("Failed to collect depth cache stats: %s", exc)
            try:
                physical_size = self._physical_size_bytes()
            except OSError:
                physical_size = 0
            return {
                "entry_count": 0,
                "size_gb": physical_size / (1024**3),
                "max_size_gb": self.max_size_gb,
                "cache_dir": str(self.cache_dir),
            }


__all__ = [
    "DEPTH_CACHE_POINTER_SCHEMA",
    "DEPTH_CACHE_SCHEMA",
    "DepthCache",
]
