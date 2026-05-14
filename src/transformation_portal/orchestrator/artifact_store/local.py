"""Filesystem-backed ``ArtifactStore`` (Phase 4.A default).

``LocalArtifactStore`` is the structural extract of the pre-Phase-4
filesystem helpers in ``transformation_portal.portal.job_artifacts``
(path-traversal semantics, SHA-256 fingerprinting, content-type
detection) plus a thin ``async`` wrapper so the surface matches the
``ArtifactStore`` contract. Explicit content-type overrides are stored
in a local metadata sidecar so ``write_bytes`` / ``head`` /
``list_for_job`` expose the same metadata contract as S3. The legacy
helpers continue to exist for
the existing ``app.py`` artifact-serving routes (they will be
rewired in Phase 4.B); this module gives the new factory-based
plumbing the same guarantees through one async-shaped facade.

Object layout on disk::

    {root}/{job_id}/{relative_path}

where ``root`` defaults to ``$TP_ARTIFACT_LOCAL_ROOT`` (or
``$XDG_STATE_HOME/transformation-portal/artifacts`` /
``/tmp/transformation-portal-artifacts`` as platform fallbacks).
Production ``app.py`` deployments will continue passing
``output_dir`` per-job; the local store accepts that as the
canonical root override on construction.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
import uuid
from pathlib import Path, PurePosixPath
from typing import AsyncIterator, List, Optional

from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.orchestrator.artifact_store.base import (
    ArtifactNotFoundError,
    ArtifactObjectMetadata,
    ArtifactPathValidationError,
    ArtifactStore,
)
from transformation_portal.portal.job_artifacts import (  # noqa: PLC2701 - intentional reuse of the validated helper
    ARTIFACT_FINGERPRINT_MAX_BYTES,
    AbsoluteArtifactPathError,
    ArtifactPathOutsideJobOutputDirError,
    InvalidArtifactPathError,
    _artifact_content_type,
)

_ARTIFACT_FINGERPRINT_CHUNK_BYTES = 1024 * 1024
_DEFAULT_LOCAL_ROOT = "/tmp/transformation-portal-artifacts"
_CONTENT_TYPE_METADATA_DIR = ".artifact-store-metadata"
_RESERVED_JOB_IDS = frozenset({_CONTENT_TYPE_METADATA_DIR})


def _normalize_job_id(job_id: str) -> str:
    """Validate ``job_id`` as one safe path/key component."""
    raw = str(job_id or "").strip()
    if not raw or raw in {".", ".."} or raw in _RESERVED_JOB_IDS:
        raise ArtifactPathValidationError("invalid_job_id")
    if "/" in raw or "\\" in raw or "\x00" in raw or raw.startswith("~"):
        raise ArtifactPathValidationError("invalid_job_id")
    candidate = PurePosixPath(raw)
    if candidate.is_absolute() or len(candidate.parts) != 1:
        raise ArtifactPathValidationError("invalid_job_id")
    if any(part in {"", ".", ".."} for part in candidate.parts):
        raise ArtifactPathValidationError("invalid_job_id")
    return raw


def _normalize_relative_path(relative_path: str) -> str:
    """Reuse the legacy validation helper's contract.

    The legacy helper raises a small hierarchy
    (``InvalidArtifactPathError`` / ``AbsoluteArtifactPathError`` /
    ``ArtifactPathOutsideJobOutputDirError``) under
    ``ArtifactPathValidationError``. The Phase 4 contract collapses
    them into ``orchestrator.artifact_store.base.ArtifactPathValidationError``
    so callers only need to import one exception class; the original
    subclass identity is preserved as the ``__cause__`` for forensics.
    """
    raw = str(relative_path or "").strip()
    if not raw or raw.startswith("~") or "\x00" in raw or "\\" in raw:
        raise ArtifactPathValidationError("invalid_artifact_path") from InvalidArtifactPathError()

    candidate = PurePosixPath(raw)
    if candidate.is_absolute():
        raise ArtifactPathValidationError("absolute_artifact_path") from AbsoluteArtifactPathError()

    normalized = candidate.as_posix()
    if normalized in {"", "."}:
        raise ArtifactPathValidationError("invalid_artifact_path") from InvalidArtifactPathError()
    if any(part == ".." for part in candidate.parts):
        raise ArtifactPathValidationError("artifact_path_outside_job_output_dir") from ArtifactPathOutsideJobOutputDirError()

    return normalized


class LocalArtifactStore(ArtifactStore):
    """Filesystem implementation of ``ArtifactStore``.

    ``root_dir`` is the directory under which each ``job_id`` gets a
    subdirectory. Production deployments override via the
    ``TP_ARTIFACT_LOCAL_ROOT`` env var (resolved at construction
    time by the factory). Tests typically pass a ``tmp_path``-rooted
    directory so cases never collide.
    """

    def __init__(self, root_dir: Optional[Path] = None) -> None:
        if root_dir is None:
            root_dir = Path(os.getenv("TP_ARTIFACT_LOCAL_ROOT", _DEFAULT_LOCAL_ROOT)).expanduser()
        self._root = Path(os.path.realpath(root_dir))
        self._root.mkdir(parents=True, exist_ok=True)
        self._metadata_root = self._root / _CONTENT_TYPE_METADATA_DIR
        self._metadata_locks_guard = threading.Lock()
        self._metadata_locks: dict[str, threading.Lock] = {}

    @property
    def backend(self) -> str:
        return "local"

    @property
    def root_dir(self) -> Path:
        return self._root

    def _resolve(self, job_id: str, relative_path: str) -> Path:
        """Resolve to an absolute path inside ``{root}/{job_id}/``.

        Reuses the legacy traversal-validation logic: the relative
        path is normalised first (``..`` rejected, no absolute /
        backslash forms), then resolved via ``os.path.realpath`` and
        verified to be a descendant of ``{root}/{job_id}``. Any
        deviation raises ``ArtifactPathValidationError``.
        """
        normalized_job_id = _normalize_job_id(job_id)
        normalized = _normalize_relative_path(relative_path)
        job_root = self._resolved_job_root(normalized_job_id)
        candidate = (job_root / normalized).resolve()
        try:
            candidate.relative_to(job_root)
        except ValueError as exc:
            raise ArtifactPathValidationError("artifact_path_outside_job_output_dir") from exc
        return candidate

    def _job_root(self, job_id: str) -> Path:
        return self._resolved_job_root(_normalize_job_id(job_id))

    def _resolved_job_root(self, normalized_job_id: str) -> Path:
        job_root = (self._root / normalized_job_id).resolve()
        try:
            job_root.relative_to(self._root)
        except ValueError as exc:
            raise ArtifactPathValidationError("job_root_outside_artifact_root") from exc
        return job_root

    async def head(self, job_id: str, relative_path: str) -> ArtifactObjectMetadata:
        return await asyncio.to_thread(self._head_sync, job_id, relative_path)

    def _head_sync(self, job_id: str, relative_path: str) -> ArtifactObjectMetadata:
        path = self._resolve(job_id, relative_path)
        if not path.is_file():
            raise ArtifactNotFoundError(f"artifact not found: {job_id}/{relative_path}")
        try:
            size_bytes: Optional[int] = path.stat().st_size
        except OSError:
            size_bytes = None
        sha256_hex, status = self._fingerprint(path, size_bytes)
        return ArtifactObjectMetadata(
            relative_path=_normalize_relative_path(relative_path),
            size_bytes=size_bytes,
            content_type=self._content_type_for(job_id, relative_path, path),
            sha256_hex=sha256_hex,
            fingerprint_status=status,
        )

    async def open_bytes(
        self,
        job_id: str,
        relative_path: str,
    ) -> AsyncIterator[bytes]:
        path = self._resolve(job_id, relative_path)
        if not path.is_file():
            raise ArtifactNotFoundError(f"artifact not found: {job_id}/{relative_path}")

        async def _stream() -> AsyncIterator[bytes]:
            chunk_size = _ARTIFACT_FINGERPRINT_CHUNK_BYTES
            handle = await asyncio.to_thread(path.open, "rb")
            try:
                while True:
                    chunk = await asyncio.to_thread(handle.read, chunk_size)
                    if not chunk:
                        return
                    yield chunk
            finally:
                await asyncio.to_thread(handle.close)

        return _stream()

    async def list_for_job(self, job_id: str) -> List[ArtifactObjectMetadata]:
        return await asyncio.to_thread(self._list_for_job_sync, job_id)

    def _list_for_job_sync(self, job_id: str) -> List[ArtifactObjectMetadata]:
        normalized_job_id = _normalize_job_id(job_id)
        job_root = self._resolved_job_root(normalized_job_id)
        if not job_root.is_dir():
            return []
        content_types = self._load_content_types(normalized_job_id)
        results: List[ArtifactObjectMetadata] = []
        for absolute in self._iter_confined_files(job_root):
            relative = absolute.relative_to(job_root).as_posix()
            try:
                size_bytes: Optional[int] = absolute.stat().st_size
            except OSError:
                size_bytes = None
            sha256_hex, status = self._fingerprint(absolute, size_bytes)
            results.append(
                ArtifactObjectMetadata(
                    relative_path=relative,
                    size_bytes=size_bytes,
                    content_type=content_types.get(relative) or _artifact_content_type(absolute),
                    sha256_hex=sha256_hex,
                    fingerprint_status=status,
                )
            )
        return results

    async def write_bytes(
        self,
        job_id: str,
        relative_path: str,
        body: bytes,
        *,
        content_type: Optional[str] = None,
    ) -> ArtifactObjectMetadata:
        return await asyncio.to_thread(self._write_bytes_sync, job_id, relative_path, body, content_type)

    def _write_bytes_sync(
        self,
        job_id: str,
        relative_path: str,
        body: bytes,
        content_type: Optional[str],
    ) -> ArtifactObjectMetadata:
        normalized_job_id = _normalize_job_id(job_id)
        normalized_relative_path = _normalize_relative_path(relative_path)
        path = self._resolve(job_id, relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write_bytes(path, body)
        size_bytes: Optional[int] = len(body)
        sha256_hex, status = self._fingerprint(path, size_bytes)
        explicit_content_type = self._normalize_content_type(content_type)
        resolved_content_type = explicit_content_type or _artifact_content_type(path)
        self._record_content_type(
            normalized_job_id,
            normalized_relative_path,
            explicit_content_type,
        )
        return ArtifactObjectMetadata(
            relative_path=normalized_relative_path,
            size_bytes=size_bytes,
            content_type=resolved_content_type,
            sha256_hex=sha256_hex,
            fingerprint_status=status,
        )

    async def write_file(
        self,
        job_id: str,
        relative_path: str,
        source_path: Path,
        *,
        content_type: Optional[str] = None,
    ) -> ArtifactObjectMetadata:
        return await asyncio.to_thread(self._write_file_sync, job_id, relative_path, source_path, content_type)

    def _write_file_sync(
        self,
        job_id: str,
        relative_path: str,
        source_path: Path,
        content_type: Optional[str],
    ) -> ArtifactObjectMetadata:
        normalized_job_id = _normalize_job_id(job_id)
        normalized_relative_path = _normalize_relative_path(relative_path)
        source = Path(source_path)
        if not source.is_file():
            raise ArtifactNotFoundError(f"source artifact not found: {source}")
        path = self._resolve(job_id, relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            source_resolved = source.resolve()
            path_resolved = path.resolve()
        except OSError:
            source_resolved = source
            path_resolved = path

        if source_resolved != path_resolved:
            self._atomic_copy_file(path, source)
        size_bytes: Optional[int]
        try:
            size_bytes = path.stat().st_size
        except OSError:
            size_bytes = None
        sha256_hex, status = self._fingerprint(path, size_bytes)
        explicit_content_type = self._normalize_content_type(content_type)
        resolved_content_type = explicit_content_type or _artifact_content_type(path)
        self._record_content_type(
            normalized_job_id,
            normalized_relative_path,
            explicit_content_type,
        )
        return ArtifactObjectMetadata(
            relative_path=normalized_relative_path,
            size_bytes=size_bytes,
            content_type=resolved_content_type,
            sha256_hex=sha256_hex,
            fingerprint_status=status,
        )

    async def delete(self, job_id: str, relative_path: Optional[str] = None) -> int:
        return await asyncio.to_thread(self._delete_sync, job_id, relative_path)

    def _delete_sync(self, job_id: str, relative_path: Optional[str]) -> int:
        if relative_path is None:
            normalized_job_id = _normalize_job_id(job_id)
            job_root = self._resolved_job_root(normalized_job_id)
            if not job_root.is_dir():
                self._delete_content_type_metadata(normalized_job_id)
                return 0
            deleted = 0
            for absolute in sorted(self._iter_confined_files(job_root), reverse=True):
                absolute.unlink()
                deleted += 1
            # Remove empty directories left behind, root-last.
            for sub in sorted(
                (p for p in job_root.rglob("*") if p.is_dir()),
                key=lambda p: len(p.parts),
                reverse=True,
            ):
                try:
                    sub.rmdir()
                except OSError:
                    pass
            try:
                job_root.rmdir()
            except OSError:
                pass
            self._delete_content_type_metadata(normalized_job_id)
            return deleted

        normalized_job_id = _normalize_job_id(job_id)
        normalized_relative_path = _normalize_relative_path(relative_path)
        path = self._resolve(job_id, relative_path)
        if not path.is_file():
            return 0
        path.unlink()
        self._record_content_type(normalized_job_id, normalized_relative_path, None)
        return 1

    async def reset(self) -> None:
        await asyncio.to_thread(self._reset_sync)

    def _reset_sync(self) -> None:
        if not self._root.is_dir():
            return
        for absolute in sorted(
            (p for p in self._root.rglob("*") if p.is_file()),
            reverse=True,
        ):
            try:
                absolute.unlink()
            except OSError:
                pass
        for sub in sorted(
            (p for p in self._root.rglob("*") if p.is_dir()),
            key=lambda p: len(p.parts),
            reverse=True,
        ):
            try:
                sub.rmdir()
            except OSError:
                pass

    def _metadata_path(self, normalized_job_id: str) -> Path:
        return self._metadata_root / f"{normalized_job_id}.json"

    def _metadata_lock(self, normalized_job_id: str) -> threading.Lock:
        with self._metadata_locks_guard:
            lock = self._metadata_locks.get(normalized_job_id)
            if lock is None:
                lock = threading.Lock()
                self._metadata_locks[normalized_job_id] = lock
            return lock

    def _iter_confined_files(self, job_root: Path) -> List[Path]:
        files: List[Path] = []
        for candidate in job_root.rglob("*"):
            if candidate.is_symlink() or not candidate.is_file():
                continue
            try:
                candidate.resolve().relative_to(job_root)
            except ValueError:
                continue
            files.append(candidate)
        return sorted(files)

    def _content_type_for(self, job_id: str, relative_path: str, path: Path) -> str:
        normalized_job_id = _normalize_job_id(job_id)
        normalized_relative_path = _normalize_relative_path(relative_path)
        return self._load_content_types(normalized_job_id).get(normalized_relative_path) or _artifact_content_type(path)

    def _load_content_types(self, normalized_job_id: str) -> dict[str, str]:
        metadata_path = self._metadata_path(normalized_job_id)
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        raw_content_types = payload.get("content_types")
        if not isinstance(raw_content_types, dict):
            return {}
        return {
            str(relative_path): str(content_type).strip()
            for relative_path, content_type in raw_content_types.items()
            if isinstance(relative_path, str) and str(content_type).strip()
        }

    def _record_content_type(
        self,
        normalized_job_id: str,
        normalized_relative_path: str,
        content_type: Optional[str],
    ) -> None:
        with self._metadata_lock(normalized_job_id):
            content_types = self._load_content_types(normalized_job_id)
            if content_type is None:
                content_types.pop(normalized_relative_path, None)
            else:
                content_types[normalized_relative_path] = content_type
            metadata_path = self._metadata_path(normalized_job_id)
            if not content_types:
                try:
                    metadata_path.unlink()
                except FileNotFoundError:
                    pass
                return
            payload = {"version": 1, "content_types": content_types}
            self._atomic_write_bytes(metadata_path, canonicalize_json(payload) + b"\n")

    def _delete_content_type_metadata(self, normalized_job_id: str) -> None:
        with self._metadata_lock(normalized_job_id):
            try:
                self._metadata_path(normalized_job_id).unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _normalize_content_type(content_type: Optional[str]) -> Optional[str]:
        if content_type is None:
            return None
        normalized = str(content_type).strip()
        return normalized or None

    @staticmethod
    def _atomic_write_bytes(path: Path, body: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
        try:
            with tmp_path.open("wb") as handle:
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        except BaseException:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
            raise

    @staticmethod
    def _atomic_copy_file(path: Path, source_path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
        try:
            with Path(source_path).open("rb") as source, tmp_path.open("wb") as handle:
                while True:
                    chunk = source.read(_ARTIFACT_FINGERPRINT_CHUNK_BYTES)
                    if not chunk:
                        break
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        except BaseException:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
            raise

    @staticmethod
    def _fingerprint(
        path: Path,
        size_bytes: Optional[int],
    ) -> tuple[Optional[str], str]:
        """SHA-256 with the bounded-bytes contract.

        Mirrors ``portal.job_artifacts._artifact_fingerprint`` —
        unavailable size returns ``("unavailable", "unavailable")``;
        oversized files return ``("skipped_size", "skipped_size")``
        without reading the bytes; everything else streams the file
        through ``hashlib.sha256``.
        """
        if size_bytes is None:
            return None, "unavailable"
        if size_bytes > ARTIFACT_FINGERPRINT_MAX_BYTES:
            return None, "skipped_size"
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                while True:
                    chunk = handle.read(_ARTIFACT_FINGERPRINT_CHUNK_BYTES)
                    if not chunk:
                        break
                    digest.update(chunk)
        except OSError:
            return None, "unavailable"
        return digest.hexdigest(), "ok"


__all__ = ["LocalArtifactStore", "_normalize_job_id", "_normalize_relative_path"]
