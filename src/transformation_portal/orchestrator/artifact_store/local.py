"""Filesystem-backed ``ArtifactStore`` (Phase 4.A default).

``LocalArtifactStore`` is the structural extract of the pre-Phase-4
filesystem helpers in ``transformation_portal.portal.job_artifacts``
(path-traversal validation, SHA-256 fingerprinting, content-type
detection) plus a thin ``async`` wrapper so the surface matches the
``ArtifactStore`` contract. The legacy helpers continue to exist for
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
import os
from pathlib import Path, PurePosixPath
from typing import AsyncIterator, List, Optional

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
        if not job_id:
            raise ArtifactPathValidationError("empty_job_id")
        normalized = _normalize_relative_path(relative_path)
        job_root = (self._root / job_id).resolve()
        candidate = (job_root / normalized).resolve()
        try:
            candidate.relative_to(job_root)
        except ValueError as exc:
            raise ArtifactPathValidationError("artifact_path_outside_job_output_dir") from exc
        return candidate

    def _job_root(self, job_id: str) -> Path:
        if not job_id:
            raise ArtifactPathValidationError("empty_job_id")
        return (self._root / job_id).resolve()

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
            content_type=_artifact_content_type(path),
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
        job_root = self._job_root(job_id)
        if not job_root.is_dir():
            return []
        results: List[ArtifactObjectMetadata] = []
        for absolute in sorted(p for p in job_root.rglob("*") if p.is_file()):
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
                    content_type=_artifact_content_type(absolute),
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
        path = self._resolve(job_id, relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        size_bytes: Optional[int] = len(body)
        sha256_hex, status = self._fingerprint(path, size_bytes)
        resolved_content_type = content_type or _artifact_content_type(path)
        return ArtifactObjectMetadata(
            relative_path=_normalize_relative_path(relative_path),
            size_bytes=size_bytes,
            content_type=resolved_content_type,
            sha256_hex=sha256_hex,
            fingerprint_status=status,
        )

    async def delete(self, job_id: str, relative_path: Optional[str] = None) -> int:
        return await asyncio.to_thread(self._delete_sync, job_id, relative_path)

    def _delete_sync(self, job_id: str, relative_path: Optional[str]) -> int:
        if relative_path is None:
            job_root = self._job_root(job_id)
            if not job_root.is_dir():
                return 0
            deleted = 0
            for absolute in sorted(
                (p for p in job_root.rglob("*") if p.is_file()),
                reverse=True,
            ):
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
            return deleted

        path = self._resolve(job_id, relative_path)
        if not path.is_file():
            return 0
        path.unlink()
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


__all__ = ["LocalArtifactStore"]
