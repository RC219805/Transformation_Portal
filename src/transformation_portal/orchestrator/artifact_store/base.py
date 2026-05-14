"""Artifact-store contract for Phase 4 (gap doc §5.4).

Phase 4 pins the ``ArtifactStore`` Protocol + a ``LocalArtifactStore``
that wraps the existing filesystem-backed primitives in
``transformation_portal.portal.job_artifacts`` and the
``transformation_portal.lux_depth_v3.artifact_manager`` Merkle helper,
plus an ``S3ArtifactStore`` selectable via
``TP_ARTIFACT_STORE=local|s3``. The app wiring mirrors terminal-job
artifacts into the selected backend, serves local artifacts through
the orchestrator, mints short-lived S3 redirects after route auth, and
records additive lifecycle metadata for retention and deletion.

The contract pins the four pre-Phase-4 guarantees the gap doc calls
out so any S3 / managed-object backend must preserve them:

- Path-traversal validation (``job_id`` must be one safe component;
  ``relative_path`` rejects ``..`` / absolute paths; resolved paths are
  confined to the job output prefix).
- SHA-256 fingerprinting per artifact, bounded to
  ``ARTIFACT_FINGERPRINT_MAX_BYTES`` and reported via
  ``fingerprint_status`` ("ok", "skipped_size", "unavailable").
- Content-type metadata is backend-consistent: inferred from the
  artifact path by default, and explicit ``write_bytes(...,
  content_type=...)`` values are returned by subsequent ``head`` and
  ``list_for_job`` calls.
- Deterministic Merkle root via
  ``transformation_portal.lux_depth_v3.artifact_manager.compute_artifact_merkle_root``
  (sort by ``relative_path``, concat sha256 bytes, sha256 the
  concatenation). The Merkle helper is intentionally
  store-agnostic — backends produce the per-artifact ``sha256``
  metadata and feed the existing helper rather than re-implementing
  it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ArtifactObjectMetadata:
    """Per-artifact metadata returned by ``ArtifactStore.head``.

    Carries the fields the legacy ``_serialize_indexed_artifact``
    wire shape needs: ``size_bytes`` (or ``None`` if unavailable),
    SHA-256 fingerprint (with a ``fingerprint_status`` describing
    whether the bytes were hashed, skipped for size, or unavailable),
    and the content-type the artifact would be served with.

    The dataclass is intentionally backend-neutral. ``LocalArtifactStore``
    populates it from ``path.stat()`` + ``path.open("rb")`` streaming
    hash; ``S3ArtifactStore`` populates it from a ``HEAD`` request
    plus a streaming download that produces the same hash.
    """

    relative_path: str
    size_bytes: Optional[int]
    content_type: str
    sha256_hex: Optional[str]
    fingerprint_status: str  # "ok" | "skipped_size" | "unavailable"


class ArtifactStoreError(RuntimeError):
    """Base class for artifact-store failures (IO, validation, etc.)."""


class ArtifactPathValidationError(ArtifactStoreError):
    """Raised when a candidate artifact path fails contract validation.

    Mirrors the legacy ``ArtifactPathValidationError`` hierarchy in
    ``transformation_portal.portal.job_artifacts``; the store's
    public surface uses this single base class so callers do not have
    to import filesystem-specific subclasses to handle the bounded set
    of validation failures.
    """


class ArtifactNotFoundError(ArtifactStoreError):
    """Raised when ``open_bytes`` / ``head`` cannot locate an artifact."""


class ArtifactStore(ABC):
    """Async ``ArtifactStore`` Protocol.

    Implementations:

    - ``LocalArtifactStore`` — filesystem-backed, wraps the existing
      ``portal.job_artifacts`` primitives. Default for single-instance
      deployments (``TP_ARTIFACT_STORE=local``).
    - ``S3ArtifactStore`` — talks to S3 (or any
      ``TP_ARTIFACT_ENDPOINT_URL``-compatible target — MinIO,
      LocalStack, R2) via lazy ``boto3`` imports so single-instance
      deployments do not pay the dependency cost.

    Object keys follow ``{prefix}/jobs/{job_id}/{relative_path}`` with
    ``prefix`` resolved from ``TP_ARTIFACT_PREFIX`` (default
    ``tp/artifacts``). The flat S3 namespace is partitioned by job_id
    so ``list_for_job`` is a single prefix scan.
    """

    @property
    @abstractmethod
    def backend(self) -> str:
        """Return ``"local"`` or ``"s3"`` — the canonical backend identifier."""

    @abstractmethod
    async def head(self, job_id: str, relative_path: str) -> ArtifactObjectMetadata:
        """Return per-artifact metadata without reading the full bytes.

        Raises ``ArtifactNotFoundError`` when no object exists at
        ``{prefix}/jobs/{job_id}/{relative_path}``;
        ``ArtifactPathValidationError`` when the relative path tries
        to escape the job prefix.
        """

    @abstractmethod
    async def open_bytes(
        self,
        job_id: str,
        relative_path: str,
    ) -> AsyncIterator[bytes]:
        """Return an async iterator yielding the artifact's raw bytes.

        Backends stream the body so a single API request never
        materializes a multi-gigabyte artifact in memory. The
        ``LocalArtifactStore`` yields fixed-size chunks from
        ``path.open("rb")``; the ``S3ArtifactStore`` yields chunks
        from the ``GetObject`` body. Callers iterate with
        ``async for chunk in store.open_bytes(...)``.
        """

    @abstractmethod
    async def list_for_job(self, job_id: str) -> List[ArtifactObjectMetadata]:
        """List every artifact for ``job_id`` with full metadata.

        Returned in ``relative_path``-sorted order so the Phase 4
        Merkle helper produces a deterministic root regardless of
        backend (S3 ``ListObjectsV2`` returns lexicographic order;
        local listings are explicitly sorted to match). The list is
        bounded by ``MAX_INDEXED_ARTIFACTS`` at the call site, not
        inside the store.
        """

    @abstractmethod
    async def write_bytes(
        self,
        job_id: str,
        relative_path: str,
        body: bytes,
        *,
        content_type: Optional[str] = None,
    ) -> ArtifactObjectMetadata:
        """Write ``body`` for ``job_id`` at ``relative_path``.

        Used by the contract tests and small artifact upload paths.
        Validates the relative path with the same rules
        ``head``/``open_bytes`` use, sets the content-type (or infers
        from the extension when omitted), and returns the resulting
        ``ArtifactObjectMetadata`` so callers can verify the
        fingerprint without an additional ``head`` round-trip.
        """

    @abstractmethod
    async def write_file(
        self,
        job_id: str,
        relative_path: str,
        source_path: Path,
        *,
        content_type: Optional[str] = None,
    ) -> ArtifactObjectMetadata:
        """Write an existing file without materializing it in memory.

        Phase 4.B app wiring mirrors runner output into the selected
        ``ArtifactStore`` after the legacy artifact index has identified
        safe, confined local files. Backends must stream from
        ``source_path`` into the destination object, validate
        ``job_id`` / ``relative_path`` with the same rules as
        ``write_bytes``, and return metadata for the stored artifact.
        """

    async def presign_get(
        self,
        job_id: str,
        relative_path: str,
        *,
        expires_seconds: int,
        content_type: Optional[str] = None,
        content_disposition: Optional[str] = None,
        cache_control: Optional[str] = None,
    ) -> Optional[str]:
        """Return an ephemeral GET URL when the backend supports it.

        Local storage intentionally returns ``None`` so callers keep
        streaming through the orchestrator. S3 overrides this to mint a
        short-lived URL only after the app has already authorized the
        caller and validated the job/path. Optional response-header
        overrides let the orchestrator preserve the legacy artifact
        download contract (content type, attachment disposition, and
        no-store cache policy) on the final S3 response.
        """
        return None

    async def check_ready(self) -> None:
        """Raise ``ArtifactStoreError`` when the backend is not usable."""
        return None

    @abstractmethod
    async def delete(self, job_id: str, relative_path: Optional[str] = None) -> int:
        """Delete a single artifact (relative_path given) or the entire job.

        Returns the number of objects deleted. ``relative_path=None``
        scans the job prefix and deletes everything underneath; useful
        for the Phase 4.B retention-sweep path.
        """

    @abstractmethod
    async def reset(self) -> None:
        """Test-only: clear all artifact state.

        Memory / local-test fixtures call this between cases.
        Production deployments must not invoke ``reset``; the method
        is documented here so the contract is uniform across
        backends.
        """

    async def close(self) -> None:
        """Optional shutdown hook for backends that hold network connections."""
        return None


__all__ = [
    "ArtifactNotFoundError",
    "ArtifactObjectMetadata",
    "ArtifactPathValidationError",
    "ArtifactStore",
    "ArtifactStoreError",
]
