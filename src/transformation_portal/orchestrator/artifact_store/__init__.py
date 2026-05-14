"""Artifact-store factory keyed off ``TP_ARTIFACT_STORE``.

Phase 4 ships the Protocol + the filesystem ``local`` backend
(default, structural extract of the pre-Phase-4 helpers in
``portal.job_artifacts``) + the ``s3`` backend (boto3, lazy-import).
``app.py`` uses this factory for artifact delivery, readiness checks,
signed S3 delivery, retention metadata, and deletion.

Supported backends:

- ``local`` (default) — filesystem under ``TP_ARTIFACT_LOCAL_ROOT``
  (default ``/tmp/transformation-portal-artifacts``).
- ``s3`` — requires ``TP_ARTIFACT_BUCKET``;
  ``TP_ARTIFACT_PREFIX`` (default ``tp/artifacts``) and
  ``TP_ARTIFACT_ENDPOINT_URL`` (MinIO / LocalStack / R2) are
  optional. ``boto3`` is loaded lazily so local deployments do not
  pay the dependency cost.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from transformation_portal.orchestrator.artifact_store.base import (
    ArtifactNotFoundError,
    ArtifactObjectMetadata,
    ArtifactPathValidationError,
    ArtifactStore,
    ArtifactStoreError,
)

_BACKEND_ENV = "TP_ARTIFACT_STORE"
_BUCKET_ENV = "TP_ARTIFACT_BUCKET"
_PREFIX_ENV = "TP_ARTIFACT_PREFIX"
_ENDPOINT_ENV = "TP_ARTIFACT_ENDPOINT_URL"
_REGION_ENV = "TP_ARTIFACT_REGION"
_LOCAL_ROOT_ENV = "TP_ARTIFACT_LOCAL_ROOT"

_store: Optional[ArtifactStore] = None


def _selected_backend() -> str:
    return os.getenv(_BACKEND_ENV, "local").strip().lower() or "local"


def get_artifact_store() -> ArtifactStore:
    """Return the singleton ``ArtifactStore``, constructing it on first use."""
    global _store
    if _store is not None:
        return _store

    backend = _selected_backend()
    if backend == "local":
        from transformation_portal.orchestrator.artifact_store.local import LocalArtifactStore

        root_env = os.getenv(_LOCAL_ROOT_ENV, "").strip()
        root_dir = Path(root_env).expanduser() if root_env else None
        _store = LocalArtifactStore(root_dir=root_dir)
        return _store

    if backend == "s3":
        from transformation_portal.orchestrator.artifact_store.s3 import S3ArtifactStore

        bucket = os.getenv(_BUCKET_ENV, "").strip()
        if not bucket:
            raise ArtifactStoreError(f"{_BACKEND_ENV}=s3 requires {_BUCKET_ENV} to be set (e.g. tp-artifacts-prod).")
        prefix = os.getenv(_PREFIX_ENV, "").strip() or "tp/artifacts"
        endpoint_url = os.getenv(_ENDPOINT_ENV, "").strip() or None
        region_name = os.getenv(_REGION_ENV, "").strip() or None
        _store = S3ArtifactStore(
            bucket=bucket,
            prefix=prefix,
            endpoint_url=endpoint_url,
            region_name=region_name,
        )
        return _store

    raise ArtifactStoreError(f"Unsupported {_BACKEND_ENV}={backend!r}; expected 'local' or 's3'.")


def reset_singleton() -> None:
    """Drop the cached singleton. Tests call this between cases."""
    global _store
    _store = None


__all__ = [
    "ArtifactNotFoundError",
    "ArtifactObjectMetadata",
    "ArtifactPathValidationError",
    "ArtifactStore",
    "ArtifactStoreError",
    "get_artifact_store",
    "reset_singleton",
]
