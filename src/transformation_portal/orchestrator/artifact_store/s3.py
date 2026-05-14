"""S3-backed ``ArtifactStore`` (Phase 4.A).

Activated by ``TP_ARTIFACT_STORE=s3`` plus the standard S3 env vars:

- ``TP_ARTIFACT_BUCKET`` (required)
- ``TP_ARTIFACT_PREFIX`` (default ``tp/artifacts``; the per-job
  layout is ``{prefix}/jobs/{job_id}/{relative_path}``)
- ``TP_ARTIFACT_ENDPOINT_URL`` (optional; lets the store target
  MinIO / LocalStack / R2 / any S3-compatible endpoint without
  changing client code)
- ``AWS_*`` for credentials (standard boto3 resolution chain)

Implementation notes:

- ``boto3`` is imported lazily so single-instance local deployments
  do not pay the dependency cost at import time. Construction
  validates the bucket name only; the first network call materializes
  the boto3 client.
- All boto3 calls are wrapped in ``asyncio.to_thread`` because
  ``boto3`` is fundamentally synchronous. The thread-pool model is
  acceptable for artifact I/O which is dominated by network latency
  rather than CPU.
- Fingerprinting streams the object body once and computes SHA-256 on
  the wire (no separate ``HeadObject`` + ``GetObject`` round trips).
  Objects larger than ``ARTIFACT_FINGERPRINT_MAX_BYTES`` are reported
  as ``fingerprint_status="skipped_size"`` without downloading the
  bytes, matching the ``LocalArtifactStore`` contract.
- Object metadata sets ``Content-Type`` so signed URL downloads in
  Phase 4.B serve the artifact with the correct type without an
  ``out-of-band`` ``HeadObject`` lookup.

The runtime dependency is part of the base package metadata and
lockfiles, but the module imports nothing eagerly, so local
deployments do not pay S3 client construction cost unless they set
``TP_ARTIFACT_STORE=s3``.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any, AsyncIterator, List, Optional

from transformation_portal.orchestrator.artifact_store.base import (
    ArtifactNotFoundError,
    ArtifactObjectMetadata,
    ArtifactStore,
    ArtifactStoreError,
)
from transformation_portal.orchestrator.artifact_store.local import _normalize_job_id, _normalize_relative_path
from transformation_portal.portal.job_artifacts import ARTIFACT_FINGERPRINT_MAX_BYTES, _artifact_content_type

_CHUNK_BYTES = 1024 * 1024
_DEFAULT_PREFIX = "tp/artifacts"


def _content_type_for(relative_path: str) -> str:
    return _artifact_content_type(Path(relative_path))


def _load_boto3() -> Any:
    """Lazy boto3 import with an actionable error when the dep is missing."""
    try:
        import boto3  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - covered by env where boto3 is uninstalled
        raise ArtifactStoreError(
            "TP_ARTIFACT_STORE=s3 requires the 'boto3' package. "
            "Install it (added to base.in for the S3 deployment lane) or "
            "fall back to TP_ARTIFACT_STORE=local."
        ) from exc
    return boto3


class S3ArtifactStore(ArtifactStore):
    """S3 implementation of ``ArtifactStore``.

    A single ``S3ArtifactStore`` instance is intended for the lifetime
    of the process. ``boto3.client("s3", ...)`` instances are
    thread-safe; the same client is reused across calls.
    """

    def __init__(
        self,
        *,
        bucket: str,
        prefix: str = _DEFAULT_PREFIX,
        endpoint_url: Optional[str] = None,
        region_name: Optional[str] = None,
        client: Optional[Any] = None,
    ) -> None:
        if not bucket:
            raise ArtifactStoreError("S3ArtifactStore requires a non-empty bucket (TP_ARTIFACT_BUCKET).")
        self._bucket = bucket
        self._prefix = (prefix or _DEFAULT_PREFIX).strip("/")
        self._endpoint_url = endpoint_url or None
        self._region_name = region_name or None
        self._client = client
        self._lock = asyncio.Lock()

    @property
    def backend(self) -> str:
        return "s3"

    @property
    def bucket(self) -> str:
        return self._bucket

    @property
    def key_prefix(self) -> str:
        return self._prefix

    async def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        async with self._lock:
            if self._client is not None:
                return self._client
            boto3 = _load_boto3()
            self._client = await asyncio.to_thread(
                boto3.client,
                "s3",
                endpoint_url=self._endpoint_url,
                region_name=self._region_name,
            )
            return self._client

    def _job_prefix(self, job_id: str) -> str:
        normalized_job_id = _normalize_job_id(job_id)
        return f"{self._prefix}/jobs/{normalized_job_id}"

    def _object_key(self, job_id: str, relative_path: str) -> str:
        normalized = _normalize_relative_path(relative_path)
        return f"{self._job_prefix(job_id)}/{normalized}"

    async def head(self, job_id: str, relative_path: str) -> ArtifactObjectMetadata:
        key = self._object_key(job_id, relative_path)
        client = await self._get_client()
        try:
            head = await asyncio.to_thread(client.head_object, Bucket=self._bucket, Key=key)
        except Exception as exc:  # noqa: BLE001 - boto3 ClientError surfaces here
            if _is_not_found(exc):
                raise ArtifactNotFoundError(f"artifact not found: {job_id}/{relative_path}") from exc
            raise ArtifactStoreError(f"S3 head_object failed for {key}") from exc

        size_bytes = int(head.get("ContentLength", 0)) if "ContentLength" in head else None
        content_type = head.get("ContentType") or _content_type_for(relative_path)

        if size_bytes is None:
            sha256_hex: Optional[str] = None
            status = "unavailable"
        elif size_bytes > ARTIFACT_FINGERPRINT_MAX_BYTES:
            sha256_hex = None
            status = "skipped_size"
        else:
            sha256_hex, status = await self._stream_sha256(client, key)

        return ArtifactObjectMetadata(
            relative_path=_normalize_relative_path(relative_path),
            size_bytes=size_bytes,
            content_type=content_type,
            sha256_hex=sha256_hex,
            fingerprint_status=status,
        )

    async def open_bytes(
        self,
        job_id: str,
        relative_path: str,
    ) -> AsyncIterator[bytes]:
        key = self._object_key(job_id, relative_path)
        client = await self._get_client()
        try:
            response = await asyncio.to_thread(client.get_object, Bucket=self._bucket, Key=key)
        except Exception as exc:  # noqa: BLE001 - boto3 ClientError
            if _is_not_found(exc):
                raise ArtifactNotFoundError(f"artifact not found: {job_id}/{relative_path}") from exc
            raise ArtifactStoreError(f"S3 get_object failed for {key}") from exc

        body = response["Body"]

        async def _stream() -> AsyncIterator[bytes]:
            try:
                while True:
                    chunk = await asyncio.to_thread(body.read, _CHUNK_BYTES)
                    if not chunk:
                        return
                    yield chunk
            finally:
                close = getattr(body, "close", None)
                if close is not None:
                    await asyncio.to_thread(close)

        return _stream()

    async def list_for_job(self, job_id: str) -> List[ArtifactObjectMetadata]:
        prefix = f"{self._job_prefix(job_id)}/"
        client = await self._get_client()
        results: List[ArtifactObjectMetadata] = []
        continuation: Optional[str] = None
        while True:
            kwargs = {"Bucket": self._bucket, "Prefix": prefix}
            if continuation:
                kwargs["ContinuationToken"] = continuation
            try:
                response = await asyncio.to_thread(client.list_objects_v2, **kwargs)
            except Exception as exc:  # noqa: BLE001 - boto3 ClientError
                raise ArtifactStoreError(f"S3 list_objects_v2 failed for {prefix}") from exc

            for entry in response.get("Contents", []) or []:
                key = entry["Key"]
                if not key.startswith(prefix):
                    continue
                relative = key[len(prefix) :]
                if not relative or relative.endswith("/"):
                    continue
                try:
                    head = await asyncio.to_thread(client.head_object, Bucket=self._bucket, Key=key)
                except Exception as exc:  # noqa: BLE001 - boto3 ClientError
                    if _is_not_found(exc):
                        continue
                    raise ArtifactStoreError(f"S3 head_object failed for {key}") from exc
                size_bytes = int(head.get("ContentLength", 0)) if "ContentLength" in head else None
                content_type = head.get("ContentType") or _content_type_for(relative)
                if size_bytes is None:
                    sha256_hex: Optional[str] = None
                    status = "unavailable"
                elif size_bytes > ARTIFACT_FINGERPRINT_MAX_BYTES:
                    sha256_hex = None
                    status = "skipped_size"
                else:
                    sha256_hex, status = await self._stream_sha256(client, key)
                results.append(
                    ArtifactObjectMetadata(
                        relative_path=relative,
                        size_bytes=size_bytes,
                        content_type=content_type,
                        sha256_hex=sha256_hex,
                        fingerprint_status=status,
                    )
                )

            if not response.get("IsTruncated"):
                break
            continuation = response.get("NextContinuationToken")
            if not continuation:
                break

        results.sort(key=lambda meta: meta.relative_path)
        return results

    async def write_bytes(
        self,
        job_id: str,
        relative_path: str,
        body: bytes,
        *,
        content_type: Optional[str] = None,
    ) -> ArtifactObjectMetadata:
        key = self._object_key(job_id, relative_path)
        client = await self._get_client()
        explicit_content_type = str(content_type).strip() if content_type is not None else ""
        resolved_content_type = explicit_content_type or _content_type_for(relative_path)
        try:
            await asyncio.to_thread(
                client.put_object,
                Bucket=self._bucket,
                Key=key,
                Body=body,
                ContentType=resolved_content_type,
            )
        except Exception as exc:  # noqa: BLE001 - boto3 ClientError
            raise ArtifactStoreError(f"S3 put_object failed for {key}") from exc

        size_bytes: Optional[int] = len(body)
        if size_bytes > ARTIFACT_FINGERPRINT_MAX_BYTES:
            sha256_hex: Optional[str] = None
            status = "skipped_size"
        else:
            sha256_hex = hashlib.sha256(body).hexdigest()
            status = "ok"
        return ArtifactObjectMetadata(
            relative_path=_normalize_relative_path(relative_path),
            size_bytes=size_bytes,
            content_type=resolved_content_type,
            sha256_hex=sha256_hex,
            fingerprint_status=status,
        )

    async def delete(self, job_id: str, relative_path: Optional[str] = None) -> int:
        if relative_path is None:
            # Bulk-delete every object under the job prefix.
            prefix = f"{self._job_prefix(job_id)}/"
            client = await self._get_client()
            deleted = 0
            continuation: Optional[str] = None
            while True:
                kwargs = {"Bucket": self._bucket, "Prefix": prefix}
                if continuation:
                    kwargs["ContinuationToken"] = continuation
                response = await asyncio.to_thread(client.list_objects_v2, **kwargs)
                keys = [{"Key": entry["Key"]} for entry in (response.get("Contents", []) or [])]
                if keys:
                    # ``delete_objects`` accepts up to 1000 keys per call.
                    for batch_start in range(0, len(keys), 1000):
                        batch = keys[batch_start : batch_start + 1000]
                        deleted += await self._delete_object_batch(client, batch)
                if not response.get("IsTruncated"):
                    break
                continuation = response.get("NextContinuationToken")
                if not continuation:
                    break
            return deleted

        key = self._object_key(job_id, relative_path)
        client = await self._get_client()
        try:
            await asyncio.to_thread(client.head_object, Bucket=self._bucket, Key=key)
        except Exception as exc:  # noqa: BLE001 - boto3 ClientError
            if _is_not_found(exc):
                return 0
            raise ArtifactStoreError(f"S3 head_object failed for {key}") from exc
        try:
            await asyncio.to_thread(client.delete_object, Bucket=self._bucket, Key=key)
        except Exception as exc:  # noqa: BLE001 - boto3 ClientError
            raise ArtifactStoreError(f"S3 delete_object failed for {key}") from exc
        return 1

    async def reset(self) -> None:
        """Test-only: delete every object under the configured prefix.

        Production deployments must not invoke ``reset`` — it is
        equivalent to ``aws s3 rm s3://{bucket}/{prefix}/jobs/ --recursive``
        and is destructive. Implemented to satisfy the contract test
        suite that uses a per-test key prefix so parallel cases
        (``pytest-xdist``) and shared-tenant buckets stay isolated.
        """
        client = await self._get_client()
        prefix = f"{self._prefix}/jobs/"
        continuation: Optional[str] = None
        while True:
            kwargs = {"Bucket": self._bucket, "Prefix": prefix}
            if continuation:
                kwargs["ContinuationToken"] = continuation
            response = await asyncio.to_thread(client.list_objects_v2, **kwargs)
            keys = [{"Key": entry["Key"]} for entry in (response.get("Contents", []) or [])]
            if keys:
                for batch_start in range(0, len(keys), 1000):
                    batch = keys[batch_start : batch_start + 1000]
                    await self._delete_object_batch(client, batch)
            if not response.get("IsTruncated"):
                break
            continuation = response.get("NextContinuationToken")
            if not continuation:
                break

    async def _delete_object_batch(self, client: Any, batch: List[dict[str, str]]) -> int:
        response = await asyncio.to_thread(
            client.delete_objects,
            Bucket=self._bucket,
            Delete={"Objects": batch, "Quiet": False},
        )
        errors = response.get("Errors", []) or []
        if errors:
            sample = ", ".join(f"{error.get('Key', '<unknown>')}:{error.get('Code', '<unknown>')}" for error in errors[:3])
            raise ArtifactStoreError(f"S3 delete_objects failed for {len(errors)} keys under {self._prefix}: {sample}")
        deleted = response.get("Deleted", []) or []
        return len(deleted)

    async def _stream_sha256(self, client: Any, key: str) -> tuple[Optional[str], str]:
        try:
            response = await asyncio.to_thread(client.get_object, Bucket=self._bucket, Key=key)
        except Exception as exc:  # noqa: BLE001 - boto3 ClientError
            if _is_not_found(exc):
                return None, "unavailable"
            return None, "unavailable"
        body = response["Body"]
        digest = hashlib.sha256()
        try:
            while True:
                chunk = await asyncio.to_thread(body.read, _CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
        except Exception:  # noqa: BLE001 - any read failure → "unavailable"
            return None, "unavailable"
        finally:
            close = getattr(body, "close", None)
            if close is not None:
                try:
                    await asyncio.to_thread(close)
                except Exception:  # noqa: BLE001 - best-effort
                    pass
        return digest.hexdigest(), "ok"


def _is_not_found(exc: Exception) -> bool:
    """Return ``True`` for boto3 ``ClientError`` 404 / NoSuchKey responses."""
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    error = response.get("Error", {}) or {}
    code = str(error.get("Code", "")).strip()
    if code in {"404", "NoSuchKey", "NotFound"}:
        return True
    status = (
        response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if isinstance(response.get("ResponseMetadata"), dict)
        else None
    )
    return status == 404


__all__ = ["S3ArtifactStore"]
