"""Immutable generation staging and fenced publication inside ArtifactStore."""

from __future__ import annotations

import asyncio
import hashlib
import json
import mimetypes
import os
import stat
import tempfile
import uuid
from pathlib import Path
from typing import Any, Mapping, Optional

from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.orchestrator.artifact_store._filesystem import open_source_file
from transformation_portal.orchestrator.artifact_store.base import ArtifactStore, ArtifactStoreError
from transformation_portal.orchestrator.artifact_store.local import _normalize_relative_path
from transformation_portal.orchestrator.dispatch import DispatchFence

MAX_GENERATION_FILES = 200
MAX_GENERATION_FILE_BYTES = 4 * 1024**3
MAX_GENERATION_BYTES = 16 * 1024**3
MAX_MANIFEST_BYTES = 1_048_576
_CHUNK = 1024 * 1024


def validate_manifest(raw: bytes, *, fence: DispatchFence, generation_id: str) -> dict[str, Any]:
    if len(raw) > MAX_MANIFEST_BYTES:
        raise ArtifactStoreError("generation manifest exceeds byte limit")
    data = json.loads(raw)
    if (
        not isinstance(data, dict)
        or set(data) != {"schema", "job_id", "tenant_id", "generation_id", "files"}
        or data["schema"] != "tp.artifact.generation.v1"
        or data["job_id"] != fence.locator.job_id
        or data["tenant_id"] != fence.locator.tenant_id
        or data["generation_id"] != generation_id
        or canonicalize_json(data) != raw
    ):
        raise ArtifactStoreError("invalid canonical generation manifest")
    if not isinstance(data["files"], list) or len(data["files"]) > MAX_GENERATION_FILES:
        raise ArtifactStoreError("generation artifact count exceeds limit")
    paths = set()
    total = 0
    for item in data["files"]:
        if not isinstance(item, dict) or set(item) != {"path", "storage_path", "size_bytes", "sha256", "content_type"}:
            raise ArtifactStoreError("invalid generation artifact fields")
        path = _normalize_relative_path(item["path"])
        if path in paths or path != item["path"]:
            raise ArtifactStoreError("duplicate or noncanonical generation path")
        paths.add(path)
        if item["storage_path"] != f"generations/{generation_id}/{path}":
            raise ArtifactStoreError("generation storage path does not match immutable namespace")
        size = item["size_bytes"]
        if type(size) is not int or not 0 <= size <= MAX_GENERATION_FILE_BYTES:
            raise ArtifactStoreError("invalid generation artifact size")
        digest = item["sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ArtifactStoreError("invalid generation artifact digest")
        if not isinstance(item["content_type"], str) or len(item["content_type"]) > 256:
            raise ArtifactStoreError("invalid generation artifact content type")
        total += size
    if total > MAX_GENERATION_BYTES:
        raise ArtifactStoreError("generation byte count exceeds limit")
    return data


def _snapshot(source: Path, target: Path, remaining: int) -> tuple[int, str]:
    fd = open_source_file(source)
    total = 0
    digest = hashlib.sha256()
    with os.fdopen(fd, "rb") as src:
        before = os.fstat(src.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > min(MAX_GENERATION_FILE_BYTES, remaining):
            raise ArtifactStoreError("generation source is not a bounded regular file")
        with target.open("xb") as dst:
            while chunk := src.read(_CHUNK):
                total += len(chunk)
                if total > min(MAX_GENERATION_FILE_BYTES, remaining):
                    raise ArtifactStoreError("generation source grew beyond byte limit")
                digest.update(chunk)
                dst.write(chunk)
            dst.flush()
            os.fsync(dst.fileno())
        after = os.fstat(src.fileno())
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ArtifactStoreError("generation source changed while staging")
    return total, digest.hexdigest()


async def _verify_staged(store: ArtifactStore, job_id: str, path: str, size: int, digest: str) -> None:
    stream = await store.open_bytes(job_id, path)
    observed = 0
    hasher = hashlib.sha256()
    try:
        async for chunk in stream:
            observed += len(chunk)
            if observed > size:
                raise ArtifactStoreError("staged artifact exceeds manifest length")
            hasher.update(chunk)
    finally:
        close = getattr(stream, "aclose", None)
        if close is not None:
            await close()
    if observed != size or hasher.hexdigest() != digest:
        raise ArtifactStoreError("staged artifact does not match manifest digest")


class GenerationPublisher:
    """Stages bytes first; only a successful database commit grants visibility."""

    def __init__(self, *, artifact_store: ArtifactStore, record_store: Any) -> None:
        self._artifacts = artifact_store
        self._records = record_store

    async def publish(
        self,
        fence: DispatchFence,
        files: Mapping[str, Path],
        *,
        state: str,
        exit_code: Optional[int],
        artifacts: dict[str, Any],
        run_summary: dict[str, Any],
        error: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if len(files) > MAX_GENERATION_FILES:
            raise ArtifactStoreError("generation artifact count exceeds limit")
        generation_id = uuid.uuid4().hex
        items = []
        total = 0
        output_root = Path(fence.output_root)
        if not output_root.is_absolute() or ".." in output_root.parts:
            raise ArtifactStoreError("generation output root must be absolute and canonical")
        with tempfile.TemporaryDirectory(prefix="tp-generation-") as temporary:
            for index, (relative, source) in enumerate(sorted(files.items())):
                relative = _normalize_relative_path(relative)
                source = Path(source)
                if not source.is_relative_to(output_root) or ".." in source.parts:
                    raise ArtifactStoreError("generation source escapes the admitted attempt output root")
                snapshot = Path(temporary) / str(index)
                size, digest = await asyncio.to_thread(_snapshot, source, snapshot, MAX_GENERATION_BYTES - total)
                total += size
                path = f"generations/{generation_id}/{relative}"
                content_type = mimetypes.guess_type(relative)[0] or "application/octet-stream"
                await self._artifacts.write_immutable_file(fence.locator.job_id, path, snapshot, content_type=content_type)
                await _verify_staged(self._artifacts, fence.locator.job_id, path, size, digest)
                items.append(
                    {
                        "path": relative,
                        "storage_path": path,
                        "size_bytes": size,
                        "sha256": digest,
                        "content_type": content_type,
                    }
                )
                snapshot.unlink()
            raw = canonicalize_json(
                {
                    "schema": "tp.artifact.generation.v1",
                    "job_id": fence.locator.job_id,
                    "tenant_id": fence.locator.tenant_id,
                    "generation_id": generation_id,
                    "files": items,
                }
            )
            validate_manifest(raw, fence=fence, generation_id=generation_id)
            manifest_path = Path(temporary) / "manifest.json"
            manifest_path.write_bytes(raw)
            await self._artifacts.write_immutable_file(
                fence.locator.job_id,
                f"generations/{generation_id}/.manifest.json",
                manifest_path,
                content_type="application/json",
            )
            await _verify_staged(
                self._artifacts,
                fence.locator.job_id,
                f"generations/{generation_id}/.manifest.json",
                len(raw),
                hashlib.sha256(raw).hexdigest(),
            )
        return await self._records.commit_generation(
            fence,
            generation_id=generation_id,
            manifest_bytes=raw,
            state=state,
            exit_code=exit_code,
            artifacts=artifacts,
            run_summary=run_summary,
            error=error,
        )
