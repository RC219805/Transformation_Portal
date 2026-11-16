"""Filesystem ownership and exact completion seals for upload correlation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
from pathlib import Path
from typing import Any

from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.ingest.upload_staging import (
    BASELINE_MANIFEST_FILENAME,
    BASELINE_MANIFEST_SCHEMA,
    CAPTURE_METADATA_FILENAME,
    UPLOAD_RECEIPT_FILENAME,
    StagedUploadResult,
    _write_bytes_atomic,
    normalize_upload_relative_path,
)
from transformation_portal.orchestrator.upload_receipts import PostgresUploadReceiptStore, UploadReservation

MARKER_FILENAME = ".upload-request.v1.json"
SEAL_FILENAME = "upload_completion.tp.upload_staging.seal.v1.json"
MAX_RECOVERY_METADATA_BYTES = 4 * 1024 * 1024
_BATCH_ID = re.compile(r"upload_[A-Za-z0-9_-]{1,56}")


class UploadBatchLock:
    """Stable lock identity lives outside the directory cleanup may delete."""

    def __init__(self, root: Path, batch_id: str) -> None:
        self.fd: int | None = None
        if _BATCH_ID.fullmatch(batch_id) is None:
            raise ValueError("invalid server upload batch ID")
        root.mkdir(parents=True, exist_ok=True)
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            try:
                os.mkdir(".upload-locks", mode=0o700, dir_fd=root_fd)
            except FileExistsError:
                pass
            directory_fd = os.open(".upload-locks", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
            try:
                self.fd = os.open(batch_id + ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600, dir_fd=directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            os.close(root_fd)
        info = os.fstat(self.fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            self.close()
            raise ValueError("invalid staged upload lock")
        self.held = False

    def acquire(self, *, blocking: bool = False) -> bool:
        import fcntl

        if self.fd is None:
            raise ValueError("staged upload lock is closed")
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            return False
        self.held = True
        return True

    def close(self) -> None:
        fd = getattr(self, "fd", None)
        if fd is not None:
            os.close(fd)
            self.fd = None
        self.held = False


def _binding(reservation: UploadReservation) -> dict[str, Any]:
    return {
        key: getattr(reservation, key)
        for key in ("batch_id", "tenant_id", "owner_sha256", "request_id", "generation", "batch_root")
    }


def initialize_batch(reservation: UploadReservation) -> None:
    root = Path(reservation.batch_root)
    root.mkdir(exist_ok=False)
    _write_bytes_atomic(root / MARKER_FILENAME, canonicalize_json(_binding(reservation)))


def seal_completion(reservation: UploadReservation, result: StagedUploadResult) -> str:
    root = Path(reservation.batch_root)
    if result.batch_id != reservation.batch_id or result.batch_root != root:
        raise ValueError("upload result differs from reserved batch")
    metadata = root / "_portal"
    digests = {
        name: _digest(metadata / name)
        for name in (BASELINE_MANIFEST_FILENAME, CAPTURE_METADATA_FILENAME, UPLOAD_RECEIPT_FILENAME)
    }
    payload = {
        "schema": "tp.upload_staging.seal.v1",
        **_binding(reservation),
        "metadata_sha256": digests,
        "response_data": result.to_response_data(),
    }
    raw = canonicalize_json(payload)
    if len(raw) > MAX_RECOVERY_METADATA_BYTES:
        raise ValueError("upload completion seal exceeds metadata bound")
    _write_bytes_atomic(metadata / SEAL_FILENAME, raw)
    return hashlib.sha256(raw).hexdigest()


def _read(path: Path) -> bytes:
    with path.open("rb") as stream:
        raw = stream.read(MAX_RECOVERY_METADATA_BYTES + 1)
    if len(raw) > MAX_RECOVERY_METADATA_BYTES:
        raise ValueError("upload recovery metadata exceeds bound")
    return raw


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_completion(reservation: UploadReservation) -> tuple[str, dict[str, Any]]:
    """Re-read exact retained bytes; a receipt alone cannot authorize ready."""
    root = Path(reservation.batch_root)
    if root.is_symlink() or root.name != reservation.batch_id:
        raise ValueError("reserved upload directory changed")
    marker = json.loads(_read(root / MARKER_FILENAME))
    if marker != _binding(reservation):
        raise ValueError("upload request binding changed")
    metadata = root / "_portal"
    if (root / "input").is_symlink() or metadata.is_symlink():
        raise ValueError("upload metadata directory changed")
    seal_raw = _read(metadata / SEAL_FILENAME)
    seal_digest = hashlib.sha256(seal_raw).hexdigest()
    if reservation.receipt_sha256 is not None and reservation.receipt_sha256 != seal_digest:
        raise ValueError("upload completion differs from its durable seal digest")
    seal = json.loads(seal_raw)
    if seal.get("schema") != "tp.upload_staging.seal.v1" or any(seal.get(k) != v for k, v in marker.items()):
        raise ValueError("upload completion binding changed")
    for name in (BASELINE_MANIFEST_FILENAME, CAPTURE_METADATA_FILENAME, UPLOAD_RECEIPT_FILENAME):
        path = metadata / name
        if (
            path.is_symlink()
            or path.stat().st_size > MAX_RECOVERY_METADATA_BYTES
            or seal.get("metadata_sha256", {}).get(name) != _digest(path)
        ):
            raise ValueError("sealed upload metadata changed")
    baseline = json.loads(_read(metadata / BASELINE_MANIFEST_FILENAME))
    records = baseline.get("records")
    if (
        baseline.get("schema") != BASELINE_MANIFEST_SCHEMA
        or not isinstance(records, list)
        or len(records) > 4096
        or baseline.get("record_count") != len(records)
    ):
        raise ValueError("invalid sealed upload inventory")
    seen = set()
    total = 0
    for record in records:
        relative = normalize_upload_relative_path(record["relative_path"])
        if relative in seen:
            raise ValueError("duplicate sealed upload path")
        seen.add(relative)
        path = root / "input" / relative
        path.resolve().relative_to((root / "input").resolve())
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != record["size_bytes"]
            or _digest(path) != record["sha256"]
        ):
            raise ValueError("sealed upload source changed")
        total += record["size_bytes"]
    data = seal.get("response_data")
    receipt = json.loads(_read(metadata / UPLOAD_RECEIPT_FILENAME))
    expected_data = {key: value for key, value in receipt.items() if key not in {"schema", "batch_root"}}
    if (
        receipt.get("schema") != "tp.orchestrator.upload_staging.v1"
        or receipt.get("batch_root") != str(root)
        or data != expected_data
    ):
        raise ValueError("sealed response differs from its upload receipt")
    if data.get("artifacts") != {
        "baseline_manifest_path": str(metadata / BASELINE_MANIFEST_FILENAME),
        "capture_metadata_path": str(metadata / CAPTURE_METADATA_FILENAME),
        "upload_receipt_path": str(metadata / UPLOAD_RECEIPT_FILENAME),
    }:
        raise ValueError("sealed upload artifact paths changed")
    if (
        not isinstance(data, dict)
        or data.get("batch_id") != reservation.batch_id
        or data.get("input_dir") != str(root / "input")
        or data.get("metadata_dir") != str(metadata)
        or data.get("summary", {}).get("file_count") != len(records)
        or data.get("summary", {}).get("total_bytes") != total
    ):
        raise ValueError("upload response differs from sealed inventory")
    return seal_digest, data


def delete_owned_batch(reservation: UploadReservation, upload_root: Path) -> None:
    root = Path(reservation.batch_root)
    if root.parent != upload_root or root.name != reservation.batch_id or root.is_symlink():
        raise ValueError("upload cleanup root differs from reserved batch")
    if root.exists():
        # Both incomplete and ready batches are ledger-owned. The caller has
        # already fenced every completion and retained reference transaction.
        marker_path = root / MARKER_FILENAME
        if marker_path.exists():
            marker = json.loads(_read(marker_path))
            if (
                marker.get("batch_id") != reservation.batch_id
                or marker.get("request_id") != reservation.request_id
                or marker.get("owner_sha256") != reservation.owner_sha256
            ):
                raise ValueError("upload cleanup ownership changed")
        elif reservation.receipt_sha256 is not None or reservation.response_data is not None:
            raise ValueError("completed upload lost its ownership marker")
        # A crash between mkdir and the initial atomic marker rename leaves
        # no marker. The ledger's never-reused server locator still owns that
        # incomplete directory; no ready job could have referenced it.
        shutil.rmtree(root)


async def cleanup_correlated_batches(
    store: PostgresUploadReceiptStore, root: Path, now: float, retained: set[str]
) -> list[str]:
    """Extend the existing cleanup pass; no competing scheduler or TTL."""
    import asyncio

    from transformation_portal.orchestrator.photographic_admission import drain_owned

    removed = []
    for candidate in await store.cleanup_candidates(now):
        if Path(candidate.batch_root).parent != root:
            continue
        lock = UploadBatchLock(root, candidate.batch_id)
        try:
            if not lock.acquire():
                continue
            claimed = await store.claim_deletion(candidate, now, retained)
            if claimed is None:
                continue

            async def finish(expected: UploadReservation = claimed) -> None:
                await asyncio.to_thread(delete_owned_batch, expected, root)
                await store.finish_deletion(expected)

            try:
                await drain_owned(finish())
                removed.append(candidate.batch_id)
            except (OSError, ValueError):
                # Leave deleting fenced; the same cleanup loop retries it.
                continue
        finally:
            lock.close()
    return removed
