"""Filesystem ownership and exact completion seals for upload correlation."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import stat
from bisect import bisect_left
from pathlib import Path
from typing import Any

from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.ingest.upload_staging import (
    BASELINE_MANIFEST_FILENAME,
    BASELINE_MANIFEST_SCHEMA,
    CAPTURE_METADATA_FILENAME,
    UPLOAD_RECEIPT_FILENAME,
    StagedUploadResult,
    UploadStagingError,
    _write_bytes_atomic,
    normalize_upload_relative_path,
)
from transformation_portal.orchestrator.upload_receipts import PostgresUploadReceiptStore, UploadReservation

MARKER_FILENAME = ".upload-request.v1.json"
SEAL_FILENAME = "upload_completion.tp.upload_staging.seal.v1.json"
MAX_RECOVERY_METADATA_BYTES = 4 * 1024 * 1024
_BATCH_ID = re.compile(r"upload_[A-Za-z0-9_-]{1,56}")
LOGGER = logging.getLogger(__name__)


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
        name: hashlib.sha256(_read(metadata / name)).hexdigest()
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
        raise UploadStagingError(
            "upload_recovery_metadata_limit", "staged upload metadata exceeds recovery capacity", status_code=413
        )
    _write_bytes_atomic(metadata / SEAL_FILENAME, raw)
    return hashlib.sha256(raw).hexdigest()


def _read(path: Path, *, verified: dict[Path, tuple[int, ...]] | None = None) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("upload recovery metadata must be an independent regular file")
        if before.st_size > MAX_RECOVERY_METADATA_BYTES:
            raise UploadStagingError(
                "upload_recovery_metadata_limit", "staged upload metadata exceeds recovery capacity", status_code=413
            )
        raw = stream.read(MAX_RECOVERY_METADATA_BYTES + 1)
        after = os.fstat(stream.fileno())
    if (
        len(raw) > MAX_RECOVERY_METADATA_BYTES
        or _identity(before) != _identity(after)
        or _identity(after) != _identity(path.lstat())
    ):
        raise ValueError("upload recovery metadata changed during verification")
    if verified is not None:
        verified[path] = _identity(after)
    return raw


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _source_digest(directory_fd: int, name: str, size: int) -> tuple[str, tuple[int, ...]]:
    hasher = hashlib.sha256()
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size != size:
            raise ValueError("sealed upload source changed")
        remaining = size
        while chunk := stream.read(min(1024 * 1024, remaining + 1)):
            if len(chunk) > remaining:
                raise ValueError("sealed upload source grew during verification")
            hasher.update(chunk)
            remaining -= len(chunk)
        after = os.fstat(stream.fileno())
        if remaining:
            raise ValueError("sealed upload source changed")
    current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if _identity(before) != _identity(after) or _identity(after) != _identity(current):
        raise ValueError("sealed upload source changed during verification")
    return hasher.hexdigest(), _identity(after)


def _exact_directory(path: Path, directories: set[str], files: set[str]) -> tuple[int, ...]:
    """Completion never includes temporary, unexpected or linked entries."""
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        before = _identity(os.fstat(descriptor))
        with os.scandir(descriptor) as entries:
            seen = set()
            for entry in entries:
                info = entry.stat(follow_symlinks=False)
                expected = stat.S_ISDIR(info.st_mode) if entry.name in directories else stat.S_ISREG(info.st_mode)
                if entry.name not in directories | files or not expected or (entry.name in files and info.st_nlink != 1):
                    raise ValueError("unexpected entry in sealed upload directory")
                seen.add(entry.name)
        if _identity(os.fstat(descriptor)) != before or _identity(path.lstat()) != before:
            raise ValueError("sealed upload directory changed during verification")
    finally:
        os.close(descriptor)
    if seen != directories | files:
        raise ValueError("sealed upload directory is incomplete")
    return before


def _verify_input_inventory(root: Path, records: list[dict[str, Any]], *, verified: dict[Path, tuple[int, ...]]) -> int:
    expected: dict[str, dict[str, Any]] = {}
    total = 0
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("relative_path"), str):
            raise ValueError("invalid sealed upload record")
        relative = normalize_upload_relative_path(record["relative_path"])
        size, digest = record.get("size_bytes"), record.get("sha256")
        if (
            relative in expected
            or type(size) is not int
            or size < 0
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise ValueError("invalid or duplicate sealed upload path")
        expected[relative] = record
        total += size
    ordered = sorted(expected)
    seen: set[str] = set()

    def visit(directory_fd: int, parent: str) -> None:
        before = _identity(os.fstat(directory_fd))
        # Stream entries rather than materializing an untrusted directory.
        # Only ancestors of sealed files may be visited; no expanded prefix
        # collection can multiply the bounded manifest's memory footprint.
        with os.scandir(directory_fd) as entries:
            for entry in entries:
                relative = parent + entry.name
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    prefix = relative + "/"
                    index = bisect_left(ordered, prefix)
                    if index == len(ordered) or not ordered[index].startswith(prefix):
                        raise ValueError("unexpected directory in sealed upload input")
                    child = os.open(entry.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory_fd)
                    try:
                        visit(child, prefix)
                    finally:
                        os.close(child)
                else:
                    record = expected.get(relative)
                    if record is None or not stat.S_ISREG(info.st_mode):
                        raise ValueError("unexpected or linked file in sealed upload input")
                    digest, identity = _source_digest(directory_fd, entry.name, record["size_bytes"])
                    if digest != record["sha256"]:
                        raise ValueError("sealed upload source changed")
                    verified[root / relative] = identity
                    seen.add(relative)
        if _identity(os.fstat(directory_fd)) != before:
            raise ValueError("sealed upload input directory changed during verification")
        verified[root / parent] = before

    directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        before = _identity(os.fstat(directory_fd))
        visit(directory_fd, "")
        if _identity(root.lstat()) != before:
            raise ValueError("sealed upload input directory changed during verification")
    finally:
        os.close(directory_fd)
    if seen != set(expected):
        raise ValueError("sealed upload input inventory is incomplete")
    return total


def verify_completion(reservation: UploadReservation, *, max_files: int | None = None) -> tuple[str, dict[str, Any]]:
    """Verify retained bytes and ending identities, without claiming an atomic snapshot.

    A receipt alone cannot authorize ready. External changes after this call
    remain subject to the existing admission and worker revalidation.
    """
    root = Path(reservation.batch_root)
    if root.is_symlink() or root.name != reservation.batch_id:
        raise ValueError("reserved upload directory changed")
    root_identity = _exact_directory(root, {"input", "_portal"}, {MARKER_FILENAME})
    verified = {root: root_identity}
    marker = json.loads(_read(root / MARKER_FILENAME, verified=verified))
    if marker != _binding(reservation):
        raise ValueError("upload request binding changed")
    metadata = root / "_portal"
    if (root / "input").is_symlink() or metadata.is_symlink():
        raise ValueError("upload metadata directory changed")
    metadata_identity = _exact_directory(
        metadata, set(), {SEAL_FILENAME, BASELINE_MANIFEST_FILENAME, CAPTURE_METADATA_FILENAME, UPLOAD_RECEIPT_FILENAME}
    )
    verified[metadata] = metadata_identity
    seal_raw = _read(metadata / SEAL_FILENAME, verified=verified)
    seal_digest = hashlib.sha256(seal_raw).hexdigest()
    if reservation.receipt_sha256 is not None and reservation.receipt_sha256 != seal_digest:
        raise ValueError("upload completion differs from its durable seal digest")
    seal = json.loads(seal_raw)
    if (
        not isinstance(seal, dict)
        or seal.get("schema") != "tp.upload_staging.seal.v1"
        or any(seal.get(k) != v for k, v in marker.items())
    ):
        raise ValueError("upload completion binding changed")
    metadata_digests = seal.get("metadata_sha256")
    if not isinstance(metadata_digests, dict):
        raise ValueError("invalid sealed upload metadata identities")
    retained_metadata = {}
    for name in (BASELINE_MANIFEST_FILENAME, CAPTURE_METADATA_FILENAME, UPLOAD_RECEIPT_FILENAME):
        raw = _read(metadata / name, verified=verified)
        if metadata_digests.get(name) != hashlib.sha256(raw).hexdigest():
            raise ValueError("sealed upload metadata changed")
        if name != CAPTURE_METADATA_FILENAME:
            retained_metadata[name] = raw
    baseline = json.loads(retained_metadata[BASELINE_MANIFEST_FILENAME])
    if not isinstance(baseline, dict):
        raise ValueError("invalid sealed upload inventory")
    records = baseline.get("records")
    data = seal.get("response_data")
    summary = data.get("summary") if isinstance(data, dict) else None
    count = summary.get("file_count") if isinstance(summary, dict) else None
    if max_files is not None and (type(max_files) is not int or max_files < 1):
        raise ValueError("upload verification file limit is invalid")
    if (
        baseline.get("schema") != BASELINE_MANIFEST_SCHEMA
        or not isinstance(records, list)
        or type(count) is not int
        or count < 1
        or len(records) != count
        or (max_files is not None and count > max_files)
        or type(baseline.get("record_count")) is not int
        or baseline.get("record_count") != len(records)
    ):
        raise ValueError("invalid sealed upload inventory")
    total = _verify_input_inventory(root / "input", records, verified=verified)
    receipt = json.loads(retained_metadata[UPLOAD_RECEIPT_FILENAME])
    if not isinstance(receipt, dict):
        raise ValueError("invalid sealed upload receipt")
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
    # Keep each identity from its actual descriptor read through the end of the
    # whole pass. A later file must not hide a rewrite of an earlier source or
    # metadata file, nor a replaced ancestor that redirects an already-read file.
    for path, identity in verified.items():
        if _identity(path.lstat()) != identity:
            raise ValueError("sealed upload file or directory changed during verification")
    return seal_digest, data


def delete_owned_batch(reservation: UploadReservation, upload_root: Path) -> None:
    root = Path(reservation.batch_root)
    if root.parent != upload_root or root.name != reservation.batch_id or root.is_symlink():
        raise ValueError("upload cleanup root differs from reserved batch")
    if root.exists():
        # Both incomplete and ready batches are ledger-owned. The caller has
        # already fenced every completion and retained reference transaction.
        marker_path = root / MARKER_FILENAME
        if marker_path.exists() or marker_path.is_symlink():
            marker = json.loads(_read(marker_path))
            if not isinstance(marker, dict) or any(
                marker.get(field) != getattr(reservation, field)
                for field in ("batch_id", "tenant_id", "request_id", "owner_sha256", "batch_root")
            ):
                raise ValueError("upload cleanup ownership changed")
        elif reservation.state != "deleting":
            raise ValueError("markerless upload deletion requires fenced ownership")
        # The caller holds the stable lock and a deleting ledger generation.
        # An interrupted rmtree may already have removed the marker; its
        # never-reused locator still belongs to that durable deletion claim.
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
        lock = None
        try:
            try:
                lock = UploadBatchLock(root, candidate.batch_id)
                acquired = lock.acquire()
            except (OSError, ValueError) as error:
                LOGGER.warning("Upload cleanup lock unavailable for batch %s (%s)", candidate.batch_id, type(error).__name__)
                continue
            if not acquired:
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
            if lock is not None:
                lock.close()
    return removed
