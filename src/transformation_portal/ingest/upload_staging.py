"""Deterministic staged upload helpers for the portal input_dir workflow."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable, Sequence

from tp.phase4 import (
    extract_capture_metadata_records,
    load_capture_metadata_config,
    normalize_relative_path,
    write_capture_metadata_artifact,
)
from transformation_portal.ingest.canonical_json import canonicalize_json

BASELINE_MANIFEST_SCHEMA = "tp.meta.baseline_manifest.v1"
UPLOAD_STAGING_SCHEMA = "tp.orchestrator.upload_staging.v1"
BASELINE_MANIFEST_FILENAME = "baseline_manifest.tp.meta.baseline_manifest.v1.json"
CAPTURE_METADATA_FILENAME = "capture_metadata.tp.meta.capture.v1.json"
UPLOAD_RECEIPT_FILENAME = "upload_receipt.tp.orchestrator.upload_staging.v1.json"
DEFAULT_CAPTURE_METADATA_CONFIG_PATH = Path(__file__).resolve().parents[3] / "tools" / "capture_metadata_config.json"
DEFAULT_CAPTURE_METADATA_SCHEMA_PATH = Path(__file__).resolve().parents[3] / "schemas" / "phase4" / "metadata.schema.json"
STREAM_CHUNK_BYTES = 1024 * 1024
# Partial uploads are written next to their destination under a short fixed-length
# name. Embedding the destination name made the temp name longer than the final
# one, so names the filesystem accepts (241-255 bytes) could not be staged.
_PARTIAL_UPLOAD_PREFIX = ".upload-"
_PARTIAL_UPLOAD_SUFFIX = ".part"
# errno values with which the staging filesystem refuses a client-chosen relative
# path. Every other OSError (no space, permissions, I/O) is a server-side fault
# and is left to propagate unchanged.
_PATH_CONFLICT_ERRNOS = frozenset({errno.EEXIST, errno.EISDIR, errno.ENOTDIR})
_PATH_UNREPRESENTABLE_ERRNOS = frozenset({errno.EILSEQ, errno.EINVAL})
# Shape of the ids produced by build_batch_id().
_BATCH_ID_PATTERN = re.compile(r"upload_\d+_[0-9a-f]{8}")
# A batch without a receipt is either still being staged or was interrupted (the
# process died mid-request). Staging runs inside one HTTP request, so a
# receipt-less batch this old can no longer be in flight. Cleanup waits for the
# longer of this grace period and the configured TTL before reaping one, which
# keeps a short TTL from deleting a batch that is still being written.
INTERRUPTED_BATCH_GRACE_SECONDS = 60 * 60
_COMPOUND_EXTENSIONS = (
    ".tar.gz",
    ".tar.bz2",
    ".tar.xz",
    ".json.gz",
    ".jsonl.gz",
    ".csv.gz",
)
_IMAGE_EXTENSIONS = {
    ".arw",
    ".cr2",
    ".dng",
    ".heic",
    ".heif",
    ".jpeg",
    ".jpg",
    ".nef",
    ".orf",
    ".png",
    ".rw2",
    ".tif",
    ".tiff",
    ".webp",
}
_ARCHIVE_EXTENSIONS = {
    ".7z",
    ".bz2",
    ".gz",
    ".rar",
    ".tar",
    ".tar.bz2",
    ".tar.gz",
    ".tar.xz",
    ".tgz",
    ".xz",
    ".zip",
}
_DOCUMENT_EXTENSIONS = {".pdf"}
_TEXT_EXTENSIONS = {
    ".csv",
    ".json",
    ".json.gz",
    ".jsonl",
    ".jsonl.gz",
    ".md",
    ".toml",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
_VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
_AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".wav"}
_STABLE_MIME_TYPES = {
    ".aac": "audio/aac",
    ".avi": "video/x-msvideo",
    ".bz2": "application/x-bzip2",
    ".csv": "text/csv",
    ".dng": "image/x-adobe-dng",
    ".flac": "audio/flac",
    ".gz": "application/gzip",
    ".heic": "image/heic",
    ".heif": "image/heif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".json": "application/json",
    ".json.gz": "application/gzip",
    ".jsonl": "application/x-ndjson",
    ".jsonl.gz": "application/gzip",
    ".m4a": "audio/mp4",
    ".m4v": "video/x-m4v",
    ".md": "text/markdown",
    ".mkv": "video/x-matroska",
    ".mov": "video/quicktime",
    ".mp3": "audio/mpeg",
    ".mp4": "video/mp4",
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".rar": "application/vnd.rar",
    ".tar": "application/x-tar",
    ".tar.bz2": "application/x-bzip2",
    ".tar.gz": "application/gzip",
    ".tar.xz": "application/x-xz",
    ".tgz": "application/gzip",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".toml": "application/toml",
    ".txt": "text/plain",
    ".wav": "audio/wav",
    ".webm": "video/webm",
    ".webp": "image/webp",
    ".xml": "application/xml",
    ".xz": "application/x-xz",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
    ".zip": "application/zip",
}


@dataclass(frozen=True)
class IncomingUpload:
    filename: str
    stream: BinaryIO
    content_type: str = ""


@dataclass(frozen=True)
class StagedUploadResult:
    batch_id: str
    batch_root: Path
    input_dir: Path
    portal_dir: Path
    baseline_manifest_path: Path
    capture_metadata_path: Path
    upload_receipt_path: Path
    file_count: int
    total_bytes: int
    capture_metadata_record_count: int
    capture_metadata_enabled: bool
    received_at_epoch_seconds: float
    top_level_roots: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_response_data(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "input_dir": str(self.input_dir),
            "metadata_dir": str(self.portal_dir),
            "artifacts": {
                "baseline_manifest_path": str(self.baseline_manifest_path),
                "capture_metadata_path": str(self.capture_metadata_path),
                "upload_receipt_path": str(self.upload_receipt_path),
            },
            "received_at_epoch_seconds": self.received_at_epoch_seconds,
            "summary": {
                "file_count": self.file_count,
                "total_bytes": self.total_bytes,
                "capture_metadata_enabled": self.capture_metadata_enabled,
                "capture_metadata_record_count": self.capture_metadata_record_count,
                "top_level_roots": list(self.top_level_roots),
                "warnings": list(self.warnings),
            },
        }


class UploadStagingError(ValueError):
    """Raised when a staged upload payload is invalid."""

    def __init__(
        self,
        reason: str,
        message: str,
        *,
        field: str = "files",
        status_code: int = 400,
    ) -> None:
        super().__init__(message)
        self.reason = str(reason)
        self.message = str(message)
        self.field = str(field)
        self.status_code = int(status_code)


def build_batch_id(now: float | None = None) -> str:
    timestamp = int(now or 0)
    return f"upload_{timestamp}_{uuid.uuid4().hex[:8]}"


def parse_client_manifest_relative_paths(
    raw_manifest: Any,
    *,
    expected_count: int,
) -> list[str] | None:
    if raw_manifest is None:
        return None
    if isinstance(raw_manifest, bytes):
        try:
            text = raw_manifest.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise UploadStagingError(
                "invalid_client_manifest",
                "client_manifest must be valid UTF-8 JSON",
                field="client_manifest",
            ) from exc
    else:
        text = str(raw_manifest)
    if not text.strip():
        return None

    try:
        payload = json.loads(text)
    except (ValueError, RecursionError) as exc:
        # json.JSONDecodeError is a ValueError, and so is the interpreter's limit
        # on integer digits; a deeply nested document exhausts the decoder's
        # recursion budget instead. All three are client errors, not crashes.
        raise UploadStagingError(
            "invalid_client_manifest",
            "client_manifest must be valid JSON",
            field="client_manifest",
        ) from exc

    if isinstance(payload, dict):
        entries = payload.get("files")
    else:
        entries = payload
    if not isinstance(entries, list):
        raise UploadStagingError(
            "invalid_client_manifest",
            "client_manifest must provide a files array",
            field="client_manifest",
        )
    if len(entries) != expected_count:
        raise UploadStagingError(
            "client_manifest_count_mismatch",
            "client_manifest file count does not match uploaded files",
            field="client_manifest",
        )

    relative_paths: list[str] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise UploadStagingError(
                "invalid_client_manifest",
                f"client_manifest entry {index} must be an object",
                field="client_manifest",
            )
        raw_path = entry.get("relative_path", entry.get("relativePath"))
        if raw_path is None:
            raise UploadStagingError(
                "invalid_client_manifest",
                f"client_manifest entry {index} is missing relative_path",
                field="client_manifest",
            )
        relative_paths.append(str(raw_path))
    return relative_paths


def normalize_upload_relative_path(raw_value: str) -> str:
    value = str(raw_value or "")
    if not value.strip():
        raise UploadStagingError("relative_path_required", "relative path is required")
    if "\x00" in value:
        raise UploadStagingError("invalid_relative_path", "relative path contains a null byte")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        # Lone surrogates (reachable through JSON escapes in client_manifest)
        # cannot be written to the UTF-8 baseline manifest.
        raise UploadStagingError("invalid_relative_path", "relative path is not valid Unicode") from exc
    try:
        normalized = normalize_relative_path(value)
    except Exception as exc:
        raise UploadStagingError("invalid_relative_path", "relative path is invalid") from exc
    normalized_path = PurePosixPath(normalized)
    if str(normalized_path) in {".", ""}:
        raise UploadStagingError("invalid_relative_path", "relative path is invalid")
    return normalized_path.as_posix()


def _deduplicate_warnings(warnings: Iterable[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for warning in warnings:
        normalized = str(warning or "").strip()
        if not normalized or normalized in seen:
            continue
        deduped.append(normalized)
        seen.add(normalized)
    return sorted(deduped)


def _normalized_extension(relative_path: str) -> str:
    lowered = str(relative_path or "").strip().lower()
    for suffix in _COMPOUND_EXTENSIONS:
        if lowered.endswith(suffix):
            return suffix
    return Path(lowered).suffix.lower()


def _stable_mime_type_for_extension(extension: str) -> str:
    return _STABLE_MIME_TYPES.get(str(extension or "").strip().lower(), "")


def _media_kind_for_path(relative_path: str) -> str:
    extension = _normalized_extension(relative_path)
    if extension in _IMAGE_EXTENSIONS:
        return "image"
    if extension in _DOCUMENT_EXTENSIONS:
        return "document"
    if extension in _ARCHIVE_EXTENSIONS:
        return "archive"
    if extension in _VIDEO_EXTENSIONS:
        return "video"
    if extension in _AUDIO_EXTENSIONS:
        return "audio"
    if extension in _TEXT_EXTENSIONS:
        return "text"
    return "file"


def _write_bytes_atomic(output_path: Path, payload: bytes) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = tempfile.NamedTemporaryFile(
        mode="wb",
        dir=str(output_path.parent),
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temp_path = Path(temp_file.name)
    try:
        with temp_file:
            temp_file.write(payload)
        os.replace(temp_path, output_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _reject_conflicting_relative_paths(relative_paths: Sequence[str]) -> None:
    """Reject payloads in which one upload is a directory prefix of another.

    ``a`` and ``a/b`` cannot both be staged: ``a`` would have to be a file and a
    directory at once. Left to the filesystem this surfaces as a raw
    ``FileExistsError`` or ``IsADirectoryError``, depending on the order.
    """
    directories: set[str] = set()
    for relative_path in relative_paths:
        directories.update(parent.as_posix() for parent in PurePosixPath(relative_path).parents)
    for relative_path in relative_paths:
        if relative_path in directories:
            raise UploadStagingError(
                "conflicting_relative_path",
                "relative path is used as both a file and a directory in staged upload payload",
                field="files",
            )


def _relative_path_error(exc: BaseException) -> UploadStagingError | None:
    """Translate a filesystem refusal caused by the upload's own relative path."""
    if isinstance(exc, RecursionError):
        return UploadStagingError("invalid_relative_path", "relative path is nested too deeply")
    if not isinstance(exc, OSError):
        return None
    if exc.errno == errno.ENAMETOOLONG:
        return UploadStagingError("invalid_relative_path", "relative path is too long for the staging filesystem")
    if exc.errno in _PATH_CONFLICT_ERRNOS:
        return UploadStagingError(
            "conflicting_relative_path",
            "relative path is used as both a file and a directory in staged upload payload",
        )
    if exc.errno in _PATH_UNREPRESENTABLE_ERRNOS:
        return UploadStagingError("invalid_relative_path", "relative path is not valid on the staging filesystem")
    return None


def _stream_upload_to_path(source: BinaryIO, destination: Path) -> tuple[str, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_file = tempfile.NamedTemporaryFile(
        mode="wb",
        dir=str(destination.parent),
        prefix=_PARTIAL_UPLOAD_PREFIX,
        suffix=_PARTIAL_UPLOAD_SUFFIX,
        delete=False,
    )
    temp_path = Path(temp_file.name)
    hasher = hashlib.sha256()
    size_bytes = 0
    try:
        with temp_file:
            while True:
                chunk = source.read(STREAM_CHUNK_BYTES)
                if not chunk:
                    break
                if not isinstance(chunk, (bytes, bytearray)):
                    raise UploadStagingError("invalid_upload_stream", "upload stream yielded a non-bytes chunk")
                temp_file.write(chunk)
                hasher.update(chunk)
                size_bytes += len(chunk)
        os.replace(temp_path, destination)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return hasher.hexdigest(), size_bytes


def _build_baseline_record(relative_path: str, _file_path: Path, sha256: str, size_bytes: int) -> dict[str, Any]:
    extension = _normalized_extension(relative_path)
    mime_type = _stable_mime_type_for_extension(extension)
    media_kind = _media_kind_for_path(relative_path)
    warnings: list[str] = []
    if not mime_type:
        warnings.append("mime_type_unresolved")

    record: dict[str, Any] = {
        "relative_path": relative_path,
        "sha256": sha256,
        "size_bytes": size_bytes,
        "extension": extension,
        "mime_type": mime_type,
        "media_kind": media_kind,
        "warnings": [],
    }

    record["warnings"] = _deduplicate_warnings(warnings)
    return record


def _build_receipt_payload(
    *,
    batch_id: str,
    batch_root: Path,
    input_dir: Path,
    portal_dir: Path,
    baseline_manifest_path: Path,
    capture_metadata_path: Path,
    upload_receipt_path: Path,
    file_count: int,
    total_bytes: int,
    capture_metadata_enabled: bool,
    capture_metadata_record_count: int,
    top_level_roots: Sequence[str],
    warnings: Sequence[str],
    received_at_epoch_seconds: float,
) -> dict[str, Any]:
    return {
        "schema": UPLOAD_STAGING_SCHEMA,
        "batch_id": batch_id,
        "received_at_epoch_seconds": received_at_epoch_seconds,
        "batch_root": str(batch_root),
        "input_dir": str(input_dir),
        "metadata_dir": str(portal_dir),
        "artifacts": {
            "baseline_manifest_path": str(baseline_manifest_path),
            "capture_metadata_path": str(capture_metadata_path),
            "upload_receipt_path": str(upload_receipt_path),
        },
        "summary": {
            "file_count": file_count,
            "total_bytes": total_bytes,
            "capture_metadata_enabled": capture_metadata_enabled,
            "capture_metadata_record_count": capture_metadata_record_count,
            "top_level_roots": list(top_level_roots),
            "warnings": _deduplicate_warnings(warnings),
        },
    }


def _resolve_relative_paths(
    uploads: Sequence[IncomingUpload],
    client_manifest_paths: Sequence[str] | None,
) -> list[str]:
    resolved: list[str] = []
    for index, upload in enumerate(uploads):
        raw_filename = str(upload.filename or "")
        manifest_path = client_manifest_paths[index] if client_manifest_paths is not None else ""
        if client_manifest_paths is not None and raw_filename:
            normalized_filename = normalize_upload_relative_path(raw_filename)
            normalized_manifest = normalize_upload_relative_path(manifest_path)
            if normalized_filename != normalized_manifest:
                raise UploadStagingError(
                    "client_manifest_mismatch",
                    "client_manifest entry does not match uploaded filename",
                    field="client_manifest",
                )
        raw_path = manifest_path or raw_filename
        resolved.append(normalize_upload_relative_path(raw_path))
    return resolved


def stage_upload_batch(
    *,
    upload_root: Path,
    uploads: Sequence[IncomingUpload],
    client_manifest_paths: Sequence[str] | None = None,
    capture_metadata_enabled: bool = False,
    capture_metadata_config_path: Path = DEFAULT_CAPTURE_METADATA_CONFIG_PATH,
    capture_metadata_schema_path: Path = DEFAULT_CAPTURE_METADATA_SCHEMA_PATH,
    now: float,
    reserved_batch_id: str | None = None,
    cleanup_on_error: bool = True,
) -> StagedUploadResult:
    if not uploads:
        raise UploadStagingError("files_required", "at least one upload file is required")

    upload_root.mkdir(parents=True, exist_ok=True)
    batch_id = reserved_batch_id or build_batch_id(now)
    if reserved_batch_id is not None and (
        not reserved_batch_id.startswith("upload_") or "/" in reserved_batch_id or "\\" in reserved_batch_id
    ):
        raise UploadStagingError("invalid_batch_id", "invalid server batch ID")
    batch_root = upload_root / batch_id
    input_dir = batch_root / "input"
    portal_dir = batch_root / "_portal"
    tmp_dir = batch_root / "tmp"
    baseline_manifest_path = portal_dir / BASELINE_MANIFEST_FILENAME
    capture_metadata_path = portal_dir / CAPTURE_METADATA_FILENAME
    upload_receipt_path = portal_dir / UPLOAD_RECEIPT_FILENAME
    relative_paths = _resolve_relative_paths(uploads, client_manifest_paths)
    _reject_conflicting_relative_paths(relative_paths)
    seen_paths: set[str] = set()
    warnings: list[str] = []
    total_bytes = 0
    records: list[dict[str, Any]] = []
    capture_metadata_record_count = 0
    top_level_roots: tuple[str, ...] = ()

    try:
        input_dir.mkdir(parents=True, exist_ok=True)
        portal_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        for upload, relative_path in zip(uploads, relative_paths):
            if relative_path in seen_paths:
                raise UploadStagingError(
                    "duplicate_relative_path",
                    "duplicate relative path in staged upload payload",
                    field="files",
                )
            seen_paths.add(relative_path)

            destination_path = input_dir / PurePosixPath(relative_path)
            try:
                destination_path.resolve().relative_to(input_dir.resolve())
            except ValueError as exc:
                raise UploadStagingError(
                    "invalid_relative_path",
                    "relative path escapes the staging root",
                    field="files",
                ) from exc

            if os.path.lexists(destination_path):
                # The batch directory is new, so an existing destination means the
                # filesystem folds two distinct relative paths into one name
                # (case-insensitive or Unicode-normalising volumes). Replacing it
                # would silently drop the earlier upload.
                raise UploadStagingError(
                    "duplicate_relative_path",
                    "relative path collides with another upload on the staging filesystem",
                    field="files",
                )

            try:
                upload.stream.seek(0)
            except (AttributeError, OSError):
                pass
            try:
                sha256, size_bytes = _stream_upload_to_path(upload.stream, destination_path)
            except (OSError, RecursionError) as exc:
                path_error = _relative_path_error(exc)
                if path_error is None:
                    raise
                raise path_error from exc
            total_bytes += size_bytes
            records.append(_build_baseline_record(relative_path, destination_path, sha256, size_bytes))

        records.sort(key=lambda item: str(item.get("relative_path") or ""))
        top_level_roots = tuple(
            sorted(
                {
                    PurePosixPath(str(item.get("relative_path") or "")).parts[0]
                    for item in records
                    if PurePosixPath(str(item.get("relative_path") or "")).parts
                }
            )
        )
        baseline_payload = {
            "schema": BASELINE_MANIFEST_SCHEMA,
            "record_count": len(records),
            "records": records,
        }
        _write_bytes_atomic(baseline_manifest_path, canonicalize_json(baseline_payload))

        if capture_metadata_enabled:
            try:
                capture_config = load_capture_metadata_config(capture_metadata_config_path)
                capture_records = extract_capture_metadata_records(
                    input_root=input_dir,
                    config=capture_config,
                    strict=False,
                    schema_path=capture_metadata_schema_path,
                )
                capture_metadata_record_count = len(capture_records)
                write_capture_metadata_artifact(capture_records, out_path=capture_metadata_path)
            except Exception:
                warnings.append("capture_metadata_extraction_failed")
                write_capture_metadata_artifact([], out_path=capture_metadata_path)
        else:
            write_capture_metadata_artifact([], out_path=capture_metadata_path)

        receipt_payload = _build_receipt_payload(
            batch_id=batch_id,
            batch_root=batch_root,
            input_dir=input_dir,
            portal_dir=portal_dir,
            baseline_manifest_path=baseline_manifest_path,
            capture_metadata_path=capture_metadata_path,
            upload_receipt_path=upload_receipt_path,
            file_count=len(records),
            total_bytes=total_bytes,
            capture_metadata_enabled=capture_metadata_enabled,
            capture_metadata_record_count=capture_metadata_record_count,
            top_level_roots=top_level_roots,
            warnings=warnings,
            received_at_epoch_seconds=now,
        )
        _write_bytes_atomic(upload_receipt_path, canonicalize_json(receipt_payload))
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return StagedUploadResult(
            batch_id=batch_id,
            batch_root=batch_root,
            input_dir=input_dir,
            portal_dir=portal_dir,
            baseline_manifest_path=baseline_manifest_path,
            capture_metadata_path=capture_metadata_path,
            upload_receipt_path=upload_receipt_path,
            file_count=len(records),
            total_bytes=total_bytes,
            capture_metadata_record_count=capture_metadata_record_count,
            capture_metadata_enabled=capture_metadata_enabled,
            received_at_epoch_seconds=now,
            top_level_roots=top_level_roots,
            warnings=tuple(_deduplicate_warnings(warnings)),
        )
    except Exception:
        if cleanup_on_error:
            shutil.rmtree(batch_root, ignore_errors=True)
        raise


def cleanup_expired_batches(
    upload_root: Path,
    *,
    now: float,
    ttl_seconds: float,
    retained_input_dirs: Iterable[str | Path],
) -> list[str]:
    if ttl_seconds <= 0:
        return []
    try:
        if not upload_root.exists():
            return []
    except OSError:
        return []

    # A job may read the whole staged input, any directory below it (the receipt
    # advertises the top-level roots) or a directory that contains the batch (for
    # example the upload root itself). A batch is therefore retained while any
    # retained path overlaps it in either direction.
    retained = {Path(os.path.realpath(str(path))) for path in retained_input_dirs if str(path).strip()}
    removed: list[str] = []
    for batch_dir in upload_root.iterdir():
        try:
            if not batch_dir.is_dir():
                continue
            if _is_managed_staged_batch_dir(batch_dir):
                min_age_seconds = ttl_seconds
            elif _is_interrupted_staged_batch_dir(batch_dir):
                min_age_seconds = max(ttl_seconds, INTERRUPTED_BATCH_GRACE_SECONDS)
            else:
                continue
            batch_root_real = Path(os.path.realpath(batch_dir))
            if any(path.is_relative_to(batch_root_real) or batch_root_real.is_relative_to(path) for path in retained):
                continue
            age_seconds = now - batch_dir.stat().st_mtime
            if age_seconds < min_age_seconds:
                continue
            shutil.rmtree(batch_dir, ignore_errors=False)
            removed.append(batch_dir.name)
        except OSError:
            continue
    return sorted(removed)


def _has_upload_ledger_identity(batch_dir: Path) -> bool:
    """Return True when a batch is owned by the upload receipt ledger.

    Legacy TTL cleanup (including the reaping of interrupted batches) has no
    authority over such a batch; the ledger fences and deletes it instead.
    """
    try:
        (batch_dir / ".upload-request.v1.json").lstat()
    except FileNotFoundError:
        pass
    else:
        # Ledger batches require an atomic ownership/reference claim first.
        return True
    # Permanent external lock identities outlive markers and partial deletion.
    # A linked/invalid lock directory or any lock entry (even a dangling link)
    # is not legacy-cleanup authority. Other filesystem errors fail closed in
    # the caller, rather than turning an unreadable identity into absence.
    lock_directory = batch_dir.parent / ".upload-locks"
    try:
        lock_info = lock_directory.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISDIR(lock_info.st_mode):
        return True
    try:
        (lock_directory / (str(batch_dir.name or "").strip() + ".lock")).lstat()
    except FileNotFoundError:
        return False
    return True


def _is_managed_staged_batch_dir(batch_dir: Path) -> bool:
    name = str(batch_dir.name or "").strip()
    if not name.startswith("upload_"):
        return False
    if _has_upload_ledger_identity(batch_dir):
        return False
    input_dir = batch_dir / "input"
    portal_dir = batch_dir / "_portal"
    if not input_dir.is_dir() or not portal_dir.is_dir():
        return False
    receipt_path = portal_dir / UPLOAD_RECEIPT_FILENAME
    return receipt_path.is_file()


def _is_interrupted_staged_batch_dir(batch_dir: Path) -> bool:
    """Return True for a batch that staging created but never finished.

    The receipt is written last, and a failed stage removes its batch, so a
    receipt-less batch is left behind only when the process is killed mid-stage.
    Only directories that carry a generated batch id and the managed layout
    qualify, and never one the upload receipt ledger owns (marker or lock
    identity); anything else in the upload root is never touched.
    """
    if _BATCH_ID_PATTERN.fullmatch(str(batch_dir.name or "")) is None:
        return False
    if _has_upload_ledger_identity(batch_dir):
        return False
    portal_dir = batch_dir / "_portal"
    if not (batch_dir / "input").is_dir() or not portal_dir.is_dir():
        return False
    return not os.path.lexists(portal_dir / UPLOAD_RECEIPT_FILENAME)
