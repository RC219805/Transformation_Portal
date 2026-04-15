"""Deterministic staged upload helpers for the portal input_dir workflow."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable, Sequence

try:
    from PIL import Image, UnidentifiedImageError
except Exception:  # pragma: no cover - import-safe fallback for optional runtime deps
    Image = None  # type: ignore[assignment]
    UnidentifiedImageError = OSError  # type: ignore[assignment]

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - import-safe fallback for optional runtime deps
    PdfReader = None  # type: ignore[assignment]

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
            "summary": {
                "file_count": self.file_count,
                "total_bytes": self.total_bytes,
                "capture_metadata_enabled": self.capture_metadata_enabled,
                "capture_metadata_record_count": self.capture_metadata_record_count,
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
    except json.JSONDecodeError as exc:
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


def _media_kind_for_path(relative_path: str, mime_type: str) -> str:
    extension = _normalized_extension(relative_path)
    if extension in _IMAGE_EXTENSIONS or mime_type.startswith("image/"):
        return "image"
    if extension in _DOCUMENT_EXTENSIONS or mime_type == "application/pdf":
        return "document"
    if extension in _ARCHIVE_EXTENSIONS:
        return "archive"
    if extension in _VIDEO_EXTENSIONS or mime_type.startswith("video/"):
        return "video"
    if extension in _AUDIO_EXTENSIONS or mime_type.startswith("audio/"):
        return "audio"
    if extension in _TEXT_EXTENSIONS or mime_type.startswith("text/"):
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


def _stream_upload_to_path(source: BinaryIO, destination: Path) -> tuple[str, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_file = tempfile.NamedTemporaryFile(
        mode="wb",
        dir=str(destination.parent),
        prefix=f".{destination.name}.",
        suffix=".part",
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


def _image_metadata_for_path(file_path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    if Image is None:
        return None, ["image_metadata_unavailable"]
    try:
        with Image.open(file_path) as image:
            image.load()
            return {
                "width": int(image.width),
                "height": int(image.height),
                "mode": str(image.mode or ""),
            }, []
    except (OSError, UnidentifiedImageError, ValueError):
        return None, ["image_metadata_unavailable"]


def _pdf_version_for_path(file_path: Path) -> str:
    with file_path.open("rb") as handle:
        header = handle.read(8)
    if header.startswith(b"%PDF-"):
        return header[5:].decode("ascii", errors="ignore").strip()
    return ""


def _pdf_metadata_for_path(file_path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    if PdfReader is None:
        return None, ["pdf_metadata_unavailable"]
    try:
        with file_path.open("rb") as handle:
            reader = PdfReader(handle, strict=False)
            return {
                "page_count": len(reader.pages),
                "encrypted": bool(reader.is_encrypted),
                "pdf_version": _pdf_version_for_path(file_path),
            }, []
    except Exception:
        return None, ["pdf_metadata_unavailable"]


def _build_baseline_record(relative_path: str, file_path: Path, sha256: str, size_bytes: int) -> dict[str, Any]:
    mime_type = mimetypes.guess_type(file_path.name, strict=False)[0] or ""
    extension = _normalized_extension(relative_path)
    media_kind = _media_kind_for_path(relative_path, mime_type)
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

    if media_kind == "image":
        image_metadata, image_warnings = _image_metadata_for_path(file_path)
        warnings.extend(image_warnings)
        if image_metadata is not None:
            record["image"] = image_metadata
    elif extension == ".pdf":
        pdf_metadata, pdf_warnings = _pdf_metadata_for_path(file_path)
        warnings.extend(pdf_warnings)
        if pdf_metadata is not None:
            record["pdf"] = pdf_metadata

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
) -> StagedUploadResult:
    if not uploads:
        raise UploadStagingError("files_required", "at least one upload file is required")

    upload_root.mkdir(parents=True, exist_ok=True)
    batch_id = build_batch_id(now)
    batch_root = upload_root / batch_id
    input_dir = batch_root / "input"
    portal_dir = batch_root / "_portal"
    tmp_dir = batch_root / "tmp"
    baseline_manifest_path = portal_dir / BASELINE_MANIFEST_FILENAME
    capture_metadata_path = portal_dir / CAPTURE_METADATA_FILENAME
    upload_receipt_path = portal_dir / UPLOAD_RECEIPT_FILENAME
    relative_paths = _resolve_relative_paths(uploads, client_manifest_paths)
    seen_paths: set[str] = set()
    warnings: list[str] = []
    total_bytes = 0
    records: list[dict[str, Any]] = []
    capture_metadata_record_count = 0

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

            try:
                upload.stream.seek(0)
            except (AttributeError, OSError):
                pass
            sha256, size_bytes = _stream_upload_to_path(upload.stream, destination_path)
            total_bytes += size_bytes
            records.append(_build_baseline_record(relative_path, destination_path, sha256, size_bytes))

        records.sort(key=lambda item: str(item.get("relative_path") or ""))
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
            warnings=tuple(_deduplicate_warnings(warnings)),
        )
    except Exception:
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

    retained = {str(Path(os.path.realpath(str(path)))) for path in retained_input_dirs if str(path).strip()}
    removed: list[str] = []
    for batch_dir in upload_root.iterdir():
        try:
            if not batch_dir.is_dir():
                continue
            input_dir = batch_dir / "input"
            input_dir_real = str(Path(os.path.realpath(input_dir)))
            if input_dir_real in retained:
                continue
            age_seconds = now - batch_dir.stat().st_mtime
            if age_seconds < ttl_seconds:
                continue
            shutil.rmtree(batch_dir, ignore_errors=False)
            removed.append(batch_dir.name)
        except OSError:
            continue
    return sorted(removed)
