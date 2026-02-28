#!/usr/bin/env python3
"""Build deterministic tp.archive.manifest.v2 JSONL artifacts."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from archive_governance_common import (  # pylint: disable=wrong-import-position
    atomic_write_text,
    deterministic_json_dumps,
    json_line,
)

EXIT_SUCCESS = 0
EXIT_INPUT_ERROR = 2
EXIT_BUILD_ERROR = 3

MIME_BY_EXTENSION = {
    ".arw": "image/x-sony-arw",
    ".avi": "video/x-msvideo",
    ".bmp": "image/bmp",
    ".cr2": "image/x-canon-cr2",
    ".cr3": "image/x-canon-cr3",
    ".csv": "text/csv",
    ".dng": "image/x-adobe-dng",
    ".flac": "audio/flac",
    ".gif": "image/gif",
    ".gz": "application/gzip",
    ".heic": "image/heic",
    ".heif": "image/heif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
    ".mkv": "video/x-matroska",
    ".mov": "video/quicktime",
    ".mp3": "audio/mpeg",
    ".mp4": "video/mp4",
    ".nef": "image/x-nikon-nef",
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".tar": "application/x-tar",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".txt": "text/plain",
    ".wav": "audio/wav",
    ".webp": "image/webp",
    ".xml": "application/xml",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
    ".zip": "application/zip",
}
MIME_BY_COMPOUND_EXTENSION = {
    ".jsonl.gz": "application/gzip",
    ".tar.gz": "application/gzip",
}


def _open_csv_reader(path: Path, *, required_columns: set[str] | None = None) -> Iterable[dict[str, str]]:
    if path.suffix == ".gz":
        handle = gzip.open(path, "rt", encoding="utf-8", newline="")
    else:
        handle = path.open("r", encoding="utf-8", newline="")

    with handle:
        header_line: str | None = None
        for line in handle:
            if not line.strip():
                continue
            if line.lstrip().startswith("#"):
                continue
            header_line = line
            break

        if header_line is None:
            raise ValueError(f"CSV has no header row: {path}")

        def _iter_lines() -> Iterable[str]:
            yield header_line
            for line in handle:
                if not line.strip():
                    continue
                if line.lstrip().startswith("#"):
                    continue
                yield line

        reader = csv.DictReader(_iter_lines())
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {path}")
        if required_columns is not None:
            available_columns = {name for name in reader.fieldnames if name}
            missing = sorted(required_columns.difference(available_columns))
            if missing:
                raise ValueError(f"CSV missing required columns ({', '.join(missing)}): {path}")
        yield from reader


def _consume_csv_rows(path: Path, *, required_columns: set[str] | None = None) -> int:
    row_count = 0
    for _ in _open_csv_reader(path, required_columns=required_columns):
        row_count += 1
    return row_count


def _materialize_relpath(relpath: str) -> tuple[Path | None, str]:
    if "\x00" in relpath:
        return None, "invalid_relpath_nul"
    normalized_raw = relpath.replace("\\", "/")
    if not normalized_raw or normalized_raw == ".":
        return None, "invalid_relpath_empty"
    if normalized_raw.startswith("/"):
        return None, "invalid_relpath_anchored"

    normalized = normalized_raw.lstrip("/")
    if not normalized or normalized == ".":
        return None, "invalid_relpath_empty"

    parts = [part for part in normalized.split("/") if part and part != "."]
    if not parts:
        return None, "invalid_relpath_empty"
    if any(part == ".." for part in parts):
        return None, "invalid_relpath_parent_ref"

    first = parts[0]
    if len(first) >= 2 and first[0].isalpha() and first[1] == ":":
        return None, "invalid_relpath_drive_spec"

    candidate = Path(*parts)
    if candidate.is_absolute() or getattr(candidate, "drive", ""):
        return None, "invalid_relpath_anchored"

    return candidate, ""


def _iso_utc(epoch_seconds: float | None) -> str | None:
    if epoch_seconds is None:
        return None
    return datetime.fromtimestamp(epoch_seconds, tz=UTC).isoformat().replace("+00:00", "Z")


def _path_times(path: Path) -> tuple[str | None, str | None, str | None, str]:
    try:
        stat_result = path.stat()
    except OSError:
        return None, None, None, "missing"

    modified_utc = _iso_utc(float(stat_result.st_mtime))
    accessed_utc = _iso_utc(float(stat_result.st_atime))

    if hasattr(stat_result, "st_birthtime"):
        created_utc = _iso_utc(float(stat_result.st_birthtime))
        created_source = "birthtime"
    elif sys.platform.startswith("win"):
        created_utc = _iso_utc(float(stat_result.st_ctime))
        created_source = "ctime"
    else:
        created_utc = None
        created_source = "unsupported"

    return created_utc, modified_utc, accessed_utc, created_source


def _normalize_relpath(relpath: str) -> tuple[str, Path | None, str]:
    relpath_obj, relpath_error = _materialize_relpath(relpath)
    if relpath_obj is not None:
        return relpath_obj.as_posix(), relpath_obj, ""
    return relpath.replace("\\", "/"), None, relpath_error


def _deterministic_mime(relpath: str) -> str:
    path = Path(relpath)
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if len(suffixes) >= 2:
        compound = f"{suffixes[-2]}{suffixes[-1]}"
        if compound in MIME_BY_COMPOUND_EXTENSION:
            return MIME_BY_COMPOUND_EXTENSION[compound]

    extension = path.suffix.lower()
    if extension in MIME_BY_EXTENSION:
        return MIME_BY_EXTENSION[extension]
    return "application/octet-stream"


def _load_rights(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}

    rights_map: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid rights JSONL at line {line_number}: {exc}") from exc

            relpath_raw = str(payload.get("relpath") or "").strip()
            if not relpath_raw:
                raise ValueError(f"Missing relpath in rights JSONL line {line_number}")
            relpath, _, _ = _normalize_relpath(relpath_raw)
            flags_raw = payload.get("rights_flags")
            if not isinstance(flags_raw, list) or not flags_raw:
                raise ValueError(f"rights_flags must be non-empty list in rights JSONL line {line_number}")
            flags = [str(flag).strip() for flag in flags_raw if str(flag).strip()]
            if not flags:
                raise ValueError(f"rights_flags must contain at least one non-empty value at line {line_number}")
            rights_map[relpath] = {
                "rights_flags": sorted(set(flags)),
                "owner": str(payload.get("owner") or "").strip() or None,
            }

    return rights_map


def _load_hash_rows(path: Path) -> list[dict[str, str]]:
    required = {
        "origin_drive",
        "partition",
        "relpath",
        "filesize_bytes",
        "sha256",
        "hash_status",
        "error",
    }
    rows = list(_open_csv_reader(path))
    if not rows:
        return []
    missing = sorted(required.difference(rows[0].keys()))
    if missing:
        raise ValueError(f"hash manifest missing required columns: {', '.join(missing)}")

    return sorted(
        rows,
        key=lambda row: (
            str(row.get("origin_drive") or ""),
            str(row.get("partition") or ""),
            str(row.get("relpath") or ""),
        ),
    )


def _build_entry(
    *,
    archive_root: Path,
    hash_row: dict[str, str],
    rights_map: dict[str, dict[str, Any]],
    collection_id: str,
    default_owner: str,
) -> dict[str, Any]:
    raw_relpath = str(hash_row.get("relpath") or "")
    relpath, relpath_obj, relpath_error = _normalize_relpath(raw_relpath)

    created_utc: str | None = None
    modified_utc: str | None = None
    accessed_utc: str | None = None
    created_source = "missing"

    if relpath_obj is not None:
        created_utc, modified_utc, accessed_utc, created_source = _path_times(archive_root / relpath_obj)
    elif relpath_error:
        created_source = relpath_error

    guessed_mime = _deterministic_mime(relpath)
    extension = Path(relpath).suffix.lower()

    rights = rights_map.get(relpath, {})
    rights_flags = rights.get("rights_flags") or ["unspecified"]
    owner = rights.get("owner") or default_owner

    provenance_seed = "\0".join(
        [
            str(hash_row.get("origin_drive") or ""),
            str(hash_row.get("partition") or ""),
            relpath,
        ]
    )

    size_raw = str(hash_row.get("filesize_bytes") or "0")
    try:
        size_bytes = int(size_raw)
    except ValueError as exc:
        raise ValueError(f"Invalid filesize_bytes for relpath={relpath!r}: {size_raw!r}") from exc

    return {
        "origin_drive": str(hash_row.get("origin_drive") or ""),
        "partition": str(hash_row.get("partition") or ""),
        "relpath": relpath,
        "size_bytes": max(size_bytes, 0),
        "sha256": str(hash_row.get("sha256") or ""),
        "hash_status": str(hash_row.get("hash_status") or ""),
        "created_utc": created_utc,
        "modified_utc": modified_utc,
        "accessed_utc": accessed_utc,
        "created_source": created_source,
        "mime": guessed_mime or "application/octet-stream",
        "extension": extension,
        "rights_flags": sorted(set(str(flag) for flag in rights_flags)),
        "owner": str(owner),
        "collection_id": collection_id,
        "provenance_id": hashlib.sha256(provenance_seed.encode("utf-8")).hexdigest(),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-index", required=True, help="Path to archive_index_normalized.csv[.gz]")
    parser.add_argument("--hash-manifest", required=True, help="Path to hash_manifest.csv[.gz]")
    parser.add_argument("--archive-root", required=True, help="Archive root path used to resolve relpath")
    parser.add_argument("--out-jsonl", required=True, help="Output path for archive_manifest_v2.jsonl")
    parser.add_argument("--out-summary", required=True, help="Output path for archive_manifest_v2.summary.json")
    parser.add_argument("--rights-jsonl", default=None, help="Optional rights JSONL from apply_rights_policy.py")
    parser.add_argument("--collection-id", default="UNSPECIFIED", help="Collection identifier")
    parser.add_argument("--owner", default="UNSPECIFIED", help="Default owner value")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    archive_root = Path(args.archive_root)
    out_jsonl = Path(args.out_jsonl)
    out_summary = Path(args.out_summary)

    try:
        # Parse both for sanity/lineage validation even though hash rows drive output.
        _consume_csv_rows(
            Path(args.archive_index),
            required_columns={"origin_drive", "partition", "relpath"},
        )
        hash_rows = _load_hash_rows(Path(args.hash_manifest))
        rights_map = _load_rights(Path(args.rights_jsonl) if args.rights_jsonl else None)
    except ValueError as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    try:
        entries = [
            _build_entry(
                archive_root=archive_root,
                hash_row=row,
                rights_map=rights_map,
                collection_id=args.collection_id,
                default_owner=args.owner,
            )
            for row in hash_rows
        ]
    except ValueError as exc:
        print(f"Build error: {exc}", file=sys.stderr)
        return EXIT_BUILD_ERROR

    by_hash_status = Counter(str(item.get("hash_status") or "") for item in entries)
    by_created_source = Counter(str(item.get("created_source") or "") for item in entries)
    rights_coverage = sum(1 for item in entries if item.get("rights_flags") and item["rights_flags"] != ["unspecified"])

    summary_payload: dict[str, Any] = {
        "schema_version": "tp.archive.manifest.v2.summary.v1",
        "entry_count": len(entries),
        "collection_id": args.collection_id,
        "archive_root": str(archive_root),
        "hash_status_counts": {key: int(value) for key, value in sorted(by_hash_status.items())},
        "created_source_counts": {key: int(value) for key, value in sorted(by_created_source.items())},
        "rights_classified_count": rights_coverage,
        "rights_unspecified_count": len(entries) - rights_coverage,
    }

    tmp_out_path = out_jsonl.with_name(f".{out_jsonl.name}.{uuid4().hex}.tmp")
    try:
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with tmp_out_path.open("w", encoding="utf-8", newline="\n") as out_handle:
            for entry in entries:
                out_handle.write(json_line(entry))
        tmp_out_path.replace(out_jsonl)
        atomic_write_text(out_summary, deterministic_json_dumps(summary_payload, pretty=True) + "\n")
    except OSError as exc:
        print(f"Build error: unable to write outputs: {exc}", file=sys.stderr)
        return EXIT_BUILD_ERROR
    finally:
        if tmp_out_path.exists():
            tmp_out_path.unlink()

    print(f"Wrote {len(entries)} manifest rows to {out_jsonl}")
    print(f"Wrote summary to {out_summary}")
    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
