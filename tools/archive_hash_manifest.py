#!/usr/bin/env python3
"""Generate deterministic Phase 3 hash outputs from a Phase 2 archive index."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from uuid import uuid4

HASH_ALGORITHM = "sha256"
HASH_MANIFEST_SCHEMA_VERSION = "1.0"
LEAF_FORMAT_VERSION = "1.0"
TREE_METHOD_VERSION = "1.0"
READ_CHUNK_BYTES = 1024 * 1024
HASH_MANIFEST_PREAMBLE = f"# hash_algorithm={HASH_ALGORITHM}"

HASH_MANIFEST_COLUMNS = [
    "origin_drive",
    "partition",
    "relpath",
    "filesize_bytes",
    "sha256",
    "hash_status",
    "error",
]

STATUS_OK = "ok"
STATUS_MISSING = "missing"
STATUS_UNREADABLE = "unreadable"
STATUS_SKIPPED = "skipped"

OUTPUT_HASH_MANIFEST = "hash_manifest.csv.gz"
OUTPUT_HASH_SUMMARY = "hash_summary.json"
OUTPUT_MERKLE_ROOTS = "merkle_roots.json"

DETERMINISTIC_GZIP_COMPRESSION = {"compresslevel": 9, "mtime": 0}
STRICT_NON_OK_EXIT_CODE = 2
STRICT_IDENTITY_EXIT_CODE = 3


@dataclass(frozen=True)
class ArchiveIndexRow:
    row_number: int
    origin_drive: str
    partition: str
    relpath: str


@dataclass(frozen=True)
class HashManifestRow:
    row_number: int
    origin_drive: str
    partition: str
    relpath: str
    filesize_bytes: int
    sha256: str
    hash_status: str
    error: str

    def as_csv_row(self) -> List[str]:
        return [
            self.origin_drive,
            self.partition,
            self.relpath,
            str(self.filesize_bytes),
            self.sha256,
            self.hash_status,
            self.error,
        ]


def atomic_write(path: Path, writer_func: Any) -> None:
    """Write `path` atomically by writing a temp file then replacing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        writer_func(tmp_path)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def write_hash_manifest_gzip(rows: Sequence[HashManifestRow], out_path: Path) -> None:
    """Write hash manifest CSV with deterministic gzip metadata."""

    def _write(tmp_path: Path) -> None:
        with tmp_path.open("wb") as raw:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw,
                compresslevel=DETERMINISTIC_GZIP_COMPRESSION["compresslevel"],
                mtime=DETERMINISTIC_GZIP_COMPRESSION["mtime"],
            ) as gz:
                with io.TextIOWrapper(gz, encoding="utf-8", newline="\n") as text:
                    writer = csv.writer(text, lineterminator="\n")
                    # Keep the custody artifact self-describing without changing CSV schema columns.
                    text.write(f"{HASH_MANIFEST_PREAMBLE}\n")
                    writer.writerow(HASH_MANIFEST_COLUMNS)
                    for row in rows:
                        writer.writerow(row.as_csv_row())

    atomic_write(out_path, _write)


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON with deterministic ordering and newline semantics."""

    def _write(tmp_path: Path) -> None:
        with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    atomic_write(path, _write)


def _open_csv_reader(path: Path) -> Iterable[Dict[str, str]]:
    if path.suffix == ".gz":
        handle = gzip.open(path, "rt", encoding="utf-8", newline="")
    else:
        handle = path.open("r", encoding="utf-8", newline="")

    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"Input CSV has no header row: {path}")

        required = {"origin_drive", "partition", "relpath"}
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise SystemExit("archive-index is missing required columns: " f"{', '.join(missing)}")

        for row in reader:
            yield row


def read_archive_index_rows(path: Path) -> List[ArchiveIndexRow]:
    """Read and canonicalize phase-2 archive index rows."""
    rows: List[ArchiveIndexRow] = []
    for idx, row in enumerate(_open_csv_reader(path), start=1):
        rows.append(
            ArchiveIndexRow(
                row_number=idx,
                origin_drive=str(row.get("origin_drive") or ""),
                partition=str(row.get("partition") or ""),
                relpath=str(row.get("relpath") or ""),
            )
        )

    return sorted(rows, key=lambda r: (r.origin_drive, r.partition, r.relpath, r.row_number))


def find_duplicate_identity_keys(rows: Sequence[ArchiveIndexRow]) -> List[Tuple[str, str, str, int]]:
    """Return duplicate identity groups in deterministic canonical order."""
    if not rows:
        return []

    duplicates: List[Tuple[str, str, str, int]] = []
    current_key = (rows[0].origin_drive, rows[0].partition, rows[0].relpath)
    current_count = 1

    for row in rows[1:]:
        key = (row.origin_drive, row.partition, row.relpath)
        if key == current_key:
            current_count += 1
            continue

        if current_count > 1:
            duplicates.append((current_key[0], current_key[1], current_key[2], current_count))
        current_key = key
        current_count = 1

    if current_count > 1:
        duplicates.append((current_key[0], current_key[1], current_key[2], current_count))

    return duplicates


def _materialize_relpath(relpath: str) -> Tuple[Path | None, str]:
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


def _sha256_for_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(READ_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def hash_one_row(archive_root: Path, row: ArchiveIndexRow) -> HashManifestRow:
    if "\x00" in row.origin_drive or "\x00" in row.partition:
        return HashManifestRow(
            row_number=row.row_number,
            origin_drive=row.origin_drive,
            partition=row.partition,
            relpath=row.relpath,
            filesize_bytes=0,
            sha256="",
            hash_status=STATUS_SKIPPED,
            error="invalid_identity_nul",
        )

    rel_path_obj, relpath_error = _materialize_relpath(row.relpath)
    if rel_path_obj is None:
        return HashManifestRow(
            row_number=row.row_number,
            origin_drive=row.origin_drive,
            partition=row.partition,
            relpath=row.relpath,
            filesize_bytes=0,
            sha256="",
            hash_status=STATUS_SKIPPED,
            error=relpath_error,
        )

    abs_path = archive_root / rel_path_obj
    try:
        stat_result = abs_path.lstat()
    except FileNotFoundError:
        return HashManifestRow(
            row_number=row.row_number,
            origin_drive=row.origin_drive,
            partition=row.partition,
            relpath=row.relpath,
            filesize_bytes=0,
            sha256="",
            hash_status=STATUS_MISSING,
            error="missing",
        )
    except PermissionError:
        return HashManifestRow(
            row_number=row.row_number,
            origin_drive=row.origin_drive,
            partition=row.partition,
            relpath=row.relpath,
            filesize_bytes=0,
            sha256="",
            hash_status=STATUS_UNREADABLE,
            error="permission_denied",
        )
    except OSError:
        return HashManifestRow(
            row_number=row.row_number,
            origin_drive=row.origin_drive,
            partition=row.partition,
            relpath=row.relpath,
            filesize_bytes=0,
            sha256="",
            hash_status=STATUS_UNREADABLE,
            error="stat_failed",
        )

    if abs_path.is_symlink():
        return HashManifestRow(
            row_number=row.row_number,
            origin_drive=row.origin_drive,
            partition=row.partition,
            relpath=row.relpath,
            filesize_bytes=0,
            sha256="",
            hash_status=STATUS_SKIPPED,
            error="symlink_skipped",
        )

    if not abs_path.is_file():
        return HashManifestRow(
            row_number=row.row_number,
            origin_drive=row.origin_drive,
            partition=row.partition,
            relpath=row.relpath,
            filesize_bytes=0,
            sha256="",
            hash_status=STATUS_SKIPPED,
            error="not_regular_file",
        )

    filesize_bytes = int(stat_result.st_size)
    try:
        digest = _sha256_for_path(abs_path)
    except PermissionError:
        return HashManifestRow(
            row_number=row.row_number,
            origin_drive=row.origin_drive,
            partition=row.partition,
            relpath=row.relpath,
            filesize_bytes=0,
            sha256="",
            hash_status=STATUS_UNREADABLE,
            error="permission_denied",
        )
    except OSError:
        return HashManifestRow(
            row_number=row.row_number,
            origin_drive=row.origin_drive,
            partition=row.partition,
            relpath=row.relpath,
            filesize_bytes=0,
            sha256="",
            hash_status=STATUS_UNREADABLE,
            error="read_failed",
        )

    return HashManifestRow(
        row_number=row.row_number,
        origin_drive=row.origin_drive,
        partition=row.partition,
        relpath=row.relpath,
        filesize_bytes=filesize_bytes,
        sha256=digest,
        hash_status=STATUS_OK,
        error="",
    )


def build_hash_manifest(
    archive_rows: Sequence[ArchiveIndexRow],
    archive_root: Path,
    workers: int,
) -> List[HashManifestRow]:
    """Hash rows with deterministic output ordering regardless of worker count."""
    if workers <= 1:
        return [hash_one_row(archive_root, row) for row in archive_rows]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        result = list(executor.map(lambda row: hash_one_row(archive_root, row), archive_rows))
    return result


def build_hash_summary(rows: Sequence[HashManifestRow]) -> Dict[str, Any]:
    hashed_ok = sum(1 for row in rows if row.hash_status == STATUS_OK)
    missing = sum(1 for row in rows if row.hash_status == STATUS_MISSING)
    unreadable = sum(1 for row in rows if row.hash_status == STATUS_UNREADABLE)
    skipped = sum(1 for row in rows if row.hash_status == STATUS_SKIPPED)

    return {
        "hash_algorithm": HASH_ALGORITHM,
        "hash_manifest_schema_version": HASH_MANIFEST_SCHEMA_VERSION,
        "hashed_ok": hashed_ok,
        "missing": missing,
        "rows_total": len(rows),
        "skipped": skipped,
        "total_bytes_hashed": sum(row.filesize_bytes for row in rows if row.hash_status == STATUS_OK),
        "unreadable": unreadable,
    }


def _leaf_preimage(row: HashManifestRow) -> bytes:
    # Canonical leaf bytes: identity + status + digest separated by NUL.
    return "\0".join(
        (
            row.origin_drive,
            row.partition,
            row.relpath,
            row.hash_status,
            row.sha256,
        )
    ).encode("utf-8")


def _merkle_root(leaf_hashes: Sequence[bytes]) -> str:
    if not leaf_hashes:
        return hashlib.sha256(b"").hexdigest()

    layer = list(leaf_hashes)
    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])

        next_layer: List[bytes] = []
        for index in range(0, len(layer), 2):
            next_layer.append(hashlib.sha256(layer[index] + layer[index + 1]).digest())
        layer = next_layer

    return layer[0].hex()


def build_merkle_roots(rows: Sequence[HashManifestRow]) -> Dict[str, Any]:
    sorted_rows = sorted(rows, key=lambda r: (r.origin_drive, r.partition, r.relpath, r.row_number))

    partition_groups: Dict[Tuple[str, str], List[HashManifestRow]] = defaultdict(list)
    for row in sorted_rows:
        partition_groups[(row.origin_drive, row.partition)].append(row)

    partitions_payload: List[Dict[str, Any]] = []
    for origin_drive, partition in sorted(partition_groups):
        partition_rows = partition_groups[(origin_drive, partition)]
        partition_leaf_hashes = [hashlib.sha256(_leaf_preimage(row)).digest() for row in partition_rows]
        partitions_payload.append(
            {
                "origin_drive": origin_drive,
                "partition": partition,
                "leaf_count": len(partition_leaf_hashes),
                "root_sha256": _merkle_root(partition_leaf_hashes),
            }
        )

    global_leaf_hashes = [hashlib.sha256(_leaf_preimage(row)).digest() for row in sorted_rows]

    return {
        "global": {
            "leaf_count": len(global_leaf_hashes),
            "root_sha256": _merkle_root(global_leaf_hashes),
        },
        "hash_algorithm": HASH_ALGORITHM,
        "leaf_format": "sha256(origin_drive\\0partition\\0relpath\\0hash_status\\0sha256_hex)",
        "leaf_format_version": LEAF_FORMAT_VERSION,
        "leaf_hash_algorithm": HASH_ALGORITHM,
        "partitions": partitions_payload,
        "tree_method": "binary_duplicate_last",
        "tree_method_version": TREE_METHOD_VERSION,
    }


def _require_jsonschema() -> Any:
    try:
        import jsonschema  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "Schema validation requested but dependency 'jsonschema' is unavailable. "
            "Install with: pip install -r requirements/tools-archive.txt"
        ) from exc
    return jsonschema


def _schema_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "docs" / "archive" / "schemas"


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_schema_type(raw_type: str) -> str:
    normalized = str(raw_type).strip().lower()
    if normalized in {"string", "str"}:
        return "string"
    if normalized in {"int", "integer"}:
        return "integer"
    if normalized in {"float", "number"}:
        return "number"
    if normalized in {"bool", "boolean"}:
        return "boolean"
    raise SystemExit(f"Unsupported schema type in hash manifest schema: {raw_type}")


def _validate_hash_manifest_contract(schema_path: Path, rows: Sequence[HashManifestRow]) -> None:
    schema = _load_json(schema_path)
    columns = schema.get("columns")
    if not isinstance(columns, list):
        raise SystemExit(f"Invalid schema format in {schema_path.name}: expected top-level 'columns' list")

    schema_specs: List[Tuple[str, str, List[str]]] = []
    expected_columns: List[str] = []
    for idx, col in enumerate(columns):
        if not isinstance(col, dict):
            raise SystemExit(f"Invalid schema format in {schema_path.name}: columns[{idx}] must be object")
        name = col.get("name")
        if not isinstance(name, str) or not name:
            raise SystemExit(f"Invalid schema format in {schema_path.name}: columns[{idx}] missing 'name'")
        col_type = _normalize_schema_type(str(col.get("type", "string")))
        enum_values = col.get("enum", [])
        if enum_values and not (isinstance(enum_values, list) and all(isinstance(item, str) for item in enum_values)):
            raise SystemExit(f"Invalid enum definition for column {name} in {schema_path.name}")
        expected_columns.append(name)
        schema_specs.append((name, col_type, list(enum_values)))

    if expected_columns != HASH_MANIFEST_COLUMNS:
        raise SystemExit("hash_manifest column order mismatch. " f"Schema={expected_columns} Tool={HASH_MANIFEST_COLUMNS}")

    for idx, row in enumerate(rows, start=1):
        row_value_map: Dict[str, Any] = {
            "origin_drive": row.origin_drive,
            "partition": row.partition,
            "relpath": row.relpath,
            "filesize_bytes": row.filesize_bytes,
            "sha256": row.sha256,
            "hash_status": row.hash_status,
            "error": row.error,
        }
        for col_name, col_type, enum_values in schema_specs:
            value = row_value_map[col_name]
            if col_type == "string":
                if not isinstance(value, str):
                    raise SystemExit(f"Row {idx} column '{col_name}' expected string, found {type(value).__name__}")
            elif col_type == "integer":
                if not isinstance(value, int):
                    raise SystemExit(f"Row {idx} column '{col_name}' expected integer, found {type(value).__name__}")
            elif col_type == "number":
                if not isinstance(value, (int, float)):
                    raise SystemExit(f"Row {idx} column '{col_name}' expected number, found {type(value).__name__}")
            elif col_type == "boolean":
                if not isinstance(value, bool):
                    raise SystemExit(f"Row {idx} column '{col_name}' expected boolean, found {type(value).__name__}")

            if enum_values and str(value) not in enum_values:
                raise SystemExit(f"Row {idx} column '{col_name}' value '{value}' not in enum domain {enum_values}")


def _validate_json_file(path: Path, schema_path: Path) -> None:
    jsonschema = _require_jsonschema()
    payload = _load_json(path)
    schema = _load_json(schema_path)
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(payload), key=lambda err: list(err.path))
    if errors:
        first = errors[0]
        location = "$"
        if first.path:
            location += "." + ".".join(str(part) for part in first.path)
        raise SystemExit(f"Schema validation failed for {schema_path.name} at {location}: {first.message}")


def validate_outputs_against_schemas(out_dir: Path, rows: Sequence[HashManifestRow]) -> None:
    schema_dir = _schema_dir()
    _validate_hash_manifest_contract(schema_dir / "hash_manifest.schema.json", rows)
    _validate_json_file(out_dir / OUTPUT_HASH_SUMMARY, schema_dir / "hash_summary.schema.json")
    _validate_json_file(out_dir / OUTPUT_MERKLE_ROOTS, schema_dir / "merkle_roots.schema.json")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-index", required=True, help="Path to archive_index_normalized.csv[.gz]")
    parser.add_argument("--archive-root", required=True, help="Archive filesystem root used to resolve relpath")
    parser.add_argument("--out-dir", required=True, help="Output directory for phase 3 artifacts")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Hashing worker count (ordering remains deterministic)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any manifest row has status != ok",
    )
    parser.add_argument(
        "--strict-identity",
        action="store_true",
        help="Exit non-zero if duplicate (origin_drive, partition, relpath) keys are present",
    )
    parser.add_argument(
        "--validate-schemas",
        action="store_true",
        help="Validate JSON outputs against docs/archive/schemas contracts",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()

    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    archive_index_path = Path(args.archive_index)
    archive_root = Path(args.archive_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    archive_rows = read_archive_index_rows(archive_index_path)
    duplicate_identity_keys = find_duplicate_identity_keys(archive_rows)
    if args.strict_identity and duplicate_identity_keys:
        print(f"Duplicate identity keys detected: {len(duplicate_identity_keys)}")
        for origin_drive, partition, relpath, count in duplicate_identity_keys:
            print(f"  {origin_drive}|{partition}|{relpath} count={count}")
        return STRICT_IDENTITY_EXIT_CODE

    manifest_rows = build_hash_manifest(archive_rows, archive_root=archive_root, workers=args.workers)

    out_hash_manifest = out_dir / OUTPUT_HASH_MANIFEST
    out_hash_summary = out_dir / OUTPUT_HASH_SUMMARY
    out_merkle_roots = out_dir / OUTPUT_MERKLE_ROOTS

    write_hash_manifest_gzip(manifest_rows, out_hash_manifest)
    write_json_atomic(out_hash_summary, build_hash_summary(manifest_rows))
    write_json_atomic(out_merkle_roots, build_merkle_roots(manifest_rows))

    if args.validate_schemas:
        validate_outputs_against_schemas(out_dir, manifest_rows)
        print("Schema validation passed for hash_summary.json and merkle_roots.json")

    non_ok = [row for row in manifest_rows if row.hash_status != STATUS_OK]
    print(f"Wrote {len(manifest_rows)} rows to {out_hash_manifest}")
    if non_ok:
        print(f"Non-ok rows: {len(non_ok)}")

    elapsed = max(time.perf_counter() - started, 1e-9)
    summary = build_hash_summary(manifest_rows)
    bytes_hashed = int(summary["total_bytes_hashed"])
    mb_per_sec = (bytes_hashed / (1024 * 1024)) / elapsed
    files_per_sec = float(summary["hashed_ok"]) / elapsed
    print(f"Elapsed: {elapsed:.3f}s | Throughput: {mb_per_sec:.2f} MiB/s | Files/s: {files_per_sec:.2f}")

    if args.strict and non_ok:
        return STRICT_NON_OK_EXIT_CODE

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
