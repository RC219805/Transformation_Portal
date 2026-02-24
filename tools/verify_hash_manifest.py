#!/usr/bin/env python3
"""Verify archive bytes against Phase 3 hash manifest output."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
from uuid import uuid4

READ_CHUNK_BYTES = 1024 * 1024
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


@dataclass(frozen=True)
class ExpectedRow:
    row_number: int
    origin_drive: str
    partition: str
    relpath: str
    filesize_bytes: int
    sha256: str
    hash_status: str
    error: str


@dataclass(frozen=True)
class ObservedRow:
    status: str
    filesize_bytes: int
    sha256: str
    error: str


def atomic_write(path: Path, writer_func: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        writer_func(tmp_path)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    def _write(tmp_path: Path) -> None:
        with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    atomic_write(path, _write)


def _open_manifest_reader(path: Path) -> List[ExpectedRow]:
    if path.suffix == ".gz":
        handle = gzip.open(path, "rt", encoding="utf-8", newline="")
    else:
        handle = path.open("r", encoding="utf-8", newline="")

    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"hash manifest has no header: {path}")

        missing = [col for col in HASH_MANIFEST_COLUMNS if col not in reader.fieldnames]
        if missing:
            raise SystemExit(f"hash manifest missing required columns: {', '.join(missing)}")

        rows: List[ExpectedRow] = []
        for idx, row in enumerate(reader, start=1):
            raw_size = str(row.get("filesize_bytes") or "0")
            try:
                filesize = int(raw_size)
            except ValueError as exc:
                raise SystemExit(f"Invalid filesize_bytes at row {idx}: {raw_size!r}") from exc

            rows.append(
                ExpectedRow(
                    row_number=idx,
                    origin_drive=str(row.get("origin_drive") or ""),
                    partition=str(row.get("partition") or ""),
                    relpath=str(row.get("relpath") or ""),
                    filesize_bytes=filesize,
                    sha256=str(row.get("sha256") or ""),
                    hash_status=str(row.get("hash_status") or ""),
                    error=str(row.get("error") or ""),
                )
            )

    return sorted(rows, key=lambda r: (r.origin_drive, r.partition, r.relpath, r.row_number))


def _materialize_relpath(relpath: str) -> Tuple[Path | None, str]:
    normalized = relpath.replace("\\", "/").lstrip("/")
    if not normalized or normalized == ".":
        return None, "invalid_relpath_empty"

    parts = [part for part in normalized.split("/") if part and part != "."]
    if not parts:
        return None, "invalid_relpath_empty"
    if any(part == ".." for part in parts):
        return None, "invalid_relpath_parent_ref"

    return Path(*parts), ""


def _sha256_for_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(READ_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def observe_row(archive_root: Path, expected_row: ExpectedRow) -> ObservedRow:
    rel_path_obj, relpath_error = _materialize_relpath(expected_row.relpath)
    if rel_path_obj is None:
        return ObservedRow(status=STATUS_SKIPPED, filesize_bytes=0, sha256="", error=relpath_error)

    abs_path = archive_root / rel_path_obj
    try:
        stat_result = abs_path.lstat()
    except FileNotFoundError:
        return ObservedRow(status=STATUS_MISSING, filesize_bytes=0, sha256="", error="missing")
    except PermissionError:
        return ObservedRow(status=STATUS_UNREADABLE, filesize_bytes=0, sha256="", error="permission_denied")
    except OSError:
        return ObservedRow(status=STATUS_UNREADABLE, filesize_bytes=0, sha256="", error="stat_failed")

    if abs_path.is_symlink():
        return ObservedRow(status=STATUS_SKIPPED, filesize_bytes=0, sha256="", error="symlink_skipped")

    if not abs_path.is_file():
        return ObservedRow(status=STATUS_SKIPPED, filesize_bytes=0, sha256="", error="not_regular_file")

    try:
        digest = _sha256_for_path(abs_path)
    except PermissionError:
        return ObservedRow(status=STATUS_UNREADABLE, filesize_bytes=0, sha256="", error="permission_denied")
    except OSError:
        return ObservedRow(status=STATUS_UNREADABLE, filesize_bytes=0, sha256="", error="read_failed")

    return ObservedRow(status=STATUS_OK, filesize_bytes=int(stat_result.st_size), sha256=digest, error="")


def compare_rows(expected: ExpectedRow, observed: ObservedRow) -> List[Dict[str, Any]]:
    mismatches: List[Dict[str, Any]] = []

    common = {
        "origin_drive": expected.origin_drive,
        "partition": expected.partition,
        "relpath": expected.relpath,
        "expected_status": expected.hash_status,
        "observed_status": observed.status,
    }

    if expected.hash_status != observed.status:
        mismatches.append({**common, "issue": "status_mismatch"})
        return mismatches

    if expected.hash_status == STATUS_OK:
        if expected.filesize_bytes != observed.filesize_bytes:
            mismatches.append(
                {
                    **common,
                    "issue": "filesize_mismatch",
                    "expected_filesize_bytes": expected.filesize_bytes,
                    "observed_filesize_bytes": observed.filesize_bytes,
                }
            )
        if expected.sha256 != observed.sha256:
            mismatches.append(
                {
                    **common,
                    "issue": "sha256_mismatch",
                    "expected_sha256": expected.sha256,
                    "observed_sha256": observed.sha256,
                }
            )
        return mismatches

    if expected.error != observed.error:
        mismatches.append(
            {
                **common,
                "issue": "error_mismatch",
                "expected_error": expected.error,
                "observed_error": observed.error,
            }
        )

    return mismatches


def verify_rows(
    selected_rows: Sequence[ExpectedRow],
    archive_root: Path,
    workers: int,
) -> List[Dict[str, Any]]:
    if workers <= 1:
        mismatches: List[Dict[str, Any]] = []
        for expected in selected_rows:
            observed = observe_row(archive_root, expected)
            mismatches.extend(compare_rows(expected, observed))
        return mismatches

    def _observe_and_compare(expected: ExpectedRow) -> List[Dict[str, Any]]:
        observed = observe_row(archive_root, expected)
        return compare_rows(expected, observed)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        nested = list(executor.map(_observe_and_compare, selected_rows))

    mismatches = []
    for item in nested:
        mismatches.extend(item)
    return mismatches


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hash-manifest", required=True, help="Path to hash_manifest.csv.gz")
    parser.add_argument("--archive-root", required=True, help="Root path used to resolve relpath entries")
    parser.add_argument(
        "--report-path",
        default="",
        help="Output JSON report path (default: alongside hash manifest as verification_report.json)",
    )
    parser.add_argument("--verify-all", action="store_true", help="Verify every row in hash manifest")
    parser.add_argument(
        "--verify-sample",
        type=int,
        default=0,
        metavar="N",
        help="Deterministically verify first N canonical rows",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Verification worker count (ordering remains deterministic)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.verify_sample < 0:
        raise SystemExit("--verify-sample must be >= 0")

    verify_mode = "all"
    if args.verify_sample > 0:
        verify_mode = "sample"
    elif not args.verify_all:
        verify_mode = "all"

    manifest_path = Path(args.hash_manifest)
    archive_root = Path(args.archive_root)
    all_rows = _open_manifest_reader(manifest_path)

    if verify_mode == "sample":
        selected_rows = all_rows[: args.verify_sample]
    else:
        selected_rows = all_rows

    mismatches = verify_rows(selected_rows, archive_root=archive_root, workers=args.workers)

    rows_checked = len(selected_rows)
    report = {
        "hash_algorithm": "sha256",
        "mismatches": mismatches,
        "rows_checked": rows_checked,
        "rows_in_manifest": len(all_rows),
        "rows_matched": rows_checked - len(mismatches),
        "rows_mismatched": len(mismatches),
        "verify_mode": verify_mode,
    }
    if verify_mode == "sample":
        report["sample_size"] = args.verify_sample

    report_path = Path(args.report_path) if args.report_path else manifest_path.parent / "verification_report.json"
    write_json_atomic(report_path, report)

    print(f"Wrote verification report: {report_path}")
    if mismatches:
        print(f"Verification failed with {len(mismatches)} mismatches")
        return 1

    print("Verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
