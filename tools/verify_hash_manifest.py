#!/usr/bin/env python3
"""Verify archive bytes against Phase 3 hash manifest output."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import stat
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
from uuid import uuid4

READ_CHUNK_BYTES = 1024 * 1024
HASH_MANIFEST_COMMENT_PREFIX = "#"
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

# Merkle method emitted by tools/archive_hash_manifest.py::build_merkle_roots. A roots file that
# declares anything else cannot be recomputed here, so the binding check rejects it.
SUPPORTED_MERKLE_METHOD = {
    "hash_algorithm": "sha256",
    "leaf_hash_algorithm": "sha256",
    "leaf_format_version": "1.0",
    "tree_method": "binary_duplicate_last",
    "tree_method_version": "1.0",
}
HASH_SUMMARY_STATUS_FIELDS = {
    "hashed_ok": STATUS_OK,
    "missing": STATUS_MISSING,
    "unreadable": STATUS_UNREADABLE,
    "skipped": STATUS_SKIPPED,
}


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
        header_line: str | None = None
        for line in handle:
            if not line.strip():
                continue
            if line.lstrip().startswith(HASH_MANIFEST_COMMENT_PREFIX):
                continue
            header_line = line
            break

        if header_line is None:
            raise SystemExit(f"hash manifest has no header: {path}")

        def _lines_with_header(first_line: str, remaining_handle: Any) -> Any:
            yield first_line
            for tail_line in remaining_handle:
                if not tail_line.strip():
                    continue
                yield tail_line

        reader = csv.DictReader(_lines_with_header(header_line, handle))
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


def _check_symlink_components(archive_root: Path, rel_path_obj: Path) -> str:
    """Detect symlink traversal under archive_root without following it."""
    current = archive_root
    for part in rel_path_obj.parts:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            return ""
        except PermissionError:
            return "permission_denied"
        except OSError:
            return "stat_failed"

        if stat.S_ISLNK(mode):
            return "symlink_skipped"
    return ""


def _resolves_within_root(archive_root: Path, abs_path: Path) -> bool:
    """Defensive boundary check: resolved target must remain under archive_root."""
    try:
        root_resolved = archive_root.resolve(strict=False)
        target_resolved = abs_path.resolve(strict=False)
    except OSError:
        return True

    try:
        target_resolved.relative_to(root_resolved)
    except ValueError:
        return False
    return True


def observe_row(archive_root: Path, expected_row: ExpectedRow) -> ObservedRow:
    if "\x00" in expected_row.origin_drive or "\x00" in expected_row.partition:
        return ObservedRow(status=STATUS_SKIPPED, filesize_bytes=0, sha256="", error="invalid_identity_nul")

    rel_path_obj, relpath_error = _materialize_relpath(expected_row.relpath)
    if rel_path_obj is None:
        return ObservedRow(status=STATUS_SKIPPED, filesize_bytes=0, sha256="", error=relpath_error)

    abs_path = archive_root / rel_path_obj
    symlink_component_error = _check_symlink_components(archive_root, rel_path_obj)
    if symlink_component_error == "symlink_skipped":
        return ObservedRow(status=STATUS_SKIPPED, filesize_bytes=0, sha256="", error="symlink_skipped")
    if symlink_component_error == "permission_denied":
        return ObservedRow(status=STATUS_UNREADABLE, filesize_bytes=0, sha256="", error="permission_denied")
    if symlink_component_error == "stat_failed":
        return ObservedRow(status=STATUS_UNREADABLE, filesize_bytes=0, sha256="", error="stat_failed")

    if not _resolves_within_root(archive_root, abs_path):
        return ObservedRow(status=STATUS_SKIPPED, filesize_bytes=0, sha256="", error="symlink_skipped")

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


def _merkle_root_sha256(leaf_hashes: Sequence[bytes]) -> str:
    """Phase 3 tree method, shared with the generator through ``tp.crypto.merkle``."""
    try:
        from tp.crypto.merkle import merkle_root_sha256
    except ImportError:
        src_root = Path(__file__).resolve().parents[1] / "src"
        if str(src_root) not in sys.path:
            sys.path.insert(0, str(src_root))
        from tp.crypto.merkle import merkle_root_sha256

    return merkle_root_sha256(leaf_hashes)


def _leaf_hash(row: ExpectedRow) -> bytes:
    # Must stay byte-identical to tools/archive_hash_manifest.py::_leaf_preimage.
    preimage = "\0".join((row.origin_drive, row.partition, row.relpath, row.hash_status, row.sha256))
    return hashlib.sha256(preimage.encode("utf-8")).digest()


def recompute_merkle_roots(rows: Sequence[ExpectedRow]) -> Dict[str, Any]:
    """Rebuild the global and per-partition Merkle roots from every manifest row."""
    canonical_rows = sorted(rows, key=lambda r: (r.origin_drive, r.partition, r.relpath, r.row_number))

    global_leaves: List[bytes] = []
    partition_leaves: Dict[Tuple[str, str], List[bytes]] = {}
    for row in canonical_rows:
        leaf = _leaf_hash(row)
        global_leaves.append(leaf)
        partition_leaves.setdefault((row.origin_drive, row.partition), []).append(leaf)

    return {
        "global": {"leaf_count": len(global_leaves), "root_sha256": _merkle_root_sha256(global_leaves)},
        "partitions": {
            key: {"leaf_count": len(leaves), "root_sha256": _merkle_root_sha256(leaves)}
            for key, leaves in partition_leaves.items()
        },
    }


def _compare_merkle_scope(scope: Dict[str, Any], expected: Dict[str, Any], observed: Dict[str, Any]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    expected_leaf_count = expected.get("leaf_count")
    if type(expected_leaf_count) is not int or expected_leaf_count != observed["leaf_count"]:
        issues.append(
            {
                **scope,
                "issue": "leaf_count_mismatch",
                "expected_leaf_count": expected_leaf_count,
                "observed_leaf_count": observed["leaf_count"],
            }
        )
    expected_root = expected.get("root_sha256")
    if expected_root != observed["root_sha256"]:
        issues.append(
            {
                **scope,
                "issue": "root_mismatch",
                "expected_root_sha256": expected_root,
                "observed_root_sha256": observed["root_sha256"],
            }
        )
    return issues


def check_merkle_binding(rows: Sequence[ExpectedRow], roots_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compare ``merkle_roots.json`` (expected) with roots recomputed from manifest rows (observed).

    The detached signature and timestamp cover ``merkle_roots.json`` only; this is the step that
    ties the hash manifest rows to those signed roots.
    """
    issues: List[Dict[str, Any]] = []
    for field, supported in SUPPORTED_MERKLE_METHOD.items():
        declared = roots_payload.get(field)
        if declared != supported:
            issues.append({"issue": "merkle_method_unsupported", "field": field, "supported": supported, "declared": declared})

    observed = recompute_merkle_roots(rows)

    global_block = roots_payload.get("global")
    if isinstance(global_block, dict):
        issues.extend(_compare_merkle_scope({"scope": "global"}, global_block, observed["global"]))
    else:
        issues.append({"issue": "merkle_roots_malformed", "field": "global"})

    partitions_block = roots_payload.get("partitions")
    if not isinstance(partitions_block, list):
        issues.append({"issue": "merkle_roots_malformed", "field": "partitions"})
        return issues

    expected_partitions: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for entry in partitions_block:
        origin_drive = entry.get("origin_drive") if isinstance(entry, dict) else None
        partition = entry.get("partition") if isinstance(entry, dict) else None
        if not isinstance(origin_drive, str) or not isinstance(partition, str):
            issues.append({"issue": "merkle_roots_malformed", "field": "partitions"})
            continue
        scope = {"scope": "partition", "origin_drive": origin_drive, "partition": partition}
        if (origin_drive, partition) in expected_partitions:
            issues.append({**scope, "issue": "partition_duplicated"})
            continue
        expected_partitions[(origin_drive, partition)] = entry

    observed_partitions = observed["partitions"]
    for key in sorted(set(expected_partitions) | set(observed_partitions)):
        scope = {"scope": "partition", "origin_drive": key[0], "partition": key[1]}
        if key not in observed_partitions:
            issues.append({**scope, "issue": "partition_missing_from_manifest"})
        elif key not in expected_partitions:
            issues.append({**scope, "issue": "partition_missing_from_roots"})
        else:
            issues.extend(_compare_merkle_scope(scope, expected_partitions[key], observed_partitions[key]))
    return issues


def check_hash_summary_binding(rows: Sequence[ExpectedRow], summary_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compare ``hash_summary.json`` (expected) with counts recomputed from manifest rows (observed)."""
    observed: Dict[str, Any] = {"hash_algorithm": "sha256", "rows_total": len(rows)}
    for field, status in HASH_SUMMARY_STATUS_FIELDS.items():
        observed[field] = sum(1 for row in rows if row.hash_status == status)
    observed["total_bytes_hashed"] = sum(row.filesize_bytes for row in rows if row.hash_status == STATUS_OK)

    issues: List[Dict[str, Any]] = []
    for field, observed_value in observed.items():
        expected_value = summary_payload.get(field)
        if type(expected_value) is not type(observed_value) or expected_value != observed_value:
            issues.append(
                {"issue": "hash_summary_mismatch", "field": field, "expected": expected_value, "observed": observed_value}
            )
    return issues


def _load_bound_json(path: Path, label: str) -> Tuple[Dict[str, Any], str]:
    """Return the JSON object at ``path`` and the SHA-256 of the exact bytes that were parsed."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SystemExit(f"Unable to read {label}: {path}: {exc}") from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{label} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} must be a JSON object: {path}")
    return payload, hashlib.sha256(raw).hexdigest()


def verify_rows(
    selected_rows: Sequence[ExpectedRow],
    archive_root: Path,
    workers: int,
) -> Tuple[List[Dict[str, Any]], int]:
    if workers <= 1:
        mismatches: List[Dict[str, Any]] = []
        mismatched_row_keys: set[tuple[str, str, str, int]] = set()
        for expected in selected_rows:
            observed = observe_row(archive_root, expected)
            row_mismatches = compare_rows(expected, observed)
            if row_mismatches:
                mismatched_row_keys.add((expected.origin_drive, expected.partition, expected.relpath, expected.row_number))
            mismatches.extend(row_mismatches)
        return mismatches, len(mismatched_row_keys)

    def _observe_and_compare(expected: ExpectedRow) -> Tuple[ExpectedRow, List[Dict[str, Any]]]:
        observed = observe_row(archive_root, expected)
        return expected, compare_rows(expected, observed)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        nested = list(executor.map(_observe_and_compare, selected_rows))

    mismatches: List[Dict[str, Any]] = []
    mismatched_row_keys: set[tuple[str, str, str, int]] = set()
    for expected, row_mismatches in nested:
        if row_mismatches:
            mismatched_row_keys.add((expected.origin_drive, expected.partition, expected.relpath, expected.row_number))
        mismatches.extend(row_mismatches)
    return mismatches, len(mismatched_row_keys)


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
    parser.add_argument(
        "--merkle-roots",
        default="",
        help=(
            "Path to the signed merkle_roots.json. Recomputes the global and per-partition Merkle roots "
            "from every manifest row and fails on any root, leaf-count or partition mismatch"
        ),
    )
    parser.add_argument(
        "--hash-summary",
        default="",
        help="Path to hash_summary.json. Fails when its row, status or byte counts disagree with the manifest rows",
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

    mismatches, rows_mismatched = verify_rows(selected_rows, archive_root=archive_root, workers=args.workers)

    rows_checked = len(selected_rows)
    report = {
        "hash_algorithm": "sha256",
        "mismatches": mismatches,
        "rows_checked": rows_checked,
        "rows_in_manifest": len(all_rows),
        "rows_matched": rows_checked - rows_mismatched,
        "rows_mismatched": rows_mismatched,
        "verify_mode": verify_mode,
    }
    if verify_mode == "sample":
        report["sample_size"] = args.verify_sample

    # Binding checks always cover every manifest row, also in --verify-sample mode.
    merkle_issues: List[Dict[str, Any]] = []
    if args.merkle_roots:
        roots_payload, roots_sha256 = _load_bound_json(Path(args.merkle_roots), "merkle roots")
        merkle_issues = check_merkle_binding(all_rows, roots_payload)
        report["merkle_binding"] = {
            "issues": merkle_issues,
            "leaf_count": len(all_rows),
            "merkle_roots_sha256": roots_sha256,
            "verified": not merkle_issues,
        }
    summary_issues: List[Dict[str, Any]] = []
    if args.hash_summary:
        summary_payload, summary_sha256 = _load_bound_json(Path(args.hash_summary), "hash summary")
        summary_issues = check_hash_summary_binding(all_rows, summary_payload)
        report["hash_summary_binding"] = {
            "hash_summary_sha256": summary_sha256,
            "issues": summary_issues,
            "verified": not summary_issues,
        }

    report_path = Path(args.report_path) if args.report_path else manifest_path.parent / "verification_report.json"
    write_json_atomic(report_path, report)

    print(f"Wrote verification report: {report_path}")
    if mismatches:
        print(f"Verification failed with {len(mismatches)} mismatches")
    if merkle_issues:
        print(f"Merkle binding failed with {len(merkle_issues)} issue(s): manifest rows do not reproduce {args.merkle_roots}")
    if summary_issues:
        print(
            f"Hash summary binding failed with {len(summary_issues)} issue(s): manifest rows disagree with {args.hash_summary}"
        )
    if mismatches or merkle_issues or summary_issues:
        return 1

    if args.merkle_roots:
        print(f"Merkle binding verified: {len(all_rows)} manifest rows reproduce {args.merkle_roots}")
    print("Verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
