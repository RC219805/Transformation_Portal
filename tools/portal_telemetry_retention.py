#!/usr/bin/env python3
"""Review and delete portal telemetry raw JSONL sinks with audit evidence."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA = "portal-telemetry-retention-evidence/v1"
CONFIRM_DELETE = "DELETE-PORTAL-TELEMETRY-RAW-LOGS"
RETENTION_WINDOW_DAYS = 14
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_STATIC_PARTS = {"public", "static"}
GLOB_CHARS = set("*?[]{}")
APPROVED_SINK_SUFFIXES = (".jsonl", ".jsonl.gz")


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _mtime_timestamp(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_pilot_end_date(value: str) -> date:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise argparse.ArgumentTypeError("must use YYYY-MM-DD")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a valid calendar date") from exc


def _has_glob_chars(value: str) -> bool:
    return any(char in GLOB_CHARS for char in value)


def _has_approved_sink_suffix(path: Path) -> bool:
    path_text = str(path)
    return any(path_text.endswith(suffix) for suffix in APPROVED_SINK_SUFFIXES)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _path_policy(path: Path) -> Dict[str, Any]:
    resolved = path.resolve(strict=False)
    project_root = PROJECT_ROOT.resolve(strict=False)
    inside_repo = _is_relative_to(resolved, project_root)
    inside_public_or_static = False
    if inside_repo:
        relative_parts = resolved.relative_to(project_root).parts
        inside_public_or_static = any(part in PUBLIC_STATIC_PARTS for part in relative_parts)

    warning: Optional[str] = None
    if inside_public_or_static:
        warning = "raw logs should not be stored inside repository public/static paths"
    elif inside_repo:
        warning = "raw logs should not be stored inside the repository"

    return {
        "inside_public_or_static": inside_public_or_static,
        "inside_repo": inside_repo,
        "warning": warning,
    }


def _validate_sink_path(raw_path: str) -> Path:
    candidate = raw_path.strip()
    if not candidate:
        raise ValueError("sink paths must not be empty")
    if _has_glob_chars(candidate):
        raise ValueError(f"sink paths must not contain glob characters: {raw_path}")

    path = Path(candidate)
    if not path.is_absolute():
        raise ValueError(f"sink paths must be absolute: {raw_path}")
    if not _has_approved_sink_suffix(path):
        raise ValueError("sink paths must end with .jsonl or .jsonl.gz")
    if path.is_symlink():
        raise ValueError(f"sink paths must not be symlinks: {raw_path}")
    if path.exists():
        if path.is_dir():
            raise ValueError(f"sink paths must not be directories: {raw_path}")
        if not path.is_file():
            raise ValueError(f"sink paths must be regular files: {raw_path}")
    return path


def _validate_evidence_path(evidence_out: str, sink_paths: List[Path]) -> Path:
    candidate = evidence_out.strip()
    if not candidate:
        raise ValueError("--evidence-out must not be empty")
    if _has_glob_chars(candidate):
        raise ValueError("--evidence-out must not contain glob characters")

    path = Path(candidate)
    if not path.is_absolute():
        raise ValueError("--evidence-out must be absolute")
    if path.is_symlink():
        raise ValueError(f"evidence output must not be a symlink: {evidence_out}")
    if path.exists() and path.is_dir():
        raise ValueError(f"evidence output must not be a directory: {evidence_out}")
    resolved_evidence = path.resolve(strict=False)
    for sink_path in sink_paths:
        if sink_path.resolve(strict=False) == resolved_evidence:
            raise ValueError("--evidence-out must not match a --sink-path")
    return path


def _preflight_evidence_output(path: Path) -> None:
    probe_path: Optional[str] = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            delete=False,
            dir=str(path.parent),
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".write-probe",
        ) as handle:
            probe_path = handle.name
            handle.write("{}\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise RuntimeError(f"evidence output is not writable: {path}: {exc}") from exc
    finally:
        if probe_path is not None:
            try:
                Path(probe_path).unlink()
            except OSError:
                pass


def _write_evidence_atomic(path: Path, evidence: Dict[str, Any]) -> None:
    payload = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            delete=False,
            dir=str(path.parent),
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
        ) as handle:
            tmp_path = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        Path(tmp_path).replace(path)
    except OSError as exc:
        if tmp_path is not None:
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass
        raise RuntimeError(f"failed to write evidence atomically: {path}: {exc}") from exc


def _sink_record(path: Path, *, delete: bool, retention_deadline: date, generated_date: date) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "deleted": False,
        "deleted_at": None,
        "delete_error": None,
        "deletion_attempted": False,
        "exists": False,
        "mtime": None,
        "path": str(path),
        "path_policy": _path_policy(path),
        "present_at_review": False,
        "retention_deadline_passed": generated_date > retention_deadline,
        "size_bytes": None,
    }

    try:
        stat_result = path.stat()
    except FileNotFoundError:
        return record
    except OSError as exc:
        record["delete_error"] = f"stat failed: {exc}"
        return record

    record.update(
        {
            "exists": True,
            "mtime": _mtime_timestamp(stat_result.st_mtime),
            "present_at_review": True,
            "size_bytes": stat_result.st_size,
        }
    )

    if delete:
        record["deletion_attempted"] = True
        try:
            path.unlink()
        except OSError as exc:
            record["delete_error"] = str(exc)
        else:
            record["deleted"] = True
            record["deleted_at"] = _utc_timestamp()

    return record


def _build_evidence(
    *,
    mode: str,
    pilot_owner: str,
    pilot_end_date: date,
    reviewer: str,
    sink_paths: List[Path],
) -> Dict[str, Any]:
    delete = mode == "delete"
    generated_at = _utc_timestamp()
    generated_date = date.fromisoformat(generated_at[:10])
    retention_deadline = pilot_end_date + timedelta(days=RETENTION_WINDOW_DAYS)
    sink_records = [
        _sink_record(
            path,
            delete=delete,
            generated_date=generated_date,
            retention_deadline=retention_deadline,
        )
        for path in sink_paths
    ]
    bytes_seen = sum(int(record["size_bytes"] or 0) for record in sink_records)
    bytes_deleted = sum(int(record["size_bytes"] or 0) for record in sink_records if record["deleted"])

    return {
        "generated_at": generated_at,
        "mode": mode,
        "pilot_end_date": pilot_end_date.isoformat(),
        "pilot_owner": pilot_owner,
        "retention_deadline": retention_deadline.isoformat(),
        "retention_window_days": RETENTION_WINDOW_DAYS,
        "reviewer": reviewer,
        "schema": SCHEMA,
        "sink_paths": sink_records,
        "summary": {
            "bytes_deleted": bytes_deleted,
            "bytes_seen": bytes_seen,
            "paths_deleted": sum(1 for record in sink_records if record["deleted"]),
            "paths_existing": sum(1 for record in sink_records if record["exists"]),
            "paths_seen": len(sink_records),
        },
    }


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review portal telemetry raw JSONL retention and write deletion evidence.")
    parser.add_argument("--pilot-owner", required=True, help="Named pilot owner.")
    parser.add_argument(
        "--pilot-end-date",
        required=True,
        type=_parse_pilot_end_date,
        help="Pilot end date in YYYY-MM-DD format.",
    )
    parser.add_argument("--reviewer", required=True, help="Reviewer or approver for the retention action.")
    parser.add_argument(
        "--sink-path",
        action="append",
        required=True,
        help="Absolute raw JSONL sink path approved for retention review. Repeat for multiple sinks.",
    )
    parser.add_argument("--evidence-out", required=True, help="Path where deterministic JSON evidence will be written.")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--dry-run", action="store_true", help="Write evidence without deleting raw logs.")
    mode_group.add_argument("--delete", action="store_true", help="Delete existing raw logs and write evidence.")
    parser.add_argument(
        "--confirm-delete",
        help=f"Required with --delete. Must equal {CONFIRM_DELETE}.",
    )
    args = parser.parse_args(argv)

    for attr, flag in (
        ("pilot_owner", "--pilot-owner"),
        ("reviewer", "--reviewer"),
        ("evidence_out", "--evidence-out"),
    ):
        value = str(getattr(args, attr) or "").strip()
        if not value:
            parser.error(f"{flag} must not be empty")
        setattr(args, attr, value)

    if args.delete and args.confirm_delete != CONFIRM_DELETE:
        parser.error(f"--delete requires --confirm-delete {CONFIRM_DELETE}")

    sink_paths: List[Path] = []
    try:
        for raw_sink_path in args.sink_path:
            sink_paths.append(_validate_sink_path(raw_sink_path))
        args.evidence_out_path = _validate_evidence_path(args.evidence_out, sink_paths)
    except ValueError as exc:
        parser.error(str(exc))
    args.sink_paths = sink_paths
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    mode = "delete" if args.delete else "dry-run"
    evidence_path: Path = args.evidence_out_path
    try:
        _preflight_evidence_output(evidence_path)
    except RuntimeError as exc:
        print(f"portal telemetry retention: {exc}", file=sys.stderr)
        return 1

    evidence = _build_evidence(
        mode=mode,
        pilot_owner=args.pilot_owner,
        pilot_end_date=args.pilot_end_date,
        reviewer=args.reviewer,
        sink_paths=args.sink_paths,
    )

    try:
        _write_evidence_atomic(evidence_path, evidence)
    except RuntimeError as exc:
        print(f"portal telemetry retention: {exc}", file=sys.stderr)
        return 1
    print(f"wrote portal telemetry retention evidence to {evidence_path}")

    if any(record.get("delete_error") for record in evidence["sink_paths"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
