#!/usr/bin/env python3
"""Deterministic BagIt build/validate tool for archive governance."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from archive_governance_common import atomic_write_text, deterministic_json_dumps

EXIT_SUCCESS = 0
EXIT_INPUT_ERROR = 2
EXIT_BUILD_ERROR = 3
EXIT_VALIDATE_ERROR = 4

READ_CHUNK_BYTES = 1024 * 1024


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


def _check_symlink_components(root: Path, rel_path_obj: Path) -> str:
    current = root
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


def _resolves_within_root(root: Path, candidate: Path) -> bool:
    try:
        root_resolved = root.resolve(strict=False)
        candidate_resolved = candidate.resolve(strict=False)
    except OSError:
        return True

    try:
        candidate_resolved.relative_to(root_resolved)
    except ValueError:
        return False
    return True


def _sha256_for_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(READ_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Manifest line {line_number} must be object")
            entries.append(payload)
    return entries


def _derive_bagging_date(entries: list[dict[str, Any]]) -> str:
    """Derive deterministic bagging date from manifest timestamps."""

    candidates: list[datetime] = []
    for entry in entries:
        for field in ("modified_utc", "created_utc", "accessed_utc"):
            value = entry.get(field)
            if not isinstance(value, str) or not value:
                continue
            normalized = value.replace("Z", "+00:00")
            try:
                parsed = datetime.fromisoformat(normalized)
            except ValueError:
                continue
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            else:
                parsed = parsed.astimezone(UTC)
            candidates.append(parsed)

    if not candidates:
        return "1970-01-01"

    return max(candidates).date().isoformat()


def _bagit_txt() -> str:
    return "BagIt-Version: 0.97\nTag-File-Character-Encoding: UTF-8\n"


def _write_manifest_file(path: Path, rows: list[tuple[str, str]]) -> None:
    content = "".join(f"{digest}  {relpath}\n" for digest, relpath in rows)
    atomic_write_text(path, content)


def _parse_manifest_file(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            try:
                digest, relpath = stripped.split("  ", 1)
            except ValueError as exc:
                raise ValueError(f"Invalid manifest line {line_number} in {path.name}: {stripped!r}") from exc
            rows.append((digest.strip(), relpath.strip()))
    return rows


def _bagit_validate_with_library(bag_dir: Path) -> tuple[bool, str | None]:
    try:
        import bagit  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency path
        return False, f"bagit library unavailable: {exc}"

    try:
        bag = bagit.Bag(str(bag_dir))
        valid = bool(bag.validate())
        return valid, None if valid else "bagit library reported invalid bag"
    except Exception as exc:  # pragma: no cover - optional dependency path
        return False, str(exc)


def _build_bag(
    *,
    manifest_path: Path,
    archive_root: Path,
    bag_dir: Path,
    report_json: Path | None,
    source_organization: str,
    validate_with_bagit_python: bool,
) -> int:
    try:
        entries = _load_manifest(manifest_path)
    except ValueError as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    payload_entries = [entry for entry in entries if str(entry.get("hash_status") or "") == "ok"]
    payload_entries = sorted(payload_entries, key=lambda row: str(row.get("relpath") or ""))
    bagging_date = _derive_bagging_date(payload_entries)

    if bag_dir.exists():
        if not bag_dir.is_dir():
            print(f"Build error: bag_dir is not a directory: {bag_dir}", file=sys.stderr)
            return EXIT_BUILD_ERROR
        if any(bag_dir.iterdir()):
            print(f"Build error: bag_dir must be empty before build: {bag_dir}", file=sys.stderr)
            return EXIT_BUILD_ERROR
    else:
        bag_dir.mkdir(parents=True, exist_ok=False)

    data_dir = bag_dir / "data"
    data_dir.mkdir(parents=False, exist_ok=False)
    archive_root_resolved = archive_root.resolve(strict=False)

    manifest_rows: list[tuple[str, str]] = []
    copied_files = 0
    payload_bytes = 0

    for entry in payload_entries:
        relpath = str(entry.get("relpath") or "")
        expected_sha = str(entry.get("sha256") or "")
        relpath_obj, relpath_error = _materialize_relpath(relpath)
        if relpath_obj is None:
            print(f"Build error: invalid relpath for relpath={relpath}: {relpath_error}", file=sys.stderr)
            return EXIT_BUILD_ERROR

        symlink_component_error = _check_symlink_components(archive_root, relpath_obj)
        if symlink_component_error:
            print(
                f"Build error: source payload rejected for relpath={relpath}: {symlink_component_error}",
                file=sys.stderr,
            )
            return EXIT_BUILD_ERROR

        source_path = archive_root / relpath_obj
        destination = data_dir / relpath_obj
        if not _resolves_within_root(data_dir, destination):
            print(f"Build error: destination escapes bag data directory for relpath={relpath}", file=sys.stderr)
            return EXIT_BUILD_ERROR
        destination.parent.mkdir(parents=True, exist_ok=True)

        try:
            source_resolved = source_path.resolve(strict=True)
        except FileNotFoundError:
            print(f"Build error: source payload missing for relpath={relpath}", file=sys.stderr)
            return EXIT_BUILD_ERROR
        except PermissionError:
            print(f"Build error: source payload unreadable for relpath={relpath}", file=sys.stderr)
            return EXIT_BUILD_ERROR
        except OSError:
            print(f"Build error: source payload stat failed for relpath={relpath}", file=sys.stderr)
            return EXIT_BUILD_ERROR

        if source_path.is_symlink():
            print(f"Build error: source payload symlink not allowed for relpath={relpath}", file=sys.stderr)
            return EXIT_BUILD_ERROR
        if not source_resolved.is_file():
            print(f"Build error: source payload missing for relpath={relpath}", file=sys.stderr)
            return EXIT_BUILD_ERROR
        try:
            source_resolved.relative_to(archive_root_resolved)
        except ValueError:
            print(f"Build error: source payload outside archive_root for relpath={relpath}", file=sys.stderr)
            return EXIT_BUILD_ERROR

        shutil.copy2(source_resolved, destination)
        observed_sha = _sha256_for_path(destination)
        if expected_sha and observed_sha != expected_sha:
            print(
                f"Build error: sha256 mismatch after copy for relpath={relpath}: expected={expected_sha} observed={observed_sha}",
                file=sys.stderr,
            )
            return EXIT_BUILD_ERROR

        payload_bytes += int(destination.stat().st_size)
        copied_files += 1
        manifest_rows.append((observed_sha, f"data/{relpath_obj.as_posix()}"))

    bagit_path = bag_dir / "bagit.txt"
    bag_info_path = bag_dir / "bag-info.txt"
    payload_manifest_path = bag_dir / "manifest-sha256.txt"
    tag_manifest_path = bag_dir / "tagmanifest-sha256.txt"

    atomic_write_text(bagit_path, _bagit_txt())

    payload_oxum = f"{payload_bytes}.{copied_files}"
    bag_info = "".join(
        [
            f"Source-Organization: {source_organization}\n",
            f"Bagging-Date: {bagging_date}\n",
            f"Payload-Oxum: {payload_oxum}\n",
        ]
    )
    atomic_write_text(bag_info_path, bag_info)
    _write_manifest_file(payload_manifest_path, manifest_rows)

    tag_rows = [
        (_sha256_for_path(bagit_path), "bagit.txt"),
        (_sha256_for_path(bag_info_path), "bag-info.txt"),
        (_sha256_for_path(payload_manifest_path), "manifest-sha256.txt"),
    ]
    _write_manifest_file(tag_manifest_path, tag_rows)

    library_validation: dict[str, Any] | None = None
    if validate_with_bagit_python:
        valid, message = _bagit_validate_with_library(bag_dir)
        library_validation = {
            "requested": True,
            "valid": valid,
            "message": message,
        }
        if not valid:
            print(f"Build error: bagit-python validation failed: {message}", file=sys.stderr)
            return EXIT_BUILD_ERROR

    report_payload: dict[str, Any] = {
        "schema_version": "tp.archive.bagit.build_report.v1",
        "bag_dir": str(bag_dir),
        "source_manifest": str(manifest_path),
        "archive_root": str(archive_root),
        "copied_files": copied_files,
        "payload_bytes": payload_bytes,
        "payload_oxum": payload_oxum,
        "payload_manifest": str(payload_manifest_path),
        "tag_manifest": str(tag_manifest_path),
    }
    if library_validation is not None:
        report_payload["bagit_python_validation"] = library_validation

    if report_json is not None:
        atomic_write_text(report_json, deterministic_json_dumps(report_payload, pretty=True) + "\n")

    print(f"Built deterministic bag at {bag_dir}")
    if report_json is not None:
        print(f"Wrote build report to {report_json}")
    return EXIT_SUCCESS


def _validate_bag(
    *,
    bag_dir: Path,
    report_json: Path,
    validate_with_bagit_python: bool,
) -> int:
    bagit_path = bag_dir / "bagit.txt"
    bag_info_path = bag_dir / "bag-info.txt"
    payload_manifest_path = bag_dir / "manifest-sha256.txt"
    tag_manifest_path = bag_dir / "tagmanifest-sha256.txt"

    missing_required = [
        str(path) for path in (bagit_path, bag_info_path, payload_manifest_path, tag_manifest_path) if not path.exists()
    ]
    if missing_required:
        report_payload = {
            "schema_version": "tp.archive.bagit.validation_report.v1",
            "bag_dir": str(bag_dir),
            "valid": False,
            "missing_required": missing_required,
        }
        atomic_write_text(report_json, deterministic_json_dumps(report_payload, pretty=True) + "\n")
        print("Validation failed: required BagIt files missing", file=sys.stderr)
        return EXIT_VALIDATE_ERROR

    mismatches: list[dict[str, Any]] = []

    try:
        payload_rows = _parse_manifest_file(payload_manifest_path)
        tag_rows = _parse_manifest_file(tag_manifest_path)
    except ValueError as exc:
        print(f"Validation error: {exc}", file=sys.stderr)
        return EXIT_VALIDATE_ERROR

    def _validate_rows(scope: str, rows: list[tuple[str, str]]) -> None:
        for expected_sha, relpath in rows:
            relpath_obj, relpath_error = _materialize_relpath(relpath)
            if relpath_obj is None:
                mismatches.append({"scope": scope, "path": relpath, "issue": relpath_error})
                continue

            symlink_component_error = _check_symlink_components(bag_dir, relpath_obj)
            if symlink_component_error:
                mismatches.append({"scope": scope, "path": relpath, "issue": symlink_component_error})
                continue

            target = bag_dir / relpath_obj
            if not _resolves_within_root(bag_dir, target):
                mismatches.append({"scope": scope, "path": relpath, "issue": "outside_bag_root"})
                continue

            if not target.exists() or not target.is_file():
                mismatches.append({"scope": scope, "path": relpath, "issue": "missing"})
                continue
            if target.is_symlink():
                mismatches.append({"scope": scope, "path": relpath, "issue": "symlink_skipped"})
                continue

            observed_sha = _sha256_for_path(target)
            if observed_sha != expected_sha:
                mismatches.append(
                    {
                        "scope": scope,
                        "path": relpath,
                        "issue": "sha256_mismatch",
                        "expected": expected_sha,
                        "observed": observed_sha,
                    }
                )

    _validate_rows("payload", payload_rows)
    _validate_rows("tag", tag_rows)

    library_validation: dict[str, Any] | None = None
    if validate_with_bagit_python:
        valid, message = _bagit_validate_with_library(bag_dir)
        library_validation = {
            "requested": True,
            "valid": valid,
            "message": message,
        }
        if not valid:
            mismatches.append(
                {
                    "scope": "bagit_python",
                    "path": str(bag_dir),
                    "issue": "library_validation_failed",
                    "message": message,
                }
            )

    report_payload = {
        "schema_version": "tp.archive.bagit.validation_report.v1",
        "bag_dir": str(bag_dir),
        "valid": len(mismatches) == 0,
        "payload_rows": len(payload_rows),
        "tag_rows": len(tag_rows),
        "mismatches": mismatches,
    }
    if library_validation is not None:
        report_payload["bagit_python_validation"] = library_validation

    atomic_write_text(report_json, deterministic_json_dumps(report_payload, pretty=True) + "\n")

    if mismatches:
        print(f"Validation failed with {len(mismatches)} mismatches", file=sys.stderr)
        return EXIT_VALIDATE_ERROR

    print(f"Bag validated: {bag_dir}")
    print(f"Validation report written to {report_json}")
    return EXIT_SUCCESS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_build = subparsers.add_parser("build", help="Build deterministic BagIt package")
    parser_build.add_argument("--manifest-jsonl", required=True, help="Input archive manifest v2 JSONL")
    parser_build.add_argument("--archive-root", required=True, help="Archive root for payload sources")
    parser_build.add_argument("--bag-dir", required=True, help="Bag output directory")
    parser_build.add_argument("--report-json", default=None, help="Optional build report JSON path")
    parser_build.add_argument("--source-organization", default="UNSPECIFIED", help="Bag-Info Source-Organization")
    parser_build.add_argument(
        "--validate-with-bagit-python",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run optional bagit-python validation after build",
    )

    parser_validate = subparsers.add_parser("validate", help="Validate deterministic BagIt package")
    parser_validate.add_argument("--bag-dir", required=True, help="Bag directory")
    parser_validate.add_argument("--report-json", required=True, help="Validation report JSON path")
    parser_validate.add_argument(
        "--validate-with-bagit-python",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run optional bagit-python validation",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.command == "build":
        return _build_bag(
            manifest_path=Path(args.manifest_jsonl),
            archive_root=Path(args.archive_root),
            bag_dir=Path(args.bag_dir),
            report_json=Path(args.report_json) if args.report_json else None,
            source_organization=args.source_organization,
            validate_with_bagit_python=bool(args.validate_with_bagit_python),
        )

    return _validate_bag(
        bag_dir=Path(args.bag_dir),
        report_json=Path(args.report_json),
        validate_with_bagit_python=bool(args.validate_with_bagit_python),
    )


if __name__ == "__main__":
    raise SystemExit(main())
