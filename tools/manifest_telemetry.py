"""Deterministic telemetry and governance utilities for Montecito manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

UNCLASSIFIED_TOKENS = {"", "unknown", "unclassified", "none", "null", "na", "n/a"}


@dataclass(frozen=True)
class ManifestRow:
    """Normalized manifest row used for telemetry and Merkle derivations."""

    filename: str
    size_bytes: int
    digest: str


def _sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _digest_algorithm_for_key(digest_key: str) -> str:
    normalized = digest_key.strip().lower()
    if normalized in {"md5", "sha256"}:
        return normalized
    return "unspecified"


def _read_manifest(manifest_csv: Path) -> tuple[list[ManifestRow], str, str]:
    with manifest_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit(f"Manifest '{manifest_csv}' does not contain headers.")

        filename_key = _pick_key(reader.fieldnames, ("filename", "path", "file", "relative_path"))
        bytes_key = _pick_key(reader.fieldnames, ("bytes", "size_bytes", "size"))
        digest_key = _pick_key(reader.fieldnames, ("md5", "sha256", "hash"))
        digest_algorithm = _digest_algorithm_for_key(digest_key)

        rows: list[ManifestRow] = []
        for raw in reader:
            filename = (raw.get(filename_key) or "").strip()
            if not filename:
                continue

            size_raw = (raw.get(bytes_key) or "0").strip()
            digest = (raw.get(digest_key) or "").strip().lower()

            try:
                size_bytes = int(size_raw)
            except ValueError as exc:
                raise SystemExit(f"Manifest row has non-integer size '{size_raw}' for '{filename}'.") from exc

            rows.append(ManifestRow(filename=filename, size_bytes=size_bytes, digest=digest))

    rows.sort(key=lambda row: row.filename)
    return rows, digest_key, digest_algorithm


def _pick_key(fieldnames: list[str], candidates: tuple[str, ...]) -> str:
    lowered = {field.lower(): field for field in fieldnames}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    raise SystemExit(f"Missing required columns. Need one of: {', '.join(candidates)}.")


def write_metrics(manifest_csv: Path, out_json: Path, out_audit_csv: Path) -> None:
    rows, digest_column, digest_algorithm = _read_manifest(manifest_csv)
    total_bytes = sum(row.size_bytes for row in rows)

    extension_stats: dict[str, dict[str, int]] = {}
    drive_stats: dict[str, dict[str, int]] = {}
    digest_counts: dict[str, int] = {}

    audit_rows: list[dict[str, str | int]] = []
    for row in rows:
        extension = Path(row.filename).suffix.lower() or "(none)"
        drive = row.filename.split("/", 1)[0] if "/" in row.filename else "."

        extension_stats.setdefault(extension, {"files": 0, "bytes": 0})
        extension_stats[extension]["files"] += 1
        extension_stats[extension]["bytes"] += row.size_bytes

        drive_stats.setdefault(drive, {"files": 0, "bytes": 0})
        drive_stats[drive]["files"] += 1
        drive_stats[drive]["bytes"] += row.size_bytes

        if row.digest:
            digest_counts[row.digest] = digest_counts.get(row.digest, 0) + 1

        audit_rows.append(
            {
                "filename": row.filename,
                "bytes": row.size_bytes,
                "digest": row.digest,
                "extension": extension,
                "top_level_dir": drive,
            }
        )

    duplicate_groups = sum(1 for count in digest_counts.values() if count > 1)
    duplicate_file_excess = sum(count - 1 for count in digest_counts.values() if count > 1)

    metrics_payload: dict[str, Any] = {
        "schema_version": "1.0",
        "manifest_name": manifest_csv.name,
        "digest_column": digest_column,
        "digest_algorithm": digest_algorithm,
        "totals": {"files": len(rows), "bytes": total_bytes},
        "duplicates": {"groups": duplicate_groups, "excess_files": duplicate_file_excess},
        "extensions": [
            {"extension": key, "files": value["files"], "bytes": value["bytes"]}
            for key, value in sorted(extension_stats.items(), key=lambda pair: pair[0])
        ],
        "top_level_dirs": [
            {"drive": key, "files": value["files"], "bytes": value["bytes"]}
            for key, value in sorted(drive_stats.items(), key=lambda pair: pair[0])
        ],
        "manifest_content_sha256": _sha256_hex("\n".join(f"{row.filename}\t{row.size_bytes}\t{row.digest}" for row in rows)),
    }

    _write_json(out_json, metrics_payload)
    _write_csv(
        out_audit_csv,
        fieldnames=["filename", "bytes", "digest", "extension", "top_level_dir"],
        rows=audit_rows,
    )


def _merkle_root(hex_leaves: list[str]) -> str:
    if not hex_leaves:
        return _sha256_hex("")

    level = list(hex_leaves)
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        level = [_sha256_hex(f"{level[index]}{level[index + 1]}") for index in range(0, len(level), 2)]
    return level[0]


def write_merkle(manifest_csv: Path, out_json: Path) -> None:
    rows, _, _ = _read_manifest(manifest_csv)
    leaf_hashes = [_sha256_hex(f"{row.filename}\n{row.size_bytes}\n{row.digest}") for row in rows]

    per_drive_leaves: dict[str, list[str]] = {}
    for row in rows:
        drive = row.filename.split("/", 1)[0] if "/" in row.filename else "."
        leaf = _sha256_hex(f"{row.filename}\n{row.size_bytes}\n{row.digest}")
        per_drive_leaves.setdefault(drive, []).append(leaf)

    payload = {
        "schema_version": "1.0",
        "manifest_name": manifest_csv.name,
        "leaf_count": len(leaf_hashes),
        "global_root": _merkle_root(leaf_hashes),
        "per_drive_roots": [
            {
                "drive": drive,
                "leaf_count": len(leaves),
                "root": _merkle_root(leaves),
            }
            for drive, leaves in sorted(per_drive_leaves.items(), key=lambda pair: pair[0])
        ],
    }
    _write_json(out_json, payload)


def _classification_column(fieldnames: list[str]) -> str | None:
    lowered = {field.lower(): field for field in fieldnames}
    for candidate in (
        "classification",
        "rights_classification",
        "privacy_classification",
        "governance_status",
        "status",
    ):
        if candidate in lowered:
            return lowered[candidate]
    return None


def _governance_metrics(governance_csv: Path) -> dict[str, Any]:
    with governance_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit(f"Governance CSV '{governance_csv}' does not contain headers.")

        classification_key = _classification_column(reader.fieldnames)
        rows = list(reader)

    total_rows = len(rows)
    if total_rows == 0:
        return {
            "schema_version": "1.0",
            "governance_csv": str(governance_csv),
            "rows_total": 0,
            "classified_rows": 0,
            "unclassified_rows": 0,
            "coverage_percent": 0.0,
            "classification_column": classification_key,
        }

    classified = 0
    for row in rows:
        if classification_key is None:
            candidate = ""
        else:
            candidate = (row.get(classification_key) or "").strip().lower()
        if candidate not in UNCLASSIFIED_TOKENS:
            classified += 1

    coverage = round((classified / total_rows) * 100.0, 2)
    return {
        "schema_version": "1.0",
        "governance_csv": str(governance_csv),
        "rows_total": total_rows,
        "classified_rows": classified,
        "unclassified_rows": total_rows - classified,
        "coverage_percent": coverage,
        "classification_column": classification_key,
    }


def write_governance_metrics(governance_csv: Path, out_json: Path) -> None:
    payload = _governance_metrics(governance_csv)
    _write_json(out_json, payload)


def run_governance_gate(governance_csv: Path, min_classified: int, out_json: Path) -> None:
    payload = _governance_metrics(governance_csv)
    payload["min_classified_required"] = min_classified
    payload["passed"] = payload["classified_rows"] >= min_classified
    _write_json(out_json, payload)
    if not payload["passed"]:
        raise SystemExit(
            "Governance completeness gate failed: "
            f"classified_rows={payload['classified_rows']} "
            f"< required={min_classified}"
        )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: str(item["filename"])):
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manifest telemetry and governance utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    metrics = subparsers.add_parser("metrics", help="Emit deterministic manifest metrics and an audit CSV.")
    metrics.add_argument("--manifest", type=Path, required=True, help="Input manifest CSV.")
    metrics.add_argument("--out-json", type=Path, required=True, help="Destination metrics JSON.")
    metrics.add_argument("--out-audit-csv", type=Path, required=True, help="Destination audit extract CSV.")

    merkle = subparsers.add_parser("merkle", help="Emit global and per-drive Merkle roots for the manifest.")
    merkle.add_argument("--manifest", type=Path, required=True, help="Input manifest CSV.")
    merkle.add_argument("--out-json", type=Path, required=True, help="Destination Merkle JSON.")

    governance_metrics = subparsers.add_parser(
        "governance-metrics",
        help="Emit governance classification coverage metrics.",
    )
    governance_metrics.add_argument("--governance-csv", type=Path, required=True, help="Input governance CSV.")
    governance_metrics.add_argument("--out-json", type=Path, required=True, help="Destination JSON path.")

    governance_gate = subparsers.add_parser(
        "governance-gate",
        help="Enforce minimum classified rows in governance CSV.",
    )
    governance_gate.add_argument("--governance-csv", type=Path, required=True, help="Input governance CSV.")
    governance_gate.add_argument(
        "--min-classified",
        type=int,
        required=True,
        help="Minimum number of classified rows required to pass.",
    )
    governance_gate.add_argument("--out-json", type=Path, required=True, help="Destination gate report JSON.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    command = args.command

    if command == "metrics":
        write_metrics(args.manifest, args.out_json, args.out_audit_csv)
    elif command == "merkle":
        write_merkle(args.manifest, args.out_json)
    elif command == "governance-metrics":
        write_governance_metrics(args.governance_csv, args.out_json)
    elif command == "governance-gate":
        run_governance_gate(args.governance_csv, args.min_classified, args.out_json)
    else:
        raise SystemExit(f"Unsupported command '{command}'.")


if __name__ == "__main__":
    main()
