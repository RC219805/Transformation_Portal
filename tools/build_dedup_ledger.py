#!/usr/bin/env python3
"""Build checksum-confirmed deduplication planning ledger from manifest v2 JSONL."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from archive_governance_common import atomic_write_text, deterministic_json_dumps

EXIT_SUCCESS = 0
EXIT_INPUT_ERROR = 2

_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_RAW_EXTENSIONS = {
    ".arw",
    ".cr2",
    ".cr3",
    ".dng",
    ".nef",
    ".orf",
    ".pef",
    ".raf",
    ".rw2",
    ".srw",
}



def _load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
                raise ValueError(f"Manifest line {line_number} must be a JSON object")
            rows.append(payload)
    return rows


def _metadata_score(entry: dict[str, Any]) -> int:
    score = 0
    extension = str(entry.get("extension") or "").lower()
    if extension in _RAW_EXTENSIONS:
        score += 20
    rights_flags = entry.get("rights_flags")
    if isinstance(rights_flags, list) and rights_flags and rights_flags != ["unspecified"]:
        score += 10
    if entry.get("created_utc"):
        score += 2
    if entry.get("modified_utc"):
        score += 2
    if entry.get("accessed_utc"):
        score += 1
    return score


def _canonical_choice(entries: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        entries,
        key=lambda entry: (
            -_metadata_score(entry),
            len(str(entry.get("relpath") or "")),
            str(entry.get("relpath") or ""),
        ),
    )[0]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-jsonl", required=True, help="Input archive_manifest_v2.jsonl path")
    parser.add_argument("--out-ledger", required=True, help="Output CSV ledger path")
    parser.add_argument("--out-summary", required=True, help="Output summary JSON path")
    parser.add_argument("--approver", default="UNSPECIFIED", help="Decision approver label")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        rows = _load_manifest(Path(args.manifest_jsonl))
    except ValueError as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sha256 = str(row.get("sha256") or "")
        hash_status = str(row.get("hash_status") or "")
        if hash_status != "ok":
            continue
        if _SHA256_RE.fullmatch(sha256) is None:
            continue
        groups[sha256].append(row)

    decision_date = datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")
    ledger_rows: list[dict[str, Any]] = []

    duplicate_groups = 0
    duplicate_excess_files = 0
    for sha256, group in sorted(groups.items(), key=lambda item: item[0]):
        if len(group) <= 1:
            continue
        duplicate_groups += 1
        duplicate_excess_files += len(group) - 1

        canonical = _canonical_choice(group)
        canonical_path = str(canonical.get("relpath") or "")
        duplicate_paths = sorted(
            str(entry.get("relpath") or "")
            for entry in group
            if str(entry.get("relpath") or "") != canonical_path
        )

        ledger_rows.append(
            {
                "sha256": sha256,
                "canonical_path": canonical_path,
                "duplicate_paths": "|".join(duplicate_paths),
                "duplicate_count": len(group),
                "decision_reason": "highest_metadata_completeness_then_shortest_path",
                "date_utc": decision_date,
                "approver": args.approver,
            }
        )

    ledger_columns = [
        "sha256",
        "canonical_path",
        "duplicate_paths",
        "duplicate_count",
        "decision_reason",
        "date_utc",
        "approver",
    ]

    out_ledger = Path(args.out_ledger)
    out_ledger.parent.mkdir(parents=True, exist_ok=True)
    with out_ledger.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ledger_columns, lineterminator="\n")
        writer.writeheader()
        for row in ledger_rows:
            writer.writerow(row)

    summary_payload = {
        "schema_version": "tp.archive.dedup.summary.v1",
        "input_rows": len(rows),
        "hashable_rows": sum(1 for row in rows if _SHA256_RE.fullmatch(str(row.get("sha256") or "")) is not None),
        "duplicate_groups": duplicate_groups,
        "duplicate_excess_files": duplicate_excess_files,
        "ledger_rows": len(ledger_rows),
        "decision_date_utc": decision_date,
    }
    atomic_write_text(Path(args.out_summary), deterministic_json_dumps(summary_payload, pretty=True) + "\n")

    print(f"Wrote dedup ledger to {out_ledger}")
    print(f"Wrote dedup summary to {args.out_summary}")
    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
