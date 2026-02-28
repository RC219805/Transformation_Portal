#!/usr/bin/env python3
"""Generate deterministic METS fileSec + structMap from archive manifest v2."""

from __future__ import annotations

import argparse
import html
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from archive_governance_common import atomic_write_text, deterministic_json_dumps

EXIT_SUCCESS = 0
EXIT_INPUT_ERROR = 2

_RAW_EXTENSIONS = {
    ".arw",
    ".cr2",
    ".cr3",
    ".dng",
    ".mov",
    ".mp4",
    ".nef",
    ".orf",
    ".raf",
    ".rw2",
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
                raise ValueError(f"Manifest line {line_number} must be object")
            rows.append(payload)
    return rows


def _escape(value: str) -> str:
    return html.escape(value, quote=True)


def _file_group_for_extension(extension: str) -> str:
    return "preservation_master" if extension.lower() in _RAW_EXTENSIONS else "access_derivative"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-jsonl", required=True, help="Input archive_manifest_v2 JSONL")
    parser.add_argument("--out-xml", required=True, help="Output METS XML path")
    parser.add_argument("--out-summary", required=True, help="Output summary JSON path")
    parser.add_argument("--href-prefix", default="data", help="Path prefix used for FLocat href values")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        manifest_rows = _load_manifest(Path(args.manifest_jsonl))
    except ValueError as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    payload_rows = [row for row in manifest_rows if str(row.get("hash_status") or "") == "ok"]
    payload_rows = sorted(payload_rows, key=lambda row: str(row.get("relpath") or ""))

    file_entries: list[dict[str, str]] = []
    for index, row in enumerate(payload_rows, start=1):
        relpath = str(row.get("relpath") or "")
        extension = str(row.get("extension") or "").lower()
        mime = str(row.get("mime") or "application/octet-stream")
        group = _file_group_for_extension(extension)
        file_id = f"F{index:06d}"
        href = f"{args.href_prefix.rstrip('/')}/{relpath}" if args.href_prefix else relpath
        file_entries.append(
            {
                "id": file_id,
                "group": group,
                "mime": mime,
                "href": href,
                "relpath": relpath,
                "partition": str(row.get("partition") or ""),
            }
        )

    grouped_files: dict[str, list[dict[str, str]]] = defaultdict(list)
    for entry in file_entries:
        grouped_files[entry["group"]].append(entry)

    partition_items: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for entry in file_entries:
        relpath = entry["relpath"]
        partition_items[entry["partition"]][relpath].append(entry["id"])

    collection_id = "UNSPECIFIED"
    if payload_rows:
        collection_id = str(payload_rows[0].get("collection_id") or "UNSPECIFIED")

    lines: list[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<mets:mets xmlns:mets="http://www.loc.gov/METS/" xmlns:xlink="http://www.w3.org/1999/xlink">')
    lines.append("  <mets:fileSec>")
    for group in sorted(grouped_files):
        lines.append(f'    <mets:fileGrp USE="{_escape(group)}">')
        for entry in sorted(grouped_files[group], key=lambda item: item["id"]):
            lines.append(f'      <mets:file ID="{entry["id"]}" MIMETYPE="{_escape(entry["mime"])}">')
            lines.append("        <mets:FLocat " f'LOCTYPE="URL" xlink:href="{_escape(entry["href"])}"/>')
            lines.append("      </mets:file>")
        lines.append("    </mets:fileGrp>")
    lines.append("  </mets:fileSec>")

    lines.append('  <mets:structMap TYPE="physical">')
    lines.append(f'    <mets:div TYPE="collection" LABEL="{_escape(collection_id)}">')
    for partition in sorted(partition_items):
        partition_label = partition or "UNSPECIFIED"
        lines.append(f'      <mets:div TYPE="partition" LABEL="{_escape(partition_label)}">')
        items = partition_items[partition]
        for item_relpath in sorted(items):
            lines.append(f'        <mets:div TYPE="item" LABEL="{_escape(item_relpath)}">')
            for file_id in sorted(set(items[item_relpath])):
                lines.append(f'          <mets:fptr FILEID="{file_id}"/>')
            lines.append("        </mets:div>")
        lines.append("      </mets:div>")
    lines.append("    </mets:div>")
    lines.append("  </mets:structMap>")
    lines.append("</mets:mets>")

    xml_text = "\n".join(lines) + "\n"
    atomic_write_text(Path(args.out_xml), xml_text)

    summary_payload = {
        "schema_version": "tp.archive.mets_export.v1",
        "manifest_rows": len(manifest_rows),
        "payload_rows": len(payload_rows),
        "file_groups": {key: len(value) for key, value in sorted(grouped_files.items())},
        "partitions": {key if key else "UNSPECIFIED": len(value) for key, value in sorted(partition_items.items())},
        "collection_id": collection_id,
        "output_xml": str(Path(args.out_xml)),
    }
    atomic_write_text(Path(args.out_summary), deterministic_json_dumps(summary_payload, pretty=True) + "\n")

    print(f"Wrote METS XML to {args.out_xml}")
    print(f"Wrote METS summary to {args.out_summary}")
    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
