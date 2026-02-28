#!/usr/bin/env python3
"""Export deterministic PROV JSON-LD and optional STAC catalog from manifest v2."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from archive_governance_common import atomic_write_text, deterministic_json_dumps

EXIT_SUCCESS = 0
EXIT_INPUT_ERROR = 2
EXIT_STAC_UNAVAILABLE = 3



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


def _try_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _build_prov_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    payload_rows = [row for row in rows if str(row.get("hash_status") or "") == "ok"]
    payload_rows = sorted(payload_rows, key=lambda row: str(row.get("relpath") or ""))

    activity_id = "urn:tp:archive:activity:manifest_export"
    agent_id = "urn:tp:archive:agent:archive_governance"

    entity_block: dict[str, Any] = {}
    generation_edges: list[dict[str, str]] = []

    for row in payload_rows:
        relpath = str(row.get("relpath") or "")
        provenance_id = str(row.get("provenance_id") or "") or f"sha256:{row.get('sha256', '')}"
        entity_id = f"urn:tp:archive:entity:{provenance_id}"
        entity_block[entity_id] = {
            "prov:type": "prov:Entity",
            "prov:label": relpath,
            "prov:location": relpath,
            "tp:sha256": str(row.get("sha256") or ""),
            "tp:collection_id": str(row.get("collection_id") or "UNSPECIFIED"),
        }
        generation_edges.append(
            {
                "prov:entity": entity_id,
                "prov:activity": activity_id,
            }
        )

    return {
        "@context": {
            "prov": "http://www.w3.org/ns/prov#",
            "tp": "https://schemas.transformation-portal.dev/archive#",
        },
        "entity": entity_block,
        "activity": {
            activity_id: {
                "prov:type": "prov:Activity",
                "prov:label": "Archive manifest export",
                "prov:wasAssociatedWith": agent_id,
            }
        },
        "agent": {
            agent_id: {
                "prov:type": "prov:SoftwareAgent",
                "prov:label": "archive_governance.py",
            }
        },
        "wasGeneratedBy": generation_edges,
    }


def _eligible_for_stac(row: dict[str, Any], *, datetime_field: str) -> tuple[bool, float | None, float | None, str | None]:
    lat = _try_float(row.get("gps_latitude"))
    lon = _try_float(row.get("gps_longitude"))
    dt_value = row.get(datetime_field)
    dt = str(dt_value) if isinstance(dt_value, str) and dt_value.strip() else None
    if lat is None or lon is None or dt is None:
        return False, None, None, None
    return True, lat, lon, dt


def _write_stac_outputs(
    *,
    rows: list[dict[str, Any]],
    catalog_path: Path,
    items_dir: Path,
    datetime_field: str,
) -> tuple[int, int]:
    item_links: list[dict[str, str]] = []
    items_written = 0

    for row in sorted(rows, key=lambda row: str(row.get("relpath") or "")):
        if str(row.get("hash_status") or "") != "ok":
            continue

        eligible, lat, lon, dt = _eligible_for_stac(row, datetime_field=datetime_field)
        if not eligible:
            continue

        relpath = str(row.get("relpath") or "")
        mime = str(row.get("mime") or "application/octet-stream")
        item_id = str(row.get("provenance_id") or "") or Path(relpath).stem or relpath.replace("/", "_")
        item_filename = f"{item_id}.json"

        item_payload = {
            "type": "Feature",
            "stac_version": "1.1.0",
            "id": item_id,
            "properties": {
                "datetime": dt,
                "collection_id": str(row.get("collection_id") or "UNSPECIFIED"),
            },
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat],
            },
            "bbox": [lon, lat, lon, lat],
            "assets": {
                "master": {
                    "href": relpath,
                    "type": mime,
                    "roles": ["data"],
                }
            },
            "links": [
                {
                    "rel": "parent",
                    "href": "../catalog.json",
                    "type": "application/json",
                }
            ],
        }

        items_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(items_dir / item_filename, deterministic_json_dumps(item_payload, pretty=True) + "\n")
        item_links.append(
            {
                "rel": "item",
                "href": f"items/{item_filename}",
                "type": "application/geo+json",
            }
        )
        items_written += 1

    catalog_payload = {
        "type": "Catalog",
        "stac_version": "1.1.0",
        "id": "tp-archive-catalog",
        "description": "Transformation Portal archive STAC catalog",
        "links": [
            {
                "rel": "self",
                "href": "catalog.json",
                "type": "application/json",
            },
            *sorted(item_links, key=lambda link: str(link["href"])),
        ],
    }
    atomic_write_text(catalog_path, deterministic_json_dumps(catalog_payload, pretty=True) + "\n")
    return items_written, len(item_links)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-jsonl", required=True, help="Input archive_manifest_v2 JSONL path")
    parser.add_argument("--out-prov-jsonld", required=True, help="Output PROV JSON-LD path")
    parser.add_argument("--out-stac-catalog", default=None, help="Optional STAC catalog output path")
    parser.add_argument("--out-stac-items-dir", default=None, help="Optional STAC items directory")
    parser.add_argument("--datetime-field", default="modified_utc", help="Datetime field for STAC items")
    parser.add_argument(
        "--require-stac",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail if STAC export requested but no eligible geometry/timestamps",
    )
    parser.add_argument("--out-summary", required=True, help="Output summary JSON path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        rows = _load_manifest(Path(args.manifest_jsonl))
    except ValueError as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    prov_payload = _build_prov_payload(rows)
    atomic_write_text(Path(args.out_prov_jsonld), deterministic_json_dumps(prov_payload, pretty=True) + "\n")

    stac_requested = bool(args.out_stac_catalog)
    items_written = 0
    stac_written = False

    if stac_requested:
        catalog_path = Path(args.out_stac_catalog)
        items_dir = Path(args.out_stac_items_dir) if args.out_stac_items_dir else catalog_path.parent / "items"

        items_written, _ = _write_stac_outputs(
            rows=rows,
            catalog_path=catalog_path,
            items_dir=items_dir,
            datetime_field=args.datetime_field,
        )

        if items_written == 0:
            if catalog_path.exists():
                catalog_path.unlink()
            if args.require_stac:
                print("STAC export unavailable: no entries with gps_latitude/gps_longitude and datetime", file=sys.stderr)
                return EXIT_STAC_UNAVAILABLE
            stac_written = False
        else:
            stac_written = True

    summary_payload = {
        "schema_version": "tp.archive.prov_stac.summary.v1",
        "manifest_rows": len(rows),
        "prov_output": str(Path(args.out_prov_jsonld)),
        "stac_requested": stac_requested,
        "stac_written": stac_written,
        "stac_items_written": items_written,
        "datetime_field": args.datetime_field,
    }
    atomic_write_text(Path(args.out_summary), deterministic_json_dumps(summary_payload, pretty=True) + "\n")

    print(f"Wrote PROV JSON-LD to {args.out_prov_jsonld}")
    if stac_requested:
        if stac_written:
            print(f"Wrote STAC catalog to {args.out_stac_catalog} ({items_written} items)")
        else:
            print("Skipped STAC catalog: insufficient geometry/datetime metadata")
    print(f"Wrote export summary to {args.out_summary}")
    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
