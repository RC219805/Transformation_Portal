"""CLI tests for tools/export_prov_stac.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Pytest markers
pytestmark = [
    pytest.mark.unit,
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "tools" / "export_prov_stac.py"


def _run_tool(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TOOL_PATH), *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def _write_manifest_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def test_out_stac_items_dir_requires_out_stac_catalog(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    prov_jsonld = tmp_path / "prov.jsonld"
    summary_json = tmp_path / "summary.json"
    _write_manifest_jsonl(
        manifest_path,
        [
            {
                "relpath": "a/file.txt",
                "hash_status": "ok",
                "sha256": "a" * 64,
            }
        ],
    )

    result = _run_tool(
        "--manifest-jsonl",
        str(manifest_path),
        "--out-prov-jsonld",
        str(prov_jsonld),
        "--out-stac-items-dir",
        str(tmp_path / "items"),
        "--out-summary",
        str(summary_json),
    )

    assert result.returncode == 2
    assert "--out-stac-items-dir requires --out-stac-catalog" in result.stderr
    assert not prov_jsonld.exists()
    assert not summary_json.exists()


def test_no_eligible_items_does_not_overwrite_existing_catalog(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    prov_jsonld = tmp_path / "prov.jsonld"
    summary_json = tmp_path / "summary.json"
    catalog_path = tmp_path / "catalog.json"
    items_dir = tmp_path / "items"
    original_catalog_bytes = b'{"sentinel":"keep-me"}\n'
    catalog_path.write_bytes(original_catalog_bytes)

    _write_manifest_jsonl(
        manifest_path,
        [
            {
                "relpath": "a/file.txt",
                "hash_status": "ok",
                "sha256": "a" * 64,
                "modified_utc": "2024-01-01T00:00:00Z",
            }
        ],
    )

    result = _run_tool(
        "--manifest-jsonl",
        str(manifest_path),
        "--out-prov-jsonld",
        str(prov_jsonld),
        "--out-stac-catalog",
        str(catalog_path),
        "--out-stac-items-dir",
        str(items_dir),
        "--out-summary",
        str(summary_json),
    )
    assert result.returncode == 0, result.stderr

    summary_payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary_payload["stac_requested"] is True
    assert summary_payload["stac_written"] is False
    assert summary_payload["stac_items_written"] == 0

    assert catalog_path.read_bytes() == original_catalog_bytes
    assert not items_dir.exists()


def test_require_stac_with_no_eligible_rows_fails_without_overwriting_outputs(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    prov_jsonld = tmp_path / "prov.jsonld"
    summary_json = tmp_path / "summary.json"
    catalog_path = tmp_path / "catalog.json"
    items_dir = tmp_path / "items"

    original_prov = b'{"sentinel":"prov"}\n'
    original_summary = b'{"sentinel":"summary"}\n'
    original_catalog = b'{"sentinel":"catalog"}\n'
    prov_jsonld.write_bytes(original_prov)
    summary_json.write_bytes(original_summary)
    catalog_path.write_bytes(original_catalog)

    _write_manifest_jsonl(
        manifest_path,
        [
            {
                "relpath": "a/file.txt",
                "hash_status": "ok",
                "sha256": "a" * 64,
                "modified_utc": "2024-01-01T00:00:00Z",
            }
        ],
    )

    result = _run_tool(
        "--manifest-jsonl",
        str(manifest_path),
        "--out-prov-jsonld",
        str(prov_jsonld),
        "--out-stac-catalog",
        str(catalog_path),
        "--out-stac-items-dir",
        str(items_dir),
        "--require-stac",
        "--out-summary",
        str(summary_json),
    )
    assert result.returncode == 3
    assert "STAC export unavailable" in result.stderr

    assert prov_jsonld.read_bytes() == original_prov
    assert summary_json.read_bytes() == original_summary
    assert catalog_path.read_bytes() == original_catalog
    assert not items_dir.exists()


def _geo_row(relpath: str, digest: str, **extra: object) -> dict[str, object]:
    row: dict[str, object] = {
        "relpath": relpath,
        "hash_status": "ok",
        "sha256": digest * 64,
        "gps_latitude": 34.4,
        "gps_longitude": -119.6,
        "modified_utc": "2026-01-01T00:00:00Z",
    }
    row.update(extra)
    return row


def test_catalog_and_item_links_resolve_for_custom_items_dir(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    catalog_path = tmp_path / "stac" / "root.json"
    items_dir = tmp_path / "elsewhere" / "stac_items"
    _write_manifest_jsonl(
        manifest_path,
        [_geo_row("a/IMG_1.jpg", "0", provenance_id="p0"), _geo_row("b/IMG_2.jpg", "1", provenance_id="p1")],
    )

    result = _run_tool(
        "--manifest-jsonl",
        str(manifest_path),
        "--out-prov-jsonld",
        str(tmp_path / "prov.jsonld"),
        "--out-stac-catalog",
        str(catalog_path),
        "--out-stac-items-dir",
        str(items_dir),
        "--out-summary",
        str(tmp_path / "summary.json"),
    )
    assert result.returncode == 0, result.stderr

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    self_links = [link["href"] for link in catalog["links"] if link["rel"] == "self"]
    assert self_links == ["root.json"]
    item_hrefs = [link["href"] for link in catalog["links"] if link["rel"] == "item"]
    assert len(item_hrefs) == 2
    for href in item_hrefs:
        item_path = (catalog_path.parent / href).resolve()
        assert item_path.parent == items_dir.resolve()
        item = json.loads(item_path.read_text(encoding="utf-8"))
        parent_href = next(link["href"] for link in item["links"] if link["rel"] == "parent")
        assert (item_path.parent / parent_href).resolve() == catalog_path.resolve()

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["stac_items_written"] == len(list(items_dir.iterdir())) == 2


def test_default_items_dir_keeps_historical_link_layout(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    catalog_path = tmp_path / "stac" / "catalog.json"
    _write_manifest_jsonl(manifest_path, [_geo_row("a/IMG_1.jpg", "0", provenance_id="p0")])

    result = _run_tool(
        "--manifest-jsonl",
        str(manifest_path),
        "--out-prov-jsonld",
        str(tmp_path / "prov.jsonld"),
        "--out-stac-catalog",
        str(catalog_path),
        "--out-summary",
        str(tmp_path / "summary.json"),
    )
    assert result.returncode == 0, result.stderr

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    assert [link["href"] for link in catalog["links"]] == ["catalog.json", "items/p0.json"]
    item = json.loads((catalog_path.parent / "items" / "p0.json").read_text(encoding="utf-8"))
    assert item["links"][0]["href"] == "../catalog.json"


def test_colliding_fallback_item_ids_are_rejected_before_writing(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    catalog_path = tmp_path / "stac" / "catalog.json"
    prov_jsonld = tmp_path / "prov.jsonld"
    _write_manifest_jsonl(manifest_path, [_geo_row("a/IMG_1.jpg", "0"), _geo_row("b/IMG_1.jpg", "1")])

    result = _run_tool(
        "--manifest-jsonl",
        str(manifest_path),
        "--out-prov-jsonld",
        str(prov_jsonld),
        "--out-stac-catalog",
        str(catalog_path),
        "--out-summary",
        str(tmp_path / "summary.json"),
    )

    assert result.returncode == 2
    assert "Duplicate STAC item id 'IMG_1'" in result.stderr
    assert not prov_jsonld.exists()
    assert not catalog_path.exists()
    assert not (tmp_path / "stac" / "items").exists()


def test_provenance_id_with_path_separator_is_rejected(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    catalog_path = tmp_path / "stac" / "catalog.json"
    _write_manifest_jsonl(manifest_path, [_geo_row("a/IMG_1.jpg", "0", provenance_id="../../escaped")])

    result = _run_tool(
        "--manifest-jsonl",
        str(manifest_path),
        "--out-prov-jsonld",
        str(tmp_path / "prov.jsonld"),
        "--out-stac-catalog",
        str(catalog_path),
        "--out-summary",
        str(tmp_path / "summary.json"),
    )

    assert result.returncode == 2
    assert "is not a safe filename" in result.stderr
    assert not (tmp_path / "escaped.json").exists()
    assert not catalog_path.exists()
