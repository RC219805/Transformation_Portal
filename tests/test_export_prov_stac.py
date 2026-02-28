"""CLI tests for tools/export_prov_stac.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

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
