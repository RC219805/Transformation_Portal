"""CLI tests for tools/build_mets_structmap.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "tools" / "build_mets_structmap.py"


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


def test_structmap_item_labels_use_full_relpath_to_avoid_stem_collisions(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    out_xml = tmp_path / "mets.xml"
    out_summary = tmp_path / "summary.json"

    _write_manifest_jsonl(
        manifest_path,
        [
            {
                "relpath": "dirA/img001.jpg",
                "hash_status": "ok",
                "mime": "image/jpeg",
                "extension": ".jpg",
                "partition": "P1",
            },
            {
                "relpath": "dirB/img001.jpg",
                "hash_status": "ok",
                "mime": "image/jpeg",
                "extension": ".jpg",
                "partition": "P1",
            },
        ],
    )

    result = _run_tool(
        "--manifest-jsonl",
        str(manifest_path),
        "--out-xml",
        str(out_xml),
        "--out-summary",
        str(out_summary),
    )
    assert result.returncode == 0, result.stderr

    xml_text = out_xml.read_text(encoding="utf-8")
    assert 'LABEL="dirA/img001.jpg"' in xml_text
    assert 'LABEL="dirB/img001.jpg"' in xml_text
    assert xml_text.count('TYPE="item"') == 2
