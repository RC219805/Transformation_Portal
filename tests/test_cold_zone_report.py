"""Unit tests for cold-zone coverage baseline generation."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]


def _load_script_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "ci" / "cold_zone_report.py"
    spec = importlib.util.spec_from_file_location("cold_zone_report", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script_module():
    return _load_script_module()


def _write_coverage(
    path: Path,
    classes: list[tuple[str, list[dict[str, str | int]]]],
    sources: list[str] | None = None,
) -> None:
    body = ['<?xml version="1.0" ?>', "<coverage>"]
    if sources:
        body.append("  <sources>")
        for source in sources:
            body.append(f"    <source>{source}</source>")
        body.append("  </sources>")
    body.extend(["  <packages>", "    <package>", "      <classes>"])
    for filename, lines in classes:
        body.append(f'        <class name="x" filename="{filename}">')
        body.append("          <lines>")
        for idx, attrs in enumerate(lines, start=1):
            branch_attrs = ""
            if "condition_coverage" in attrs:
                branch_attrs = f' branch="true" condition-coverage="{attrs["condition_coverage"]}"'
            body.append(f'            <line number="{idx}" hits="{attrs["hits"]}"{branch_attrs}/>')
        body.append("          </lines>")
        body.append("        </class>")
    body.extend(["      </classes>", "    </package>", "  </packages>", "</coverage>"])
    path.write_text("\n".join(body), encoding="utf-8")


def test_report_writes_markdown_and_json_with_missed_ranges(script_module, tmp_path: Path):
    src_streaming = tmp_path / "src" / "transformation_portal" / "streaming"
    src_streaming.mkdir(parents=True)
    (src_streaming / "stages.py").write_text("# source file\n", encoding="utf-8")
    coverage = tmp_path / "coverage.xml"
    _write_coverage(
        coverage,
        [
            (
                "stages.py",
                [
                    {"hits": 1},
                    {"hits": 0, "condition_coverage": "50% (1/2)"},
                    {"hits": 0},
                    {"hits": 1, "condition_coverage": "100% (2/2)"},
                ],
            )
        ],
        sources=[str(src_streaming)],
    )
    markdown_out = tmp_path / "cold_zone_baseline_2026-05-12.md"
    json_out = tmp_path / "cold_zone_baseline_2026-05-12.json"

    rc = script_module.main(
        [
            str(coverage),
            "--markdown-out",
            str(markdown_out),
            "--json-out",
            str(json_out),
            "--baseline-date",
            "2026-05-12",
        ]
    )

    assert rc == 0
    markdown = markdown_out.read_text(encoding="utf-8")
    assert "# Cold-Zone Coverage Baseline (2026-05-12)" in markdown
    assert "`src/transformation_portal/streaming/stages.py`" in markdown
    assert "| `src/transformation_portal/streaming/stages.py` | `unit` | 2/4 | 50.00% | 3/4 | 75.00% | 2-3 | 1 |" in markdown

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    streaming_row = payload["targets"][0]
    assert streaming_row["line_coverage_percent"] == 50.0
    assert streaming_row["branch_coverage_percent"] == 75.0
    assert streaming_row["missed_line_ranges"] == ["2-3"]
    assert streaming_row["missed_branch_count"] == 1


def test_missing_coverage_xml_returns_2(script_module, tmp_path: Path):
    assert script_module.main([str(tmp_path / "missing.xml")]) == 2


def test_baseline_date_inferred_from_markdown_filename(script_module, tmp_path: Path):
    markdown_out = tmp_path / "cold_zone_baseline_2026-05-12.md"

    assert script_module._infer_baseline_date(markdown_out) == "2026-05-12"
