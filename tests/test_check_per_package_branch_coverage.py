"""Unit tests for branch-coverage checker."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]


def _load_script_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "ci" / "check_per_package_branch_coverage.py"
    spec = importlib.util.spec_from_file_location("check_per_package_branch_coverage", script_path)
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


def test_aggregate_reads_branch_condition_counts_with_source_relative_paths(script_module, tmp_path: Path):
    src_pkg = tmp_path / "src" / "pkg"
    src_pkg.mkdir(parents=True)
    (src_pkg / "branchy.py").write_text("# source file\n", encoding="utf-8")
    coverage = tmp_path / "coverage.xml"
    _write_coverage(
        coverage,
        [
            (
                "branchy.py",
                [
                    {"hits": 1, "condition_coverage": "50% (1/2)"},
                    {"hits": 1, "condition_coverage": "100% (2/2)"},
                ],
            )
        ],
        sources=[str(src_pkg)],
    )

    results = script_module.aggregate(coverage, (script_module.BranchFloor("src/pkg/", 75.0),))

    assert results[0].covered == 3
    assert results[0].valid == 4
    assert results[0].percentage == 75.0
    assert results[0].passed is True


def test_dry_run_reports_floor_miss_without_failing(script_module, tmp_path: Path, monkeypatch, capsys):
    coverage = tmp_path / "coverage.xml"
    _write_coverage(
        coverage,
        [
            (
                "src/pkg/branchy.py",
                [
                    {"hits": 1, "condition_coverage": "25% (1/4)"},
                ],
            )
        ],
    )
    monkeypatch.setattr(script_module, "BRANCH_FLOORS", (script_module.BranchFloor("src/pkg/", 90.0),))

    rc = script_module.main([str(coverage), "--dry-run"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "DRY-RUN" in captured.out
    assert "not enforced" in captured.out


def test_missing_coverage_xml_returns_2(script_module, tmp_path: Path):
    assert script_module.main([str(tmp_path / "missing.xml"), "--dry-run"]) == 2


def test_default_branch_floors_cover_cold_zone_prefixes(script_module):
    assert tuple((floor.prefix, floor.floor) for floor in script_module.BRANCH_FLOORS) == (
        ("src/transformation_portal/plugins/", 36.0),
        ("src/transformation_portal/stage_graph/", 63.0),
        ("src/transformation_portal/vlm/", 55.0),
        ("src/transformation_portal/depth/", 40.0),
        ("src/transformation_portal/streaming/", 29.0),
        ("src/transformation_portal/spatial_ai/reconstruction/", 47.0),
    )


def test_empty_branch_floors_report_dry_run_baselines(script_module, tmp_path: Path, monkeypatch, capsys):
    coverage = tmp_path / "coverage.xml"
    _write_coverage(
        coverage,
        [
            (
                "src/transformation_portal/plugins/loader.py",
                [{"hits": 1, "condition_coverage": "50% (1/2)"}],
            ),
            (
                "src/transformation_portal/stage_graph/policy.py",
                [{"hits": 1, "condition_coverage": "100% (2/2)"}],
            ),
        ],
    )
    monkeypatch.setattr(script_module, "BRANCH_FLOORS", ())

    rc = script_module.main([str(coverage), "--dry-run"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "No per-package branch coverage floors configured" in captured.out
    assert "src/transformation_portal/plugins/" in captured.out
    assert "src/transformation_portal/stage_graph/" in captured.out
    assert "1/2" in captured.out
    assert "  N/A" in captured.out
    assert "DRY-RUN" in captured.out


def test_empty_branch_floors_without_dry_run_is_success(script_module, tmp_path: Path, monkeypatch, capsys):
    coverage = tmp_path / "coverage.xml"
    _write_coverage(coverage, [("src/pkg/x.py", [{"hits": 1}])])
    monkeypatch.setattr(script_module, "BRANCH_FLOORS", ())

    rc = script_module.main([str(coverage)])

    captured = capsys.readouterr()
    assert rc == 0
    assert "No per-package branch coverage floors configured" in captured.out
    assert "Package" not in captured.out


def test_enforced_branch_floor_miss_returns_1(script_module, tmp_path: Path, monkeypatch, capsys):
    coverage = tmp_path / "coverage.xml"
    _write_coverage(
        coverage,
        [
            (
                "src/pkg/branchy.py",
                [{"hits": 1, "condition_coverage": "25% (1/4)"}],
            )
        ],
    )
    monkeypatch.setattr(script_module, "BRANCH_FLOORS", (script_module.BranchFloor("src/pkg/", 90.0),))

    rc = script_module.main([str(coverage)])

    captured = capsys.readouterr()
    assert rc == 1
    assert "FAIL" in captured.out
    assert "src/pkg/: 25.00% < floor 90.0%" in captured.err
