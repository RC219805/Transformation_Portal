"""Unit tests for cold-zone touched-file coverage evidence reporting."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.unit]


def _load_script_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "ci" / "check_cold_zone_touched_files.py"
    spec = importlib.util.spec_from_file_location("check_cold_zone_touched_files", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module", name="script_module")
def _script_module_fixture():
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


def test_normalize_repo_path_only_removes_explicit_relative_prefixes(script_module):
    assert script_module._normalize_repo_path("./src/transformation_portal/depth/tools.py") == (
        "src/transformation_portal/depth/tools.py"
    )
    assert script_module._normalize_repo_path(".github/workflows/build.yml") == ".github/workflows/build.yml"
    assert script_module._normalize_repo_path("../src/transformation_portal/depth/tools.py") == (
        "../src/transformation_portal/depth/tools.py"
    )
    assert script_module._normalize_repo_path("/src/transformation_portal/depth/tools.py") == (
        "src/transformation_portal/depth/tools.py"
    )


def test_build_report_filters_to_touched_cold_zone_sources(script_module, tmp_path: Path):
    coverage = tmp_path / "coverage.xml"
    _write_coverage(
        coverage,
        [
            (
                "src/transformation_portal/depth/tools.py",
                [
                    {"hits": 1},
                    {"hits": 0, "condition_coverage": "50% (1/2)"},
                    {"hits": 0},
                    {"hits": 1, "condition_coverage": "100% (2/2)"},
                ],
            ),
            ("src/transformation_portal/plugins/registry.py", [{"hits": 1}]),
        ],
    )

    report = script_module.build_report(
        coverage,
        [
            "README.md",
            "tests/unit/depth/test_tools.py",
            "src/transformation_portal/depth/tools.py",
            "src/transformation_portal/not_cold.py",
        ],
        compare_ref="origin/main",
    )

    assert report.touched_files == ("src/transformation_portal/depth/tools.py",)
    assert report.missing_files == ()
    assert len(report.measured_files) == 1
    measured = report.measured_files[0]
    assert measured.covered_lines == 2
    assert measured.valid_lines == 4
    assert measured.line_percentage == 50.0
    assert measured.covered_branches == 3
    assert measured.valid_branches == 4
    assert measured.branch_percentage == 75.0
    assert measured.missed_line_ranges == ("2-3",)
    assert measured.missed_branch_count == 1


def test_render_report_includes_missed_ranges_and_branch_counts(script_module, tmp_path: Path):
    coverage = tmp_path / "coverage.xml"
    _write_coverage(
        coverage,
        [
            (
                "src/transformation_portal/streaming/stages.py",
                [
                    {"hits": 1},
                    {"hits": 0, "condition_coverage": "50% (1/2)"},
                    {"hits": 1},
                ],
            )
        ],
    )
    report = script_module.build_report(
        coverage,
        ["src/transformation_portal/streaming/stages.py"],
        compare_ref="origin/main",
    )

    rendered = script_module.render_report(report)

    assert "Cold-zone touched-file coverage report" in rendered
    assert "Touched cold-zone source files: 1" in rendered
    assert "src/transformation_portal/streaming/stages.py" in rendered
    assert "2/3" in rendered
    assert "66.67%" in rendered
    assert "1/2" in rendered
    assert "50.00%" in rendered
    assert "2" in rendered
    assert "Reviewer note:" in rendered


def test_missing_touched_coverage_entry_fails_closed(script_module, tmp_path: Path, capsys, monkeypatch):
    coverage = tmp_path / "coverage.xml"
    _write_coverage(coverage, [("src/transformation_portal/depth/other.py", [{"hits": 1}])])
    monkeypatch.setattr(
        script_module,
        "collect_changed_files",
        lambda compare_ref, *, repo_root: ("src/transformation_portal/depth/tools.py",),
    )

    rc = script_module.main([str(coverage), "--compare-ref", "origin/main", "--repo-root", str(tmp_path)])

    captured = capsys.readouterr()
    assert rc == 1
    assert "Missing coverage entries:" in captured.out
    assert "src/transformation_portal/depth/tools.py" in captured.out
    assert "touched cold-zone files are missing from coverage.xml" in captured.err


def test_no_touched_cold_zone_sources_passes(script_module, tmp_path: Path, capsys, monkeypatch):
    coverage = tmp_path / "coverage.xml"
    _write_coverage(coverage, [("src/transformation_portal/depth/tools.py", [{"hits": 1}])])
    monkeypatch.setattr(script_module, "collect_changed_files", lambda compare_ref, *, repo_root: ("README.md",))

    rc = script_module.main([str(coverage), "--compare-ref", "origin/main", "--repo-root", str(tmp_path)])

    captured = capsys.readouterr()
    assert rc == 0
    assert "Touched cold-zone source files: 0" in captured.out
    assert "No cold-zone source files changed" in captured.out


def test_missing_coverage_xml_returns_2(script_module, tmp_path: Path):
    assert script_module.main([str(tmp_path / "missing.xml")]) == 2


def test_git_diff_failure_returns_2(script_module, tmp_path: Path, capsys, monkeypatch):
    coverage = tmp_path / "coverage.xml"
    _write_coverage(coverage, [("src/transformation_portal/depth/tools.py", [{"hits": 1}])])

    def fail_diff(compare_ref, *, repo_root):
        raise RuntimeError("bad revision")

    monkeypatch.setattr(script_module, "collect_changed_files", fail_diff)

    rc = script_module.main([str(coverage), "--compare-ref", "origin/missing", "--repo-root", str(tmp_path)])

    captured = capsys.readouterr()
    assert rc == 2
    assert "unable to diff against origin/missing: bad revision" in captured.err


def test_collect_changed_files_falls_back_when_shallow_checkout_has_no_merge_base(script_module, tmp_path: Path, monkeypatch):
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        del kwargs
        calls.append(cmd)
        if cmd[-1] == "origin/main...HEAD":
            return SimpleNamespace(returncode=128, stdout="", stderr="fatal: origin/main...HEAD: no merge base")
        return SimpleNamespace(returncode=0, stdout="./src/transformation_portal/depth/tools.py\n", stderr="")

    monkeypatch.setattr(script_module.subprocess, "run", fake_run)

    changed_files = script_module.collect_changed_files("origin/main", repo_root=tmp_path)

    assert changed_files == ("src/transformation_portal/depth/tools.py",)
    assert calls == [
        ["git", "diff", "--name-only", "--diff-filter=ACMRT", "origin/main...HEAD"],
        ["git", "diff", "--name-only", "--diff-filter=ACMRT", "origin/main", "HEAD"],
    ]
