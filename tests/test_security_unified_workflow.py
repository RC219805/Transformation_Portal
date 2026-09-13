from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.unit, pytest.mark.security]

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "security-unified.yml"
AUDIT_STEP_NAME = "Run pip-audit (BLOCKING on any advisory)"


def _load_workflow() -> dict:
    return yaml.load(WORKFLOW_PATH.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _workflow_step(name: str, job: str = "dependency-scan") -> dict:
    steps = _load_workflow()["jobs"][job]["steps"]
    return next(step for step in steps if step.get("name") == name)


def _run_audit_step(tmp_path: Path, report: dict | None, audit_exit: int) -> subprocess.CompletedProcess[str]:
    fixture_path = tmp_path / "pip-audit-fixture.json"
    if report is not None:
        fixture_path.write_text(json.dumps(report), encoding="utf-8")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    pip_audit_path = bin_dir / "pip-audit"
    pip_audit_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'if [ -n "${PIP_AUDIT_FIXTURE:-}" ]; then cp "$PIP_AUDIT_FIXTURE" audit-report.json; fi',
                'exit "$PIP_AUDIT_EXIT"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    pip_audit_path.chmod(pip_audit_path.stat().st_mode | stat.S_IXUSR)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env.get('PATH', '')}",
            "PIP_AUDIT_EXIT": str(audit_exit),
            "PIP_AUDIT_FIXTURE": str(fixture_path) if report is not None else "",
        }
    )
    return subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", _workflow_step(AUDIT_STEP_NAME)["run"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _report_with_vulnerabilities() -> dict:
    return {
        "dependencies": [
            {
                "name": "example-one",
                "version": "1.0",
                "vulns": [{"id": "PYSEC-1"}, {"id": "PYSEC-2"}],
            },
            {
                "name": "example-two",
                "version": "2.0",
                "vulns": [{"id": "PYSEC-3"}],
            },
            {"name": "local-package", "skip_reason": "not on PyPI"},
        ],
        "fixes": [],
    }


def test_pip_audit_step_blocks_and_counts_nested_findings(tmp_path: Path) -> None:
    result = _run_audit_step(tmp_path, _report_with_vulnerabilities(), audit_exit=1)

    assert result.returncode == 1
    assert "BLOCKED: Found 3 known vulnerability advisories" in result.stdout


def test_pip_audit_step_propagates_operational_failure(tmp_path: Path) -> None:
    report = {"dependencies": [{"name": "safe", "version": "1.0", "vulns": []}], "fixes": []}

    result = _run_audit_step(tmp_path, report, audit_exit=2)

    assert result.returncode == 2
    assert "BLOCKED: pip-audit failed with exit code 2" in result.stdout


def test_pip_audit_step_allows_a_clean_report(tmp_path: Path) -> None:
    report = {"dependencies": [{"name": "safe", "version": "1.0", "vulns": []}], "fixes": []}

    result = _run_audit_step(tmp_path, report, audit_exit=0)

    assert result.returncode == 0, result.stderr
    assert "No known vulnerability advisories found" in result.stdout


def test_pip_audit_step_defensively_blocks_findings_on_zero_exit(tmp_path: Path) -> None:
    result = _run_audit_step(tmp_path, _report_with_vulnerabilities(), audit_exit=0)

    assert result.returncode == 1
    assert "BLOCKED: Found 3 known vulnerability advisories" in result.stdout


def test_pip_audit_step_blocks_a_missing_report(tmp_path: Path) -> None:
    result = _run_audit_step(tmp_path, report=None, audit_exit=0)

    assert result.returncode == 1
    assert "BLOCKED: pip-audit did not produce audit-report.json" in result.stdout


def test_pip_audit_step_blocks_a_report_with_the_wrong_schema(tmp_path: Path) -> None:
    result = _run_audit_step(tmp_path, {"unexpected": []}, audit_exit=0)

    assert result.returncode != 0


def test_pip_audit_summary_counts_nested_findings(tmp_path: Path) -> None:
    (tmp_path / "audit-report.json").write_text(json.dumps(_report_with_vulnerabilities()), encoding="utf-8")
    summary_path = tmp_path / "step-summary.md"
    env = os.environ.copy()
    env["GITHUB_STEP_SUMMARY"] = str(summary_path)

    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", _workflow_step("Security Scan Summary")["run"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "- Total vulnerabilities: 3" in summary_path.read_text(encoding="utf-8")


def test_pip_audit_workflow_contract_has_no_severity_or_top_level_count_fallback() -> None:
    audit_run = _workflow_step(AUDIT_STEP_NAME)["run"]
    summary_run = _workflow_step("Security Scan Summary")["run"]

    assert "AUDIT_EXIT=0" in audit_run
    assert 'exit "$AUDIT_EXIT"' in audit_run
    assert "[.dependencies[] | .vulns[]?] | length" in audit_run
    assert "[.dependencies[] | .vulns[]?] | length" in summary_run
    assert ".vulnerabilities" not in audit_run + summary_run
    assert ".severity" not in audit_run + summary_run


def test_security_tool_install_pins_a_non_vulnerable_setuptools() -> None:
    install_run = _workflow_step("Install security tools")["run"]

    assert '"pip==26.2.1"' in install_run
    assert '"setuptools==83.0.0"' in install_run


@pytest.mark.parametrize("filename", ["zz_last.py", "with spaces.sh", "with\nnewline.yaml", "deprecated/example.md"])
def test_unicode_workflow_scans_all_tracked_files_without_splitting(tmp_path: Path, filename: str) -> None:
    script_path = Path("scripts/validation/check_unicode_controls.py")
    (tmp_path / script_path.parent).mkdir(parents=True)
    shutil.copyfile(REPO_ROOT / script_path, tmp_path / script_path)
    for index in range(105):
        (tmp_path / f"clean{index:03}.py").write_text("# clean\n", encoding="utf-8")
    bad_file = tmp_path / filename
    bad_file.parent.mkdir(parents=True, exist_ok=True)
    bad_file.write_text("# hidden\u202e\n", encoding="utf-8")
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    env = os.environ.copy()
    env["PATH"] = f"{Path(sys.executable).parent}:{env.get('PATH', '')}"
    step = _workflow_step("Check for bidirectional Unicode", job="security-gates")

    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", step["run"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "Bidirectional Unicode U+202E" in result.stderr
