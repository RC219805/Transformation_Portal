"""Contracts for the hosted APEX performance workflow."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "apex_performance.yml"
DA3_PYTHON = ".runtime/Depth-Anything-3/.venv-da3/bin/python"


def _load_workflow() -> dict:
    return yaml.load(WORKFLOW_PATH.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _step(job: dict, name: str) -> dict:
    return next(step for step in job["steps"] if step.get("name") == name)


def _run_results_check(tmp_path: Path, *, matrix_result: str, result_names: list[str]) -> dict[str, str]:
    workflow = _load_workflow()
    script = _step(workflow["jobs"]["apex_gate"], "Check downloaded results")["run"]
    results_dir = tmp_path / "apex_downloaded_results"
    results_dir.mkdir()
    for result_name in result_names:
        (results_dir / result_name).write_text("{}\n", encoding="utf-8")

    output_path = tmp_path / "github-output.txt"
    env = os.environ.copy()
    env.update({"GITHUB_OUTPUT": str(output_path), "MATRIX_RESULT": matrix_result})
    completed = subprocess.run(
        ["bash", "-c", script],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    return dict(line.split("=", 1) for line in output_path.read_text(encoding="utf-8").splitlines())


def test_real_da3_lane_bootstraps_only_the_governed_isolated_runtime() -> None:
    workflow = _load_workflow()
    matrix_job = workflow["jobs"]["apex_matrix"]
    install_step = _step(matrix_job, "Install dependencies (ML tier)")
    run_step = _step(matrix_job, "Run APEX Matrix")
    install_script = install_step["run"]

    assert install_step["if"] == "github.event.inputs.mode == 'real' || github.event_name == 'schedule'"
    assert matrix_job["steps"].index(install_step) < matrix_job["steps"].index(run_step)
    assert "./scripts/setup/install_da3_runtime.sh --profile baseline" in install_script
    assert DA3_PYTHON in install_script
    assert "TRANSFORMATION_PORTAL_DA3_PYTHON" in install_script
    assert "TP_STRICT_MODEL_LOCK=1" in install_script
    assert 'if [[ "${BACKEND_ID}" == "da3" ]]' in install_script
    assert 'python -m pip install -e ".[ml]"' in install_script


def test_gate_reports_matrix_failure_instead_of_claiming_transient_artifact_loss() -> None:
    workflow = _load_workflow()
    gate_job = workflow["jobs"]["apex_gate"]
    results_script = _step(gate_job, "Check downloaded results")["run"]
    report_script = _step(gate_job, "Generate PR Comment")["run"]
    ledger_upload = _step(gate_job, "Upload Ledger Artifact")

    assert "observation_v1_local.json observation_v2_local.json" in results_script
    assert '"${MATRIX_RESULT}" == "success"' in results_script
    assert "evidence_complete=true" in results_script
    assert "transient Actions artifact issue" not in report_script
    assert "APEX matrix execution failed before producing complete result artifacts" in report_script
    assert "APEX result artifacts were not transferred after a successful matrix run" in report_script
    assert "needs.apex_matrix.result" in report_script
    assert gate_job["outputs"]["evidence_complete"] == "${{ steps.results_check.outputs.evidence_complete }}"
    assert "steps.results_check.outputs.evidence_complete == 'true'" in ledger_upload["if"]
    assert "steps.rebuild_ledger.outcome == 'success'" in ledger_upload["if"]
    assert "steps.aggregate_stats.outcome == 'success'" in ledger_upload["if"]


@pytest.mark.parametrize(
    ("matrix_result", "result_names", "expected_complete"),
    [
        ("failure", ["observation_v1_local.json"], "false"),
        ("success", ["observation_v1_local.json"], "false"),
        ("success", ["observation_v1_local.json", "observation_v2_local.json"], "true"),
    ],
)
def test_gate_requires_successful_complete_v1_v2_evidence(
    tmp_path: Path,
    matrix_result: str,
    result_names: list[str],
    expected_complete: str,
) -> None:
    outputs = _run_results_check(tmp_path, matrix_result=matrix_result, result_names=result_names)

    assert outputs["evidence_complete"] == expected_complete


def test_real_evidence_consumers_require_current_run_results() -> None:
    workflow = _load_workflow()
    dashboard_job = workflow["jobs"]["dashboard_deploy"]
    backup_job = workflow["jobs"]["weekly_backup"]
    download_step = _step(backup_job, "Download Current Run Ledger")

    assert dashboard_job["needs"] == "apex_gate"
    assert "needs.apex_gate.outputs.evidence_complete == 'true'" in dashboard_job["if"]
    assert "github.event_name == 'schedule'" in dashboard_job["if"]
    assert "github.event.inputs.mode == 'real'" in dashboard_job["if"]

    assert backup_job["needs"] == "apex_gate"
    assert backup_job["if"] == "github.event_name == 'schedule' && needs.apex_gate.outputs.evidence_complete == 'true'"
    assert download_step["uses"].startswith("actions/download-artifact@")
    assert download_step["with"] == {"name": "apex-ledger"}
    assert "dawidd6/action-download-artifact" not in WORKFLOW_PATH.read_text(encoding="utf-8")
