"""Contracts for the hosted APEX performance workflow."""

from __future__ import annotations

import json
import os
import subprocess
import sys
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


def _read_outputs(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines())


def _run_matrix_stage_check(
    tmp_path: Path,
    *,
    runner_outcomes: list[str | None],
    upload_outcomes: list[str | None],
    jobs_payload_override: str | None = None,
    gh_failure: bool = False,
) -> dict[str, str]:
    workflow = _load_workflow()
    script = _step(workflow["jobs"]["apex_gate"], "Resolve Matrix Stage Outcomes")["run"]
    jobs = [
        {
            "name": f"Run (v{index} / local)",
            "steps": [
                {"name": "Run APEX Matrix", "conclusion": runner_outcome},
                {"name": "Upload Results", "conclusion": upload_outcome},
            ],
        }
        for index, (runner_outcome, upload_outcome) in enumerate(zip(runner_outcomes, upload_outcomes, strict=True), start=1)
    ]
    jobs_path = tmp_path / "jobs.json"
    jobs_path.write_text(
        jobs_payload_override if jobs_payload_override is not None else json.dumps({"jobs": jobs}),
        encoding="utf-8",
    )
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    fake_gh = fakebin / "gh"
    fake_gh.write_text(
        '#!/bin/sh\n[ "$FAKE_GH_FAIL" != "1" ] || exit 1\ncat "$FAKE_GH_JOBS"\n',
        encoding="utf-8",
    )
    fake_gh.chmod(0o755)

    output_path = tmp_path / "stage-output.txt"
    env = os.environ.copy()
    env.update(
        {
            "EXPECTED_MATRIX_JOBS": "2",
            "FAKE_GH_FAIL": "1" if gh_failure else "0",
            "FAKE_GH_JOBS": str(jobs_path),
            "GITHUB_OUTPUT": str(output_path),
            "GITHUB_REPOSITORY": "example/repo",
            "GITHUB_RUN_ID": "123",
            "PATH": f"{fakebin}{os.pathsep}{Path(sys.executable).parent}{os.pathsep}{env['PATH']}",
            "RUNNER_TEMP": str(tmp_path),
        }
    )
    completed = subprocess.run(
        ["bash", "-c", script],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    return _read_outputs(output_path)


def _run_results_check(
    tmp_path: Path,
    *,
    runner_outcome: str,
    upload_outcome: str,
    download_outcome: str,
    result_names: list[str],
    matrix_result: str = "success",
) -> dict[str, str]:
    workflow = _load_workflow()
    script = _step(workflow["jobs"]["apex_gate"], "Check downloaded results")["run"]
    results_dir = tmp_path / "apex_downloaded_results"
    results_dir.mkdir()
    for result_name in result_names:
        (results_dir / result_name).write_text("{}\n", encoding="utf-8")

    output_path = tmp_path / "github-output.txt"
    env = os.environ.copy()
    env.update(
        {
            "DOWNLOAD_OUTCOME": download_outcome,
            "GITHUB_OUTPUT": str(output_path),
            "MATRIX_RESULT": matrix_result,
            "RUNNER_OUTCOME": runner_outcome,
            "UPLOAD_OUTCOME": upload_outcome,
        }
    )
    completed = subprocess.run(
        ["bash", "-c", script],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    return _read_outputs(output_path)


def _run_partial_report(
    tmp_path: Path,
    *,
    runner_outcome: str,
    upload_outcome: str,
    download_outcome: str,
) -> str:
    workflow = _load_workflow()
    script = _step(workflow["jobs"]["apex_gate"], "Generate PR Comment")["run"]
    script = script.replace("${{ steps.results_check.outputs.evidence_complete }}", "false", 1)
    env = os.environ.copy()
    env.update(
        {
            "DOWNLOAD_OUTCOME": download_outcome,
            "MATRIX_RESULT": "failure",
            "RUNNER_OUTCOME": runner_outcome,
            "UPLOAD_OUTCOME": upload_outcome,
        }
    )
    completed = subprocess.run(
        ["bash", "-c", script],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    return (tmp_path / "apex_comment.md").read_text(encoding="utf-8")


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


def test_gate_exposes_each_transfer_stage_and_fail_closed_publishers() -> None:
    workflow = _load_workflow()
    matrix_job = workflow["jobs"]["apex_matrix"]
    gate_job = workflow["jobs"]["apex_gate"]
    stage_script = _step(gate_job, "Resolve Matrix Stage Outcomes")["run"]
    results_script = _step(gate_job, "Check downloaded results")["run"]
    report_script = _step(gate_job, "Generate PR Comment")["run"]
    ledger_upload = _step(gate_job, "Upload Ledger Artifact")

    assert _step(matrix_job, "Run APEX Matrix")["id"] == "run_apex"
    assert _step(matrix_job, "Upload Results")["id"] == "upload_results"
    assert _step(matrix_job, "Upload Results")["with"]["if-no-files-found"] == "error"
    assert gate_job["permissions"]["actions"] == "read"
    assert "actions/runs/${GITHUB_RUN_ID}/jobs?per_page=100" in stage_script
    assert "aggregate_step('Run APEX Matrix')" in stage_script
    assert "aggregate_step('Upload Results')" in stage_script
    assert "observation_v1_local.json observation_v2_local.json" in results_script
    assert '"${RUNNER_OUTCOME}" == "success"' in results_script
    assert '"${UPLOAD_OUTCOME}" == "success"' in results_script
    assert '"${DOWNLOAD_OUTCOME}" == "success"' in results_script
    assert "evidence_complete=true" in results_script
    assert "The `Run APEX Matrix` stage did not complete successfully" in report_script
    assert "The current-run matrix runner outcome could not be resolved" in report_script
    assert "APEX result upload failed after successful matrix execution" in report_script
    assert "The current-run result upload outcome could not be resolved" in report_script
    assert "APEX result download failed after successful matrix execution and upload" in report_script
    assert "The current-run result download outcome could not be resolved" in report_script
    assert "Complete V1/V2 APEX evidence was not present after successful transfer" in report_script
    assert gate_job["outputs"]["evidence_complete"] == "${{ steps.results_check.outputs.evidence_complete }}"
    assert gate_job["outputs"]["runner_outcome"] == "${{ steps.matrix_stages.outputs.runner_outcome }}"
    assert gate_job["outputs"]["upload_outcome"] == "${{ steps.matrix_stages.outputs.upload_outcome }}"
    assert gate_job["outputs"]["download_outcome"] == "${{ steps.results_check.outputs.download_outcome }}"
    assert "steps.results_check.outputs.evidence_complete == 'true'" in ledger_upload["if"]
    assert "steps.rebuild_ledger.outcome == 'success'" in ledger_upload["if"]
    assert "steps.aggregate_stats.outcome == 'success'" in ledger_upload["if"]


@pytest.mark.parametrize(
    ("runner_outcomes", "upload_outcomes", "expected_runner", "expected_upload"),
    [
        (["failure", "success"], ["success", "success"], "failure", "success"),
        (["success", "success"], ["failure", "success"], "success", "failure"),
        (["success", "success"], ["success", "success"], "success", "success"),
    ],
)
def test_gate_resolves_stage_outcomes_across_every_matrix_leg(
    tmp_path: Path,
    runner_outcomes: list[str],
    upload_outcomes: list[str],
    expected_runner: str,
    expected_upload: str,
) -> None:
    outputs = _run_matrix_stage_check(
        tmp_path,
        runner_outcomes=runner_outcomes,
        upload_outcomes=upload_outcomes,
    )

    assert outputs["runner_outcome"] == expected_runner
    assert outputs["upload_outcome"] == expected_upload
    assert outputs["matrix_job_count"] == "2"


def test_gate_stage_lookup_is_unknown_when_a_matrix_leg_is_missing(tmp_path: Path) -> None:
    outputs = _run_matrix_stage_check(
        tmp_path,
        runner_outcomes=["success"],
        upload_outcomes=["success"],
    )

    assert outputs["runner_outcome"] == "unknown"
    assert outputs["upload_outcome"] == "unknown"


def test_gate_stage_lookup_is_unknown_when_step_conclusion_is_missing(tmp_path: Path) -> None:
    outputs = _run_matrix_stage_check(
        tmp_path,
        runner_outcomes=["success", "success"],
        upload_outcomes=["success", None],
    )

    assert outputs["runner_outcome"] == "success"
    assert outputs["upload_outcome"] == "unknown"


@pytest.mark.parametrize(
    ("jobs_payload_override", "gh_failure"),
    [
        pytest.param(None, True, id="jobs-api-failure"),
        pytest.param("{malformed", False, id="malformed-jobs-json"),
    ],
)
def test_gate_stage_lookup_fails_closed_when_job_evidence_is_unavailable(
    tmp_path: Path,
    jobs_payload_override: str | None,
    gh_failure: bool,
) -> None:
    outputs = _run_matrix_stage_check(
        tmp_path,
        runner_outcomes=["success", "success"],
        upload_outcomes=["success", "success"],
        jobs_payload_override=jobs_payload_override,
        gh_failure=gh_failure,
    )

    assert outputs["runner_outcome"] == "unknown"
    assert outputs["upload_outcome"] == "unknown"


@pytest.mark.parametrize(
    ("runner_outcome", "upload_outcome", "download_outcome", "result_names", "expected_complete"),
    [
        ("failure", "success", "success", ["observation_v1_local.json", "observation_v2_local.json"], "false"),
        ("success", "failure", "success", ["observation_v1_local.json", "observation_v2_local.json"], "false"),
        ("success", "success", "failure", ["observation_v1_local.json", "observation_v2_local.json"], "false"),
        ("success", "success", "success", ["observation_v1_local.json"], "false"),
        ("success", "success", "success", ["observation_v1_local.json", "observation_v2_local.json"], "true"),
    ],
)
def test_gate_requires_successful_complete_v1_v2_evidence(
    tmp_path: Path,
    runner_outcome: str,
    upload_outcome: str,
    download_outcome: str,
    result_names: list[str],
    expected_complete: str,
) -> None:
    outputs = _run_results_check(
        tmp_path,
        runner_outcome=runner_outcome,
        upload_outcome=upload_outcome,
        download_outcome=download_outcome,
        result_names=result_names,
    )

    assert outputs["evidence_complete"] == expected_complete


@pytest.mark.parametrize(
    ("runner_outcome", "upload_outcome", "download_outcome", "expected_message"),
    [
        ("failure", "success", "success", "The `Run APEX Matrix` stage did not complete successfully"),
        ("unknown", "unknown", "success", "The current-run matrix runner outcome could not be resolved"),
        ("success", "failure", "success", "APEX result upload failed after successful matrix execution"),
        ("success", "unknown", "success", "The current-run result upload outcome could not be resolved"),
        ("success", "success", "failure", "APEX result download failed after successful matrix execution and upload"),
        ("success", "success", "unknown", "The current-run result download outcome could not be resolved"),
        ("success", "success", "success", "Complete V1/V2 APEX evidence was not present after successful transfer"),
    ],
)
def test_partial_report_names_the_failed_stage(
    tmp_path: Path,
    runner_outcome: str,
    upload_outcome: str,
    download_outcome: str,
    expected_message: str,
) -> None:
    report = _run_partial_report(
        tmp_path,
        runner_outcome=runner_outcome,
        upload_outcome=upload_outcome,
        download_outcome=download_outcome,
    )

    assert expected_message in report


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
