"""Structural contract for the scheduled performance gate."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.regression]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "performance-monitor.yml"
POLICY_PATH = PROJECT_ROOT / "docs" / "performance" / "GATE_POLICY.md"


def _workflow_step(workflow_text: str, step_name: str) -> str:
    match = re.search(
        rf"(?ms)^      - name: {re.escape(step_name)}\n(?P<body>.*?)(?=^      - name: |\Z)",
        workflow_text,
    )
    assert match is not None, f"performance-monitor workflow must keep an explicit {step_name!r} step"
    return match.group("body")


def _performance_gate_step(workflow_text: str) -> str:
    return _workflow_step(workflow_text, "Enforce nightly performance gate")


def _workflow_on_block(workflow_text: str) -> str:
    match = re.search(r"(?ms)^on:\n(?P<body>.*?)(?=^[A-Za-z_][\w-]*:|\Z)", workflow_text)
    assert match is not None, "performance-monitor workflow must keep an explicit trigger block"
    return match.group("body")


def test_performance_monitor_stays_schedule_or_manual_only() -> None:
    workflow_text = WORKFLOW_PATH.read_text(encoding="utf-8")
    policy_text = POLICY_PATH.read_text(encoding="utf-8")
    on_block = _workflow_on_block(workflow_text)

    assert "schedule:" in on_block
    assert "workflow_dispatch:" in on_block
    assert "pull_request:" not in on_block
    assert "push:" not in on_block
    assert "schedule/manual only" in policy_text


def test_performance_monitor_blocks_regressions_and_benchmark_failures() -> None:
    workflow_text = WORKFLOW_PATH.read_text(encoding="utf-8")
    step = _performance_gate_step(workflow_text)

    assert "steps.benchmark.outputs.status == 'regression'" in step
    assert "steps.benchmark.outputs.status == 'failed'" in step
    assert "not product-green evidence" in step
    assert "exit 1" in step


def test_benchmark_step_keeps_status_output_contract_for_gate_consumers() -> None:
    workflow_text = WORKFLOW_PATH.read_text(encoding="utf-8")
    benchmark_step = _workflow_step(workflow_text, "Run performance tests")
    report_step = _workflow_step(workflow_text, "Generate performance report")
    issue_step = _workflow_step(workflow_text, "Create issue on regression (deduplicated)")
    gate_step = _performance_gate_step(workflow_text)

    assert re.search(r"(?m)^        id: benchmark$", benchmark_step)
    for status in ("no_tests", "passed", "regression", "failed"):
        assert f'echo "status={status}" >> $GITHUB_OUTPUT' in benchmark_step

    assert "${{ steps.benchmark.outputs.status }}" in report_step
    assert "steps.benchmark.outputs.status == 'regression'" in issue_step
    assert "steps.benchmark.outputs.status == 'regression'" in gate_step
    assert "steps.benchmark.outputs.status == 'failed'" in gate_step


def test_performance_gate_policy_matches_workflow_failure_modes() -> None:
    workflow_text = WORKFLOW_PATH.read_text(encoding="utf-8")
    policy_text = POLICY_PATH.read_text(encoding="utf-8")
    step = _performance_gate_step(workflow_text)

    assert "status == 'failed'" in step
    assert "regression beyond its documented threshold" in policy_text
    assert "benchmark execution fails" in policy_text
    assert "not valid evidence" in policy_text


def test_invalid_benchmark_runs_fail_gate_without_opening_regression_issue() -> None:
    workflow_text = WORKFLOW_PATH.read_text(encoding="utf-8")
    policy_text = POLICY_PATH.read_text(encoding="utf-8")
    issue_step = _workflow_step(workflow_text, "Create issue on regression (deduplicated)")

    assert "steps.benchmark.outputs.status == 'regression'" in issue_step
    assert "steps.benchmark.outputs.status == 'failed'" not in issue_step
    assert "Performance Regression Detected" in issue_step
    assert "tooling/environment blockers" in policy_text
