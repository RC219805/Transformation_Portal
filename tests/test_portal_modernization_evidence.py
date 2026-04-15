"""Tests for tools/portal_modernization_evidence.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "tools" / "portal_modernization_evidence.py"


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TOOL_PATH), *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def _write_jsonl(path: Path, records: list[object]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def test_portal_modernization_evidence_reports_repo_gates_in_text_output(tmp_path: Path) -> None:
    rum_path = tmp_path / "portal-rum.jsonl"
    event_path = tmp_path / "portal-events.jsonl"
    _write_jsonl(
        rum_path,
        [
            {
                "schema": "tp.orchestrator.portal_rum.v1",
                "event_type": "core_web_vital",
                "metric": "lcp",
                "value": 2200.0,
            },
            {
                "schema": "tp.orchestrator.portal_rum.v1",
                "event_type": "core_web_vital",
                "metric": "inp",
                "value": 150.0,
            },
            {
                "schema": "tp.orchestrator.portal_rum.v1",
                "event_type": "core_web_vital",
                "metric": "cls",
                "value": 0.04,
            },
            {
                "schema": "tp.orchestrator.portal_rum.v1",
                "event_type": "bootstrap_ready",
                "metric": "duration",
                "value": 180.0,
            },
            {
                "schema": "tp.orchestrator.portal_rum.v1",
                "event_type": "first_view_interactive",
                "metric": "duration",
                "value": 240.0,
            },
            {
                "schema": "tp.orchestrator.portal_rum.v1",
                "event_type": "portal_shell_rendered",
                "metric": "duration",
                "value": 95.0,
            },
            {
                "schema": "tp.orchestrator.portal_rum.v1",
                "event_type": "queue_request",
                "metric": "submit",
                "value": 70.0,
            },
            {
                "schema": "tp.orchestrator.portal_rum.v1",
                "event_type": "sse_reconnect",
                "value": 1.0,
            },
            {
                "schema": "tp.orchestrator.portal_event.v1",
                "event_type": "ignored_in_rum_sink",
            },
        ],
    )
    event_records = [
        {
            "schema": "tp.orchestrator.portal_event.v1",
            "event_type": "artifact_viewer_opened",
            "surface": "artifact_review",
            "metadata": {"job_id": f"job_{index:04d}", "viewer_mode": "modal"},
        }
        for index in range(20)
    ]
    event_records.append(
        {
            "schema": "tp.orchestrator.portal_event.v1",
            "event_type": "artifact_viewer_fallback",
            "surface": "artifact_review",
            "metadata": {"job_id": "job_0001", "viewer_mode": "modal", "fallback_reason": "inline_preview_unavailable"},
        }
    )
    event_records.append(
        {
            "schema": "tp.orchestrator.portal_rum.v1",
            "event_type": "ignored_in_event_sink",
        }
    )
    _write_jsonl(event_path, event_records)

    result = _run_cli("--rum-log", str(rum_path), "--event-log", str(event_path), "--operator-hours", "4")

    assert result.returncode == 0, result.stderr
    assert "m1_measurement_foundation status=pass" in result.stdout
    assert "m4_performance status=pass" in result.stdout
    assert "metric name=sse_reconnect_rate_per_operator_hour status=pass value=0.25" in result.stdout
    assert (
        "m5_artifact_review status=pass viewer_open_count=20 viewer_fallback_count=1 viewer_success_rate_pct=95.00"
        in result.stdout
    )
    assert "m5_fallback_reason reason=inline_preview_unavailable count=1" in result.stdout


def test_portal_modernization_evidence_json_marks_sse_threshold_insufficient_without_operator_hours(
    tmp_path: Path,
) -> None:
    rum_path = tmp_path / "portal-rum.jsonl"
    _write_jsonl(
        rum_path,
        [
            {
                "schema": "tp.orchestrator.portal_rum.v1",
                "event_type": "core_web_vital",
                "metric": "lcp",
                "value": 2200.0,
            },
            {
                "schema": "tp.orchestrator.portal_rum.v1",
                "event_type": "core_web_vital",
                "metric": "inp",
                "value": 150.0,
            },
            {
                "schema": "tp.orchestrator.portal_rum.v1",
                "event_type": "core_web_vital",
                "metric": "cls",
                "value": 0.04,
            },
            {
                "schema": "tp.orchestrator.portal_rum.v1",
                "event_type": "bootstrap_ready",
                "value": 180.0,
            },
            {
                "schema": "tp.orchestrator.portal_rum.v1",
                "event_type": "first_view_interactive",
                "value": 240.0,
            },
            {
                "schema": "tp.orchestrator.portal_rum.v1",
                "event_type": "portal_shell_rendered",
                "value": 95.0,
            },
            {
                "schema": "tp.orchestrator.portal_rum.v1",
                "event_type": "queue_request",
                "metric": "submit",
                "value": 70.0,
            },
            {
                "schema": "tp.orchestrator.portal_rum.v1",
                "event_type": "sse_reconnect",
                "value": 1.0,
            },
        ],
    )

    result = _run_cli("--rum-log", str(rum_path), "--format", "json")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["milestones"]["m1_measurement_foundation"]["status"] == "pass"
    assert payload["metrics"]["sse_reconnect_rate_per_operator_hour"]["status"] == "insufficient_data"
    assert payload["milestones"]["m5_artifact_review"]["status"] == "insufficient_data"


def test_portal_modernization_evidence_skips_non_object_json_lines(tmp_path: Path) -> None:
    rum_path = tmp_path / "portal-rum.jsonl"
    _write_jsonl(
        rum_path,
        [
            {
                "schema": "tp.orchestrator.portal_rum.v1",
                "event_type": "core_web_vital",
                "metric": "lcp",
                "value": 2200.0,
            },
            {
                "schema": "tp.orchestrator.portal_rum.v1",
                "event_type": "core_web_vital",
                "metric": "inp",
                "value": 150.0,
            },
            {
                "schema": "tp.orchestrator.portal_rum.v1",
                "event_type": "core_web_vital",
                "metric": "cls",
                "value": 0.04,
            },
            {
                "schema": "tp.orchestrator.portal_rum.v1",
                "event_type": "bootstrap_ready",
                "value": 180.0,
            },
            {
                "schema": "tp.orchestrator.portal_rum.v1",
                "event_type": "first_view_interactive",
                "value": 240.0,
            },
            {
                "schema": "tp.orchestrator.portal_rum.v1",
                "event_type": "portal_shell_rendered",
                "value": 95.0,
            },
            {
                "schema": "tp.orchestrator.portal_rum.v1",
                "event_type": "queue_request",
                "metric": "submit",
                "value": 70.0,
            },
            ["not", "an", "object"],
        ],
    )

    result = _run_cli("--rum-log", str(rum_path), "--format", "json")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["milestones"]["m1_measurement_foundation"]["status"] == "pass"
    assert "skipped non-object json line" in result.stderr


def test_portal_modernization_evidence_keeps_m1_visible_when_thresholds_fail(tmp_path: Path) -> None:
    rum_path = tmp_path / "portal-rum.jsonl"
    _write_jsonl(
        rum_path,
        [
            {
                "schema": "tp.orchestrator.portal_rum.v1",
                "event_type": "core_web_vital",
                "metric": "lcp",
                "value": 4100.0,
            },
            {
                "schema": "tp.orchestrator.portal_rum.v1",
                "event_type": "core_web_vital",
                "metric": "inp",
                "value": 320.0,
            },
            {
                "schema": "tp.orchestrator.portal_rum.v1",
                "event_type": "core_web_vital",
                "metric": "cls",
                "value": 0.22,
            },
            {
                "schema": "tp.orchestrator.portal_rum.v1",
                "event_type": "bootstrap_ready",
                "value": 180.0,
            },
            {
                "schema": "tp.orchestrator.portal_rum.v1",
                "event_type": "first_view_interactive",
                "value": 240.0,
            },
            {
                "schema": "tp.orchestrator.portal_rum.v1",
                "event_type": "portal_shell_rendered",
                "value": 95.0,
            },
            {
                "schema": "tp.orchestrator.portal_rum.v1",
                "event_type": "queue_request",
                "metric": "submit",
                "value": 225.0,
            },
        ],
    )

    result = _run_cli("--rum-log", str(rum_path), "--format", "json")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["milestones"]["m1_measurement_foundation"] == {
        "rum_visibility_confirmed": True,
        "status": "pass",
    }
    assert payload["metrics"]["lcp_p75_ms"]["status"] == "fail"
    assert payload["milestones"]["m4_performance"]["status"] == "fail"


def test_portal_modernization_evidence_marks_m5_fail_when_viewer_success_rate_misses_target(tmp_path: Path) -> None:
    rum_path = tmp_path / "portal-rum.jsonl"
    event_path = tmp_path / "portal-events.jsonl"
    _write_jsonl(
        rum_path,
        [
            {
                "schema": "tp.orchestrator.portal_rum.v1",
                "event_type": "core_web_vital",
                "metric": "lcp",
                "value": 2200.0,
            },
            {
                "schema": "tp.orchestrator.portal_rum.v1",
                "event_type": "core_web_vital",
                "metric": "inp",
                "value": 150.0,
            },
            {
                "schema": "tp.orchestrator.portal_rum.v1",
                "event_type": "core_web_vital",
                "metric": "cls",
                "value": 0.04,
            },
            {
                "schema": "tp.orchestrator.portal_rum.v1",
                "event_type": "bootstrap_ready",
                "value": 180.0,
            },
            {
                "schema": "tp.orchestrator.portal_rum.v1",
                "event_type": "first_view_interactive",
                "value": 240.0,
            },
            {
                "schema": "tp.orchestrator.portal_rum.v1",
                "event_type": "portal_shell_rendered",
                "value": 95.0,
            },
            {
                "schema": "tp.orchestrator.portal_rum.v1",
                "event_type": "queue_request",
                "metric": "submit",
                "value": 70.0,
            },
        ],
    )
    event_records = [
        {
            "schema": "tp.orchestrator.portal_event.v1",
            "event_type": "artifact_viewer_opened",
            "surface": "artifact_review",
            "metadata": {"job_id": f"job_{index:04d}", "viewer_mode": "modal"},
        }
        for index in range(10)
    ]
    event_records.extend(
        [
            {
                "schema": "tp.orchestrator.portal_event.v1",
                "event_type": "artifact_viewer_fallback",
                "surface": "artifact_review",
                "metadata": {"job_id": "job_0001", "viewer_mode": "modal", "fallback_reason": "inline_preview_unavailable"},
            },
            {
                "schema": "tp.orchestrator.portal_event.v1",
                "event_type": "artifact_viewer_fallback",
                "surface": "artifact_review",
                "metadata": {"job_id": "job_0002", "viewer_mode": "modal", "fallback_reason": "asset_url_unavailable"},
            },
        ]
    )
    _write_jsonl(event_path, event_records)

    result = _run_cli("--rum-log", str(rum_path), "--event-log", str(event_path), "--format", "json")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["milestones"]["m5_artifact_review"]["status"] == "fail"
    assert payload["milestones"]["m5_artifact_review"]["viewer_success_rate_pct"] == 80.0
    assert payload["milestones"]["m5_artifact_review"]["fallback_reasons"] == {
        "asset_url_unavailable": 1,
        "inline_preview_unavailable": 1,
    }
