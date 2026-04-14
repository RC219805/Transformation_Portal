"""Tests for tools/portal_rum_summary.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "tools" / "portal_rum_summary.py"


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TOOL_PATH), *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def test_portal_rum_summary_groups_records_and_prints_p75_metrics(tmp_path: Path) -> None:
    rum_path = tmp_path / "portal-rum.jsonl"
    records = [
        {
            "schema": "tp.orchestrator.portal_rum.v1",
            "auth_mode": "managed",
            "route": "/portal",
            "view": "build",
            "cohort_bucket": 12,
            "event_type": "core_web_vital",
            "metric": "lcp",
            "value": 200.0,
        },
        {
            "schema": "tp.orchestrator.portal_rum.v1",
            "auth_mode": "managed",
            "route": "/portal",
            "view": "build",
            "cohort_bucket": 12,
            "event_type": "core_web_vital",
            "metric": "lcp",
            "value": 300.0,
        },
        {
            "schema": "tp.orchestrator.portal_rum.v1",
            "auth_mode": "managed",
            "route": "/portal",
            "view": "build",
            "cohort_bucket": 12,
            "event_type": "bootstrap_ready",
            "metric": "duration",
            "value": 150.0,
        },
        {
            "schema": "tp.orchestrator.portal_rum.v1",
            "auth_mode": "managed",
            "route": "/portal",
            "view": "build",
            "cohort_bucket": 12,
            "event_type": "queue_request",
            "metric": "submit",
            "value": 45.0,
        },
        {
            "schema": "tp.orchestrator.portal_rum.v1",
            "auth_mode": "managed",
            "route": "/portal",
            "view": "build",
            "cohort_bucket": 12,
            "event_type": "sse_reconnect",
            "value": 1,
        },
        {
            "schema": "tp.orchestrator.portal_event.v1",
            "event_type": "config_exported",
        },
    ]
    rum_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    result = _run_cli("--input", str(rum_path))

    assert result.returncode == 0, result.stderr
    assert "auth_mode=managed route=/portal view=build cohort=12 samples=5" in result.stdout
    assert "lcp_p75_ms=275.00" in result.stdout
    assert "bootstrap_ready_p75_ms=150.00" in result.stdout
    assert "queue_submit_p75_ms=45.00" in result.stdout
    assert "sse_reconnect_count=1" in result.stdout


def test_portal_rum_summary_handles_empty_logs(tmp_path: Path) -> None:
    rum_path = tmp_path / "portal-rum.jsonl"
    rum_path.write_text("", encoding="utf-8")

    result = _run_cli("--input", str(rum_path))

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "portal rum summary: no records"


def test_portal_rum_summary_skips_invalid_json_lines(tmp_path: Path) -> None:
    rum_path = tmp_path / "portal-rum.jsonl"
    rum_path.write_text(
        "\n".join(
            [
                '{"schema":"tp.orchestrator.portal_rum.v1","auth_mode":"managed","route":"/portal","view":"build","cohort_bucket":12,"event_type":"bootstrap_ready","value":150}',
                '{"schema":"tp.orchestrator.portal_rum.v1","broken":',
                '{"schema":"tp.orchestrator.portal_event.v1","event_type":"config_exported"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = _run_cli("--input", str(rum_path))

    assert result.returncode == 0, result.stderr
    assert "auth_mode=managed route=/portal view=build cohort=12 samples=1" in result.stdout
    assert "portal rum summary: skipped invalid json line 2" in result.stderr
