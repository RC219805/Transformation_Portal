"""Tests for tools/portal_telemetry_retention.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "tools" / "portal_telemetry_retention.py"
CONFIRM_DELETE = "DELETE-PORTAL-TELEMETRY-RAW-LOGS"


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TOOL_PATH), *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def _base_args(tmp_path: Path, sink_path: Path, evidence_path: Path) -> list[str]:
    return [
        "--pilot-owner",
        "RC219805",
        "--pilot-end-date",
        "2026-05-09",
        "--reviewer",
        "RC219805",
        "--sink-path",
        str(sink_path),
        "--evidence-out",
        str(evidence_path),
    ]


def _load_evidence(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_dry_run_produces_evidence_and_deletes_nothing(tmp_path: Path) -> None:
    sink_path = tmp_path / "portal-rum.jsonl"
    evidence_path = tmp_path / "deletion-evidence.json"
    sink_path.write_text('{"schema":"tp.orchestrator.portal_rum.v1"}\n', encoding="utf-8")

    result = _run_cli(*_base_args(tmp_path, sink_path, evidence_path), "--dry-run")

    assert result.returncode == 0, result.stderr
    assert sink_path.exists()
    payload = _load_evidence(evidence_path)
    assert payload["schema"] == "portal-telemetry-retention-evidence/v1"
    assert payload["mode"] == "dry-run"
    assert payload["retention_deadline"] == "2026-05-23"
    assert payload["retention_window_days"] == 14
    assert payload["summary"] == {
        "bytes_deleted": 0,
        "bytes_seen": sink_path.stat().st_size,
        "paths_deleted": 0,
        "paths_existing": 1,
        "paths_seen": 1,
    }
    sink_record = payload["sink_paths"][0]
    assert sink_record["exists"] is True
    assert sink_record["eligible_for_deletion"] is True
    assert sink_record["deleted"] is False
    assert sink_record["deletion_attempted"] is False


def test_delete_mode_without_confirmation_fails_and_preserves_file(tmp_path: Path) -> None:
    sink_path = tmp_path / "portal-events.jsonl"
    evidence_path = tmp_path / "deletion-evidence.json"
    sink_path.write_text('{"schema":"tp.orchestrator.portal_event.v1"}\n', encoding="utf-8")

    result = _run_cli(*_base_args(tmp_path, sink_path, evidence_path), "--delete")

    assert result.returncode != 0
    assert "--delete requires --confirm-delete DELETE-PORTAL-TELEMETRY-RAW-LOGS" in result.stderr
    assert sink_path.exists()
    assert not evidence_path.exists()


def test_delete_mode_with_confirmation_removes_existing_raw_log_files(tmp_path: Path) -> None:
    rum_path = tmp_path / "portal-rum.jsonl"
    event_path = tmp_path / "portal-events.jsonl"
    evidence_path = tmp_path / "deletion-evidence.json"
    rum_path.write_text('{"schema":"tp.orchestrator.portal_rum.v1"}\n', encoding="utf-8")
    event_path.write_text('{"schema":"tp.orchestrator.portal_event.v1"}\n', encoding="utf-8")
    bytes_seen = rum_path.stat().st_size + event_path.stat().st_size

    result = _run_cli(
        "--pilot-owner",
        "RC219805",
        "--pilot-end-date",
        "2026-05-09",
        "--reviewer",
        "RC219805",
        "--sink-path",
        str(rum_path),
        "--sink-path",
        str(event_path),
        "--evidence-out",
        str(evidence_path),
        "--delete",
        "--confirm-delete",
        CONFIRM_DELETE,
    )

    assert result.returncode == 0, result.stderr
    assert not rum_path.exists()
    assert not event_path.exists()
    payload = _load_evidence(evidence_path)
    assert payload["mode"] == "delete"
    assert payload["summary"] == {
        "bytes_deleted": bytes_seen,
        "bytes_seen": bytes_seen,
        "paths_deleted": 2,
        "paths_existing": 2,
        "paths_seen": 2,
    }
    for sink_record in payload["sink_paths"]:
        assert sink_record["deletion_attempted"] is True
        assert sink_record["deleted"] is True
        assert sink_record["deleted_at"].endswith("Z")
        assert sink_record["delete_error"] is None


def test_evidence_never_includes_raw_jsonl_contents(tmp_path: Path) -> None:
    sink_path = tmp_path / "portal-rum.jsonl"
    evidence_path = tmp_path / "deletion-evidence.json"
    sink_path.write_text('{"secret_raw_record":"do-not-preserve"}\n', encoding="utf-8")

    result = _run_cli(*_base_args(tmp_path, sink_path, evidence_path), "--dry-run")

    assert result.returncode == 0, result.stderr
    evidence_text = evidence_path.read_text(encoding="utf-8")
    assert "secret_raw_record" not in evidence_text
    assert "do-not-preserve" not in evidence_text


@pytest.mark.parametrize(
    ("raw_sink_path", "expected_error"),
    [
        ("relative/portal-rum.jsonl", "sink paths must be absolute"),
        ("/tmp/*.jsonl", "sink paths must not contain glob characters"),
    ],
)
def test_sink_path_rejects_relative_paths_and_globs(
    tmp_path: Path,
    raw_sink_path: str,
    expected_error: str,
) -> None:
    result = _run_cli(
        "--pilot-owner",
        "RC219805",
        "--pilot-end-date",
        "2026-05-09",
        "--reviewer",
        "RC219805",
        "--sink-path",
        raw_sink_path,
        "--evidence-out",
        str(tmp_path / "deletion-evidence.json"),
        "--dry-run",
    )

    assert result.returncode != 0
    assert expected_error in result.stderr


def test_directories_are_rejected(tmp_path: Path) -> None:
    sink_path = tmp_path / "portal-rum-directory"
    sink_path.mkdir()

    result = _run_cli(*_base_args(tmp_path, sink_path, tmp_path / "deletion-evidence.json"), "--dry-run")

    assert result.returncode != 0
    assert "sink paths must not be directories" in result.stderr


def test_symlinks_are_rejected(tmp_path: Path) -> None:
    target_path = tmp_path / "portal-rum.jsonl"
    sink_path = tmp_path / "portal-rum-link.jsonl"
    target_path.write_text("{}\n", encoding="utf-8")
    sink_path.symlink_to(target_path)

    result = _run_cli(*_base_args(tmp_path, sink_path, tmp_path / "deletion-evidence.json"), "--dry-run")

    assert result.returncode != 0
    assert "sink paths must not be symlinks" in result.stderr
    assert target_path.exists()


@pytest.mark.parametrize(
    "omitted_flag",
    ["--pilot-owner", "--pilot-end-date", "--reviewer"],
)
def test_required_pilot_metadata_is_required(tmp_path: Path, omitted_flag: str) -> None:
    sink_path = tmp_path / "portal-rum.jsonl"
    evidence_path = tmp_path / "deletion-evidence.json"
    sink_path.write_text("{}\n", encoding="utf-8")
    args = _base_args(tmp_path, sink_path, evidence_path)
    flag_index = args.index(omitted_flag)
    del args[flag_index : flag_index + 2]

    result = _run_cli(*args, "--dry-run")

    assert result.returncode != 0
    assert omitted_flag in result.stderr


def test_empty_pilot_metadata_is_rejected(tmp_path: Path) -> None:
    sink_path = tmp_path / "portal-rum.jsonl"
    evidence_path = tmp_path / "deletion-evidence.json"
    sink_path.write_text("{}\n", encoding="utf-8")

    result = _run_cli(*_base_args(tmp_path, sink_path, evidence_path), "--pilot-owner", "", "--dry-run")

    assert result.returncode != 0
    assert "--pilot-owner must not be empty" in result.stderr


def test_missing_sink_path_is_represented_safely(tmp_path: Path) -> None:
    sink_path = tmp_path / "missing-portal-rum.jsonl"
    evidence_path = tmp_path / "deletion-evidence.json"

    result = _run_cli(*_base_args(tmp_path, sink_path, evidence_path), "--dry-run")

    assert result.returncode == 0, result.stderr
    payload = _load_evidence(evidence_path)
    assert payload["summary"] == {
        "bytes_deleted": 0,
        "bytes_seen": 0,
        "paths_deleted": 0,
        "paths_existing": 0,
        "paths_seen": 1,
    }
    sink_record = payload["sink_paths"][0]
    assert sink_record["exists"] is False
    assert sink_record["eligible_for_deletion"] is False
    assert sink_record["deleted"] is False
    assert sink_record["delete_error"] is None


def test_path_policy_classification_is_report_only_for_repo_public_paths(tmp_path: Path) -> None:
    sink_path = PROJECT_ROOT / "web" / "secure-landing" / "public" / "portal-rum.jsonl"
    evidence_path = tmp_path / "deletion-evidence.json"

    result = _run_cli(*_base_args(tmp_path, sink_path, evidence_path), "--dry-run")

    assert result.returncode == 0, result.stderr
    payload = _load_evidence(evidence_path)
    path_policy = payload["sink_paths"][0]["path_policy"]
    assert path_policy["inside_repo"] is True
    assert path_policy["inside_public_or_static"] is True
    assert path_policy["warning"] == "raw logs should not be stored inside repository public/static paths"
    assert payload["sink_paths"][0]["exists"] is False
