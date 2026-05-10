"""Tests for portal telemetry sink path policy."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from transformation_portal.portal import path_security
from transformation_portal.portal.path_security import PathSecurityValidationError

pytestmark = [pytest.mark.unit, pytest.mark.security]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
orchestrator_app = importlib.import_module("app")


def _repo_root(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    return repo_root


def _external_sink(tmp_path: Path, name: str = "portal-rum.jsonl") -> Path:
    sink_root = tmp_path / "operator-owned"
    sink_root.mkdir()
    return sink_root / name


def _assert_policy_reason(path_value: str, reason: str, *, repo_root: Path) -> None:
    with pytest.raises(PathSecurityValidationError) as exc_info:
        path_security.validate_portal_telemetry_sink_path(path_value, repo_root=repo_root)

    assert exc_info.value.reason == reason


def test_unset_sink_path_remains_noop_at_config_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TP_PORTAL_RUM_LOG_PATH", raising=False)
    monkeypatch.delenv("TP_PORTAL_EVENT_LOG_PATH", raising=False)

    assert orchestrator_app._portal_telemetry_sink_path_from_env("TP_PORTAL_RUM_LOG_PATH") is None
    assert orchestrator_app._portal_telemetry_sink_path_from_env("TP_PORTAL_EVENT_LOG_PATH") is None


def test_external_jsonl_sink_path_is_accepted(tmp_path: Path) -> None:
    repo_root = _repo_root(tmp_path)
    sink_path = _external_sink(tmp_path)

    assert path_security.validate_portal_telemetry_sink_path(str(sink_path), repo_root=repo_root) == Path(
        os.path.realpath(sink_path)
    )


def test_external_jsonl_gz_sink_path_is_accepted(tmp_path: Path) -> None:
    repo_root = _repo_root(tmp_path)
    sink_path = _external_sink(tmp_path, "portal-rum.jsonl.gz")

    assert path_security.validate_portal_telemetry_sink_path(str(sink_path), repo_root=repo_root) == Path(
        os.path.realpath(sink_path)
    )


def test_relative_sink_path_is_rejected(tmp_path: Path) -> None:
    _assert_policy_reason("logs/portal-rum.jsonl", "relative_portal_telemetry_sink_path", repo_root=_repo_root(tmp_path))


def test_empty_sink_path_is_rejected(tmp_path: Path) -> None:
    _assert_policy_reason("", "empty_portal_telemetry_sink_path", repo_root=_repo_root(tmp_path))


def test_glob_like_sink_path_is_rejected(tmp_path: Path) -> None:
    _assert_policy_reason("/tmp/portal-*.jsonl", "glob_portal_telemetry_sink_path", repo_root=_repo_root(tmp_path))


def test_non_jsonl_sink_suffix_is_rejected(tmp_path: Path) -> None:
    _assert_policy_reason(
        str(_external_sink(tmp_path, "portal-rum.txt")),
        "invalid_portal_telemetry_sink_suffix",
        repo_root=_repo_root(tmp_path),
    )


def test_existing_directory_sink_path_is_rejected(tmp_path: Path) -> None:
    repo_root = _repo_root(tmp_path)
    sink_path = _external_sink(tmp_path, "portal-rum.jsonl")
    sink_path.mkdir()

    _assert_policy_reason(str(sink_path), "directory_portal_telemetry_sink_path", repo_root=repo_root)


def test_symlink_sink_path_is_rejected(tmp_path: Path) -> None:
    repo_root = _repo_root(tmp_path)
    target_path = _external_sink(tmp_path, "target.jsonl")
    target_path.write_text("{}\n", encoding="utf-8")
    symlink_path = target_path.parent / "portal-rum.jsonl"
    symlink_path.symlink_to(target_path)

    _assert_policy_reason(str(symlink_path), "symlink_portal_telemetry_sink_path", repo_root=repo_root)


def test_repo_root_sink_path_is_rejected(tmp_path: Path) -> None:
    repo_root = _repo_root(tmp_path)

    _assert_policy_reason(str(repo_root / "portal-rum.jsonl"), "repo_portal_telemetry_sink_path", repo_root=repo_root)


def test_nested_repo_sink_path_is_rejected(tmp_path: Path) -> None:
    repo_root = _repo_root(tmp_path)

    _assert_policy_reason(
        str(repo_root / "var" / "portal-rum.jsonl"),
        "repo_portal_telemetry_sink_path",
        repo_root=repo_root,
    )


def test_repo_symlink_parent_to_external_sink_is_rejected(tmp_path: Path) -> None:
    repo_root = _repo_root(tmp_path)
    external_root = tmp_path / "operator-owned"
    external_root.mkdir()
    link_path = repo_root / "logs-link"
    link_path.symlink_to(external_root, target_is_directory=True)

    _assert_policy_reason(
        str(link_path / "portal-rum.jsonl"),
        "repo_portal_telemetry_sink_path",
        repo_root=repo_root,
    )


def test_frontdoor_public_sink_path_is_rejected(tmp_path: Path) -> None:
    repo_root = _repo_root(tmp_path)

    _assert_policy_reason(
        str(repo_root / "web" / "secure-landing" / "public" / "portal-rum.jsonl"),
        "public_static_portal_telemetry_sink_path",
        repo_root=repo_root,
    )


def test_frontdoor_public_symlink_parent_to_external_sink_is_rejected(tmp_path: Path) -> None:
    repo_root = _repo_root(tmp_path)
    external_root = tmp_path / "operator-owned"
    external_root.mkdir()
    public_root = repo_root / "web" / "secure-landing" / "public"
    public_root.mkdir(parents=True)
    link_path = public_root / "logs-link"
    link_path.symlink_to(external_root, target_is_directory=True)

    _assert_policy_reason(
        str(link_path / "portal-rum.jsonl"),
        "public_static_portal_telemetry_sink_path",
        repo_root=repo_root,
    )


def test_static_asset_sink_path_is_rejected(tmp_path: Path) -> None:
    repo_root = _repo_root(tmp_path)

    _assert_policy_reason(
        str(repo_root / "static" / "portal-rum.jsonl"),
        "public_static_portal_telemetry_sink_path",
        repo_root=repo_root,
    )


def test_github_workspace_sink_path_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = _repo_root(tmp_path)
    github_workspace = tmp_path / "github-workspace"
    github_workspace.mkdir()
    monkeypatch.setenv("GITHUB_WORKSPACE", str(github_workspace))

    _assert_policy_reason(
        str(github_workspace / "portal-rum.jsonl"),
        "ci_portal_telemetry_sink_path",
        repo_root=repo_root,
    )


def test_runner_temp_sink_path_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = _repo_root(tmp_path)
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    monkeypatch.setenv("RUNNER_TEMP", str(runner_temp))

    _assert_policy_reason(
        str(runner_temp / "portal-events.jsonl"),
        "ci_portal_telemetry_sink_path",
        repo_root=repo_root,
    )


def test_configured_invalid_sink_fails_app_import_without_leaking_contents(tmp_path: Path) -> None:
    unsafe_content = "do-not-leak-raw-log-content"
    unsafe_path = tmp_path / "portal-rum.txt"
    unsafe_path.write_text(unsafe_content, encoding="utf-8")
    env = os.environ.copy()
    env["TP_PORTAL_RUM_LOG_PATH"] = str(unsafe_path)
    env.pop("TP_PORTAL_EVENT_LOG_PATH", None)

    result = subprocess.run(
        [sys.executable, "-c", "import app"],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    combined_output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "TP_PORTAL_RUM_LOG_PATH violates portal telemetry sink path policy" in combined_output
    assert "must end with .jsonl or .jsonl.gz" in combined_output
    assert unsafe_content not in combined_output
