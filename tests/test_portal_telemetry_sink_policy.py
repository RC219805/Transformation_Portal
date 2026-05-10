"""Tests for portal telemetry sink path policy."""

from __future__ import annotations

import importlib
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from transformation_portal.portal import path_security
from transformation_portal.portal.path_security import PathSecurityValidationError

pytestmark = [pytest.mark.unit, pytest.mark.security]

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def _stat_result(*, mode: int, dev: int = 1, ino: int = 10) -> os.stat_result:
    return os.stat_result((mode, ino, dev, 1, 0, 0, 0, 0, 0, 0))


def test_unset_sink_path_remains_noop_at_config_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TP_PORTAL_RUM_LOG_PATH", raising=False)
    monkeypatch.delenv("TP_PORTAL_EVENT_LOG_PATH", raising=False)
    orchestrator_app = importlib.import_module("app")

    assert orchestrator_app._portal_telemetry_sink_path_from_env("TP_PORTAL_RUM_LOG_PATH") is None
    assert orchestrator_app._portal_telemetry_sink_path_from_env("TP_PORTAL_EVENT_LOG_PATH") is None


def test_persist_portal_event_record_opens_sink_relative_to_parent_dir_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator_app = importlib.import_module("app")
    sink_path = _external_sink(tmp_path)
    open_calls = []
    closed_fds = []
    next_fd = iter([101, 102])
    real_stat = orchestrator_app.os.stat
    parent_stat = orchestrator_app.os.stat(sink_path.parent, follow_symlinks=False)

    def fake_open(path, flags, mode=0o777, *, dir_fd=None):
        fd = next(next_fd)
        open_calls.append(
            {
                "path": path,
                "flags": flags,
                "mode": mode,
                "dir_fd": dir_fd,
                "fd": fd,
            }
        )
        return fd

    def fake_fstat(fd):
        if fd == 101:
            return parent_stat
        return _stat_result(mode=stat.S_IFREG | 0o600, dev=parent_stat.st_dev, ino=parent_stat.st_ino + 1)

    def fake_stat(path, *args, follow_symlinks=True, **kwargs):
        if path == sink_path.name and kwargs.get("dir_fd") == 101:
            raise FileNotFoundError(path)
        return real_stat(path, *args, follow_symlinks=follow_symlinks, **kwargs)

    monkeypatch.setattr(orchestrator_app.os, "open", fake_open)
    monkeypatch.setattr(orchestrator_app.os, "supports_dir_fd", {*orchestrator_app.os.supports_dir_fd, fake_open})
    monkeypatch.setattr(orchestrator_app.os, "stat", fake_stat)
    monkeypatch.setattr(orchestrator_app.os, "fstat", fake_fstat)
    monkeypatch.setattr(orchestrator_app.os, "write", lambda _fd, data: len(data))
    monkeypatch.setattr(orchestrator_app.os, "close", lambda fd: closed_fds.append(fd))

    orchestrator_app._persist_portal_event_record({"schema": "test"}, sink_path)

    assert open_calls[0]["path"] == Path(os.path.realpath(sink_path)).parent
    assert open_calls[0]["dir_fd"] is None
    assert open_calls[1]["path"] == sink_path.name
    assert open_calls[1]["dir_fd"] == open_calls[0]["fd"]
    no_follow_flag = getattr(orchestrator_app.os, "O_NOFOLLOW", 0)
    if no_follow_flag:
        assert open_calls[0]["flags"] & no_follow_flag
        assert open_calls[1]["flags"] & no_follow_flag
    assert closed_fds == [open_calls[1]["fd"], open_calls[0]["fd"]]


def test_persist_portal_event_record_rejects_parent_fd_identity_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    orchestrator_app = importlib.import_module("app")
    sink_path = _external_sink(tmp_path)
    unsafe_content = "do-not-leak-raw-log-content"
    open_calls = []
    close_calls = []
    write_calls = []
    real_stat = orchestrator_app.os.stat
    real_fstat = orchestrator_app.os.fstat
    expected_parent_stat = _stat_result(mode=stat.S_IFDIR | 0o700, dev=1, ino=10)
    changed_parent_stat = _stat_result(mode=stat.S_IFDIR | 0o700, dev=2, ino=20)

    monkeypatch.setattr(
        orchestrator_app._portal_path_security,
        "validate_portal_telemetry_sink_path",
        lambda _path, *, repo_root=None: sink_path,
    )

    def fake_stat(path, *args, follow_symlinks=True, **kwargs):
        if isinstance(path, (str, os.PathLike)) and Path(path) == sink_path.parent:
            return expected_parent_stat
        return real_stat(path, *args, follow_symlinks=follow_symlinks, **kwargs)

    def fake_fstat(fd):
        if fd == 101:
            return changed_parent_stat
        return real_fstat(fd)

    def fake_open(path, flags, mode=0o777, *, dir_fd=None):
        open_calls.append((path, flags, mode, dir_fd))
        return 101

    monkeypatch.setattr(orchestrator_app.os, "open", fake_open)
    monkeypatch.setattr(orchestrator_app.os, "supports_dir_fd", {*orchestrator_app.os.supports_dir_fd, fake_open})
    monkeypatch.setattr(orchestrator_app.os, "stat", fake_stat)
    monkeypatch.setattr(orchestrator_app.os, "fstat", fake_fstat)
    monkeypatch.setattr(orchestrator_app.os, "write", lambda fd, data: write_calls.append((fd, data)) or len(data))
    monkeypatch.setattr(orchestrator_app.os, "close", lambda fd: close_calls.append(fd))

    with caplog.at_level("WARNING"):
        orchestrator_app._persist_portal_event_record({"schema": "test", "raw": unsafe_content}, sink_path)

    assert len(open_calls) == 1
    assert write_calls == []
    assert close_calls == [101]
    assert "failed to persist portal event telemetry" in caplog.text
    assert unsafe_content not in caplog.text


def test_persist_portal_event_record_rejects_non_regular_existing_sink_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    orchestrator_app = importlib.import_module("app")
    sink_path = _external_sink(tmp_path)
    unsafe_content = "do-not-leak-raw-log-content"
    open_calls = []
    close_calls = []
    write_calls = []
    real_stat = orchestrator_app.os.stat
    real_fstat = orchestrator_app.os.fstat
    parent_stat = _stat_result(mode=stat.S_IFDIR | 0o700, dev=1, ino=10)
    fifo_stat = _stat_result(mode=stat.S_IFIFO | 0o600, dev=1, ino=11)

    monkeypatch.setattr(
        orchestrator_app._portal_path_security,
        "validate_portal_telemetry_sink_path",
        lambda _path, *, repo_root=None: sink_path,
    )

    def fake_stat(path, *args, follow_symlinks=True, **kwargs):
        if isinstance(path, (str, os.PathLike)) and Path(path) == sink_path.parent:
            return parent_stat
        if path == sink_path.name and kwargs.get("dir_fd") == 101:
            return fifo_stat
        return real_stat(path, *args, follow_symlinks=follow_symlinks, **kwargs)

    def fake_fstat(fd):
        if fd == 101:
            return parent_stat
        return real_fstat(fd)

    def fake_open(path, flags, mode=0o777, *, dir_fd=None):
        fd = 101 if dir_fd is None else 102
        open_calls.append((path, flags, mode, dir_fd, fd))
        return fd

    monkeypatch.setattr(orchestrator_app.os, "stat", fake_stat)
    monkeypatch.setattr(orchestrator_app.os, "fstat", fake_fstat)
    monkeypatch.setattr(orchestrator_app.os, "open", fake_open)
    monkeypatch.setattr(orchestrator_app.os, "supports_dir_fd", {*orchestrator_app.os.supports_dir_fd, fake_open})
    monkeypatch.setattr(orchestrator_app.os, "write", lambda fd, data: write_calls.append((fd, data)) or len(data))
    monkeypatch.setattr(orchestrator_app.os, "close", lambda fd: close_calls.append(fd))

    with caplog.at_level("WARNING"):
        orchestrator_app._persist_portal_event_record({"schema": "test", "raw": unsafe_content}, sink_path)

    assert len(open_calls) == 1
    assert write_calls == []
    assert close_calls == [101]
    assert "failed to persist portal event telemetry" in caplog.text
    assert unsafe_content not in caplog.text


def test_persist_portal_event_record_requires_regular_opened_file_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    orchestrator_app = importlib.import_module("app")
    sink_path = _external_sink(tmp_path)
    unsafe_content = "do-not-leak-raw-log-content"
    open_calls = []
    close_calls = []
    write_calls = []
    real_stat = orchestrator_app.os.stat
    real_fstat = orchestrator_app.os.fstat
    parent_stat = _stat_result(mode=stat.S_IFDIR | 0o700, dev=1, ino=10)
    fifo_stat = _stat_result(mode=stat.S_IFIFO | 0o600, dev=1, ino=11)

    monkeypatch.setattr(
        orchestrator_app._portal_path_security,
        "validate_portal_telemetry_sink_path",
        lambda _path, *, repo_root=None: sink_path,
    )

    def fake_stat(path, *args, follow_symlinks=True, **kwargs):
        if isinstance(path, (str, os.PathLike)) and Path(path) == sink_path.parent:
            return parent_stat
        if path == sink_path.name and kwargs.get("dir_fd") == 101:
            raise FileNotFoundError(path)
        return real_stat(path, *args, follow_symlinks=follow_symlinks, **kwargs)

    def fake_fstat(fd):
        if fd == 101:
            return parent_stat
        if fd == 102:
            return fifo_stat
        return real_fstat(fd)

    def fake_open(path, flags, mode=0o777, *, dir_fd=None):
        fd = 101 if dir_fd is None else 102
        open_calls.append((path, flags, mode, dir_fd, fd))
        return fd

    monkeypatch.setattr(orchestrator_app.os, "fstat", fake_fstat)
    monkeypatch.setattr(orchestrator_app.os, "stat", fake_stat)
    monkeypatch.setattr(orchestrator_app.os, "open", fake_open)
    monkeypatch.setattr(orchestrator_app.os, "supports_dir_fd", {*orchestrator_app.os.supports_dir_fd, fake_open})
    monkeypatch.setattr(orchestrator_app.os, "write", lambda fd, data: write_calls.append((fd, data)) or len(data))
    monkeypatch.setattr(orchestrator_app.os, "close", lambda fd: close_calls.append(fd))

    with caplog.at_level("WARNING"):
        orchestrator_app._persist_portal_event_record({"schema": "test", "raw": unsafe_content}, sink_path)

    assert len(open_calls) == 2
    assert write_calls == []
    assert close_calls == [102, 101]
    assert "failed to persist portal event telemetry" in caplog.text
    assert unsafe_content not in caplog.text


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


def test_missing_parent_sink_path_is_rejected(tmp_path: Path) -> None:
    repo_root = _repo_root(tmp_path)
    sink_path = tmp_path / "operator-owned" / "missing-parent" / "portal-rum.jsonl"

    _assert_policy_reason(str(sink_path), "missing_portal_telemetry_sink_parent", repo_root=repo_root)


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


def test_external_symlink_parent_sink_path_is_rejected(tmp_path: Path) -> None:
    repo_root = _repo_root(tmp_path)
    operator_root = tmp_path / "operator-owned"
    operator_root.mkdir()
    target_root = tmp_path / "operator-target"
    target_root.mkdir()
    link_path = operator_root / "logs-link"
    link_path.symlink_to(target_root, target_is_directory=True)

    _assert_policy_reason(
        str(link_path / "portal-rum.jsonl"),
        "symlink_portal_telemetry_sink_parent",
        repo_root=repo_root,
    )


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
