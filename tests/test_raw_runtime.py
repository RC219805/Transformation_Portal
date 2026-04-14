"""Unit tests for the shared RAW subprocess runtime helpers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from transformation_portal.core import raw_runtime

pytestmark = [pytest.mark.unit]


def test_check_raw_runtime_timeout_surfaces_output(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        raw_runtime,
        "resolve_raw_python_for_execution",
        lambda python_executable, start: "/tmp/raw-python",
    )
    monkeypatch.setattr(raw_runtime, "build_raw_worker_env", lambda start: {})
    monkeypatch.setattr(raw_runtime, "_worker_cwd", lambda start: Path("/tmp/repo"))

    timeout_exc = subprocess.TimeoutExpired(
        cmd=["/tmp/raw-python", "-m", raw_runtime.RAW_WORKER_MODULE, "--check"],
        timeout=raw_runtime.RAW_RUNTIME_CHECK_TIMEOUT_SECONDS,
        output="partial stdout",
        stderr="partial stderr",
    )

    with patch("subprocess.run", side_effect=timeout_exc) as mock_run:
        with pytest.raises(RuntimeError, match="RAW subprocess environment is not ready") as exc_info:
            raw_runtime.check_raw_runtime("./.venv-raw/bin/python", start=Path(__file__))

    assert mock_run.call_args.kwargs["timeout"] == raw_runtime.RAW_RUNTIME_CHECK_TIMEOUT_SECONDS
    message = str(exc_info.value)
    assert f"Timeout: {raw_runtime.RAW_RUNTIME_CHECK_TIMEOUT_SECONDS}s" in message
    assert "partial stdout" in message
    assert "partial stderr" in message


def test_run_raw_worker_resolves_relative_input_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    input_path = tmp_path / "inputs" / "sample.cr2"
    input_path.parent.mkdir(parents=True)
    input_path.write_bytes(b"raw")
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(
        raw_runtime,
        "resolve_raw_python_for_execution",
        lambda python_executable, start: "/tmp/raw-python",
    )
    monkeypatch.setattr(raw_runtime, "build_raw_worker_env", lambda start: {})
    monkeypatch.setattr(raw_runtime, "_worker_cwd", lambda start: tmp_path)

    expected_array = np.full((2, 3, 3), 0.5, dtype=np.float32)
    expected_metadata = {"dtype": "float32", "shape": [2, 3, 3]}

    def fake_run(command, **kwargs):
        assert kwargs["timeout"] == raw_runtime.RAW_WORKER_TIMEOUT_SECONDS
        input_index = command.index("--input-path") + 1
        output_array_index = command.index("--output-array") + 1
        output_json_index = command.index("--output-json") + 1
        assert command[input_index] == str(input_path.resolve())
        np.save(command[output_array_index], expected_array, allow_pickle=False)
        Path(command[output_json_index]).write_text(json.dumps(expected_metadata), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=fake_run):
        array, metadata = raw_runtime.run_raw_worker(
            python_executable="./.venv-raw/bin/python",
            command_name="load_rgb",
            input_path=Path("inputs/sample.cr2"),
            payload={"output_bps": 16},
            start=Path(__file__),
        )

    assert np.array_equal(array, expected_array)
    assert metadata == expected_metadata


def test_run_raw_worker_timeout_surfaces_output(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        raw_runtime,
        "resolve_raw_python_for_execution",
        lambda python_executable, start: "/tmp/raw-python",
    )
    monkeypatch.setattr(raw_runtime, "build_raw_worker_env", lambda start: {})
    monkeypatch.setattr(raw_runtime, "_worker_cwd", lambda start: Path("/tmp/repo"))

    timeout_exc = subprocess.TimeoutExpired(
        cmd=["/tmp/raw-python", "-m", raw_runtime.RAW_WORKER_MODULE, "--command", "load_rgb"],
        timeout=raw_runtime.RAW_WORKER_TIMEOUT_SECONDS,
        output="worker stdout",
        stderr="worker stderr",
    )

    with patch("subprocess.run", side_effect=timeout_exc):
        with pytest.raises(RuntimeError, match="RAW subprocess worker failed") as exc_info:
            raw_runtime.run_raw_worker(
                python_executable="./.venv-raw/bin/python",
                command_name="load_rgb",
                input_path=Path("sample.cr2"),
                payload={},
                start=Path(__file__),
            )

    message = str(exc_info.value)
    assert f"Timeout: {raw_runtime.RAW_WORKER_TIMEOUT_SECONDS}s" in message
    assert "worker stdout" in message
    assert "worker stderr" in message
