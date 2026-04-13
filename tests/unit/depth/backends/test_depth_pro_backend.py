"""Focused unit tests for DepthProBackend subprocess readiness."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from transformation_portal.depth.backends.depth_pro import DepthProBackend
from transformation_portal.depth.backends.depth_pro_worker import _check_device_availability
from transformation_portal.lux_depth_v3.config import EnhanceConfig

pytestmark = [
    pytest.mark.unit,
]


def _depth_pro_config(checkpoint: Path, *, device: str = "mps") -> EnhanceConfig:
    return EnhanceConfig(
        depth_backend="depth_pro",
        depth_device=device,
        depth_pro_checkpoint_path=str(checkpoint),
        depth_pro_python_executable=sys.executable,
        non_commercial_ok=True,
        accept_apple_depth_pro_research_license=True,
        enable_v2=False,
    )


def test_subprocess_availability_check_passes_requested_device(tmp_path: Path) -> None:
    checkpoint = tmp_path / "depth_pro.pt"
    checkpoint.write_bytes(b"checkpoint")
    backend = DepthProBackend(_depth_pro_config(checkpoint, device="mps"))

    with patch("transformation_portal.depth.backends.depth_pro.subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["python"],
            returncode=0,
            stdout="",
            stderr="",
        )

        backend.ensure_available()

    command = mock_run.call_args.args[0]
    assert "--check" in command
    assert "--device" in command
    assert command[command.index("--device") + 1] == "mps"


def test_subprocess_availability_failure_includes_worker_diagnostics(tmp_path: Path) -> None:
    checkpoint = tmp_path / "depth_pro.pt"
    checkpoint.write_bytes(b"checkpoint")
    backend = DepthProBackend(_depth_pro_config(checkpoint, device="mps"))

    diagnostic = json.dumps(
        {
            "status": "unavailable",
            "reason": "PyTorch MPS backend is not available in this runtime.",
            "device": "mps",
            "mps_built": True,
            "mps_available": False,
            "torch_version": "2.11.0",
        },
        sort_keys=True,
    )
    with patch("transformation_portal.depth.backends.depth_pro.subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["python"],
            returncode=1,
            stdout="",
            stderr=diagnostic,
        )

        with pytest.raises(ImportError) as exc_info:
            backend.ensure_available()

    message = str(exc_info.value)
    assert "--device mps" in message
    assert '"mps_available": false' in message
    assert "Depth Pro subprocess environment is not ready." in message


def test_worker_reports_structured_mps_diagnostics(capsys: pytest.CaptureFixture[str]) -> None:
    diagnostics = {
        "device": "mps",
        "mps_built": True,
        "mps_available": False,
        "torch_version": "2.11.0",
    }
    with patch(
        "transformation_portal.depth.backends.depth_pro_worker._torch_diagnostics",
        return_value=diagnostics,
    ):
        returncode = _check_device_availability("mps")

    assert returncode == 1
    payload = json.loads(capsys.readouterr().err)
    assert payload["status"] == "unavailable"
    assert payload["device"] == "mps"
    assert payload["mps_available"] is False
