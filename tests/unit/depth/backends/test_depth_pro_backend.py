"""Focused unit tests for DepthProBackend subprocess readiness."""

from __future__ import annotations

import builtins
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from transformation_portal.depth.backends.depth_pro import DepthProBackend
from transformation_portal.depth.backends.depth_pro_worker import (
    _check_availability,
    _check_device_availability,
    _torch_diagnostics,
)
from transformation_portal.depth.backends.protocol import DepthResult
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
            "torch_version": "2.12.0",
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
        "torch_version": "2.12.0",
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


def test_torch_diagnostics_handles_missing_optional_accelerator_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = SimpleNamespace(
        __version__="test",
        backends=SimpleNamespace(),
        cuda=None,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    diagnostics = _torch_diagnostics("mps")

    assert diagnostics["torch_version"] == "test"
    assert diagnostics["mps_built"] is False
    assert diagnostics["mps_available"] is False
    assert diagnostics["cuda_available"] is False


def test_check_availability_reports_missing_checkpoint_before_imports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing_checkpoint = tmp_path / "missing-depth-pro.pt"
    real_import = builtins.__import__

    def _guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "depth_pro":
            raise AssertionError("depth_pro import should not run when checkpoint is missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)

    returncode = _check_availability(missing_checkpoint, "cpu")

    assert returncode == 1
    assert f"Checkpoint not found: {missing_checkpoint}" in capsys.readouterr().err


def test_compute_normalizes_device_override_for_readiness_and_subprocess(tmp_path: Path) -> None:
    checkpoint = tmp_path / "depth_pro.pt"
    checkpoint.write_bytes(b"checkpoint")
    backend = DepthProBackend(_depth_pro_config(checkpoint, device="cpu"))

    with (
        patch.object(backend, "_ensure_runtime_available") as mock_ready,
        patch.object(
            backend,
            "_compute_subprocess",
            return_value=DepthResult(
                depth_map=np.zeros((1, 1), dtype=np.float32),
                original_image=np.zeros((1, 1, 3), dtype=np.uint8),
                metadata={},
                depth_units="meters",
                focal_length_px=None,
                field_of_view_deg=None,
                backend_id=backend.name,
                device="mps",
                dtype="float32",
                input_size=(1, 1),
            ),
        ) as mock_subprocess,
    ):
        image = np.zeros((1, 1, 3), dtype=np.uint8)
        backend.compute(image, device="  MPS  ")

    mock_ready.assert_called_once_with(device="mps")
    mock_subprocess.assert_called_once()
    assert mock_subprocess.call_args.args[1] == "mps"
