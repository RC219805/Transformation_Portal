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
            "torch_version": "2.13.0",
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
        "torch_version": "2.13.0",
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


def test_check_availability_rejects_missing_plan_validator_before_model_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoint = tmp_path / "depth_pro.pt"
    checkpoint.write_bytes(b"checkpoint")
    real_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "jsonschema":
            raise ModuleNotFoundError("No module named 'jsonschema'")
        if name == "depth_pro":
            raise AssertionError("model imports must wait for execution-plan support")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert _check_availability(checkpoint, "cpu") == 1
    failure = json.loads(capsys.readouterr().err)
    assert failure["status"] == "unavailable"
    assert "install_depth_pro_runtime.sh" in failure["reason"]
    assert "jsonschema" in failure["execution_plan_import_error"]


def test_check_availability_validates_packaged_execution_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from transformation_portal.core import execution_plan
    from transformation_portal.depth.backends import depth_pro_worker

    checkpoint = tmp_path / "depth_pro.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setitem(sys.modules, "depth_pro", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "transformation_portal.stage_graph.stages.depth_pro", SimpleNamespace(DepthProStage=object)
    )
    monkeypatch.setattr(depth_pro_worker, "_check_device_availability", lambda device: 0)
    assert _check_availability(checkpoint, "cpu") == 0
    # An invalid packaged schema must never pass readiness either.
    monkeypatch.setattr(execution_plan, "load_execution_plan_schema", lambda: {"type": "invalid-schema-type"})
    import jsonschema

    with pytest.raises(jsonschema.SchemaError):
        _check_availability(checkpoint, "cpu")


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


@pytest.mark.parametrize(
    "dtype, values, expected",
    [
        ("uint16", [0, 255, 256, 32768, 65535], [0, 1, 1, 128, 255]),
        (">u2", [0, 255, 256, 32768, 65535], [0, 1, 1, 128, 255]),
        ("uint16", [0, 1], [0, 0]),
        ("uint8", [0, 1], [0, 1]),
    ],
)
def test_prepare_image_preserves_integer_intensity_order(dtype, values, expected):
    backend = DepthProBackend.__new__(DepthProBackend)
    image = np.repeat(np.array(values, dtype=dtype)[None, :, None], 3, axis=2)
    pil_image, prepared = backend._prepare_image(image)
    assert prepared[0, :, 0].tolist() == expected
    np.testing.assert_array_equal(np.asarray(pil_image), prepared)


def test_worker_envelope_retains_stage_focal_metadata(tmp_path, monkeypatch):
    from PIL import Image

    from transformation_portal.depth.backends import depth_pro_worker
    from transformation_portal.stage_graph.stage import StageResult, StageStatus
    from transformation_portal.stage_graph.stages import depth_pro as stage_module

    focal = 812.0
    camera = {"focal_length_px": focal, "fov_deg": 61.0, "focal_length_source": "model_estimated"}
    result = StageResult(
        stage_name="depth_pro_estimation",
        stage_version="1.0.0",
        status=StageStatus.COMPLETED,
        artifacts={"depth_map": np.ones((14, 28), np.float32), "depth_provenance": {"camera": camera}},
        metadata=camera,
    )
    monkeypatch.setattr(stage_module, "DepthProStage", lambda **kwargs: SimpleNamespace(compute=lambda context: result))
    image = tmp_path / "image.png"
    Image.new("RGB", (28, 14)).save(image)
    output = tmp_path / "result.json"
    assert depth_pro_worker._run_inference(image, tmp_path / "depth.npy", output, tmp_path / "weights.pt", "cpu") == 0
    payload = json.loads(output.read_text())
    assert payload["focal_length_px"] == focal
    assert payload["field_of_view_deg"] == 61.0
    assert payload["provenance"]["camera"]["focal_length_source"] == "model_estimated"


def _fake_depth_pro_worker(write_depth, payload):
    def fake_run(command, **kwargs):
        del kwargs
        write_depth(Path(command[command.index("--output-depth") + 1]))
        Path(command[command.index("--output-json") + 1]).write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="worker log")

    return fake_run


@pytest.mark.parametrize(
    ("depth", "payload"),
    [
        (np.full((4, 5), np.nan, dtype=np.float32), {"provenance": {}}),
        (np.full((4, 5), np.inf, dtype=np.float32), {"provenance": {}}),
        (np.zeros((4, 5, 3), dtype=np.float32), {"provenance": {}}),
        (np.zeros((4, 0), dtype=np.float32), {"provenance": {}}),
        (np.zeros((4, 5), dtype=np.float32), []),
        (None, {"provenance": {}}),
        (np.zeros((4, 5), dtype=np.float32), {"input_size": ["x", "y"]}),
    ],
    ids=["nan", "inf", "three_dimensional", "empty", "json_list", "empty_npy", "bad_input_size"],
)
def test_subprocess_rejects_malformed_exit_zero_output(tmp_path: Path, depth, payload) -> None:
    checkpoint = tmp_path / "depth_pro.pt"
    checkpoint.write_bytes(b"checkpoint")
    backend = DepthProBackend(_depth_pro_config(checkpoint, device="cpu"))

    def write_depth(path: Path) -> None:
        if depth is None:
            path.write_bytes(b"")
        else:
            np.save(path, depth, allow_pickle=False)

    with patch(
        "transformation_portal.depth.backends.depth_pro.subprocess.run",
        side_effect=_fake_depth_pro_worker(write_depth, payload),
    ):
        with pytest.raises(RuntimeError, match="Depth Pro subprocess produced malformed output") as exc_info:
            backend._compute_subprocess(np.zeros((4, 5, 3), dtype=np.uint8))

    assert "worker log" in str(exc_info.value)


def test_subprocess_accepts_valid_exit_zero_output(tmp_path: Path) -> None:
    checkpoint = tmp_path / "depth_pro.pt"
    checkpoint.write_bytes(b"checkpoint")
    backend = DepthProBackend(_depth_pro_config(checkpoint, device="cpu"))
    depth = np.arange(20, dtype=np.float64).reshape(4, 5) + 1.0

    with patch(
        "transformation_portal.depth.backends.depth_pro.subprocess.run",
        side_effect=_fake_depth_pro_worker(
            lambda path: np.save(path, depth, allow_pickle=False),
            {"provenance": {}, "input_size": [4, 5], "device": "cpu"},
        ),
    ):
        result = backend._compute_subprocess(np.zeros((4, 5, 3), dtype=np.uint8))

    assert result.depth_map.dtype == np.float32
    np.testing.assert_array_equal(result.depth_map, depth.astype(np.float32))
    assert result.input_size == (4, 5)


def test_worker_env_strips_parent_interpreter_configuration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = tmp_path / "depth_pro.pt"
    checkpoint.write_bytes(b"checkpoint")
    backend = DepthProBackend(_depth_pro_config(checkpoint, device="cpu"))
    monkeypatch.setenv("PYTHONPATH", "/parent/site-packages")
    monkeypatch.setenv("PYTHONHOME", "/parent/home")
    monkeypatch.setenv("VIRTUAL_ENV", "/parent/.venv")
    monkeypatch.setenv("TP_UNRELATED_SETTING", "kept")

    env = backend._build_worker_env()

    assert "PYTHONHOME" not in env
    assert "VIRTUAL_ENV" not in env
    assert "/parent/site-packages" not in env.get("PYTHONPATH", "")
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["PYTHONSAFEPATH"] == "1"
    assert env["TP_UNRELATED_SETTING"] == "kept"
    if backend._repo_src is not None and backend._repo_src.exists():
        assert env["PYTHONPATH"] == str(backend._repo_src)


def test_subprocess_calls_carry_timeout_and_fail_on_hung_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = tmp_path / "depth_pro.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setenv("TRANSFORMATION_PORTAL_DEPTH_PRO_TIMEOUT_SECONDS", "7")
    backend = DepthProBackend(_depth_pro_config(checkpoint, device="cpu"))
    timeouts: list[object] = []

    def hung_run(command, **kwargs):
        timeouts.append(kwargs.get("timeout"))
        raise subprocess.TimeoutExpired(command, kwargs.get("timeout") or 0, output=b"partial", stderr=b"stuck")

    with patch("transformation_portal.depth.backends.depth_pro.subprocess.run", side_effect=hung_run):
        with pytest.raises(ImportError, match="timed out after 7s"):
            backend.ensure_available()
        with pytest.raises(RuntimeError, match="Depth Pro subprocess timed out") as exc_info:
            backend._compute_subprocess(np.zeros((4, 5, 3), dtype=np.uint8))

    assert timeouts == [7, 7]
    assert "stuck" in str(exc_info.value)


def test_subprocess_timeout_defaults_and_rejects_invalid_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = tmp_path / "depth_pro.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.delenv("TRANSFORMATION_PORTAL_DEPTH_PRO_TIMEOUT_SECONDS", raising=False)
    assert DepthProBackend(_depth_pro_config(checkpoint))._subprocess_timeout_seconds == 900

    for invalid in ("0", "-3", "soon"):
        monkeypatch.setenv("TRANSFORMATION_PORTAL_DEPTH_PRO_TIMEOUT_SECONDS", invalid)
        with pytest.raises(ValueError, match="Depth Pro subprocess timeout"):
            DepthProBackend(_depth_pro_config(checkpoint))
