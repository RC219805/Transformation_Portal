"""Tests for DA3Backend adapter.

Tests that DA3Backend implements the DepthBackend protocol correctly
and integrates with the registry.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

# Import availability helpers from conftest
from tests.conftest import can_run_da3_compute
from transformation_portal.core.platform_matrix import PlatformAccel, PlatformISA, PlatformMatrix, PlatformOS
from transformation_portal.depth.backends.da3 import DA3Backend
from transformation_portal.depth.backends.protocol import DepthResult, LicenseType
from transformation_portal.depth.backends.registry import DepthBackendRegistry
from transformation_portal.lux_depth_v3.model_resolution import BackendCapabilityError

# Mark all tests in this module as ML tier (require torch + transformers)
pytestmark = pytest.mark.ml


def _licensed_da3_config(**overrides):
    from transformation_portal.lux_depth_v3.config import EnhanceConfig

    defaults = {
        "depth_device": "cpu",
        "non_commercial_ok": True,
    }
    defaults.update(overrides)
    return EnhanceConfig(**defaults)


def _install_fake_depth_anything3(monkeypatch):
    """Install a lightweight fake depth_anything_3 module for device smoke tests."""
    import types

    class FakeDepthAnything3:
        def __init__(self):
            self.loaded_device = None

        @classmethod
        def from_pretrained(cls, model_id):
            del model_id
            return cls()

        def to(self, device):
            dev = str(device)
            if "cuda" in dev:
                raise RuntimeError("Unexpected CUDA path in DA3 smoke test")
            self.loaded_device = dev
            return self

        def eval(self):
            return self

        def inference(self, images):
            del images
            if self.loaded_device is None:
                raise RuntimeError("Model device not set before inference")
            if "cuda" in str(self.loaded_device):
                raise RuntimeError("Torch not compiled with CUDA enabled")
            depth = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape((1, 64, 64))
            return SimpleNamespace(depth=depth)

    fake_pkg = types.ModuleType("depth_anything_3")
    fake_api = types.ModuleType("depth_anything_3.api")
    fake_pkg.DepthAnything3 = FakeDepthAnything3
    fake_api.DepthAnything3 = FakeDepthAnything3

    monkeypatch.setitem(sys.modules, "depth_anything_3", fake_pkg)
    monkeypatch.setitem(sys.modules, "depth_anything_3.api", fake_api)


def test_da3_backend_implements_protocol():
    """DA3Backend implements DepthBackend protocol."""
    backend = DA3Backend(_licensed_da3_config())
    assert backend.name == "da3"
    assert backend.license_type == LicenseType.MODEL_DEPENDENT
    assert backend.requires_checkpoint is False


def test_da3_backend_availability():
    """DA3Backend has ensure_available() method.

    Verifies the method exists and is callable.
    Actual error handling is tested in test_da3_backend_availability_missing_transformers.
    """
    backend = DA3Backend(_licensed_da3_config())

    # Verify method exists
    assert hasattr(backend, "ensure_available")
    assert callable(backend.ensure_available)


def test_da3_backend_availability_missing_transformers(monkeypatch):
    """DA3Backend.ensure_available() detects missing transformers dependency.

    Uses monkeypatch to manage sys.modules, simulating missing dependency.
    """
    # Use monkeypatch to safely modify sys.modules
    monkeypatch.delitem(sys.modules, "transformers", raising=False)
    monkeypatch.setitem(sys.modules, "transformers", None)

    backend = DA3Backend(_licensed_da3_config())

    # This should raise ImportError when ensure_available tries to import transformers
    with pytest.raises(ImportError, match="transformers"):
        backend.ensure_available()


def test_da3_backend_python_executable_resolution_from_config(tmp_path):
    """Dedicated DA3 Python path should be resolved from config."""
    python_executable = tmp_path / ".venv-da3" / "bin" / "python"
    python_executable.parent.mkdir(parents=True)
    python_executable.write_text("#!/bin/sh\n", encoding="utf-8")

    backend = DA3Backend(
        _licensed_da3_config(
            depth_device="cpu",
            da3_python_executable=str(python_executable),
        )
    )

    assert backend._python_executable == str(python_executable.resolve())


def test_da3_backend_missing_configured_python_never_falls_back_to_ambient(tmp_path):
    """A missing configured interpreter must fail instead of using host Python."""
    from unittest.mock import patch

    missing_python = tmp_path / ".venv-da3" / "bin" / "python"

    with patch(
        "transformation_portal.depth.backends.da3.shutil.which",
        side_effect=AssertionError("configured paths must not fall back to PATH"),
    ):
        with pytest.raises(FileNotFoundError, match="DA3 Python executable not found"):
            DA3Backend(
                _licensed_da3_config(
                    depth_device="cpu",
                    da3_python_executable=str(missing_python),
                )
            )


def test_da3_backend_python_executable_preserves_venv_symlink(tmp_path):
    """Configured DA3 Python should preserve the venv launcher symlink path."""
    target_python = tmp_path / "python3.11"
    target_python.write_text("#!/bin/sh\n", encoding="utf-8")
    python_executable = tmp_path / ".venv-da3" / "bin" / "python"
    python_executable.parent.mkdir(parents=True)
    python_executable.symlink_to(target_python)

    backend = DA3Backend(
        _licensed_da3_config(
            depth_device="cpu",
            da3_python_executable=str(python_executable),
        )
    )

    assert backend._python_executable == str(python_executable.absolute())


def test_da3_backend_relative_python_executable_anchors_to_repo_root(monkeypatch, tmp_path):
    """Relative DA3 Python paths should anchor to the discovered repo root."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    python_executable = repo_root / ".runtime" / "Depth-Anything-3" / ".venv-da3" / "bin" / "python"
    python_executable.parent.mkdir(parents=True)
    python_executable.write_text("#!/bin/sh\n", encoding="utf-8")

    other_cwd = tmp_path / "other"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)
    monkeypatch.setattr(DA3Backend, "_find_repo_root", lambda self: repo_root)

    backend = DA3Backend(
        _licensed_da3_config(
            depth_device="cpu",
            da3_python_executable="./.runtime/Depth-Anything-3/.venv-da3/bin/python",
        )
    )

    assert backend._python_executable == str(python_executable.absolute())


def test_da3_backend_required_packages_drop_transformers_when_env_points_to_subprocess(monkeypatch):
    """Class-level dependency declaration should honor isolated DA3 subprocess mode."""
    monkeypatch.setenv("TRANSFORMATION_PORTAL_DA3_PYTHON", "/tmp/fake-da3-python")

    assert DA3Backend.required_packages() == []


def test_da3_backend_subprocess_ensure_available_skips_local_dependency_checks(tmp_path):
    """Dedicated subprocess mode should not require local DA3 package imports."""
    from unittest.mock import MagicMock, patch

    python_executable = tmp_path / ".venv-da3" / "bin" / "python"
    python_executable.parent.mkdir(parents=True)
    python_executable.write_text("#!/bin/sh\n", encoding="utf-8")

    backend = DA3Backend(
        # Explicit research selection: repair 1.2 (#2066) moved the
        # no-selector default to da3_metric, and this test's argv
        # assertions exercise research model-key + license-flag propagation.
        _licensed_da3_config(
            depth_device="cpu",
            model_key="da3-research",
            da3_python_executable=str(python_executable),
            da3_subprocess_timeout_seconds=321,
        )
    )

    with patch.object(
        backend,
        "_ensure_local_package_available",
        side_effect=AssertionError("local DA3 package import should not be used"),
    ):
        with patch(
            "transformation_portal.depth.backends.da3.ensure_dependency_importable",
            side_effect=AssertionError("local dependency import checks should not be used"),
        ):
            with patch("transformation_portal.depth.backends.da3.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
                backend.ensure_available()

    command = mock_run.call_args.args[0]
    assert command[0] == str(python_executable.absolute())
    assert command[0] != sys.executable
    assert "--check" in command
    assert "METRIC_LARGE" in command
    assert "da3_research" in command
    assert "--non-commercial-ok" in command
    assert mock_run.call_args.kwargs["timeout"] == 321


def test_da3_backend_runtime_required_packages_drop_transformers_in_subprocess(tmp_path):
    """Instance dependency declaration should match subprocess-backed execution."""
    python_executable = tmp_path / ".venv-da3" / "bin" / "python"
    python_executable.parent.mkdir(parents=True)
    python_executable.write_text("#!/bin/sh\n", encoding="utf-8")

    backend = DA3Backend(
        _licensed_da3_config(
            depth_device="cpu",
            da3_python_executable=str(python_executable),
        )
    )

    assert backend.runtime_required_packages() == []


def test_da3_backend_subprocess_worker_env_sets_runtime_guards(monkeypatch, tmp_path):
    """Subprocess DA3 mode should add repo/runtime env needed on macOS."""
    import transformation_portal.depth.backends.da3 as da3_module

    python_executable = tmp_path / ".venv-da3" / "bin" / "python"
    python_executable.parent.mkdir(parents=True)
    python_executable.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setattr(
        da3_module,
        "CURRENT_PLATFORM",
        PlatformMatrix(PlatformOS.DARWIN, PlatformISA.ARM64, PlatformAccel.MPS),
    )
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "True")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "attacker-python"))
    monkeypatch.setenv("PYTHONSTARTUP", str(tmp_path / "attacker-startup.py"))
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "attacker-venv"))

    backend = DA3Backend(
        _licensed_da3_config(
            depth_device="cpu",
            da3_python_executable=str(python_executable),
        )
    )
    monkeypatch.setenv("TP_MODEL_LOCK_MANIFEST", str(tmp_path / "model-lock.yaml"))
    monkeypatch.setenv("TP_STRICT_MODEL_LOCK", "1")

    env = backend._build_worker_env()

    assert env["PYTHONPATH"] == str(backend._repo_src)
    assert env["PYTHONSAFEPATH"] == "1"
    assert env["PYTHONNOUSERSITE"] == "1"
    assert "PYTHONSTARTUP" not in env
    assert "VIRTUAL_ENV" not in env
    assert env["KMP_DUPLICATE_LIB_OK"] == "TRUE"
    assert env["MPLCONFIGDIR"].endswith(".runtime/mplconfig")
    assert env["TP_MODEL_LOCK_MANIFEST"] == str(tmp_path / "model-lock.yaml")
    assert env["TP_STRICT_MODEL_LOCK"] == "1"


def test_da3_backend_subprocess_model_lock_failure_never_falls_back_to_host(tmp_path):
    """Worker model-lock failures must stop without trying host dependencies."""
    from unittest.mock import MagicMock, patch

    python_executable = tmp_path / ".venv-da3" / "bin" / "python"
    python_executable.parent.mkdir(parents=True)
    python_executable.write_text("#!/bin/sh\n", encoding="utf-8")
    backend = DA3Backend(
        _licensed_da3_config(
            depth_device="cpu",
            da3_python_executable=str(python_executable),
        )
    )

    with patch.object(
        backend,
        "_ensure_local_package_available",
        side_effect=AssertionError("host DA3 must not be used as fallback"),
    ):
        with patch(
            "transformation_portal.depth.backends.da3.ensure_dependency_importable",
            side_effect=AssertionError("host dependencies must not be used as fallback"),
        ):
            with patch("transformation_portal.depth.backends.da3.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=1,
                    stdout="",
                    stderr=(
                        "ModelLockError: repo 'depth-anything/DA3NESTED-GIANT-LARGE-1.1' "
                        "revision mismatch with model lock manifest"
                    ),
                )
                with pytest.raises(ImportError) as exc_info:
                    backend.ensure_available()

    assert "Failure category: startup_failed" in str(exc_info.value)
    assert "revision mismatch with model lock manifest" in str(exc_info.value)
    assert mock_run.call_count == 1
    assert mock_run.call_args.args[0][0] == str(python_executable.absolute())


def test_da3_backend_subprocess_dependency_failure_reports_category(tmp_path):
    """Dependency failures should report the normalized subprocess category."""
    from unittest.mock import MagicMock, patch

    python_executable = tmp_path / ".venv-da3" / "bin" / "python"
    python_executable.parent.mkdir(parents=True)
    python_executable.write_text("#!/bin/sh\n", encoding="utf-8")

    backend = DA3Backend(
        _licensed_da3_config(
            depth_device="cpu",
            da3_python_executable=str(python_executable),
        )
    )

    with patch("transformation_portal.depth.backends.da3.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="ModuleNotFoundError: No module named 'transformers'",
        )
        with pytest.raises(ImportError, match="Failure category: dependency_missing"):
            backend.ensure_available()


def test_da3_backend_subprocess_launch_oserror_reports_category(tmp_path):
    """Launch-time OS errors should still map to a stable failure category."""
    from unittest.mock import patch

    python_executable = tmp_path / ".venv-da3" / "bin" / "python"
    python_executable.parent.mkdir(parents=True)
    python_executable.write_text("#!/bin/sh\n", encoding="utf-8")

    backend = DA3Backend(
        _licensed_da3_config(
            depth_device="cpu",
            da3_python_executable=str(python_executable),
            da3_subprocess_timeout_seconds=45,
        )
    )

    with patch(
        "transformation_portal.depth.backends.da3.subprocess.run",
        side_effect=PermissionError("Permission denied"),
    ) as mock_run:
        with pytest.raises(ImportError, match="Failure category: startup_failed"):
            backend.ensure_available()

    assert mock_run.call_args.kwargs["timeout"] == 45


def test_da3_backend_subprocess_protocol_error_reports_category(tmp_path):
    """Missing worker outputs should be normalized as protocol errors."""
    from unittest.mock import MagicMock, patch

    python_executable = tmp_path / ".venv-da3" / "bin" / "python"
    python_executable.parent.mkdir(parents=True)
    python_executable.write_text("#!/bin/sh\n", encoding="utf-8")

    backend = DA3Backend(
        _licensed_da3_config(
            depth_device="cpu",
            da3_python_executable=str(python_executable),
        )
    )

    def fake_run(command, **kwargs):
        del kwargs
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("transformation_portal.depth.backends.da3.subprocess.run", side_effect=fake_run):
        with pytest.raises(RuntimeError, match="Failure category: protocol_error"):
            backend.compute(np.zeros((4, 5, 3), dtype=np.uint8))


def test_da3_backend_subprocess_compute_returns_depth_result(tmp_path):
    """Subprocess worker output should map back to the DepthResult contract."""
    from unittest.mock import MagicMock, patch

    python_executable = tmp_path / ".venv-da3" / "bin" / "python"
    python_executable.parent.mkdir(parents=True)
    python_executable.write_text("#!/bin/sh\n", encoding="utf-8")

    backend = DA3Backend(
        # Explicit research selection: repair 1.2 (#2066) moved the
        # no-selector default to da3_metric, and this test's argv
        # assertions exercise research model-key + license-flag propagation.
        _licensed_da3_config(
            depth_device="cpu",
            model_key="da3-research",
            da3_python_executable=str(python_executable),
            da3_subprocess_timeout_seconds=123,
        )
    )

    seen_commands = []
    seen_timeouts = []

    def fake_run(command, **kwargs):
        seen_commands.append(command)
        seen_timeouts.append(kwargs["timeout"])
        if "--check" in command:
            return MagicMock(returncode=0, stdout="", stderr="")

        output_depth_path = Path(command[command.index("--output-depth") + 1])
        output_json_path = Path(command[command.index("--output-json") + 1])
        np.save(output_depth_path, np.full((4, 5), 0.5, dtype=np.float32), allow_pickle=False)
        output_json_path.write_text(
            json.dumps(
                {
                    "metadata": {
                        "resolved_model_id": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
                        "device": "cpu",
                    },
                    "device": "cpu",
                    "dtype": "float32",
                    "input_size": [4, 5],
                }
            ),
            encoding="utf-8",
        )
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("transformation_portal.depth.backends.da3.subprocess.run", side_effect=fake_run):
        result = backend.compute(np.zeros((4, 5, 3), dtype=np.uint8))

    assert isinstance(result, DepthResult)
    assert result.depth_units == "relative"
    assert result.depth_map.shape == (4, 5)
    assert result.device == "cpu"
    assert result.input_size == (4, 5)
    assert result.metadata["runner"]["mode"] == "subprocess"
    assert result.metadata["runner"]["python_executable"] == backend._python_executable
    assert any("normalized to relative" in warning for warning in result.warnings)
    assert len(seen_commands) == 2
    for command in seen_commands:
        assert "--model-variant" in command
        assert command[command.index("--model-variant") + 1] == "METRIC_LARGE"
        assert "--model-key" in command
        assert command[command.index("--model-key") + 1] == "da3_research"
        assert "--non-commercial-ok" in command
    assert seen_timeouts == [123, 123]


def test_da3_worker_inference_propagates_non_commercial_ack(monkeypatch, tmp_path):
    """Worker subprocess config should preserve parent non-commercial acknowledgement."""
    import transformation_portal.depth.backends.da3_worker as da3_worker
    import transformation_portal.lux_depth_v3.inference as inference_module

    captured = {}
    input_path = tmp_path / "input.png"
    output_depth = tmp_path / "depth.npy"
    output_json = tmp_path / "result.json"
    Image.new("RGB", (4, 5), color="white").save(input_path)

    class FakeEngine:
        def __init__(self, *, config, commercial_use, validate_license_strict, model_key, non_commercial_ok):
            captured["config_non_commercial_ok"] = config.non_commercial_ok
            captured["engine_non_commercial_ok"] = non_commercial_ok
            captured["model_key"] = model_key
            captured["commercial_use"] = commercial_use
            captured["validate_license_strict"] = validate_license_strict

        def predict(self, image):
            del image
            return SimpleNamespace(
                depth_map=np.full((4, 5), 0.25, dtype=np.float32),
                metadata={"device": "cpu"},
                original_image=np.zeros((4, 5, 3), dtype=np.uint8),
            )

    monkeypatch.setattr(inference_module, "DA3InferenceEngine", FakeEngine)

    exit_code = da3_worker._run_inference(
        input_image=input_path,
        output_depth=output_depth,
        output_json=output_json,
        model_variant_name="METRIC_LARGE",
        model_key="da3_research",
        device="cpu",
        use_coreml=False,
        non_commercial_ok=True,
    )

    assert exit_code == 0
    assert captured == {
        "config_non_commercial_ok": True,
        "engine_non_commercial_ok": True,
        "model_key": "da3_research",
        "commercial_use": True,
        "validate_license_strict": False,
    }


def test_da3_backend_subprocess_compute_launch_oserror_reports_category(tmp_path):
    """Inference launch OS errors should stay normalized for operators."""
    from unittest.mock import patch

    python_executable = tmp_path / ".venv-da3" / "bin" / "python"
    python_executable.parent.mkdir(parents=True)
    python_executable.write_text("#!/bin/sh\n", encoding="utf-8")

    backend = DA3Backend(
        _licensed_da3_config(
            depth_device="cpu",
            da3_python_executable=str(python_executable),
            da3_subprocess_timeout_seconds=17,
        )
    )
    backend._subprocess_available_checked = True

    with patch(
        "transformation_portal.depth.backends.da3.subprocess.run",
        side_effect=PermissionError("Permission denied"),
    ) as mock_run:
        with pytest.raises(RuntimeError, match="Failure category: startup_failed"):
            backend.compute(np.zeros((4, 5, 3), dtype=np.uint8))

    assert mock_run.call_args.kwargs["timeout"] == 17


def test_da3_backend_prepare_image_preserves_dark_uint8_arrays():
    """Dark uint8 images should not be reinterpreted as normalized floats."""
    backend = DA3Backend(_licensed_da3_config())
    image = np.array(
        [
            [[0, 1, 0], [1, 0, 1]],
            [[1, 1, 0], [0, 0, 1]],
        ],
        dtype=np.uint8,
    )

    image_pil, image_array = backend._prepare_image(image)

    assert image_pil.mode == "RGB"
    assert np.array_equal(image_array, image)


def test_da3_backend_prepare_image_scales_normalized_float_arrays():
    """Normalized float images should still be scaled into uint8 pixel space."""
    backend = DA3Backend(_licensed_da3_config())
    image = np.array(
        [
            [[0.0, 0.5, 1.0], [0.25, 0.75, 0.1]],
        ],
        dtype=np.float32,
    )

    _, image_array = backend._prepare_image(image)

    expected = np.array(
        [
            [[0, 127, 255], [63, 191, 25]],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(image_array, expected)


@pytest.mark.skipif(
    not can_run_da3_compute(),
    reason="DA3 compute requires depth_anything_3 + transformers + online mode",
)
def test_da3_backend_compute():
    """DA3Backend.compute() returns DepthResult."""
    config = _licensed_da3_config(depth_device="cpu")
    backend = DA3Backend(config)

    # Create test image
    image = Image.new("RGB", (64, 64), color="white")

    # Run inference
    result = backend.compute(image)

    assert isinstance(result, DepthResult)
    # DA3 may resize the input, so check that we got a depth map
    assert len(result.depth_map.shape) == 2  # 2D depth map
    assert result.depth_map.dtype == np.float32
    assert result.depth_units == "relative"
    assert result.focal_length_px is None  # DA3 doesn't provide focal length
    assert result.backend_id == "da3"


@pytest.mark.skipif(
    not can_run_da3_compute(),
    reason="DA3 compute requires depth_anything_3 + transformers + online mode",
)
def test_da3_backend_compute_numpy():
    """DA3Backend.compute() accepts numpy arrays."""
    backend = DA3Backend(_licensed_da3_config())

    # Create test image as numpy array
    image_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

    # Run inference
    result = backend.compute(image_array)

    assert isinstance(result, DepthResult)
    # DA3 may resize the input
    assert len(result.depth_map.shape) == 2  # 2D depth map


def test_da3_backend_cache_key():
    """DA3Backend generates consistent cache keys."""
    backend = DA3Backend(_licensed_da3_config())

    image = Image.new("RGB", (64, 64))

    key1 = backend.get_cache_key(image)
    key2 = backend.get_cache_key(image)

    assert key1 == key2
    assert key1.startswith("da3_")


def test_da3_backend_coreml_opt_in_fails_closed():
    """DA3 backends should reject CoreML opt-in during registry resolution."""
    with pytest.raises(BackendCapabilityError, match="CoreML backend is not supported"):
        DA3Backend(_licensed_da3_config(use_coreml_backend=True))


def test_da3_backend_registry_integration():
    """DA3Backend is registered in DepthBackendRegistry."""
    registry = DepthBackendRegistry()

    backends = registry.list_backends()
    assert "da3" in backends
    assert backends["da3"]["license_type"] == "model_dependent"
    assert backends["da3"]["requires_checkpoint"] is False


def test_da3_backend_registry_instantiation_without_config_defers_license_gate():
    """Registry metadata access should not enforce DA3 research-license acknowledgement."""
    registry = DepthBackendRegistry()

    backend = registry.get_backend("da3")

    assert isinstance(backend, DA3Backend)
    # Repair 1.2 (#2066): the no-selector default is the commercial-safe
    # metric model, so metadata instantiation resolves cleanly.
    assert backend._resolved_model_contract.canonical_key == "da3_metric"


@pytest.mark.skipif(
    not can_run_da3_compute(),
    reason="DA3 compute requires depth_anything_3 + transformers + online mode",
)
def test_da3_backend_via_registry():
    """DA3Backend can be instantiated via registry."""
    config = _licensed_da3_config(depth_device="cpu")
    registry = DepthBackendRegistry()

    backend = registry.get_backend("da3", config)

    assert isinstance(backend, DA3Backend)
    assert backend.name == "da3"


@pytest.mark.skipif(
    not can_run_da3_compute(),
    reason="DA3 compute requires depth_anything_3 + transformers + online mode",
)
def test_da3_backend_device_override():
    """DA3Backend respects device parameter in compute()."""
    config = _licensed_da3_config(depth_device="cpu")
    backend = DA3Backend(config)

    image = Image.new("RGB", (64, 64))

    # Should not raise even if device override is specified
    result = backend.compute(image, device="cpu")
    assert result.device == "cpu"


def test_da3_backend_unit_contract_metadata(monkeypatch):
    """DA3 adapter should expose source/output unit semantics in metadata."""
    config = _licensed_da3_config(depth_device="cpu")
    backend = DA3Backend(config)

    # Avoid dependency checks and heavy model loading.
    monkeypatch.setattr(backend, "ensure_available", lambda: None)
    backend._engine = SimpleNamespace(
        predict=lambda _image: SimpleNamespace(
            depth_map=np.ones((32, 32), dtype=np.float32),
            metadata={"resolved_model_id": "depth-anything/DA3NESTED-GIANT-LARGE-1.1"},
        )
    )

    result = backend.compute(Image.new("RGB", (32, 32), color="white"))

    assert result.depth_units == "relative"
    assert result.metadata["source_depth_units"] == "meters"
    assert result.metadata["output_depth_units"] == "relative"
    assert result.metadata["output_normalization"] == "minmax_0_1_per_image"
    assert any("normalized to relative" in warning for warning in result.warnings)


def test_da3_backend_uses_engine_effective_device_metadata(monkeypatch):
    """DA3 adapter should report the engine's effective runtime device."""
    config = _licensed_da3_config(depth_device="cpu")
    backend = DA3Backend(config)

    monkeypatch.setattr(backend, "ensure_available", lambda: None)
    backend._engine = SimpleNamespace(
        predict=lambda _image: SimpleNamespace(
            depth_map=np.ones((32, 32), dtype=np.float32),
            metadata={
                "backend": "da3",
                "device": "cpu",
                "resolved_model_id": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
            },
        )
    )

    result = backend.compute(Image.new("RGB", (32, 32), color="white"))

    assert result.device == "cpu"
    assert result.metadata["backend"] == "da3"
    assert result.metadata["device"] == "cpu"


def test_da3_backend_smoke_cpu_no_hidden_cuda(monkeypatch):
    """CPU DA3 inference path should not invoke CUDA implicitly."""
    pytest.importorskip("torch")
    import transformation_portal.lux_depth_v3.inference as inference_module

    _install_fake_depth_anything3(monkeypatch)
    monkeypatch.setattr(DA3Backend, "ensure_available", lambda self: None)
    monkeypatch.setattr(inference_module, "TRANSFORMERS_AVAILABLE", True)

    backend = DA3Backend(_licensed_da3_config(depth_device="cpu"))
    result = backend.compute(Image.new("RGB", (64, 64), color="white"))

    assert result.device == "cpu"
    assert result.depth_map.shape == (64, 64)


def test_da3_backend_smoke_mps_no_hidden_cuda(monkeypatch):
    """MPS DA3 inference path should not invoke CUDA implicitly."""
    torch = pytest.importorskip("torch")
    import transformation_portal.lux_depth_v3.inference as inference_module

    _install_fake_depth_anything3(monkeypatch)
    monkeypatch.setattr(DA3Backend, "ensure_available", lambda self: None)
    monkeypatch.setattr(inference_module, "TRANSFORMERS_AVAILABLE", True)

    if hasattr(torch.backends, "mps"):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    else:
        monkeypatch.setattr(torch.backends, "mps", SimpleNamespace(is_available=lambda: True), raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    backend = DA3Backend(_licensed_da3_config(depth_device="mps"))
    result = backend.compute(Image.new("RGB", (64, 64), color="white"))

    assert result.device == "mps"
    assert result.depth_map.shape == (64, 64)


def test_da3_backend_rejects_coreml_opt_in_on_apple_silicon(monkeypatch):
    """Apple Silicon should still fail closed for DA3 CoreML requests."""
    import transformation_portal.depth.backends.da3 as da3_module

    monkeypatch.setattr(
        da3_module,
        "CURRENT_PLATFORM",
        PlatformMatrix(PlatformOS.DARWIN, PlatformISA.ARM64, PlatformAccel.MPS),
    )

    with pytest.raises(BackendCapabilityError, match="CoreML backend is not supported"):
        DA3Backend(_licensed_da3_config(depth_device="cpu", use_coreml_backend=True))


def test_da3_backend_rejects_coreml_opt_in_off_apple_silicon(monkeypatch):
    """Intel/Linux lanes should also fail closed for DA3 CoreML requests."""
    import transformation_portal.depth.backends.da3 as da3_module

    monkeypatch.setattr(
        da3_module,
        "CURRENT_PLATFORM",
        PlatformMatrix(PlatformOS.DARWIN, PlatformISA.X86_64, PlatformAccel.CPU),
    )

    with pytest.raises(BackendCapabilityError, match="CoreML backend is not supported"):
        DA3Backend(_licensed_da3_config(depth_device="cpu", use_coreml_backend=True))
