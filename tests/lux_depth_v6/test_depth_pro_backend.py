"""Native Depth Pro transport rejects mismatched authority and bounded outputs."""

import copy
import io
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from transformation_portal.depth.backends.depth_pro import DepthProBackend
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v6 import depth_pro_backend as backend

pytestmark = pytest.mark.unit


@pytest.fixture
def authority(tmp_path, monkeypatch):
    root = tmp_path / ".runtime" / "input"
    root.mkdir(parents=True)
    source = root / "source.png"
    Image.new("RGB", (5, 3)).save(source)
    checkpoint = tmp_path / "depth_pro.pt"
    checkpoint.write_bytes(b"test-only-checkpoint")
    monkeypatch.setattr(
        backend,
        "_checkpoint_identity",
        lambda path: {"path": str(path), "sha256": DepthProBackend.EXPECTED_SHA256, "size_bytes": 20},
    )
    return backend.prepare_authority(
        root,
        [source],
        python_executable=Path(sys.executable),
        checkpoint=checkpoint,
        device="cpu",
        non_commercial_ok=True,
        accept_license=True,
    )


def _metadata(authority, shape=(3, 5)):
    plan = backend.validate_authority(authority)
    model = backend.backend_candidate_authority(plan, "depth_pro").model_contract
    focal = 6.5
    fov = math.degrees(2 * math.atan(shape[1] / (2 * focal)))
    return {
        "depth_units": "meters",
        "dtype": "float32",
        "device": "cpu",
        "input_size": list(shape),
        "focal_length_px": focal,
        "field_of_view_deg": fov,
        "warnings": [],
        "execution_authority": {
            "plan_fingerprint_sha256": plan.plan_fingerprint_sha256,
            "candidate_id": "depth_pro",
            "model_backend_id": None,
            "executed_backend_id": "depth_pro",
        },
        "provenance": {
            "status": "ok",
            "engine": "apple_depth_pro",
            "device": "cpu",
            "checkpoint": {"path": model.artifact_path, "sha256": model.artifact_sha256},
            "outputs": {"depth_shape": list(shape), "depth_dtype": "float32"},
            "camera": {
                "focal_length_px": focal,
                "fov_deg": fov,
                "focal_length_source": "model_estimated",
                "coordinate_space": "input_image",
            },
        },
    }


def _model_input(shape=(3, 5)):
    buffer = io.BytesIO()
    Image.new("RGB", (shape[1], shape[0])).save(buffer, format="PNG")
    return buffer.getvalue()


def _worker(monkeypatch, authority, array=None, metadata=None):
    if array is None:
        array = np.arange(15, dtype=np.float32).reshape(3, 5) + 1
    if metadata is None:
        metadata = _metadata(authority)

    def run(command, root, **kwargs):
        assert "--execution-plan-stdin" in command
        assert "--checkpoint" not in command
        assert kwargs["authority_bytes"] == authority
        np.save(root / "depth.npy", array, allow_pickle=False)
        (root / "metadata.json").write_bytes(canonicalize_json(metadata))

    monkeypatch.setattr(backend, "_run_bounded", run)


def test_authority_is_canonical_and_accepts_explicit_hidden_input(authority):
    plan = backend.validate_authority(authority)
    assert plan.to_canonical_json().encode() == authority
    assert plan.license_acknowledgements.non_commercial_ok is True
    assert plan.license_acknowledgements.apple_depth_pro_research is True
    assert [item.backend_id for item in plan.backend_candidates] == ["depth_pro"]
    with pytest.raises(ValueError, match="canonical"):
        backend.validate_authority(authority + b"\n")


@pytest.mark.parametrize("non_commercial,license_ok", [(False, True), (True, False), (1, True), (True, "true")])
def test_admission_requires_exact_acknowledgements(tmp_path, non_commercial, license_ok):
    with pytest.raises(ValueError, match="acknowledgements"):
        backend.prepare_authority(
            tmp_path,
            [],
            python_executable=Path(sys.executable),
            checkpoint=tmp_path / "missing.pt",
            device="cpu",
            non_commercial_ok=non_commercial,
            accept_license=license_ok,
        )


def test_actual_checkpoint_hash_is_pinned(tmp_path):
    model = tmp_path / "forged.pt"
    model.write_bytes(b"not-the-authorized-model")
    with pytest.raises(ValueError, match="pinned SHA"):
        backend._checkpoint_identity(model)


def test_infer_preserves_exact_float32_grid(authority, monkeypatch):
    _worker(monkeypatch, authority)
    checkpoints = []
    depth, metadata = backend.infer(
        authority, _model_input(), (3, 5), memory_mib=4096, checkpoint=lambda: checkpoints.append(True)
    )
    assert depth.dtype == np.float32
    np.testing.assert_array_equal(depth, np.arange(15, dtype=np.float32).reshape(3, 5) + 1)
    assert metadata == _metadata(authority)
    assert checkpoints


@pytest.mark.parametrize("array", [np.ones((3, 5), dtype=np.float64), np.ones((5, 3), dtype=np.float32)])
def test_infer_rejects_worker_dtype_or_shape(authority, monkeypatch, array):
    _worker(monkeypatch, authority, array=array)
    with pytest.raises(ValueError, match="NPY header"):
        backend.infer(authority, _model_input(), (3, 5), memory_mib=4096, checkpoint=lambda: None)


@pytest.mark.parametrize(
    "field,value",
    [
        ("depth_units", "relative"),
        ("dtype", "float64"),
        ("device", "mps"),
        ("input_size", [5, 3]),
        ("execution_authority", {"candidate_id": "depth_pro"}),
        ("warnings", ["fallback"]),
        ("focal_length_px", float("nan")),
        ("focal_length_px", True),
        ("field_of_view_deg", 360),
    ],
)
def test_worker_metadata_rejects_authority_and_camera_drift(authority, field, value):
    metadata = _metadata(authority)
    metadata[field] = value
    with pytest.raises(ValueError):
        backend.validate_worker_metadata(metadata, authority, (3, 5))


def test_model_provenance_cannot_borrow_other_checkpoint(authority):
    metadata = copy.deepcopy(_metadata(authority))
    metadata["provenance"]["checkpoint"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="pinned model"):
        backend.validate_worker_metadata(metadata, authority, (3, 5))


def test_native_grid_budget_rejected_before_worker(authority, monkeypatch):
    monkeypatch.setattr(backend, "_run_bounded", lambda *_args, **_kwargs: pytest.fail("worker must not start"))
    with pytest.raises(ValueError, match="memory budget"):
        backend.infer(authority, _model_input(), (10000, 10000), memory_mib=256, checkpoint=lambda: None)
    with pytest.raises(ValueError, match="native grid"):
        backend.infer(authority, _model_input(), (5, 3), memory_mib=4096, checkpoint=lambda: None)


def test_npy_trailing_data_and_links_rejected(tmp_path):
    output = tmp_path / "depth.npy"
    np.save(output, np.ones((3, 5), dtype=np.float32), allow_pickle=False)
    with output.open("ab") as stream:
        stream.write(b"unbound-data")
    with pytest.raises(ValueError, match="payload size"):
        backend._read_depth(output, (3, 5))
    link = tmp_path / "linked.npy"
    link.symlink_to(output)
    with pytest.raises(OSError):
        backend._read_depth(link, (3, 5))


def test_worker_environment_discards_python_and_loader_injection(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/untrusted")
    monkeypatch.setenv("PYTHONSTARTUP", "/untrusted/startup.py")
    monkeypatch.setenv("DYLD_LIBRARY_PATH", "/untrusted")
    env = backend._environment(tmp_path)
    assert env["PYTHONPATH"] == str(Path(backend.__file__).resolve().parents[2])
    assert env["PYTHONPYCACHEPREFIX"] == str(tmp_path)
    assert "DYLD_LIBRARY_PATH" not in env
    assert "PYTHONSTARTUP" not in env


def _runtime_identity():
    return {
        "schema": "tp.lux.depth_pro.runtime.v1",
        "python_executable": "/runtime/bin/python",
        "checkpoint": {"path": "/model/depth_pro.pt", "sha256": DepthProBackend.EXPECTED_SHA256, "size_bytes": 100},
        "python": "3.12.14",
        "python_prefix": "/runtime",
        "python_binary_sha256": "1" * 64,
        "python_binary_size_bytes": 100,
        "source_sha256": "2" * 64,
        "distributions": [
            {
                "name": name,
                "version": "1.0.0",
                "direct_url_sha256": "3" * 64,
                "record_sha256": "4" * 64,
                "installed_files_sha256": "5" * 64,
            }
            for name in ("depth-pro", "jsonschema", "numpy", "pillow", "torch", "torchvision")
        ],
    }


def test_retained_runtime_validation_needs_no_local_model_or_environment():
    backend.validate_runtime_identity(_runtime_identity())


def test_runtime_identity_honors_caller_cancellation_before_model_read(tmp_path):
    def cancelled():
        raise TimeoutError("caller deadline expired")

    with pytest.raises(TimeoutError, match="caller deadline expired"):
        backend.runtime_identity(Path(sys.executable), tmp_path / "missing.pt", checkpoint_callback=cancelled)


def test_runtime_identity_preserves_caller_memory_and_guard(monkeypatch):
    payload = _runtime_identity()
    calls = []

    def model(_path, *, checkpoint_callback):
        checkpoint_callback()
        return payload["checkpoint"]

    def probe(_command, root, *, memory_mib, checkpoint, **_kwargs):
        assert memory_mib == 512
        checkpoint()
        retained = {key: value for key, value in payload.items() if key not in {"schema", "python_executable", "checkpoint"}}
        (root / "identity.json").write_bytes(canonicalize_json(retained))

    monkeypatch.setattr(backend, "_checkpoint_identity", model)
    monkeypatch.setattr(backend, "_run_bounded", probe)
    result = backend.runtime_identity(
        Path("/runtime/bin/python"),
        Path("/model/depth_pro.pt"),
        checkpoint_callback=lambda: calls.append(True),
        memory_mib=512,
    )
    assert result == payload
    assert len(calls) >= 4


def test_checkpoint_hash_checks_cancellation_between_chunks(tmp_path):
    model = tmp_path / "model.pt"
    model.write_bytes(b"x" * (3 * 1024**2))
    calls = 0

    def cancelled():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise TimeoutError("cancelled during hash")

    with pytest.raises(TimeoutError, match="cancelled during hash"):
        backend._hash_file(model, 4 * 1024**2, checkpoint_callback=cancelled)
    assert calls == 3


@pytest.mark.parametrize("mutation", ["checkpoint", "duplicate", "missing", "digest", "unknown", "relative"])
def test_retained_runtime_identity_is_closed(mutation):
    payload = _runtime_identity()
    if mutation == "checkpoint":
        payload["checkpoint"]["sha256"] = "0" * 64
    elif mutation == "duplicate":
        payload["distributions"].append(payload["distributions"][-1])
    elif mutation == "missing":
        payload["distributions"].pop()
    elif mutation == "digest":
        payload["source_sha256"] = "untrusted"
    elif mutation == "unknown":
        payload["cache_authorized"] = True
    else:
        payload["python_executable"] = "python"
    with pytest.raises(ValueError):
        backend.validate_runtime_identity(payload)


def test_real_worker_cancellation_reaps_process_group(tmp_path):
    pytest.importorskip("psutil", reason="Native process supervision requires the optional ML-core profile")
    counter = 0

    def cancel():
        nonlocal counter
        counter += 1
        if counter >= 3:
            raise TimeoutError("cancelled by parent")

    with pytest.raises(TimeoutError, match="cancelled by parent"):
        backend._run_bounded(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            tmp_path,
            authority_bytes=b"{}",
            outputs={},
            memory_mib=16384,
            checkpoint=cancel,
        )


def test_real_worker_oversized_output_is_rejected(tmp_path):
    pytest.importorskip("psutil", reason="Native process supervision requires the optional ML-core profile")
    output = tmp_path / "oversized.bin"
    with pytest.raises(RuntimeError, match="output exceeded"):
        backend._run_bounded(
            [sys.executable, "-c", "from pathlib import Path; Path('oversized.bin').write_bytes(b'x'*1000)"],
            tmp_path,
            authority_bytes=b"{}",
            outputs={output: 32},
            memory_mib=16384,
            checkpoint=lambda: None,
        )
