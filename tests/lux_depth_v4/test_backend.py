"""Native worker boundaries, deterministic sampling, and cancellable IPC."""

from __future__ import annotations

import hashlib
import io
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from transformation_portal.lux_depth_v4 import backend, worker

pytestmark = pytest.mark.unit


def _process_session(script, *, timeout=3, cancellation=None):
    pytest.importorskip("psutil", reason="Native process supervision requires the optional ML-core profile")
    session = backend.DA3Session.__new__(backend.DA3Session)
    session._temporary = tempfile.TemporaryDirectory(prefix="tp-v4-ipc-test-")
    session.root = Path(session._temporary.name)
    session._stderr = (session.root / "stderr.log").open("w+b")
    session.process = subprocess.Popen(
        [sys.executable, "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=session._stderr,
        start_new_session=True,
    )
    session._buffer = b""
    session.resources = {"memory_mib": 1024}
    session.deadline = time.monotonic() + timeout
    session.cancellation = cancellation
    os.set_blocking(session.process.stdin.fileno(), False)
    return session


def test_blocked_request_write_obeys_timeout():
    session = _process_session("import time; time.sleep(30)", timeout=0.2)
    try:
        with pytest.raises(TimeoutError, match="wall-time"):
            session._rpc({"command": "prepare", "plan": "x" * 2_000_000})
    finally:
        session.close()
    assert session.process.poll() is not None


def test_blocked_request_write_obeys_cancellation():
    start = time.monotonic()
    session = _process_session("import time; time.sleep(30)", cancellation=lambda: time.monotonic() - start > 0.2)
    try:
        with pytest.raises(RuntimeError, match="cancelled"):
            session._rpc({"command": "prepare", "plan": "x" * 2_000_000})
    finally:
        session.close()


def test_rpc_returns_response_and_rejects_duplicate_keys():
    session = _process_session(
        'import sys; sys.stdin.readline(); print(\'{"ok":true,"result":{"verified":true}}\', flush=True)'
    )
    try:
        assert session._rpc({"command": "verify"}) == {"verified": True}
    finally:
        session.close()
    session = _process_session('import sys; sys.stdin.readline(); print(\'{"ok":false,"ok":true,"result":{}}\', flush=True)')
    try:
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            session._rpc({"command": "verify"})
    finally:
        session.close()


def test_close_terminates_descendants_after_leader_exit_and_is_idempotent(tmp_path):
    psutil = pytest.importorskip("psutil", reason="Native process supervision requires the optional ML-core profile")
    pid_file = tmp_path / "child.pid"
    session = _process_session(
        "import pathlib, subprocess, sys; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
        f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid))"
    )
    session.process.wait(timeout=3)
    pid = int(pid_file.read_text())
    session.close()
    session.close()
    try:
        psutil.Process(pid).wait(timeout=3)
    except psutil.NoSuchProcess:
        pass


def _array_bytes(shape=(14, 28), dtype="<f4", *, header_only=False):
    buffer = io.BytesIO()
    if header_only:
        np.lib.format.write_array_header_1_0(buffer, {"shape": shape, "fortran_order": False, "descr": dtype})
    else:
        np.save(buffer, np.ones(shape, dtype=dtype), allow_pickle=False)
    return buffer.getvalue()


def _archive_session(tmp_path, members, *, response_updates=None):
    session = backend.DA3Session.__new__(backend.DA3Session)
    session.root = tmp_path
    session._counter = 0
    session.runtime = SimpleNamespace(runtime_identity_sha256="a" * 64)
    session.plan = SimpleNamespace(plan_fingerprint_sha256="b" * 64)

    def rpc(request):
        path = Path(request["output"])
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, payload in members:
                archive.writestr(name, payload)
        payload = path.read_bytes()
        return {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
            "runtime_identity_sha256": "a" * 64,
            "plan_fingerprint_sha256": "b" * 64,
            **(response_updates or {}),
        }

    session._rpc = rpc
    return session


def test_worker_archive_accepts_native_depth_without_invented_confidence(tmp_path):
    session = _archive_session(tmp_path, [("native_depth.npy", _array_bytes())])
    arrays, _ = session.compute(np.zeros((14, 28, 3), np.uint8))
    assert set(arrays) == {"native_depth"}
    np.testing.assert_array_equal(arrays["native_depth"], np.ones((14, 28), np.float32))


@pytest.mark.parametrize(
    "members, error",
    [
        ([("native_depth.npy", _array_bytes((1_000_000, 1_000_000), header_only=True))], "header differs"),
        ([("native_depth.npy", _array_bytes(dtype="|O", header_only=True))], "header differs"),
        ([("native_depth.npy", b"x" * 100_000)], "expanded array budget"),
        ([("other.npy", _array_bytes())], "expanded array budget"),
        ([], "member inventory"),
    ],
)
def test_worker_archive_rejects_unbounded_payload_before_numpy_load(tmp_path, monkeypatch, members, error):
    session = _archive_session(tmp_path, members)

    def forbidden_load(*_args, **_kwargs):
        pytest.fail("np.load must not run before header and expanded-size validation")

    monkeypatch.setattr(np, "load", forbidden_load)
    with pytest.raises(RuntimeError, match=error):
        session.compute(np.zeros((14, 28, 3), np.uint8))


@pytest.mark.parametrize("field", ["sha256", "runtime_identity_sha256", "plan_fingerprint_sha256"])
def test_worker_archive_rejects_unbound_receipt(tmp_path, field):
    session = _archive_session(tmp_path, [("native_depth.npy", _array_bytes())], response_updates={field: "f" * 64})
    with pytest.raises(RuntimeError, match="identity mismatch"):
        session.compute(np.zeros((14, 28, 3), np.uint8))


def _native_worker(monkeypatch, *, confidence=None):
    from transformation_portal.depth.backends import da3_worker

    native_worker = worker.NativeDepthWorker.__new__(worker.NativeDepthWorker)
    native_worker.plan = SimpleNamespace(plan_fingerprint_sha256="b" * 64)
    native_worker.evidence = SimpleNamespace(runtime_identity_sha256="a" * 64)
    native_worker.verify = lambda: None
    monkeypatch.setattr(da3_worker, "_seed_isolated_inference", lambda _image: np.random.seed(317))

    class Engine:
        def __init__(self):
            self.model = None

        def _load_model(self):
            if self.model is None:
                np.random.random(117)
                self.model = self

        def inference(self, images, *, process_res, process_res_method):
            image = images[0]
            assert process_res == max(image.size)
            assert process_res_method == "upper_bound_resize"
            return SimpleNamespace(depth=[np.random.random((image.height, image.width)).astype(np.float32)], conf=confidence)

    native_worker.engine = Engine()
    return native_worker


def test_native_worker_cold_and_warm_sampling_are_equal(tmp_path, monkeypatch):
    native_worker = _native_worker(monkeypatch)
    input_path = tmp_path / "proxy.png"
    Image.fromarray(np.zeros((14, 28, 3), np.uint8)).save(input_path)
    first = tmp_path / "cold.npz"
    second = tmp_path / "warm.npz"
    native_worker.infer(input_path, first)
    native_worker.infer(input_path, second)
    with np.load(first) as cold, np.load(second) as warm:
        np.testing.assert_array_equal(cold["native_depth"], warm["native_depth"])


@pytest.mark.parametrize(
    "confidence, available",
    [(None, False), ([np.ones((14, 28), np.float32) * 5], False), ([np.ones((14, 28), np.float32) * 0.7], False)],
)
def test_native_worker_confidence_remains_unavailable_unless_bounded(tmp_path, monkeypatch, confidence, available):
    native_worker = _native_worker(monkeypatch, confidence=confidence)
    input_path = tmp_path / "proxy.png"
    Image.fromarray(np.zeros((14, 28, 3), np.uint8)).save(input_path)
    report = native_worker.infer(input_path, tmp_path / "depth.npz")
    assert report["confidence_available"] is available


def test_worker_rejects_proxy_dimensions_before_pixel_decode(monkeypatch, tmp_path):
    native_worker = _native_worker(monkeypatch)

    class OversizedImage:
        mode = "RGB"
        size = (140_000, 140_000)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def load(self):
            pytest.fail("Oversized proxy pixels must not be decoded")

    monkeypatch.setattr(Image, "open", lambda _path: OversizedImage())
    with pytest.raises(ValueError, match="bounded RGB model proxy"):
        native_worker.infer(tmp_path / "large.png", tmp_path / "result.npz")


def test_worker_environment_removes_ambient_python_import_paths(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/untrusted")
    monkeypatch.setenv("PYTHONHOME", "/untrusted")
    monkeypatch.setenv("VIRTUAL_ENV", "/untrusted")
    env = backend.worker_environment()
    assert env["PYTHONPATH"] != "/untrusted"
    assert "PYTHONHOME" not in env and "VIRTUAL_ENV" not in env
    assert env["HF_HUB_OFFLINE"] == env["TP_STRICT_MODEL_LOCK"] == "1"
