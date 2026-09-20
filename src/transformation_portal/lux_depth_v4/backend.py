"""One cancellable isolated worker per photographic batch, with native depth."""

from __future__ import annotations

import hashlib
import io
import os
import selectors
import signal
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path
from types import ModuleType, TracebackType
from typing import Callable

import numpy as np
from PIL import Image

from transformation_portal.core.execution_plan import decode_bounded_json_object
from transformation_portal.core.execution_plan_v2 import ExecutionPlanV2
from transformation_portal.ingest.canonical_json import dumps_json


def require_process_supervisor() -> ModuleType:
    """Load the execution-only monitor without making core planning require ML."""
    try:
        import psutil
    except ImportError as exc:
        raise RuntimeError(
            "V4 supervised execution requires the native ML-core parent profile; run make install-ml-core"
        ) from exc
    return psutil


def worker_environment() -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("PYTHON") and key.upper() not in {"VIRTUAL_ENV", "__PYVENV_LAUNCHER__"}
    }
    env.update(
        {
            "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TP_STRICT_MODEL_LOCK": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    return env


def probe_device(interpreter: str, requested: str) -> str:
    if not interpreter or not Path(interpreter).is_file():
        raise RuntimeError("Governed DA3 Python is missing; set TRANSFORMATION_PORTAL_DA3_PYTHON")
    result = subprocess.run(
        [interpreter, "-m", "transformation_portal.lux_depth_v4.worker", "--probe-device", requested],
        env=worker_environment(),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"Device preflight failed: {result.stderr[-2000:]}")
    device = decode_bounded_json_object(result.stdout)["device"]
    if device not in {"cpu", "mps"} or requested == "mps" and device != "mps":
        raise RuntimeError("Device preflight returned an unauthorized device")
    return device


def _signal_process_group(process: subprocess.Popen, signum: int) -> None:
    """Reap a zombie leader before retrying macOS's transient EPERM response."""
    process.poll()
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        return
    except PermissionError:
        # A zombie session leader can produce EPERM until its parent reaps it.
        # Never suppress a permission failure while that leader is still alive.
        if process.poll() is None:
            raise
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass


class DA3Session:
    """Persistent batch worker; cancellation always terminates its process group."""

    def __init__(self, interpreter: str, plan: ExecutionPlanV2, *, cancellation: Callable[[], bool] | None = None) -> None:
        if not interpreter or not Path(interpreter).is_file():
            raise RuntimeError("Governed DA3 Python is missing; set TRANSFORMATION_PORTAL_DA3_PYTHON")
        self.plan = plan
        self.cancellation = cancellation
        self.resources = plan.to_payload()["resources"]
        self.deadline = time.monotonic() + self.resources["wall_time_seconds"]
        self._temporary = tempfile.TemporaryDirectory(prefix="tp-lux-v4-worker-")
        self.root = Path(self._temporary.name)
        self._stderr = (self.root / "worker.log").open("w+b")
        self.process = subprocess.Popen(
            [interpreter, "-m", "transformation_portal.lux_depth_v4.worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            env=worker_environment(),
            start_new_session=True,
        )
        self._buffer = b""
        assert self.process.stdin is not None
        os.set_blocking(self.process.stdin.fileno(), False)
        self._counter = 0
        try:
            from transformation_portal.depth.backends.da3_runtime_identity import DA3RuntimeIdentityEvidence

            raw = self._rpc({"command": "prepare", "plan": plan.to_payload()})["runtime_evidence"]
            self.runtime = DA3RuntimeIdentityEvidence.from_mapping(raw)
            if not self.runtime.cacheable:
                raise RuntimeError("Worker did not materialize complete runtime identity")
            backend = raw["backend_identity"]
            model = plan.to_payload()["model"]
            if (
                backend["model_canonical_key"] != model["canonical_key"]
                or backend["model_lock_revision"] != model["revision"]
                or backend["actual_device"] != plan.to_payload()["device"]
            ):
                raise RuntimeError("Worker runtime identity differs from carried plan")
        except BaseException:
            self.close()
            raise

    def checkpoint(self) -> None:
        if self.cancellation is not None and self.cancellation():
            raise RuntimeError("Photographic execution cancelled")
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Photographic worker exceeded its wall-time budget")
        psutil = require_process_supervisor()

        try:
            parent = psutil.Process(self.process.pid)
            processes = [parent, *parent.children(recursive=True)]
            resident = psutil.Process().memory_info().rss + sum(
                item.memory_info().rss for item in processes if item.is_running()
            )
            if resident > self.resources["memory_mib"] * 1024 * 1024:
                raise RuntimeError("Photographic worker exceeded its observed memory budget")
        except psutil.NoSuchProcess:
            pass
        if os.fstat(self._stderr.fileno()).st_size > 16 * 1024 * 1024:
            raise RuntimeError("Worker diagnostic output exceeded its budget")

    def _rpc(self, request: dict) -> dict:
        assert self.process.stdin is not None and self.process.stdout is not None
        self.checkpoint()
        frame = dumps_json(request, allow_nan=False).encode() + b"\n"
        if len(frame) > 8 * 1024 * 1024:
            raise RuntimeError("Worker request exceeds byte budget")
        offset = 0
        with selectors.DefaultSelector() as writer:
            writer.register(self.process.stdin, selectors.EVENT_WRITE)
            while offset < len(frame):
                self.checkpoint()
                if writer.select(timeout=0.1):
                    try:
                        offset += os.write(self.process.stdin.fileno(), frame[offset : offset + 65536])
                    except BlockingIOError:
                        continue
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            while b"\n" not in self._buffer:
                self.checkpoint()
                if selector.select(timeout=0.1):
                    chunk = os.read(self.process.stdout.fileno(), 65536)
                    if not chunk:
                        self._stderr.seek(0)
                        raise RuntimeError(f"DA3 worker exited: {self._stderr.read()[-3000:].decode(errors='replace')}")
                    self._buffer += chunk
                    if len(self._buffer) > 8 * 1024 * 1024:
                        raise RuntimeError("Worker response exceeds byte budget")
        line, self._buffer = self._buffer.split(b"\n", 1)
        response = decode_bounded_json_object(line)
        if response.get("ok") is not True:
            raise RuntimeError(f"DA3 worker failed: {response.get('error')}")
        return response["result"]

    def verify(self) -> None:
        if self._rpc({"command": "verify"}) != {"verified": True}:
            raise RuntimeError("Worker runtime verification failed")

    def compute(self, proxy: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        self._counter += 1
        input_path = self.root / f"input-{self._counter}.png"
        output_path = self.root / f"depth-{self._counter}.npz"
        Image.fromarray(proxy).save(input_path)
        response = self._rpc({"command": "infer", "input": str(input_path), "output": str(output_path)})
        if output_path.stat().st_size > 64 * 1024 * 1024:
            raise RuntimeError("Worker depth artifact exceeds byte budget")
        raw = output_path.read_bytes()
        if (
            hashlib.sha256(raw).hexdigest() != response["sha256"]
            or len(raw) != response["size_bytes"]
            or response["runtime_identity_sha256"] != self.runtime.runtime_identity_sha256
            or response["plan_fingerprint_sha256"] != self.plan.plan_fingerprint_sha256
        ):
            raise RuntimeError("Worker output identity mismatch")
        with zipfile.ZipFile(io.BytesIO(raw)) as container:
            members = container.infolist()
            expected_names = {"native_depth.npy", "confidence.npy"}
            if not 1 <= len(members) <= 2 or len({member.filename for member in members}) != len(members):
                raise RuntimeError("Worker depth archive has an invalid member inventory")
            for member in members:
                if member.filename not in expected_names or member.file_size > proxy.shape[0] * proxy.shape[1] * 4 + 1024:
                    raise RuntimeError("Worker depth archive exceeds its expanded array budget")
                with container.open(member) as source:
                    version = np.lib.format.read_magic(source)
                    if version == (1, 0):
                        shape, _fortran, dtype = np.lib.format.read_array_header_1_0(source, max_header_size=1024)
                    elif version == (2, 0):
                        shape, _fortran, dtype = np.lib.format.read_array_header_2_0(source, max_header_size=1024)
                    else:
                        raise RuntimeError("Unsupported worker array header")
                    if shape != proxy.shape[:2] or dtype != np.dtype("float32"):
                        raise RuntimeError("Worker array header differs from prepared proxy")
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            if set(archive.files) not in ({"native_depth"}, {"native_depth", "confidence"}):
                raise RuntimeError("Worker returned unknown depth arrays")
            arrays = {name: archive[name].copy() for name in archive.files}
        if any(
            array.dtype != np.float32 or array.shape != proxy.shape[:2] or not np.isfinite(array).all()
            for array in arrays.values()
        ):
            raise RuntimeError("Worker returned invalid depth arrays")
        input_path.unlink()
        output_path.unlink()
        return arrays, response

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        # Descendants can outlive an exited leader. Signal the dedicated group.
        _signal_process_group(self.process, signal.SIGTERM)
        if self.process.poll() is None:
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
        _signal_process_group(self.process, signal.SIGKILL)
        self.process.wait(timeout=3)
        for handle in (self.process.stdin, self.process.stdout, self._stderr):
            if handle is not None:
                handle.close()
        self._temporary.cleanup()

    def __enter__(self) -> DA3Session:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()
