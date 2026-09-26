"""Bounded Depth Pro subprocess transport with canonical execution authority."""

from __future__ import annotations

import hashlib
import io
import math
import os
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from transformation_portal.core.execution_plan import CanonicalExecutionPlan, decode_bounded_json_object
from transformation_portal.core.execution_plan_v2 import require_digest
from transformation_portal.depth.backends.depth_pro import DepthProBackend
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v3.config import EnhanceConfig
from transformation_portal.lux_depth_v3.execution_lifecycle import (
    backend_candidate_authority,
    consume_lux_execution_plan,
    consume_lux_worker_execution_plan,
    prepare_lux_execution,
    runtime_config_from_execution_plan,
)
from transformation_portal.lux_depth_v4.backend import _signal_process_group, require_process_supervisor

_MAX_LOG_BYTES = 1024 * 1024
_MAX_METADATA_BYTES = 256 * 1024
_MAX_HEADER_BYTES = 4096
_MAX_CHECKPOINT_BYTES = 3 * 1024**3
_WORKER_MODULE = "transformation_portal.depth.backends.depth_pro_worker"


def _hash_file(path: Path, maximum: int, *, checkpoint_callback: Callable[[], None] | None = None) -> tuple[str, int]:
    from transformation_portal.depth.backends.da3_runtime_identity import _hash_regular_file

    if checkpoint_callback is None:
        return _hash_regular_file(path, maximum_bytes=maximum)
    checkpoint_callback()
    digest, count = hashlib.sha256(), 0
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 <= before.st_size <= maximum:
            raise ValueError("Depth Pro identity input exceeds its regular-file byte bound")
        while True:
            checkpoint_callback()
            block = handle.read(1024 * 1024)
            if not block:
                break
            count += len(block)
            if count > maximum:
                raise ValueError("Depth Pro identity input exceeded its byte bound while hashing")
            digest.update(block)
        after = os.fstat(handle.fileno())
    fields = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    if count != before.st_size or fields(before) != fields(after) or fields(before) != fields(path.lstat()):
        raise ValueError("Depth Pro identity input changed during hashing")
    checkpoint_callback()
    return digest.hexdigest(), count


def _checkpoint_identity(path: Path, *, checkpoint_callback: Callable[[], None] | None = None) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    sha256, size = _hash_file(resolved, _MAX_CHECKPOINT_BYTES, checkpoint_callback=checkpoint_callback)
    if sha256 != DepthProBackend.EXPECTED_SHA256:
        raise ValueError("Depth Pro checkpoint differs from its pinned SHA-256")
    return {"path": str(resolved), "sha256": sha256, "size_bytes": size}


def prepare_authority(
    input_root: Path,
    input_files: list[Path],
    *,
    python_executable: Path,
    checkpoint: Path,
    device: str,
    non_commercial_ok: bool,
    accept_license: bool,
) -> bytes:
    """Freeze the existing core worker authority under an exact input selection."""
    if non_commercial_ok is not True or accept_license is not True:
        raise ValueError("Depth Pro requires explicit non-commercial and Apple research-license acknowledgements")
    if device not in {"cpu", "mps", "cuda"}:
        raise ValueError("Depth Pro requires an explicit cpu, mps, or cuda device")
    python = Path(python_executable).expanduser().absolute()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError("Depth Pro Python executable is unavailable")
    model = _checkpoint_identity(checkpoint)
    config = EnhanceConfig(
        depth_backend="depth_pro",
        depth_device=device,
        depth_pro_checkpoint_path=model["path"],
        depth_pro_python_executable=str(python),
        non_commercial_ok=True,
        accept_apple_depth_pro_research_license=True,
        enable_v2=False,
        allow_synthetic_fallback=False,
        depth_fallback="fail",
    )
    prepared = prepare_lux_execution(config, input_root, input_files)
    # Re-consume the exact bytes under the parent's original-input authority.
    consumed = consume_lux_execution_plan(prepared.canonical_plan_bytes, authorized_input_root=input_root)
    validate_authority(consumed.canonical_plan_bytes)
    return consumed.canonical_plan_bytes


def validate_authority(data: bytes) -> CanonicalExecutionPlan:
    """Validate replayable model/license intent without reading a live runtime."""
    if type(data) is not bytes:
        raise ValueError("Depth Pro authority must be canonical bytes")
    plan = consume_lux_worker_execution_plan(data)
    licenses = plan.license_acknowledgements
    if licenses.non_commercial_ok is not True or licenses.apple_depth_pro_research is not True:
        raise ValueError("Depth Pro authority lacks its research-license acknowledgements")
    if len(plan.backend_candidates) != 1 or plan.backend_candidates[0].backend_id != "depth_pro":
        raise ValueError("V6 Depth Pro authority requires one exact Depth Pro candidate")
    candidate = backend_candidate_authority(plan, "depth_pro")
    model = candidate.model_contract
    config = runtime_config_from_execution_plan(plan)
    if (
        model is None
        or model.backend_id != "depth_pro"
        or model.artifact_sha256 != DepthProBackend.EXPECTED_SHA256
        or model.artifact_path is None
        or not Path(model.artifact_path).is_absolute()
        or candidate.device not in {"cpu", "mps", "cuda"}
        or not config.depth_pro_python_executable
        or not Path(config.depth_pro_python_executable).is_absolute()
    ):
        raise ValueError("Depth Pro authority has incomplete executable model identity")
    return plan


def _environment(cache_root: Path) -> dict[str, str]:
    """Only fixed runtime controls and OS essentials cross the worker boundary."""
    env = {key: os.environ[key] for key in ("HOME", "PATH", "TMPDIR", "SYSTEMROOT") if key in os.environ}
    env.update(
        {
            "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
            "PYTHONPYCACHEPREFIX": str(cache_root),
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


def _runtime_probe() -> dict[str, Any]:
    """Hash installed bytes inside the selected interpreter, without model import."""
    from transformation_portal.depth.backends import da3_runtime_identity as identity

    installed = identity._installed_distribution_index()
    if not {"depth-pro", "torch", "torchvision", "numpy", "pillow", "jsonschema"} <= installed.keys():
        raise ValueError("Depth Pro runtime lacks required installed distributions")
    distributions = [
        identity._distribution_record(name, distribution=installed[name][0], verify_record_hashes=True)
        for name in sorted(installed)
    ]
    source_root = Path(__file__).resolve().parents[1]
    sources = []
    remaining = 256 * 1024 * 1024
    for path in sorted(source_root.rglob("*")):
        if path.suffix not in {".py", ".json", ".yaml"}:
            continue
        if len(sources) >= 8192 or path.is_symlink():
            raise ValueError("Depth Pro source inventory is unbounded or contains a link")
        digest, size = _hash_file(path, remaining)
        remaining -= size
        sources.append({"path": path.relative_to(source_root).as_posix(), "sha256": digest, "size_bytes": size})
    binary_hash, binary_size = _hash_file(Path(sys.executable).resolve(strict=True), 256 * 1024 * 1024)
    return {
        "python": sys.version,
        "python_prefix": sys.prefix,
        "python_binary_sha256": binary_hash,
        "python_binary_size_bytes": binary_size,
        "distributions": distributions,
        "source_sha256": hashlib.sha256(canonicalize_json(sources)).hexdigest(),
    }


def _read_regular(path: Path, maximum: int) -> bytes:
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum:
            raise ValueError("Depth Pro worker output exceeds its regular-file byte bound")
        data = handle.read(maximum + 1)
        after = os.fstat(handle.fileno())
    fields = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    if len(data) != before.st_size or fields(before) != fields(after) or fields(before) != fields(path.lstat()):
        raise ValueError("Depth Pro worker output changed during read")
    return data


def _run_bounded(
    command: list[str],
    root: Path,
    *,
    authority_bytes: bytes,
    outputs: dict[Path, int],
    memory_mib: int,
    checkpoint: Callable[[], None],
) -> None:
    supervisor = require_process_supervisor()
    stdin_path, log_path = root / "authority.json", root / "worker.log"
    stdin_path.write_bytes(authority_bytes)
    checkpoint()
    with stdin_path.open("rb") as stdin, log_path.open("w+b") as log:
        process = subprocess.Popen(
            command,
            stdin=stdin,
            stdout=log,
            stderr=log,
            cwd=root,
            env=_environment(root / "empty-bytecode"),
            start_new_session=True,
        )
        try:
            while True:
                checkpoint()
                try:
                    parent = supervisor.Process(process.pid)
                    members = [parent, *parent.children(recursive=True)]
                    resident = supervisor.Process().memory_info().rss
                    for member in members:
                        try:
                            resident += member.memory_info().rss
                        except supervisor.NoSuchProcess:
                            continue
                    if resident > memory_mib * 1024**2:
                        raise RuntimeError("Depth Pro worker exceeded its observed memory budget")
                except supervisor.NoSuchProcess:
                    pass
                if os.fstat(log.fileno()).st_size > _MAX_LOG_BYTES:
                    raise RuntimeError("Depth Pro worker diagnostic output exceeded its byte budget")
                for path, maximum in outputs.items():
                    if path.exists() and (path.is_symlink() or not path.is_file() or path.stat().st_size > maximum):
                        raise RuntimeError("Depth Pro worker output exceeded its byte budget")
                if process.poll() is not None:
                    break
                time.sleep(0.05)
            if process.returncode:
                log.seek(max(0, os.fstat(log.fileno()).st_size - 2000))
                message = log.read(2000).decode(errors="replace")
                raise RuntimeError(f"Depth Pro worker failed with exit {process.returncode}: {message}")
        finally:
            _signal_process_group(process, signal.SIGTERM)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                _signal_process_group(process, signal.SIGKILL)
                process.wait(timeout=2)
            # A terminated leader can leave child processes in its owned group.
            _signal_process_group(process, signal.SIGKILL)


def runtime_identity(
    python_executable: Path,
    checkpoint: Path,
    *,
    checkpoint_callback: Callable[[], None] | None = None,
    memory_mib: int = 16384,
) -> dict[str, Any]:
    """Materialize inference-only runtime identity; this cannot authorize DA3 cache."""
    python = Path(python_executable).expanduser().absolute()
    if type(memory_mib) is not int or memory_mib < 256:
        raise ValueError("Depth Pro runtime identity requires an admitted memory budget")
    deadline = time.monotonic() + 180

    def check() -> None:
        if checkpoint_callback is not None:
            checkpoint_callback()
        if time.monotonic() >= deadline:
            raise TimeoutError("Depth Pro runtime identity exceeded its time budget")

    check()
    model = _checkpoint_identity(checkpoint, checkpoint_callback=check)
    with tempfile.TemporaryDirectory(prefix="tp-depth-pro-identity-") as directory:
        root = Path(directory)
        output = root / "identity.json"
        script = (
            "from pathlib import Path; "
            "from transformation_portal.lux_depth_v6.depth_pro_backend import _runtime_probe; "
            "from transformation_portal.ingest.canonical_json import canonicalize_json; "
            "Path('identity.json').write_bytes(canonicalize_json(_runtime_probe()))"
        )
        _run_bounded(
            [str(python), "-c", script],
            root,
            authority_bytes=b"",
            outputs={output: _MAX_METADATA_BYTES},
            memory_mib=memory_mib,
            checkpoint=check,
        )
        identity = decode_bounded_json_object(_read_regular(output, _MAX_METADATA_BYTES))
        check()
    payload = {"schema": "tp.lux.depth_pro.runtime.v1", "python_executable": str(python), "checkpoint": model, **identity}
    validate_runtime_identity(payload)
    return payload


def validate_runtime_identity(payload: dict[str, Any]) -> None:
    """Validate retained inference-only identity without requiring its interpreter."""
    keys = {
        "schema",
        "python_executable",
        "checkpoint",
        "python",
        "python_prefix",
        "python_binary_sha256",
        "python_binary_size_bytes",
        "distributions",
        "source_sha256",
    }
    if type(payload) is not dict or set(payload) != keys or payload["schema"] != "tp.lux.depth_pro.runtime.v1":
        raise ValueError("Invalid closed Depth Pro runtime identity")
    for name in ("python_executable", "python_prefix"):
        value = payload[name]
        if type(value) is not str or not 1 <= len(value) <= 4096 or not Path(value).is_absolute():
            raise ValueError("Depth Pro runtime identity requires absolute runtime paths")
    if type(payload["python"]) is not str or not 1 <= len(payload["python"]) <= 4096:
        raise ValueError("Depth Pro runtime identity has invalid Python version")
    if type(payload["python_binary_size_bytes"]) is not int or not 0 < payload["python_binary_size_bytes"] <= 256 * 1024**2:
        raise ValueError("Depth Pro runtime identity has invalid interpreter size")
    require_digest(payload["python_binary_sha256"])
    require_digest(payload["source_sha256"])
    model = payload["checkpoint"]
    if (
        type(model) is not dict
        or set(model) != {"path", "sha256", "size_bytes"}
        or type(model["path"]) is not str
        or not Path(model["path"]).is_absolute()
        or model["sha256"] != DepthProBackend.EXPECTED_SHA256
        or type(model["size_bytes"]) is not int
        or not 0 < model["size_bytes"] <= _MAX_CHECKPOINT_BYTES
    ):
        raise ValueError("Depth Pro runtime identity has invalid pinned checkpoint")
    distributions = payload["distributions"]
    if type(distributions) is not list or not 1 <= len(distributions) <= 2048:
        raise ValueError("Depth Pro runtime identity requires bounded installed distributions")
    names = []
    for item in distributions:
        if type(item) is not dict or set(item) != {
            "name",
            "version",
            "direct_url_sha256",
            "record_sha256",
            "installed_files_sha256",
        }:
            raise ValueError("Depth Pro runtime identity has invalid distribution fields")
        for key in ("name", "version"):
            if type(item[key]) is not str or not 1 <= len(item[key]) <= 1024:
                raise ValueError("Depth Pro runtime identity has invalid distribution identity")
        for key in ("direct_url_sha256", "record_sha256", "installed_files_sha256"):
            require_digest(item[key])
        names.append(item["name"])
    if names != sorted(set(names)) or not {"depth-pro", "torch", "torchvision", "numpy", "pillow", "jsonschema"} <= set(names):
        raise ValueError("Depth Pro runtime identity has missing or duplicate distributions")


def validate_worker_metadata(metadata: dict, authority_bytes: bytes, shape: tuple[int, int]) -> None:
    """Bind retained model output to its exact canonical worker and native grid."""
    plan = validate_authority(authority_bytes)
    candidate = backend_candidate_authority(plan, "depth_pro")
    expected = {
        "plan_fingerprint_sha256": plan.plan_fingerprint_sha256,
        "candidate_id": "depth_pro",
        "model_backend_id": None,
        "executed_backend_id": "depth_pro",
    }
    if (
        type(metadata) is not dict
        or metadata.get("execution_authority") != expected
        or metadata.get("depth_units") != "meters"
        or metadata.get("dtype") != "float32"
        or metadata.get("input_size") != list(shape)
        or metadata.get("device") != candidate.device
        or metadata.get("warnings") != []
    ):
        raise ValueError("Depth Pro worker receipt differs from its canonical authority or native grid")
    provenance = metadata.get("provenance")
    model = candidate.model_contract
    assert model is not None
    if (
        type(provenance) is not dict
        or provenance.get("status") != "ok"
        or provenance.get("engine") != "apple_depth_pro"
        or provenance.get("device") != candidate.device
        or type(provenance.get("checkpoint")) is not dict
        or type(provenance.get("outputs")) is not dict
        or provenance.get("checkpoint", {}).get("sha256") != model.artifact_sha256
        or provenance.get("checkpoint", {}).get("path") != model.artifact_path
        or provenance.get("outputs", {}).get("depth_shape") != list(shape)
        or provenance.get("outputs", {}).get("depth_dtype") != "float32"
    ):
        raise ValueError("Depth Pro provenance differs from the pinned model contract")
    focal, fov = metadata.get("focal_length_px"), metadata.get("field_of_view_deg")
    if any(type(value) not in {float, int} or not math.isfinite(value) or value <= 0 for value in (focal, fov)):
        raise ValueError("Depth Pro receipt requires a finite positive estimated focal length and field of view")
    expected_fov = math.degrees(2 * math.atan(shape[1] / (2 * focal)))
    camera = provenance.get("camera")
    if (
        type(camera) is not dict
        or camera.get("focal_length_source") != "model_estimated"
        or camera.get("coordinate_space") != "input_image"
        or camera.get("focal_length_px") != focal
        or camera.get("fov_deg") != fov
        or not math.isclose(fov, expected_fov, rel_tol=1e-6, abs_tol=1e-6)
    ):
        raise ValueError("Depth Pro camera receipt is inconsistent with estimated focal length")


def _read_depth(path: Path, shape: tuple[int, int]) -> np.ndarray:
    data = _read_regular(path, shape[0] * shape[1] * 4 + _MAX_HEADER_BYTES)
    stream = io.BytesIO(data)
    version = np.lib.format.read_magic(stream)
    if version not in {(1, 0), (2, 0)}:
        raise ValueError("Depth Pro NPY format version is unsupported")
    reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
    actual_shape, fortran, dtype = reader(stream, max_header_size=_MAX_HEADER_BYTES)
    if actual_shape != shape or fortran or dtype != np.dtype("float32"):
        raise ValueError("Depth Pro NPY header differs from the float32 native grid")
    if len(data) - stream.tell() != shape[0] * shape[1] * 4:
        raise ValueError("Depth Pro NPY payload size differs from its native grid")
    return np.frombuffer(data, dtype=np.float32, offset=stream.tell()).reshape(shape).copy()


def infer(
    authority_bytes: bytes,
    model_input: bytes,
    shape: tuple[int, int],
    *,
    memory_mib: int,
    checkpoint: Callable[[], None],
) -> tuple[np.ndarray, dict]:
    """Run the frozen worker once with bounded transport and cooperative cancellation."""
    plan = validate_authority(authority_bytes)
    if type(shape) is not tuple or len(shape) != 2 or any(type(value) is not int or value <= 0 for value in shape):
        raise ValueError("Depth Pro native shape must contain two positive integers")
    if type(memory_mib) is not int or memory_mib < 256 or shape[0] * shape[1] * 32 > memory_mib * 1024**2:
        raise ValueError("Depth Pro native grid exceeds its memory budget")
    if type(model_input) is not bytes or not 0 < len(model_input) <= shape[0] * shape[1] * 4 + 65536:
        raise ValueError("Depth Pro model-input PNG exceeds its byte budget")
    with Image.open(io.BytesIO(model_input)) as image:
        if image.format != "PNG" or image.mode != "RGB" or image.size != (shape[1], shape[0]):
            raise ValueError("Depth Pro model input must be an RGB PNG on the native grid")
        image.verify()
    config = runtime_config_from_execution_plan(plan)
    with tempfile.TemporaryDirectory(prefix="tp-lux-v6-depth-pro-") as directory:
        root = Path(directory)
        input_path, depth_path, metadata_path = root / "input.png", root / "depth.npy", root / "metadata.json"
        input_path.write_bytes(model_input)
        command = [
            str(config.depth_pro_python_executable),
            "-m",
            _WORKER_MODULE,
            "--execution-plan-stdin",
            "--candidate-id",
            "depth_pro",
            "--input-image",
            str(input_path),
            "--output-depth",
            str(depth_path),
            "--output-json",
            str(metadata_path),
        ]
        _run_bounded(
            command,
            root,
            authority_bytes=authority_bytes,
            outputs={depth_path: shape[0] * shape[1] * 4 + _MAX_HEADER_BYTES, metadata_path: _MAX_METADATA_BYTES},
            memory_mib=memory_mib,
            checkpoint=checkpoint,
        )
        checkpoint()
        metadata = decode_bounded_json_object(_read_regular(metadata_path, _MAX_METADATA_BYTES))
        validate_worker_metadata(metadata, authority_bytes, shape)
        depth = _read_depth(depth_path, shape)
        checkpoint()
    return depth, metadata
