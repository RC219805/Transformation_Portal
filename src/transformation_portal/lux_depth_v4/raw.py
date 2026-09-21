"""Bounded, cancellable bridge to the existing strict RAW decoder.

The RAW runtime has no cache-authorizing dependency lock. Its decoded pixels and
ingest fingerprint are evidence, not permission to reuse a decoder cache. LibRaw
performs orientation and emits 16-bit linear sRGB; sensor highlight reconstruction
and recovery of precision beyond that decoder output are not claimed here.
"""

from __future__ import annotations

import hashlib
import os
import signal
import stat
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping, NotRequired, TypedDict

import numpy as np

from transformation_portal.core.execution_plan import decode_bounded_json_object
from transformation_portal.core.raw_runtime import RAW_WORKER_MODULE, resolve_raw_python_for_execution
from transformation_portal.ingest.canonical_json import dumps_json
from transformation_portal.lux_depth_v4.backend import (
    _signal_process_group,
    require_process_supervisor,
    validate_process_group_ownership,
    worker_environment,
)
from transformation_portal.lux_depth_v4.io import snapshot

_RAW_SUFFIXES = frozenset({".arw", ".cr2", ".dng", ".nef"})
_MAX_METADATA_BYTES = 64 * 1024
_MAX_LOG_BYTES = 1024 * 1024
_DECODE_PAYLOAD = {"gamma": 1.0, "bit_depth": 32, "strict_ingest": True, "demosaic": "AHD"}


class _WorkerArguments(TypedDict):
    python: str
    root: Path
    source_path: Path
    resources: Mapping[str, Any]
    cancellation: Callable[[], bool] | None
    deadline: float
    own_process_group: NotRequired[bool]


def _positive_limit(resources: Mapping[str, Any], key: str) -> int:
    value = resources.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"RAW {key} must be a positive integer")
    return value


def _stop_group(process: subprocess.Popen, *, own_process_group: bool = True) -> None:
    """Reap the leader and terminate descendants even when the leader exited."""
    _signal_process_group(process, signal.SIGTERM, own_process_group=own_process_group)
    try:
        process.wait(timeout=0.5)
    except subprocess.TimeoutExpired:
        pass
    finally:
        _signal_process_group(process, signal.SIGKILL, own_process_group=own_process_group)
        process.wait(timeout=3)


def _run_command(
    command: list[str],
    *,
    root: Path,
    resources: Mapping[str, Any],
    cancellation: Callable[[], bool] | None,
    deadline: float,
    log_name: str,
    own_process_group: bool = True,
) -> None:
    """Observe resource use; these watchdogs are not hard OS isolation."""
    validate_process_group_ownership(own_process_group)
    psutil = require_process_supervisor()
    memory_limit = _positive_limit(resources, "memory_mib") * 1024 * 1024
    output_limit = _positive_limit(resources, "max_output_bytes")
    if cancellation is not None and cancellation():
        raise RuntimeError("RAW decode cancelled")
    if time.monotonic() >= deadline:
        raise TimeoutError("RAW decode exceeded its wall-time budget")
    log_path = root / log_name
    with log_path.open("xb") as log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=root,
            env=worker_environment(),
            start_new_session=own_process_group,
        )
        try:
            while True:
                if cancellation is not None and cancellation():
                    raise RuntimeError("RAW decode cancelled")
                if time.monotonic() >= deadline:
                    raise TimeoutError("RAW decode exceeded its wall-time budget")
                try:
                    parent = psutil.Process(process.pid)
                    resident = sum(item.memory_info().rss for item in [parent, *parent.children(recursive=True)])
                    if resident > memory_limit:
                        raise RuntimeError("RAW worker exceeded its observed memory budget")
                except psutil.NoSuchProcess:
                    pass
                if os.fstat(log.fileno()).st_size > _MAX_LOG_BYTES:
                    raise RuntimeError("RAW diagnostic output exceeded its byte budget")
                # Output files have fixed names; source bytes are a separate input
                # budget. Check during execution, not only after deserialization.
                output_size = sum(path.lstat().st_size for path in root.iterdir() if path.name.endswith((".npy", ".json")))
                if output_size > output_limit:
                    raise RuntimeError("RAW worker output exceeded its observed disk budget")
                if process.poll() is not None:
                    break
                try:
                    process.wait(timeout=0.05)
                except subprocess.TimeoutExpired:
                    continue
            if process.returncode:
                with log_path.open("rb") as diagnostic:
                    diagnostic.seek(max(0, os.fstat(diagnostic.fileno()).st_size - 3000))
                    tail = diagnostic.read(3000).decode(errors="replace")
                raise RuntimeError(f"Strict RAW worker failed: {tail}")
        finally:
            _stop_group(process, own_process_group=own_process_group)


def _run_worker(
    command_name: str,
    *,
    python: str,
    root: Path,
    source_path: Path,
    resources: Mapping[str, Any],
    cancellation: Callable[[], bool] | None,
    deadline: float,
    own_process_group: bool = True,
) -> tuple[Path, dict[str, Any]]:
    output_array = root / f"{command_name}.npy"
    output_json = root / f"{command_name}.json"
    payload_path = root / "payload.json"
    command = [
        python,
        "-m",
        RAW_WORKER_MODULE,
        "--command",
        command_name,
        "--input-path",
        str(source_path),
        "--payload-json",
        str(payload_path),
        "--output-array",
        str(output_array),
        "--output-json",
        str(output_json),
    ]
    _run_command(
        command,
        root=root,
        resources=resources,
        cancellation=cancellation,
        deadline=deadline,
        log_name=f"{command_name}.log",
        own_process_group=own_process_group,
    )
    raw, _ = snapshot(root, output_json, maximum_bytes=_MAX_METADATA_BYTES)
    return output_array, decode_bounded_json_object(raw)


def _dimensions(metadata: Mapping[str, Any], max_pixels: int) -> tuple[int, int]:
    size = metadata.get("input_size")
    if not isinstance(size, list) or len(size) != 2 or any(type(item) is not int or item <= 0 for item in size):
        raise ValueError("RAW worker returned invalid image dimensions")
    if size[0] * size[1] > max_pixels:
        raise ValueError("RAW image dimensions exceed max_pixels")
    return size[0], size[1]


def _load_pixels(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    """Validate the NPY header before allocating any worker-declared tensor."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("RAW pixels must be a regular file without link aliases")
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle, max_header_size=4096)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle, max_header_size=4096)
        else:
            raise ValueError("Unsupported RAW pixel array format")
        if dtype != np.dtype("float32") or shape != (*expected_shape, 3) or fortran:
            raise ValueError("RAW pixels must be bounded contiguous float32 RGB")
        expected_bytes = expected_shape[0] * expected_shape[1] * 3 * 4
        if before.st_size - handle.tell() != expected_bytes:
            raise ValueError("RAW pixel payload length differs from its shape")
        data = handle.read(expected_bytes + 1)
        after = os.fstat(handle.fileno())
        if len(data) != expected_bytes or (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError("RAW pixels changed during loading")
    pixels = np.frombuffer(data, dtype=np.float32).reshape(*expected_shape, 3)
    if not np.isfinite(pixels).all():
        raise ValueError("RAW pixels must be finite")
    return pixels


def decode_raw(
    source_bytes: bytes,
    source_name: str,
    *,
    python: str,
    resources: Mapping[str, Any],
    cancellation: Callable[[], bool] | None = None,
    own_process_group: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode only the admitted immutable snapshot through the strict RAW worker."""
    validate_process_group_ownership(own_process_group)
    maximum_bytes = _positive_limit(resources, "max_input_bytes")
    max_pixels = _positive_limit(resources, "max_pixels")
    seconds = _positive_limit(resources, "wall_time_seconds")
    _positive_limit(resources, "memory_mib")
    _positive_limit(resources, "max_output_bytes")
    if not isinstance(source_bytes, bytes) or not 0 < len(source_bytes) <= maximum_bytes:
        raise ValueError("RAW source must be nonempty immutable bytes within max_input_bytes")
    suffix = Path(source_name).suffix.lower()
    if suffix not in _RAW_SUFFIXES:
        raise ValueError("Strict RAW decoder supports ARW, CR2, DNG, and NEF inputs")
    worker_python = resolve_raw_python_for_execution(python, start=Path(__file__))
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    deadline = time.monotonic() + seconds
    with tempfile.TemporaryDirectory(prefix="tp-lux-v4-raw-") as directory:
        root = Path(directory)
        source_path = root / f"source{suffix}"
        with source_path.open("xb") as handle:
            handle.write(source_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        source_path.chmod(0o400)
        (root / "payload.json").write_text(dumps_json(_DECODE_PAYLOAD, allow_nan=False), encoding="utf-8")
        common: _WorkerArguments = {
            "python": worker_python,
            "root": root,
            "source_path": source_path,
            "resources": resources,
            "cancellation": cancellation,
            "deadline": deadline,
        }
        if not own_process_group:
            common["own_process_group"] = False
        _, probe = _run_worker("probe", **common)
        unrotated_size = _dimensions(probe, max_pixels)
        if unrotated_size[0] * unrotated_size[1] * 12 + 8192 > resources["max_output_bytes"]:
            raise ValueError("Decoded RAW image exceeds the output byte budget")
        output_path, metadata = _run_worker("linear_decode", **common)
        oriented_size = _dimensions(metadata, max_pixels)
        if oriented_size not in {unrotated_size, unrotated_size[::-1]}:
            raise ValueError("RAW geometry changed beyond LibRaw orientation")
        if str(metadata.get("color_space", "")).lower() != "linear_srgb" or metadata.get("dtype") != "float32":
            raise ValueError("RAW decoder did not return linear-sRGB float32 pixels")
        fingerprint = metadata.get("ingest_fingerprint")
        if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint):
            raise ValueError("RAW decoder did not provide an ingest fingerprint")
        _, observed = snapshot(root, source_path, maximum_bytes=maximum_bytes, retain_bytes=False)
        if observed["sha256"] != source_digest:
            raise ValueError("RAW source snapshot changed during decode")
        pixels = _load_pixels(output_path, oriented_size)
    return pixels, {
        **metadata,
        "color_space": "linear_srgb",
        "source_sha256": source_digest,
        "source_bit_depth": 16,
        "native_sensor_bit_depth": None,
        "source_precision_basis": "LibRaw output_bps=16; native sensor precision unavailable",
        "orientation_normalized": True,
        "orientation_authority": "LibRaw default metadata orientation",
        "source_orientation": None,
        "raw_visible_shape": list(unrotated_size),
        "raw_decode": {**_DECODE_PAYLOAD, "white_balance": "camera", "auto_white_balance_fallback": False},
        "raw_highlight_policy": "LibRaw clip; no highlight reconstruction",
        "raw_runtime_cache_authorized": False,
        "resource_enforcement": "observed process-group memory and disk watchdog; not hard isolation",
    }
