"""Strict RAW ingest preserves snapshots and fails closed on resource limits."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from transformation_portal.lux_depth_v4 import raw

pytestmark = pytest.mark.unit


@pytest.fixture
def resources():
    return {
        "max_pixels": 10000,
        "max_input_bytes": 1024,
        "max_output_bytes": 1024 * 1024,
        "wall_time_seconds": 10,
        "memory_mib": 512,
    }


def _worker(monkeypatch, *, pixels=None, dimensions=(4, 7), metadata_updates=None, mutate_source=False):
    calls = []
    pixels = np.linspace(0, 1, 84, dtype=np.float32).reshape((7, 4, 3)) if pixels is None else pixels

    def run(command_name, **kwargs):
        calls.append(command_name)
        assert kwargs["source_path"].read_bytes() == b"immutable raw bytes"
        assert kwargs["source_path"].name == "source.dng"
        assert kwargs["source_path"].stat().st_mode & 0o777 == 0o400
        payload = json.loads((kwargs["root"] / "payload.json").read_text())
        assert payload == {"gamma": 1.0, "bit_depth": 32, "strict_ingest": True, "demosaic": "AHD"}
        output = kwargs["root"] / (command_name + ".npy")
        if command_name == "probe":
            return output, {"input_size": list(dimensions)}
        np.save(output, pixels, allow_pickle=False)
        if mutate_source:
            kwargs["source_path"].chmod(0o600)
            kwargs["source_path"].write_bytes(b"changed raw bytes")
        return output, {
            "input_size": list(pixels.shape[:2]),
            "color_space": "linear_sRGB",
            "dtype": "float32",
            "ingest_fingerprint": "f" * 64,
            **(metadata_updates or {}),
        }

    monkeypatch.setattr(raw, "_run_worker", run)
    return calls, pixels


def test_raw_uses_snapshot_strict_decode_and_preserves_already_oriented_pixels(monkeypatch, resources):
    calls, expected = _worker(monkeypatch)
    pixels, metadata = raw.decode_raw(
        b"immutable raw bytes", "../private-camera.DNG", python=sys.executable, resources=resources
    )
    assert calls == ["probe", "linear_decode"]
    np.testing.assert_array_equal(pixels, expected)
    assert not pixels.flags.writeable
    assert metadata["orientation_normalized"]
    assert metadata["raw_visible_shape"] == [4, 7]
    assert metadata["native_sensor_bit_depth"] is None
    assert metadata["source_bit_depth"] == 16
    assert metadata["raw_runtime_cache_authorized"] is False
    assert metadata["raw_decode"]["auto_white_balance_fallback"] is False


def test_raw_rejects_pixel_limit_before_decode(monkeypatch, resources):
    calls, _ = _worker(monkeypatch, dimensions=(10000, 10000))
    with pytest.raises(ValueError, match="max_pixels"):
        raw.decode_raw(b"immutable raw bytes", "source.dng", python=sys.executable, resources=resources)
    assert calls == ["probe"]


def test_raw_rejects_output_limit_before_decode(monkeypatch, resources):
    calls, _ = _worker(monkeypatch)
    resources["max_output_bytes"] = 64
    with pytest.raises(ValueError, match="output byte budget"):
        raw.decode_raw(b"immutable raw bytes", "source.dng", python=sys.executable, resources=resources)
    assert calls == ["probe"]


@pytest.mark.parametrize(
    "updates, message",
    [
        ({"color_space": "camera_native_linear"}, "linear-sRGB"),
        ({"dtype": "uint8"}, "linear-sRGB"),
        ({"ingest_fingerprint": None}, "fingerprint"),
        ({"ingest_fingerprint": "g" * 64}, "fingerprint"),
        ({"input_size": [True, 2]}, "dimensions"),
        ({"input_size": [5, 4]}, "geometry changed"),
    ],
)
def test_raw_rejects_unattested_decode(monkeypatch, resources, updates, message):
    _worker(monkeypatch, metadata_updates=updates)
    with pytest.raises(ValueError, match=message):
        raw.decode_raw(b"immutable raw bytes", "source.dng", python=sys.executable, resources=resources)


def test_raw_detects_mutated_snapshot(monkeypatch, resources):
    _worker(monkeypatch, mutate_source=True)
    with pytest.raises(ValueError, match="snapshot changed"):
        raw.decode_raw(b"immutable raw bytes", "source.dng", python=sys.executable, resources=resources)


@pytest.mark.parametrize("dtype", [np.uint8, np.float64, object])
def test_raw_array_header_rejects_dtype_without_loading_payload(tmp_path, dtype):
    path = tmp_path / "pixels.npy"
    with path.open("wb") as handle:
        np.lib.format.write_array_header_1_0(
            handle, {"shape": (7, 4, 3), "fortran_order": False, "descr": np.dtype(dtype).str}
        )
    with pytest.raises(ValueError, match="bounded contiguous float32"):
        raw._load_pixels(path, (7, 4))


def test_raw_array_header_rejects_oversized_shape_before_allocation(tmp_path):
    path = tmp_path / "pixels.npy"
    with path.open("wb") as handle:
        np.lib.format.write_array_header_1_0(
            handle, {"shape": (1_000_000_000, 1_000_000_000, 3), "fortran_order": False, "descr": "<f4"}
        )
    with pytest.raises(ValueError, match="bounded contiguous float32"):
        raw._load_pixels(path, (7, 4))


def test_raw_array_rejects_trailing_data_and_symlink(tmp_path):
    path = tmp_path / "pixels.npy"
    np.save(path, np.zeros((7, 4, 3), np.float32))
    alias = tmp_path / "alias.npy"
    alias.symlink_to(path)
    with pytest.raises(OSError):
        raw._load_pixels(alias, (7, 4))
    with path.open("ab") as handle:
        handle.write(b"unbound bytes")
    with pytest.raises(ValueError, match="length differs"):
        raw._load_pixels(path, (7, 4))


def test_raw_array_rejects_nonfinite(tmp_path):
    path = tmp_path / "pixels.npy"
    np.save(path, np.full((7, 4, 3), np.nan, np.float32))
    with pytest.raises(ValueError, match="finite"):
        raw._load_pixels(path, (7, 4))


@pytest.mark.parametrize("key", ["max_pixels", "max_input_bytes", "max_output_bytes", "wall_time_seconds", "memory_mib"])
def test_raw_resource_limits_are_positive_integers(resources, key):
    resources[key] = True
    with pytest.raises(ValueError, match=key):
        raw.decode_raw(b"source", "source.dng", python=sys.executable, resources=resources)


def test_worker_protocol_uses_existing_module_and_bounded_metadata(monkeypatch, tmp_path, resources):
    def command(argv, **kwargs):
        assert argv[:3] == [sys.executable, "-m", raw.RAW_WORKER_MODULE]
        assert argv[argv.index("--command") + 1] == "probe"
        path = Path(argv[argv.index("--output-json") + 1])
        path.write_text('{"input_size": [4, 7]}')

    monkeypatch.setattr(raw, "_run_command", command)
    _, metadata = raw._run_worker(
        "probe",
        python=sys.executable,
        root=tmp_path,
        source_path=tmp_path / "source.dng",
        resources=resources,
        cancellation=None,
        deadline=time.monotonic() + 1,
    )
    assert metadata == {"input_size": [4, 7]}


def _run_script(tmp_path, resources, script, *, cancellation=None, timeout=3):
    pytest.importorskip("psutil", reason="Native process supervision requires the optional ML-core profile")
    return raw._run_command(
        [sys.executable, "-c", script],
        root=tmp_path,
        resources=resources,
        cancellation=cancellation,
        deadline=time.monotonic() + timeout,
        log_name="worker.log",
    )


def test_subprocess_reports_failure_and_never_retries_wb(tmp_path, resources):
    with pytest.raises(RuntimeError, match="invalid camera WB"):
        _run_script(tmp_path, resources, "raise ValueError('invalid camera WB')")


def test_subprocess_timeout_reaps_leader_and_descendant(tmp_path, resources):
    psutil = pytest.importorskip("psutil", reason="Native process supervision requires the optional ML-core profile")
    child_file = tmp_path / "child.pid"
    leader_file = tmp_path / "leader.pid"
    script = (
        "import pathlib, subprocess, sys, time, os; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
        f"pathlib.Path({str(child_file)!r}).write_text(str(child.pid)); "
        f"pathlib.Path({str(leader_file)!r}).write_text(str(os.getpid())); "
        "time.sleep(30)"
    )
    with pytest.raises(TimeoutError, match="wall-time"):
        _run_script(tmp_path, resources, script, timeout=0.5)
    for path in (leader_file, child_file):
        pid = int(path.read_text())
        try:
            assert psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            pass


def test_subprocess_cancel_kills_running_process(tmp_path, resources):
    start = time.monotonic()
    with pytest.raises(RuntimeError, match="cancelled"):
        _run_script(tmp_path, resources, "import time; time.sleep(30)", cancellation=lambda: time.monotonic() - start > 0.2)


def test_subprocess_memory_watchdog_reaps_worker(tmp_path, resources):
    resources["memory_mib"] = 30
    with pytest.raises(RuntimeError, match="observed memory budget"):
        _run_script(tmp_path, resources, "import time; pixels = bytearray(80 * 1024 * 1024); time.sleep(30)")


def test_subprocess_diagnostic_budget_reaps_worker(tmp_path, resources):
    with pytest.raises(RuntimeError, match="diagnostic output"):
        _run_script(tmp_path, resources, "import os, time; os.write(1, b'x' * 2000000); time.sleep(30)")


def test_subprocess_disk_budget_reaps_worker(tmp_path, resources):
    resources["max_output_bytes"] = 1024
    with pytest.raises(RuntimeError, match="disk budget"):
        _run_script(
            tmp_path, resources, "import pathlib, time; pathlib.Path('pixels.npy').write_bytes(b'x' * 2000); time.sleep(30)"
        )
