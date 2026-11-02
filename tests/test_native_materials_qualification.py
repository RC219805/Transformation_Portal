"""Offline native CPU measurements preserve source and managed authority boundaries."""

from __future__ import annotations

import hashlib
import os
import signal
import time
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from transformation_portal.lux_depth_v4.photography import output_srgb_icc
from transformation_portal.photographic_delivery import qualification as q

pytestmark = pytest.mark.unit


@pytest.fixture(name="case")
def qualification_case(tmp_path, monkeypatch):
    psutil = pytest.importorskip("psutil")
    from transformation_portal.orchestrator.photographic_admission import ReviewAdmissionConfig

    for descriptor in fields(ReviewAdmissionConfig):
        monkeypatch.delenv("TP_PHOTO_REVIEW_" + descriptor.name.upper(), raising=False)
    # Synthetic headroom only: the tiny real subprocess does not qualify a host.
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=64 * q.GIB))
    source_root, output_root = tmp_path / "retained", tmp_path / "qualification"
    source_root.mkdir()
    output_root.mkdir()
    source = source_root / "source.tif"
    samples = np.full((9, 13, 4), 22000, np.uint16)
    samples[..., 3] = 65535
    samples[0, :, 3] = 0
    samples[1, :, 3] = 32768
    tifffile.imwrite(source, samples, photometric="rgb", extrasamples="unassalpha", iccprofile=output_srgb_icc())
    recipe = {
        "schema": "tp.photo.materials_recipe.v1",
        "input_stage": "display_referred",
        "regions": [{"id": "preservation", "label": "glass", "region": [0, 0, 1, 1], "strength": 0}],
    }
    return SimpleNamespace(
        source=source,
        digest=hashlib.sha256(source.read_bytes()).hexdigest(),
        recipe=recipe,
        output=output_root / "trial",
        samples=samples,
    )


@pytest.mark.parametrize("bits", [8, 16])
def test_real_worker_verifies_rgba_products_without_managed_or_human_approval(case, bits):
    expected = case.samples
    if bits == 8:
        samples = (case.samples // 257).astype(np.uint8)
        tifffile.imwrite(case.source, samples, photometric="rgb", extrasamples="unassalpha", iccprofile=output_srgb_icc())
        case.digest = hashlib.sha256(case.source.read_bytes()).hexdigest()
        expected = samples.astype(np.uint16) * 257
    before = dict(os.environ)
    handlers = {event: signal.getsignal(event) for event in (signal.SIGINT, signal.SIGTERM)}
    report = q.qualify(case.source, case.digest, case.recipe, case.output, repeats=1)
    assert report["schema"] == "tp.photo.materials_cpu_qualification.v1"
    assert report["status"] == "measured" and report["source_unchanged"]
    assert report["analytical_reservation_covers_measured_margin"]
    assert report["managed_profile"]["memory_eligible"]
    assert report["managed_profile"]["managed_grant"] is False
    assert report["managed_profile"]["live_server_configuration_checked"] is False
    assert report["photographic_acceptance"] == "not_established"
    assert report["production_flags_changed"] is report["model_inference_performed"] is False
    assert dict(os.environ) == before
    assert {event: signal.getsignal(event) for event in handlers} == handlers
    assert report["normalized_pixels_repeatable"] is None
    assert case.digest == hashlib.sha256(case.source.read_bytes()).hexdigest()
    trial = report["trials"][0]
    assert trial["zero_effect"] and trial["maximum_16bit_code_delta"] <= 1
    assert trial["observed_peak_rss_bytes"] > 0
    checkpoints = trial["rss_high_water_checkpoints"]
    assert checkpoints["semantics"] == "cumulative_process_high_water_not_phase_exclusive"
    assert checkpoints["scope"] == "offline_worker_not_managed_pipeline"
    assert (
        0
        < checkpoints["operation_return_bytes"]
        <= checkpoints["independent_verification_complete_bytes"]
        <= trial["peak_rss_bytes"]
        <= trial["observed_peak_rss_bytes"]
    )
    assert trial["sampled_peak_accounted_bytes"] >= trial["sampled_peak_allocated_bytes"] > 0
    assert {entry["filename"] for entry in trial["products"]} == {
        "master.tif",
        "receipt.json",
        "materials-preview.png",
        "materials-overlay.png",
        "materials-diagnostics.json",
    }
    np.testing.assert_array_equal(tifffile.imread(case.output / "trial-1/master.tif"), expected)
    assert not (case.output / "trial-1/source.snapshot.tif").exists()


def test_worker_rss_checkpoints_preserve_cumulative_phase_order_and_total_metrics(case, monkeypatch):
    import resource

    from transformation_portal.lux_depth_v4 import photography
    from transformation_portal.photographic_delivery import materials

    case.output.mkdir()
    specification = case.output / "specification.json"
    q._write_json(
        specification,
        {
            "source": str(case.source),
            "bound": q._source(case.source, case.digest),
            "recipe": case.recipe,
            "runtime": q._runtime_identity(),
        },
    )
    # Sampling has cumulative semantics, including source preparation and imports.
    # Exclusive phase allocation or managed-pipeline reservations cannot be inferred.
    samples = iter([10, 20, 30, 40])
    events = []

    def observe(usage):
        assert usage == resource.RUSAGE_SELF
        sample = next(samples)
        events.append(f"rss:{sample}")
        return SimpleNamespace(ru_maxrss=sample)

    create_revision = materials.create_materials_revision
    decode_master = photography.decode_master

    def create(*args, **kwargs):
        result = create_revision(*args, **kwargs)
        events.append("operation_return")
        return result

    def decode(*args, **kwargs):
        result = decode_master(*args, **kwargs)
        if kwargs.get("source_name") == "master.tif":
            events.append("independent_reopen")
        return result

    monkeypatch.setattr(resource, "getrusage", observe)
    monkeypatch.setattr(materials, "create_materials_revision", create)
    monkeypatch.setattr(photography, "decode_master", decode)
    q._worker(specification)
    result = q._read_json(case.output / "result.json", 256 * 1024)
    multiplier = 1 if q.sys.platform == "darwin" else 1024
    assert events == ["rss:10", "operation_return", "rss:20", "independent_reopen", "rss:30", "rss:40"]
    assert result["rss_high_water_checkpoints"] == {
        "semantics": "cumulative_process_high_water_not_phase_exclusive",
        "scope": "offline_worker_not_managed_pipeline",
        "operation_return_bytes": 20 * multiplier,
        "independent_verification_complete_bytes": 30 * multiplier,
    }
    assert result["baseline_rss_bytes"] == 10 * multiplier
    assert result["peak_rss_bytes"] == 40 * multiplier
    assert result["incremental_peak_rss_bytes"] == 30 * multiplier
    assert result["zero_effect"] and result["maximum_16bit_code_delta"] <= 1
    assert len(result["products"]) == 5


@pytest.mark.parametrize("limit", ["memory", "offline"])
def test_capacity_rejection_precedes_pixel_decode_and_worker_launch(case, monkeypatch, limit):
    import psutil

    def forbidden(*args, **kwargs):
        pytest.fail("Blocked qualification must not decode or start a worker")

    monkeypatch.setattr(tifffile.TiffPage, "asarray", forbidden)
    monkeypatch.setattr(q, "_run_trial", forbidden)
    if limit == "memory":
        monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=q.GIB))
    limits = q.QualificationLimits(memory_bytes=1) if limit == "offline" else q.QualificationLimits()
    report = q.qualify(case.source, case.digest, case.recipe, case.output, limits=limits)
    assert report["status"] == "blocked" and not report["trials"]
    assert report["source_unchanged"]
    assert set(path.name for path in case.output.iterdir()) == {"qualification.json"}


def test_special_input_cannot_block_before_resource_supervision(case):
    fifo = case.source.parent / "not-a-regular-file.tif"
    os.mkfifo(fifo)
    environment = {**os.environ, "PYTHONPATH": str(Path(q.__file__).resolve().parents[2])}
    result = q.subprocess.run(
        [
            q.sys.executable,
            "-c",
            "from pathlib import Path; from transformation_portal.photographic_delivery.qualification import _open_regular; "
            "_open_regular(Path(__import__('sys').argv[1]), 1024)",
            str(fifo),
        ],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode != 0 and "bounded regular files" in result.stderr


def test_bad_source_digest_never_creates_output_or_allocates_pixels(case, monkeypatch):
    monkeypatch.setattr(tifffile.TiffPage, "asarray", lambda *args, **kwargs: pytest.fail("Must not decode"))
    with pytest.raises(ValueError, match="SHA-256"):
        q.qualify(case.source, "a" * 64, case.recipe, case.output)
    assert not case.output.exists()


def test_existing_output_and_retained_source_namespace_are_preserved(case):
    case.output.mkdir()
    sentinel = case.output / "retained.txt"
    sentinel.write_text("preserve")
    with pytest.raises(ValueError, match="fresh directory"):
        q.qualify(case.source, case.digest, case.recipe, case.output)
    assert sentinel.read_text() == "preserve"
    with pytest.raises(ValueError, match="separate"):
        q.qualify(case.source, case.digest, case.recipe, case.source.parent / "output")


def test_symlink_source_and_output_ancestor_fail_closed(case, tmp_path):
    linked = case.source.parent / "linked.tif"
    linked.symlink_to(case.source)
    with pytest.raises(ValueError, match="symbolic links"):
        q.qualify(linked, case.digest, case.recipe, case.output)
    target = tmp_path / "linked"
    target.symlink_to(case.output.parent)
    with pytest.raises(ValueError, match="non-symlink"):
        q.qualify(case.source, case.digest, case.recipe, target / "new")


def test_float_tiff_rejected_by_headers_before_native_allocation(case, monkeypatch):
    source = case.source.parent / "float.tif"
    tifffile.imwrite(source, np.full((3, 4, 3), 0.25, np.float32), photometric="rgb", iccprofile=output_srgb_icc())
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(tifffile.TiffPage, "asarray", lambda *args, **kwargs: pytest.fail("Must not decode"))
    with pytest.raises(ValueError, match="integer 8/16-bit"):
        q.qualify(source, digest, case.recipe, case.output)
    assert not case.output.exists()


def test_preflight_does_not_execute_or_grant_managed_capacity(case, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Preflight must not execute")

    monkeypatch.setattr(q, "_run_trial", forbidden)
    report = q.qualify(case.source, case.digest, case.recipe, case.output, preflight_only=True)
    assert report["status"] == "planned" and not report["trials"]
    assert report["managed_profile"]["managed_grant"] is False


def test_measured_peak_margin_cannot_automatically_expand_reservation(case, monkeypatch):
    monkeypatch.setattr(
        q,
        "_run_trial",
        lambda *args, **kwargs: {"incremental_peak_rss_bytes": 12 * q.GIB, "normalized_pixel_sha256": "a" * 64},
    )
    report = q.qualify(case.source, case.digest, case.recipe, case.output, repeats=1)
    assert report["status"] == "not_qualified"
    assert report["blocking_reasons"] == ["measured_margin_exceeds_analytical_reservation"]
    assert report["limits"]["memory_bytes"] == 12 * q.GIB
    assert report["offline_memory_reservation_bytes"] < 12 * q.GIB


def test_runtime_change_is_rejected_by_real_child_before_source_decode(case, monkeypatch):
    monkeypatch.setattr(q, "_runtime_identity", lambda: {"changed": True})
    with pytest.raises(RuntimeError, match="worker failed"):
        q.qualify(case.source, case.digest, case.recipe, case.output, repeats=1)
    assert not (case.output / "trial-1/master.tif").exists()
    report = q._read_json(case.output / "qualification.json", 256 * 1024)
    assert report["status"] == "failed" and report["source_unchanged"]
    assert "runtime changed" in (case.output / "trial-1/worker.log").read_text()


@pytest.mark.parametrize("value", [0, 6, True])
def test_trial_count_is_bounded_before_creating_namespace(case, value):
    with pytest.raises(ValueError, match="one to five"):
        q.qualify(case.source, case.digest, case.recipe, case.output, repeats=value)
    assert not case.output.exists()


@pytest.mark.parametrize(
    "name,value",
    [("memory_bytes", 13 * q.GIB), ("scratch_bytes", 65 * q.GIB), ("timeout_seconds", 301), ("memory_bytes", True)],
)
def test_offline_envelope_cannot_be_increased_silently(name, value):
    with pytest.raises(ValueError, match="positive integer"):
        q.QualificationLimits(**{name: value})


def test_recipe_json_rejects_duplicate_keys(tmp_path):
    path = tmp_path / "recipe.json"
    path.write_text('{"schema":"one","schema":"two"}')
    with pytest.raises(ValueError, match="Duplicate"):
        q._read_json(path)


def test_source_change_after_planning_is_rejected_before_snapshot_use(case):
    bound = q._source(case.source, case.digest)
    case.source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity changed"):
        q._copy_source(case.source, case.output.parent / "snapshot.tif", bound)


def test_source_revalidation_failure_cannot_leave_a_measured_report(case, monkeypatch):
    def changed(*args, **kwargs):
        case.source.write_bytes(b"changed")
        return {"incremental_peak_rss_bytes": 1, "normalized_pixel_sha256": "a" * 64}

    monkeypatch.setattr(q, "_run_trial", changed)
    with pytest.raises(ValueError, match="Retained source changed"):
        q.qualify(case.source, case.digest, case.recipe, case.output, repeats=1)
    report = q._read_json(case.output / "qualification.json", 256 * 1024)
    assert report["status"] == "failed" and report["source_unchanged"] is False


def test_identical_bytes_with_replaced_inode_cannot_leave_measured_report(case, monkeypatch):
    def replaced(*args, **kwargs):
        replacement = case.source.with_name("replacement.tif")
        replacement.write_bytes(case.source.read_bytes())
        replacement.replace(case.source)
        return {"incremental_peak_rss_bytes": 1, "normalized_pixel_sha256": "a" * 64}

    monkeypatch.setattr(q, "_run_trial", replaced)
    with pytest.raises(ValueError, match="Retained source changed"):
        q.qualify(case.source, case.digest, case.recipe, case.output, repeats=1)
    report = q._read_json(case.output / "qualification.json", 256 * 1024)
    assert report["status"] == "failed" and report["source_unchanged"] is False


def test_cli_preflight_cannot_claim_execution_or_managed_acceptance(case):
    recipe = case.output.parent / "recipe.json"
    recipe.write_bytes(q._canonical(case.recipe))
    environment = {**os.environ, "PYTHONPATH": str(Path(q.__file__).resolve().parents[2])}
    result = q.subprocess.run(
        [
            q.sys.executable,
            "-m",
            q._MODULES[0],
            "--source",
            str(case.source),
            "--source-sha256",
            case.digest,
            "--recipe",
            str(recipe),
            "--output",
            str(case.output),
            "--preflight-only",
        ],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode in {0, 2}, result.stderr
    report = q._read_json(case.output / "qualification.json", 256 * 1024)
    assert report["status"] in {"planned", "blocked"} and not report["trials"]
    assert report["managed_eligibility"] == "not_established_offline_measurement_only"


@pytest.mark.parametrize("failure", ["timeout", "memory", "scratch", "cancel", "children", "observer"])
def test_failure_drains_only_the_owned_worker(tmp_path, monkeypatch, failure):
    psutil = pytest.importorskip("psutil")
    state = SimpleNamespace(returncode=None, killed=[], waited=False)

    class Process:
        pid = 424242

        def poll(self):
            return state.returncode

        def wait(self, timeout=None):
            state.waited = True
            return state.returncode

    def launch(*args, **kwargs):
        assert kwargs["start_new_session"] and kwargs["stdin"] == q.subprocess.DEVNULL
        return Process()

    def kill(pid, event):
        if event == 0:
            raise ProcessLookupError()
        state.killed.append((pid, event))
        state.returncode = -event

    clock = iter([0])
    monkeypatch.setattr(q.subprocess, "Popen", launch)
    monkeypatch.setattr(q.time, "monotonic", lambda: next(clock, 1000 if failure == "timeout" else 0))
    monkeypatch.setattr(q.os, "killpg", kill)

    def observe(pid):
        if failure == "observer":
            raise PermissionError("Process inspection denied")
        if failure == "cancel":
            raise KeyboardInterrupt()
        return SimpleNamespace(
            memory_info=lambda: SimpleNamespace(rss=2 * q.GIB if failure == "memory" else 1),
            children=lambda **kwargs: [object()] if failure == "children" else [],
        )

    monkeypatch.setattr(psutil, "Process", observe)
    if failure == "scratch":
        monkeypatch.setattr(
            q,
            "_scratch_usage",
            lambda root: {"logical_bytes": 2 * q.GIB, "allocated_bytes": 2 * q.GIB, "accounted_bytes": 2 * q.GIB},
        )
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=100 * q.GIB))
    expected = {
        "timeout": TimeoutError,
        "memory": RuntimeError,
        "scratch": RuntimeError,
        "cancel": KeyboardInterrupt,
        "children": RuntimeError,
        "observer": PermissionError,
    }[failure]
    with pytest.raises(expected):
        q._run_trial(tmp_path / "trial", {}, q.QualificationLimits(), q.GIB, q.GIB, 1, 1)
    assert state.killed == [(424242, signal.SIGTERM), (424242, signal.SIGKILL)] and state.waited


def test_runtime_identity_binds_delegated_color_implementations(tmp_path, monkeypatch):
    pytest.importorskip("psutil")
    from transformation_portal.lux_depth_v4 import photography

    before = q._runtime_identity()
    for function in (photography.apply_input_color, photography.linear_to_srgb, photography.output_srgb_icc):
        assert function.__module__ in before["module_sha256"]
    actual_import = q.importlib.import_module
    replacement = tmp_path / "color.py"
    replacement.write_text("changed color implementation")
    monkeypatch.setattr(
        q.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(__file__=str(replacement))
            if name == photography.linear_to_srgb.__module__
            else actual_import(name)
        ),
    )
    assert q._runtime_identity() != before


@pytest.mark.parametrize("ignore_worker_termination,trigger", [(False, "signal"), (True, "signal"), (True, "timeout")])
def test_sigterm_and_repeated_cancellation_drain_a_real_owned_worker(case, tmp_path, ignore_worker_termination, trigger):
    psutil = pytest.importorskip("psutil")
    marker, worker_ready = tmp_path / "owned.pid", tmp_path / "worker.ready"
    draining = tmp_path / "worker.draining"
    worker_code = (
        "import signal,time; from pathlib import Path; "
        + (
            f"signal.signal(signal.SIGTERM, lambda *_: Path({str(draining)!r}).write_text('draining')); "
            if ignore_worker_termination
            else ""
        )
        + "signal.pthread_sigmask(signal.SIG_UNBLOCK,{signal.SIGINT,signal.SIGTERM}); "
        + f"Path({str(worker_ready)!r}).write_text('ready'); time.sleep(60)"
    )
    script = tmp_path / "supervisor.py"
    script.write_text(
        "from pathlib import Path\n"
        "from transformation_portal.photographic_delivery import qualification as q\n"
        "launch = q.subprocess.Popen\n"
        "def worker(*args, **kwargs):\n"
        "    if args[0][1:3] != ['-m', q._MODULES[0]]: return launch(*args, **kwargs)\n"
        f"    process = launch([q.sys.executable, '-c', {worker_code!r}], **kwargs)\n"
        f"    Path({str(marker)!r}).write_text(str(process.pid))\n"
        "    return process\n"
        "q.subprocess.Popen = worker\n"
        "import psutil; from types import SimpleNamespace\n"
        "psutil.virtual_memory = lambda: SimpleNamespace(available=64*q.GIB)\n"
        f"q.qualify(Path({str(case.source)!r}), {case.digest!r}, {case.recipe!r}, Path({str(case.output)!r}), "
        f"repeats=1, limits=q.QualificationLimits(timeout_seconds={1 if trigger == 'timeout' else 300}))\n"
    )
    environment = {**os.environ, "PYTHONPATH": str(Path(q.__file__).resolve().parents[2])}
    process = q.subprocess.Popen(
        [q.sys.executable, str(script)],
        env=environment,
        stdout=q.subprocess.PIPE,
        stderr=q.subprocess.PIPE,
        start_new_session=True,
    )
    worker_pid = None
    try:
        deadline = time.monotonic() + 10
        while not (marker.exists() and worker_ready.exists()):
            if process.poll() is not None or time.monotonic() >= deadline:
                pytest.fail("Owned worker did not reach its deterministic cancellation barrier")
            time.sleep(0.01)
        worker_pid = int(marker.read_text())
        if trigger == "signal":
            process.send_signal(signal.SIGTERM)
        if ignore_worker_termination:
            while not draining.exists():
                if process.poll() is not None or time.monotonic() >= deadline:
                    pytest.fail("Worker did not reach the owned-drain barrier")
                time.sleep(0.01)
            process.send_signal(signal.SIGTERM)
            if trigger == "timeout":
                time.sleep(0.1)
                process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=8)
        assert process.returncode == 128 + signal.SIGTERM, stderr.decode()
        assert not psutil.pid_exists(worker_pid)
        report = q._read_json(case.output / "qualification.json", 256 * 1024)
        assert report["status"] == "failed" and report["source_unchanged"]
        assert report["error"]["type"] == "SystemExit"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if worker_pid is not None and psutil.pid_exists(worker_pid):
            os.kill(worker_pid, signal.SIGKILL)


@pytest.mark.parametrize("repeatable", [False, True])
def test_repeatability_requires_comparison_of_independent_trials(case, monkeypatch, repeatable):
    pixels = iter(["a" * 64, ("a" if repeatable else "b") * 64])
    monkeypatch.setattr(
        q, "_run_trial", lambda *args, **kwargs: {"incremental_peak_rss_bytes": 1, "normalized_pixel_sha256": next(pixels)}
    )
    report = q.qualify(case.source, case.digest, case.recipe, case.output, repeats=2)
    assert report["normalized_pixels_repeatable"] is repeatable
    assert report["status"] == ("measured" if repeatable else "not_qualified")


@pytest.mark.parametrize("interruption", [SystemExit(143), KeyboardInterrupt(), RuntimeError("verification failed")])
def test_interrupted_final_source_verification_cannot_persist_measured(case, monkeypatch, interruption):
    original_source = q._source
    calls = 0

    def verify(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise interruption
        return original_source(*args, **kwargs)

    monkeypatch.setattr(q, "_source", verify)
    monkeypatch.setattr(
        q, "_run_trial", lambda *args, **kwargs: {"incremental_peak_rss_bytes": 1, "normalized_pixel_sha256": "a" * 64}
    )
    with pytest.raises(type(interruption)):
        q.qualify(case.source, case.digest, case.recipe, case.output, repeats=1)
    report = q._read_json(case.output / "qualification.json", 256 * 1024)
    assert report["status"] == "failed" and report["source_unchanged"] is False
    assert report["source_revalidation_error"]["type"] == type(interruption).__name__


def test_runtime_identity_does_not_launch_external_platform_helpers(monkeypatch):
    pytest.importorskip("psutil")
    q._runtime_identity()  # Import the selected implementations before isolating OS identity reads.

    def forbidden(*args, **kwargs):
        pytest.fail("Runtime identity must not fork external platform helpers")

    monkeypatch.setattr(q.subprocess, "Popen", forbidden)
    monkeypatch.setattr(q.platform, "_platform_cache", {})
    monkeypatch.setattr(q.platform, "_uname_cache", None)
    identity = q._runtime_identity()
    assert identity["platform"]["system"] in {"Darwin", "Linux"}


@pytest.mark.parametrize("phase", ["serialization", "after_publication"])
def test_signal_during_report_publication_leaves_atomic_failed_record(case, monkeypatch, phase):
    monkeypatch.setattr(
        q, "_run_trial", lambda *args, **kwargs: {"incremental_peak_rss_bytes": 1, "normalized_pixel_sha256": "a" * 64}
    )
    if phase == "serialization":
        original = q._canonical

        def serialize(value):
            if value.get("schema") == q.SCHEMA and value["status"] == "measured":
                os.kill(os.getpid(), signal.SIGTERM)
            return original(value)

        monkeypatch.setattr(q, "_canonical", serialize)
    else:
        original_write = q._write_json

        def write(path, value, **kwargs):
            original_write(path, value, **kwargs)
            if path.name == "qualification.json" and value["status"] == "measured":
                os.kill(os.getpid(), signal.SIGTERM)

        monkeypatch.setattr(q, "_write_json", write)
    with pytest.raises(SystemExit):
        q.qualify(case.source, case.digest, case.recipe, case.output, repeats=1)
    report = q._read_json(case.output / "qualification.json", 256 * 1024)
    assert report["status"] == "failed" and report["source_unchanged"]
    assert report["report_publication_error"]["type"] == "SystemExit"
    assert not list(case.output.glob(".qualification.json.*"))


def test_atomic_json_never_replaces_existing_unowned_record(tmp_path):
    record = tmp_path / "retained.json"
    record.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        q._write_json(record, {"new": "record"})
    assert record.read_bytes() == b"preserve"
    assert not list(tmp_path.glob(".retained.json.*"))


@pytest.mark.parametrize("leader_exits", [False, True])
def test_unexpected_descendant_is_drained_after_its_leader_exits(case, tmp_path, leader_exits):
    psutil = pytest.importorskip("psutil")
    ready = tmp_path / "descendant.pid"
    child_code = (
        "import os,signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    leader = tmp_path / "leader.py"
    leader.write_text(
        "import os,subprocess,sys,time\nfrom pathlib import Path\n"
        f"child = subprocess.Popen([sys.executable,'-c',{child_code!r}])\n"
        f"while not Path({str(ready)!r}).exists(): time.sleep(.01)\n"
        + ("os._exit(0)\n" if leader_exits else "time.sleep(60)\n")
    )
    script = tmp_path / "descendant_supervisor.py"
    script.write_text(
        "from pathlib import Path\nfrom types import SimpleNamespace\nimport time,psutil\n"
        "from transformation_portal.photographic_delivery import qualification as q\n"
        "launch = q.subprocess.Popen\n"
        "def worker(*args, **kwargs):\n"
        "    if args[0][1:3] != ['-m',q._MODULES[0]]: return launch(*args, **kwargs)\n"
        f"    return launch([q.sys.executable,{str(leader)!r}], **kwargs)\n"
        "actual_process = psutil.Process\n"
        "def observe(pid):\n"
        "    owned = actual_process(pid)\n"
        "    def children(**kwargs):\n"
        "        deadline=time.monotonic()+10\n"
        f"        while not Path({str(ready)!r}).exists():\n"
        "            if time.monotonic()>deadline: raise RuntimeError('Descendant barrier failed')\n"
        "            time.sleep(.01)\n"
        f"        return [actual_process(int(Path({str(ready)!r}).read_text()))]\n"
        "    return SimpleNamespace(memory_info=owned.memory_info, children=children)\n"
        "psutil.Process=observe\nq.subprocess.Popen=worker\n"
        "psutil.virtual_memory=lambda:SimpleNamespace(available=64*q.GIB)\n"
        f"q.qualify(Path({str(case.source)!r}),{case.digest!r},{case.recipe!r},Path({str(case.output)!r}),repeats=1)\n"
    )
    environment = {**os.environ, "PYTHONPATH": str(Path(q.__file__).resolve().parents[2])}
    process = q.subprocess.Popen(
        [q.sys.executable, str(script)],
        env=environment,
        stdout=q.subprocess.PIPE,
        stderr=q.subprocess.PIPE,
        start_new_session=True,
    )
    child_pid = None
    try:
        stdout, stderr = process.communicate(timeout=20)
        assert process.returncode != 0, stdout.decode()
        assert ready.exists(), stderr.decode()
        child_pid = int(ready.read_text())
        try:
            assert psutil.Process(child_pid).status() in {psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD}
        except psutil.NoSuchProcess:
            pass
        report = q._read_json(case.output / "qualification.json", 256 * 1024)
        assert report["status"] == "failed" and report["source_unchanged"]
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if ready.exists() and child_pid is None:
            child_pid = int(ready.read_text())
        if child_pid is not None:
            try:
                psutil.Process(child_pid).kill()
            except psutil.NoSuchProcess:
                pass
