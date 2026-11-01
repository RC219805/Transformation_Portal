"""Offline native CPU measurements preserve source and managed authority boundaries."""

from __future__ import annotations

import hashlib
import os
import signal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from transformation_portal.lux_depth_v4.photography import output_srgb_icc
from transformation_portal.photographic_delivery import qualification as q

pytestmark = pytest.mark.unit


@pytest.fixture
def case(tmp_path, monkeypatch):
    psutil = pytest.importorskip("psutil")
    from transformation_portal.orchestrator.photographic_admission import ReviewAdmissionConfig

    for name in ReviewAdmissionConfig.__dataclass_fields__:
        monkeypatch.delenv("TP_PHOTO_REVIEW_" + name.upper(), raising=False)
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
    report = q.qualify(case.source, case.digest, case.recipe, case.output, repeats=1)
    assert report["status"] == "measured" and report["source_unchanged"]
    assert report["analytical_reservation_covers_measured_margin"]
    assert report["managed_profile"]["memory_eligible"]
    assert report["managed_profile"]["managed_grant"] is False
    assert report["managed_profile"]["live_server_configuration_checked"] is False
    assert report["photographic_acceptance"] == "not_established"
    assert report["production_flags_changed"] is report["model_inference_performed"] is False
    assert dict(os.environ) == before
    assert case.digest == hashlib.sha256(case.source.read_bytes()).hexdigest()
    trial = report["trials"][0]
    assert trial["zero_effect"] and trial["maximum_16bit_code_delta"] <= 1
    assert trial["observed_peak_rss_bytes"] > 0
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
    assert report["status"] == "planned" and report["trials"] == []
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
        state.killed.append((pid, event))
        state.returncode = -event

    clock = iter([0, 1000 if failure == "timeout" else 0])
    monkeypatch.setattr(q.subprocess, "Popen", launch)
    monkeypatch.setattr(q.time, "monotonic", lambda: next(clock))
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
    assert state.killed == [(424242, signal.SIGTERM)] and state.waited
