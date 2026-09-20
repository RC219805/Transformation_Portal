"""Full photographic graph contracts with an explicit controlled backend fixture."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
from PIL import Image

from transformation_portal.lux_depth_v4 import LuxDepthV4Request, pipeline, prepare

pytestmark = pytest.mark.unit


class ParentFixture:
    sha256 = "a" * 64
    source_sha256 = "b" * 64

    def verify(self):
        pass


class SessionFixture:
    calls = 0

    def __init__(self, _python, plan, *, cancellation):
        self.plan = plan
        self.runtime = SimpleNamespace(
            runtime_identity_sha256="c" * 64,
            to_mapping=lambda: {"backend_identity": {"model": "fixture", "actual_device": "cpu"}},
        )

    def compute(self, proxy):
        type(self).calls += 1
        return {
            "native_depth": np.linspace(1, 9, proxy.shape[0] * proxy.shape[1], dtype=np.float32).reshape(proxy.shape[:2])
        }, {"native_semantics": "da3_metric_uncalibrated"}

    def checkpoint(self):
        pass

    def verify(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    values = np.arange(1176, dtype=np.uint16).reshape(28, 42) + 10000
    tifffile.imwrite(inputs / "ramp.tif", np.repeat(values[..., None], 3, axis=-1), photometric="rgb")
    monkeypatch.setattr(pipeline, "DA3Session", SessionFixture)
    monkeypatch.setattr(pipeline, "PhotographyRuntime", ParentFixture)
    monkeypatch.setattr(
        pipeline,
        "require_process_supervisor",
        lambda: SimpleNamespace(Process=lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=0))),
    )
    SessionFixture.calls = 0
    return prepare(
        LuxDepthV4Request(
            inputs, tmp_path / "output", input_color="srgb", strength=0, target_size=56, cache_dir=tmp_path / "cache"
        )
    )


def test_complete_graph_preserves_precision_and_native_depth(prepared):
    result = pipeline.run(prepared)
    image = tifffile.imread(result.output_root / "input-0000/delivery.tif")
    assert image.dtype == np.uint16 and image.shape == (28, 42, 3)
    assert len(np.unique(image[..., 0])) == 1176
    native = np.load(result.output_root / "input-0000/native-depth.npy")
    assert native.max() == 9 and native.min() == 1
    evidence = json.loads(result.evidence_path.read_bytes())
    assert evidence["complete"] is True
    assert evidence["plan_fingerprint_sha256"] == prepared.plan.plan_fingerprint_sha256
    assert (result.output_root / "execution-plan.json").read_bytes() == prepared.canonical_plan_bytes
    descriptor = json.loads((result.output_root / "input-0000/photograph.json").read_bytes())
    assert descriptor["depth"]["has_metric_depth"] is False
    assert descriptor["materials"]["status"] == "abstained"


def test_warm_cache_reuses_only_native_depth_and_isolates_sources(prepared):
    first = pipeline.run(prepared)
    second = pipeline.run(replace(prepared, output_root=prepared.output_root.with_name("second")))
    assert first.depth_cache_misses == second.depth_cache_hits == SessionFixture.calls == 1
    source = prepared.input_root / "ramp.tif"
    pixels = tifffile.imread(source)
    tifffile.imwrite(source, pixels + 1, photometric="rgb")
    changed = prepare(
        LuxDepthV4Request(
            prepared.input_root,
            prepared.output_root.with_name("changed"),
            input_color="srgb",
            strength=0,
            target_size=56,
            cache_dir=prepared.cache_root,
        )
    )
    result = pipeline.run(changed)
    assert result.depth_cache_misses == 1 and SessionFixture.calls == 2


def test_source_drift_fails_before_outputs_or_inference(prepared):
    (prepared.input_root / "ramp.tif").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="changed after preparation"):
        pipeline.run(prepared)
    assert not prepared.output_root.exists() and SessionFixture.calls == 0


def test_cancellation_never_publishes_completion(prepared):
    with pytest.raises(RuntimeError, match="cancelled"):
        pipeline.run(prepared, cancellation=lambda: True)
    assert not prepared.output_root.exists()


def test_disk_budget_failure_preserves_incomplete_evidence(prepared):
    limited = prepare(
        LuxDepthV4Request(prepared.input_root, prepared.output_root, input_color="srgb", target_size=56, max_output_bytes=5000)
    )
    with pytest.raises(RuntimeError, match="disk budget"):
        pipeline.run(limited)
    assert not (prepared.output_root / "execution-evidence.json").exists()
    assert json.loads((prepared.output_root / "failure.json").read_bytes())["complete"] is False


def test_backend_failure_never_becomes_success(prepared, monkeypatch):
    def failed(*_args):
        raise RuntimeError("backend failure fixture")

    monkeypatch.setattr(SessionFixture, "compute", failed)
    with pytest.raises(RuntimeError, match="backend failure fixture"):
        pipeline.run(prepared)
    assert not (prepared.output_root / "execution-evidence.json").exists()


def test_cli_help_keeps_v3_entrypoint_and_lists_plan():
    from transformation_portal.lux_depth_v4.__main__ import main

    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_own_namespace_setup_does_not_invalidate_frozen_runtime(prepared, monkeypatch):
    class NamespaceRuntime(ParentFixture):
        def __init__(self):
            self.names = {path.name for path in prepared.output_root.parent.iterdir()}

        def verify(self):
            if self.names != {path.name for path in prepared.output_root.parent.iterdir()}:
                raise RuntimeError("Runtime import namespace changed")

    monkeypatch.setattr(pipeline, "PhotographyRuntime", NamespaceRuntime)
    assert pipeline.run(prepared).input_count == 1


def test_cancellation_during_final_verification_cannot_publish_success(prepared, monkeypatch):
    original_snapshot = pipeline.snapshot
    descriptor_reads = 0
    cancelled = False

    def cancelling_snapshot(root, path, **kwargs):
        nonlocal descriptor_reads, cancelled
        result = original_snapshot(root, path, **kwargs)
        if path.name == "photograph.json":
            descriptor_reads += 1
            cancelled = descriptor_reads == 2
        return result

    monkeypatch.setattr(pipeline, "snapshot", cancelling_snapshot)
    with pytest.raises(RuntimeError, match="cancelled"):
        pipeline.run(prepared, cancellation=lambda: cancelled)
    assert not (prepared.output_root / "execution-evidence.json").exists()


def _with_companions(prepared, *, calibration=True, materials=True, confidence=True):
    import hashlib

    from transformation_portal.ingest.canonical_json import canonicalize_json

    root = prepared.output_root.parent / "companions"
    root.mkdir(exist_ok=True)
    record = {"path": "ramp.tif", "source_sha256": prepared.plan.to_payload()["inputs"][0]["sha256"]}
    if calibration:
        record["calibration"] = {
            "width": 42,
            "height": 28,
            "fx": 300.0,
            "fy": 600.0,
            "cx": 20.5,
            "cy": 13.5,
            "source": "measured camera calibration fixture",
            "coordinate_space": "canonical_master",
        }
    if materials:
        mask = root / "water.npy"
        np.save(mask, np.ones((28, 42), dtype=np.float32))
        record["materials"] = {
            "masks": {"water": {"path": "water.npy", "sha256": hashlib.sha256(mask.read_bytes()).hexdigest()}},
            "confidences": {"water": 0.99} if confidence else {},
            "coordinate_space": "canonical_master",
        }
    manifest = root / "companions.json"
    manifest.write_bytes(canonicalize_json({"schema": "tp.lux.companions.v1", "inputs": [record]}))
    return (
        prepare(
            LuxDepthV4Request(
                prepared.input_root,
                prepared.output_root,
                input_color="srgb",
                strength=0,
                target_size=56,
                cache_dir=prepared.cache_root,
                companions_manifest=manifest,
            )
        ),
        manifest,
    )


def test_optional_inputs_produce_bound_calibrated_depth_and_material_report(prepared):
    prepared, manifest = _with_companions(prepared)
    payload = prepared.plan.to_payload()
    assert payload["nodes"][0]["inputs"] == {"source": "$input", "calibration": "$calibration", "materials": "$materials"}
    assert payload["nodes"][1]["inputs"]["calibration"] == "$calibration"
    assert payload["nodes"][2]["inputs"]["materials"] == "$materials"
    result = pipeline.run(prepared)
    destination = result.output_root / "input-0000"
    native = np.load(destination / "native-depth.npy")
    metric = np.load(destination / "metric-depth-m.npy")
    assert native.min() == 1 and native.max() == 9
    np.testing.assert_array_equal(metric, native * 1.5)
    np.testing.assert_array_equal(np.load(destination / "aligned-metric-depth-m.npy"), metric)
    descriptor = json.loads((destination / "photograph.json").read_bytes())
    assert descriptor["depth"]["has_metric_depth"] is True
    assert descriptor["depth"]["has_confidence"] is False
    assert descriptor["depth"]["intrinsics"]["fx"] == 300
    assert descriptor["materials"]["status"] == "applied"
    assert descriptor["materials"]["segmentation_inferred"] is False
    assert descriptor["materials"]["materials"]["water"]["confidence"] == 0.99


def test_optional_missing_confidence_abstains_and_missing_calibration_stays_unavailable(prepared):
    prepared, _ = _with_companions(prepared, calibration=False, confidence=False)
    result = pipeline.run(prepared)
    descriptor = json.loads((result.output_root / "input-0000/photograph.json").read_bytes())
    assert descriptor["materials"]["materials"]["water"]["reason"] == "missing_confidence"
    assert descriptor["materials"]["status"] == "abstained"
    assert descriptor["depth"]["has_metric_depth"] is False
    assert not (result.output_root / "input-0000/metric-depth-m.npy").exists()


def test_invalid_calibrated_depth_does_not_change_photographic_pixels(prepared, monkeypatch):
    prepared, manifest = _with_companions(prepared, materials=False)
    prepared = prepare(
        LuxDepthV4Request(
            prepared.input_root,
            prepared.output_root,
            input_color="srgb",
            strength=1,
            target_size=14,
            companions_manifest=manifest,
        )
    )

    def depth_with_hole(self, proxy):
        native = np.linspace(1, 9, proxy.shape[0] * proxy.shape[1], dtype=np.float32).reshape(proxy.shape[:2])
        native[3:7, 4:10] = -1000
        return {"native_depth": native}, {"native_semantics": "da3_metric_uncalibrated"}

    monkeypatch.setattr(SessionFixture, "compute", depth_with_hole)
    result = pipeline.run(prepared)
    destination = result.output_root / "input-0000"
    original = np.load(destination / "source-master.npy")
    finished = np.load(destination / "master.npy")
    valid = np.load(destination / "aligned-depth-valid.npy")
    assert valid.shape == original.shape[:2] and valid.dtype == np.bool_
    assert valid.any() and (~valid).any()
    np.testing.assert_array_equal(finished[~valid], original[~valid])
    assert np.any(finished[valid] != original[valid])
    relative = np.load(destination / "relative-depth.npy")
    metric = np.load(destination / "aligned-metric-depth-m.npy")
    np.testing.assert_array_equal(relative[~valid], 0)
    np.testing.assert_array_equal(metric[~valid], 0)
    assert (metric[valid] > 0).all()
    assert (np.load(destination / "native-depth.npy") < 0).any()
    descriptor = json.loads((destination / "photograph.json").read_bytes())
    assert descriptor["aligned_depth"]["validity_path"] == "input-0000/aligned-depth-valid.npy"
    evidence = json.loads(result.evidence_path.read_bytes())
    assert any(item["path"] == "input-0000/aligned-depth-valid.npy" for item in evidence["artifacts"])


def test_changed_material_bytes_fail_before_worker_or_outputs(prepared):
    prepared, manifest = _with_companions(prepared)
    mask = manifest.parent / "water.npy"
    np.save(mask, np.zeros((28, 42), dtype=np.float32))
    with pytest.raises(ValueError, match="digest differs from the manifest"):
        pipeline.run(prepared)
    assert SessionFixture.calls == 0 and not prepared.output_root.exists()


def test_wrong_canonical_geometry_fails_before_inference(prepared):
    from transformation_portal.ingest.canonical_json import canonicalize_json

    _, manifest = _with_companions(prepared, materials=False)
    payload = json.loads(manifest.read_bytes())
    calibration = payload["inputs"][0]["calibration"]
    calibration["height"], calibration["width"] = 42, 28
    manifest.write_bytes(canonicalize_json(payload))
    wrong = prepare(
        LuxDepthV4Request(prepared.input_root, prepared.output_root, input_color="srgb", companions_manifest=manifest)
    )
    with pytest.raises(RuntimeError, match="orientation-normalized master"):
        pipeline.run(wrong)
    assert SessionFixture.calls == 0


def test_calibration_changes_depth_identity_and_derived_meters(prepared):
    from transformation_portal.ingest.canonical_json import canonicalize_json

    prepared, manifest = _with_companions(prepared, materials=False)
    first = pipeline.run(prepared)
    repeat = pipeline.run(replace(prepared, output_root=prepared.output_root.with_name("calibration-repeat")))
    assert repeat.depth_cache_hits == 1
    payload = json.loads(manifest.read_bytes())
    payload["inputs"][0]["calibration"]["fx"] *= 2
    manifest.write_bytes(canonicalize_json(payload))
    changed = prepare(
        LuxDepthV4Request(
            prepared.input_root,
            prepared.output_root.with_name("recalibrated"),
            input_color="srgb",
            strength=0,
            target_size=56,
            cache_dir=prepared.cache_root,
            companions_manifest=manifest,
        )
    )
    second = pipeline.run(changed)
    assert second.depth_cache_misses == 1 and SessionFixture.calls == 2
    np.testing.assert_array_equal(
        np.load(first.output_root / "input-0000/native-depth.npy"),
        np.load(second.output_root / "input-0000/native-depth.npy"),
    )
    assert not np.array_equal(
        np.load(first.output_root / "input-0000/metric-depth-m.npy"),
        np.load(second.output_root / "input-0000/metric-depth-m.npy"),
    )


def test_changed_companion_manifest_fails_before_worker_or_outputs(prepared):
    prepared, manifest = _with_companions(prepared)
    manifest.write_text("this is no longer the prepared request")
    with pytest.raises(RuntimeError, match="manifest changed after preparation"):
        pipeline.run(prepared)
    assert SessionFixture.calls == 0 and not prepared.output_root.exists()


def test_forged_companion_semantics_cannot_hide_behind_valid_manifest_receipt(prepared):
    from transformation_portal.core.execution_plan_v2 import ExecutionPlanV2, digest_payload

    prepared, _ = _with_companions(prepared, materials=False)
    payload = prepared.plan.to_payload()
    payload["inputs"][0]["companions"]["calibration"]["fx"] = 123
    payload.pop("plan_fingerprint_sha256")
    payload["plan_fingerprint_sha256"] = digest_payload(payload)
    forged = replace(prepared, plan=ExecutionPlanV2.from_payload(payload))
    with pytest.raises(RuntimeError, match="semantics differ from the prepared plan"):
        pipeline.run(forged)
    assert SessionFixture.calls == 0 and not prepared.output_root.exists()
