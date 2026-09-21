"""Complete V5 graph and semantic publication evidence with a controlled worker.

Only worker inference, process monitoring, and runtime materialization are
fixtures; real preprocessing, cache, response, serialization, and independent
semantic verification execute end to end.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v4 import evidence as shared_evidence
from transformation_portal.lux_depth_v4 import pipeline as shared_pipeline
from transformation_portal.lux_depth_v5 import pipeline
from transformation_portal.lux_depth_v5.evidence import verify_execution_evidence_v3
from transformation_portal.lux_depth_v5.lifecycle import LuxDepthV5Request, prepare

pytestmark = pytest.mark.unit


class ParentFixture:
    sha256 = "a" * 64
    source_sha256 = "b" * 64

    def verify(self):
        pass


class SessionFixture:
    calls = 0
    sky_available = True

    def __init__(self, _python, plan, *, cancellation):
        self.plan = plan
        payload = plan.to_payload()
        backend = {
            "model_canonical_key": payload["model"]["canonical_key"],
            "model_repo_id": payload["model"]["repo_id"],
            "model_lock_revision": payload["model"]["revision"],
            "actual_device": payload["device"],
        }
        self.runtime = SimpleNamespace(
            runtime_identity_sha256="c" * 64,
            to_mapping=lambda: {"backend_identity": copy.deepcopy(backend)},
        )

    def compute(self, proxy):
        type(self).calls += 1
        shape = proxy.shape[:2]
        native = np.linspace(1, 9, shape[0] * shape[1], dtype=np.float32).reshape(shape)
        native[4:6, 4:6] = 0
        sky = np.zeros(shape, bool)
        sky[:2] = True
        arrays = {"native_depth": native}
        if type(self).sky_available:
            arrays["sky_mask"] = sky
        return arrays, {
            "native_semantics": "da3_metric_uncalibrated",
            "precision": self.plan.to_payload()["configuration"]["depth"]["precision"],
            "inference_recipe": "tp.da3.explicit_precision_sky.v1",
            "sky_available": type(self).sky_available,
        }

    def checkpoint(self):
        pass

    def verify(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass


@pytest.fixture
def request_case(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    levels = np.arange(1176, dtype=np.uint16).reshape(28, 42) + 10000
    tifffile.imwrite(inputs / "ramp.tif", np.repeat(levels[..., None], 3, axis=-1), photometric="rgb")
    monkeypatch.setattr(pipeline._ExecutionProfile, "session_type", SessionFixture)
    monkeypatch.setattr(shared_pipeline, "PhotographyRuntime", ParentFixture)
    monkeypatch.setattr(
        shared_pipeline,
        "require_process_supervisor",
        lambda: SimpleNamespace(Process=lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=0))),
    )
    # Runtime cryptographic/environment validation is independently covered by
    # worker tests. Preserve its source/model binding checks in this fixture.
    monkeypatch.setattr(
        shared_evidence.DA3RuntimeIdentityEvidence,
        "from_mapping",
        lambda value: SimpleNamespace(cacheable=True, to_mapping=lambda: copy.deepcopy(value)),
    )
    SessionFixture.calls = 0
    SessionFixture.sky_available = True
    return LuxDepthV5Request(
        inputs,
        tmp_path / "output",
        input_color="srgb",
        target_size=56,
        cache_dir=tmp_path / "cache",
        strength=0.25,
        clarity=0.2,
    )


def execute(request):
    prepared = prepare(request)
    result = pipeline.run(prepared)
    verified = verify_execution_evidence_v3(result.output_root, expected_plan_sha256=result.plan_fingerprint_sha256)
    assert {item.path for item in verified.artifacts} == set(result.artifact_paths)
    assert verified.to_payload()["production_acceptance"] == "pending"
    return result


def photograph(result):
    return json.loads((result.output_root / "input-0000/photograph.json").read_bytes())


def array(result, name):
    return np.load(result.output_root / "input-0000" / name, allow_pickle=False)


def test_cold_graph_serializes_native_status_and_verifies_all_derivatives(request_case):
    result = execute(request_case)
    descriptor = photograph(result)
    assert result.depth_cache_misses == SessionFixture.calls == 1
    assert descriptor["schema"] == "tp.lux.photograph.v3"
    assert descriptor["browser_preview"]["path"] == "input-0000/preview.png"
    assert "input-0000/preview.png" in result.artifact_paths
    assert descriptor["depth"]["schema"] == "tp.depth.artifact.v3"
    assert descriptor["depth"]["sky_status"] == "model_mask"
    assert descriptor["depth"]["confidence_status"] == "unavailable"
    assert descriptor["depth_response"]["schema"] == "tp.depth.response.v1"
    assert descriptor["depth"]["has_metric_depth"] is False
    assert descriptor["aligned_depth"]["metric_path"] is None
    assert array(result, "native-depth.npy").max() == 9
    assert array(result, "native-numeric-valid.npy")[:2].all()
    assert not array(result, "depth-valid.npy")[:2].any()
    assert array(result, "native-sky.npy")[:2].all()
    assert tifffile.imread(result.output_root / "input-0000/delivery.tif").dtype == np.uint16


def test_legacy_plan_executes_original_inventory_without_browser_preview(request_case):
    from transformation_portal.core.execution_plan_v2 import digest_payload
    from transformation_portal.core.execution_plan_v4 import ExecutionPlanV4, depth_photography_nodes

    prepared = prepare(request_case)
    payload = prepared.plan.to_payload()
    payload["configuration"].pop("browser_preview")
    payload["nodes"] = depth_photography_nodes(payload["configuration"])
    payload.pop("plan_fingerprint_sha256")
    payload["plan_fingerprint_sha256"] = digest_payload(payload)
    result = pipeline.run(replace(prepared, plan=ExecutionPlanV4.from_payload(payload)))
    verify_execution_evidence_v3(result.output_root, expected_plan_sha256=result.plan_fingerprint_sha256)
    assert photograph(result)["schema"] == "tp.lux.photograph.v2"
    assert "input-0000/preview.png" not in result.artifact_paths


def test_sky_invalid_and_unknown_pixels_receive_no_depth_or_clarity_edit(request_case):
    result = execute(request_case)
    valid = array(result, "aligned-depth-valid.npy")
    assert not valid[:2].any() and not valid[4:6, 4:6].any()
    source = array(result, "source-master.npy")
    np.testing.assert_array_equal(array(result, "master.npy")[~valid], source[~valid])
    np.testing.assert_array_equal(array(result, "relative-depth.npy")[~valid], 0)
    np.testing.assert_array_equal(array(result, "depth-support-score.npy")[~valid], 0)


def test_native_cache_survives_finishing_and_refinement_changes(request_case):
    first = execute(request_case)
    changed = replace(
        request_case, output_dir=request_case.output_dir.with_name("changed"), strength=0.7, clarity=0.0, refinement="bilinear"
    )
    second = execute(changed)
    assert first.depth_cache_misses == second.depth_cache_hits == SessionFixture.calls == 1
    assert first.plan_fingerprint_sha256 != second.plan_fingerprint_sha256
    np.testing.assert_array_equal(array(first, "native-depth.npy"), array(second, "native-depth.npy"))
    assert not np.array_equal(array(first, "master.npy"), array(second, "master.npy"))
    assert photograph(second)["aligned_depth"]["evidence"]["metadata"]["refinement"] == "bilinear"


def test_precision_changes_cannot_reuse_another_native_inference(request_case):
    first = execute(request_case)
    second = execute(replace(request_case, output_dir=request_case.output_dir.with_name("fp16"), precision="fp16"))
    assert first.depth_cache_misses == second.depth_cache_misses == 1
    assert second.depth_cache_hits == 0 and SessionFixture.calls == 2
    assert photograph(second)["depth"]["precision"]["compute"] == "fp16"


def test_missing_sky_is_explicit_abstention_and_retains_native_precision(request_case):
    SessionFixture.sky_available = False
    result = execute(request_case)
    descriptor = photograph(result)
    assert descriptor["depth"]["sky_status"] == "unavailable"
    assert not array(result, "aligned-depth-valid.npy").any()
    assert not array(result, "native-sky.npy").any()
    np.testing.assert_array_equal(array(result, "master.npy"), array(result, "source-master.npy"))
    assert len(np.unique(tifffile.imread(result.output_root / "input-0000/delivery.tif")[..., 0])) == 1176
    assert descriptor["depth_response"]["changed_pixels"] == 0


def add_calibration(request, factor=1):
    root = request.output_dir.parent / "calibration"
    root.mkdir(exist_ok=True)
    source_hash = hashlib.sha256((request.input_dir / "ramp.tif").read_bytes()).hexdigest()
    path = root / "companions.json"
    path.write_bytes(
        canonicalize_json(
            {
                "schema": "tp.lux.companions.v1",
                "inputs": [
                    {
                        "path": "ramp.tif",
                        "source_sha256": source_hash,
                        "calibration": {
                            "width": 42,
                            "height": 28,
                            "fx": 300 * factor,
                            "fy": 300 * factor,
                            "cx": 20.5,
                            "cy": 13.5,
                            "coordinate_space": "canonical_master",
                            "source": "measured fixture camera",
                        },
                    }
                ],
            }
        )
    )
    return replace(request, companions_manifest=path)


def test_adding_calibration_reuses_native_and_preserves_sky_and_numeric_masks(request_case):
    first = execute(request_case)
    calibrated = add_calibration(replace(request_case, output_dir=request_case.output_dir.with_name("metric")))
    second = execute(calibrated)
    assert second.depth_cache_hits == 1 and SessionFixture.calls == 1
    for name in ("native-depth.npy", "native-sky.npy", "native-numeric-valid.npy", "depth-valid.npy", "relative-depth.npy"):
        np.testing.assert_array_equal(array(first, name), array(second, name))
    native = array(second, "native-depth.npy")
    np.testing.assert_array_equal(array(second, "metric-depth-m.npy"), native)
    valid = array(second, "aligned-depth-valid.npy")
    np.testing.assert_array_equal(array(second, "aligned-metric-depth-m.npy")[~valid], 0)
    assert photograph(second)["depth"]["metric_status"] == "inferred_with_supplied_camera_calibration"


def test_calibration_change_changes_meters_without_repeating_inference(request_case):
    first_request = add_calibration(request_case)
    first = execute(first_request)
    second_request = add_calibration(replace(request_case, output_dir=request_case.output_dir.with_name("twice")), factor=2)
    second = execute(second_request)
    assert second.depth_cache_hits == 1 and SessionFixture.calls == 1
    np.testing.assert_array_equal(array(second, "metric-depth-m.npy"), array(first, "metric-depth-m.npy") * 2)
    np.testing.assert_array_equal(array(second, "master.npy"), array(first, "master.npy"))


def test_optional_previews_pass_independent_valid_neighborhood_verification(request_case):
    result = execute(replace(request_case, preview_maps=True))
    report = photograph(result)["preview_maps"]
    assert report["physical_material_estimate"] is False
    assert report["support_policy"] == "complete_usable_surface_filter_neighborhood"
    assert array(result, "preview-normal.npy").dtype == np.uint8
    np.testing.assert_array_equal(array(result, "preview-normal.npy")[4, 3], [128, 128, 255])


def test_materials_v4_applies_after_depth_baseline_with_separate_evidence(request_case):
    from transformation_portal.materials_v4.artifacts import write_evidence
    from transformation_portal.materials_v4.contracts import MaterialEvidence, RegionEvidence

    source_hash = hashlib.sha256((request_case.input_dir / "ramp.tif").read_bytes()).hexdigest()
    mask = np.zeros((28, 42), np.float32)
    # Exceed MaterialsV4's 500-pixel eligibility floor while preserving the
    # fixture's sky and invalid-depth holes as separately protected regions.
    mask[2:, :21] = 1
    mask[4:6, 4:6] = 0
    evidence = MaterialEvidence(source_hash, (28, 42), (RegionEvidence("glass-1", "glass", mask, 0.99),))
    root = request_case.output_dir.parent / "materials"
    root.mkdir()
    bundle = root / "evidence.json"
    write_evidence(evidence, bundle)
    manifest = root / "manifest.json"
    manifest.write_bytes(
        canonicalize_json(
            {
                "schema": "tp.lux.materials_manifest.v1",
                "inputs": [
                    {
                        "path": "ramp.tif",
                        "source_sha256": source_hash,
                        "evidence_path": bundle.name,
                        "evidence_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
                        "shape": [28, 42],
                    }
                ],
            }
        )
    )
    result = execute(replace(request_case, strength=0.0, clarity=0.0, materials_manifest=manifest))
    original = array(result, "source-master.npy")
    baseline = array(result, "depth-baseline.npy")
    final = array(result, "master.npy")
    np.testing.assert_array_equal(baseline, original)
    np.testing.assert_array_equal(array(result, "materials-baseline.npy"), baseline)
    np.testing.assert_array_equal(final[mask == 0], original[mask == 0])
    assert np.any(final[mask == 1] != original[mask == 1])
    assert photograph(result)["materials"]["status"] == "applied"


def test_changed_input_rejects_prepared_plan_before_inference(request_case):
    prepared = prepare(request_case)
    (request_case.input_dir / "ramp.tif").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="changed after preparation"):
        pipeline.run(prepared)
    assert SessionFixture.calls == 0
    assert not request_case.output_dir.exists()


def test_pre_cancelled_v5_does_not_publish_outputs(request_case):
    with pytest.raises(RuntimeError, match="cancelled"):
        pipeline.run(prepare(request_case), cancellation=lambda: True)
    assert SessionFixture.calls == 0
    assert not request_case.output_dir.exists()
