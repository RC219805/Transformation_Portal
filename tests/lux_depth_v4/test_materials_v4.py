"""Materials V4 admission, drift, and high-precision Lux graph regressions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from tests.lux_depth_v4.test_pipeline import ParentFixture, SessionFixture
from transformation_portal.core.execution_plan import ExecutionPlanError
from transformation_portal.core.execution_plan_v2 import ExecutionPlanV2, digest_payload, parse_execution_plan
from transformation_portal.core.execution_plan_v3 import ExecutionPlanV3
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v4 import LuxDepthV4Request, pipeline, prepare
from transformation_portal.materials_v4.artifacts import write_evidence
from transformation_portal.materials_v4.contracts import MaterialEvidence, RegionEvidence
from transformation_portal.materials_v4.engine import ResponsePolicy

pytestmark = pytest.mark.unit


@pytest.fixture
def material_case(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    source = inputs / "ramp.tif"
    ramp = np.arange(32 * 40, dtype=np.uint16).reshape(32, 40) + 10000
    pixels = np.repeat(ramp[..., None], 3, axis=-1)
    tifffile.imwrite(source, pixels, photometric="rgb")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    masks = tmp_path / "materials"
    masks.mkdir()
    mask = np.zeros((32, 40), dtype=np.float32)
    mask[:, :20] = 1
    evidence = MaterialEvidence(source_hash, (32, 40), (RegionEvidence("glass-1", "glass", mask, 0.99),))
    bundle = masks / "evidence.json"
    write_evidence(evidence, bundle)
    manifest = masks / "materials.json"
    manifest.write_bytes(
        canonicalize_json(
            {
                "schema": "tp.lux.materials_manifest.v1",
                "inputs": [
                    {
                        "path": "ramp.tif",
                        "source_sha256": source_hash,
                        "evidence_path": "evidence.json",
                        "evidence_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
                        "shape": [32, 40],
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(pipeline, "DA3Session", SessionFixture)
    monkeypatch.setattr(pipeline, "PhotographyRuntime", ParentFixture)
    monkeypatch.setattr(
        pipeline,
        "require_process_supervisor",
        lambda: SimpleNamespace(Process=lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=0))),
    )
    SessionFixture.calls = 0
    request = LuxDepthV4Request(
        inputs,
        tmp_path / "output",
        input_color="srgb",
        target_size=56,
        strength=0,
        materials_manifest=manifest,
        cache_dir=tmp_path / "cache",
    )
    return SimpleNamespace(request=request, manifest=manifest, bundle=bundle, evidence=evidence, pixels=pixels)


def test_opt_in_freezes_v3_without_changing_default_v2(material_case):
    request = material_case.request
    legacy = prepare(replace(request, materials_manifest=None))
    prepared = prepare(request)
    assert type(legacy.plan) is ExecutionPlanV2
    assert type(prepared.plan) is ExecutionPlanV3
    assert type(parse_execution_plan(prepared.canonical_plan_bytes)) is ExecutionPlanV3
    payload = prepared.plan.to_payload()
    assert payload["inputs"][0]["materials_v4"]["content_sha256"] == material_case.evidence.content_hash()
    assert payload["configuration"]["materials_v4"] == ResponsePolicy().to_payload()
    assert payload["nodes"][2]["stage"] == "tp.stage.lux.enhance.v2"
    assert "$materials" not in payload["nodes"][2]["inputs"].values()
    assert not prepared.output_root.exists() and SessionFixture.calls == 0


def test_real_graph_composes_once_and_preserves_unmasked_precision(material_case):
    prepared = prepare(material_case.request)
    result = pipeline.run(prepared)
    delivery = tifffile.imread(result.output_root / "input-0000/delivery.tif")
    assert delivery.dtype == np.uint16
    np.testing.assert_array_equal(delivery[:, 20:], material_case.pixels[:, 20:])
    assert np.all(delivery[:, :20] > material_case.pixels[:, :20])
    assert len(np.unique(delivery[..., 0])) > 1000
    descriptor = json.loads((result.output_root / "input-0000/photograph.json").read_bytes())
    receipt = descriptor["materials"]
    assert receipt["schema"] == "tp.materials.execution.v1"
    assert receipt["status"] == "applied" and receipt["changed_pixels"] == 640
    assert 0 < receipt["max_abs_delta"] <= ResponsePolicy().max_abs_delta
    assert receipt["evidence_sha256"] == material_case.evidence.content_hash()
    repeat = pipeline.run(replace(prepared, output_root=prepared.output_root.with_name("repeat")))
    assert repeat.depth_cache_hits == 1 and SessionFixture.calls == 1
    np.testing.assert_array_equal(tifffile.imread(repeat.output_root / "input-0000/delivery.tif"), delivery)


@pytest.mark.parametrize("kind", ["manifest", "bundle", "mask"])
def test_material_drift_refuses_before_inference_or_outputs(material_case, kind):
    prepared = prepare(material_case.request)
    target = {
        "manifest": material_case.manifest,
        "bundle": material_case.bundle,
        "mask": next(material_case.bundle.parent.glob("mask-*.npy")),
    }[kind]
    content = target.read_bytes()
    target.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
    with pytest.raises((ValueError, RuntimeError)):
        pipeline.run(prepared)
    assert not prepared.output_root.exists() and SessionFixture.calls == 0


@pytest.mark.parametrize("kind", ["schema", "graph", "source", "policy", "unknown", "budget", "shape"])
def test_resigned_invalid_plan_never_authorizes(material_case, kind):
    payload = prepare(material_case.request).plan.to_payload()
    if kind == "schema":
        payload["schema"] = "tp.execution.plan.v2"
    elif kind == "graph":
        payload["nodes"][2]["inputs"]["materials_v4"] = "$materials"
    elif kind == "source":
        payload["inputs"][0]["materials_v4"]["source_sha256"] = "f" * 64
    elif kind == "policy":
        payload["configuration"]["materials_v4"]["allow_supplied_confidence"] = 1
    elif kind == "unknown":
        payload["inputs"][0]["materials_v4"]["override"] = True
    elif kind == "budget":
        payload["inputs"][0]["materials_v4"]["size_bytes"] = True
    elif kind == "shape":
        payload["inputs"][0]["materials_v4"]["shape"] = [True, 40]
    payload.pop("plan_fingerprint_sha256")
    payload["plan_fingerprint_sha256"] = digest_payload(payload)
    with pytest.raises((ExecutionPlanError, ValueError)):
        ExecutionPlanV3.from_payload(payload)


def test_policy_change_invalidates_execution_identity(material_case):
    original = prepare(material_case.request)
    changed = prepare(replace(material_case.request, materials_policy=ResponsePolicy(max_abs_delta=0.001)))
    assert original.plan.plan_fingerprint_sha256 != changed.plan.plan_fingerprint_sha256
    first = pipeline.run(original)
    second = pipeline.run(replace(changed, output_root=changed.output_root.with_name("changed")))
    assert first.depth_cache_misses == second.depth_cache_misses == 1


def test_caller_cannot_rebind_material_root(material_case, tmp_path):
    prepared = prepare(material_case.request)
    with pytest.raises(ValueError, match="namespace"):
        pipeline.run(replace(prepared, materials_root=None))
    with pytest.raises(ValueError, match="separate from outputs"):
        pipeline.run(replace(prepared, output_root=material_case.bundle.parent / "output"))


def test_source_without_material_evidence_abstains(material_case):
    source = material_case.request.input_dir / "second.tif"
    tifffile.imwrite(source, material_case.pixels, photometric="rgb")
    result = pipeline.run(prepare(material_case.request))
    second = json.loads((result.output_root / "input-0001/photograph.json").read_bytes())
    assert second["materials"]["status"] == "abstained"
    assert second["materials"]["changed_pixels"] == 0
    np.testing.assert_array_equal(tifffile.imread(result.output_root / "input-0001/delivery.tif"), material_case.pixels)


def test_policy_requires_explicit_materials_manifest(material_case):
    with pytest.raises(ValueError, match="explicit materials manifest"):
        prepare(replace(material_case.request, materials_manifest=None, materials_policy=ResponsePolicy()))


def test_transient_bundle_swap_cannot_mix_manifest_and_semantic_identity(material_case, monkeypatch):
    from transformation_portal.lux_depth_v4 import materials

    original_bytes = material_case.bundle.read_bytes()
    original_loader = materials.load_evidence
    swapped = replace(material_case.evidence, regions=(replace(material_case.evidence.regions[0], semantic_confidence=0.01),))
    replacement = json.loads(original_bytes)
    replacement["regions"][0]["semantic_confidence"] = 0.01
    replacement["content_sha256"] = swapped.content_hash()

    def swap_during_load(*args, **kwargs):
        material_case.bundle.write_bytes(canonicalize_json(replacement))
        try:
            return original_loader(*args, **kwargs)
        finally:
            material_case.bundle.write_bytes(original_bytes)

    monkeypatch.setattr(materials, "load_evidence", swap_during_load)
    with pytest.raises(ValueError, match="content|semantic"):
        prepare(material_case.request)
    assert not material_case.request.output_dir.exists()
