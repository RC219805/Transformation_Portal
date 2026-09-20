"""V5 plans admit one exact graph and retain existing filesystem/model constraints."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from PIL import Image

from transformation_portal.core.execution_identity_v5 import materialize_stage_identity
from transformation_portal.core.execution_plan_v2 import digest_payload, parse_execution_plan
from transformation_portal.core.execution_plan_v4 import ExecutionPlanV4
from transformation_portal.lux_depth_v4.lifecycle import LuxDepthV4Request
from transformation_portal.lux_depth_v5 import LuxDepthV5Request, prepare

pytestmark = pytest.mark.unit


@pytest.fixture
def request_v5(tmp_path):
    source = tmp_path / "inputs"
    source.mkdir()
    Image.new("RGB", (42, 28), (45, 78, 91)).save(source / "photo.png")
    return LuxDepthV5Request(source, tmp_path / "output", input_color="srgb", cache_dir=tmp_path / "cache")


def test_planning_is_read_only_and_has_distinct_authority(request_v5, monkeypatch):
    monkeypatch.setattr("subprocess.Popen", lambda *a, **kw: pytest.fail("planning loaded a model"))
    prepared = prepare(request_v5)
    payload = prepared.plan.to_payload()
    assert payload["schema"] == "tp.execution.plan.v4" and payload["pipeline"] == "lux_depth_v5"
    assert payload["configuration"]["depth"] == {"precision": "fp32", "refinement": "guided_bilinear"}
    assert not prepared.output_root.exists() and not prepared.cache_root.exists()
    assert (
        prepare(replace(request_v5, output_dir=request_v5.output_dir.with_name("other"))).canonical_plan_bytes
        == prepared.canonical_plan_bytes
    )
    with pytest.raises(TypeError):
        prepare(LuxDepthV4Request(request_v5.input_dir, request_v5.output_dir))


@pytest.mark.parametrize(
    "change",
    [
        {"precision": "automatic"},
        {"precision": True},
        {"refinement": "invent_detail"},
        {"target_size": 1000},
        {"model_key": "da3-research"},
    ],
)
def test_invalid_policy_or_model_never_creates_outputs(request_v5, change):
    with pytest.raises((ValueError, TypeError)):
        prepare(replace(request_v5, **change))
    assert not request_v5.output_dir.exists()


def test_plan_rejects_rehashed_graph_and_policy_forgery(request_v5):
    payload = prepare(request_v5).plan.to_payload()
    payload["nodes"][1]["configuration"]["sky_mask_policy"] = "ignore_sky"
    payload.pop("plan_fingerprint_sha256")
    payload["plan_fingerprint_sha256"] = digest_payload(payload)
    with pytest.raises(ValueError, match="closed depth evidence graph"):
        ExecutionPlanV4.from_payload(payload)


def _depth_identity(plan):
    node = plan.to_payload()["nodes"][1]
    return materialize_stage_identity(
        plan,
        SimpleNamespace(name="depth", version=node["stage"]),
        "input-0000",
        {"proxy": "e" * 64},
        "a" * 64,
        "b" * 64,
        "c" * 64,
    )


def test_native_identity_reuses_inference_for_artistic_changes_only(request_v5):
    original = prepare(request_v5).plan
    finish = prepare(replace(request_v5, strength=0.75, clarity=0.3, refinement="bilinear")).plan
    precision = prepare(replace(request_v5, precision="fp16")).plan
    assert original.plan_fingerprint_sha256 != finish.plan_fingerprint_sha256
    assert _depth_identity(original) == _depth_identity(finish)
    assert _depth_identity(original) != _depth_identity(precision)


def test_v4_executor_rejects_v5_carrier(request_v5):
    from transformation_portal.lux_depth_v4.pipeline import run

    with pytest.raises(TypeError):
        run(prepare(request_v5))


def test_core_version_dispatch_retains_v5_authority(request_v5):
    prepared = prepare(request_v5)
    parsed = parse_execution_plan(prepared.canonical_plan_bytes)
    assert type(parsed) is ExecutionPlanV4
    assert parsed.canonical_bytes == prepared.canonical_plan_bytes


def test_python_plan_input_does_not_coerce_tuples_into_json_arrays(request_v5):
    payload = prepare(request_v5).plan.to_payload()
    payload["inputs"] = tuple(payload["inputs"])
    with pytest.raises(ValueError, match="explicit JSON"):
        ExecutionPlanV4.from_payload(payload)


def test_legacy_material_companions_fail_before_device_probe(request_v5, tmp_path, monkeypatch):
    import hashlib

    import numpy as np

    from transformation_portal.ingest.canonical_json import canonicalize_json

    root = tmp_path / "companions"
    root.mkdir()
    mask = root / "water.npy"
    np.save(mask, np.ones((28, 42), np.float32))
    record = {
        "path": "photo.png",
        "source_sha256": hashlib.sha256((request_v5.input_dir / "photo.png").read_bytes()).hexdigest(),
        "materials": {
            "masks": {"water": {"path": mask.name, "sha256": hashlib.sha256(mask.read_bytes()).hexdigest()}},
            "confidences": {"water": 0.99},
            "coordinate_space": "canonical_master",
        },
    }
    manifest = root / "companions.json"
    manifest.write_bytes(canonicalize_json({"schema": "tp.lux.companions.v1", "inputs": [record]}))
    monkeypatch.setattr(
        "transformation_portal.lux_depth_v4.backend.probe_device", lambda *_: pytest.fail("invalid companion probed device")
    )
    with pytest.raises(ValueError, match="legacy companion"):
        prepare(replace(request_v5, companions_manifest=manifest, device="mps"))
    assert not request_v5.output_dir.exists()
