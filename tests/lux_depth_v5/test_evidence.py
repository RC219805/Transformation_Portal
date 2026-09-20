"""Rehashed forgeries must fail independent depth and photographic reconstruction."""

from __future__ import annotations

import copy
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from transformation_portal.core.depth_evidence import build_depth_evidence
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v4 import evidence as shared_evidence
from transformation_portal.lux_depth_v4.photography import create_proxy, decode_master, output_srgb_icc, write_delivery
from transformation_portal.lux_depth_v5.evidence import verify_execution_evidence_v3
from transformation_portal.lux_depth_v5.lifecycle import LuxDepthV5Request, prepare
from transformation_portal.lux_depth_v5.photography import align_depth, enhance_master_v5, generate_preview_maps_v5
from transformation_portal.lux_depth_v5.pipeline import LuxDepthV5Result
from transformation_portal.orchestrator.artifact_store.generation import GenerationPublisher
from transformation_portal.orchestrator.artifact_store.local import LocalArtifactStore
from transformation_portal.orchestrator.dispatch import DispatchFence, DispatchLocator

pytestmark = pytest.mark.unit


@pytest.fixture
def completed(tmp_path, monkeypatch, request):
    options = getattr(request, "param", {})
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    pixels = np.linspace(start=0.1, stop=0.7, num=29 * 43 * 3, dtype=np.float32).reshape((29, 43, 3))
    if options.get("alpha"):
        alpha = np.ones((29, 43, 1), np.float32)
        alpha[:3] = 0.5
        pixels = np.concatenate([pixels, alpha], axis=2)
    image_path = inputs / "photo.tif"
    tifffile.imwrite(
        image_path,
        pixels,
        photometric="rgb",
        extrasamples="unassalpha" if options.get("alpha") else None,
        iccprofile=output_srgb_icc() if options.get("icc") else None,
    )
    companion_path = None
    if options.get("calibration"):
        (tmp_path / "companions").mkdir()
        companion_path = tmp_path / "companions" / "companions.json"
        companion_path.write_bytes(
            canonicalize_json(
                {
                    "schema": "tp.lux.companions.v1",
                    "inputs": [
                        {
                            "path": image_path.name,
                            "source_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                            "calibration": {
                                "width": 43,
                                "height": 29,
                                "fx": 300.0,
                                "fy": 310.0,
                                "cx": 21.0,
                                "cy": 14.0,
                                "source": "synthetic camera contract fixture",
                                "coordinate_space": "canonical_master",
                            },
                        }
                    ],
                }
            )
        )
    publisher = GenerationPublisher(artifact_store=LocalArtifactStore(root_dir=tmp_path / "prepare-store"), record_store=None)
    prepared = prepare(
        LuxDepthV5Request(
            inputs,
            tmp_path / "attempt",
            input_color="linear_srgb",
            target_size=56,
            clarity=0.2,
            companions_manifest=companion_path,
            preview_maps=options.get("previews", False),
        ),
        publisher=publisher if options.get("managed", True) else None,
    )
    root = prepared.output_root
    root.mkdir()
    plan = prepared.plan.to_payload()
    source = plan["inputs"][0]
    input_id = source["id"]
    destination = root / input_id
    destination.mkdir()
    original = decode_master(image_path.read_bytes(), source_name=image_path.name, input_color="linear_srgb")
    proxy = create_proxy(original, 56)
    native = np.linspace(1, 9, np.prod(proxy.transform.padded_shape), dtype=np.float32).reshape(proxy.transform.padded_shape)
    sky = None if options.get("unknown_sky") else np.zeros(native.shape, bool)
    if sky is not None:
        sky[:3] = True
    depth = build_depth_evidence(native, sky, proxy, source["sha256"], companion=source.get("companions"))
    aligned = align_depth(depth, original, proxy)
    master, response = enhance_master_v5(original, aligned, clarity=0.2)
    arrays = {
        "source-master.npy": original.pixels,
        "master.npy": master.pixels,
        "native-depth.npy": native,
        "depth-valid.npy": depth.valid_mask,
        "native-numeric-valid.npy": depth.numeric_valid,
        "native-support.npy": depth.support_mask,
        "native-sky.npy": np.zeros(native.shape, bool) if sky is None else sky,
        "relative-depth.npy": aligned.relative_depth,
        "aligned-depth-valid.npy": aligned.valid_mask,
        "aligned-depth-support.npy": aligned.support_mask,
        "depth-support-score.npy": aligned.support_confidence,
        "depth-baseline.npy": master.pixels,
    }
    if depth.metric_map_m is not None:
        arrays.update({"metric-depth-m.npy": depth.metric_map_m, "aligned-metric-depth-m.npy": aligned.metric_map_m})
    if original.alpha is not None:
        arrays["alpha.npy"] = original.alpha
    if original.source_icc is not None:
        arrays["source-icc.npy"] = np.frombuffer(original.source_icc, np.uint8)
    if options.get("previews"):
        previews, preview_report = generate_preview_maps_v5(depth)
        arrays.update({f"preview-{name}.npy": values for name, values in previews.items()})
    else:
        preview_report = {"status": "disabled", "classification": "depth_derived_preview"}
    for name, array in arrays.items():
        np.save(destination / name, array, allow_pickle=False)
    delivery = write_delivery(master, destination / "delivery.tif")
    delivery["path"] = f"{input_id}/delivery.tif"
    descriptor = {
        "schema": "tp.lux.photograph.v2",
        "input_id": input_id,
        "source": original.to_payload(),
        "master": master.to_payload(),
        "depth": depth.to_payload(),
        "depth_content_sha256": depth.content_hash(),
        "depth_baseline": master.to_payload(),
        "depth_response": response,
        "aligned_depth": {
            "evidence": aligned.to_payload(),
            "content_sha256": aligned.content_hash(),
            "relative_path": f"{input_id}/relative-depth.npy",
            "validity_path": f"{input_id}/aligned-depth-valid.npy",
            "support_path": f"{input_id}/aligned-depth-support.npy",
            "support_score_path": f"{input_id}/depth-support-score.npy",
            "metric_path": f"{input_id}/aligned-metric-depth-m.npy" if aligned.metric_map_m is not None else None,
        },
        "delivery": delivery,
        "materials": {"status": "abstained", "reason": "no_authoritative_masks"},
        "preview_maps": preview_report,
    }
    (destination / "photograph.json").write_bytes(canonicalize_json(descriptor))
    (root / "execution-plan.json").write_bytes(prepared.canonical_plan_bytes)
    artifacts = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        kind = (
            "plan"
            if path.name == "execution-plan.json"
            else "descriptor" if path.suffix == ".json" else "image" if path.suffix == ".tif" else "array"
        )
        raw = path.read_bytes()
        artifacts.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
                "kind": kind,
                "input_id": None if kind == "plan" else input_id,
            }
        )
    runtime = {
        "backend_identity": {
            "model_canonical_key": plan["model"]["canonical_key"],
            "model_repo_id": plan["model"]["repo_id"],
            "model_lock_revision": plan["model"]["revision"],
            "actual_device": plan["device"],
        }
    }
    monkeypatch.setattr(
        shared_evidence.DA3RuntimeIdentityEvidence,
        "from_mapping",
        lambda value: SimpleNamespace(cacheable=True, to_mapping=lambda: copy.deepcopy(value)),
    )
    evidence = {
        "schema": "tp.lux.execution.evidence.v3",
        "complete": True,
        "synthetic": False,
        "plan_schema": plan["schema"],
        "plan_fingerprint_sha256": plan["plan_fingerprint_sha256"],
        "parent_runtime_sha256": "b" * 64,
        "worker_runtime": runtime,
        "inputs": plan["inputs"],
        "artifacts": artifacts,
        "executions": [
            {
                "input_id": input_id,
                "depth_cache_hit": False,
                "identities": {name: "c" * 64 for name in ("preprocess", "depth", "enhance", "output")},
            }
        ],
        "duration_seconds": 1.0,
        "cache": {"namespace": "identity-v5", "enabled": False, "hits": 0, "misses": 1},
        "production_acceptance": "pending",
    }
    evidence_path = root / "execution-evidence.json"
    evidence_path.write_bytes(canonicalize_json(evidence))
    result = LuxDepthV5Result(
        root,
        evidence_path,
        plan["plan_fingerprint_sha256"],
        tuple(row["path"] for row in artifacts) + ("execution-evidence.json",),
        1,
        0,
        1,
    )
    locator = DispatchLocator(
        "job_fixture", "attempt", "dispatch", hashlib.sha256(prepared.canonical_plan_bytes).hexdigest(), "tenant"
    )
    fence = DispatchFence(locator, "worker", 1, 0.0, str(root), str(tmp_path / "requested"))
    return SimpleNamespace(result=result, evidence=evidence, descriptor=descriptor, prepared=prepared, fence=fence)


def rehash(completed, relative):
    path = completed.result.output_root / relative
    record = next(item for item in completed.evidence["artifacts"] if item["path"] == relative)
    raw = path.read_bytes()
    record.update(sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    completed.result.evidence_path.write_bytes(canonicalize_json(completed.evidence))


def verify(completed):
    return verify_execution_evidence_v3(
        completed.result.output_root, expected_plan_sha256=completed.result.plan_fingerprint_sha256
    )


@pytest.mark.parametrize(
    "completed",
    [{}, {"alpha": True, "icc": True, "previews": True, "calibration": True}, {"unknown_sky": True}],
    indirect=True,
)
def test_complete_products_are_reconstructed_with_optional_geometry(completed):
    result = verify(completed)
    assert {artifact.path for artifact in result.artifacts} == set(completed.result.artifact_paths)
    assert result.to_payload()["schema"] == "tp.lux.execution.evidence.v3"


@pytest.mark.parametrize(
    "name",
    [
        "relative-depth.npy",
        "aligned-depth-valid.npy",
        "aligned-depth-support.npy",
        "depth-support-score.npy",
        "native-numeric-valid.npy",
        "native-support.npy",
        "depth-valid.npy",
        "depth-baseline.npy",
        "master.npy",
    ],
)
def test_rehashed_numeric_forgery_fails_independent_reconstruction(completed, name):
    relative = f"input-0000/{name}"
    values = np.load(completed.result.output_root / relative)
    index = tuple(size // 2 for size in values.shape)
    values[index] = not values[index] if values.dtype == bool else values[index] + 0.03125
    np.save(completed.result.output_root / relative, values, allow_pickle=False)
    rehash(completed, relative)
    with pytest.raises(ValueError, match="independently reconstructed"):
        verify(completed)


@pytest.mark.parametrize("completed", [{"calibration": True}], indirect=True)
@pytest.mark.parametrize("name", ["metric-depth-m.npy", "aligned-metric-depth-m.npy"])
def test_rehashed_metric_scale_forgery_is_rejected(completed, name):
    relative = f"input-0000/{name}"
    values = np.load(completed.result.output_root / relative)
    np.save(completed.result.output_root / relative, values * 1.5, allow_pickle=False)
    rehash(completed, relative)
    with pytest.raises(ValueError, match="metric.*independently reconstructed"):
        verify(completed)


@pytest.mark.parametrize("field", ["depth", "aligned_depth", "depth_response", "depth_baseline", "delivery"])
def test_rehashed_semantic_descriptor_forgery_is_rejected(completed, field):
    completed.descriptor[field]["forged"] = True
    relative = "input-0000/photograph.json"
    (completed.result.output_root / relative).write_bytes(canonicalize_json(completed.descriptor))
    rehash(completed, relative)
    with pytest.raises(ValueError, match="differs"):
        verify(completed)


def test_rehashed_delivery_pixels_are_bound_to_final_master(completed):
    relative = "input-0000/delivery.tif"
    image = tifffile.imread(completed.result.output_root / relative)
    image[12, 15] = 0
    tifffile.imwrite(
        completed.result.output_root / relative,
        image,
        photometric="rgb",
        iccprofile=output_srgb_icc(),
        extratags=[(274, "H", 1, 1, False)],
    )
    rehash(completed, relative)
    with pytest.raises(ValueError, match="delivery pixels"):
        verify(completed)


def test_unknown_declared_product_is_rejected_even_when_hashed(completed):
    relative = "input-0000/invented-confidence.npy"
    path = completed.result.output_root / relative
    np.save(path, np.ones((29, 43), np.float32), allow_pickle=False)
    completed.evidence["artifacts"].append(
        {
            "path": relative,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
            "kind": "array",
            "input_id": "input-0000",
        }
    )
    completed.result.evidence_path.write_bytes(canonicalize_json(completed.evidence))
    with pytest.raises(ValueError, match="exact admitted"):
        verify(completed)


def test_rehashed_npy_header_bomb_is_rejected_before_allocation(completed, monkeypatch):
    relative = "input-0000/relative-depth.npy"
    (completed.result.output_root / relative).write_bytes(b"\x93NUMPY\x02\x00" + (2**32 - 1).to_bytes(4, "little") + b"short")
    rehash(completed, relative)
    monkeypatch.setattr(np.lib.format, "read_array_header_2_0", lambda *a, **k: pytest.fail("Unbounded reader reached"))
    with pytest.raises(ValueError, match="bounded V5 array header"):
        verify(completed)


def test_runtime_identity_admission_remains_authoritative(completed, monkeypatch):
    monkeypatch.undo()
    with pytest.raises(ValueError, match="runtime-identity report"):
        verify(completed)


def test_v5_receipt_cannot_pass_the_unchanged_v4_schema(completed):
    with pytest.raises(ValueError, match="completion evidence"):
        shared_evidence.verify_execution_evidence_v2(
            completed.result.output_root, expected_plan_sha256=completed.result.plan_fingerprint_sha256
        )
