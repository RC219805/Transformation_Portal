"""Verified candidate generations reuse the existing fenced visibility transaction."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v4 import evidence as ev
from transformation_portal.lux_depth_v4.lifecycle import LuxDepthV4Request, prepare
from transformation_portal.lux_depth_v4.pipeline import LuxDepthV4Result
from transformation_portal.lux_depth_v4.publication import publish_result
from transformation_portal.orchestrator.artifact_store.base import ArtifactStoreError
from transformation_portal.orchestrator.artifact_store.generation import GenerationPublisher, validate_manifest
from transformation_portal.orchestrator.artifact_store.local import LocalArtifactStore
from transformation_portal.orchestrator.dispatch import DispatchFence, DispatchLocator

pytestmark = pytest.mark.unit


@pytest.fixture
def completed(tmp_path, monkeypatch, request):
    """Unit fixture stubs only the separately tested runtime identity deserializer."""
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (3, 2)).save(source / "image.png")
    root = tmp_path / "attempt"
    publisher = GenerationPublisher(
        artifact_store=LocalArtifactStore(root_dir=tmp_path / "preparation-store"), record_store=None
    )
    prepared = prepare(
        LuxDepthV4Request(source, root, input_color="srgb"),
        publisher=publisher if getattr(request, "param", True) else None,
    )
    plan = prepared.plan.to_payload()
    root.mkdir()
    (root / "execution-plan.json").write_bytes(prepared.canonical_plan_bytes)
    artifacts = []

    def observe(path, kind, input_id=None):
        data = path.read_bytes()
        artifacts.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(data).hexdigest(),
                "size_bytes": len(data),
                "kind": kind,
                "input_id": input_id,
            }
        )

    observe(root / "execution-plan.json", "plan")
    item = plan["inputs"][0]
    input_id = item["id"]
    (root / input_id).mkdir()
    for name in (
        "source-master.npy",
        "master.npy",
        "native-depth.npy",
        "depth-valid.npy",
        "relative-depth.npy",
        "aligned-depth-valid.npy",
    ):
        path = root / input_id / name
        path.write_bytes(b"opaque test array bytes")
        observe(path, "array", input_id)
    image = root / input_id / "delivery.tif"
    image.write_bytes(b"opaque test delivery bytes")
    observe(image, "image", input_id)
    descriptor = {
        "schema": "tp.lux.photograph.v1",
        "input_id": input_id,
        "source": {"source_sha256": item["sha256"], "shape": [2, 3]},
        "master": {"source_sha256": item["sha256"], "shape": [2, 3]},
        "depth": {"source_sha256": item["sha256"], "has_metric_depth": False},
        "aligned_depth": {
            "shape": [2, 3],
            "validity_path": f"{input_id}/aligned-depth-valid.npy",
            "relative_path": f"{input_id}/relative-depth.npy",
            "metric_path": None,
            "invalid_value": 0,
            "interpolation": "valid_weighted_bilinear_nearest_validity",
        },
        "delivery": {"path": f"{input_id}/delivery.tif"},
    }
    description = root / input_id / "photograph.json"
    description.write_bytes(canonicalize_json(descriptor))
    observe(description, "descriptor", input_id)
    runtime = {
        "backend_identity": {
            "model_canonical_key": plan["model"]["canonical_key"],
            "model_repo_id": plan["model"]["repo_id"],
            "model_lock_revision": plan["model"]["revision"],
            "actual_device": plan["device"],
        }
    }
    monkeypatch.setattr(
        ev.DA3RuntimeIdentityEvidence,
        "from_mapping",
        lambda value: SimpleNamespace(cacheable=True, to_mapping=lambda: copy.deepcopy(value)),
    )
    evidence = {
        "schema": "tp.lux.execution.evidence.v2",
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
        "cache": {"namespace": "identity-v4", "enabled": False, "hits": 0, "misses": 1},
        "production_acceptance": "pending",
    }
    evidence_path = root / "execution-evidence.json"
    evidence_path.write_bytes(canonicalize_json(evidence))
    result = LuxDepthV4Result(
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
    return result, fence, evidence


def _rewrite(result, payload):
    result.evidence_path.write_bytes(canonicalize_json(payload))


def test_complete_evidence_binds_every_artifact_and_plan(completed):
    result, _, _ = completed
    verified = ev.verify_execution_evidence_v2(result.output_root, expected_plan_sha256=result.plan_fingerprint_sha256)
    assert {item.path for item in verified.artifacts} == set(result.artifact_paths)
    assert verified.to_payload()["production_acceptance"] == "pending"


@pytest.mark.parametrize("mutation", ["missing", "changed", "descriptor"])
def test_aligned_depth_validity_is_required_bound_and_verified(completed, mutation):
    result, _, payload = completed
    relative = "input-0000/aligned-depth-valid.npy"
    if mutation == "missing":
        payload["artifacts"] = [item for item in payload["artifacts"] if item["path"] != relative]
        (result.output_root / relative).unlink()
    elif mutation == "changed":
        (result.output_root / relative).write_bytes(b"changed mask")
    else:
        path = result.output_root / "input-0000/photograph.json"
        descriptor = json.loads(path.read_bytes())
        descriptor["aligned_depth"]["validity_path"] = "input-0000/depth-valid.npy"
        path.write_bytes(canonicalize_json(descriptor))
        row = next(item for item in payload["artifacts"] if item["kind"] == "descriptor")
        row.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size_bytes=path.stat().st_size)
    _rewrite(result, payload)
    with pytest.raises(ValueError, match="required photographic artifact|differs|verified validity"):
        ev.verify_execution_evidence_v2(result.output_root, expected_plan_sha256=result.plan_fingerprint_sha256)


@pytest.mark.parametrize(
    "mutation",
    ["synthetic", "incomplete", "unknown", "inputs", "duplicate", "coverage", "cache", "worker", "path", "no_image", "hash"],
)
def test_incomplete_or_changed_completion_is_rejected(completed, mutation):
    result, _, payload = completed
    if mutation == "synthetic":
        payload["synthetic"] = True
    elif mutation == "incomplete":
        payload["complete"] = False
    elif mutation == "unknown":
        payload["unexpected"] = "field"
    elif mutation == "inputs":
        payload["inputs"][0]["sha256"] = "d" * 64
    elif mutation == "duplicate":
        payload["artifacts"].append(copy.deepcopy(payload["artifacts"][0]))
    elif mutation == "coverage":
        payload["executions"].append(copy.deepcopy(payload["executions"][0]))
    elif mutation == "cache":
        payload["cache"]["hits"] = 1
    elif mutation == "worker":
        payload["worker_runtime"]["backend_identity"]["model_lock_revision"] = "d" * 40
    elif mutation == "path":
        payload["artifacts"][1]["path"] = "../outside.npy"
    elif mutation == "no_image":
        payload["artifacts"] = [item for item in payload["artifacts"] if item["kind"] != "image"]
    else:
        payload["artifacts"][1]["sha256"] = "d" * 64
    _rewrite(result, payload)
    with pytest.raises(ValueError):
        ev.verify_execution_evidence_v2(result.output_root, expected_plan_sha256=result.plan_fingerprint_sha256)


def test_unlisted_output_and_changed_image_are_rejected(completed):
    result, _, _ = completed
    (result.output_root / "undeclared.txt").write_bytes(b"private")
    with pytest.raises(ValueError, match="inventory differs"):
        ev.verify_execution_evidence_v2(result.output_root, expected_plan_sha256=result.plan_fingerprint_sha256)
    (result.output_root / "undeclared.txt").unlink()
    (result.output_root / "input-0000/delivery.tif").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="differs"):
        ev.verify_execution_evidence_v2(result.output_root, expected_plan_sha256=result.plan_fingerprint_sha256)


def test_forged_descriptor_source_is_rejected_even_when_artifact_hash_is_updated(completed):
    result, _, payload = completed
    path = result.output_root / "input-0000/photograph.json"
    descriptor = json.loads(path.read_bytes())
    descriptor["depth"]["source_sha256"] = "f" * 64
    path.write_bytes(canonicalize_json(descriptor))
    row = next(item for item in payload["artifacts"] if item["kind"] == "descriptor")
    row.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size_bytes=path.stat().st_size)
    _rewrite(result, payload)
    with pytest.raises(ValueError, match="another source"):
        ev.verify_execution_evidence_v2(result.output_root, expected_plan_sha256=result.plan_fingerprint_sha256)


def test_runtime_identity_parser_is_not_bypassed(completed, monkeypatch):
    result, _, _ = completed
    monkeypatch.undo()
    with pytest.raises(ValueError, match="runtime-identity report"):
        ev.verify_execution_evidence_v2(result.output_root, expected_plan_sha256=result.plan_fingerprint_sha256)


def test_publication_reuses_generation_commit_and_preserves_pending_acceptance(completed, tmp_path):
    result, fence, _ = completed
    assert fence.locator.plan_digest == hashlib.sha256((result.output_root / "execution-plan.json").read_bytes()).hexdigest()
    assert fence.locator.plan_digest != result.plan_fingerprint_sha256
    commits = []

    class Records:
        async def commit_generation(self, observed, **kwargs):
            assert observed == fence
            commits.append(kwargs)
            return validate_manifest(kwargs["manifest_bytes"], fence=fence, generation_id=kwargs["generation_id"])

    publisher = GenerationPublisher(artifact_store=LocalArtifactStore(root_dir=tmp_path / "store"), record_store=Records())
    manifest = asyncio.run(publish_result(result, publisher=publisher, fence=fence))
    assert {item["path"] for item in manifest["files"]} == set(result.artifact_paths)
    assert len(commits) == 1
    assert commits[0]["run_summary"]["production_acceptance"] == "pending"


def test_verification_allows_worker_heartbeat_before_async_publication(completed, tmp_path, monkeypatch):
    from transformation_portal.lux_depth_v4 import publication

    result, fence, _ = completed
    verifying, heartbeat = threading.Event(), threading.Event()
    snapshot = publication.snapshot

    def slow_snapshot(*args, **kwargs):
        verifying.set()
        assert heartbeat.wait(timeout=3), "Verification blocked the worker heartbeat event loop"
        return snapshot(*args, **kwargs)

    monkeypatch.setattr(publication, "snapshot", slow_snapshot)

    class Records:
        async def commit_generation(self, observed, **kwargs):
            assert observed == fence and heartbeat.is_set()
            return validate_manifest(kwargs["manifest_bytes"], fence=fence, generation_id=kwargs["generation_id"])

    publisher = GenerationPublisher(artifact_store=LocalArtifactStore(root_dir=tmp_path / "store"), record_store=Records())

    async def exercise():
        async def renew_heartbeat():
            assert await asyncio.to_thread(verifying.wait, 3)
            heartbeat.set()

        completed_publication, _ = await asyncio.gather(
            publish_result(result, publisher=publisher, fence=fence), renew_heartbeat()
        )
        return completed_publication

    manifest = asyncio.run(exercise())
    assert {item["path"] for item in manifest["files"]} == set(result.artifact_paths)


@pytest.mark.parametrize("completed", [False], indirect=True)
def test_standalone_completion_cannot_skip_managed_preparation(completed, tmp_path):
    result, fence, _ = completed
    publisher = GenerationPublisher(artifact_store=LocalArtifactStore(root_dir=tmp_path / "store"), record_store=None)
    with pytest.raises(ValueError, match="requires prepare"):
        asyncio.run(publish_result(result, publisher=publisher, fence=fence))


def test_publication_rechecks_active_limits_before_staging(completed, tmp_path, monkeypatch):
    from transformation_portal.orchestrator.artifact_store import generation

    result, fence, _ = completed
    publisher = GenerationPublisher(artifact_store=LocalArtifactStore(root_dir=tmp_path / "store"), record_store=None)
    monkeypatch.setattr(generation, "MAX_GENERATION_FILES", 199)
    with pytest.raises(ValueError, match="limits changed after managed preparation"):
        asyncio.run(publish_result(result, publisher=publisher, fence=fence))


def test_managed_publication_rejects_unreserved_artifacts(completed, tmp_path):
    result, fence, payload = completed
    path = result.output_root / "input-0000/unreserved.npy"
    path.write_bytes(b"unexpected but hash-consistent output")
    relative = path.relative_to(result.output_root).as_posix()
    payload["artifacts"].append(
        {
            "path": relative,
            "kind": "array",
            "input_id": "input-0000",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
    )
    _rewrite(result, payload)
    result = replace(result, artifact_paths=result.artifact_paths + (relative,))
    publisher = GenerationPublisher(artifact_store=LocalArtifactStore(root_dir=tmp_path / "store"), record_store=None)
    with pytest.raises(ValueError, match="publication reservation"):
        asyncio.run(publish_result(result, publisher=publisher, fence=fence))


@pytest.mark.parametrize("mutation", ["root", "plan", "semantic", "inventory", "summary"])
def test_fence_or_result_mismatch_never_calls_publisher(completed, mutation):
    result, fence, _ = completed
    if mutation == "root":
        fence = replace(fence, output_root=str(result.output_root.parent))
    elif mutation == "plan":
        fence = replace(fence, locator=replace(fence.locator, plan_digest="f" * 64))
    elif mutation == "semantic":
        result = replace(result, plan_fingerprint_sha256="f" * 64)
    elif mutation == "inventory":
        result = replace(result, artifact_paths=result.artifact_paths[:-1])
    else:
        result = replace(result, input_count=2)

    class ForbiddenPublisher:
        async def publish(self, *args, **kwargs):
            pytest.fail("Invalid completion reached publication")

    with pytest.raises(ValueError):
        asyncio.run(publish_result(result, publisher=ForbiddenPublisher(), fence=fence))


@pytest.mark.parametrize("mutation", ["semantic_digest", "changed_plan_bytes"])
def test_dispatch_binds_exact_plan_bytes_independently_of_semantic_fingerprint(completed, mutation):
    result, fence, payload = completed
    if mutation == "semantic_digest":
        fence = replace(fence, locator=replace(fence.locator, plan_digest=result.plan_fingerprint_sha256))
    else:
        # Whitespace preserves parsed semantics but changes the admitted bytes.
        path = result.output_root / "execution-plan.json"
        raw = path.read_bytes() + b"\n"
        path.write_bytes(raw)
        row = next(item for item in payload["artifacts"] if item["kind"] == "plan")
        row.update(sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
        _rewrite(result, payload)
    # Completion can remain semantically valid while the dispatch binding fails.
    ev.verify_execution_evidence_v2(result.output_root, expected_plan_sha256=result.plan_fingerprint_sha256)

    class ForbiddenPublisher:
        async def publish(self, *args, **kwargs):
            pytest.fail("A plan with the wrong byte digest reached publication")

    with pytest.raises(ValueError, match="plan bytes do not match the admitted dispatch plan"):
        asyncio.run(publish_result(result, publisher=ForbiddenPublisher(), fence=fence))


def test_verified_plan_record_must_match_dispatch_even_when_admitted_bytes_are_restored(completed, monkeypatch):
    from transformation_portal.lux_depth_v4 import publication

    result, fence, payload = completed
    path = result.output_root / "execution-plan.json"
    admitted_bytes = path.read_bytes()
    changed_bytes = admitted_bytes + b"\n"
    path.write_bytes(changed_bytes)
    row = next(item for item in payload["artifacts"] if item["kind"] == "plan")
    row.update(sha256=hashlib.sha256(changed_bytes).hexdigest(), size_bytes=len(changed_bytes))
    _rewrite(result, payload)
    verify = publication.verify_execution_evidence_v2

    def verify_then_restore(*args, **kwargs):
        verified = verify(*args, **kwargs)
        path.write_bytes(admitted_bytes)
        return verified

    monkeypatch.setattr(publication, "verify_execution_evidence_v2", verify_then_restore)

    class ForbiddenPublisher:
        async def publish(self, *args, **kwargs):
            pytest.fail("Verified staging hash differed from the admitted plan")

    with pytest.raises(ValueError, match="plan bytes do not match the admitted dispatch plan"):
        asyncio.run(publish_result(result, publisher=ForbiddenPublisher(), fence=fence))


def test_staging_hash_check_closes_verification_to_publication_race(completed, tmp_path, monkeypatch):
    from transformation_portal.lux_depth_v4 import publication

    result, fence, _ = completed
    verify = publication.verify_execution_evidence_v2

    def verify_then_change(*args, **kwargs):
        verified = verify(*args, **kwargs)
        (result.output_root / "input-0000/delivery.tif").write_bytes(b"changed after verification")
        return verified

    monkeypatch.setattr(publication, "verify_execution_evidence_v2", verify_then_change)

    class Records:
        async def commit_generation(self, *args, **kwargs):
            pytest.fail("Changed artifact granted managed visibility")

    publisher = GenerationPublisher(artifact_store=LocalArtifactStore(root_dir=tmp_path / "store"), record_store=Records())
    with pytest.raises(ArtifactStoreError, match="changed after execution evidence"):
        asyncio.run(publish_result(result, publisher=publisher, fence=fence))


@pytest.mark.parametrize("with_expected", [False, True])
def test_existing_generation_callers_remain_compatible(tmp_path, with_expected):
    source = tmp_path / "attempt"
    source.mkdir()
    path = source / "output.txt"
    path.write_bytes(b"unchanged")
    fence = DispatchFence(
        DispatchLocator("job_compat", "attempt", "dispatch", "a" * 64, "tenant"), "worker", 1, 0.0, str(source), str(tmp_path)
    )

    class Records:
        async def commit_generation(self, _fence, **kwargs):
            return {"state": kwargs["state"]}

    publisher = GenerationPublisher(artifact_store=LocalArtifactStore(root_dir=tmp_path / "store"), record_store=Records())
    kwargs = {}
    if with_expected:
        kwargs["expected_file_integrity"] = {
            "output.txt": {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size_bytes": 9}
        }
    result = asyncio.run(
        publisher.publish(fence, {"output.txt": path}, state="succeeded", exit_code=0, artifacts={}, run_summary={}, **kwargs)
    )
    assert result == {"state": "succeeded"}


def test_metric_claim_without_admitted_calibration_is_rejected(completed):
    result, _, payload = completed
    path = result.output_root / "input-0000/photograph.json"
    descriptor = json.loads(path.read_bytes())
    descriptor["depth"]["has_metric_depth"] = True
    path.write_bytes(canonicalize_json(descriptor))
    row = next(item for item in payload["artifacts"] if item["kind"] == "descriptor")
    row.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size_bytes=path.stat().st_size)
    _rewrite(result, payload)
    with pytest.raises(ValueError, match="admitted calibration"):
        ev.verify_execution_evidence_v2(result.output_root, expected_plan_sha256=result.plan_fingerprint_sha256)


def test_admitted_calibration_requires_both_metric_artifacts(completed):
    from transformation_portal.core.execution_plan_v2 import ExecutionPlanV2, digest_payload, photography_nodes

    result, _, payload = completed
    path = result.output_root / "execution-plan.json"
    plan = json.loads(path.read_bytes())
    source = plan["inputs"][0]
    calibration = {
        "width": 3,
        "height": 2,
        "fx": 300,
        "fy": 300,
        "cx": 1,
        "cy": 1,
        "source": "measured fixture",
        "coordinate_space": "canonical_master",
    }
    source["companions"] = {"path": source["path"], "source_sha256": source["sha256"], "calibration": calibration}
    plan["nodes"] = photography_nodes(plan["configuration"], companions=True)
    plan["companions_manifest"] = {"path": "companions.json", "sha256": "b" * 64, "size_bytes": 100}
    plan.pop("plan_fingerprint_sha256")
    plan["plan_fingerprint_sha256"] = digest_payload(plan)
    plan = ExecutionPlanV2.from_payload(plan)
    path.write_bytes(plan.canonical_bytes)
    row = next(item for item in payload["artifacts"] if item["kind"] == "plan")
    row.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size_bytes=path.stat().st_size)
    payload["inputs"] = plan.to_payload()["inputs"]
    payload["plan_fingerprint_sha256"] = plan.plan_fingerprint_sha256
    path = result.output_root / "input-0000/photograph.json"
    descriptor = json.loads(path.read_bytes())
    descriptor["depth"].update(has_metric_depth=True, calibration={"input_intrinsics": calibration})
    descriptor["aligned_depth"]["metric_path"] = "input-0000/aligned-metric-depth-m.npy"
    path.write_bytes(canonicalize_json(descriptor))
    row = next(item for item in payload["artifacts"] if item["kind"] == "descriptor")
    row.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size_bytes=path.stat().st_size)
    _rewrite(result, payload)
    with pytest.raises(ValueError, match="omits an admitted calibrated"):
        ev.verify_execution_evidence_v2(result.output_root, expected_plan_sha256=plan.plan_fingerprint_sha256)
    for name in ("metric-depth-m.npy", "aligned-metric-depth-m.npy"):
        path = result.output_root / "input-0000" / name
        path.write_bytes(b"opaque calibrated array fixture")
        payload["artifacts"].append(
            {
                "path": f"input-0000/{name}",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
                "kind": "array",
                "input_id": "input-0000",
            }
        )
    _rewrite(result, payload)
    assert ev.verify_execution_evidence_v2(result.output_root, expected_plan_sha256=plan.plan_fingerprint_sha256)
