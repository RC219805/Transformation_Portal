"""V5 semantic admission must precede the unchanged fenced generation commit."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import numpy as np
import pytest

from tests.lux_depth_v5.test_evidence import completed, rehash  # noqa: F401 - shared fixture registration
from transformation_portal.lux_depth_v5.publication import publication_paths, publish_result, validate_publication_plan
from transformation_portal.orchestrator.artifact_store.generation import GenerationPublisher, validate_manifest
from transformation_portal.orchestrator.artifact_store.local import LocalArtifactStore

pytestmark = pytest.mark.unit


def test_publication_reuses_the_fenced_generation_commit(completed, tmp_path):
    commits = []

    class Records:
        async def commit_generation(self, fence, **kwargs):
            assert fence == completed.fence
            commits.append(kwargs)
            return validate_manifest(kwargs["manifest_bytes"], fence=fence, generation_id=kwargs["generation_id"])

    publisher = GenerationPublisher(artifact_store=LocalArtifactStore(root_dir=tmp_path / "store"), record_store=Records())
    manifest = asyncio.run(publish_result(completed.result, publisher=publisher, fence=completed.fence))
    assert {item["path"] for item in manifest["files"]} == set(completed.result.artifact_paths)
    assert len(commits) == 1
    assert commits[0]["run_summary"]["pipeline"] == "lux_depth_v5"
    assert commits[0]["run_summary"]["production_acceptance"] == "pending"
    assert commits[0]["artifacts"]["schema"] == "tp.lux.delivery.v3"


def test_reservation_accounts_for_all_unknown_optional_carriers(completed):
    payload = completed.prepared.plan.to_payload()
    paths = publication_paths(payload)
    assert len(paths) == 18
    assert "input-0000/source-icc.npy" in paths
    assert "input-0000/alpha.npy" in paths
    assert "input-0000/depth-confidence.npy" not in paths
    limits = GenerationPublisher(artifact_store=None, record_store=None).limits
    with pytest.raises(ValueError, match="limits changed"):
        validate_publication_plan(payload, replace(limits, max_files=17))
    payload["publication"]["max_files"] = 17
    with pytest.raises(ValueError, match="reserves 18"):
        validate_publication_plan(payload, replace(limits, max_files=17))


@pytest.mark.parametrize("completed", [{"managed": False}], indirect=True)
def test_standalone_result_cannot_skip_managed_admission(completed, tmp_path):
    publisher = GenerationPublisher(artifact_store=LocalArtifactStore(root_dir=tmp_path / "store"), record_store=None)
    with pytest.raises(ValueError, match="requires prepare"):
        asyncio.run(publish_result(completed.result, publisher=publisher, fence=completed.fence))


def test_semantic_forgery_never_reaches_publisher(completed, tmp_path, monkeypatch):
    path = completed.result.output_root / "input-0000/relative-depth.npy"
    values = np.load(path)
    values[10, 10] += 0.1
    np.save(path, values, allow_pickle=False)
    rehash(completed, "input-0000/relative-depth.npy")
    publisher = GenerationPublisher(artifact_store=LocalArtifactStore(root_dir=tmp_path / "store"), record_store=None)
    monkeypatch.setattr(publisher, "publish", lambda *args, **kwargs: pytest.fail("Invalid depth reached publication"))
    with pytest.raises(ValueError, match="independently reconstructed"):
        asyncio.run(publish_result(completed.result, publisher=publisher, fence=completed.fence))


@pytest.mark.parametrize("mutation", ["plan", "root", "inventory", "summary"])
def test_result_and_dispatch_fence_bind_the_verified_generation(completed, tmp_path, mutation):
    result = completed.result
    if mutation == "plan":
        result = replace(result, plan_fingerprint_sha256="f" * 64)
    elif mutation == "root":
        result = replace(result, output_root=tmp_path)
    elif mutation == "inventory":
        result = replace(result, artifact_paths=result.artifact_paths[:-1])
    else:
        result = replace(result, input_count=2)
    publisher = GenerationPublisher(artifact_store=LocalArtifactStore(root_dir=tmp_path / "store"), record_store=None)
    with pytest.raises(ValueError, match="does not match|differs"):
        asyncio.run(publish_result(result, publisher=publisher, fence=completed.fence))
