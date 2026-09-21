"""Portal descriptors are a bounded projection of the verified photographic inventory."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from transformation_portal.orchestrator.artifact_store.base import ArtifactStoreError
from transformation_portal.orchestrator.artifact_store.local import LocalArtifactStore
from transformation_portal.orchestrator.dispatch import DispatchFence, DispatchLocator
from transformation_portal.orchestrator.photography_adapter import ManagedPhotographyPublisher
from transformation_portal.portal import job_artifacts

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_projection_matches_manifest_without_scanning_or_discovering_preview_siblings(tmp_path, monkeypatch):
    output = tmp_path / "attempt"
    files = {}
    integrity = {}
    for relative, data in {
        "execution-plan.json": b"admitted plan",
        "execution-evidence.json": b"verified evidence",
        "input-0000/delivery.tif": b"verified photographic pixels",
    }.items():
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        files[relative] = path
        integrity[relative] = {"size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    (output / "input-0000/unverified.png").write_bytes(b"must not be listed")
    (output / "input-0000/delivery.tif.preview.png").write_bytes(b"unverified preview")
    monkeypatch.setattr(job_artifacts, "_index_job_artifacts", lambda **_kw: pytest.fail("scanned output tree"))
    monkeypatch.setattr(job_artifacts, "_artifact_preview_proxy_path", lambda *_a: pytest.fail("discovered preview sibling"))
    locator = DispatchLocator("job_items", "attempt", "dispatch", "a" * 64, "tenant")
    fence = DispatchFence(locator, "worker", 1, 100, str(output), str(tmp_path))
    captured = []

    class Records:
        async def commit_generation(self, observed, **kwargs):
            assert observed == fence
            captured.append(kwargs)
            return json.loads(kwargs["manifest_bytes"])

    publisher = ManagedPhotographyPublisher(
        artifact_store=LocalArtifactStore(root_dir=tmp_path / "store"), record_store=Records()
    )
    original = {
        "schema": "tp.lux.delivery.v3",
        "paths": list(files),
        "execution_evidence": "execution-evidence.json",
        "items": [{"path": "unverified.png"}],
    }
    manifest = await publisher.publish(
        fence,
        files,
        state="succeeded",
        exit_code=0,
        artifacts=original,
        run_summary={"pipeline": "lux_depth_v5"},
        expected_file_integrity=integrity,
    )
    projected = captured[0]["artifacts"]
    assert projected["schema"] == original["schema"]
    assert projected["paths"] == original["paths"]
    assert projected["execution_evidence"] == "execution-evidence.json"
    assert projected["indexed_count"] == len(files)
    assert projected["truncated"] is False
    assert original["items"] == [{"path": "unverified.png"}]
    by_path = {item["path"]: item for item in projected["items"]}
    assert list(by_path) == sorted(files)
    assert set(by_path) == {item["path"] for item in manifest["files"]}
    for entry in manifest["files"]:
        item = by_path[entry["path"]]
        assert item["relative_path"] == entry["path"]
        assert item["name"] == Path(entry["path"]).name
        assert item["size_bytes"] == entry["size_bytes"]
        assert item["sha256"] == entry["sha256"]
        assert item["fingerprint_status"] == "ok"
        assert item["url"] == item["download_url"] == job_artifacts._artifact_url(locator.job_id, entry["path"])
    image = by_path["input-0000/delivery.tif"]
    assert image["artifact_type"] == image["media_kind"] == "image"
    assert image["previewable"] is True
    assert image["browser_previewable"] is False
    assert "preview_url" not in image


@pytest.mark.asyncio
@pytest.mark.parametrize("mismatch", ["missing_integrity", "unverified_path", "wrong_profile"])
async def test_projection_rejects_inventory_that_is_not_the_exact_verified_v5_set(tmp_path, mismatch):
    files = {"execution-evidence.json": tmp_path / "does-not-exist"}
    integrity = {"execution-evidence.json": {"size_bytes": 1, "sha256": "a" * 64}}
    artifacts = {"schema": "tp.lux.delivery.v3", "paths": list(files), "execution_evidence": "execution-evidence.json"}
    if mismatch == "missing_integrity":
        integrity = None
    elif mismatch == "unverified_path":
        artifacts["paths"].append("unverified.png")
    else:
        artifacts["schema"] = "tp.lux.delivery.v2"
    locator = DispatchLocator("job_rejected_items", "attempt", "dispatch", "a" * 64, "tenant")
    fence = DispatchFence(locator, "worker", 1, 100, str(tmp_path), str(tmp_path))
    publisher = ManagedPhotographyPublisher(artifact_store=None, record_store=None)
    with pytest.raises(ArtifactStoreError, match="exact verified V5 inventory"):
        await publisher.publish(
            fence,
            files,
            state="succeeded",
            exit_code=0,
            artifacts=artifacts,
            run_summary={},
            expected_file_integrity=integrity,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "preview_maps,materials,calibration", [(p, m, c) for p in (False, True) for m in (False, True) for c in (False, True)]
)
@pytest.mark.parametrize("input_count", [1, 9])
async def test_existing_admission_reservation_already_bounds_complete_portal_projection(
    tmp_path, monkeypatch, preview_maps, materials, calibration, input_count
):
    from dataclasses import replace

    from transformation_portal.core.execution_plan_v2 import MAX_INPUTS
    from transformation_portal.ingest.canonical_json import canonicalize_json
    from transformation_portal.lux_depth_v4 import publication as shared_publication
    from transformation_portal.lux_depth_v5.publication import publication_paths, validate_publication_plan
    from transformation_portal.orchestrator.artifact_store.generation import GenerationPublisher

    limits = GenerationPublisher(artifact_store=None, record_store=None).limits
    configuration = {"target_size": 56, "preview_maps": preview_maps}
    if materials:
        configuration["materials_v4"] = {}
    payload = {
        "inputs": [
            {"id": f"input-{index:04d}", **({"companions": {"calibration": {}}} if calibration else {})}
            for index in range(input_count)
        ],
        "configuration": configuration,
        "resources": {"max_pixels": 1, "max_output_bytes": limits.max_total_bytes},
        "publication": limits.to_payload(),
    }
    captured = []

    def capture_reservation(value):
        captured.append(value)
        return canonicalize_json(value)

    monkeypatch.setattr(shared_publication, "canonicalize_json", capture_reservation)
    validate_publication_plan(payload, limits)
    reservation = captured[0]
    reserved_bytes = len(canonicalize_json(reservation))
    exact_limits = replace(limits, max_manifest_bytes=reserved_bytes)
    payload["publication"] = exact_limits.to_payload()
    validate_publication_plan(payload, exact_limits)

    class ExactlyAdmittedPublisher(ManagedPhotographyPublisher):
        @property
        def limits(self):
            return exact_limits

    async def capture_projection(_self, _fence, _files, **kwargs):
        return kwargs["artifacts"]

    monkeypatch.setattr(GenerationPublisher, "publish", capture_projection)
    paths = publication_paths(payload)
    locator = DispatchLocator("j" * 64, "attempt", "dispatch", "a" * 64, "t" * 64)
    fence = DispatchFence(locator, "worker", 1, 100, str(tmp_path), str(tmp_path))
    projected = await ExactlyAdmittedPublisher(artifact_store=None, record_store=None).publish(
        fence,
        {path: tmp_path / path for path in paths},
        state="succeeded",
        exit_code=0,
        artifacts={"schema": "tp.lux.delivery.v3", "paths": list(paths), "execution_evidence": "execution-evidence.json"},
        run_summary={},
        expected_file_integrity={path: {"size_bytes": limits.max_file_bytes, "sha256": "f" * 64} for path in paths},
    )
    assert len(canonicalize_json(projected)) <= reserved_bytes
    reserved_by_path = {entry["path"]: entry for entry in reservation["files"]}
    # Each reserved manifest record dominates BOTH one descriptor and its
    # separate paths entry, including their extra array delimiter. Thus the
    # inequality extends to any admitted subset or configured batch count.
    for item in projected["items"]:
        descriptor_bytes = len(canonicalize_json(item)) + len(canonicalize_json(item["path"])) + 1
        assert descriptor_bytes < len(canonicalize_json(reserved_by_path[item["path"]]))
    # All schema-admitted ordinal IDs have four digits. The fixed envelope
    # also fits, even with the largest possible inventory count (all options).
    assert len(f"input-{MAX_INPUTS - 1:04d}") == len("input-0000")
    projection_header = {**projected, "paths": [], "items": [], "indexed_count": 2 + MAX_INPUTS * 22}
    reservation_header = {**reservation, "files": []}
    assert len(canonicalize_json(projection_header)) <= len(canonicalize_json(reservation_header))
