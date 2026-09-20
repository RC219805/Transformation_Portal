"""Managed V4 work must fit the active publisher before inference begins."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest
from PIL import Image

from transformation_portal.core.execution_plan import ExecutionPlanError
from transformation_portal.core.execution_plan_v2 import ExecutionPlanV2, digest_payload
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v4 import LuxDepthV4Request, backend, pipeline, prepare, run
from transformation_portal.lux_depth_v4.publication import publication_paths, validate_publication_plan
from transformation_portal.orchestrator.artifact_store import generation
from transformation_portal.orchestrator.artifact_store.generation import GenerationPublisher
from transformation_portal.orchestrator.artifact_store.local import LocalArtifactStore

pytestmark = pytest.mark.unit


@pytest.fixture
def publisher(tmp_path):
    return GenerationPublisher(artifact_store=LocalArtifactStore(root_dir=tmp_path / "store"), record_store=None)


def _request(tmp_path, count, *, calibrated=0, previews=False):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    records = []
    for index in range(count):
        path = inputs / f"image-{index:04d}.png"
        Image.new("RGB", (3, 2)).save(path)
        if index < calibrated:
            records.append(
                {
                    "path": path.name,
                    "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "calibration": {
                        "width": 3,
                        "height": 2,
                        "fx": 3.0,
                        "fy": 3.0,
                        "cx": 1.0,
                        "cy": 0.5,
                        "source": "measured contract fixture",
                        "coordinate_space": "canonical_master",
                    },
                }
            )
    manifest = None
    if records:
        companions = tmp_path / "companions"
        companions.mkdir()
        manifest = companions / "companions.json"
        manifest.write_bytes(canonicalize_json({"schema": "tp.lux.companions.v1", "inputs": records}))
    return LuxDepthV4Request(
        inputs, tmp_path / "output", input_color="srgb", preview_maps=previews, companions_manifest=manifest
    )


@pytest.mark.parametrize(
    "count,calibrated,previews,reserved",
    [
        (19, 0, False, 192),
        (16, 16, False, 194),
        (15, 0, True, 197),
        (13, 13, True, 197),
        (18, 9, False, 200),
    ],
)
def test_managed_boundaries_bind_exact_budget_without_outputs(tmp_path, publisher, count, calibrated, previews, reserved):
    request = _request(tmp_path, count, calibrated=calibrated, previews=previews)
    prepared = prepare(request, publisher=publisher)
    payload = json.loads(prepared.canonical_plan_bytes)
    assert payload["publication"] == publisher.limits.to_payload()
    assert payload["resources"]["max_output_bytes"] == 16 * 1024**3
    assert request.max_output_bytes == 64 * 1024**3
    assert len(publication_paths(payload)) == reserved
    assert not prepared.output_root.exists()


@pytest.mark.parametrize(
    "count,calibrated,previews",
    [
        (20, 0, False),
        (25, 0, False),
        (17, 17, False),
        (16, 0, True),
        (14, 14, True),
        (18, 10, False),
    ],
)
def test_oversized_managed_batches_fail_before_device_probe(tmp_path, publisher, monkeypatch, count, calibrated, previews):
    request = replace(_request(tmp_path, count, calibrated=calibrated, previews=previews), device="mps")
    monkeypatch.setattr(backend, "probe_device", lambda *_args: pytest.fail("Oversized batch probed the native runtime"))
    with pytest.raises(ValueError, match="artifacts, exceeding publisher limit"):
        prepare(request, publisher=publisher)
    assert not request.output_dir.exists()


def test_standalone_limits_and_plan_bytes_remain_unbound(tmp_path):
    prepared = prepare(_request(tmp_path, 1024))
    payload = prepared.plan.to_payload()
    assert len(payload["inputs"]) == 1024
    assert payload["resources"]["max_output_bytes"] == 64 * 1024**3
    assert "publication" not in payload


@pytest.mark.parametrize(
    "requested,expected", [(1024**3, 1024**3), (16 * 1024**3, 16 * 1024**3), (64 * 1024**3, 16 * 1024**3)]
)
def test_managed_total_ceiling_is_canonical_and_preserves_lower_requested_budget(tmp_path, publisher, requested, expected):
    prepared = prepare(replace(_request(tmp_path, 1), max_output_bytes=requested), publisher=publisher)
    assert json.loads(prepared.canonical_plan_bytes)["resources"]["max_output_bytes"] == expected


def test_custom_file_count_limit_is_used_before_probe(tmp_path, publisher, monkeypatch):
    monkeypatch.setattr(generation, "MAX_GENERATION_FILES", 11)
    monkeypatch.setattr(backend, "probe_device", lambda *_args: pytest.fail("Invalid count reached device probe"))
    with pytest.raises(ValueError, match="12 artifacts, exceeding publisher limit 11"):
        prepare(replace(_request(tmp_path, 1), device="mps"), publisher=publisher)


def test_per_file_limit_is_checked_before_probe(tmp_path, publisher, monkeypatch):
    monkeypatch.setattr(generation, "MAX_GENERATION_FILE_BYTES", 1024**3)
    monkeypatch.setattr(backend, "probe_device", lambda *_args: pytest.fail("Invalid size reached device probe"))
    with pytest.raises(ValueError, match="per-file bound"):
        prepare(replace(_request(tmp_path, 1), device="mps"), publisher=publisher)


def test_manifest_limit_cannot_be_bypassed_by_raising_file_count(tmp_path, publisher, monkeypatch):
    monkeypatch.setattr(generation, "MAX_GENERATION_FILES", 20000)
    with pytest.raises(ValueError, match="manifest-byte limit"):
        prepare(_request(tmp_path, 1024), publisher=publisher)


@pytest.mark.parametrize(
    "changed", ["MAX_GENERATION_FILES", "MAX_GENERATION_FILE_BYTES", "MAX_GENERATION_BYTES", "MAX_MANIFEST_BYTES"]
)
def test_limit_drift_fails_before_backend_or_output(tmp_path, publisher, monkeypatch, changed):
    prepared = prepare(_request(tmp_path, 1), publisher=publisher)
    monkeypatch.setattr(generation, changed, getattr(generation, changed) + 1)
    monkeypatch.setattr(pipeline, "DA3Session", lambda *_args, **_kwargs: pytest.fail("Drift reached inference"))
    with pytest.raises(ValueError, match="limits changed after managed preparation"):
        run(prepared, publisher=publisher)
    assert not prepared.output_root.exists()


def test_managed_run_requires_publisher_and_standalone_cannot_silently_upgrade(tmp_path, publisher, monkeypatch):
    request = _request(tmp_path, 1)
    managed = prepare(request, publisher=publisher)
    standalone = prepare(request)
    monkeypatch.setattr(pipeline, "DA3Session", lambda *_args, **_kwargs: pytest.fail("Unbound execution reached inference"))
    with pytest.raises(ValueError, match="requires its publisher"):
        run(managed)
    with pytest.raises(ValueError, match="requires prepare"):
        run(standalone, publisher=publisher)
    assert not request.output_dir.exists()


def test_publication_binding_is_part_of_plan_fingerprint_and_closed_schema(tmp_path, publisher):
    managed = prepare(_request(tmp_path, 1), publisher=publisher)
    payload = managed.plan.to_payload()
    payload["publication"]["max_files"] += 1
    with pytest.raises(ExecutionPlanError, match="fingerprint"):
        ExecutionPlanV2.from_payload(payload)
    payload.pop("plan_fingerprint_sha256")
    payload["publication"]["max_files"] = 200.0
    payload["plan_fingerprint_sha256"] = digest_payload(payload)
    with pytest.raises(ExecutionPlanError, match="exact integer"):
        ExecutionPlanV2.from_payload(payload)


def test_valid_rehashed_plan_cannot_claim_over_limit_budget(tmp_path, publisher):
    payload = prepare(_request(tmp_path, 1), publisher=publisher).plan.to_payload()
    payload["resources"]["max_output_bytes"] += 1
    payload.pop("plan_fingerprint_sha256")
    payload["plan_fingerprint_sha256"] = digest_payload(payload)
    with pytest.raises(ExecutionPlanError, match="publication limit"):
        ExecutionPlanV2.from_payload(payload)


def test_per_file_exact_boundary_is_admissible(tmp_path, publisher):
    payload = prepare(_request(tmp_path, 1), publisher=publisher).plan.to_payload()
    maximum_file = payload["resources"]["max_pixels"] * 12 + 256
    limits = replace(publisher.limits, max_file_bytes=maximum_file)
    payload["publication"] = limits.to_payload()
    validate_publication_plan(payload, limits)
    smaller = replace(limits, max_file_bytes=maximum_file - 1)
    payload["publication"] = smaller.to_payload()
    with pytest.raises(ValueError, match="per-file bound"):
        validate_publication_plan(payload, smaller)
