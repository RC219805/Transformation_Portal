"""Pure local admission preserves managed composition and publication boundaries."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tests.lux_depth_v5 import test_pipeline as v5_pipeline
from transformation_portal.core.execution_plan_v5 import ExecutionPlanV5, stage_output_budget
from transformation_portal.lux_depth_v4 import lifecycle as shared_lifecycle
from transformation_portal.lux_depth_v5.lifecycle import prepare as prepare_v5
from transformation_portal.lux_depth_v6.managed import ManagedLuxDepthV6Request, prepare, run, verify_managed_evidence
from transformation_portal.orchestrator.artifact_store.generation import GenerationPublicationLimits, GenerationPublisher

pytestmark = pytest.mark.unit
request_case = v5_pipeline.request_case


@pytest.fixture
def publication_limits():
    return GenerationPublicationLimits(
        max_files=200,
        max_file_bytes=4 * 1024**3,
        max_total_bytes=16 * 1024**3,
        max_manifest_bytes=1024**2,
    )


def test_local_composition_needs_no_publisher_or_store(request_case, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Local preparation must not construct a publisher or run inference")

    monkeypatch.setattr(GenerationPublisher, "__init__", forbidden)
    monkeypatch.setattr(v5_pipeline.SessionFixture, "__init__", forbidden)
    prepared = prepare(ManagedLuxDepthV6Request(request_case))
    payload = prepared.plan.to_payload()
    assert type(prepared.plan) is ExecutionPlanV5
    assert payload["pipeline"] == "lux_depth_v6"
    assert payload["inference"]["pipeline"] == "lux_depth_v5"
    assert payload["inference"]["publication"] == payload["publication"]
    assert payload["inference"]["resources"]["max_output_bytes"] == stage_output_budget(
        payload["resources"]["max_output_bytes"]
    )
    assert not prepared.output_root.exists()
    assert not request_case.cache_dir.exists()


@pytest.mark.parametrize("composition", [False, True])
def test_explicit_limits_preserve_managed_plan_bytes(request_case, composition):
    publisher = GenerationPublisher(artifact_store=None, record_store=None)
    request = ManagedLuxDepthV6Request(request_case) if composition else request_case
    prepare_request = prepare if composition else prepare_v5
    managed = prepare_request(request, publisher=publisher)
    local = prepare_request(request, publication_limits=publisher.limits)
    assert managed.canonical_plan_bytes == local.canonical_plan_bytes
    assert not request_case.output_dir.exists()


@pytest.mark.parametrize("composition", [False, True])
def test_ambiguous_publication_policy_fails_before_discovery(request_case, publication_limits, monkeypatch, composition):
    def forbidden(*args, **kwargs):
        pytest.fail("Ambiguous admission must fail before input discovery")

    monkeypatch.setattr(shared_lifecycle, "_discover_inputs", forbidden)
    publisher = GenerationPublisher(artifact_store=None, record_store=None)
    request = ManagedLuxDepthV6Request(request_case) if composition else request_case
    with pytest.raises(ValueError, match="never both"):
        (prepare if composition else prepare_v5)(request, publisher=publisher, publication_limits=publication_limits)
    assert not request_case.output_dir.exists()


@pytest.mark.parametrize("composition", [False, True])
def test_untyped_publication_policy_fails_before_discovery(request_case, publication_limits, monkeypatch, composition):
    def forbidden(*args, **kwargs):
        pytest.fail("Untyped admission must fail before input discovery")

    monkeypatch.setattr(shared_lifecycle, "_discover_inputs", forbidden)
    request = ManagedLuxDepthV6Request(request_case) if composition else request_case
    with pytest.raises(TypeError, match="exact GenerationPublicationLimits"):
        (prepare if composition else prepare_v5)(request, publication_limits=publication_limits.to_payload())
    assert not request_case.output_dir.exists()


@pytest.mark.parametrize("composition", [False, True])
@pytest.mark.parametrize("limit", ["max_files", "max_file_bytes", "max_manifest_bytes"])
def test_explicit_policy_is_enforced_before_model_resolution(
    request_case, publication_limits, monkeypatch, composition, limit
):
    def forbidden(*args, **kwargs):
        pytest.fail("Unpublishable admission must fail before model resolution")

    monkeypatch.setattr(shared_lifecycle, "resolve_model_contract", forbidden)
    request = ManagedLuxDepthV6Request(request_case) if composition else request_case
    with pytest.raises(ValueError, match="publisher"):
        (prepare if composition else prepare_v5)(request, publication_limits=replace(publication_limits, **{limit: 1}))
    assert not request_case.output_dir.exists()


def test_local_composite_runs_one_inference_and_replays_complete_inventory(request_case, publication_limits, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Local execution must not publish managed artifacts")

    monkeypatch.setattr(GenerationPublisher, "publish", forbidden)
    prepared = prepare(ManagedLuxDepthV6Request(request_case), publication_limits=publication_limits)
    result = run(prepared)
    verified = verify_managed_evidence(result.output_root, expected_plan_bytes=prepared.canonical_plan_bytes)
    assert v5_pipeline.SessionFixture.calls == 1
    assert verified.canonical_bytes == result.canonical_bytes
    assert verified.to_payload()["production_acceptance"] == "not_established"
    paths = {record.path for record in verified.artifacts}
    assert {
        "source-v5/execution-evidence.json",
        "v6/input-0000/delivery.tif",
        "v6/input-0000/preview.png",
        "v6/input-0000/depth-relative.tif",
        "v6/evidence.json",
        "execution-evidence.json",
    } <= paths


def test_local_composite_rejects_changed_execution_policy(request_case, publication_limits):
    prepared = prepare(ManagedLuxDepthV6Request(request_case), publication_limits=publication_limits)
    with pytest.raises(ValueError, match="publisher policy changed"):
        run(prepared, publication_limits=replace(publication_limits, max_files=publication_limits.max_files + 1))
    assert v5_pipeline.SessionFixture.calls == 0
    assert not prepared.output_root.exists()


def test_v5_without_publication_policy_keeps_standalone_contract(request_case):
    prepared = prepare_v5(request_case)
    assert "publication" not in prepared.plan.to_payload()
    assert prepared.plan.to_payload()["resources"]["max_output_bytes"] == request_case.max_output_bytes
    assert not prepared.output_root.exists()
