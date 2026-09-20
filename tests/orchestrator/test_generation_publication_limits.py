"""Managed admission observes the publisher's immutable, active resource limits."""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError, replace

import pytest

from transformation_portal.orchestrator.artifact_store import generation
from transformation_portal.orchestrator.artifact_store.base import ArtifactStoreError
from transformation_portal.orchestrator.artifact_store.generation import GenerationPublicationLimits, GenerationPublisher
from transformation_portal.orchestrator.artifact_store.local import LocalArtifactStore
from transformation_portal.orchestrator.dispatch import DispatchFence, DispatchLocator

pytestmark = pytest.mark.unit


@pytest.fixture(name="publisher")
def generation_publisher_fixture(tmp_path):
    return GenerationPublisher(artifact_store=LocalArtifactStore(root_dir=tmp_path / "store"), record_store=object())


def test_limits_snapshot_matches_active_enforcement_and_remains_immutable(publisher, monkeypatch):
    monkeypatch.setattr(generation, "MAX_GENERATION_FILES", 23)
    monkeypatch.setattr(generation, "MAX_GENERATION_FILE_BYTES", 1024)
    monkeypatch.setattr(generation, "MAX_GENERATION_BYTES", 8192)
    monkeypatch.setattr(generation, "MAX_MANIFEST_BYTES", 4096)
    snapshot = publisher.limits
    assert snapshot == GenerationPublicationLimits(23, 1024, 8192, 4096)
    monkeypatch.setattr(generation, "MAX_GENERATION_FILES", 7)
    assert snapshot.max_files == 23
    assert publisher.limits.max_files == 7
    with pytest.raises(FrozenInstanceError):
        snapshot.max_files = 99
    with pytest.raises(AttributeError):
        publisher.limits = snapshot


def test_limits_do_not_reresolve_startup_environment(publisher, monkeypatch):
    snapshot = publisher.limits
    monkeypatch.setenv("TP_MAX_INDEXED_ARTIFACTS", str(snapshot.max_files + 1))
    assert publisher.limits == snapshot


def test_limits_payload_roundtrip_is_independent(publisher):
    snapshot = publisher.limits
    payload = snapshot.to_payload()
    assert GenerationPublicationLimits.from_payload(payload) == snapshot
    payload["max_files"] += 1
    assert publisher.limits == snapshot
    assert payload != snapshot.to_payload()


@pytest.mark.parametrize("field", ["max_files", "max_file_bytes", "max_total_bytes", "max_manifest_bytes"])
@pytest.mark.parametrize("value", [0, -1, True, False, 1.0, "1", None])
def test_limits_require_positive_exact_integers(field, value):
    valid = GenerationPublicationLimits(1, 1, 1, 1)
    with pytest.raises(ValueError, match="positive exact integer"):
        replace(valid, **{field: value})
    with pytest.raises(ValueError, match="positive exact integer"):
        GenerationPublicationLimits.from_payload({**valid.to_payload(), field: value})


@pytest.mark.parametrize("mutation", ["missing", "unknown", "list", "none"])
def test_limits_reject_open_or_non_mapping_payloads(mutation):
    payload = GenerationPublicationLimits(1, 1, 1, 1).to_payload()
    if mutation == "missing":
        del payload["max_files"]
    elif mutation == "unknown":
        payload["unsupported"] = 1
    elif mutation == "list":
        payload = list(payload.items())
    else:
        payload = None
    with pytest.raises(ValueError, match="exactly the four supported fields"):
        GenerationPublicationLimits.from_payload(payload)


def test_observed_count_limit_is_the_enforced_publication_limit(publisher, tmp_path, monkeypatch):
    monkeypatch.setattr(generation, "MAX_GENERATION_FILES", 1)
    assert publisher.limits.max_files == 1
    fence = DispatchFence(
        DispatchLocator("job_limits", "attempt", "dispatch", "a" * 64, "tenant"),
        "worker",
        1,
        0.0,
        str(tmp_path / "attempt"),
        str(tmp_path / "requested"),
    )
    with pytest.raises(ArtifactStoreError, match="count exceeds limit"):
        asyncio.run(
            publisher.publish(
                fence,
                {"one.txt": tmp_path / "missing-one", "two.txt": tmp_path / "missing-two"},
                state="succeeded",
                exit_code=0,
                artifacts={},
                run_summary={},
            )
        )
