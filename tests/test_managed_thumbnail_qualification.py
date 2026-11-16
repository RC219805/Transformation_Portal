"""Mixed-workload qualification is opt-in, exact-source and metadata-only."""

from __future__ import annotations

import asyncio
import hashlib
import io
from dataclasses import replace
from types import SimpleNamespace
from urllib.parse import quote

import httpx
import pytest
from PIL import Image

from tests.test_managed_materials_qualification import fixture_case  # noqa: F401 - existing retained fixture
from transformation_portal.orchestrator import photographic_review_service as review
from transformation_portal.orchestrator.photographic_admission import (
    PhotographicAdmissionController,
    ReviewAdmissionConfig,
    ReviewCost,
    current_review_lease,
    operation_cost,
)
from transformation_portal.photographic_delivery import DeliveryImage
from transformation_portal.photographic_delivery import managed_qualification as managed

pytestmark = pytest.mark.unit


def test_mixed_preflight_declares_scope_without_launching_or_enabling_parent(case, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Preflight cannot launch a measurement worker")

    monkeypatch.setattr(managed, "_run_process", forbidden)
    before = dict(managed.os.environ)
    report = managed.qualify(case.manifest, case.output, "a" * 64, concurrent_thumbnails=True, preflight_only=True)
    assert report["status"] == "planned" and report["concurrent_thumbnails"] is True
    assert report["mixed_contention_source_index"] == 0
    assert report["production_flags_changed"] is False and dict(managed.os.environ) == before
    assert report["managed_profile_qualification"] == "not_established"
    assert report["config"]["memory_bytes"] == 12 * managed.offline.GIB
    assert report["config"]["noninteractive_slots"] == 1


@pytest.mark.parametrize(
    "options",
    [
        {"concurrent_thumbnails": 1},
        {"concurrent_thumbnails": True, "concurrent_inspection": False},
        {"concurrent_thumbnails": True, "scenario": "cancel_helper", "repeats": 1},
    ],
)
def test_mixed_option_requires_explicit_boolean_success_and_protected_inspection(case, options):
    with pytest.raises(ValueError):
        managed.qualify(case.manifest, case.output, "a" * 64, preflight_only=True, **options)
    assert not case.output.exists()


@pytest.mark.parametrize("enabled", [False, True])
def test_cli_forwards_only_explicit_mixed_opt_in(monkeypatch, enabled):
    observed = {}

    def qualify(*_args, **kwargs):
        observed.update(kwargs)
        return {"status": "planned", "blocking_reasons": []}

    monkeypatch.setattr(managed, "qualify", qualify)
    monkeypatch.setattr(
        managed.sys,
        "argv",
        [
            "qualification",
            "--manifest",
            "/private/manifest.json",
            "--output",
            "/private/fresh",
            "--expected-profile-identity",
            "a" * 64,
            *(["--concurrent-thumbnails"] if enabled else []),
        ],
    )
    assert managed.main() == 0
    assert observed["concurrent_thumbnails"] is enabled


@pytest.mark.parametrize("enabled", [False, True])
def test_only_private_worker_receives_thumbnail_flag(tmp_path, monkeypatch, enabled):
    environment = {"TP_PHOTO_REVIEW_THUMBNAILS_ENABLED": "1"}
    monkeypatch.setattr(managed.os, "environ", environment)
    monkeypatch.setattr(managed, "_database_url", lambda: "postgresql+asyncpg://unused/qualification_test")
    managed.offline._write_json(tmp_path / "specification.json", {"concurrent_thumbnails": enabled})
    managed._configure_worker(tmp_path)
    assert environment["TP_PHOTO_REVIEW_THUMBNAILS_ENABLED"] == ("1" if enabled else "0")


@pytest.mark.parametrize("verified", [False, True])
def test_mixed_report_requires_observed_contention_evidence(case, monkeypatch, verified):
    def run(root, mode, **_kwargs):
        if mode == "_worker":
            managed.offline._write_json(
                root / "managed-result.json",
                {
                    "runtime": {"managed_profile_identity": "a" * 64},
                    "baseline_rss_bytes": 0,
                    "peak_rss_bytes": 4 * managed.offline.GIB,
                    "transport_ownership_verified": True,
                    "mixed_thumbnail_contention_verified": verified,
                },
            )
        else:
            managed.offline._write_json(
                root / "verification.json", {"results": [{"source_index": 0, "product_identity_sha256": "a" * 64}] * 3}
            )
        return {}

    monkeypatch.setattr(managed, "_run_process", run)
    report = managed.qualify(case.manifest, case.output, "a" * 64, concurrent_thumbnails=True)
    assert (report["status"] == "measured") is verified
    assert report["blocking_reasons"] == ([] if verified else ["mixed_thumbnail_contention_unverified"])


def response_case():
    buffer = io.BytesIO()
    with Image.new("RGB", (16, 8), "white") as image:
        image.save(buffer, format="PNG")
    hint = {"artifact_path": "v6/input-0000/delivery.tif", "sha256": "a" * 64, "source_identity_sha256": "a" * 64}
    headers = {
        "content-type": "image/png",
        "cache-control": "no-store",
        "x-content-type-options": "nosniff",
        "x-tp-catalogue-identity-sha256": "c" * 64,
        "x-tp-thumbnail-max-edge": "256",
        "x-tp-thumbnail-max-bytes": "1048576",
        "x-tp-thumbnail-renderer": "tp.photo.thumbnail.v1",
        "x-tp-presentation-view": "clip_srgb",
        "x-tp-display-transform": "linear_srgb_to_8bit_srgb_clip_v1",
        "x-tp-native-width": "1024",
        "x-tp-native-height": "512",
        "x-tp-scene-id": "input-0000",
        "x-tp-artifact-path": quote(hint["artifact_path"], safe=""),
        "x-tp-source-sha256": "a" * 64,
        "x-tp-source-identity-sha256": "a" * 64,
        "x-tp-presentation-policy-sha256": "none",
        "x-tp-thumbnail-width": "16",
        "x-tp-thumbnail-height": "8",
    }
    return httpx.Response(200, headers=headers, content=buffer.getvalue()), hint


def test_thumbnail_measurement_retains_only_verified_identity_metadata():
    response, hint = response_case()
    evidence = managed._thumbnail_read_evidence(response, hint, "input-0000", "c" * 64, {"width": 1024, "height": 512})
    assert evidence["source_sha256"] == "a" * 64 and evidence["sha256"] == hashlib.sha256(response.content).hexdigest()
    assert evidence["size_bytes"] == len(response.content) and (evidence["width"], evidence["height"]) == (16, 8)
    assert all(isinstance(value, (str, int)) for value in evidence.values())


@pytest.mark.parametrize(
    "field,value",
    [
        ("x-tp-source-sha256", "b" * 64),
        ("x-tp-source-identity-sha256", "b" * 64),
        ("x-tp-catalogue-identity-sha256", "b" * 64),
        ("x-tp-presentation-policy-sha256", "invalid"),
        ("x-tp-native-width", "2048"),
        ("x-tp-thumbnail-width", "1000"),
        ("x-tp-thumbnail-renderer", "unqualified"),
    ],
)
def test_thumbnail_measurement_rejects_changed_identity_or_variant(field, value):
    response, hint = response_case()
    response.headers[field] = value
    with pytest.raises(ValueError):
        managed._thumbnail_read_evidence(response, hint, "input-0000", "c" * 64, {"width": 1024, "height": 512})


def test_thumbnail_measurement_rejects_oversize_body():
    response, hint = response_case()
    response = httpx.Response(200, headers=response.headers, content=b"x" * (managed.offline.MIB + 1))
    with pytest.raises(ValueError, match="byte bound"):
        managed._thumbnail_read_evidence(response, hint, "input-0000", "c" * 64, {"width": 1024, "height": 512})


@pytest.mark.asyncio
async def test_raster_observer_holds_real_decode_under_bulk_lease_and_restores_methods(tmp_path):
    service = review.PhotographicReviewService(ReviewAdmissionConfig(enabled=True))
    observer = managed._Observer(tmp_path, service)
    phases = []
    observer.capture = lambda phase, **_metadata: phases.append(phase)
    observer.pause_thumbnail = True
    observer.thumbnail_release.set()
    response, _ = response_case()
    image = DeliveryImage(
        "source.png", hashlib.sha256(response.content).hexdigest(), "scene", source_bytes=response.content, input_color="srgb"
    )
    display, decode = review._display_with_geometry, DeliveryImage.decode
    lease = await service.admission.acquire("tenant", operation_cost("thumbnail", service.admission.config))
    token = current_review_lease.set(lease)
    try:
        with managed._observe_operations(observer):
            result = await asyncio.to_thread(review._display_with_geometry, image, None, 256)
        assert result[1:3] == (16, 8)
        assert phases == ["thumbnail_decoded_raster_held"] and observer.thumbnail_entered.is_set()
        assert review._display_with_geometry is display and DeliveryImage.decode is decode
    finally:
        current_review_lease.reset(token)
        await lease.release()
        await service.close()


def test_queue_barrier_requires_observed_scheduling_queue():
    state = {"queued": 0}
    event = managed._AdmissionQueueObserved(SimpleNamespace(admission=SimpleNamespace(snapshot=lambda: state)))
    assert not event.is_set()
    state["queued"] = 1
    assert event.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("lost_ticket", [None, "native", "thumbnail"])
async def test_native_outer_transport_keeps_both_owners_with_one_known_queued_thumbnail(lost_ticket):
    controller = PhotographicAdmissionController(ReviewAdmissionConfig(enabled=True))
    native_ticket = await controller.begin_request("tenant")
    thumbnail_ticket = await controller.begin_request("tenant")
    native_lease = await controller.acquire("tenant", ReviewCost(6 * managed.offline.GIB, 0, interactive=False))
    finish = asyncio.Event()
    native = asyncio.create_task(finish.wait())
    thumbnail = asyncio.create_task(controller.acquire("tenant", operation_cost("thumbnail", controller.config)))
    observer = SimpleNamespace(service=SimpleNamespace(admission=controller))
    try:
        for _ in range(20):
            if controller.snapshot()["queued"]:
                break
            await asyncio.sleep(0)
        assert controller.snapshot()["queued"] == 1
        # The baseline probe correctly cannot account for an unannounced
        # second owner. Mixed mode must identify its known queued request.
        with pytest.raises(RuntimeError, match="before outer send"):
            managed._require_transport_reservation(observer, native)
        managed._require_transport_reservation(observer, native, queued_thumbnail=thumbnail)
        if lost_ticket is not None:
            await (native_ticket if lost_ticket == "native" else thumbnail_ticket).release()
            with pytest.raises(RuntimeError, match="before outer send"):
                managed._require_transport_reservation(observer, native, queued_thumbnail=thumbnail)
    finally:
        thumbnail.cancel()
        await asyncio.gather(thumbnail, return_exceptions=True)
        finish.set()
        await native
        await native_lease.release()
        await native_ticket.release()
        await thumbnail_ticket.release()
        await controller.close()


@pytest.mark.asyncio
async def test_transport_probe_cannot_substitute_a_completed_thumbnail_for_queued_owner():
    finish = asyncio.Event()
    native = asyncio.create_task(finish.wait())
    thumbnail = asyncio.create_task(asyncio.sleep(0))
    await thumbnail
    observer = SimpleNamespace(
        service=SimpleNamespace(
            admission=SimpleNamespace(
                snapshot=lambda: {
                    "active": 1,
                    "queued": 1,
                    "outstanding": 2,
                    "memory_reserved_bytes": 6 * managed.offline.GIB,
                }
            )
        )
    )
    try:
        with pytest.raises(RuntimeError, match="before outer send"):
            managed._require_transport_reservation(observer, native, queued_thumbnail=thumbnail)
    finally:
        finish.set()
        await native


def test_mixed_pressure_uses_largest_declared_native_vector_instead_of_first_source():
    def source(width, height, regions):
        return {"bound": {"width": width, "height": height, "alpha": False}, "recipe": {"regions": [{}] * regions}}

    small, large, more_regions = source(64, 96, 1), source(5000, 6000, 1), source(5000, 6000, 4)
    assert managed._contention_source_index([small, large, more_regions]) == 2
    assert managed._contention_source_index([more_regions, small, large]) == 0


@pytest.mark.parametrize(
    "field,value", [("memory_bytes", 14 * managed.offline.GIB), ("bulk_wait_seconds", 60), ("noninteractive_slots", 2)]
)
def test_actual_worker_config_cannot_inherit_different_admission_limits(field, value):
    config = ReviewAdmissionConfig(enabled=True)
    if field == "noninteractive_slots":
        config = replace(config, io_slots=3)
    service = SimpleNamespace(admission=SimpleNamespace(config=replace(config, **{field: value})))
    with pytest.raises(ValueError, match="unchanged governed admission budgets"):
        managed._require_qualification_config(service)
    managed._require_qualification_config(
        SimpleNamespace(admission=SimpleNamespace(config=ReviewAdmissionConfig(enabled=True)))
    )
