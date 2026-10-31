"""Authenticated native Materials revisions reuse bounded review and delivery."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import threading
import uuid
import zipfile
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from starlette.responses import JSONResponse

from transformation_portal.api.routes.photographic_review import create_photographic_review_router
from transformation_portal.api.v1.photographic_review import RevisionRequest
from transformation_portal.lux_depth_v4.photography import output_srgb_icc
from transformation_portal.orchestrator.photographic_admission import ReviewAdmissionConfig
from transformation_portal.orchestrator.photographic_review_access import ReviewAccessError, ReviewContext, ReviewRuntime
from transformation_portal.orchestrator.photographic_review_service import PhotographicReviewService
from transformation_portal.orchestrator.photographic_review_store import MemoryReviewStore, ReviewConflict

pytestmark = pytest.mark.unit


def recipe():
    return {
        "schema": "tp.photo.materials_recipe.v1",
        "input_stage": "display_referred",
        "regions": [{"id": "stone-1", "label": "stone", "region": [0.1, 0.1, 0.9, 0.9], "strength": 0.03}],
    }


@pytest.fixture
def review_factory(monkeypatch):
    from transformation_portal.orchestrator import photographic_review_service as service_module
    from transformation_portal.orchestrator import photographic_review_store as stores

    @contextmanager
    def create(*, enabled="1", admission="1", version="v1", alpha=False):
        monkeypatch.setenv("TP_PHOTO_REVIEW_NATIVE_MATERIALS_ENABLED", enabled)
        monkeypatch.setenv("TP_PHOTO_REVIEW_ENABLED", admission)
        store = MemoryReviewStore()
        monkeypatch.setattr(stores, "get_review_store", lambda: store)
        monkeypatch.setattr(service_module, "get_review_store", lambda: store)
        pixels = np.full((64, 96, 3), 24000, dtype=np.uint16)
        pixels[:, :, 0] += np.arange(96, dtype=np.uint16)[None, :] * 200
        if alpha:
            opacity = np.full((64, 96, 1), 65535, dtype=np.uint16)
            opacity[:, :8] = 0
            opacity[:, 8:16] = 32768
            pixels = np.concatenate([pixels, opacity], axis=2)
        encoded = io.BytesIO()
        tifffile.imwrite(
            encoded,
            pixels,
            photometric="rgb",
            extrasamples="unassalpha" if alpha else None,
            iccprofile=output_srgb_icc(),
            metadata=None,
        )
        raw = encoded.getvalue()
        path = "v6/input-0000/delivery.tif"
        sha = hashlib.sha256(raw).hexdigest()
        ref = {
            "artifact_path": path,
            "sha256": sha,
            "source_identity_sha256": sha,
            "size_bytes": len(raw),
            "scene_id": "input-0000",
            "source_name": "delivery.tif",
            "label": "Stone room",
            "role": "final",
            "input_color": "auto",
            "kind": "indexed",
        }
        blobs, assets, reads, writes, discarded = {path: raw}, {}, [], [], []
        controls = SimpleNamespace(authorized=True, fail_publish=None)

        async def revalidate():
            if not controls.authorized:
                raise ReviewAccessError("Parent job access withdrawn", status=403, code="FORBIDDEN")

        async def materialize(selected, destination, limit):
            await revalidate()
            reads.append(selected.get("artifact_path", selected.get("storage_path")))
            source = assets.get(selected.get("storage_path")) if selected["kind"] == "review_asset" else blobs.get(path)
            if source is None or len(source) > limit or len(source) != selected["size_bytes"]:
                raise ReviewAccessError("Retained artifact unavailable")
            digest = hashlib.sha256(source).hexdigest()
            if digest != selected["sha256"]:
                raise ReviewAccessError("Retained artifact changed")
            destination.write_bytes(source)
            return {"sha256": digest, "size_bytes": len(source)}

        async def publish(storage, source, mime):
            await revalidate()
            if controls.fail_publish == len(writes) + 1:
                raise OSError("fixture immutable publication interrupted")
            assert storage not in assets
            assets[storage] = source.read_bytes()
            writes.append((storage, mime))

        async def discard(paths):
            discarded.extend(paths)
            for storage in paths:
                assets.pop(storage, None)

        job = "job_materials_" + uuid.uuid4().hex
        context = ReviewContext(
            job,
            "tenant_a",
            {"username": "reviewer", "assurance": "fixture_authenticated"},
            [ref],
            materialize,
            publish,
            discard,
            revalidate,
            binding_identity="a" * 64,
        )

        def error(status, **values):
            return JSONResponse({"success": False, "data": None, "error": values}, status_code=status)

        async def authorize(request, job_id):
            if request.headers.get("x-api-key") != "fixture-key":
                return error(401, code="UNAUTHORIZED")
            if request.headers.get("x-tenant", "tenant_a") != "tenant_a":
                return error(403, code="FORBIDDEN")
            if job_id != job:
                return error(404, code="NOT_FOUND")
            await revalidate()
            return context

        app = FastAPI()
        app.include_router(create_photographic_review_router(ReviewRuntime(authorize, error, authorize_admission=authorize)))
        with TestClient(app, headers={"x-api-key": "fixture-key"}) as client:
            yield SimpleNamespace(
                client=client,
                base=f"/{version}/jobs/{job}/photographic-review",
                job_id=job,
                path=path,
                sha=sha,
                context=context,
                store=store,
                blobs=blobs,
                assets=assets,
                reads=reads,
                writes=writes,
                discarded=discarded,
                controls=controls,
            )

    return create


def payload(case, **updates):
    return {
        "artifact_path": case.path,
        "sha256": case.sha,
        "source_identity_sha256": case.sha,
        "scene_id": "input-0000",
        "expected_version": 0,
        "recipe": recipe(),
        "preview_only": False,
        **updates,
    }


def inspect_source(case):
    response = case.client.get(case.base)
    assert response.status_code == 200, response.text
    return response.json()["data"]


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_native_revision_preview_save_inspect_approve_deliver_and_revoke(review_factory, version):
    with review_factory(version=version) as case:
        catalogue = inspect_source(case)
        capability = catalogue["revision_constraints"]["native_materials"]
        assert capability == {
            "enabled": True,
            "recipe_schema": "tp.photo.materials_recipe.v1",
            "labels": ["foliage", "glass", "sky", "stone", "water"],
            "max_regions": 4,
        }
        preview = case.client.post(case.base + "/revisions", json=payload(case, preview_only=True))
        assert preview.status_code == 200, preview.text
        assert preview.json()["data"]["preview_data_url"].startswith("data:image/png;base64,")
        assert not case.assets and not case.writes
        saved = case.client.post(case.base + "/revisions", json=payload(case))
        assert saved.status_code == 200, saved.text
        revision = saved.json()["data"]["revision"]
        assert revision["receipt"]["schema"] == "tp.photo.materials_revision.v1"
        assert revision["review_status"] == "unreviewed" and revision["receipt"]["approval_transferred"] is False
        assert revision["receipt"]["output_bit_depth"] == 16
        assert len(case.assets) == 4 and "storage_path" not in saved.text
        current = inspect_source(case)
        final = next(row for row in current["scenes"][0]["variants"] if row["role"] == "final")
        assert final["artifact_path"] == revision["artifact_path"] and final["bit_depth"] == 16
        assert final["diagnostics"] == revision["diagnostics"]
        crop = case.client.get(
            case.base + "/native-crop",
            params={
                "artifact_path": revision["artifact_path"],
                "sha256": revision["sha256"],
                "x": 3,
                "y": 5,
                "width": 11,
                "height": 9,
                "view": "recorded",
            },
        )
        assert crop.status_code == 200, crop.text
        assert crop.headers["x-tp-native-width"] == "11" and crop.headers["x-tp-native-height"] == "9"
        for product in revision["diagnostics"]:
            response = case.client.get(product["url"])
            assert response.status_code == 200, response.text
            assert hashlib.sha256(response.content).hexdigest() == product["sha256"]
            assert response.headers["cache-control"] == "no-store"
            if product["content_type"] == "application/json":
                diagnostic = response.json()
                assert diagnostic["diagnostic_only"] is True and diagnostic["approval_authority"] is False
        export = {
            "property_name": "Stone room",
            "presets": ["photographic_master", "website"],
            "approved_sha256": [revision["sha256"]],
            "expected_version": 1,
        }
        assert case.client.post(case.base + "/delivery", json=export).status_code == 400
        decision = {
            "artifact_path": revision["artifact_path"],
            "sha256": revision["sha256"],
            "status": "approved",
            "expected_version": 1,
        }
        approved = case.client.post(case.base + "/decisions", json=decision)
        assert approved.status_code == 200, approved.text
        delivered = case.client.post(case.base + "/delivery", json={**export, "expected_version": 2})
        assert delivered.status_code == 200, delivered.text
        delivery = delivered.json()["data"]
        downloaded = case.client.get(delivery["download_url"])
        assert downloaded.status_code == 200
        with zipfile.ZipFile(io.BytesIO(downloaded.content)) as archive:
            assert archive.testzip() is None
            approval = next(name for name in archive.namelist() if name.endswith("/approval-record.json"))
            assert json.loads(archive.read(approval))["masters"][0]["sha256"] == revision["sha256"]
            sums = next(name for name in archive.namelist() if name.endswith("/_package/SHA256SUMS.txt"))
            for line in archive.read(sums).decode().splitlines():
                digest, member = line.split("  ", 1)
                assert hashlib.sha256(archive.read(member)).hexdigest() == digest
        revoked = case.client.post(
            case.base + "/decisions",
            json={
                **decision,
                "status": "needs_revision",
                "expected_version": 3,
            },
        )
        assert revoked.status_code == 200
        assert case.client.get(delivery["download_url"]).status_code == 409
        assert case.client.get(revision["diagnostics"][0]["url"]).status_code == 200
        assert hashlib.sha256(case.blobs[case.path]).hexdigest() == case.sha


@pytest.mark.parametrize("enabled,admission", [("0", "1"), ("1", "0"), ("true", "1")])
def test_native_requires_both_explicit_feature_and_admission_before_source_reads(review_factory, enabled, admission):
    with review_factory(enabled=enabled, admission=admission) as case:
        response = case.client.post(case.base + "/revisions", json=payload(case))
        assert response.status_code == 503 and response.json()["error"]["code"] == "FEATURE_DISABLED"
        assert not case.reads and not case.writes and not case.assets


@pytest.mark.parametrize(
    "mutation",
    [
        {"regions": []},
        {"mask_path": "/private/mask.npy"},
        {"input_stage": "working_linear"},
        {"regions": [{"id": "x", "label": "unsupported", "region": [0, 0, 1, 1], "strength": 0.03}]},
    ],
)
def test_native_recipe_is_closed_and_validated_before_source_reads(review_factory, mutation):
    with review_factory() as case:
        response = case.client.post(case.base + "/revisions", json=payload(case, recipe={**recipe(), **mutation}))
        assert response.status_code == 400 and not case.reads and not case.writes


def test_native_rejects_reference_and_stale_request_before_materialization(review_factory):
    with review_factory() as case:
        reference = case.client.post(case.base + "/revisions", json=payload(case, reference_sha256=case.sha))
        assert reference.status_code == 400 and not case.reads
        stale = case.client.post(case.base + "/revisions", json=payload(case, expected_version=1))
        assert stale.status_code == 409 and not case.reads and not case.writes


@pytest.mark.parametrize("source_kind", ["float32", "recorded_working"])
def test_managed_native_rejects_unsupported_or_conflicting_retained_source(review_factory, source_kind):
    with review_factory() as case:
        if source_kind == "float32":
            encoded = io.BytesIO()
            tifffile.imwrite(
                encoded,
                np.full((64, 96, 3), 0.4, dtype=np.float32),
                photometric="rgb",
                iccprofile=output_srgb_icc(),
                metadata=None,
            )
            data = encoded.getvalue()
            case.blobs[case.path] = data
            case.sha = hashlib.sha256(data).hexdigest()
            case.context.references[0].update(sha256=case.sha, source_identity_sha256=case.sha, size_bytes=len(data))
        else:
            from transformation_portal.lux_depth_v6.color import RenderRecipe
            from transformation_portal.photographic_delivery.presentation import PresentationPolicy

            policy = PresentationPolicy("working_linear", RenderRecipe(), "revision_v2", "d" * 64, case.sha, case.sha)
            case.context.references[0].update(
                presentation_policy=policy.to_payload(), presentation_policy_sha256=policy.sha256
            )
        inspect_source(case)
        response = case.client.post(case.base + "/revisions", json=payload(case))
        assert response.status_code == 400 and not case.writes and not case.assets


def test_native_diagnostics_revalidate_parent_and_tenant_without_granting_approval(review_factory):
    with review_factory() as case:
        inspect_source(case)
        saved = case.client.post(case.base + "/revisions", json=payload(case)).json()["data"]["revision"]
        url = saved["diagnostics"][0]["url"]
        assert case.client.get(url, headers={"x-api-key": "invalid"}).status_code == 401
        assert case.client.get(url, headers={"x-tenant": "tenant_b"}).status_code == 403
        assert case.client.get(url.replace(case.job_id, "job_other")).status_code == 404
        assert case.client.get(url.replace(saved["id"], "unknown")).status_code == 404
        assert case.client.get(url.rsplit("/", 1)[0] + "/master.tif").status_code == 404
        case.controls.authorized = False
        assert case.client.get(url).status_code == 403
        case.controls.authorized = True
        case.blobs[case.path] = b"changed"
        changed = case.client.get(url)
        assert changed.status_code == 409
        assert not case.client.get(case.base).json().get("data")


def test_native_saved_preview_matches_inline_bytes_and_preserves_straight_alpha(review_factory):
    with review_factory(alpha=True) as case:
        inspect_source(case)
        original = case.blobs[case.path]
        saved = case.client.post(case.base + "/revisions", json=payload(case))
        assert saved.status_code == 200, saved.text
        data = saved.json()["data"]
        revision = data["revision"]
        preview_ref = next(row for row in revision["diagnostics"] if row["filename"] == "materials-preview.png")
        preview = case.client.get(preview_ref["url"])
        assert preview.status_code == 200, preview.text
        assert preview.content == base64.b64decode(data["preview_data_url"].split(",", 1)[1])
        assert hashlib.sha256(preview.content).hexdigest() == revision["receipt"]["preview"]["sha256"]
        with Image.open(io.BytesIO(preview.content)) as display:
            assert display.mode == "RGBA" and display.size == (96, 64)
            assert np.array_equal(
                np.asarray(display)[..., 3], np.rint(tifffile.imread(io.BytesIO(original))[..., 3] / 257).astype(np.uint8)
            )
        master = next(value for key, value in case.assets.items() if key.endswith("/master.tif"))
        actual = tifffile.imread(io.BytesIO(master))
        assert actual.dtype == np.uint16 and actual.shape == (64, 96, 4)
        assert np.array_equal(actual[..., 3], tifffile.imread(io.BytesIO(original))[..., 3])
        assert case.blobs[case.path] == original


@pytest.mark.parametrize("failure", ["conflict", "publication"])
def test_native_atomic_publication_discards_only_uncommitted_products(review_factory, monkeypatch, failure):
    with review_factory() as case:
        inspect_source(case)
        if failure == "publication":
            case.controls.fail_publish = 3
        else:

            async def conflict(*_args, **_kwargs):
                raise ReviewConflict("concurrent fixture change")

            monkeypatch.setattr(case.store, "mutate", conflict)
        response = case.client.post(case.base + "/revisions", json=payload(case))
        assert response.status_code == (409 if failure == "conflict" else 503)
        assert len(case.discarded) == (4 if failure == "conflict" else 2)
        assert not case.assets and inspect_source(case)["state_version"] == 0


@pytest.mark.parametrize(
    "corruption", ["receipt_schema", "receipt_hash", "product_hash", "product_mime", "product_path", "missing_product"]
)
def test_native_diagnostic_authority_corruption_fails_before_source_or_product_reads(review_factory, corruption):
    with review_factory() as case:
        inspect_source(case)
        saved = case.client.post(case.base + "/revisions", json=payload(case)).json()["data"]["revision"]
        entry = case.store.states[case.job_id, "tenant_a"]["revisions"][0]
        if corruption == "receipt_schema":
            entry["receipt"]["schema"] = "tp.photo.materials_revision.v999"
        elif corruption == "receipt_hash":
            entry["receipt"]["revision_sha256"] = "0" * 64
        elif corruption == "product_hash":
            entry["native_products"][0]["sha256"] = "0" * 64
        elif corruption == "product_mime":
            entry["native_products"][0]["content_type"] = "text/html"
        elif corruption == "product_path":
            entry["native_products"][0]["storage_path"] = "photographic-review/other/product.png"
        else:
            entry["native_products"].pop()
        reads = len(case.reads)
        response = case.client.get(saved["diagnostics"][0]["url"])
        assert response.status_code == 409 and len(case.reads) == reads
        response = case.client.get(case.base)
        assert response.status_code == 409 and len(case.reads) == reads


@pytest.mark.asyncio
async def test_native_direct_call_requires_owned_reservation_before_source_loading(monkeypatch):
    from unittest.mock import AsyncMock

    from transformation_portal.orchestrator.photographic_admission import ReviewOverload

    monkeypatch.setenv("TP_PHOTO_REVIEW_NATIVE_MATERIALS_ENABLED", "1")
    service = PhotographicReviewService(ReviewAdmissionConfig(enabled=True))
    context = SimpleNamespace(materialize=AsyncMock(side_effect=AssertionError("No unreserved source reads")))
    request = RevisionRequest(
        artifact_path="final.tif", sha256="a" * 64, scene_id="input-0000", expected_version=0, recipe=recipe()
    )
    try:
        with pytest.raises(ReviewOverload, match="capacity"):
            await service.revision(context, request)
        context.materialize.assert_not_awaited()
    finally:
        await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "lease_kind", ["render", "foreign", "released", "memory", "scratch", "decode", "io", "profile", "tenant"]
)
async def test_native_direct_call_rejects_inapplicable_or_incomplete_live_lease(review_factory, lease_kind):
    from dataclasses import replace

    from transformation_portal.orchestrator.photographic_admission import ReviewOverload, current_review_lease, operation_cost

    with review_factory() as case:
        service = PhotographicReviewService(ReviewAdmissionConfig(enabled=True))
        foreign = PhotographicReviewService(ReviewAdmissionConfig(enabled=True))
        geometry = {"width": 96, "height": 64, "alpha": False}
        service._remember_inspection(case.context, case.context.references[0], geometry)
        request = RevisionRequest(**payload(case))
        state = await service.state(case.context)
        cost = service._resource_cost(case.context, state, "revision", (request,), [geometry])
        if lease_kind == "render":
            cost = operation_cost("render", service.admission.config, [geometry])
        elif lease_kind == "memory":
            cost = replace(cost, memory_bytes=1)
        elif lease_kind == "scratch":
            cost = replace(cost, scratch_bytes=1, filesystems=())
        elif lease_kind == "decode":
            cost = replace(cost, decode_slots=0)
        elif lease_kind == "io":
            cost = replace(cost, io_slots=0)
        elif lease_kind == "profile":
            cost = replace(cost, native_materials_regions=0)
        owner = foreign.admission if lease_kind == "foreign" else service.admission
        lease = await owner.acquire("tenant_b" if lease_kind == "tenant" else "tenant_a", cost)
        if lease_kind == "released":
            await lease.release()
        token = current_review_lease.set(lease)
        try:
            with pytest.raises(ReviewOverload, match="capacity"):
                await service.revision(case.context, request)
            assert not case.reads and not case.writes
        finally:
            current_review_lease.reset(token)
            await lease.release()
            await service.close()
            await foreign.close()


@pytest.mark.asyncio
async def test_native_cancellation_drains_worker_before_releasing_lease(review_factory, monkeypatch):
    from transformation_portal.photographic_delivery import materials

    with review_factory() as case:
        service = PhotographicReviewService(ReviewAdmissionConfig(enabled=True))
        service._remember_inspection(case.context, case.context.references[0], {"width": 96, "height": 64, "alpha": False})
        started, finish = threading.Event(), threading.Event()
        original = materials.create_materials_revision

        def controlled(image, selected_recipe):
            started.set()
            if not finish.wait(5):
                raise AssertionError("Fixture worker did not receive completion")
            return original(image, selected_recipe)

        monkeypatch.setattr(materials, "create_materials_revision", controlled)
        request = RevisionRequest(**payload(case))
        task = asyncio.create_task(service.execute(case.context, service.revision, request))
        try:
            assert await asyncio.to_thread(started.wait, 2)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done() and service.admission.snapshot()["active"] == 1
            finish.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert service.admission.snapshot()["active"] == 0
            assert not case.assets and not case.writes
        finally:
            finish.set()
            await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["save", "repeated_cancellation"])
async def test_native_direct_sibling_cannot_reuse_consumed_lease(review_factory, monkeypatch, outcome):
    from transformation_portal.orchestrator.photographic_admission import ReviewOverload, current_review_lease
    from transformation_portal.photographic_delivery import materials

    with review_factory() as case:
        service = PhotographicReviewService(ReviewAdmissionConfig(enabled=True))
        geometry = {"width": 96, "height": 64, "alpha": False}
        service._remember_inspection(case.context, case.context.references[0], geometry)
        request = RevisionRequest(**payload(case))
        state = await service.state(case.context)
        cost = service._resource_cost(case.context, state, "revision", (request,), [geometry])
        lease = await service.admission.acquire("tenant_a", cost)
        started, finish = threading.Event(), threading.Event()
        original = materials.create_materials_revision

        def controlled(image, selected_recipe):
            started.set()
            if not finish.wait(5):
                raise AssertionError("Fixture worker did not receive completion")
            return original(image, selected_recipe)

        monkeypatch.setattr(materials, "create_materials_revision", controlled)
        token = current_review_lease.set(lease)
        task = asyncio.create_task(service.revision(case.context, request))
        try:
            assert await asyncio.to_thread(started.wait, 2)
            claim = lease.native_revision_owner
            reads = len(case.reads)
            with pytest.raises(ReviewOverload) as rejected:
                await service.revision(case.context, request)
            assert rejected.value.reason == "revision_reservation_already_used"
            legacy_request = RevisionRequest(**payload(case, recipe={"exposure_stops": 0.1}))
            with pytest.raises(ReviewOverload) as rejected:
                await service.revision(case.context, legacy_request)
            assert rejected.value.reason == "revision_reservation_already_used"
            assert lease.native_revision_owner is claim and len(case.reads) == reads
            assert service.admission.snapshot()["active"] == 1
            if outcome == "repeated_cancellation":
                task.cancel()
                await asyncio.sleep(0)
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done() and service.admission.snapshot()["active"] == 1
                with pytest.raises(ReviewOverload) as rejected:
                    await service.revision(case.context, request)
                assert rejected.value.reason == "revision_reservation_already_used"
            finish.set()
            if outcome == "save":
                result = await task
                assert result["revision"]["review_status"] == "unreviewed"
                assert lease.future_scratch_finished and lease.cost.memory_bytes == 0
            else:
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert not case.assets and not case.writes
            with pytest.raises(ReviewOverload) as rejected:
                await service.revision(case.context, request)
            assert rejected.value.reason == "revision_reservation_already_used"
            assert len(case.reads) == reads and lease.native_revision_owner is claim
        finally:
            finish.set()
            current_review_lease.reset(token)
            if not task.done():
                await asyncio.gather(task, return_exceptions=True)
            await lease.release()
            await service.close()


@pytest.mark.asyncio
async def test_adapter_reads_only_exact_committed_native_diagnostic_refs(tmp_path, monkeypatch):
    from copy import deepcopy
    from unittest.mock import AsyncMock

    from transformation_portal.portal import photographic_review_access as adapter

    data = b'{"diagnostic_only":true}'
    product = {
        "filename": "materials-diagnostics.json",
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "content_type": "application/json",
        "kind": "review_asset",
        "storage_path": "photographic-review/committed/materials-diagnostics.json",
    }
    state = {"revisions": [{"native_products": [product]}], "deliveries": []}
    review_store = SimpleNamespace(get=AsyncMock(return_value=state))
    monkeypatch.setattr(adapter, "get_review_store", lambda: review_store)

    async def chunks(value):
        yield value

    artifact_store = SimpleNamespace(open_bytes=AsyncMock(side_effect=lambda *_: chunks(data)))
    app = SimpleNamespace(
        _uses_dispatch_authority=lambda: False,
        _artifact_storage_job_id=lambda job: job.id,
        _artifact_store=lambda: artifact_store,
    )
    access = adapter.AppReviewAccess(app)
    job = SimpleNamespace(id="job_native")
    access._job = AsyncMock(return_value=(job, None))
    destination = tmp_path / "verified.json"
    observed = await access._materialize_verified(None, job, "tenant_a", product, destination, 4096)
    assert destination.read_bytes() == data and observed["sha256"] == product["sha256"]
    review_store.get.assert_awaited_with(job.id, "tenant_a")
    artifact_store.open_bytes.assert_awaited_once_with(job.id, product["storage_path"])
    forged = {**product, "storage_path": "photographic-review/another-tenant/materials-diagnostics.json"}
    with pytest.raises(ReviewAccessError, match="saved review state"):
        await access._materialize_verified(None, job, "tenant_a", forged, tmp_path / "forged", 4096)
    artifact_store.open_bytes.assert_awaited_once()
    corrupted = deepcopy(state)
    corrupted["revisions"][0]["native_products"][0]["sha256"] = "0" * 64
    review_store.get.return_value = corrupted
    changed = corrupted["revisions"][0]["native_products"][0]
    with pytest.raises(ReviewAccessError, match="changed"):
        await access._materialize_verified(None, job, "tenant_a", changed, tmp_path / "changed", 4096)
    assert not (tmp_path / "changed").exists()
