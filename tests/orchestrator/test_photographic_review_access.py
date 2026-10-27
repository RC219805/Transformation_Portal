"""App-adapter authorization and verified artifact materialization contracts."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import threading
import tracemalloc
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse

from tests.lux_depth_v5 import test_pipeline as controlled_pipeline
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.orchestrator.photographic_review_access import ReviewAccessError, ReviewContext
from transformation_portal.portal import photographic_review_access as adapter

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]
request_case = controlled_pipeline.request_case


def _request():
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/jobs/job/review",
            "query_string": b"",
            "headers": [(b"x-tp-actor", b"forged-human"), (b"x-tp-artifact-delivery", b"redirect")],
        }
    )


async def _chunks(data):
    yield data[:3]
    yield data[3:]


def _reference(data=b"photograph"):
    return {
        "artifact_path": "v6/input-0000/delivery.tif",
        "sha256": hashlib.sha256(data).hexdigest(),
        "source_identity_sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "scene_id": "input-0000",
        "role": "final",
        "source_name": "delivery.tif",
        "label": "input-0000",
        "input_color": "auto",
        "kind": "indexed",
        "identity_status": "retained_artifact_identity",
        "original_unavailable_reason": "immutable_original_binding_unavailable",
    }


async def test_history_authorizes_same_job_scope_without_discovering_or_reading_sources(fake_app):
    app, job, store, _ = fake_app
    access = adapter.AppReviewAccess(app)
    access._references = AsyncMock(side_effect=AssertionError("History must not discover image references"))
    access._binding_identity = AsyncMock(side_effect=AssertionError("History must not require an artifact inventory"))
    app._artifact_store = Mock(side_effect=AssertionError("History must not require photographic storage"))
    context = await access.authorize_history(_request(), job.id)
    assert isinstance(context, ReviewContext) and context.references == []
    assert context.actor["username"] == "api_key_operator"
    assert context.binding_identity is None and context.publication_root is None
    await context.revalidate()
    access._references.assert_not_awaited()
    access._binding_identity.assert_not_awaited()
    app._artifact_store.assert_not_called()
    app._get_job_artifact.assert_not_awaited()
    store.open_bytes.assert_not_awaited()
    app._has_valid_api_key = lambda _: False
    denied = await access.authorize_history(_request(), job.id)
    assert denied.status_code == 401


@pytest.fixture
def fake_app(tmp_path):
    data = b"photograph"
    reference = _reference(data)
    job = SimpleNamespace(
        id="job",
        state="succeeded",
        artifacts={
            "items": [
                {"path": reference["artifact_path"], "sha256": reference["sha256"], "size_bytes": reference["size_bytes"]},
                {"path": "v6/input-0000/preview.png", "sha256": "a" * 64, "size_bytes": 10},
                {"path": "v6/input-0000/depth.tif", "sha256": "b" * 64, "size_bytes": 10},
            ]
        },
    )
    store = SimpleNamespace(
        open_bytes=AsyncMock(side_effect=lambda *_: _chunks(data)),
        write_immutable_file=AsyncMock(),
        delete=AsyncMock(return_value=1),
    )

    def error(status, **kwargs):
        return JSONResponse({"success": False, "error": kwargs}, status_code=status)

    app = SimpleNamespace(
        _error_response=error,
        _cleanup_expired_jobs=AsyncMock(),
        _now=lambda: 100.0,
        _pilot_tenant_from_request=AsyncMock(return_value=(None, None)),
        _load_job_view=AsyncMock(return_value=job),
        _pilot_enforce_job_tenant=AsyncMock(return_value=None),
        _artifacts_deleted=lambda _: False,
        _job_tenant_id=lambda _: None,
        API_KEY_SECRET="test-backend-secret",
        _has_valid_api_key=lambda _: True,
        PILOT_CONTROL_PLANE_ENABLED=False,
        LOGGER=SimpleNamespace(exception=Mock()),
        _uses_dispatch_authority=lambda: False,
        _artifact_storage_job_id=lambda value: value.id,
        _artifact_store=lambda: store,
        _get_job_artifact=AsyncMock(side_effect=lambda *_args, **_kwargs: StreamingResponse(_chunks(data))),
        _job_output_dir=lambda _: tmp_path,
    )
    return app, job, store, data


@pytest.mark.parametrize("condition,status", [("missing-config", 503), ("invalid-key", 401)])
async def test_review_requires_auth_even_when_legacy_local_job_routes_are_open(fake_app, condition, status):
    app, _, _, _ = fake_app
    if condition == "missing-config":
        app.API_KEY_SECRET = ""
    else:
        app._has_valid_api_key = lambda _: False
    response = await adapter.AppReviewAccess(app).authorize(_request(), "job")
    assert response.status_code == status
    app._load_job_view.assert_not_awaited()


async def test_sources_only_include_verified_final_photographs_and_explicit_missing_original_reason(fake_app):
    app, _, _, _ = fake_app
    context = await adapter.AppReviewAccess(app).authorize(_request(), "job")

    assert isinstance(context, ReviewContext)
    assert context.references == [_reference()]
    assert context.actor == {"username": "api_key_operator", "role": "operator", "assurance": "backend_api_key"}
    assert context.tenant_id == "local"
    app._cleanup_expired_jobs.assert_awaited_once()


async def test_authorized_source_name_retains_format_required_by_delivery_image(fake_app):
    from transformation_portal.photographic_delivery import DeliveryImage

    app, _, _, data = fake_app
    context = await adapter.AppReviewAccess(app).authorize(_request(), "job")
    source = context.references[0]
    image = DeliveryImage(
        source_name=source["source_name"],
        source_sha256=source["sha256"],
        source_bytes=data,
        scene=source["scene_id"],
        input_color=source["input_color"],
    )
    assert image.source_name == "delivery.tif"
    assert source["label"] == "input-0000"


async def test_unified_v5_evidence_is_not_offered_as_a_previous_revision(fake_app):
    app, job, _, _ = fake_app
    job.artifacts["items"].append({**job.artifacts["items"][0], "path": "source-v5/input-0000/enhanced.tif"})
    context = await adapter.AppReviewAccess(app).authorize(_request(), "job")
    assert [ref["artifact_path"] for ref in context.references] == ["v6/input-0000/delivery.tif"]


async def test_ambiguous_duplicate_finals_require_explicit_source_contract(fake_app):
    app, job, _, _ = fake_app
    job.artifacts["items"].append({**job.artifacts["items"][0], "path": "v6/input-0000/enhanced.tif"})
    response = await adapter.AppReviewAccess(app).authorize(_request(), "job")
    assert response.status_code == 409


@pytest.mark.parametrize("condition,status", [("missing", 404), ("active", 409), ("deleted", 410), ("tenant", 404)])
async def test_auth_lifecycle_and_tenant_failures_precede_artifact_access(fake_app, condition, status):
    app, job, store, _ = fake_app
    if condition == "missing":
        app._load_job_view.return_value = None
    elif condition == "active":
        job.state = "running"
    elif condition == "deleted":
        app._artifacts_deleted = lambda _: True
    else:
        app._pilot_enforce_job_tenant.return_value = JSONResponse({}, status_code=404)

    response = await adapter.AppReviewAccess(app).authorize(_request(), "job")

    assert isinstance(response, Response)
    assert response.status_code == status
    app._get_job_artifact.assert_not_awaited()
    store.open_bytes.assert_not_awaited()


async def test_configuration_failure_is_typed_and_does_not_expose_paths_or_credentials(fake_app):
    app, _, _, _ = fake_app
    app._load_job_view.side_effect = RuntimeError("/private/path AWS_SECRET_DO_NOT_EXPOSE")

    response = await adapter.AppReviewAccess(app).authorize(_request(), "job")

    assert response.status_code == 503
    assert b"REVIEW_UNAVAILABLE" in response.body
    assert b"private/path" not in response.body
    assert b"AWS_SECRET" not in response.body


async def test_indexed_materialization_reuses_authenticated_stream_and_verifies_complete_bytes(fake_app, tmp_path):
    app, _, _, data = fake_app
    request = _request()
    context = await adapter.AppReviewAccess(app).authorize(request, "job")
    output = tmp_path / "snapshot.tif"

    identity = await context.materialize(context.references[0], output, 1024)

    assert output.read_bytes() == data
    assert identity == {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
    forwarded = app._get_job_artifact.await_args.kwargs["request"]
    assert forwarded.headers["x-tp-artifact-delivery"] == "stream"
    assert request.headers["x-tp-artifact-delivery"] == "redirect"


@pytest.mark.parametrize("condition", ["changed", "truncated", "overflow", "http-error"])
async def test_incomplete_or_changed_artifact_never_leaves_materialized_output(fake_app, tmp_path, condition):
    app, _, _, data = fake_app
    context = await adapter.AppReviewAccess(app).authorize(_request(), "job")
    if condition == "http-error":
        app._get_job_artifact.side_effect = None
        app._get_job_artifact.return_value = JSONResponse({}, status_code=503)
    else:
        changed = {"changed": data.upper(), "truncated": data[:-1], "overflow": data + b"more"}[condition]
        app._get_job_artifact.side_effect = lambda *_args, **_kwargs: StreamingResponse(_chunks(changed))
    output = tmp_path / "snapshot.tif"

    with pytest.raises(ReviewAccessError):
        await context.materialize(context.references[0], output, 1024)

    assert not output.exists()


@pytest.mark.parametrize("invalid", ["unexpected text", memoryview(b"unexpected buffer")])
async def test_binary_snapshot_rejects_broader_response_chunks_and_closes_stream(fake_app, tmp_path, invalid):
    app, _, _, data = fake_app
    closed = asyncio.Event()

    async def invalid_stream():
        try:
            yield data[:3]
            yield invalid
        finally:
            closed.set()

    app._get_job_artifact.side_effect = lambda *_args, **_kwargs: StreamingResponse(invalid_stream())
    context = await adapter.AppReviewAccess(app).authorize(_request(), "job")
    output = tmp_path / "invalid.tif"
    with pytest.raises(ReviewAccessError, match="Invalid artifact stream"):
        await context.materialize(context.references[0], output, 1024)
    assert closed.is_set()
    assert not output.exists()


async def test_materialization_does_not_overwrite_existing_destination(fake_app, tmp_path):
    app, _, _, _ = fake_app
    context = await adapter.AppReviewAccess(app).authorize(_request(), "job")
    output = tmp_path / "snapshot.tif"
    output.write_bytes(b"keep existing")

    with pytest.raises(ReviewAccessError, match="unavailable"):
        await context.materialize(context.references[0], output, 1024)

    assert output.read_bytes() == b"keep existing"


async def test_recorded_size_budget_is_enforced_before_reading(fake_app, tmp_path):
    app, _, _, _ = fake_app
    context = await adapter.AppReviewAccess(app).authorize(_request(), "job")
    with pytest.raises(ReviewAccessError, match="budget"):
        await context.materialize(context.references[0], tmp_path / "snapshot.tif", 1)
    app._get_job_artifact.assert_not_awaited()


async def test_legacy_missing_fingerprint_observes_complete_authorized_stream_without_native_acceptance(fake_app, tmp_path):
    app, job, _, _ = fake_app
    # The ordinary legacy index skips full hashing above its 16 MiB budget.
    content = b"p" * (18 * 1024**2)
    job.artifacts["items"][0]["sha256"] = None
    job.artifacts["items"][0]["size_bytes"] = len(content)
    job.artifacts["items"][0]["fingerprint_status"] = "skipped_size"
    app._get_job_artifact.side_effect = lambda *_args, **_kwargs: StreamingResponse(_chunks(content))

    context = await adapter.AppReviewAccess(app).authorize(_request(), "job")

    assert isinstance(context, ReviewContext)
    ref = context.references[0]
    assert ref["sha256"] == hashlib.sha256(content).hexdigest()
    assert ref["size_bytes"] == len(content)
    assert ref["identity_status"] == "fresh_snapshot_not_native_acceptance"
    assert job.artifacts["items"][0]["sha256"] is None
    assert app._get_job_artifact.await_args.kwargs["request"].headers["x-tp-artifact-delivery"] == "stream"
    await context.materialize(ref, tmp_path / "snapshot.tif", len(content))
    assert (tmp_path / "snapshot.tif").stat().st_size == len(content)


async def test_changed_legacy_bytes_cannot_inherit_an_old_hash_bound_decision(fake_app, tmp_path):
    from transformation_portal.orchestrator.photographic_review_service import resolve
    from transformation_portal.orchestrator.photographic_review_store import empty_state

    app, job, _, data = fake_app
    job.artifacts["items"][0]["sha256"] = None
    access = adapter.AppReviewAccess(app)
    first = await access.authorize(_request(), "job")
    old_ref = first.references[0]
    app._get_job_artifact.side_effect = lambda *_args, **_kwargs: StreamingResponse(_chunks(data.upper()))

    current = await access.authorize(_request(), "job")

    assert current.references[0]["sha256"] != old_ref["sha256"]
    with pytest.raises(ReviewAccessError, match="changed"):
        resolve(current, empty_state(), old_ref["artifact_path"], old_ref["sha256"])
    with pytest.raises(ReviewAccessError, match="changed"):
        await first.materialize(old_ref, tmp_path / "old.tif", 1024)
    assert not (tmp_path / "old.tif").exists()


@pytest.mark.parametrize("condition", ["truncated", "oversized", "missing-size", "too-large", "denied", "bad-sha"])
async def test_fresh_legacy_identity_never_bypasses_stream_length_auth_or_budget(fake_app, condition):
    app, job, _, data = fake_app
    item = job.artifacts["items"][0]
    item["sha256"] = None
    if condition in {"truncated", "oversized"}:
        content = data[:-1] if condition == "truncated" else data + b"extra"
        app._get_job_artifact.side_effect = lambda *_args, **_kwargs: StreamingResponse(_chunks(content))
    elif condition == "missing-size":
        item["size_bytes"] = None
    elif condition == "too-large":
        item["size_bytes"] = adapter.MAX_REVIEW_IMAGE_BYTES + 1
    elif condition == "bad-sha":
        item["sha256"] = "malformed-retained-hash"
    else:
        app._get_job_artifact.side_effect = None
        app._get_job_artifact.return_value = JSONResponse({}, status_code=404)

    response = await adapter.AppReviewAccess(app).authorize(_request(), "job")

    assert isinstance(response, Response)
    assert response.status_code in {404, 409, 413}
    if condition in {"missing-size", "too-large", "bad-sha"}:
        app._get_job_artifact.assert_not_awaited()


async def test_legacy_snapshot_cannot_discover_unindexed_files(fake_app):
    app, job, _, _ = fake_app
    job.artifacts["items"] = []
    context = await adapter.AppReviewAccess(app).authorize(_request(), "job")
    assert context.references == []
    app._get_job_artifact.assert_not_awaited()


async def test_zip_publication_has_a_separate_bounded_budget_without_changing_generation_policy(
    fake_app, tmp_path, monkeypatch
):
    from transformation_portal.orchestrator.artifact_store import generation

    app, _, store, data = fake_app
    source = tmp_path / "delivery.zip"
    source.write_bytes(data)
    context = await adapter.AppReviewAccess(app).authorize(_request(), "job")
    original_generation_limit = generation.MAX_GENERATION_FILE_BYTES
    monkeypatch.setattr(adapter, "MAX_REVIEW_IMAGE_BYTES", 1)

    await context.publish("photographic-review/new/delivery.zip", source, "application/zip")

    store.open_bytes.assert_awaited_once()
    assert adapter.MAX_REVIEW_ZIP_BYTES == 16 * 1024**3
    assert generation.MAX_GENERATION_FILE_BYTES == original_generation_limit
    with pytest.raises(ReviewAccessError):
        await context.publish("photographic-review/new/not-a-zip.tif", source, "image/tiff")


async def test_zip_publication_enforces_zip_budget(fake_app, tmp_path, monkeypatch):
    app, _, store, data = fake_app
    source = tmp_path / "delivery.zip"
    source.write_bytes(data)
    context = await adapter.AppReviewAccess(app).authorize(_request(), "job")
    monkeypatch.setattr(adapter, "MAX_REVIEW_ZIP_BYTES", len(data) - 1)
    with pytest.raises(ReviewAccessError, match="bounded regular"):
        await context.publish("photographic-review/new/delivery.zip", source, "application/zip")
    store.write_immutable_file.assert_not_awaited()


async def test_file_response_fallback_uses_confined_snapshot(fake_app, tmp_path):
    app, _, _, data = fake_app
    source = tmp_path / "source.tif"
    source.write_bytes(data)
    app._validate_resolved_job_artifact_path = Mock(return_value=(tmp_path, source, "source.tif"))
    app._get_job_artifact.side_effect = None
    app._get_job_artifact.return_value = FileResponse(source)
    context = await adapter.AppReviewAccess(app).authorize(_request(), "job")
    output = tmp_path / "snapshot.tif"

    await context.materialize(context.references[0], output, 1024)

    assert output.read_bytes() == data
    app._validate_resolved_job_artifact_path.assert_called_once()


@pytest.mark.parametrize("reported_large_file", [False, True], ids=["complete", "512mib-truncated-header"])
async def test_legacy_file_response_identity_keeps_download_memory_bounded(
    fake_app, tmp_path, monkeypatch, reported_large_file
):
    from transformation_portal.orchestrator.photographic_admission import (
        PhotographicAdmissionController,
        ReviewAdmissionConfig,
        ReviewCost,
    )

    app, job, _, _ = fake_app
    content = b"p" * (8 * 1024**2 + 17)
    source = tmp_path / "legacy.tif"
    source.write_bytes(content)
    indexed_size = adapter.MAX_REVIEW_IMAGE_BYTES if reported_large_file else len(content)
    job.artifacts["items"][0].update(sha256=None, size_bytes=indexed_size, fingerprint_status="skipped_size")
    app._validate_resolved_job_artifact_path = Mock(return_value=(tmp_path, source, source.name))
    app._get_job_artifact.side_effect = None
    app._get_job_artifact.return_value = FileResponse(source)
    original_stat, original_read = adapter.os.fstat, adapter.os.read
    identity = source.stat()
    reads = []

    def file_stat(descriptor):
        observed = original_stat(descriptor)
        if reported_large_file and (observed.st_dev, observed.st_ino) == (identity.st_dev, identity.st_ino):
            fields = {name: getattr(observed, name) for name in dir(observed) if name.startswith("st_")}
            fields["st_size"] = indexed_size
            return SimpleNamespace(**fields)
        return observed

    def file_read(descriptor, maximum):
        observed = original_stat(descriptor)
        if (observed.st_dev, observed.st_ino) == (identity.st_dev, identity.st_ino):
            reads.append(maximum)
        return original_read(descriptor, maximum)

    monkeypatch.setattr(adapter.os, "fstat", file_stat)
    monkeypatch.setattr(adapter.os, "read", file_read)
    controller = PhotographicAdmissionController(ReviewAdmissionConfig(enabled=True))
    lease = await controller.acquire("local", ReviewCost(256 * 1024**2, 32 * 1024**2, decode_slots=0, interactive=False))
    token = adapter.current_review_lease.set(lease)
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    try:
        result = await adapter.AppReviewAccess(app).authorize(_request(), job.id)
        _, peak = tracemalloc.get_traced_memory()
        assert peak - before < 6 * 1024**2, "Legacy identity must not retain the complete artifact in a transfer lease"
        assert reads and max(reads) <= 1024**2
        if reported_large_file:
            assert isinstance(result, Response) and result.status_code == 503
        else:
            assert isinstance(result, ReviewContext)
            assert result.references[0]["sha256"] == hashlib.sha256(content).hexdigest()
            assert result.references[0]["size_bytes"] == len(content)
        assert source.read_bytes() == content
        assert job.artifacts["items"][0]["sha256"] is None
    finally:
        if not was_tracing:
            tracemalloc.stop()
        adapter.current_review_lease.reset(token)
        await lease.release()
        await controller.close()


@pytest.mark.parametrize("mutation", ["replace", "symlink", "grow"])
async def test_file_response_snapshot_rejects_source_change_before_using_private_bytes(
    fake_app, tmp_path, monkeypatch, mutation
):
    app, _, _, data = fake_app
    source = tmp_path / "source.tif"
    source.write_bytes(data)
    app._validate_resolved_job_artifact_path = Mock(return_value=(tmp_path, source, source.name))
    app._get_job_artifact.side_effect = None
    app._get_job_artifact.return_value = FileResponse(source)
    context = await adapter.AppReviewAccess(app).authorize(_request(), "job")
    original_read = adapter.os.read
    changed = False

    def mutate_after_read(descriptor, maximum):
        nonlocal changed
        chunk = original_read(descriptor, maximum)
        if chunk and not changed:
            changed = True
            if mutation == "grow":
                with source.open("ab") as handle:
                    handle.write(b"!")
            else:
                retained = source.with_name("retained.tif")
                source.rename(retained)
                if mutation == "symlink":
                    source.symlink_to(retained)
                else:
                    source.write_bytes(data)
        return chunk

    monkeypatch.setattr(adapter.os, "read", mutate_after_read)
    destination = tmp_path / "private.tif"
    with pytest.raises(ReviewAccessError):
        await context.materialize(context.references[0], destination, 1024)
    assert changed and not destination.exists()


async def test_original_sha_failure_drops_nested_snapshot_bytes_before_final_lease_release(fake_app, tmp_path, monkeypatch):
    from transformation_portal.orchestrator import photographic_admission as admission
    from transformation_portal.orchestrator import photographic_review_service as service_module

    app, job, _, _ = fake_app
    source = tmp_path / "original.tif"
    source.write_bytes(b"x" * (8 * 1024**2))
    ref = {
        **_reference(),
        "kind": "original",
        "artifact_path": source.name,
        "source_name": source.name,
        "sha256": "a" * 64,
        "source_identity_sha256": "a" * 64,
        "size_bytes": source.stat().st_size,
        "private_original_root": str(tmp_path),
        "private_original_path": source.name,
    }
    access = adapter.AppReviewAccess(app)
    monkeypatch.setattr(access, "_originals", AsyncMock(return_value=([ref], None)))
    failures = []
    stream = adapter._stream_snapshot

    async def capture_failure(*args):
        try:
            return await stream(*args)
        except BaseException as error:
            failures.append(error)
            raise

    monkeypatch.setattr(adapter, "_stream_snapshot", capture_failure)

    async def materialize(selected, destination, maximum):
        return await access._materialize(_request(), job, "local", selected, destination, maximum)

    context = ReviewContext(
        job_id=job.id,
        tenant_id="local",
        actor={"username": "operator"},
        references=[ref],
        materialize=materialize,
        publish=AsyncMock(),
        revalidate=AsyncMock(),
    )
    service = service_module.PhotographicReviewService(admission.ReviewAdmissionConfig(enabled=True))
    monkeypatch.setattr(
        service,
        "state",
        AsyncMock(
            return_value={"state_version": 0, "decisions": [{**ref, "inspection": {"width": 8, "height": 6, "alpha": False}}]}
        ),
    )
    monkeypatch.setattr(admission.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * admission.GIB))
    retained = []
    release = admission.ReviewLease.release

    async def inspect_release(lease):
        if failures:
            trace = failures[-1].__traceback__
            while trace is not None:
                frame = trace.tb_frame
                if frame.f_code.co_name in {"_materialize_verified", "_stream_snapshot"}:
                    for name in ("data", "chunk"):
                        value = frame.f_locals.get(name)
                        if isinstance(value, bytes) and len(value) >= source.stat().st_size:
                            retained.append((frame.f_code.co_name, name, len(value)))
                trace = trace.tb_next
        await release(lease)

    monkeypatch.setattr(admission.ReviewLease, "release", inspect_release)

    async def render(case, *_args):
        await case.materialize(ref, tmp_path / "private.tif", adapter.MAX_REVIEW_IMAGE_BYTES)

    try:
        with pytest.raises(ReviewAccessError, match="changed"):
            await service.execute(context, render, ref["artifact_path"], ref["sha256"])
        assert failures and not retained, f"nested adapter frames retained encoded buffers at final release: {retained}"
        assert not (tmp_path / "private.tif").exists()
    finally:
        await service.close()


@pytest.mark.parametrize("worker_failure", [False, True], ids=["discarded-success", "failed-worker"])
async def test_cancelled_original_snapshot_drops_completed_worker_buffers_before_release(
    fake_app, tmp_path, monkeypatch, worker_failure
):
    import numpy as np

    from transformation_portal.orchestrator import photographic_admission as admission

    app, job, _, _ = fake_app
    ref = {**_reference(), "kind": "original", "private_original_root": str(tmp_path), "private_original_path": "original.tif"}
    access = adapter.AppReviewAccess(app)
    monkeypatch.setattr(access, "_originals", AsyncMock(return_value=([ref], None)))
    started, finish = threading.Event(), threading.Event()
    buffers, failures = [], []

    def blocked_snapshot(*_args, **_kwargs):
        values = np.ones(1024 * 1024, dtype=np.float64)
        buffers.append(weakref.ref(values))
        started.set()
        assert finish.wait(timeout=5), "controlled snapshot worker was not released"
        if worker_failure:
            error = RuntimeError("original snapshot failed after cancellation")
            failures.append(error)
            raise error
        return values, {}

    monkeypatch.setattr(adapter, "snapshot", blocked_snapshot)
    controller = admission.PhotographicAdmissionController(admission.ReviewAdmissionConfig(enabled=True))
    lease = await controller.acquire("local", admission.ReviewCost(256 * admission.MIB, 32 * admission.MIB, interactive=False))
    token = adapter.current_review_lease.set(lease)
    task = asyncio.create_task(access._materialize_verified(_request(), job, "local", ref, tmp_path / "private.tif", 1024))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert lease.cost.memory_bytes == 256 * admission.MIB
        assert buffers and all(
            buffer() is None for buffer in buffers
        ), "quiescent owned snapshot worker retained decoded buffers"
        assert not (tmp_path / "private.tif").exists()
        if worker_failure:
            assert type(failures[0]) is RuntimeError and str(failures[0]) == "original snapshot failed after cancellation"
            trace = failures[0].__traceback__
            names = []
            while trace is not None:
                names.append(trace.tb_frame.f_code.co_name)
                trace = trace.tb_next
            assert "blocked_snapshot" in names
    finally:
        finish.set()
        await asyncio.gather(task, return_exceptions=True)
        adapter.current_review_lease.reset(token)
        await lease.release()
        await controller.close()


async def test_review_derivative_requires_exact_saved_server_reference(fake_app, monkeypatch, tmp_path):
    app, _, store, data = fake_app
    source = {**_reference(data), "kind": "review_asset", "storage_path": "photographic-review/rev-one/image.tif"}
    state = {"revisions": [{"variant": source}]}
    monkeypatch.setattr(adapter, "get_review_store", lambda: SimpleNamespace(get=AsyncMock(return_value=state)))
    context = await adapter.AppReviewAccess(app).authorize(_request(), "job")

    await context.materialize(source, tmp_path / "snapshot.tif", 1024)
    store.open_bytes.assert_awaited_once_with("job", "photographic-review/rev-one/image.tif")
    forged = {**source, "storage_path": "photographic-review/other-tenant/image.tif"}
    with pytest.raises(ReviewAccessError, match="saved review state"):
        await context.materialize(forged, tmp_path / "forged.tif", 1024)
    assert store.open_bytes.await_count == 1


async def test_saved_revision_reopens_through_real_adapter_with_receipt_derived_presentation(fake_app, monkeypatch, tmp_path):
    from transformation_portal.orchestrator.photographic_review_service import _display, load_image, revision_refs
    from transformation_portal.photographic_delivery import DeliveryImage, RevisionRecipe, create_revision

    app, _, store, _ = fake_app
    bodies, _, _ = _native_inventory(False, False)
    raw = bodies["v6/input-0000/delivery.tif"]
    result = create_revision(
        DeliveryImage("source.tif", hashlib.sha256(raw).hexdigest(), "Room", source_bytes=raw), RevisionRecipe()
    )
    source = {
        **_reference(result["source_bytes"]),
        "kind": "review_asset",
        "storage_path": "photographic-review/rev-one/master.tif",
        "source_name": "master.tif",
    }
    state = {"revisions": [{"variant": source, "receipt": result["receipt"]}], "deliveries": []}
    monkeypatch.setattr(adapter, "get_review_store", lambda: SimpleNamespace(get=AsyncMock(return_value=state)))
    store.open_bytes.side_effect = lambda *_: _chunks(result["source_bytes"])
    context = await adapter.AppReviewAccess(app).authorize(_request(), "job")
    ref = revision_refs(state)[0]
    assert "presentation_policy" in ref and "presentation_policy" not in source
    reopened = await load_image(context, ref)
    assert _display(reopened, view="recorded")[0]
    count = store.open_bytes.await_count
    for changed in ({"storage_path": "photographic-review/forged/master.tif"}, {"sha256": "0" * 64}):
        with pytest.raises(ReviewAccessError, match="saved review state"):
            await context.materialize({**ref, **changed}, tmp_path / "forged.tif", 1024**2)
    with pytest.raises(ReviewAccessError, match="committed receipt"):
        await context.materialize({**ref, "presentation_policy_sha256": "0" * 64}, tmp_path / "forged.tif", 1024**2)
    assert store.open_bytes.await_count == count


async def test_arbitrary_nested_review_state_cannot_claim_a_derivative(fake_app, monkeypatch, tmp_path):
    app, _, store, data = fake_app
    source = {**_reference(data), "kind": "review_asset", "storage_path": "photographic-review/forged/image.tif"}
    state = {"decisions": [{"notes": {"forged": source}}], "revisions": [], "deliveries": []}
    monkeypatch.setattr(adapter, "get_review_store", lambda: SimpleNamespace(get=AsyncMock(return_value=state)))
    context = await adapter.AppReviewAccess(app).authorize(_request(), "job")

    with pytest.raises(ReviewAccessError, match="saved review state"):
        await context.materialize(source, tmp_path / "refused.tif", 1024)

    store.open_bytes.assert_not_awaited()


@pytest.mark.parametrize(
    "storage_path", ["other/image.tif", "photographic-review/../escape.tif", "/photographic-review/image.tif"]
)
async def test_publication_cannot_escape_owned_namespace(fake_app, tmp_path, storage_path):
    app, _, store, _ = fake_app
    context = await adapter.AppReviewAccess(app).authorize(_request(), "job")
    with pytest.raises(ReviewAccessError):
        await context.publish(storage_path, tmp_path / "anything.tif", "image/tiff")
    store.write_immutable_file.assert_not_awaited()


async def test_parent_deletion_after_authorization_revokes_materialization_and_publication(fake_app, tmp_path):
    app, _, store, _ = fake_app
    context = await adapter.AppReviewAccess(app).authorize(_request(), "job")
    app._artifacts_deleted = lambda _: True
    with pytest.raises(ReviewAccessError):
        await context.materialize(context.references[0], tmp_path / "snapshot.tif", 1024)
    with pytest.raises(ReviewAccessError):
        await context.publish("photographic-review/new/image.tif", tmp_path / "snapshot.tif", "image/tiff")
    store.write_immutable_file.assert_not_awaited()


async def test_parent_revocation_during_stream_rejects_the_completed_snapshot(fake_app, tmp_path):
    app, _, _, data = fake_app
    context = await adapter.AppReviewAccess(app).authorize(_request(), "job")

    async def revoked_stream():
        yield data
        app._artifacts_deleted = lambda _: True

    app._get_job_artifact.side_effect = lambda *_args, **_kwargs: StreamingResponse(revoked_stream())
    output = tmp_path / "revoked.tif"
    with pytest.raises(ReviewAccessError, match="became unavailable") as exc:
        await context.materialize(context.references[0], output, 1024)
    assert exc.value.status == 410
    assert not output.exists()


async def test_publication_reads_back_complete_store_bytes_before_success(fake_app, tmp_path):
    app, _, store, data = fake_app
    source = tmp_path / "published.tif"
    source.write_bytes(data)
    context = await adapter.AppReviewAccess(app).authorize(_request(), "job")

    await context.publish("photographic-review/new/image.tif", source, "image/tiff")

    store.write_immutable_file.assert_awaited_once()
    store.open_bytes.assert_awaited_once_with("job", "photographic-review/new/image.tif")
    staged = store.write_immutable_file.await_args.args[2]
    assert staged != source
    assert not staged.exists()
    assert source.read_bytes() == data


@pytest.mark.parametrize("condition", ["corrupt", "truncated", "oversized", "read-error"])
async def test_publication_rejects_incomplete_or_corrupt_readback(fake_app, tmp_path, condition):
    app, _, store, data = fake_app
    source = tmp_path / "published.tif"
    source.write_bytes(data)
    context = await adapter.AppReviewAccess(app).authorize(_request(), "job")
    if condition == "read-error":
        store.open_bytes.side_effect = OSError("private object key and internal credentials")
    else:
        content = {"corrupt": data.upper(), "truncated": data[:-1], "oversized": data + b"extra"}[condition]
        store.open_bytes.side_effect = lambda *_: _chunks(content)

    with pytest.raises(ReviewAccessError, match="publication unavailable") as exc:
        await context.publish("photographic-review/new/image.tif", source, "image/tiff")

    assert exc.value.status == 503
    assert "credentials" not in str(exc.value)
    assert not store.write_immutable_file.await_args.args[2].exists()
    store.delete.assert_awaited_once_with("job", "photographic-review/new/image.tif")


async def test_publication_rechecks_parent_visibility_after_store_readback(fake_app, tmp_path):
    app, _, store, data = fake_app
    source = tmp_path / "published.tif"
    source.write_bytes(data)
    context = await adapter.AppReviewAccess(app).authorize(_request(), "job")

    async def readback():
        yield data
        app._artifacts_deleted = lambda _: True

    store.open_bytes.side_effect = lambda *_: readback()
    with pytest.raises(ReviewAccessError, match="became unavailable") as exc:
        await context.publish("photographic-review/new/image.tif", source, "image/tiff")
    assert exc.value.status == 410
    store.delete.assert_awaited_once_with("job", "photographic-review/new/image.tif")


async def test_failed_immutable_collision_never_deletes_an_existing_object(fake_app, tmp_path):
    app, _, store, data = fake_app
    source = tmp_path / "published.tif"
    source.write_bytes(data)
    store.write_immutable_file.side_effect = RuntimeError("immutable key exists")
    context = await adapter.AppReviewAccess(app).authorize(_request(), "job")
    with pytest.raises(ReviewAccessError):
        await context.publish("photographic-review/new/image.tif", source, "image/tiff")
    store.delete.assert_not_awaited()


async def test_context_rollback_deletes_only_own_unreferenced_publications(fake_app, tmp_path, monkeypatch):
    app, _, store, data = fake_app
    state_store = SimpleNamespace(get=AsyncMock(return_value={"revisions": [], "deliveries": []}))
    monkeypatch.setattr(adapter, "get_review_store", lambda: state_store)
    source = tmp_path / "published.tif"
    source.write_bytes(data)
    context = await adapter.AppReviewAccess(app).authorize(_request(), "job")
    path = "photographic-review/request-a/image.tif"
    await context.publish(path, source, "image/tiff")

    await context.discard([path])

    store.delete.assert_awaited_once_with("job", path)
    with pytest.raises(ReviewAccessError, match="unowned"):
        await context.discard(["photographic-review/request-b/image.tif"])
    assert store.delete.await_count == 1


async def test_uncertain_commit_or_saved_reference_prevents_rollback_deletion(fake_app, tmp_path, monkeypatch):
    app, _, store, data = fake_app
    state_store = SimpleNamespace(get=AsyncMock())
    monkeypatch.setattr(adapter, "get_review_store", lambda: state_store)
    source = tmp_path / "published.tif"
    source.write_bytes(data)
    context = await adapter.AppReviewAccess(app).authorize(_request(), "job")
    path = "photographic-review/request-a/image.tif"
    await context.publish(path, source, "image/tiff")
    state_store.get.return_value = {"revisions": [{"variant": {"kind": "review_asset", "storage_path": path}}]}
    await context.discard([path])
    store.delete.assert_not_awaited()
    state_store.get.side_effect = RuntimeError("state unavailable after uncertain commit")
    with pytest.raises(RuntimeError, match="uncertain commit"):
        await context.discard([path])
    store.delete.assert_not_awaited()


async def test_review_publication_and_rollback_share_legacy_parent_tenant_namespace(fake_app, tmp_path, monkeypatch):
    app, _, store, data = fake_app
    app._artifact_storage_job_id = lambda job: "tenant-a__" + job.id
    state_store = SimpleNamespace(get=AsyncMock(return_value={"revisions": [], "deliveries": []}))
    monkeypatch.setattr(adapter, "get_review_store", lambda: state_store)
    source = tmp_path / "published.tif"
    source.write_bytes(data)
    context = await adapter.AppReviewAccess(app).authorize(_request(), "job")
    path = "photographic-review/request-a/image.tif"

    await context.publish(path, source, "image/tiff")
    await context.discard([path])

    assert store.write_immutable_file.await_args.args[:2] == ("tenant-a__job", path)
    store.open_bytes.assert_awaited_once_with("tenant-a__job", path)
    store.delete.assert_awaited_once_with("tenant-a__job", path)


async def test_parent_retention_releases_only_its_review_memory_metadata(fake_app, monkeypatch):
    from transformation_portal.orchestrator.photographic_review_store import MemoryReviewStore

    app, job, _, _ = fake_app
    store = MemoryReviewStore()
    await store.mutate("job", "local", 0, lambda state: state, {"action": "decision"})
    await store.mutate("other-job", "tenant-b", 0, lambda state: state, {"action": "decision"})
    monkeypatch.setattr(adapter, "get_review_store", lambda: store)

    await adapter.forget_review_for_job(app, job)

    assert (await store.get("job", "local"))["state_version"] == 0
    assert (await store.get("other-job", "tenant-b"))["state_version"] == 1
    assert ("job", "local") not in store.events


async def test_losing_cas_publication_rolls_back_without_deleting_winner(fake_app, tmp_path, monkeypatch):
    from transformation_portal.orchestrator.photographic_review_store import MemoryReviewStore, ReviewConflict

    app, _, store, data = fake_app
    state_store = MemoryReviewStore()
    monkeypatch.setattr(adapter, "get_review_store", lambda: state_store)
    source = tmp_path / "published.tif"
    source.write_bytes(data)
    winner = await adapter.AppReviewAccess(app).authorize(_request(), "job")
    loser = await adapter.AppReviewAccess(app).authorize(_request(), "job")
    winner_path, loser_path = "photographic-review/winner/image.tif", "photographic-review/loser/image.tif"
    await winner.publish(winner_path, source, "image/tiff")
    await loser.publish(loser_path, source, "image/tiff")

    def retain(state):
        state["revisions"].append({"variant": {"kind": "review_asset", "storage_path": winner_path}})
        return state

    await state_store.mutate("job", "local", 0, retain, {"action": "revision"})
    with pytest.raises(ReviewConflict):
        await state_store.mutate("job", "local", 0, lambda state: state, {"action": "revision"})
    await loser.discard([loser_path])
    await winner.discard([winner_path])

    store.delete.assert_awaited_once_with("job", loser_path)
    assert len(state_store.events[("job", "local")]) == 1


def _native_inventory(alpha=True, icc=True):
    import io

    import numpy as np

    from transformation_portal.core.image_artifact import ImageMaster
    from transformation_portal.lux_depth_v6.color import RenderRecipe, render_master
    from transformation_portal.lux_depth_v6.products import _delivery

    pixels = np.full((8, 12, 3), 1.25, dtype=np.float32)
    opacity = np.full((8, 12), 0.6, dtype=np.float32) if alpha else None
    profile = b"synthetic-icc-profile-for-carrier-test" if icc else None
    master = ImageMaster(pixels, "1" * 64, 16, opacity, {"purpose": "synthetic-test"}, profile)
    display, render = render_master(master, RenderRecipe())
    delivery, delivery_descriptor = _delivery(display, "input-0000/delivery.tif")
    descriptor = {
        "schema": "tp.lux.graded_photograph.v2",
        "input_id": "input-0000",
        "master": master.to_payload(),
        "display": display.to_payload(),
        "render": render,
        "delivery": delivery_descriptor,
    }
    bodies = {"v6/input-0000/delivery.tif": delivery, "v6/input-0000/photograph.json": canonicalize_json(descriptor)}
    arrays = {"master.npy": pixels}
    if opacity is not None:
        arrays["alpha.npy"] = opacity
    if profile is not None:
        arrays["source-icc.npy"] = np.frombuffer(profile, dtype=np.uint8)
    for name, array in arrays.items():
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        bodies["v6/input-0000/" + name] = buffer.getvalue()
    inventory = [
        {"path": path, "sha256": hashlib.sha256(body).hexdigest(), "size_bytes": len(body)} for path, body in bodies.items()
    ]
    return bodies, inventory, profile


def _install_native_inventory(app, monkeypatch, bodies, inventory):
    app._uses_dispatch_authority = lambda: True
    records = SimpleNamespace(
        get_locator=AsyncMock(return_value=SimpleNamespace(tenant_id="local")),
        committed_manifest=AsyncMock(return_value={"files": inventory}),
    )
    app.get_operational_record_store = lambda: records
    app._get_job_artifact.side_effect = lambda _job_id, path, **_kwargs: StreamingResponse(_chunks(bodies[path]))
    monkeypatch.setattr(adapter.AppReviewAccess, "_originals", AsyncMock(return_value=([], "original_unavailable")))


async def test_legacy_minimal_descriptor_keeps_clip_readable_without_invented_recorded_authority(fake_app, monkeypatch):
    app, *_ = fake_app
    bodies, inventory, _ = _native_inventory(False, False)
    path = "v6/input-0000/photograph.json"
    descriptor = json.loads(bodies[path])
    for key in ("render", "display", "delivery"):
        descriptor.pop(key)
    bodies[path] = canonicalize_json(descriptor)
    for row in inventory:
        if row["path"] == path:
            row.update(sha256=hashlib.sha256(bodies[path]).hexdigest(), size_bytes=len(bodies[path]))
    _install_native_inventory(app, monkeypatch, bodies, inventory)
    context = await adapter.AppReviewAccess(app).authorize(_request(), "job")
    assert all("presentation_policy" not in ref for ref in context.references)
    from transformation_portal.orchestrator.photographic_review_service import _display, load_image

    image = await load_image(context, next(ref for ref in context.references if ref["role"] == "working_master"))
    assert _display(image)[0]
    with pytest.raises(ReviewAccessError, match="Recorded presentation is unavailable"):
        _display(image, view="recorded")


@pytest.mark.parametrize("alpha,icc", [(False, False), (True, False), (True, True)])
async def test_native_master_uses_only_manifest_bound_descriptor_alpha_and_actual_icc(
    fake_app, monkeypatch, tmp_path, alpha, icc
):
    from transformation_portal.photographic_delivery.native import NativeWorkingMaster

    app, _, _, _ = fake_app
    bodies, inventory, profile = _native_inventory(alpha, icc)
    _install_native_inventory(app, monkeypatch, bodies, inventory)

    context = await adapter.AppReviewAccess(app).authorize(_request(), "job")

    native = next(ref for ref in context.references if ref["role"] == "working_master")
    assert native["source_name"] == "master.npy"
    assert native["input_color"] == "linear_srgb"
    components = {}
    for name, ref in native["native"].items():
        if ref is not None:
            target = tmp_path / ref["source_name"]
            await context.materialize(ref, target, adapter.MAX_REVIEW_IMAGE_BYTES)
            assert hashlib.sha256(target.read_bytes()).hexdigest() == ref["sha256"]
            components[name + "_sha256"] = ref["sha256"]
            components[name + "_path"] = target
    if profile is not None:
        assert components["source_icc_path"].read_bytes() == profile
    working = NativeWorkingMaster(**components)
    assert native["source_identity_sha256"] == working.identity(native["sha256"])
    assert native["source_identity_sha256"] != native["sha256"]
    main = tmp_path / "master.npy"
    await context.materialize(native, main, adapter.MAX_REVIEW_IMAGE_BYTES)
    observed, provenance = working.decode(main.read_bytes(), native["sha256"])
    assert float(observed.pixels.max()) == pytest.approx(1.25)
    assert observed.source_bit_depth == 32
    assert provenance["original_source_bit_depth"] == 16
    assert provenance["original_source_icc_verified"] is icc


@pytest.mark.parametrize("missing", ["photograph.json", "alpha.npy"])
async def test_native_master_never_guesses_an_unindexed_descriptor_or_alpha(fake_app, monkeypatch, missing):
    app, _, _, _ = fake_app
    bodies, inventory, _ = _native_inventory()
    inventory[:] = [row for row in inventory if not row["path"].endswith("/" + missing)]
    _install_native_inventory(app, monkeypatch, bodies, inventory)

    context = await adapter.AppReviewAccess(app).authorize(_request(), "job")

    assert not any(ref["role"] == "working_master" for ref in context.references)
    assert context.references[0]["working_master_unavailable_reason"]
    assert all(call.args[1] != "v6/input-0000/" + missing for call in app._get_job_artifact.await_args_list)


async def test_descriptor_change_invalidates_native_composite_identity_with_same_pixel_file(fake_app, monkeypatch):
    app, _, _, _ = fake_app
    bodies, inventory, _ = _native_inventory()
    _install_native_inventory(app, monkeypatch, bodies, inventory)
    first = await adapter.AppReviewAccess(app).authorize(_request(), "job")
    before = next(ref for ref in first.references if ref["role"] == "working_master")
    descriptor_path = "v6/input-0000/photograph.json"
    descriptor = json.loads(bodies[descriptor_path])
    descriptor["master"]["metadata"]["review-note"] = "new descriptor bytes"
    bodies[descriptor_path] = canonicalize_json(descriptor)
    for row in inventory:
        if row["path"] == descriptor_path:
            row.update(sha256=hashlib.sha256(bodies[descriptor_path]).hexdigest(), size_bytes=len(bodies[descriptor_path]))
    second = await adapter.AppReviewAccess(app).authorize(_request(), "job")
    after = next(ref for ref in second.references if ref["role"] == "working_master")
    assert before["sha256"] == after["sha256"]
    assert before["source_identity_sha256"] != after["source_identity_sha256"]


@pytest.mark.parametrize("alpha,icc", [(False, False), (True, False), (True, True)])
async def test_real_service_spools_optional_native_components_and_preserves_bundle_identity(
    fake_app, monkeypatch, tmp_path, alpha, icc
):
    from transformation_portal.orchestrator.photographic_review_service import spool_image

    app, _, _, _ = fake_app
    bodies, inventory, _ = _native_inventory(alpha, icc)
    _install_native_inventory(app, monkeypatch, bodies, inventory)
    context = await adapter.AppReviewAccess(app).authorize(_request(), "job")
    ref = next(item for item in context.references if item["role"] == "working_master")

    image = await spool_image(context, ref, tmp_path / "spooled")

    assert image.source_identity_sha256 == ref["source_identity_sha256"]
    decoded = image.decode()
    assert float(decoded.pixels.max()) == pytest.approx(1.25)
    assert (decoded.alpha is not None) is alpha
    assert image.native is not None
    assert (image.native.source_icc_sha256 is not None) is icc


@pytest.mark.parametrize("condition", ["symlink", "hardlink", "oversized", "replaced"])
async def test_publication_source_requires_pinned_stable_bounded_regular_file(fake_app, tmp_path, monkeypatch, condition):
    app, _, store, data = fake_app
    source = tmp_path / "published.tif"
    source.write_bytes(data)
    if condition == "symlink":
        linked = tmp_path / "linked.tif"
        linked.symlink_to(source)
        source = linked
    elif condition == "hardlink":
        import os

        os.link(source, tmp_path / "linked.tif")
    elif condition == "oversized":
        monkeypatch.setattr(adapter, "MAX_REVIEW_IMAGE_BYTES", len(data) - 1)
    else:
        original_read = adapter.os.read
        replaced = False

        def read_and_replace(descriptor, size):
            nonlocal replaced
            chunk = original_read(descriptor, size)
            if chunk and not replaced:
                replaced = True
                replacement = tmp_path / "replacement.tif"
                replacement.write_bytes(data)
                replacement.replace(source)
            return chunk

        monkeypatch.setattr(adapter.os, "read", read_and_replace)
    context = await adapter.AppReviewAccess(app).authorize(_request(), "job")

    with pytest.raises(ReviewAccessError):
        await context.publish("photographic-review/new/image.tif", source, "image/tiff")

    store.write_immutable_file.assert_not_awaited()


async def test_original_is_derived_only_from_immutable_bound_plan_and_revalidated_policy(fake_app, monkeypatch, tmp_path):
    from transformation_portal.orchestrator import photography_adapter

    app, _, _, data = fake_app
    originals = tmp_path / "originals"
    originals.mkdir()
    (originals / "source.jpg").write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()
    plan = {
        "schema": "tp.execution.plan.v4",
        "configuration": {"input_color": "srgb"},
        "inputs": [{"id": "input-0000", "path": "source.jpg", "sha256": sha, "size_bytes": len(data)}],
    }
    bindings = photography_adapter.PhotographyBindings.from_payload(
        {
            "schema": photography_adapter.PHOTOGRAPHY_BINDINGS_SCHEMA,
            "input_root": str(originals),
            "runtime_python": "/trusted/python",
            "raw_python": None,
            "cache_root": None,
            "companion_root": None,
            "materials_root": None,
        }
    ).canonical_bytes
    validate = Mock()
    monkeypatch.setattr(photography_adapter, "validate_photography_dispatch", validate)
    locator = SimpleNamespace(tenant_id="local")
    records = SimpleNamespace(
        get_locator=AsyncMock(return_value=locator),
        fetch_plan=AsyncMock(return_value=canonicalize_json(plan)),
        fetch_execution_bindings=AsyncMock(return_value=bindings),
        committed_manifest=AsyncMock(return_value={"files": []}),
    )
    policy = SimpleNamespace(input_roots=(originals,), resolve_path=Mock(return_value=originals))
    app._uses_dispatch_authority = lambda: True
    app.get_operational_record_store = lambda: records
    app._execution_policy = lambda: policy

    context = await adapter.AppReviewAccess(app).authorize(_request(), "job")

    assert context.references[0]["artifact_path"] == "original/input-0000.jpg"
    assert context.references[0]["role"] == "original"
    assert context.references[0]["sha256"] == sha
    validate.assert_called_once_with(canonicalize_json(plan), bindings)
    output = tmp_path / "original-copy.jpg"
    await context.materialize(context.references[0], output, 1024)
    assert output.read_bytes() == data
    policy.resolve_path.return_value = tmp_path / "not-authorized"
    with pytest.raises(ReviewAccessError, match="no longer authorized"):
        await context.materialize(context.references[0], tmp_path / "refused.jpg", 1024)
    assert not (tmp_path / "refused.jpg").exists()


@pytest.fixture(params=["v5", "v6", "unified_infer", "unified_process"])
def admitted_originals(request, request_case, fake_app, monkeypatch):
    from transformation_portal.lux_depth_v6.managed import ManagedLuxDepthV6Request
    from transformation_portal.orchestrator.artifact_store.generation import GenerationPublisher
    from transformation_portal.orchestrator.lux_depth_adapter import prepare_unified_dispatch
    from transformation_portal.orchestrator.photography_adapter import prepare_photography_dispatch
    from transformation_portal.orchestrator.photography_v6_adapter import prepare_v6_dispatch

    monkeypatch.setenv("TRANSFORMATION_PORTAL_DA3_PYTHON", sys.executable)
    monkeypatch.delenv("TRANSFORMATION_PORTAL_RAW_PYTHON", raising=False)
    publisher = GenerationPublisher(artifact_store=None, record_store=None)
    if request.param == "v5":
        admitted = prepare_photography_dispatch(request_case, publisher=publisher)
    elif request.param == "v6":
        admitted = prepare_v6_dispatch(ManagedLuxDepthV6Request(request_case), publisher=publisher)
    else:
        workflow = request.param.removeprefix("unified_")
        native_request = ManagedLuxDepthV6Request(request_case) if workflow == "process" else request_case
        admitted = prepare_unified_dispatch(native_request, workflow=workflow, publisher=publisher)
    records = SimpleNamespace(
        get_locator=AsyncMock(return_value=SimpleNamespace(tenant_id="local")),
        fetch_plan=AsyncMock(return_value=admitted.plan_bytes),
        fetch_execution_bindings=AsyncMock(return_value=admitted.bindings_bytes),
        committed_manifest=AsyncMock(return_value={"files": []}),
    )
    policy = SimpleNamespace(input_roots=(request_case.input_dir,), resolve_path=Mock(return_value=request_case.input_dir))
    app, job, _, _ = fake_app
    app._uses_dispatch_authority = lambda: True
    app.get_operational_record_store = lambda: records
    app._execution_policy = lambda: policy
    return app, job, records, policy, request_case


async def test_originals_use_exact_validated_native_inference_authority(admitted_originals, tmp_path):
    app, _, records, policy, request_case = admitted_originals
    context = await adapter.AppReviewAccess(app).authorize(_request(), "job")

    assert isinstance(context, ReviewContext)
    assert len(context.references) == 1
    reference = context.references[0]
    plan = json.loads(records.fetch_plan.return_value)
    inference = plan["inference"] if plan["schema"] == "tp.execution.plan.v5" else plan
    admitted_input = inference["inputs"][0]
    assert reference["artifact_path"] == "original/input-0000.tif"
    assert reference["role"] == "original"
    assert reference["sha256"] == reference["source_identity_sha256"] == admitted_input["sha256"]
    assert reference["size_bytes"] == admitted_input["size_bytes"]
    assert reference["input_color"] == inference["configuration"]["input_color"] == "srgb"
    output = tmp_path / "original-copy.tif"
    await context.materialize(reference, output, 16 * 1024)
    assert output.read_bytes() == (request_case.input_dir / admitted_input["path"]).read_bytes()
    assert not request_case.output_dir.exists()

    policy.resolve_path.return_value = tmp_path / "revoked"
    with pytest.raises(ReviewAccessError, match="no longer authorized"):
        await context.materialize(reference, tmp_path / "refused.tif", 16 * 1024)
    assert not (tmp_path / "refused.tif").exists()


@pytest.mark.parametrize("field,value", [("sha256", "f" * 64), ("input_color", "linear_srgb")])
async def test_originals_reject_changed_inference_before_physical_access(admitted_originals, field, value):
    app, _, records, policy, _ = admitted_originals
    plan = json.loads(records.fetch_plan.return_value)
    inference = plan["inference"] if plan["schema"] == "tp.execution.plan.v5" else plan
    if field == "sha256":
        inference["inputs"][0][field] = value
    else:
        inference["configuration"][field] = value
    records.fetch_plan.return_value = canonicalize_json(plan)

    response = await adapter.AppReviewAccess(app).authorize(_request(), "job")

    assert isinstance(response, Response)
    assert response.status_code == 503
    assert json.loads(response.body)["error"]["code"] == "REVIEW_UNAVAILABLE"
    policy.resolve_path.assert_not_called()


async def test_originals_never_infer_missing_physical_bindings(admitted_originals):
    app, job, records, policy, _ = admitted_originals
    records.fetch_execution_bindings.return_value = None

    originals, reason = await adapter.AppReviewAccess(app)._originals(job, None)

    assert originals == []
    assert reason == "immutable_original_binding_unavailable"
    policy.resolve_path.assert_not_called()


async def test_registered_actual_app_routes_reject_unauthenticated_and_return_typed_missing_job(monkeypatch):
    from httpx import ASGITransport, AsyncClient

    import app

    monkeypatch.setattr(app, "API_KEY_SECRET", "review-contract-secret")
    monkeypatch.setattr(app, "ENFORCE_JOB_API_KEY", True)
    monkeypatch.setattr(app, "PILOT_CONTROL_PLANE_ENABLED", False)
    monkeypatch.setattr(app, "RATE_LIMIT_PER_MINUTE", 0)
    monkeypatch.setattr(app, "_cleanup_expired_jobs", AsyncMock())
    monkeypatch.setattr(app, "_load_job_view", AsyncMock(return_value=None))
    async with AsyncClient(transport=ASGITransport(app=app.app), base_url="http://testserver") as client:
        denied = await client.get("/v1/jobs/missing/photographic-review")
        missing = await client.get("/v1/jobs/missing/photographic-review", headers={"x-api-key": "review-contract-secret"})

    assert denied.status_code == 401
    assert denied.json()["error"]["code"] == "UNAUTHORIZED"
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "NOT_FOUND"
    assert missing.headers["cache-control"] == "no-store"
    assert "/v1/jobs/{job_id}/photographic-review/native-crop" in app.app.openapi()["paths"]


def _change_native_component(bodies, inventory, component):
    import io

    import numpy as np

    from transformation_portal.core.image_artifact import artifact_content_hash

    path = "v6/input-0000/" + ("photograph.json" if component == "descriptor" else "alpha.npy")
    if component == "descriptor":
        descriptor = json.loads(bodies[path])
        descriptor["master"]["metadata"]["current-revision"] = "different descriptor bytes"
        bodies[path] = canonicalize_json(descriptor)
    else:
        buffer = io.BytesIO()
        np.save(buffer, np.full((8, 12), 0.3, dtype=np.float32), allow_pickle=False)
        bodies[path] = buffer.getvalue()
    descriptor_path = "v6/input-0000/photograph.json"
    descriptor = json.loads(bodies[descriptor_path])
    descriptor["render"]["input_master_sha256"] = artifact_content_hash(
        descriptor["master"],
        {
            "pixels": np.load(io.BytesIO(bodies["v6/input-0000/master.npy"]), allow_pickle=False),
            "alpha": np.load(io.BytesIO(bodies["v6/input-0000/alpha.npy"]), allow_pickle=False),
        },
    )
    bodies[descriptor_path] = canonicalize_json(descriptor)
    for row in inventory:
        if row["path"] in {path, descriptor_path}:
            current = bodies[row["path"]]
            row.update(sha256=hashlib.sha256(current).hexdigest(), size_bytes=len(current))


@pytest.mark.parametrize("component", ["descriptor", "alpha"])
@pytest.mark.parametrize("operation", ["render", "native-crop", "decisions", "revisions"])
async def test_public_native_operations_reject_stale_bundle_identity_before_approval(
    fake_app, monkeypatch, component, operation
):
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from transformation_portal.api.routes.photographic_review import create_photographic_review_router
    from transformation_portal.orchestrator import photographic_review_service as service
    from transformation_portal.orchestrator.photographic_review_store import MemoryReviewStore

    app, _, _, _ = fake_app
    bodies, inventory, _ = _native_inventory()
    _install_native_inventory(app, monkeypatch, bodies, inventory)
    state_store = MemoryReviewStore()
    monkeypatch.setattr(service, "get_review_store", lambda: state_store)
    access = adapter.AppReviewAccess(app)
    before = await access.authorize(_request(), "job")
    selected = next(ref for ref in before.references if ref["role"] == "working_master")
    _change_native_component(bodies, inventory, component)
    api = FastAPI()
    api.include_router(create_photographic_review_router(access.runtime()))
    common = {
        "artifact_path": selected["artifact_path"],
        "sha256": selected["sha256"],
        "source_identity_sha256": selected["source_identity_sha256"],
    }
    async with AsyncClient(transport=ASGITransport(app=api), base_url="http://testserver") as client:
        path = "/v1/jobs/job/photographic-review/" + operation
        if operation in {"render", "native-crop"}:
            params = {**common, **({"x": 0, "y": 0, "width": 2, "height": 2} if operation == "native-crop" else {})}
            response = await client.get(path, params=params)
        else:
            payload = {**common, "expected_version": 0}
            if operation == "decisions":
                payload["status"] = "approved"
            else:
                payload.update(scene_id=selected["scene_id"], recipe={"exposure_stops": 0.0}, preview_only=False)
            response = await client.post(path, json=payload)

    assert response.status_code == 409, response.text
    assert "identity" in response.text
    assert (await state_store.get("job", "local"))["decisions"] == []
    assert (await state_store.get("job", "local"))["state_version"] == 0


async def test_native_approval_and_delivery_are_revoked_by_changed_companion_identity(fake_app, monkeypatch):
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from transformation_portal.api.routes.photographic_review import create_photographic_review_router
    from transformation_portal.orchestrator import photographic_review_service as service
    from transformation_portal.orchestrator.photographic_review_store import MemoryReviewStore

    app, _, store, _ = fake_app
    bodies, inventory, _ = _native_inventory()
    _install_native_inventory(app, monkeypatch, bodies, inventory)
    state_store, published = MemoryReviewStore(), {}
    monkeypatch.setattr(service, "get_review_store", lambda: state_store)
    monkeypatch.setattr(adapter, "get_review_store", lambda: state_store)

    async def write(_job_id, key, source, **_kwargs):
        assert key not in published
        published[key] = source.read_bytes()

    store.write_immutable_file.side_effect = write
    store.open_bytes.side_effect = lambda _job_id, key: _chunks(published[key])
    access = adapter.AppReviewAccess(app)
    before = await access.authorize(_request(), "job")
    selected = next(ref for ref in before.references if ref["role"] == "working_master")
    api = FastAPI()
    api.include_router(create_photographic_review_router(access.runtime()))
    base = "/v1/jobs/job/photographic-review"
    async with AsyncClient(transport=ASGITransport(app=api), base_url="http://testserver") as client:
        decision = await client.post(
            base + "/decisions",
            json={
                "artifact_path": selected["artifact_path"],
                "sha256": selected["sha256"],
                "source_identity_sha256": selected["source_identity_sha256"],
                "status": "approved",
                "expected_version": 0,
            },
        )
        assert decision.status_code == 200, decision.text
        assert decision.json()["data"]["decisions"][0]["source_identity_sha256"] == selected["source_identity_sha256"]
        bundle = await client.post(
            base + "/delivery",
            json={
                "property_name": "Synthetic native",
                "presets": ["photographic_master", "retouch_print"],
                "approved_sha256": [selected["sha256"]],
                "expected_version": 1,
            },
        )
        assert bundle.status_code == 200, bundle.text
        download = bundle.json()["data"]["download_url"]
        assert (await client.get(download)).status_code == 200
        _change_native_component(bodies, inventory, "alpha")
        withdrawn = await client.get(download)
        assert withdrawn.status_code == 409, withdrawn.text
        refreshed = await access.authorize(_request(), "job")
        current = next(ref for ref in refreshed.references if ref["role"] == "working_master")
        reapproved = await client.post(
            base + "/decisions",
            json={
                "artifact_path": current["artifact_path"],
                "sha256": current["sha256"],
                "source_identity_sha256": current["source_identity_sha256"],
                "status": "approved",
                "expected_version": 2,
            },
        )
        assert reapproved.status_code == 200, reapproved.text
        old_bundle = await client.get(download)

    assert old_bundle.status_code == 409, "approving a new native bundle must not revive an earlier bundle's download"
    assert (await state_store.get("job", "local"))["state_version"] == 3


async def test_exact_delivery_selectors_export_distinct_native_bundles_with_shared_pixel_file(fake_app, monkeypatch):
    import io
    import zipfile

    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from transformation_portal.api.routes.photographic_review import create_photographic_review_router
    from transformation_portal.orchestrator import photographic_review_service as service
    from transformation_portal.orchestrator.photographic_review_store import MemoryReviewStore

    app, _, store, _ = fake_app
    bodies, inventory, _ = _native_inventory(False, False)
    for path, content in list(bodies.items()):
        second_path = path.replace("input-0000", "input-0001")
        if path.endswith("photograph.json"):
            descriptor = json.loads(content)
            descriptor["input_id"] = "input-0001"
            descriptor["delivery"]["path"] = "input-0001/delivery.tif"
            content = canonicalize_json(descriptor)
        bodies[second_path] = content
        inventory.append({"path": second_path, "sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)})
    _install_native_inventory(app, monkeypatch, bodies, inventory)
    state_store, published = MemoryReviewStore(), {}
    monkeypatch.setattr(service, "get_review_store", lambda: state_store)
    monkeypatch.setattr(adapter, "get_review_store", lambda: state_store)

    async def write(_job_id, key, source, **_kwargs):
        assert key not in published
        published[key] = source.read_bytes()

    store.write_immutable_file.side_effect = write
    store.open_bytes.side_effect = lambda _job_id, key: _chunks(published[key])
    access = adapter.AppReviewAccess(app)
    context = await access.authorize(_request(), "job")
    refs = [ref for ref in context.references if ref["role"] == "working_master"]
    assert len(refs) == 2
    assert refs[0]["sha256"] == refs[1]["sha256"]
    assert refs[0]["source_identity_sha256"] != refs[1]["source_identity_sha256"]
    selectors = [{key: ref[key] for key in ("artifact_path", "sha256", "source_identity_sha256")} for ref in refs]
    api = FastAPI()
    api.include_router(create_photographic_review_router(access.runtime()))
    base = "/v1/jobs/job/photographic-review"
    async with AsyncClient(transport=ASGITransport(app=api), base_url="http://testserver") as client:
        first = await client.post(base + "/decisions", json={**selectors[0], "status": "approved", "expected_version": 0})
        assert first.status_code == 200, first.text
        request = {
            "property_name": "Synthetic shared pixels",
            "presets": ["photographic_master", "retouch_print"],
            "approved_sources": selectors,
            "expected_version": 1,
        }
        unapproved = await client.post(base + "/delivery", json=request)
        assert unapproved.status_code == 400, unapproved.text
        assert "current approval" in unapproved.text
        store.write_immutable_file.assert_not_awaited()
        second = await client.post(base + "/decisions", json={**selectors[1], "status": "approved", "expected_version": 1})
        assert second.status_code == 200, second.text
        request["expected_version"] = 2
        invalid_requests = [
            ({**request, "approved_sources": [{**selectors[0], "source_identity_sha256": "0" * 64}, selectors[1]]}, 409),
            ({**request, "approved_sha256": ["f" * 64]}, 400),
            ({**request, "approved_sha256": [selectors[0]["sha256"], selectors[0]["sha256"]]}, 400),
            ({**request, "approved_sources": [selectors[0], selectors[0]]}, 400),
        ]
        for invalid, status in invalid_requests:
            response = await client.post(base + "/delivery", json=invalid)
            assert response.status_code == status, response.text
        assert (await state_store.get("job", "local"))["state_version"] == 2
        store.write_immutable_file.assert_not_awaited()
        bundle = await client.post(base + "/delivery", json=request)
        assert bundle.status_code == 200, bundle.text
        download = await client.get(bundle.json()["data"]["download_url"])

    assert download.status_code == 200, download.text
    expected = {(row["artifact_path"], row["sha256"], row["source_identity_sha256"]) for row in selectors}
    with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
        assert archive.testzip() is None
        approval_path = next(name for name in archive.namelist() if name.endswith("/approval-record.json"))
        approval = json.loads(archive.read(approval_path))
        assert {
            (row["artifact_path"], row["sha256"], row["source_identity_sha256"]) for row in approval["masters"]
        } == expected
        inventory_path = next(name for name in archive.namelist() if name.endswith("/inventory.json"))
        inventory = json.loads(archive.read(inventory_path))
        assert {row["source_identity_sha256"] for row in inventory["sources"]} == {row[2] for row in expected}
        assert len(inventory["sources"]) == 2
        assert {row["source_master_sha256"] for row in inventory["sources"]} == {refs[0]["sha256"]}
        archive_root = inventory_path.removesuffix("inventory.json")
        masters = [row for row in inventory["files"] if row["path"].endswith(".npy")]
        assert {row["source_identity_sha256"] for row in masters} == {row[2] for row in expected}
        for row in [*masters, *inventory["native_components"]]:
            actual = archive.read(archive_root + row["path"])
            assert hashlib.sha256(actual).hexdigest() == row["sha256"]
            assert len(actual) == row["size_bytes"]
        retained_descriptors = {
            row["source_identity_sha256"]: row["sha256"]
            for row in inventory["native_components"]
            if row["component"] == "descriptor"
        }
        assert retained_descriptors == {ref["source_identity_sha256"]: ref["native"]["descriptor"]["sha256"] for ref in refs}
        checksum_path = next(name for name in archive.namelist() if name.endswith("/_package/SHA256SUMS.txt"))
        for line in archive.read(checksum_path).decode().splitlines():
            digest, member = line.split("  ", 1)
            assert hashlib.sha256(archive.read(member)).hexdigest() == digest
    assert (await state_store.get("job", "local"))["state_version"] == 3


@pytest.mark.parametrize("revocation", ["artifacts-deleted", "parent-deleted"])
async def test_parent_revocation_during_inspection_cannot_return_successful_approval(fake_app, monkeypatch, revocation):
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from transformation_portal.api.routes.photographic_review import create_photographic_review_router
    from transformation_portal.orchestrator import photographic_review_service as service
    from transformation_portal.orchestrator.photographic_review_store import MemoryReviewStore

    app, _, _, _ = fake_app
    bodies, inventory, _ = _native_inventory(False, False)
    _install_native_inventory(app, monkeypatch, bodies, inventory)
    state_store = MemoryReviewStore()
    monkeypatch.setattr(service, "get_review_store", lambda: state_store)
    monkeypatch.setattr(adapter, "get_review_store", lambda: state_store)
    metadata = service._metadata

    def inspect_then_revoke(image):
        result = metadata(image)
        if revocation == "artifacts-deleted":
            app._artifacts_deleted = lambda _: True
        else:
            app._load_job_view.return_value = None
        return result

    monkeypatch.setattr(service, "_metadata", inspect_then_revoke)
    access = adapter.AppReviewAccess(app)
    context = await access.authorize(_request(), "job")
    selected = next(ref for ref in context.references if ref["role"] == "working_master")
    api = FastAPI()
    api.include_router(create_photographic_review_router(access.runtime()))
    async with AsyncClient(transport=ASGITransport(app=api), base_url="http://testserver") as client:
        response = await client.post(
            "/v1/jobs/job/photographic-review/decisions",
            json={
                "artifact_path": selected["artifact_path"],
                "sha256": selected["sha256"],
                "source_identity_sha256": selected["source_identity_sha256"],
                "status": "approved",
                "expected_version": 0,
            },
        )
    assert response.status_code == (410 if revocation == "artifacts-deleted" else 404), response.text
    assert (await state_store.get("job", "local"))["decisions"] == []
    assert not state_store.events


@pytest.mark.parametrize("source_kind", ["original", "file-response"])
async def test_cancelled_snapshot_drains_its_thread_before_removing_owned_destination(
    fake_app, monkeypatch, tmp_path, source_kind
):
    app, _, _, data = fake_app
    source = tmp_path / "source.tif"
    source.write_bytes(data)
    if source_kind == "original":
        original = {
            **_reference(data),
            "artifact_path": "original/input-0000.tif",
            "source_name": "source.tif",
            "role": "original",
            "kind": "original",
            "private_original_root": str(tmp_path),
            "private_original_path": source.name,
        }
        monkeypatch.setattr(adapter.AppReviewAccess, "_originals", AsyncMock(return_value=([original], None)))
    else:
        app._validate_resolved_job_artifact_path = Mock(return_value=(tmp_path, source, source.name))
        app._get_job_artifact.side_effect = None
        app._get_job_artifact.return_value = FileResponse(source)
    context = await adapter.AppReviewAccess(app).authorize(_request(), "job")
    ref = next(row for row in context.references if row["role"] == ("original" if source_kind == "original" else "final"))
    operation = adapter.snapshot if source_kind == "original" else adapter.os.read
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def blocked_snapshot(*args, **kwargs):
        started.set()
        try:
            assert release.wait(timeout=5), "test did not release the controlled snapshot thread"
            assert source.is_file()
            return operation(*args, **kwargs)
        finally:
            finished.set()

    if source_kind == "original":
        monkeypatch.setattr(adapter, "snapshot", blocked_snapshot)
    else:
        monkeypatch.setattr(adapter.os, "read", blocked_snapshot)
    destination = tmp_path / "owned-snapshot.tif"
    task = asyncio.create_task(context.materialize(ref, destination, 1024))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done(), "snapshot ownership ended while its thread was still running"
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()
        assert not destination.exists()
        assert source.read_bytes() == data
    finally:
        release.set()
        await asyncio.to_thread(finished.wait, 2)
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("phase", ["staging", "write-success", "write-failure", "readback"])
async def test_cancelled_publication_drains_owned_work_and_rolls_back_only_successful_new_key(
    fake_app, monkeypatch, tmp_path, phase
):
    app, _, store, data = fake_app
    source = tmp_path / "delivery.tif"
    source.write_bytes(data)
    key = "photographic-review/cancelled/delivery.tif"
    existing = b"existing immutable bytes"
    published = {key: existing} if phase == "write-failure" else {}
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    delete_started, delete_release = threading.Event(), threading.Event()
    paths = {}
    original_snapshot = adapter._publication_snapshot

    def wait_for_release():
        started.set()
        assert release.wait(timeout=5), "test did not release the controlled publication operation"

    def publication_snapshot(input_path, staged, maximum):
        paths["staged"] = staged
        if phase == "staging":
            try:
                wait_for_release()
                assert staged.parent.is_dir()
                return original_snapshot(input_path, staged, maximum)
            finally:
                finished.set()
        return original_snapshot(input_path, staged, maximum)

    def blocking_write(staged):
        try:
            wait_for_release()
            assert staged.is_file(), "staging was removed before the immutable write stopped"
            if phase == "write-failure":
                raise RuntimeError("immutable key already exists")
            published[key] = staged.read_bytes()
        finally:
            finished.set()

    async def write(_job_id, actual_key, staged, **_kwargs):
        assert actual_key == key
        if phase.startswith("write-"):
            await asyncio.to_thread(blocking_write, staged)
        else:
            published[key] = staged.read_bytes()

    async def readback():
        try:
            if phase == "readback":
                await asyncio.to_thread(wait_for_release)
            yield published[key]
        finally:
            if phase == "readback":
                finished.set()

    def blocking_delete():
        delete_started.set()
        assert delete_release.wait(timeout=5), "test did not release controlled publication rollback"
        return int(published.pop(key, None) is not None)

    async def delete(_job_id, actual_key):
        assert actual_key == key
        if phase == "write-success":
            return await asyncio.to_thread(blocking_delete)
        return int(published.pop(key, None) is not None)

    monkeypatch.setattr(adapter, "_publication_snapshot", publication_snapshot)
    store.write_immutable_file.side_effect = write
    store.open_bytes.side_effect = lambda *_args: readback()
    store.delete.side_effect = delete
    context = await adapter.AppReviewAccess(app).authorize(_request(), "job")
    task = asyncio.create_task(context.publish(key, source, "image/tiff"))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done(), "publication released staging before its owned operation stopped"
        assert paths["staged"].parent.is_dir()
        release.set()
        if phase == "write-success":
            assert await asyncio.to_thread(delete_started.wait, 2)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done(), "repeated cancellation abandoned the successful write's rollback"
            delete_release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()
        assert not paths["staged"].parent.exists()
        assert source.read_bytes() == data
        if phase in {"write-success", "readback"}:
            assert published == {}
            store.delete.assert_awaited_once_with("job", key)
        else:
            assert published == ({key: existing} if phase == "write-failure" else {})
            store.delete.assert_not_awaited()
        if phase == "staging":
            store.write_immutable_file.assert_not_awaited()
    finally:
        release.set()
        delete_release.set()
        await asyncio.to_thread(finished.wait, 2)
        await asyncio.gather(task, return_exceptions=True)


async def test_admission_authenticates_before_discovery_and_binds_manifest_identity(fake_app, monkeypatch):
    app, job, _, _ = fake_app
    access = adapter.AppReviewAccess(app)
    discover = AsyncMock(return_value=[_reference()])
    monkeypatch.setattr(access, "_references", discover)
    context = await access.authorize_admission(_request(), "job")
    assert isinstance(context, ReviewContext) and context.references == []
    assert context.actor["assurance"] == "backend_api_key"
    assert context.binding_identity == hashlib.sha256(canonicalize_json(job.artifacts)).hexdigest()
    discover.assert_not_awaited()
    assert await context.discover_references() == [_reference()]
    discover.assert_awaited_once()
    await context.revalidate()
    job.artifacts["items"][0]["sha256"] = "f" * 64
    with pytest.raises(ReviewAccessError, match="bindings changed"):
        await context.revalidate()


async def test_admission_invalid_credentials_never_discover_or_queue(fake_app, monkeypatch):
    app, _, _, _ = fake_app
    app._has_valid_api_key = lambda _: False
    access = adapter.AppReviewAccess(app)
    discover = AsyncMock()
    monkeypatch.setattr(access, "_references", discover)
    result = await access.authorize_admission(_request(), "job")
    assert result.status_code == 401
    discover.assert_not_awaited()
    app._load_job_view.assert_not_awaited()


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("phase", ["before-write", "readback"])
async def test_http_publication_overload_preserves_retry_envelope_and_owned_cleanup(fake_app, monkeypatch, version, phase):
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from transformation_portal.api.routes.photographic_review import create_photographic_review_router
    from transformation_portal.orchestrator import photographic_admission as admission
    from transformation_portal.orchestrator import photographic_review_service as service
    from transformation_portal.orchestrator.photographic_review_store import MemoryReviewStore

    monkeypatch.setenv("TP_PHOTO_REVIEW_ENABLED", "1")
    app, _, store, _ = fake_app
    bodies, inventory, _ = _native_inventory(False, False)
    original_bodies = dict(bodies)
    _install_native_inventory(app, monkeypatch, bodies, inventory)
    state_store, published, staged_paths, owners = MemoryReviewStore(), {}, [], []
    monkeypatch.setattr(service, "get_review_store", lambda: state_store)
    monkeypatch.setattr(adapter, "get_review_store", lambda: state_store)
    capacity_available = True
    monkeypatch.setattr(
        admission.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * admission.GIB if capacity_available else 1)
    )
    original_snapshot, original_verify = adapter._publication_snapshot, adapter._verify_staged

    def snapshot_then_drop_capacity(source, staged, maximum):
        nonlocal capacity_available
        result = original_snapshot(source, staged, maximum)
        staged_paths.append(staged)
        owners.append(admission.current_review_lease.get())
        if phase == "before-write":
            capacity_available = False
        return result

    async def write(_job, key, source, **_kwargs):
        published[key] = source.read_bytes()

    async def delete(_job, key):
        return int(published.pop(key, None) is not None)

    async def verify_then_overload(*args):
        await original_verify(*args)
        raise admission.ReviewOverload("scratch_unavailable")

    monkeypatch.setattr(adapter, "_publication_snapshot", snapshot_then_drop_capacity)
    if phase == "readback":
        monkeypatch.setattr(adapter, "_verify_staged", verify_then_overload)
    store.write_immutable_file.side_effect = write
    store.open_bytes.side_effect = lambda _job, key: _chunks(published[key])
    store.delete.side_effect = delete
    api = FastAPI()
    api.include_router(create_photographic_review_router(adapter.AppReviewAccess(app).runtime()))
    base = f"/{version}/jobs/job/photographic-review"
    async with api.router.lifespan_context(api):
        async with AsyncClient(transport=ASGITransport(app=api), base_url="http://testserver") as client:
            catalogue = await client.get(base)
            assert catalogue.status_code == 200, catalogue.text
            native = next(row for row in catalogue.json()["data"]["scenes"][0]["variants"] if row["role"] == "working_master")
            response = await client.post(
                base + "/revisions",
                json={
                    **{key: native[key] for key in ("artifact_path", "sha256", "source_identity_sha256")},
                    "scene_id": "input-0000",
                    "expected_version": 0,
                    "recipe": {"exposure_stops": -0.1},
                    "preview_only": False,
                },
            )
    assert response.status_code == 503 and response.headers.get("Retry-After") == "1", response.text
    error = response.json()["error"]
    assert error["code"] == "RESOURCE_EXHAUSTED" and error["retriable"] is True
    assert error["details"] == {"job_id": "job", "reason": "scratch_unavailable", "scope": "process"}
    assert staged_paths and all(not path.parent.exists() for path in staged_paths)
    assert all(owner is not None and owner.released for owner in owners)
    assert published == {} and bodies == original_bodies
    assert (await state_store.get("job", "local"))["state_version"] == 0
    if phase == "before-write":
        store.write_immutable_file.assert_not_awaited()
        store.delete.assert_not_awaited()
    else:
        store.write_immutable_file.assert_awaited_once()
        store.delete.assert_awaited_once()
