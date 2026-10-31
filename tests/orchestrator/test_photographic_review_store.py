"""Atomic review projection/audit contracts; optional dedicated-Postgres lane.

The live lane never resets a database or applies migrations. Set
TP_TEST_POSTGRES_URL to an already migrated dedicated *_test database. Each
case owns one unique job row, removed by primary key when the test finishes.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import delete, select, update
from sqlalchemy.engine import make_url

from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.orchestrator.models import JobModel, PhotographicReviewEventModel, PhotographicReviewStateModel
from transformation_portal.orchestrator.photographic_review_store import (
    MemoryReviewStore,
    PostgresReviewStore,
    ReviewConflict,
    empty_state,
    get_review_store,
)
from transformation_portal.orchestrator.storage.postgres import _SharedEngine

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest_asyncio.fixture(params=["memory", "postgres"])
async def review_store(request):
    job_id = f"review-test-{uuid4().hex}"
    if request.param == "memory":
        yield MemoryReviewStore(), job_id
        return
    url = os.environ.get("TP_TEST_POSTGRES_URL", "").strip()
    if not url:
        pytest.skip("TP_TEST_POSTGRES_URL missing; durable review integration requires a dedicated migrated service")
    database = make_url(url).database or ""
    if not database.endswith("_test"):
        pytest.fail("durable review integration requires a dedicated *_test database")
    _, factory = await _SharedEngine.get(url)
    async with factory() as session, session.begin():
        session.add(JobModel(id=job_id, created_at=1.0, state="succeeded"))
    try:
        yield PostgresReviewStore(url), job_id
    finally:
        _, cleanup_factory = await _SharedEngine.get(url)
        async with cleanup_factory() as session, session.begin():
            await session.execute(delete(JobModel).where(JobModel.id == job_id))
        await _SharedEngine.dispose(url)


async def _events(store, job_id):
    if isinstance(store, MemoryReviewStore):
        return list(store.events.get((job_id, "tenant-a"), []))
    _, factory = await _SharedEngine.get(store.database_url)
    async with factory() as session:
        rows = (
            await session.scalars(
                select(PhotographicReviewEventModel)
                .where(PhotographicReviewEventModel.job_id == job_id)
                .order_by(PhotographicReviewEventModel.version)
            )
        ).all()
        for row in rows:
            assert hashlib.sha256(row.canonical_bytes).hexdigest() == row.sha256
            assert json.loads(row.canonical_bytes)["version"] == row.version
            assert row.tenant_id == "tenant-a"
        return [row.canonical_bytes for row in rows]


def _approve(state):
    state["decisions"].append({"artifact_sha256": "a" * 64, "decision": "approved", "notes": {"actor": "reviewer-a"}})
    return state


async def test_new_projection_is_empty_and_isolated(review_store):
    store, job_id = review_store
    snapshot = await store.get(job_id, "tenant-a")
    snapshot["decisions"].append({"decision": "approved"})
    assert await store.get(job_id, "tenant-a") == empty_state()
    assert await _events(store, job_id) == []


async def test_mutation_and_projection_audit_commit_together(review_store):
    store, job_id = review_store
    event = {"action": "decision", "actor": "reviewer-a", "artifact_sha256": "a" * 64}

    updated = await store.mutate(job_id, "tenant-a", 0, _approve, event)

    assert updated["state_version"] == 1
    assert await store.get(job_id, "tenant-a") == updated
    raw_events = await _events(store, job_id)
    assert len(raw_events) == 1
    audit = json.loads(raw_events[0])
    assert audit["schema"] == "tp.photographic_review.event.v1"
    assert audit["version"] == 1
    assert audit["projection_sha256"] == hashlib.sha256(canonicalize_json(updated)).hexdigest()
    assert raw_events[0] == canonicalize_json(audit)
    updated["decisions"][0]["notes"]["actor"] = "returned-value-mutated"
    event["actor"] = "caller-mutated"
    assert (await store.get(job_id, "tenant-a"))["decisions"][0]["notes"]["actor"] == "reviewer-a"
    assert json.loads((await _events(store, job_id))[0])["actor"] == "reviewer-a"


async def test_concurrent_same_version_has_exactly_one_winner_and_event(review_store):
    store, job_id = review_store

    outcomes = await asyncio.gather(
        store.mutate(job_id, "tenant-a", 0, _approve, {"actor": "one"}),
        store.mutate(job_id, "tenant-a", 0, _approve, {"actor": "two"}),
        return_exceptions=True,
    )

    assert sum(isinstance(outcome, dict) for outcome in outcomes) == 1
    assert sum(isinstance(outcome, ReviewConflict) for outcome in outcomes) == 1
    assert (await store.get(job_id, "tenant-a"))["state_version"] == 1
    assert len((await store.get(job_id, "tenant-a"))["decisions"]) == 1
    assert len(await _events(store, job_id)) == 1


@pytest.mark.parametrize("version", [False, True, -1, 1, 0.0, "0"])
async def test_invalid_or_stale_version_cannot_change_projection_or_append_event(review_store, version):
    store, job_id = review_store
    with pytest.raises(ReviewConflict):
        await store.mutate(job_id, "tenant-a", version, _approve, {"action": "decision"})
    assert await store.get(job_id, "tenant-a") == empty_state()
    assert await _events(store, job_id) == []


async def test_throwing_mutator_cannot_leave_partially_changed_approval(review_store):
    store, job_id = review_store
    await store.mutate(job_id, "tenant-a", 0, _approve, {"action": "approve"})
    before = await store.get(job_id, "tenant-a")
    before_events = await _events(store, job_id)

    def fail(state):
        state["decisions"].clear()
        raise RuntimeError("cancelled mutation")

    with pytest.raises(RuntimeError, match="cancelled mutation"):
        await store.mutate(job_id, "tenant-a", 1, fail, {"action": "revision"})

    assert await store.get(job_id, "tenant-a") == before
    assert await _events(store, job_id) == before_events


@pytest.mark.parametrize("field", ["schema", "version", "projection_sha256"])
async def test_event_cannot_override_reserved_authority_fields(review_store, field):
    store, job_id = review_store
    with pytest.raises(ValueError, match="authority fields"):
        await store.mutate(job_id, "tenant-a", 0, _approve, {field: "spoofed"})
    assert await store.get(job_id, "tenant-a") == empty_state()
    assert await _events(store, job_id) == []


@pytest.mark.parametrize("target", ["event", "projection"])
async def test_oversized_mutation_is_atomic_rejection(review_store, target):
    store, job_id = review_store
    event = {"action": "decision", "notes": "x" * (129 * 1024) if target == "event" else "safe"}

    def change(state):
        state["notes"] = "x" * (2 * 1024 * 1024) if target == "projection" else "safe"
        return state

    with pytest.raises(ValueError, match="bounded size"):
        await store.mutate(job_id, "tenant-a", 0, change, event)
    assert await store.get(job_id, "tenant-a") == empty_state()
    assert await _events(store, job_id) == []


async def test_cross_tenant_cannot_read_or_mutate_owned_projection(review_store):
    store, job_id = review_store
    await store.mutate(job_id, "tenant-a", 0, _approve, {"action": "approve"})
    with pytest.raises(ValueError, match="tenant mismatch"):
        await store.get(job_id, "tenant-b")
    with pytest.raises(ValueError, match="tenant mismatch"):
        await store.mutate(job_id, "tenant-b", 0, _approve, {"action": "approve"})
    assert len(await _events(store, job_id)) == 1
    assert (await store.get(job_id, "tenant-a"))["state_version"] == 1


@pytest.mark.parametrize("payload_version", [False, True, -1, "1", None, "missing", 1])
async def test_postgres_corrupt_projection_cannot_be_read_repaired_or_audited(review_store, payload_version):
    store, job_id = review_store
    if isinstance(store, MemoryReviewStore):
        pytest.skip("persistent-row corruption is exercised only in the isolated dedicated Postgres lane")
    await store.mutate(job_id, "tenant-a", 0, _approve, {"action": "approve"})
    before_events = await _events(store, job_id)
    _, factory = await _SharedEngine.get(store.database_url)
    async with factory() as session, session.begin():
        row = await session.get(PhotographicReviewStateModel, job_id)
        corrupt_payload = {**row.payload, "state_version": payload_version}
        if payload_version == "missing":
            corrupt_payload.pop("state_version")
        # The valid-payload case specifically guards the locked mutation path's
        # independent database-column/projection invariant.
        corrupt_column_version = 99 if type(payload_version) is int and payload_version == 1 else 1
        # Force the deliberately corrupt JSONB bytes to be written: Python's
        # True == 1 equality can make ORM change detection omit that mutation.
        await session.execute(
            update(PhotographicReviewStateModel)
            .where(PhotographicReviewStateModel.job_id == job_id)
            .values(payload=corrupt_payload, version=corrupt_column_version)
        )
    mutator_called = False

    def forbidden_change(state):
        nonlocal mutator_called
        mutator_called = True
        return _approve(state)

    with pytest.raises(ValueError, match="Review projection version"):
        await store.get(job_id, "tenant-a")
    with pytest.raises(ValueError, match="Review projection version"):
        await store.mutate(job_id, "tenant-a", 1, forbidden_change, {"action": "must-not-append"})

    assert not mutator_called
    assert await _events(store, job_id) == before_events
    async with factory() as session:
        row = await session.get(PhotographicReviewStateModel, job_id)
        assert row.version == corrupt_column_version
        assert row.payload == corrupt_payload


async def test_memory_forget_requires_owner_and_removes_only_expired_job_lifetime():
    store = MemoryReviewStore()
    before = await store.mutate("expired", "tenant-a", 0, _approve, {"action": "approve"})
    other = await store.mutate("retained", "tenant-b", 0, _approve, {"action": "approve"})
    with pytest.raises(ValueError, match="tenant mismatch"):
        await store.forget("expired", "tenant-b")
    assert await store.get("expired", "tenant-a") == before
    assert len(store.events[("expired", "tenant-a")]) == 1

    await store.forget("expired", "tenant-a")

    assert await store.get("expired", "tenant-a") == empty_state()
    assert ("expired", "tenant-a") not in store.states
    assert ("expired", "tenant-a") not in store.events
    assert "expired" not in store.owners
    assert await store.get("retained", "tenant-b") == other
    assert len(store.events[("retained", "tenant-b")]) == 1
    # After confirmed parent retention, an empty memory projection cannot
    # resurrect the previous lifetime's approval or audit history.
    recreated = await store.mutate("expired", "tenant-a", 0, lambda state: state, {"action": "new-lifetime"})
    assert recreated == {**empty_state(), "state_version": 1}
    assert len(store.events[("expired", "tenant-a")]) == 1
    assert json.loads(store.events[("expired", "tenant-a")][0])["action"] == "new-lifetime"


async def test_memory_forget_missing_projection_is_idempotent_and_does_not_claim_owner():
    store = MemoryReviewStore()
    await store.forget("absent", "tenant-a")
    await store.forget("absent", "tenant-a")
    assert store.states == store.events == store.owners == {}


async def test_new_postgres_store_instance_observes_committed_review_and_audit(review_store):
    store, job_id = review_store
    if isinstance(store, MemoryReviewStore):
        pytest.skip("durability is a Postgres-only claim; memory reviews have process lifetime")
    approved = await store.mutate(job_id, "tenant-a", 0, _approve, {"action": "approve"})
    await _SharedEngine.dispose(store.database_url)
    restarted = PostgresReviewStore(store.database_url)
    assert await restarted.get(job_id, "tenant-a") == approved
    assert len(await _events(restarted, job_id)) == 1


async def test_delivery_recovery_metadata_and_unknown_fields_survive_legacy_style_mutation(review_store):
    store, job_id = review_store
    delivery = {
        "id": "saved",
        "request_correlation_id": "attempt",
        "request_fingerprint_schema": "tp.photo.delivery_intent.v1",
        "request_fingerprint_sha256": "a" * 64,
        "approval_version": 0,
        "future_optional_field": {"preserve": True},
    }

    def save(state):
        state["deliveries"].append(delivery)
        state["future_projection_field"] = {"preserve": True}
        return state

    first = await store.mutate(
        job_id, "tenant-a", 0, save, {"kind": "delivery", "details": {"request_correlation_id": "attempt"}}
    )
    updated = await store.mutate(job_id, "tenant-a", 1, _approve, {"kind": "decision"})
    assert updated["deliveries"] == first["deliveries"] == [delivery]
    assert updated["future_projection_field"] == {"preserve": True}
    assert json.loads((await _events(store, job_id))[0])["details"]["request_correlation_id"] == "attempt"


def _generated_native_materials_record():
    """Use real native SDR math and all three exact diagnostic products."""
    import io

    import numpy as np
    import tifffile

    from transformation_portal.lux_depth_v4.photography import output_srgb_icc
    from transformation_portal.lux_depth_v6.color import RenderRecipe
    from transformation_portal.photographic_delivery import DeliveryImage
    from transformation_portal.photographic_delivery.materials import MaterialsRecipe, create_materials_revision
    from transformation_portal.photographic_delivery.presentation import PresentationPolicy

    grid = np.indices((64, 96)).sum(axis=0) % 2
    pixels = np.empty((64, 96, 4), dtype=np.uint16)
    pixels[..., :3] = (20000 + grid * 15000)[..., None]
    pixels[..., 3] = 65535
    pixels[0, :, 3] = 0
    pixels[1, :, 3] = 32768
    buffer = io.BytesIO()
    tifffile.imwrite(buffer, pixels, photometric="rgb", extrasamples="unassalpha", iccprofile=output_srgb_icc())
    source = buffer.getvalue()
    source_sha = hashlib.sha256(source).hexdigest()
    image = DeliveryImage(
        "source.tif",
        source_sha,
        "Room",
        source_bytes=source,
        presentation=PresentationPolicy(
            "display_referred", RenderRecipe("soft_srgb", 0.7), "v6_delivery", "b" * 64, source_sha, source_sha
        ),
    )
    recipe = MaterialsRecipe.from_payload(
        {
            "schema": "tp.photo.materials_recipe.v1",
            "input_stage": "display_referred",
            "regions": [
                {"id": f"region-{index}-" + "x" * 55, "label": label, "region": region, "strength": 0.03}
                for index, (label, region) in enumerate(
                    [
                        ("glass", [0, 0, 0.5, 0.5]),
                        ("stone", [0.5, 0, 1, 0.5]),
                        ("water", [0, 0.5, 0.5, 1]),
                        ("foliage", [0.5, 0.5, 1, 1]),
                    ],
                    start=1,
                )
            ],
        }
    )
    result = create_materials_revision(image, recipe)
    identity = "native-materials-store-record"
    record = {
        "id": identity,
        "variant": {
            "artifact_path": f"revisions/{identity}/master.tif",
            "sha256": result["receipt"]["revision_sha256"],
            "size_bytes": len(result["source_bytes"]),
            "source_name": "master.tif",
            "version": "r001",
            "scene_id": "room",
            "label": "Room",
            "role": "final",
            "kind": "review_asset",
            "storage_path": f"photographic-review/{identity}/master.tif",
            "input_color": "auto",
        },
        "receipt": result["receipt"],
        "native_source": {
            "artifact_path": "retained/source.tif",
            "sha256": source_sha,
            "source_identity_sha256": image.source_identity_sha256,
        },
        "native_products": [
            {
                "filename": filename,
                "sha256": hashlib.sha256(data).hexdigest(),
                "size_bytes": len(data),
                "content_type": mime,
                "kind": "review_asset",
                "storage_path": f"photographic-review/{identity}/{filename}",
            }
            for filename, (data, mime) in result["diagnostic_files"].items()
        ],
        "actor": {"username": "synthetic-reviewer", "assurance": "test_authenticated_operator"},
        "created_at": 1.0,
    }
    assert len(record["receipt"]["recipe"]["regions"]) == 4
    assert {product["filename"] for product in record["native_products"]} == {
        "materials-overlay.png",
        "materials-diagnostics.json",
        "materials-preview.png",
    }
    assert len(record["receipt"]["diagnostics"]) == 3
    for product in record["native_products"]:
        diagnostic = next(row for row in record["receipt"]["diagnostics"] if row["filename"] == product["filename"])
        assert all(product[key] == value for key, value in diagnostic.items())
    return record


async def _save_native_materials_record(store, job_id):
    record = _generated_native_materials_record()

    def save(state):
        state["revisions"].append(record)
        return state

    updated = await store.mutate(
        job_id,
        "tenant-a",
        0,
        save,
        {"kind": "revision", "actor": record["actor"], "created_at": 1.0, "details": record},
    )
    return record, updated


async def test_generated_native_materials_receipt_and_three_product_closure_survive_projection_and_audit(review_store):
    store, job_id = review_store
    record, updated = await _save_native_materials_record(store, job_id)
    assert await store.get(job_id, "tenant-a") == updated
    assert updated["revisions"] == [record]
    raw = (await _events(store, job_id))[0]
    audit = json.loads(raw)
    assert raw == canonicalize_json(audit)
    assert audit["details"] == record
    assert audit["projection_sha256"] == hashlib.sha256(canonicalize_json(updated)).hexdigest()
    # A real maximum-four-region receipt (including 64-character region IDs)
    # and all product metadata retain ample space within both store limits.
    assert len(raw) < 128 * 1024 // 2
    assert len(canonicalize_json(updated)) < 2 * 1024 * 1024 // 8
    snapshot_sha = hashlib.sha256(raw).hexdigest()
    changed = await store.mutate(job_id, "tenant-a", 1, _approve, {"kind": "decision"})
    assert changed["revisions"][0] == record
    assert hashlib.sha256((await _events(store, job_id))[0]).hexdigest() == snapshot_sha
    # Detached caller containers cannot mutate a retained receipt or product
    # identity after the atomic projection/event write.
    record["receipt"]["execution"]["status"] = "caller-mutated"
    record["native_products"][0]["sha256"] = "0" * 64
    assert (await store.get(job_id, "tenant-a"))["revisions"] == updated["revisions"]
    assert json.loads((await _events(store, job_id))[0])["details"] == updated["revisions"][0]


async def test_postgres_native_materials_receipt_and_diagnostic_closure_survive_separate_process_restart(review_store):
    from pathlib import Path

    store, job_id = review_store
    if isinstance(store, MemoryReviewStore):
        pytest.skip("durable native Materials review requires the migrated dedicated Postgres lane")
    _, expected = await _save_native_materials_record(store, job_id)
    expected_events = [
        {"sha256": hashlib.sha256(raw).hexdigest(), "payload": json.loads(raw)} for raw in await _events(store, job_id)
    ]
    await _SharedEngine.dispose(store.database_url)
    code = """
import asyncio, hashlib, json, os
from sqlalchemy import select
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.orchestrator.models import PhotographicReviewEventModel
from transformation_portal.orchestrator.photographic_review_store import PostgresReviewStore
from transformation_portal.orchestrator.storage.postgres import _SharedEngine
async def run():
    url, job_id = os.environ['TP_TEST_POSTGRES_URL'], os.environ['TP_REVIEW_TEST_JOB_ID']
    state = await PostgresReviewStore(url).get(job_id, 'tenant-a')
    _, factory = await _SharedEngine.get(url)
    async with factory() as session:
        rows = (await session.scalars(select(PhotographicReviewEventModel)
            .where(PhotographicReviewEventModel.job_id == job_id)
            .order_by(PhotographicReviewEventModel.version))).all()
        events = []
        for row in rows:
            payload = json.loads(row.canonical_bytes)
            assert hashlib.sha256(row.canonical_bytes).hexdigest() == row.sha256
            assert canonicalize_json(payload) == row.canonical_bytes
            events.append({'sha256': row.sha256, 'payload': payload})
    print(json.dumps({'state': state, 'events': events}, sort_keys=True))
    await _SharedEngine.dispose(url)
asyncio.run(run())
"""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        code,
        env={
            **os.environ,
            "TP_REVIEW_TEST_JOB_ID": job_id,
            "PYTHONPATH": os.pathsep.join(
                [str(Path(__file__).resolve().parents[2] / "src"), os.environ.get("PYTHONPATH", "")]
            ),
        },
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await asyncio.wait_for(process.communicate(), timeout=30)
    assert process.returncode == 0, "separate native Materials review process failed"
    assert json.loads(stdout) == {"state": expected, "events": expected_events}


async def test_postgres_delivery_recovery_metadata_survives_separate_process_restart(review_store):
    store, job_id = review_store
    if isinstance(store, MemoryReviewStore):
        pytest.skip("durable delivery recovery requires the migrated Postgres lane")

    def save(state):
        state["deliveries"].append(
            {
                "id": "saved",
                "approval_version": 0,
                "request_correlation_id": "original-attempt",
                "request_fingerprint_schema": "tp.photo.delivery_intent.v1",
                "request_fingerprint_sha256": "a" * 64,
                "future_field": {"preserved": True},
            }
        )
        return state

    expected = await store.mutate(job_id, "tenant-a", 0, save, {"kind": "delivery"})
    code = """
import asyncio, json, os
from transformation_portal.orchestrator.photographic_review_store import PostgresReviewStore
from transformation_portal.orchestrator.storage.postgres import _SharedEngine
async def run():
    url = os.environ['TP_TEST_POSTGRES_URL']
    state = await PostgresReviewStore(url).get(os.environ['TP_REVIEW_TEST_JOB_ID'], 'tenant-a')
    print(json.dumps(state, sort_keys=True))
    await _SharedEngine.dispose(url)
asyncio.run(run())
"""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        code,
        env={**os.environ, "TP_REVIEW_TEST_JOB_ID": job_id},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await asyncio.wait_for(process.communicate(), timeout=30)
    assert process.returncode == 0, "separate delivery-history process failed"
    assert json.loads(stdout) == expected


async def test_postgres_approval_survives_separate_python_process_restart(review_store):
    store, job_id = review_store
    if isinstance(store, MemoryReviewStore):
        pytest.skip("cross-process persistence requires the migrated dedicated Postgres lane")
    approved = await store.mutate(job_id, "tenant-a", 0, _approve, {"action": "approve"})
    code = """
import asyncio, json, os
from transformation_portal.orchestrator.photographic_review_store import PostgresReviewStore
from transformation_portal.orchestrator.storage.postgres import _SharedEngine
async def run():
    url = os.environ['TP_TEST_POSTGRES_URL']
    value = await PostgresReviewStore(url).get(os.environ['TP_REVIEW_TEST_JOB_ID'], 'tenant-a')
    print(json.dumps(value, sort_keys=True))
    await _SharedEngine.dispose(url)
asyncio.run(run())
"""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        code,
        env={**os.environ, "TP_REVIEW_TEST_JOB_ID": job_id},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await asyncio.wait_for(process.communicate(), timeout=30)
    assert process.returncode == 0, "separate review-store process failed"
    assert json.loads(stdout) == approved
    assert len(await _events(store, job_id)) == 1


async def test_memory_event_is_snapshotted_before_waiting_for_writer_lock():
    store = MemoryReviewStore()
    event = {"action": "approve", "notes": {"intent": "original"}}
    await store.lock.acquire()
    task = asyncio.create_task(store.mutate("job", "tenant-a", 0, _approve, event))
    try:
        await asyncio.sleep(0)
        event["notes"]["intent"] = "mutated-while-queued"
    finally:
        store.lock.release()
    await task
    assert json.loads(store.events[("job", "tenant-a")][0])["notes"]["intent"] == "original"


@pytest.mark.parametrize("backend", ["sqlite", "unknown", "postgres"])
async def test_store_factory_fails_closed_without_supported_configuration(monkeypatch, backend):
    monkeypatch.setenv("TP_ORCHESTRATOR_STATE_BACKEND", backend)
    monkeypatch.delenv("TP_DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="configured orchestrator"):
        get_review_store()
