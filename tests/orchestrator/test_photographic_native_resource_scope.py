"""Qualified native costing binds real server authorities, never public claims.

These are isolated contract tests. Runtime matching and the documented prior
cost floor are synthetic; no database, provider, photographic or resource
measurement acceptance is established here.
"""

from __future__ import annotations

import copy
import hashlib
import io
import logging
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
import tifffile

from transformation_portal.api.v1.photographic_review import RevisionRequest
from transformation_portal.lux_depth_v4.color_preparation import output_srgb_icc
from transformation_portal.orchestrator import photographic_admission as admission
from transformation_portal.orchestrator import photographic_review_service as review
from transformation_portal.orchestrator import photographic_review_store as review_store
from transformation_portal.orchestrator.artifact_store.local import LocalArtifactStore
from transformation_portal.orchestrator.artifact_store.s3 import S3ArtifactStore
from transformation_portal.orchestrator.photographic_admission import GIB, ReviewAdmissionConfig, ReviewCost, ReviewOverload
from transformation_portal.orchestrator.photographic_review_access import NativeResourceScope, ReviewContext
from transformation_portal.orchestrator.photographic_review_store import PostgresReviewStore
from transformation_portal.orchestrator.storage.postgres import PostgresJobRepository
from transformation_portal.photographic_delivery import native_resources as resources
from transformation_portal.photographic_delivery.native_resource_inputs import inspect_native_raster
from transformation_portal.portal import photographic_review_access as access

pytestmark = pytest.mark.unit
DATABASE = "postgresql+asyncpg://operator:private-secret@localhost/qualified_test"


@pytest.fixture(name="resource_case")
def _resource_case(monkeypatch, tmp_path):
    monkeypatch.setenv("TP_ORCHESTRATOR_STATE_BACKEND", "postgres")
    monkeypatch.setenv("TP_DATABASE_URL", DATABASE)
    monkeypatch.setenv("TP_PHOTO_REVIEW_NATIVE_MATERIALS_ENABLED", "1")
    monkeypatch.setenv("TP_PHOTO_REVIEW_ENABLED", "1")
    # Runtime matching is synthetic. The source closure and qualification
    # functions remain real, and no test output is qualification evidence.
    runtime = copy.deepcopy(resources.RESOURCE_CONTRACT["runtime"])
    monkeypatch.setattr(resources, "_runtime_identity", lambda: copy.deepcopy(runtime))
    monkeypatch.setattr(resources, "RESOURCE_CONTRACT", resources.capture_contract())
    original_operation_cost = review.operation_cost

    def synthetic_prior_cost(*args, **kwargs):
        cost = original_operation_cost(*args, **kwargs)
        if kwargs.get("native_materials_regions") is not None:
            # Exercise the profile selector without allocating a 30MP raster.
            # Real small-image fallback tests explicitly restore the estimator.
            return replace(cost, memory_bytes=max(cost.memory_bytes, 7 * GIB))
        return cost

    monkeypatch.setattr(review, "operation_cost", synthetic_prior_cost)
    encoded = io.BytesIO()
    tifffile.imwrite(
        encoded,
        np.full((6, 8, 3), 24000, np.uint16),
        photometric="rgb",
        iccprofile=output_srgb_icc(),
        metadata={"color_space": "srgb", "color_domain": "display_encoded_srgb"},
    )
    raw = encoded.getvalue()
    sha = hashlib.sha256(raw).hexdigest()
    evidence = inspect_native_raster(raw, sha)
    assert evidence is not None
    store = LocalArtifactStore(tmp_path / "objects")
    repository = PostgresJobRepository(database_url=DATABASE)
    live = SimpleNamespace(store=store, repository=repository, dispatch=False)
    app = SimpleNamespace(
        get_artifact_store=lambda: live.store,
        _job_repository=lambda: live.repository,
        _configured_job_repository=lambda: live.repository,
        _uses_dispatch_authority=lambda: live.dispatch,
    )
    adapter = access.AppReviewAccess(app)
    job = SimpleNamespace(
        id="job",
        request={},
        effective_request={},
        artifacts={},
        logs_tail=[],
        run_summary={},
        error=None,
        artifact_lookup={},
    )
    ref = {
        "artifact_path": "source/final.tif",
        "source_name": "final.tif",
        "sha256": sha,
        "source_identity_sha256": sha,
        "size_bytes": len(raw),
        "scene_id": "scene",
        "role": "final",
        "kind": "indexed",
        "input_color": "auto",
    }
    scope = adapter._native_resource_scope(job)
    assert isinstance(scope, NativeResourceScope) and scope.store is store
    assert scope() is not None
    context = ReviewContext(
        "job",
        "tenant",
        {"username": "operator"},
        [ref],
        AsyncMock(),
        AsyncMock(),
        revalidate=AsyncMock(),
        binding_identity="b" * 64,
        publication_root=store.root_dir,
        native_resource_scope=scope,
        native_artifact_store=scope.store,
    )
    payload = RevisionRequest(
        expected_version=0,
        artifact_path=ref["artifact_path"],
        sha256=sha,
        source_identity_sha256=sha,
        scene_id="scene",
        recipe={
            "schema": "tp.photo.materials_recipe.v1",
            "input_stage": "display_referred",
            "regions": [
                {"id": f"stone-{index}", "label": "stone", "region": [0, 0, 1, 1], "strength": 0} for index in range(4)
            ],
        },
    )
    state = {"state_version": 0, "decisions": [], "groups": [], "revisions": [], "deliveries": []}
    service = review.PhotographicReviewService(ReviewAdmissionConfig(enabled=True))
    geometry = {"width": evidence.width, "height": evidence.height, "alpha": evidence.alpha}
    service._remember_inspection(context, ref, geometry, evidence)
    case = SimpleNamespace(
        runtime=runtime,
        raw=raw,
        evidence=evidence,
        store=store,
        repository=repository,
        live=live,
        app=app,
        adapter=adapter,
        job=job,
        scope=scope,
        context=context,
        payload=payload,
        state=state,
        service=service,
        geometry=geometry,
        ref=ref,
        tmp_path=tmp_path,
        original_operation_cost=original_operation_cost,
    )

    def cost(*, context=None, payload=None, response_owned=True, state=None):
        context = case.context if context is None else context
        payload = case.payload if payload is None else payload
        state = case.state if state is None else state
        inspections = service._admission_inspections(context, state, "revision", (payload,), {})
        return service._resource_cost(context, state, "revision", (payload,), inspections, response_owned=response_owned)

    case.cost = cost
    return case


def test_actual_postgres_factory_wrappers_share_one_unchanged_durable_scope(resource_case):
    case = resource_case
    assert type(access.get_configured_review_store()) is PostgresReviewStore  # pylint: disable=unidiomatic-typecheck
    assert case.scope() == case.scope() and case.scope.store is case.store
    assert case.cost().native_resource_qualification is not None


def test_qualified_native_profile_is_fixed_response_owned_and_keeps_legacy_floors(resource_case):
    case = resource_case
    before = copy.deepcopy(admission._MEASURED_40MP)
    cost = case.cost()
    assert cost.native_resource_qualification.source_sha256 == case.payload.sha256
    assert cost.native_resource_qualification.raster_identity == case.evidence.content_hash()
    assert cost.memory_bytes == 6 * GIB and cost.max_pixels == 48
    assert cost.decode_slots == cost.io_slots == 1 and not cost.interactive
    assert case.cost(response_owned=False).native_resource_qualification is None
    assert admission._MEASURED_40MP == before
    assert before["revision"] == (3_987_947_520, 6_153_633_792)


def test_private_header_authority_is_not_persisted_or_exposed_as_an_inspection(resource_case):
    case = resource_case
    cached = next(iter(case.service._inspection_costs.values()))
    assert cached["_native_resource_evidence"] is case.evidence
    assert "_native_resource_evidence" not in case.geometry
    decision = {**case.ref, "status": "selected", "inspection": dict(case.geometry)}
    state = {**case.state, "decisions": [decision]}
    public = review.public_state(state, case.context)
    assert "_native_resource_evidence" not in repr(public)
    assert decision["inspection"] == case.geometry


def test_persisted_or_public_evidence_never_substitutes_for_private_cache(resource_case):
    case = resource_case
    case.service._inspection_costs.clear()
    decision = {
        **case.ref,
        "status": "selected",
        "inspection": {**case.geometry, "_native_resource_evidence": case.evidence.to_payload()},
    }
    assert case.cost(state={**case.state, "decisions": [decision]}).native_resource_qualification is None
    assert case.cost().native_resource_qualification is None


@pytest.mark.parametrize("field", ["tenant", "job", "binding", "path", "sha", "identity"])
def test_exact_actor_job_inventory_and_source_selectors_are_required(resource_case, field):
    case = resource_case
    context, payload = case.context, case.payload
    if field == "tenant":
        context = replace(context, tenant_id="other")
    elif field == "job":
        context = replace(context, job_id="other")
    elif field == "binding":
        context = replace(context, binding_identity="c" * 64)
    else:
        attribute = {"path": "artifact_path", "sha": "sha256", "identity": "source_identity_sha256"}[field]
        payload = payload.model_copy(update={attribute: "other/final.tif" if field == "path" else "d" * 64})
    assert case.cost(context=context, payload=payload).native_resource_qualification is None


@pytest.mark.parametrize("change", ["hash", "width", "height", "alpha", "composite"])
def test_mismatched_header_inspection_cannot_enter_private_qualified_cache(resource_case, change):
    case = resource_case
    inspection, evidence, ref = dict(case.geometry), case.evidence, dict(case.ref)
    if change == "hash":
        evidence = replace(evidence, source_sha256="e" * 64)
    elif change == "composite":
        ref["source_identity_sha256"] = "e" * 64
    else:
        inspection[change] = not inspection[change] if change == "alpha" else inspection[change] + 1
    case.service._inspection_costs.clear()
    case.service._remember_inspection(case.context, ref, inspection, evidence)
    assert all("_native_resource_evidence" not in row for row in case.service._inspection_costs.values())
    assert case.cost().native_resource_qualification is None


def test_runtime_drift_discards_qualified_profile(resource_case):
    case = resource_case
    case.runtime["version"] = "unqualified runtime change"
    assert resources.runtime_profile_identity() is None
    assert case.cost().native_resource_qualification is None


@pytest.mark.parametrize("kind", ["callable", "different_store", "store_subclass"])
def test_scope_capability_must_bind_the_exact_qualified_store(resource_case, kind):
    case = resource_case
    if kind == "callable":
        context = replace(case.context, native_resource_scope=case.scope.revalidate)
    else:
        if kind == "store_subclass":

            class DerivedStore(LocalArtifactStore):
                pass

            store = DerivedStore(case.tmp_path / "derived")
        else:
            store = LocalArtifactStore(case.tmp_path / "other-store")
        context = replace(case.context, native_artifact_store=store)
    assert case.cost(context=context).native_resource_qualification is None


@pytest.mark.parametrize("kind", ["s3", "local_subclass", "repository_subclass", "review_subclass"])
def test_unsupported_provider_or_durable_classes_cannot_create_scope(resource_case, monkeypatch, kind):
    case = resource_case
    if kind == "s3":
        case.live.store = S3ArtifactStore(bucket="unit-fixture", client=SimpleNamespace())
    elif kind == "local_subclass":

        class DerivedStore(LocalArtifactStore):
            pass

        case.live.store = DerivedStore(case.tmp_path / "derived")
    elif kind == "repository_subclass":

        class DerivedRepository(PostgresJobRepository):
            pass

        case.live.repository = DerivedRepository(database_url=DATABASE)
    else:

        class DerivedReviewStore(PostgresReviewStore):
            pass

        monkeypatch.setattr(access, "get_configured_review_store", lambda: DerivedReviewStore(DATABASE))
    assert case.adapter._native_resource_scope(case.job) is None


@pytest.mark.parametrize("change", ["store", "repository", "review_database", "repository_database", "dispatch", "root"])
def test_scope_detects_backing_authority_or_root_inode_drift(resource_case, monkeypatch, change):
    case = resource_case
    if change == "store":
        case.live.store = LocalArtifactStore(case.tmp_path / "other-store")
    elif change == "repository":
        case.live.repository = PostgresJobRepository(database_url=DATABASE)
    elif change == "review_database":
        monkeypatch.setenv("TP_DATABASE_URL", DATABASE + "_changed")
    elif change == "repository_database":
        case.repository._database_url = DATABASE + "_changed"
    elif change == "dispatch":
        case.live.dispatch = not case.live.dispatch
    else:
        # Preserve every old object/file under another name; delete nothing.
        case.store.root_dir.rename(case.tmp_path / "old-objects")
        case.store.root_dir.mkdir()
    assert case.scope() is None
    assert case.cost().native_resource_qualification is None


def test_initial_review_and_job_database_mismatch_cannot_create_scope(resource_case):
    case = resource_case
    case.live.repository = PostgresJobRepository(database_url=DATABASE + "_different")
    assert case.adapter._native_resource_scope(case.job) is None


def test_canonical_dispatch_keeps_generic_native_cost_without_a_qualified_scope(resource_case, monkeypatch):
    case = resource_case
    case.live.dispatch = True
    assert case.adapter._native_resource_scope(case.job) is None
    context = replace(case.context, native_resource_scope=None, native_artifact_store=None)
    monkeypatch.setattr(review, "operation_cost", case.original_operation_cost)
    cost = case.cost(context=context)
    expected = case.original_operation_cost(
        "revision", case.service.admission.config, [case.geometry], native_materials_regions=4
    )
    assert cost.native_resource_qualification is None
    assert cost.memory_bytes == expected.memory_bytes < 2 * GIB


@pytest.mark.parametrize("field", ["request", "artifact_lookup"])
def test_scope_rechecks_bounded_job_metadata(resource_case, field):
    case = resource_case
    if field == "request":
        case.job.request = {"over_limit": "x" * resources.MAX_STATE_BYTES}
    else:
        case.job.artifact_lookup = {str(index): Path("/unit") for index in range(1025)}
    assert case.scope() is None
    assert case.cost().native_resource_qualification is None


def test_over_thirty_megapixels_retains_generic_capacity_rejection(resource_case):
    case = resource_case
    pixels = 6001 * 5000
    # Synthetic corrupted cache geometry only; no large image or acceptance.
    evidence = replace(
        case.evidence,
        source_width=6001,
        source_height=5000,
        width=6001,
        height=5000,
        raster_bytes=pixels * 6,
        size_bytes=case.evidence.header_bytes + pixels * 6,
    )
    case.service._remember_inspection(case.context, case.ref, {"width": 6001, "height": 5000, "alpha": False}, evidence)
    cost = case.cost()
    assert cost.native_resource_qualification is None and cost.memory_bytes > 6 * GIB
    with pytest.raises(ReviewOverload) as error:
        case.service.admission._validate_cost(cost)
    assert error.value.status == 413


@pytest.mark.asyncio
async def test_qualified_memory_never_expands_smaller_limits_or_inspection_headroom(resource_case):
    case = resource_case
    config = ReviewAdmissionConfig(enabled=True, memory_bytes=10 * GIB, bulk_memory_bytes=6 * GIB)
    controller = admission.PhotographicAdmissionController(config)
    try:
        cost = case.cost()
        assert cost.memory_bytes == 6 * GIB and controller.interactive_reserve == 6 * GIB
        with pytest.raises(ReviewOverload) as error:
            await controller.acquire("tenant", cost)
        assert error.value.status == 413
        assert controller.config.memory_bytes == 10 * GIB
        assert controller.interactive_reserve == 6 * GIB and controller.snapshot()["active"] == 0
    finally:
        await controller.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("memory_gib,bulk_gib", [(12, 8), (10, 6), (8, 4)])
async def test_real_small_image_keeps_its_lower_generic_cost_and_remains_admissible(
    resource_case, monkeypatch, memory_gib, bulk_gib
):
    case = resource_case
    monkeypatch.setattr(review, "operation_cost", case.original_operation_cost)
    assert review.operation_cost is case.original_operation_cost
    expected = case.original_operation_cost(
        "revision",
        case.service.admission.config,
        [case.geometry],
        native_materials_regions=4,
    )
    cost = case.cost()
    assert cost.native_resource_qualification is None
    assert cost.memory_bytes == expected.memory_bytes < 2 * GIB
    config = ReviewAdmissionConfig(enabled=True, memory_bytes=memory_gib * GIB, bulk_memory_bytes=bulk_gib * GIB)
    controller = admission.PhotographicAdmissionController(config)
    lease = None
    try:
        lease = await controller.acquire("tenant", cost)
        assert lease.cost is cost
        assert controller.interactive_reserve == 6 * GIB
        assert controller.config.memory_bytes == memory_gib * GIB
    finally:
        if lease is not None:
            await lease.release()
        await controller.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["scope", "runtime"])
async def test_native_lease_recompute_rejects_drift_before_materialization(resource_case, change):
    case = resource_case
    lease = await case.service.admission.acquire("tenant", case.cost())
    lease.native_response_owned = True
    lease.native_artifact_store = case.store
    token = admission.current_review_lease.set(lease)
    try:
        if change == "scope":
            case.live.store = LocalArtifactStore(case.tmp_path / "other-store")
        else:
            case.runtime["version"] = "drifted"
        with pytest.raises(ReviewOverload) as error:
            case.service._require_native_materials_lease(case.context, case.state, case.payload)
        assert error.value.status == 413 and error.value.reason == "native_resource_scope_changed"
        case.context.materialize.assert_not_awaited()
    finally:
        admission.current_review_lease.reset(token)
        await lease.release()
        await case.service.close()


@pytest.mark.asyncio
async def test_actual_app_uses_pinned_store_across_factory_swaps_and_clears_it_on_release(resource_case, monkeypatch):
    import app as app_module

    case = resource_case
    replacement = LocalArtifactStore(case.tmp_path / "replacement")
    monkeypatch.setattr(app_module, "get_artifact_store", lambda: replacement)
    lease = await case.service.admission.acquire("tenant", case.cost())
    lease.native_artifact_store = case.store
    token = admission.current_review_lease.set(lease)
    try:
        assert app_module._artifact_store() is case.store
        await lease.release()
        assert lease.native_artifact_store is None
        with pytest.raises(RuntimeError, match="authority"):
            app_module._artifact_store()
    finally:
        admission.current_review_lease.reset(token)
        await lease.release()
        await case.service.close()
    assert app_module._artifact_store() is replacement


@pytest.mark.asyncio
async def test_actual_app_rejects_qualified_lease_without_pinned_store(resource_case):
    import app as app_module

    case = resource_case
    lease = await case.service.admission.acquire("tenant", case.cost())
    token = admission.current_review_lease.set(lease)
    try:
        with pytest.raises(RuntimeError, match="authority"):
            app_module._artifact_store()
    finally:
        admission.current_review_lease.reset(token)
        await lease.release()
        await case.service.close()


@pytest.mark.asyncio
async def test_unqualified_lease_keeps_existing_artifact_factory_behaviour(resource_case, monkeypatch):
    import app as app_module

    case = resource_case
    replacement = LocalArtifactStore(case.tmp_path / "replacement")
    monkeypatch.setattr(app_module, "get_artifact_store", lambda: replacement)
    lease = await case.service.admission.acquire("tenant", ReviewCost(1, 0))
    lease.native_artifact_store = case.store
    token = admission.current_review_lease.set(lease)
    try:
        assert app_module._artifact_store() is replacement
    finally:
        admission.current_review_lease.reset(token)
        await lease.release()
        await case.service.close()


@pytest.mark.asyncio
async def test_qualified_lease_pins_all_authorities_through_global_configuration_drift(resource_case, monkeypatch):
    import app as app_module

    case = resource_case
    replacement_store = LocalArtifactStore(case.tmp_path / "replacement")
    replacement_repository = PostgresJobRepository(database_url=DATABASE + "_changed")
    replacement_reviews = PostgresReviewStore(DATABASE + "_changed")
    monkeypatch.setattr(app_module, "get_artifact_store", lambda: replacement_store)
    monkeypatch.setattr(app_module, "_configured_job_repository", lambda: replacement_repository)
    monkeypatch.setattr(review_store, "get_configured_review_store", lambda: replacement_reviews)
    lease = await case.service.admission.acquire("tenant", case.cost())
    lease.native_artifact_store = case.scope.store
    lease.native_review_store = case.scope.review_store
    lease.native_job_repository = case.scope.job_repository
    token = admission.current_review_lease.set(lease)
    try:
        assert app_module._artifact_store() is case.store
        assert app_module._job_repository() is case.repository
        assert review_store.get_review_store() is case.scope.review_store
        await lease.release()
        assert lease.native_artifact_store is lease.native_review_store is lease.native_job_repository is None
        for getter in (app_module._artifact_store, app_module._job_repository, review_store.get_review_store):
            with pytest.raises(RuntimeError, match="authority"):
                getter()
    finally:
        admission.current_review_lease.reset(token)
        await lease.release()
        await case.service.close()
    assert app_module._artifact_store() is replacement_store
    assert app_module._job_repository() is replacement_repository
    assert review_store.get_review_store() is replacement_reviews


@pytest.mark.asyncio
@pytest.mark.parametrize("authority", ["review", "job"])
async def test_qualified_lease_cannot_fall_back_when_a_durable_pin_is_missing(resource_case, authority):
    import app as app_module

    case = resource_case
    lease = await case.service.admission.acquire("tenant", case.cost())
    token = admission.current_review_lease.set(lease)
    getter = app_module._job_repository if authority == "job" else review_store.get_review_store
    try:
        with pytest.raises(RuntimeError, match="authority"):
            getter()
    finally:
        admission.current_review_lease.reset(token)
        await lease.release()
        await case.service.close()


@pytest.mark.asyncio
async def test_context_publication_filesystem_comes_from_captured_store_not_a_fresh_factory(resource_case, monkeypatch):
    case = resource_case
    replacement = LocalArtifactStore(case.tmp_path / "other-filesystem-authority")
    factory_calls = 0

    def changing_factory():
        nonlocal factory_calls
        factory_calls += 1
        # The first two calls capture and validate the capsule. A third fresh
        # lookup would choose another publication root without changing its pin.
        return replacement if factory_calls == 3 else case.store

    case.app.get_artifact_store = case.app._artifact_store = changing_factory
    case.app._cleanup_expired_jobs = AsyncMock()
    case.app._now = lambda: 0
    case.app.PILOT_CONTROL_PLANE_ENABLED = False
    case.app._job_tenant_id = lambda job: "tenant"
    case.app.LOGGER = logging.getLogger(__name__)
    monkeypatch.setattr(case.adapter, "_job", AsyncMock(return_value=(case.job, None)))
    monkeypatch.setattr(case.adapter, "_binding_identity", AsyncMock(return_value=case.context.binding_identity))
    context = await case.adapter.authorize_admission(None, case.job.id)
    assert isinstance(context, ReviewContext)
    assert context.native_artifact_store is case.store
    assert context.publication_root == context.native_artifact_store.root_dir


@pytest.mark.asyncio
async def test_scope_drift_after_successful_publication_cleans_only_new_owned_key(resource_case, monkeypatch):
    case = resource_case
    storage_path = "photographic-review/revisions/private/output.tif"
    source = case.tmp_path / "candidate.tif"
    source.write_bytes(case.raw)
    case.app._artifact_store = lambda: case.store
    case.app._artifact_storage_job_id = lambda job: job.id
    case.app.LOGGER = logging.getLogger(__name__)
    monkeypatch.setattr(case.adapter, "_job", AsyncMock(return_value=case.job))
    original_write = case.store.write_immutable_file

    async def write_then_drift(*args, **kwargs):
        await original_write(*args, **kwargs)
        case.live.store = LocalArtifactStore(case.tmp_path / "changed-authority")

    monkeypatch.setattr(case.store, "write_immutable_file", write_then_drift)
    expected_scope = case.scope()
    with pytest.raises(access.ReviewAccessError, match="resource backing changed"):
        await case.adapter._publish(
            None,
            case.job,
            storage_path,
            source,
            "image/tiff",
            resource_scope=case.scope,
            expected_resource_scope=expected_scope,
        )
    assert case.scope() is None
    assert not (case.store.root_dir / case.job.id / storage_path).exists()


@pytest.mark.asyncio
async def test_publication_collision_preserves_preexisting_retained_bytes(resource_case, monkeypatch):
    case = resource_case
    storage_path = "photographic-review/revisions/retained/output.tif"
    source = case.tmp_path / "candidate.tif"
    source.write_bytes(case.raw)
    await case.store.write_bytes(case.job.id, storage_path, b"retained authority", content_type="image/tiff")
    case.app._artifact_store = lambda: case.store
    case.app._artifact_storage_job_id = lambda job: job.id
    case.app.LOGGER = logging.getLogger(__name__)
    monkeypatch.setattr(case.adapter, "_job", AsyncMock(return_value=case.job))
    with pytest.raises(access.ReviewAccessError, match="publication unavailable"):
        await case.adapter._publish(
            None,
            case.job,
            storage_path,
            source,
            "image/tiff",
            resource_scope=case.scope,
            expected_resource_scope=case.scope(),
        )
    assert (case.store.root_dir / case.job.id / storage_path).read_bytes() == b"retained authority"
