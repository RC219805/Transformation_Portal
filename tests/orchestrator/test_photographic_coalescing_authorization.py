"""A shared photographic read never shares its subscribers' authorization."""

from __future__ import annotations

import asyncio
import hashlib
import io
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from transformation_portal.orchestrator import photographic_admission as admission
from transformation_portal.orchestrator import photographic_review_service as review
from transformation_portal.orchestrator.photographic_admission import GIB, ReviewAdmissionConfig, current_review_lease
from transformation_portal.orchestrator.photographic_review_access import ReviewAccessError, ReviewContext

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
async def shared_render(monkeypatch):
    """Use real source spooling/identity checks, with a controlled render worker."""
    buffer = io.BytesIO()
    with Image.new("RGB", (8, 6), "white") as image:
        image.save(buffer, format="PNG")
    source = buffer.getvalue()
    sha = hashlib.sha256(source).hexdigest()
    ref = {
        "artifact_path": "input-0000/final.png",
        "source_name": "final.png",
        "scene_id": "input-0000",
        "sha256": sha,
        "input_color": "srgb",
    }
    monkeypatch.setattr(admission.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * GIB))
    service = review.PhotographicReviewService(ReviewAdmissionConfig(enabled=True))
    monkeypatch.setattr(service, "state", AsyncMock(return_value={"state_version": 1, "decisions": [], "revisions": []}))
    case = SimpleNamespace(
        service=service,
        source=ref,
        revoked=set(),
        auth_status=403,
        revoke_on_read=None,
        binding_changed=False,
        wrong_hash=False,
        require_lease=True,
        block_at="display",
        io_started=asyncio.Event(),
        io_release=asyncio.Event(),
        read_barriers={},
        initial_lease=None,
        display_started=threading.Event(),
        display_release=threading.Event(),
        display_finished=threading.Event(),
        discoveries=[],
        reads=[],
        computations=[],
        auth_checks=[],
        tasks=[],
    )

    def check_access(actor):
        case.auth_checks.append(actor)
        if actor in case.revoked:
            raise ReviewAccessError("Actor access revoked", status=case.auth_status, code="FORBIDDEN")
        if case.binding_changed:
            raise ReviewAccessError("Source binding changed")

    def context(actor):
        async def revalidate():
            check_access(actor)

        async def discover():
            case.discoveries.append(actor)
            if actor == "leader":
                case.initial_lease = current_review_lease.get()
            elif case.initial_lease is not None:
                assert case.initial_lease.released, "a private fallback cannot overlap the previous shared owner"
            if actor == "leader" and case.block_at == "discovery":
                case.io_started.set()
                await case.io_release.wait()
            check_access(actor)
            return [dict(ref)]

        async def materialize(selected, path, limit):
            case.reads.append(actor)
            assert not case.require_lease or current_review_lease.get() is not None
            if actor == "leader" and case.block_at == "materialize":
                case.io_started.set()
                await case.io_release.wait()
            if actor in case.read_barriers:
                started, release = case.read_barriers[actor]
                started.set()
                await release.wait()
            if actor == case.revoke_on_read:
                case.revoked.add(actor)
            check_access(actor)
            assert selected["sha256"] == sha and len(source) <= limit
            path.write_bytes(source)
            return {"sha256": "f" * 64 if case.wrong_hash else sha, "size_bytes": len(source)}

        return ReviewContext(
            job_id="job",
            tenant_id="tenant",
            actor={"username": actor},
            references=[dict(ref)],
            materialize=materialize,
            publish=AsyncMock(side_effect=AssertionError("A render cannot publish")),
            revalidate=revalidate,
            discover_references=discover,
            binding_identity="same-verified-source-binding",
        )

    def display(*_args):
        lease = current_review_lease.get()
        assert not case.require_lease or lease is not None
        assert lease is None or not lease.released
        case.computations.append(lease)
        case.display_started.set()
        try:
            assert case.display_release.wait(5)
            assert lease is None or not lease.released
            return b"verified-pixels", 8, 6
        finally:
            case.display_finished.set()

    monkeypatch.setattr(review, "_display", display)
    case.context = context

    def start(actor):
        task = asyncio.create_task(service.execute(context(actor), service.render, ref["artifact_path"], sha))
        case.tasks.append(task)
        return task

    case.start = start
    try:
        yield case
    finally:
        case.io_release.set()
        for _started, release in case.read_barriers.values():
            release.set()
        case.display_release.set()
        await asyncio.gather(*case.tasks, return_exceptions=True)
        await service.close()


async def join_subscribers(case):
    leader = case.start("leader")
    if case.block_at == "display":
        assert await asyncio.to_thread(case.display_started.wait, 2)
    else:
        await asyncio.wait_for(case.io_started.wait(), 2)
    follower = case.start("follower")
    for _ in range(10):
        if case.service.admission.snapshot()["coalesced"]:
            break
        await asyncio.sleep(0)
    assert case.service.admission.snapshot()["coalesced"] == 1
    return leader, follower


def assert_quiescent(case):
    snapshot = case.service.admission.snapshot()
    assert snapshot["active"] == snapshot["queued"] == snapshot["outstanding"] == 0
    assert all(lease is None or lease.released for lease in case.computations)


async def test_revoked_leader_cannot_poison_authorized_follower_after_shared_computation(shared_render):
    case = shared_render
    leader, follower = await join_subscribers(case)
    case.revoked.add("leader")
    case.display_release.set()
    with pytest.raises(ReviewAccessError, match="Actor access revoked"):
        await leader
    response = await follower
    assert response.body == b"verified-pixels"
    assert response.headers["X-TP-Source-SHA256"] == case.source["sha256"]
    assert len(case.computations) == 1
    assert case.reads == case.discoveries == ["leader"]
    assert case.auth_checks.count("follower") == 2
    assert_quiescent(case)


@pytest.mark.parametrize("block_at", ["discovery", "materialize"])
@pytest.mark.parametrize("auth_status", [401, 403])
async def test_early_leader_auth_loss_allows_one_fresh_leased_read_under_authorized_follower(
    shared_render, block_at, auth_status
):
    case = shared_render
    case.block_at = block_at
    case.auth_status = auth_status
    leader, follower = await join_subscribers(case)
    case.revoked.add("leader")
    case.display_release.set()
    case.io_release.set()
    with pytest.raises(ReviewAccessError, match="Actor access revoked"):
        await leader
    assert (await follower).body == b"verified-pixels"
    assert case.discoveries == ["leader", "follower"]
    assert case.reads == (["follower"] if block_at == "discovery" else ["leader", "follower"])
    assert len(case.computations) == 1
    assert case.service.admission.snapshot()["admitted"] == 2
    assert_quiescent(case)


async def test_a_second_authorization_failure_during_follower_read_is_not_retried(shared_render):
    case = shared_render
    case.block_at = "materialize"
    leader, follower = await join_subscribers(case)
    case.revoked.add("leader")
    case.revoke_on_read = "follower"
    case.io_release.set()
    outcomes = await asyncio.gather(leader, follower, return_exceptions=True)
    assert all(isinstance(error, ReviewAccessError) and error.status == 403 for error in outcomes)
    assert case.reads == ["leader", "follower"] and not case.computations
    assert case.service.admission.snapshot()["admitted"] == 2
    assert_quiescent(case)


async def test_private_fallback_survives_sequential_revocation_of_two_other_subscribers(shared_render):
    case = shared_render
    case.block_at = "materialize"
    case.read_barriers = {actor: (asyncio.Event(), asyncio.Event()) for actor in ["follower", "survivor"]}
    leader, follower = await join_subscribers(case)
    survivor = case.start("survivor")
    for _ in range(10):
        if case.service.admission.snapshot()["coalesced"] == 2:
            break
        await asyncio.sleep(0)
    assert case.service.admission.snapshot()["coalesced"] == 2
    case.revoked.add("leader")
    case.io_release.set()
    # Both fallback readers must own their own authorization and full vector;
    # neither may subscribe to another actor's second shared attempt.
    await asyncio.wait_for(case.read_barriers["follower"][0].wait(), 2)
    await asyncio.wait_for(case.read_barriers["survivor"][0].wait(), 2)
    assert case.service.admission.snapshot()["active"] == 2
    case.revoked.add("follower")
    for _started, release in case.read_barriers.values():
        release.set()
    case.display_release.set()
    outcomes = await asyncio.gather(leader, follower, survivor, return_exceptions=True)
    assert all(isinstance(error, ReviewAccessError) and error.status == 403 for error in outcomes[:2])
    assert outcomes[2].body == b"verified-pixels"
    assert sorted(case.reads) == ["follower", "leader", "survivor"]
    assert len(case.computations) == 1
    assert case.service.admission.snapshot()["admitted"] == 3
    assert case.service.admission.snapshot()["coalesced"] == 2, "only the original attempt was shared"
    assert_quiescent(case)


@pytest.mark.parametrize("revoked", [{"follower"}, {"leader", "follower"}])
async def test_each_revoked_subscriber_is_denied_even_when_shared_pixels_exist(shared_render, revoked):
    case = shared_render
    tasks = await join_subscribers(case)
    case.revoked.update(revoked)
    case.display_release.set()
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)
    for actor, outcome in zip(["leader", "follower"], outcomes):
        if actor in revoked:
            assert isinstance(outcome, ReviewAccessError) and outcome.status == 403
        else:
            assert outcome.body == b"verified-pixels"
    assert len(case.computations) == 1
    assert_quiescent(case)


async def test_changed_source_binding_denies_every_subscriber_without_retry(shared_render):
    case = shared_render
    tasks = await join_subscribers(case)
    case.binding_changed = True
    case.display_release.set()
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)
    assert all(isinstance(error, ReviewAccessError) and error.status == 409 for error in outcomes)
    assert len(case.computations) == 1
    assert case.service.admission.snapshot()["admitted"] == 1
    assert_quiescent(case)


async def test_source_integrity_failure_does_not_retry_under_another_actor(shared_render):
    case = shared_render
    case.block_at = "materialize"
    leader, follower = await join_subscribers(case)
    case.wrong_hash = True
    case.io_release.set()
    outcomes = await asyncio.gather(leader, follower, return_exceptions=True)
    assert all(isinstance(error, ReviewAccessError) and error.status == 409 for error in outcomes)
    assert case.reads == ["leader"] and not case.computations
    assert case.service.admission.snapshot()["admitted"] == 1
    assert_quiescent(case)


@pytest.mark.parametrize("cancel_actor", ["leader", "follower"])
async def test_canceling_a_subscriber_retains_shared_worker_ownership_and_independent_auth(shared_render, cancel_actor):
    case = shared_render
    leader, follower = await join_subscribers(case)
    canceled, retained = (leader, follower) if cancel_actor == "leader" else (follower, leader)
    canceled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await canceled
    assert case.service.admission.snapshot()["active"] == 1
    assert not case.computations[0].released and not case.display_finished.is_set()
    case.revoked.add(cancel_actor)
    case.display_release.set()
    assert (await retained).body == b"verified-pixels"
    assert len(case.computations) == 1
    assert_quiescent(case)


async def test_direct_render_keeps_its_final_actor_check(shared_render):
    case = shared_render
    # Direct/disabled callers retain their normal gate and final auth boundary.
    case.require_lease = False
    context = case.context("leader")
    task = asyncio.create_task(case.service.render(context, case.source["artifact_path"], case.source["sha256"]))
    case.tasks.append(task)
    assert await asyncio.to_thread(case.display_started.wait, 2)
    case.revoked.add("leader")
    case.display_release.set()
    with pytest.raises(ReviewAccessError, match="Actor access revoked"):
        await task
    assert len(case.computations) == 1
    assert_quiescent(case)
