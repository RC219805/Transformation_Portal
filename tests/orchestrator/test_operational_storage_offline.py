"""Offline transaction and rejection contracts for durable dispatch storage.

Scripted SQLAlchemy results exercise Python transaction orchestration; real
Postgres tests remain authoritative for SQL isolation, locks, and triggers.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from PIL import Image

from tests.core.test_execution_plan import _valid_payload
from transformation_portal.core.execution_plan import CanonicalExecutionPlan
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v5.lifecycle import LuxDepthV5Request, prepare
from transformation_portal.orchestrator.artifact_store.generation import GenerationPublisher
from transformation_portal.orchestrator.dispatch import DispatchFence, DispatchLocator
from transformation_portal.orchestrator.models import (
    AdmissionCapacityModel,
    DispatchAttemptModel,
    JobEventModel,
    JobModel,
    OperationalOutboxModel,
    OperationalRecordModel,
)
from transformation_portal.orchestrator.photography_adapter import PhotographyBindings
from transformation_portal.orchestrator.storage.base import JobRecord, RepositoryError
from transformation_portal.orchestrator.storage.operational import (
    AdmissionRejected,
    DispatchAuthorityLost,
    PostgresOperationalRecordStore,
)

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class Result:
    def __init__(self, rows=(), value=41):
        self.rows, self.value = list(rows), value

    def scalar_one(self):
        return self.value

    def scalars(self):
        return self

    def all(self):
        return self.rows

    def one_or_none(self):
        return self.rows[0] if self.rows else None


class Session:
    def __init__(self, scalars=(), gets=(), rows=()):
        self.scalar_values, self.get_values = deque(scalars), deque(gets)
        self.rows = list(rows)
        self.trace, self.added = [], []

    @asynccontextmanager
    async def begin(self):
        self.trace.append("BEGIN")
        try:
            yield
        except BaseException:
            self.trace.append("ROLLBACK")
            raise
        else:
            self.trace.append("COMMIT")

    async def scalar(self, statement):
        self.trace.append(str(statement.compile(compile_kwargs={"literal_binds": True})))
        assert self.scalar_values, f"unexpected scalar query: {statement}"
        return self.scalar_values.popleft()

    async def scalars(self, statement):
        self.trace.append(str(statement))
        return Result(self.rows)

    async def get(self, model, key):
        self.trace.append(f"GET {model.__tablename__} {key}")
        assert self.get_values, "unexpected get"
        return self.get_values.popleft()

    async def execute(self, statement):
        self.trace.append(str(statement))
        return Result(self.rows)

    def add(self, model):
        self.trace.append(f"ADD {model.__tablename__}")
        self.added.append(model)

    async def flush(self):
        self.trace.append("FLUSH")


def _store(session):
    store = PostgresOperationalRecordStore(database_url="postgresql+asyncpg://unused/unused")

    @asynccontextmanager
    async def local_session():
        yield session

    store._session = local_session
    return store


def _plan():
    return CanonicalExecutionPlan.from_payload(_valid_payload()).to_canonical_json().encode()


def _photography_authority(tmp_path):
    source = tmp_path / "photographs"
    source.mkdir()
    Image.new("RGB", (3, 2)).save(source / "image.png")
    prepared = prepare(
        LuxDepthV5Request(source, tmp_path / "output", input_color="srgb"),
        publisher=GenerationPublisher(artifact_store=None, record_store=None),
    )
    # Operational storage must validate bindings before the worker's filesystem
    # is mounted. Runtime and source existence are execution-time checks.
    bindings = PhotographyBindings.from_payload(
        {
            "schema": "tp.job.photography.bindings.v1",
            "input_root": "/not/mounted/worker/inputs",
            "runtime_python": "/not/mounted/worker/runtime/bin/python",
            "raw_python": None,
            "cache_root": None,
            "companion_root": None,
            "materials_root": None,
        }
    )
    return prepared.canonical_plan_bytes, bindings.canonical_bytes


def _record():
    return JobRecord(
        id="job_offline",
        created_at=1.0,
        request={"args": {"nested": [1]}},
        effective_request={"tenant_id": "tenant", "args": {}},
        artifacts={"items": []},
    )


def _attempt(**overrides):
    values = dict(
        job_id="job_offline",
        attempt_id="attempt",
        dispatch_id="dispatch",
        tenant_id="tenant",
        plan_digest=hashlib.sha256(_plan()).hexdigest(),
        execution_bindings=None,
        execution_bindings_digest=None,
        api_version="v1",
        state="running",
        holder="worker",
        lease_epoch=7,
        lease_valid_until=200.0,
        admitted_at=1.0,
        output_root="/outputs/.tp-attempts/job_offline",
        requested_output_root="/outputs",
        generation_id=None,
        manifest_digest=None,
        cleaned_at=None,
        cleanup_checked_at=None,
    )
    return DispatchAttemptModel(**{**values, **overrides})


def _fence(attempt=None):
    attempt = attempt or _attempt()
    return DispatchFence(
        PostgresOperationalRecordStore._locator(attempt),
        "worker",
        7,
        200.0,
        attempt.output_root,
        attempt.requested_output_root,
    )


def _capacities(active=1):
    return [AdmissionCapacityModel(scope=scope, limit=2, active=active) for scope in ("global", "tenant:tenant")]


def _job(**overrides):
    return JobModel(
        **{
            **dict(
                id="job_offline",
                created_at=1.0,
                state="running",
                progress=0,
                artifacts={"items": []},
                run_summary={},
                error=None,
            ),
            **overrides,
        }
    )


async def _admit(store, **overrides):
    values = dict(
        record=_record(),
        plan_bytes=_plan(),
        tenant_id="tenant",
        output_root="/outputs/attempt",
        requested_output_root="/outputs",
        global_limit=2,
        tenant_limit=2,
    )
    return await store.admit(**{**values, **overrides})


@pytest.mark.parametrize("invalid", ["noncanonical", "not_queued", "tenant_mismatch", "relative", "nul"])
async def test_invalid_admission_never_opens_transaction(invalid):
    session = Session()
    values = {}
    if invalid == "noncanonical":
        values["plan_bytes"] = _plan() + b"\n"
    elif invalid in {"not_queued", "tenant_mismatch"}:
        record = _record()
        if invalid == "not_queued":
            record.state = "running"
        else:
            record.effective_request["tenant_id"] = "other"
        values["record"] = record
    else:
        values["output_root"] = "relative" if invalid == "relative" else "/outputs/\x00"
    with pytest.raises(RepositoryError):
        await _admit(_store(session), **values)
    assert session.trace == []


@pytest.mark.parametrize("limit", [0, -1, True])
async def test_capacity_limits_fail_closed_before_reservation(limit):
    session = Session()
    with pytest.raises(RepositoryError, match="positive"):
        await _admit(_store(session), global_limit=limit)
    assert session.trace == ["BEGIN", "ROLLBACK"]


@pytest.mark.parametrize("case", ["missing", "conflict", "global_full", "tenant_full", "no_clock", "digest_collision"])
async def test_admission_failures_roll_back_without_outbox(case):
    capacities = _capacities(active=0)
    if case == "missing":
        scalars = [None]
    elif case == "conflict":
        capacities[0].limit = 3
        scalars = [capacities[0]]
    else:
        if case.endswith("full"):
            capacities[0 if case == "global_full" else 1].active = 2
        scalars = [*capacities, None if case == "no_clock" else 100.0]
    session = Session(scalars=scalars, gets=[SimpleNamespace(canonical_bytes=b"collision")])
    expected = AdmissionRejected if case.endswith("full") else RepositoryError
    with pytest.raises(expected):
        await _admit(_store(session))
    assert session.trace[-1] == "ROLLBACK"
    assert not session.added


async def test_admission_reserves_both_scopes_before_atomic_copied_record_and_outbox():
    capacities = _capacities(active=0)
    session = Session(scalars=[*capacities, 100.0], gets=[SimpleNamespace(canonical_bytes=_plan())])
    record = _record()
    locator = await _admit(_store(session), record=record)
    record.request["args"]["nested"].append(2)
    job = next(model for model in session.added if isinstance(model, JobModel))
    assert job.request["args"]["nested"] == [1]
    assert [cap.active for cap in capacities] == [1, 1]
    outbox = next(model for model in session.added if isinstance(model, OperationalOutboxModel))
    assert DispatchLocator.from_json(json.dumps(outbox.payload)) == locator
    evidence = next(model for model in session.added if isinstance(model, OperationalRecordModel))
    assert hashlib.sha256(evidence.canonical_bytes).hexdigest() == evidence.digest
    locks = [query for query in session.trace if "FOR UPDATE" in query]
    assert len(locks) == 2
    assert "= 'global'" in locks[0] and "= 'tenant:tenant'" in locks[1]
    assert session.trace[-1] == "COMMIT"


async def test_photography_admission_atomically_binds_physical_carrier_without_changing_plan_digest(tmp_path):
    plan_bytes, bindings = _photography_authority(tmp_path)
    changed = json.loads(bindings)
    changed["input_root"] = "/another/worker/input/mount"
    binding_digests = []
    for bound in (bindings, canonicalize_json(changed)):
        session = Session(scalars=[*_capacities(active=0), 100.0], gets=[SimpleNamespace(canonical_bytes=plan_bytes)])
        locator = await _admit(_store(session), plan_bytes=plan_bytes, execution_bindings=bound)
        assert locator.plan_digest == hashlib.sha256(plan_bytes).hexdigest()
        assert locator.plan_digest != json.loads(plan_bytes)["plan_fingerprint_sha256"]
        attempt = next(model for model in session.added if isinstance(model, DispatchAttemptModel))
        assert attempt.execution_bindings == bound
        assert attempt.execution_bindings_digest == hashlib.sha256(bound).hexdigest()
        evidence = next(model for model in session.added if isinstance(model, OperationalRecordModel))
        assert json.loads(evidence.canonical_bytes)["execution_bindings_digest"] == attempt.execution_bindings_digest
        binding_digests.append(attempt.execution_bindings_digest)
        outbox = next(model for model in session.added if isinstance(model, OperationalOutboxModel))
        assert outbox.payload == locator.to_payload()
        assert session.trace[-1] == "COMMIT"
    assert binding_digests[0] != binding_digests[1]


@pytest.mark.parametrize("invalid", ["missing", "extra", "noncanonical", "oversized", "legacy"])
async def test_invalid_photography_authority_never_reserves_capacity(tmp_path, invalid):
    plan_bytes, bindings = _photography_authority(tmp_path)
    if invalid == "missing":
        bindings = None
    elif invalid == "extra":
        bindings = canonicalize_json({**json.loads(bindings), "argv": ["untrusted"]})
    elif invalid == "noncanonical":
        bindings += b"\n"
    elif invalid == "oversized":
        bindings += b" " * 65536
    else:
        plan_bytes = _plan()
    session = Session()
    with pytest.raises(RepositoryError, match="canonical execution authority and bindings"):
        await _admit(_store(session), plan_bytes=plan_bytes, execution_bindings=bindings)
    assert session.trace == []


@pytest.mark.parametrize("photography", [False, True])
async def test_plan_and_bindings_readback_preserves_exact_authority_without_local_paths(tmp_path, photography):
    plan_bytes, bindings = _photography_authority(tmp_path) if photography else (_plan(), None)
    attempt = _attempt(
        plan_digest=hashlib.sha256(plan_bytes).hexdigest(),
        execution_bindings=bindings,
        execution_bindings_digest=None if bindings is None else hashlib.sha256(bindings).hexdigest(),
    )
    model = SimpleNamespace(canonical_bytes=plan_bytes)
    store = _store(Session(gets=[attempt, model, attempt, model]))
    locator = _fence(attempt).locator
    assert await store.fetch_plan(locator) == plan_bytes
    assert await store.fetch_execution_bindings(locator) == bindings


@pytest.mark.parametrize("fetch_method", ["fetch_plan", "fetch_execution_bindings"])
@pytest.mark.parametrize(
    "invalid",
    [
        "missing_attempt",
        "wrong_locator",
        "missing_plan",
        "wrong_plan_digest",
        "noncanonical_plan",
        "missing_bindings",
        "unpaired_digest",
        "wrong_binding_digest",
        "forged_binding_schema",
        "legacy_binding",
    ],
)
async def test_plan_or_bindings_corruption_never_returns_partial_execution_authority(tmp_path, fetch_method, invalid):
    plan_bytes, bindings = _photography_authority(tmp_path)
    if invalid == "legacy_binding":
        plan_bytes = _plan()
    if invalid == "noncanonical_plan":
        plan_bytes += b"\n"
    attempt = _attempt(
        plan_digest=hashlib.sha256(plan_bytes).hexdigest(),
        execution_bindings=bindings,
        execution_bindings_digest=hashlib.sha256(bindings).hexdigest(),
    )
    locator = _fence(attempt).locator
    model = SimpleNamespace(canonical_bytes=plan_bytes)
    if invalid == "missing_attempt":
        attempt = None
    elif invalid == "wrong_locator":
        attempt.dispatch_id = "other"
    elif invalid == "missing_plan":
        model = None
    elif invalid == "wrong_plan_digest":
        model.canonical_bytes = _plan()
    elif invalid == "missing_bindings":
        attempt.execution_bindings = attempt.execution_bindings_digest = None
    elif invalid == "unpaired_digest":
        attempt.execution_bindings_digest = None
    elif invalid == "wrong_binding_digest":
        attempt.execution_bindings_digest = "f" * 64
    elif invalid == "forged_binding_schema":
        attempt.execution_bindings = canonicalize_json({**json.loads(bindings), "schema": "tp.job.photography.bindings.v99"})
        attempt.execution_bindings_digest = hashlib.sha256(attempt.execution_bindings).hexdigest()
    with pytest.raises(DispatchAuthorityLost):
        await getattr(_store(Session(gets=[attempt, model])), fetch_method)(locator)


@pytest.mark.parametrize(
    "holder,duration", [("", 1), ("x" * 129, 1), (None, 1), ("worker", 0), ("worker", float("inf")), ("worker", float("nan"))]
)
async def test_invalid_claim_lease_never_opens_transaction(holder, duration):
    session = Session()
    with pytest.raises(RepositoryError, match="invalid holder"):
        await _store(session).claim_dispatch(_fence().locator, holder, lease_seconds=duration)
    assert not session.trace


@pytest.mark.parametrize("case", ["missing", "wrong_locator", "terminal", "claimed"])
async def test_claim_rejects_unknown_or_consumed_attempt_without_job_write(case):
    attempt = _attempt(state="queued", lease_epoch=0)
    locator = _fence(attempt).locator
    if case == "wrong_locator":
        attempt.dispatch_id = "other"
    elif case == "terminal":
        attempt.state = "failed"
    elif case == "claimed":
        attempt.lease_epoch = 1
    session = Session(scalars=[None if case == "missing" else attempt])
    with pytest.raises(DispatchAuthorityLost):
        await _store(session).claim_dispatch(locator, "worker", lease_seconds=10)
    assert session.trace[-1] == "ROLLBACK"
    assert not session.added


@pytest.mark.parametrize("job_state", [None, "running", "queued"])
async def test_claim_uses_db_epoch_and_locks_projection_before_authorized_write(job_state):
    attempt = _attempt(state="queued", lease_epoch=0)
    job = None if job_state is None else _job(state=job_state)
    session = Session(scalars=[attempt, 100.0, 42, job])
    store = _store(session)
    if job_state != "queued":
        with pytest.raises(DispatchAuthorityLost, match="projection"):
            await store.claim_dispatch(_fence(attempt).locator, "worker", lease_seconds=10)
        assert session.trace[-1] == "ROLLBACK"
        assert not session.added
    else:
        fence = await store.claim_dispatch(_fence(attempt).locator, "worker", lease_seconds=10)
        assert fence.lease_epoch == 42 and fence.lease_valid_until == 110.0
        assert job.state == "running" and job.started_at == 100.0
        assert next(i for i, q in enumerate(session.trace) if "FROM jobs" in q) < next(
            i for i, q in enumerate(session.trace) if "set_config" in q
        )
        assert session.trace[-1] == "COMMIT"


@pytest.mark.parametrize("case", ["missing", "locator", "state", "holder", "epoch", "no_deadline", "expired"])
async def test_renewal_rejects_every_revoked_fence_dimension(case):
    attempt = _attempt()
    fence = _fence(attempt)
    mutations = {
        "locator": ("dispatch_id", "other"),
        "state": ("state", "failed"),
        "holder": ("holder", "other"),
        "epoch": ("lease_epoch", 8),
        "no_deadline": ("lease_valid_until", None),
        "expired": ("lease_valid_until", 100.0),
    }
    if case in mutations:
        setattr(attempt, *mutations[case])
    session = Session(scalars=[None if case == "missing" else attempt, 100.0])
    with pytest.raises(DispatchAuthorityLost):
        await _store(session).renew_dispatch(fence, lease_seconds=30)
    assert session.trace[-1] == "ROLLBACK"


async def test_renewal_uses_current_database_clock():
    attempt = _attempt()
    session = Session(scalars=[attempt, 125.0])
    await _store(session).renew_dispatch(_fence(attempt), lease_seconds=30)
    assert attempt.lease_valid_until == 155.0
    assert session.trace[-1] == "COMMIT"


async def test_assert_dispatch_authority_reads_live_claim_without_renewal_or_projection_writes():
    attempt, job = _attempt(), _job(cancel_requested=False)
    session = Session(scalars=[attempt, 125.0, job, 126.0])
    await _store(session).assert_dispatch_authority(_fence(attempt))
    assert attempt.lease_valid_until == 200.0
    assert attempt.state == job.state == "running"
    assert not session.added
    assert session.trace[0] == "BEGIN" and session.trace[-1] == "COMMIT"
    assert len([query for query in session.trace if "FOR UPDATE" in query]) == 2
    assert not any("set_config" in query or query.startswith(("UPDATE", "INSERT")) for query in session.trace)


@pytest.mark.parametrize("case", ["missing", "locator", "state", "holder", "epoch", "no_deadline", "expired"])
async def test_assert_dispatch_authority_rejects_revoked_claim_before_reading_projection(case):
    attempt = _attempt()
    fence = _fence(attempt)
    mutations = {
        "locator": ("dispatch_id", "other"),
        "state": ("state", "canceled"),
        "holder": ("holder", "other"),
        "epoch": ("lease_epoch", 8),
        "no_deadline": ("lease_valid_until", None),
        "expired": ("lease_valid_until", 100.0),
    }
    if case in mutations:
        setattr(attempt, *mutations[case])
    session = Session(scalars=[None if case == "missing" else attempt, 100.0])
    with pytest.raises(DispatchAuthorityLost):
        await _store(session).assert_dispatch_authority(fence)
    assert session.trace[-1] == "ROLLBACK"
    assert not any("FROM jobs" in query for query in session.trace)
    assert not session.added


@pytest.mark.parametrize("case", ["missing_job", "terminal_job", "cancel_requested", "expired_after_job_lock"])
async def test_assert_dispatch_authority_rechecks_cancellation_and_clock_after_projection_lock(case):
    attempt = _attempt()
    job = _job(cancel_requested=case == "cancel_requested", state="canceled" if case == "terminal_job" else "running")
    session = Session(
        scalars=[attempt, 100.0, None if case == "missing_job" else job, 200.0 if case == "expired_after_job_lock" else 101.0]
    )
    with pytest.raises(DispatchAuthorityLost, match="expired or canceled"):
        await _store(session).assert_dispatch_authority(_fence(attempt))
    assert session.trace[-1] == "ROLLBACK"
    assert not session.added
    assert attempt.lease_valid_until == 200.0


@pytest.mark.parametrize("state", ["succeeded", "failed", "canceled"])
async def test_terminal_event_counter_projection_and_outbox_share_transaction(state):
    attempt, job, capacities = _attempt(), _job(), _capacities()
    session = Session(scalars=[*capacities, attempt, 100.0, job, 101.0])
    error = {"details": {"reason": "original"}}
    result = await _store(session).finish_dispatch(_fence(attempt), state=state, exit_code=0, error=error)
    error["details"]["reason"] = "mutated"
    assert job.error["details"]["reason"] == "original"
    assert result["error"]["details"]["reason"] == "original"
    assert job.state == attempt.state == state
    assert job.finished_at == job.done_published_at == 101.0
    assert job.cancel_requested == (state == "canceled")
    assert job.progress == (100 if state == "succeeded" else 0)
    assert [cap.active for cap in capacities] == [0, 0]
    events = [model for model in session.added if isinstance(model, JobEventModel)]
    outboxes = [model for model in session.added if isinstance(model, OperationalOutboxModel)]
    assert len(events) == len(outboxes) == 1
    assert events[0].seq == outboxes[0].payload["seq"] == 41
    assert events[0].event_type == "done"
    job_lock = next(i for i, q in enumerate(session.trace) if "FROM jobs" in q)
    event_counter = next(i for i, q in enumerate(session.trace) if "INSERT INTO job_event_sequences" in q)
    assert job_lock < event_counter < session.trace.index("ADD job_events") < session.trace.index("ADD operational_outbox")
    assert session.trace[-1] == "COMMIT"


@pytest.mark.parametrize("case", ["invalid_state", "missing_job", "expired_after_lock", "capacity_underflow"])
async def test_terminal_rechecks_authority_after_projection_lock_and_rolls_back(case):
    attempt, capacities = _attempt(), _capacities()
    if case == "capacity_underflow":
        capacities[0].active = 0
    session = Session(
        scalars=[
            *capacities,
            attempt,
            100.0,
            None if case == "missing_job" else _job(),
            200.0 if case == "expired_after_lock" else 101.0,
        ]
    )
    with pytest.raises(RepositoryError):
        await _store(session).finish_dispatch(_fence(attempt), state="running" if case == "invalid_state" else "failed")
    assert session.trace[-1] == "ROLLBACK"
    assert not any(isinstance(model, (JobEventModel, OperationalOutboxModel)) for model in session.added)


@pytest.mark.parametrize("case", ["missing", "other_tenant", "terminal", "active"])
async def test_cancel_is_tenant_bound_and_terminal_idempotent(case):
    attempt = _attempt()
    if case == "other_tenant":
        attempt.tenant_id = "other"
    elif case == "terminal":
        attempt.state = "succeeded"
    capacities = _capacities()
    session = Session(scalars=[*capacities, None if case == "missing" else attempt, 100.0, _job(), 101.0])
    if case in {"missing", "other_tenant"}:
        with pytest.raises(DispatchAuthorityLost):
            await _store(session).cancel_dispatch("job_offline", "tenant")
    else:
        assert await _store(session).cancel_dispatch("job_offline", "tenant") is (case == "active")
        assert [cap.active for cap in capacities] == ([0, 0] if case == "active" else [1, 1])
    if case != "active":
        assert not session.added


@pytest.mark.parametrize("legacy_count", [0, 1])
async def test_cutover_refuses_live_legacy_projection(legacy_count):
    session = Session(scalars=[legacy_count])
    if legacy_count:
        with pytest.raises(RepositoryError, match="legacy active jobs"):
            await _store(session).validate_cutover()
    else:
        await _store(session).validate_cutover()
    assert "NOT (EXISTS" in session.trace[0]
