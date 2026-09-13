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

from tests.core.test_execution_plan import _valid_payload
from transformation_portal.core.execution_plan import CanonicalExecutionPlan
from transformation_portal.orchestrator.dispatch import DispatchFence, DispatchLocator
from transformation_portal.orchestrator.models import (
    AdmissionCapacityModel,
    DispatchAttemptModel,
    JobEventModel,
    JobModel,
    OperationalOutboxModel,
    OperationalRecordModel,
)
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
