"""Offline Postgres storage contract tests.

These tests exercise persistence-boundary behavior without requiring a live
Postgres service. The live repository contract still owns SQL integration; this
file pins pure Python copy/snapshot guarantees that can regress before a query
ever reaches the database.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator, Iterable

import pytest

from transformation_portal.orchestrator import JobRecord, OperationalAuditRecord
from transformation_portal.orchestrator.models import JobEventModel, JobModel, OperationalAuditEventModel
from transformation_portal.orchestrator.storage.base import JobNotFoundError, RepositoryError
from transformation_portal.orchestrator.storage.postgres import (
    PostgresJobEventStore,
    PostgresJobRepository,
    PostgresOperationalAuditStore,
)

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class _ScalarResult:
    def __init__(self, value: int) -> None:
        self._value = value

    def scalar_one(self) -> int:
        return self._value


class _AsyncRows:
    def __init__(self, rows: Iterable[Any]) -> None:
        self._rows = iter(rows)

    def __aiter__(self) -> "_AsyncRows":
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._rows)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class _FakePostgresSession:
    def __init__(self, *, current_max_seq: int = 0, rows: Iterable[Any] = ()) -> None:
        self.current_max_seq = current_max_seq
        self.rows = list(rows)
        self.added: list[Any] = []
        self.commits = 0
        self.executed: list[Any] = []
        self.streamed: list[Any] = []

    def add(self, model: Any) -> None:
        self.added.append(model)

    async def execute(self, statement: Any) -> _ScalarResult:
        self.executed.append(statement)
        if getattr(statement, "is_insert", False):
            self.current_max_seq += 1
        return _ScalarResult(self.current_max_seq)

    async def stream_scalars(self, statement: Any) -> _AsyncRows:
        self.streamed.append(statement)
        return _AsyncRows(self.rows)

    async def flush(self) -> None:
        pass

    @asynccontextmanager
    async def begin(self) -> AsyncIterator[None]:
        yield
        self.commits += 1

    async def commit(self) -> None:
        self.commits += 1


def _install_fake_session(target: Any, session: _FakePostgresSession) -> None:
    @asynccontextmanager
    async def fake_session() -> AsyncIterator[_FakePostgresSession]:
        yield session

    target._session = fake_session  # type: ignore[method-assign]  # noqa: SLF001


async def test_postgres_repository_create_snapshots_mutable_record_fields() -> None:
    repo = PostgresJobRepository(database_url="postgresql+asyncpg://user:pw@host/db")
    session = _FakePostgresSession()
    _install_fake_session(repo, session)
    record = JobRecord(
        id="job-pg-create-copy",
        created_at=1.0,
        request={"args": {"quality": "premium"}},
        effective_request={"args": {"resolved_backend": "da3"}},
        logs_tail=["line-1"],
        artifacts={"items": [{"relative_path": "report.json"}]},
        artifact_lookup={"out/report.json": Path("/abs/out/report.json")},
        run_summary={"counts": {"succeeded": 1}},
        error={"details": {"code": "original"}},
    )

    await repo.create(record)
    record.request["args"]["quality"] = "mutated"
    record.effective_request["args"]["resolved_backend"] = "mutated"
    record.logs_tail.append("mutated")
    record.artifacts["items"][0]["relative_path"] = "mutated.json"
    record.artifact_lookup["out/report.json"] = Path("/mutated/report.json")
    record.run_summary["counts"]["succeeded"] = 99
    assert record.error is not None
    record.error["details"]["code"] = "mutated"

    job_model = next(model for model in session.added if isinstance(model, JobModel))
    artifact_model = next(model for model in session.added if not isinstance(model, JobModel))

    assert session.commits == 1
    assert job_model.request == {"args": {"quality": "premium"}}
    assert job_model.effective_request == {"args": {"resolved_backend": "da3"}}
    assert job_model.logs_tail == ["line-1"]
    assert job_model.artifacts == {"items": [{"relative_path": "report.json"}]}
    assert job_model.run_summary == {"counts": {"succeeded": 1}}
    assert job_model.error == {"details": {"code": "original"}}
    assert artifact_model.path == "out/report.json"
    assert artifact_model.absolute_path == "/abs/out/report.json"


async def test_postgres_event_append_snapshots_payload_for_db_and_return_value() -> None:
    store = PostgresJobEventStore(database_url="postgresql+asyncpg://user:pw@host/db")
    session = _FakePostgresSession(current_max_seq=41)
    _install_fake_session(store, session)
    payload = {"nested": {"state": "running"}, "items": [{"path": "out/report.json"}]}

    event = await store.append("job-pg-event-copy", "state", payload, created_at=123.0)
    payload["nested"]["state"] = "mutated"
    payload["items"][0]["path"] = "mutated.json"

    event_model = next(model for model in session.added if isinstance(model, JobEventModel))
    assert session.commits == 1
    assert event.seq == 42
    assert event.payload == {"nested": {"state": "running"}, "items": [{"path": "out/report.json"}]}
    assert event_model.payload == {"nested": {"state": "running"}, "items": [{"path": "out/report.json"}]}

    event.payload["nested"]["state"] = "returned-mutated"
    assert event_model.payload == {"nested": {"state": "running"}, "items": [{"path": "out/report.json"}]}


async def test_postgres_events_since_yields_payload_copies() -> None:
    row = SimpleNamespace(
        job_id="job-pg-replay-copy",
        seq=7,
        event_type="progress",
        payload={"nested": {"percent": 50}},
        created_at=200.0,
    )
    store = PostgresJobEventStore(database_url="postgresql+asyncpg://user:pw@host/db")
    session = _FakePostgresSession(rows=[row])
    _install_fake_session(store, session)

    events = [event async for event in store.events_since("job-pg-replay-copy", after_seq=6)]

    assert len(events) == 1
    assert events[0].seq == 7
    assert events[0].payload == {"nested": {"percent": 50}}
    events[0].payload["nested"]["percent"] = 99
    assert row.payload == {"nested": {"percent": 50}}


async def test_postgres_operational_audit_append_snapshots_record_fields() -> None:
    store = PostgresOperationalAuditStore(database_url="postgresql+asyncpg://user:pw@host/db")
    session = _FakePostgresSession()
    _install_fake_session(store, session)
    record = OperationalAuditRecord(
        created_at=300.0,
        action="job_create",
        decision="accepted",
        tenant_id="tenant_a",
        job_id="job_audit",
        actor={"username": "operator"},
        request_context={"path": "/v1/jobs"},
        details={"nested": {"pipeline": "lux-depth-v3"}},
    )

    await store.append(record)
    record.actor["username"] = "mutated"
    record.request_context["path"] = "/mutated"
    record.details["nested"]["pipeline"] = "mutated"

    audit_model = next(model for model in session.added if isinstance(model, OperationalAuditEventModel))
    assert session.commits == 1
    assert audit_model.created_at == 300.0
    assert audit_model.action == "job_create"
    assert audit_model.decision == "accepted"
    assert audit_model.tenant_id == "tenant_a"
    assert audit_model.job_id == "job_audit"
    assert audit_model.actor == {"username": "operator"}
    assert audit_model.request_context == {"path": "/v1/jobs"}
    assert audit_model.details == {"nested": {"pipeline": "lux-depth-v3"}}


async def test_postgres_append_snapshots_payload_before_waiting_for_session() -> None:
    store = PostgresJobEventStore(database_url="postgresql+asyncpg://user:pw@host/db")
    payload = {"nested": {"state": "original"}}
    session = _FakePostgresSession()

    @asynccontextmanager
    async def delayed_session() -> AsyncIterator[_FakePostgresSession]:
        payload["nested"]["state"] = "mutated-during-acquisition"
        yield session

    store._session = delayed_session
    event = await store.append("job-session-copy", "state", payload, created_at=1)
    assert event.payload == {"nested": {"state": "original"}}


class _QueryResult:
    def __init__(self, value=None, rows=(), rowcount=1):
        self.value, self.rows, self.rowcount = value, list(rows), rowcount

    def scalar_one(self):
        return self.value

    def scalar_one_or_none(self):
        return self.value

    def scalars(self):
        return self

    def all(self):
        return self.rows


class _RepositorySession(_FakePostgresSession):
    def __init__(self, *, results=(), model=None, commit_error=None):
        super().__init__()
        self.results = iter(results)
        self.model = model
        self.commit_error = commit_error
        self.rollbacks = 0
        self.refreshed = []

    async def execute(self, statement):
        self.executed.append(statement)
        return next(self.results, _QueryResult())

    async def get(self, model_type, key):
        assert model_type is JobModel
        return self.model

    async def refresh(self, model, attributes):
        self.refreshed.append((model, attributes))

    async def rollback(self):
        self.rollbacks += 1

    async def commit(self):
        if self.commit_error is not None:
            raise self.commit_error
        self.commits += 1


def _job_model():
    return JobModel(
        id="job_adapter",
        created_at=1.0,
        state="queued",
        progress=0,
        request={"args": {}},
        effective_request={"args": {}},
        logs_tail=["old"],
        artifacts={"items": []},
        run_summary={},
        error=None,
        cancel_requested=False,
        version=1,
        artifact_index=[],
    )


def _repository(session):
    repo = PostgresJobRepository(database_url="postgresql+asyncpg://unused/unused")
    _install_fake_session(repo, session)
    return repo


async def test_duplicate_job_insert_rolls_back_and_translates_integrity_error():
    from sqlalchemy.exc import IntegrityError

    session = _RepositorySession(commit_error=IntegrityError("insert", {}, ValueError("duplicate id")))
    with pytest.raises(RepositoryError, match="already exists"):
        await _repository(session).create(JobRecord(id="duplicate", created_at=1.0, request={}))
    assert session.commits == 0 and session.rollbacks == 1


@pytest.mark.parametrize("present", [False, True])
async def test_get_refreshes_artifact_index_and_returns_independent_data(present):
    model = _job_model() if present else None
    session = _RepositorySession(model=model)
    result = await _repository(session).get("job_adapter")
    if not present:
        assert result is None and not session.refreshed
    else:
        assert session.refreshed == [(model, ["artifact_index"])]
        result.request["args"]["changed"] = True
        assert model.request == {"args": {}}


@pytest.mark.parametrize("limit", [None, 1])
async def test_listing_preserves_total_and_limits_only_rows(limit):
    session = _RepositorySession(results=[_QueryResult(value=5), _QueryResult(rows=[_job_model()])])
    records, total = await _repository(session).list(limit=limit)
    assert total == 5 and len(records) == 1
    assert "LIMIT" not in str(session.executed[0])
    assert ("LIMIT" in str(session.executed[1])) is (limit is not None)


@pytest.mark.parametrize("case", ["unknown_field", "empty_missing", "empty_existing", "missing_update"])
async def test_update_rejects_invalid_fields_and_missing_jobs(case):
    session = _RepositorySession(model=_job_model() if case == "empty_existing" else None, results=[_QueryResult()])
    repo = _repository(session)
    if case == "empty_existing":
        assert (await repo.update("job_adapter")).id == "job_adapter"
    else:
        fields = {"unknown": 1} if case == "unknown_field" else {"progress": 50} if case == "missing_update" else {}
        with pytest.raises(RepositoryError if case == "unknown_field" else JobNotFoundError):
            await repo.update("job_adapter", **fields)
    assert session.commits == 0


async def test_update_locks_copies_nested_values_and_increments_version():
    model = _job_model()
    session = _RepositorySession(results=[_QueryResult(value=model)])
    payload = {
        "request": {"args": {"changed": [1]}},
        "effective_request": {"args": {}},
        "artifacts": {"items": [1]},
        "run_summary": {"stats": [1]},
        "logs_tail": ["new"],
        "error": {"details": [1]},
    }
    record = await _repository(session).update("job_adapter", **payload)
    for key in ("request", "effective_request", "artifacts", "run_summary", "error"):
        assert getattr(model, key) == payload[key]
        assert getattr(model, key) is not payload[key]
    payload["request"]["args"]["changed"].append(2)
    record.logs_tail.append("caller mutation")
    assert model.request["args"]["changed"] == [1]
    assert model.logs_tail == ["new"] and model.version == 2
    assert "FOR UPDATE" in str(session.executed[0])
    assert session.commits == 1


@pytest.mark.parametrize("single", [False, True])
async def test_nonpositive_log_tail_limit_is_rejected_before_query(single):
    session = _RepositorySession()
    repo = _repository(session)
    with pytest.raises(RepositoryError, match="positive"):
        if single:
            await repo.append_log("job", "line", tail_limit=0)
        else:
            await repo.append_logs("job", ["line"], tail_limit=0)
    assert not session.executed


@pytest.mark.parametrize("case", ["empty", "missing", "append", "trim"])
async def test_log_append_is_bounded_and_versioned(case):
    model = _job_model()
    session = _RepositorySession(results=[_QueryResult(value=None if case == "missing" else model)])
    repo = _repository(session)
    if case == "missing":
        with pytest.raises(JobNotFoundError):
            await repo.append_log("job_adapter", "new", tail_limit=2)
    elif case == "empty":
        await repo.append_logs("job_adapter", [], tail_limit=2)
        assert not session.executed
    else:
        await repo.append_logs("job_adapter", ["new1", "new2"], tail_limit=2 if case == "trim" else 4)
        assert model.logs_tail == (["new1", "new2"] if case == "trim" else ["old", "new1", "new2"])
        assert model.version == 2 and session.commits == 1


@pytest.mark.parametrize("case", ["missing", "conflicts", "retry_success"])
async def test_artifact_compare_and_swap_is_bounded_and_publishes_only_after_success(case):
    model = _job_model() if case != "missing" else None
    counts = [0, 0, 0] if case == "conflicts" else [0, 1, 1]
    session = _RepositorySession(model=model, results=[_QueryResult(rowcount=count) for count in counts])
    repo = _repository(session)
    if case in {"missing", "conflicts"}:
        with pytest.raises(JobNotFoundError if case == "missing" else RepositoryError):
            await repo.set_artifacts("job_adapter", {"items": [1]}, {"result.txt": Path("/result.txt")})
        assert not session.added and session.commits == 0
        assert session.rollbacks == (3 if case == "conflicts" else 0)
    else:
        await repo.set_artifacts("job_adapter", {"items": [1]}, {"result.txt": Path("/result.txt")})
        assert session.rollbacks == 1 and session.commits == 1
        assert len(session.added) == 1 and session.added[0].absolute_path == "/result.txt"
        assert "jobs.version =" in str(session.executed[0])


@pytest.mark.parametrize("active", [False, True])
async def test_delete_locks_job_and_preserves_active_durable_dispatch(active):
    session = _RepositorySession(results=[_QueryResult(), _QueryResult(value=active)])
    repo = _repository(session)
    if active:
        with pytest.raises(RepositoryError, match="active durable dispatch"):
            await repo.delete("job_adapter")
        assert not any(getattr(query, "is_delete", False) for query in session.executed)
    else:
        await repo.delete("job_adapter")
        deletes = [query.table.name for query in session.executed if getattr(query, "is_delete", False)]
        assert deletes == ["job_event_sequences", "job_events", "jobs"]
        assert session.commits == 1
    assert "FOR UPDATE" in str(session.executed[0])


@pytest.mark.parametrize("ids", [[], ["expired_a", "expired_b"]])
async def test_retention_filters_active_authority_and_deletes_only_selected_jobs(ids):
    session = _RepositorySession(results=[_QueryResult(rows=ids)])
    assert await _repository(session).cleanup_expired(100.0, 10.0) == ids
    selection = str(session.executed[0])
    assert "FOR UPDATE" in selection and "dispatch_attempts" in selection and "NOT (EXISTS" in selection
    deletes = [query for query in session.executed if getattr(query, "is_delete", False)]
    assert len(deletes) == (3 if ids else 0)
    for query in deletes:
        assert ids in query.compile().params.values()
    assert session.commits == (1 if ids else 0)
