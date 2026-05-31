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

from transformation_portal.orchestrator import JobRecord
from transformation_portal.orchestrator.models import JobEventModel, JobModel
from transformation_portal.orchestrator.storage.postgres import PostgresJobEventStore, PostgresJobRepository

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
        return _ScalarResult(self.current_max_seq)

    async def stream_scalars(self, statement: Any) -> _AsyncRows:
        self.streamed.append(statement)
        return _AsyncRows(self.rows)

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
