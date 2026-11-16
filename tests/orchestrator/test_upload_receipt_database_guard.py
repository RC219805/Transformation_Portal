"""Upload service cleanup is authorized by its own active database session."""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tests.orchestrator import test_upload_receipt_services as services

pytestmark = pytest.mark.unit


class Session:
    def __init__(self, database):
        self.database = database
        self.statements = []
        self.transaction = False

    @asynccontextmanager
    async def begin(self):
        self.transaction = True
        try:
            yield
        finally:
            self.transaction = False

    async def scalar(self, statement):
        assert self.transaction
        self.statements.append(str(statement))
        return self.database

    async def execute(self, statement):
        assert self.transaction
        self.statements.append(str(statement))


@pytest.mark.parametrize(
    "url",
    [
        None,
        "",
        "postgresql+asyncpg://tester@localhost",
        "postgresql+asyncpg://tester@localhost/production",
        "postgresql+asyncpg://tester@localhost/production?application_name=disposable_test",
    ],
)
def test_url_database_suffix_cannot_come_from_query_or_missing_name(url):
    with pytest.raises(RuntimeError, match="refusing to alter"):
        services._validate_upload_test_database_url(url)
    services._validate_upload_test_database_url("postgresql+asyncpg://tester@localhost/disposable_test?application_name=guard")


@pytest.mark.asyncio
@pytest.mark.parametrize("database", [None, "", "production", "production_test?sslmode=require"])
async def test_non_test_actual_database_never_executes_truncate(database):
    session = Session(database)
    async with session.begin():
        with pytest.raises(RuntimeError, match="refusing to alter"):
            await services._clear_upload_tables(session)
    assert session.statements == ["SELECT current_database()"]


@pytest.mark.asyncio
async def test_each_clear_rechecks_the_provided_session_before_truncating():
    session = Session("disposable_test")
    async with session.begin():
        await services._clear_upload_tables(session)
        assert session.statements[0] == "SELECT current_database()"
        assert session.statements[1].startswith("TRUNCATE upload_job_references,")
        session.database = "production"
        with pytest.raises(RuntimeError, match="refusing to alter"):
            await services._clear_upload_tables(session)
    assert len(session.statements) == 3


@pytest.mark.asyncio
async def test_fixture_checks_and_truncates_inside_its_same_session_transaction(tmp_path, monkeypatch):
    session = Session("disposable_test")

    @asynccontextmanager
    async def open_session():
        yield session

    item = SimpleNamespace(_session=open_session)
    dispose = AsyncMock()
    monkeypatch.setenv("TP_UPLOAD_RECEIPT_TEST_DATABASE_URL", "postgresql+asyncpg://tester@localhost/disposable_test")
    monkeypatch.setattr(services, "PostgresUploadReceiptStore", lambda **_kwargs: item)
    monkeypatch.setattr(services._SharedEngine, "dispose", dispose)
    fixture = services.store.__wrapped__(tmp_path, monkeypatch)
    assert await anext(fixture) is item
    assert session.transaction is False
    assert session.statements[0] == "SELECT current_database()"
    assert session.statements[1].startswith("TRUNCATE upload_job_references,")
    await fixture.aclose()
    dispose.assert_awaited_once()
