"""Destructive service fixtures verify the database on their own connection."""

import pytest
from sqlalchemy.engine import make_url

from tests.orchestrator.test_job_submission_services import _clear_submission_tables

pytestmark = pytest.mark.unit


class Connection:
    def __init__(self, database):
        self.database = database
        self.statements = []

    async def scalar(self, statement):
        self.statements.append(str(statement))
        return self.database

    async def execute(self, statement):
        self.statements.append(str(statement))


@pytest.mark.asyncio
@pytest.mark.parametrize("database", [None, "", "production", "production_test?sslmode=require"])
async def test_non_test_actual_database_never_executes_truncate(database):
    connection = Connection(database)
    with pytest.raises(RuntimeError, match="refusing to truncate"):
        await _clear_submission_tables(connection)
    assert connection.statements == ["SELECT current_database()"]


@pytest.mark.asyncio
async def test_each_clear_checks_the_same_connection_again_before_truncating():
    connection = Connection("disposable_test")
    await _clear_submission_tables(connection)
    assert connection.statements[0] == "SELECT current_database()"
    assert connection.statements[1].startswith("TRUNCATE jobs,")
    connection.database = "production"
    with pytest.raises(RuntimeError, match="refusing to truncate"):
        await _clear_submission_tables(connection)
    assert len(connection.statements) == 3


def test_url_query_cannot_supply_the_database_suffix():
    url = make_url("postgresql+asyncpg://tester@localhost/production?password=disposable_test")
    assert url.database == "production"
    assert not url.database.endswith("_test")
