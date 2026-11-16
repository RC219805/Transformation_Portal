"""Dedicated migrated Postgres evidence for consumed submission identities.

Set TP_SUBMISSION_TEST_DATABASE_URL to a disposable *_test database. This
suite truncates only that explicit database, never any application database.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import DBAPIError

from tests.core.test_execution_plan import _valid_payload
from tests.orchestrator.test_job_submissions import _headers
from transformation_portal.core.execution_plan import CanonicalExecutionPlan
from transformation_portal.orchestrator.storage.base import JobRecord
from transformation_portal.orchestrator.storage.operational import PostgresOperationalRecordStore
from transformation_portal.orchestrator.storage.postgres import PostgresJobRepository, _SharedEngine
from transformation_portal.orchestrator.submissions import SubmissionConflict, SubmissionIdentity, submission_intent

pytestmark = pytest.mark.integration


async def _clear_submission_tables(connection):
    # Recheck the actual database on the connection performing each truncate.
    # URL query parameters and routing proxies cannot establish this authority.
    database = await connection.scalar(text("SELECT current_database()"))
    if not isinstance(database, str) or not database.endswith("_test"):
        raise RuntimeError("refusing to truncate a database without the _test suffix")
    await connection.execute(
        text(
            "TRUNCATE jobs, job_events, job_event_sequences, admission_capacity, dispatch_attempts, "
            "dispatch_plans, operational_records, operational_outbox, committed_generations "
            "RESTART IDENTITY CASCADE"
        )
    )


@pytest_asyncio.fixture
async def store():
    url = os.getenv("TP_SUBMISSION_TEST_DATABASE_URL", "")
    if not url:
        pytest.skip("dedicated migrated TP_SUBMISSION_TEST_DATABASE_URL is required")
    if not (make_url(url).database or "").endswith("_test"):
        pytest.fail("refusing to truncate a database without the _test suffix")
    engine, _ = await _SharedEngine.get(url)

    async def clear():
        async with engine.begin() as connection:
            await _clear_submission_tables(connection)

    await clear()
    try:
        yield PostgresOperationalRecordStore(database_url=url), url
    finally:
        await clear()
        await _SharedEngine.dispose(url)


def _identity(*, owner="a", intent=None):
    digest = submission_intent({"pipeline": "lux-depth-v3"}) if intent is None else intent * 64
    return SubmissionIdentity("same-key", owner * 64, digest)


async def _admit(store, root: Path, identity):
    job_id = "job_" + uuid.uuid4().hex
    record = JobRecord(
        id=job_id,
        created_at=time.time(),
        request={"pipeline": "lux-depth-v3"},
        effective_request={"pipeline": "lux-depth-v3", "tenant_id": "tenant_a", "args": {}},
    )
    plan = CanonicalExecutionPlan.from_payload(_valid_payload()).to_canonical_json().encode()
    return await store.admit(
        record,
        plan,
        tenant_id="tenant_a",
        output_root=str(root / job_id),
        requested_output_root=str(root),
        global_limit=1,
        tenant_limit=1,
        submission=identity,
    )


@pytest.mark.asyncio
async def test_concurrent_independent_sessions_admit_one_job_one_capacity_one_outbox(store, tmp_path):
    authority, url = store
    locators = await asyncio.gather(
        *[_admit(PostgresOperationalRecordStore(database_url=url), tmp_path, _identity()) for _ in range(8)]
    )
    assert len({locator.job_id for locator in locators}) == 1
    engine, _ = await _SharedEngine.get(url)
    async with engine.connect() as connection:
        for table in ("jobs", "dispatch_attempts", "operational_records", "operational_outbox"):
            assert await connection.scalar(text(f"SELECT count(*) FROM {table}")) == 1
        assert await connection.scalar(text("SELECT active FROM admission_capacity WHERE scope='global'")) == 1
    assert not list(tmp_path.iterdir()), "admission must not create output directories"


@pytest.mark.asyncio
async def test_consumed_key_conflicts_before_full_capacity_and_retains_original_intent(store, tmp_path):
    authority, _ = store
    original = await _admit(authority, tmp_path, _identity())
    with pytest.raises(SubmissionConflict):
        await _admit(authority, tmp_path, _identity(intent="c"))
    receipt = await authority.submission_receipt("tenant_a", "a" * 64, "same-key")
    assert receipt.job_id == original.job_id and receipt.intent_sha256 == _identity().intent_sha256
    assert await authority.submission_receipt("tenant_a", "d" * 64, "same-key") is None
    assert await authority.submission_receipt("tenant_b", "a" * 64, "same-key") is None


@pytest.mark.asyncio
async def test_restart_and_job_projection_retirement_never_make_key_reusable(store, tmp_path):
    authority, url = store
    original = await _admit(authority, tmp_path, _identity())
    assert await authority.cancel_dispatch(original.job_id, "tenant_a")
    reopened = PostgresOperationalRecordStore(database_url=url)
    receipt = await reopened.submission_receipt("tenant_a", "a" * 64, "same-key")
    assert receipt.state == "canceled" and receipt.retired is False
    await PostgresJobRepository(database_url=url).delete(original.job_id)
    receipt = await reopened.submission_receipt("tenant_a", "a" * 64, "same-key")
    assert receipt.retired is True and receipt.state == "canceled"
    assert (await _admit(reopened, tmp_path, _identity())).job_id == original.job_id
    engine, _ = await _SharedEngine.get(url)
    async with engine.connect() as connection:
        assert await connection.scalar(text("SELECT count(*) FROM jobs")) == 0
        assert await connection.scalar(text("SELECT count(*) FROM dispatch_attempts")) == 1
        assert await connection.scalar(text("SELECT active FROM admission_capacity WHERE scope='global'")) == 0


@pytest.mark.asyncio
async def test_failure_before_transaction_commit_leaves_no_consumed_key_or_dispatch(store, tmp_path, monkeypatch):
    authority, url = store

    def fail(*_args, **_kwargs):
        raise RuntimeError("injected record failure")

    with monkeypatch.context() as patch:
        patch.setattr(authority, "_record", fail)
        with pytest.raises(RuntimeError):
            await _admit(authority, tmp_path, _identity())
    assert await authority.submission_receipt("tenant_a", "a" * 64, "same-key") is None
    engine, _ = await _SharedEngine.get(url)
    async with engine.connect() as connection:
        for table in ("jobs", "dispatch_attempts", "operational_outbox", "admission_capacity"):
            assert await connection.scalar(text(f"SELECT count(*) FROM {table}")) == 0
    assert await _admit(authority, tmp_path, _identity())


@pytest.mark.asyncio
async def test_database_rejects_submission_identity_mutation(store, tmp_path):
    authority, url = store
    locator = await _admit(authority, tmp_path, _identity())
    engine, _ = await _SharedEngine.get(url)
    async with engine.begin() as connection:
        with pytest.raises(DBAPIError, match="immutable dispatch identity"):
            await connection.execute(
                text("UPDATE dispatch_attempts SET submission_owner_sha256=:owner WHERE job_id=:job"),
                {"owner": "c" * 64, "job": locator.job_id},
            )


@pytest.mark.asyncio
async def test_process_exit_after_commit_keeps_receipt_and_only_one_dispatch(store, tmp_path):
    authority, url = store
    script = """
import asyncio, os
from pathlib import Path
from tests.orchestrator.test_job_submission_services import _admit, _identity
from transformation_portal.orchestrator.storage.operational import PostgresOperationalRecordStore
async def run():
    await _admit(PostgresOperationalRecordStore(database_url=os.environ['TP_SUBMISSION_TEST_DATABASE_URL']),
                 Path(os.environ['TP_SUBMISSION_TEST_OUTPUT_ROOT']), _identity())
    os._exit(0)  # abrupt API-process loss: no response or graceful provider shutdown
asyncio.run(run())
"""
    root = Path(__file__).resolve().parents[2]
    result = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-B", "-c", script],
        cwd=root,
        env={
            **os.environ,
            "TP_SUBMISSION_TEST_DATABASE_URL": url,
            "TP_SUBMISSION_TEST_OUTPUT_ROOT": str(tmp_path),
            "PYTHONPATH": f"{root / 'src'}:{root}",
        },
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    receipt = await authority.submission_receipt("tenant_a", "a" * 64, "same-key")
    assert receipt is not None and not receipt.retired
    assert (await _admit(authority, tmp_path, _identity())).job_id == receipt.job_id
    engine, _ = await _SharedEngine.get(url)
    async with engine.connect() as connection:
        assert await connection.scalar(text("SELECT count(*) FROM jobs")) == 1
        assert await connection.scalar(text("SELECT count(*) FROM operational_outbox")) == 1


@pytest.mark.asyncio
async def test_http_lost_response_replays_committed_job_without_preview_files_or_redis(store, tmp_path, monkeypatch):
    import app as application

    authority, url = store
    workspaces = tmp_path / "workspaces"
    fixture = workspaces / "tenant_a" / "archive"
    shutil.copytree(Path(__file__).parents[1] / "fixtures" / "archive_small", fixture)
    output = workspaces / "tenant_a" / "outputs"
    overrides = {
        "API_KEY_SECRET": "contract-secret",
        "ENFORCE_JOB_API_KEY": True,
        "PILOT_CONTROL_PLANE_ENABLED": True,
        "FRONTDOOR_IDENTITY_SECRET": "submission-test-identity-secret-32bytes",
        "PILOT_ACTOR_TENANTS_JSON": '{"shared@example.com":"tenant_a"}',
        "PILOT_ALLOWED_PIPELINES": {"archive-gate-a"},
        "PILOT_ALLOWED_TENANTS": {"tenant_a"},
        "PILOT_MAX_ACTIVE_JOBS_PER_TENANT": 1,
        "MAX_CONCURRENT_JOBS": 1,
        "PILOT_TENANT_WORKSPACE_ROOT": workspaces,
        "PILOT_TENANT_CAS_ROOT": tmp_path / "cas",
        "_PILOT_TENANT_MANAGER": None,
        "RATE_LIMIT_PER_MINUTE": 0,
        "ALLOWED_INPUT_ROOTS": [tmp_path],
        "ALLOWED_OUTPUT_ROOTS": [tmp_path],
        "ALLOWED_PATH_ROOTS": [tmp_path],
        "JOBS": {},
        "EVENT_SUBSCRIBERS": {},
    }
    for name, value in overrides.items():
        monkeypatch.setattr(application, name, value)
    for name, value in {
        "TP_JOB_SUBMISSION_RECOVERY_ENABLED": "1",
        "TP_DATABASE_URL": url,
        "TP_ORCHESTRATOR_STATE_BACKEND": "postgres",
        "TP_ORCHESTRATOR_QUEUE_BACKEND": "redis",
        "TP_ORCHESTRATOR_EXECUTION_ROOT": str(tmp_path / "execution"),
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(application, "get_operational_record_store", lambda: authority)
    monkeypatch.setattr(application, "_job_repository", lambda: PostgresJobRepository(database_url=url))

    async def unavailable_queue():
        raise RuntimeError("Redis temporarily unavailable after commit")

    monkeypatch.setattr(application, "_flush_operational_outbox_best_effort", unavailable_queue)
    body = {
        "pipeline": "archive-gate-a",
        "args": {
            "input_dir": str(fixture / "archive_root"),
            "output_dir": str(output),
            "archive_command": "fixity-scan",
            "archive_index": str(fixture / "archive_index_normalized.csv.gz"),
            "validate_schemas": False,
        },
    }
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application.app), base_url="http://testserver") as client:
        admitted = await client.post("/v1/jobs", json=body, headers=_headers("POST", "/v1/jobs", key="lost-response"))
        assert admitted.status_code == 200, admitted.text
        original_id = admitted.json()["data"]["id"]
        application.JOBS.clear()
        fixture.rename(tmp_path / "removed-source")

        async def forbidden(*_args, **_kwargs):
            pytest.fail("replay must not rerun preview")

        monkeypatch.setattr(application, "_build_config_preview_threaded", forbidden)
        replay = await client.post("/v2/jobs", json=body, headers=_headers("POST", "/v2/jobs", key="lost-response"))
        path = "/v2/job-submissions/lost-response"
        lookup = await client.get(path, headers=_headers("GET", path))
    assert replay.status_code == 200 and replay.json()["data"]["id"] == original_id
    assert lookup.json()["data"]["job"]["id"] == original_id
    assert not output.exists()
    engine, _ = await _SharedEngine.get(url)
    async with engine.connect() as connection:
        assert await connection.scalar(text("SELECT count(*) FROM jobs")) == 1
        assert await connection.scalar(text("SELECT count(*) FROM operational_outbox")) == 1
