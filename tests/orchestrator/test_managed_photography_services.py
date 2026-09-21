"""Managed V5 HTTP -> Postgres -> Redis -> external worker -> generation proof.

Opt in with TP_PHOTOGRAPHY_TEST_DATABASE_URL (a dedicated migrated *_test DB)
and TP_DISPATCH_TEST_REDIS_URL. No database is truncated; each run owns a unique
queue namespace. Only inference/runtime materialization are controlled fixtures.
The separate worker, fixed consumer, photo graph, verifier, publisher and HTTP
artifact retrieval execute normally. This is integration, not native-quality evidence.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx
import pytest
from redis.asyncio import Redis
from sqlalchemy import text

from tests.lux_depth_v5 import test_pipeline as controlled_pipeline
from transformation_portal.orchestrator.dispatch import DispatchLocator

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]
request_case = controlled_pipeline.request_case


# Loaded only in the explicitly spawned test interpreter and its fixed dispatch
# consumer. Production code has no fixture switch or replaceable worker module.
_CONTROLLED_NATIVE = """
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
from tests.lux_depth_v5.test_pipeline import ParentFixture, SessionFixture
from transformation_portal.lux_depth_v4 import evidence, pipeline as shared_pipeline
from transformation_portal.lux_depth_v5 import lifecycle, pipeline
from transformation_portal.lux_depth_v4.backend import validate_process_group_ownership

class ControlledManagedSession(SessionFixture):
    def __init__(self, python, plan, *, cancellation, own_process_group=True):
        assert own_process_group is False, "Managed native workers must inherit the owned group"
        validate_process_group_ownership(own_process_group)
        Path(os.environ["TP_TEST_PHOTOGRAPHY_MARKER"]).write_text(json.dumps({
            "pid": os.getpid(), "group": os.getpgrp(), "session": os.getsid(0),
            "plan_sha256": __import__("hashlib").sha256(plan.canonical_bytes).hexdigest(),
        }))
        super().__init__(python, plan, cancellation=cancellation)

def forbidden_prepare(*args, **kwargs):
    raise AssertionError("Worker must never re-prepare an admitted photography request")

pipeline._ExecutionProfile.session_type = ControlledManagedSession
shared_pipeline.PhotographyRuntime = ParentFixture
shared_pipeline.require_process_supervisor = lambda: SimpleNamespace(
    Process=lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=0)))
evidence.DA3RuntimeIdentityEvidence.from_mapping = lambda value: SimpleNamespace(
    cacheable=True, to_mapping=lambda: copy.deepcopy(value))
lifecycle.prepare = forbidden_prepare
"""


@pytest.mark.parametrize("api_prefix", ["/v1", "/v2"])
async def test_managed_photography_round_trip_uses_exact_admission_and_fenced_artifacts(
    request_case, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, api_prefix: str
) -> None:
    database_url = os.getenv("TP_PHOTOGRAPHY_TEST_DATABASE_URL")
    redis_url = os.getenv("TP_DISPATCH_TEST_REDIS_URL")
    if not database_url or not redis_url:
        pytest.skip("dedicated migrated photography Postgres and Redis are required")
    if not database_url.rsplit("/", 1)[-1].endswith("_test"):
        pytest.fail("managed photography integration requires a dedicated *_test database")

    import app as application
    from transformation_portal.orchestrator.artifact_store import reset_singleton as reset_artifacts
    from transformation_portal.orchestrator.queue import get_queue_broker
    from transformation_portal.orchestrator.queue import reset_singleton as reset_queue
    from transformation_portal.orchestrator.queue.locator import RedisLocatorQueueBroker
    from transformation_portal.orchestrator.storage import reset_singletons
    from transformation_portal.orchestrator.storage.operational import PostgresOperationalRecordStore
    from transformation_portal.orchestrator.storage.postgres import _SharedEngine

    repo_root = Path(__file__).resolve().parents[2]
    prefix = "photography-test-" + uuid.uuid4().hex
    api_key = "photography-service-test-api-key"
    marker = tmp_path / "native-fixture.json"
    injection = tmp_path / "controlled-native"
    injection.mkdir()
    (injection / "sitecustomize.py").write_text(_CONTROLLED_NATIVE, encoding="utf-8")
    environment = {
        "TP_ORCHESTRATOR_STATE_BACKEND": "postgres",
        "TP_ORCHESTRATOR_QUEUE_BACKEND": "redis",
        "TP_ORCHESTRATOR_IN_PROCESS_WORKERS_ENABLED": "0",
        "TP_ORCHESTRATOR_EXECUTION_ROOT": str(tmp_path / "execution-storage"),
        "TP_DATABASE_URL": database_url,
        "TP_REDIS_URL": redis_url,
        "TP_REDIS_KEY_PREFIX": prefix,
        "TP_LUX_V5_MANAGED_ENABLED": "1",
        "TP_LUX_V5_CACHE_DIR": str(tmp_path / "managed-cache"),
        "TRANSFORMATION_PORTAL_DA3_PYTHON": sys.executable,
        "TP_ARTIFACT_STORE": "local",
        "TP_ARTIFACT_LOCAL_ROOT": str(tmp_path / "artifact-store"),
        "TP_ALLOWED_INPUT_ROOTS": str(tmp_path),
        "TP_ALLOWED_OUTPUT_ROOTS": str(tmp_path),
        "TP_PILOT_CONTROL_PLANE_ENABLED": "0",
        "TP_PILOT_MAX_ACTIVE_JOBS_PER_TENANT": "2",
        "TP_WORKER_LEASE_SECONDS": "10",
        "TP_WORKER_HEARTBEAT_SECONDS": "1",
        "TP_TEST_PHOTOGRAPHY_MARKER": str(marker),
        "PYTHONPATH": f"{injection}:{repo_root / 'src'}:{repo_root}",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("TRANSFORMATION_PORTAL_RAW_PYTHON", raising=False)
    overrides = {
        "PILOT_CONTROL_PLANE_ENABLED": False,
        "PILOT_MAX_ACTIVE_JOBS_PER_TENANT": 2,
        "ALLOWED_INPUT_ROOTS": [tmp_path],
        "ALLOWED_OUTPUT_ROOTS": [tmp_path],
        "ALLOWED_PATH_ROOTS": [tmp_path],
        "API_KEY_SECRET": api_key,
        "RATE_LIMIT_PER_MINUTE": 0,
        "JOBS": {},
        "EVENT_SUBSCRIBERS": {},
    }
    for name, value in overrides.items():
        monkeypatch.setattr(application, name, value)
    monkeypatch.setattr(application.app.state, "job_repository", None, raising=False)
    monkeypatch.setattr(application.app.state, "job_repository_unavailable", False, raising=False)
    reset_singletons()
    reset_queue()
    reset_artifacts()
    broker = RedisLocatorQueueBroker(redis_url=redis_url, key_prefix=prefix + ":dispatch:v1:")
    records = PostgresOperationalRecordStore(database_url=database_url)
    redis = Redis.from_url(redis_url, decode_responses=True)
    worker = None
    log_path = tmp_path / "photography-worker.log"
    job_id = None
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application.app),
            base_url="http://testserver",
            headers={"x-api-key": api_key},
        ) as client:
            response = await client.post(
                f"{api_prefix}/jobs",
                json={
                    "pipeline": "lux-depth-v5",
                    "args": {
                        "input_dir": str(request_case.input_dir),
                        "output_dir": str(request_case.output_dir),
                        "input_color": "srgb",
                        "target_size": 56,
                        "strength": 0.25,
                        "clarity": 0.2,
                    },
                },
            )
            assert response.status_code == 200, response.text
            job_id = response.json()["data"]["id"]
            assert not request_case.output_dir.exists(), "admission must not create photographic outputs"
            locator = await records.get_locator(job_id)
            assert locator is not None
            plan_bytes = await records.fetch_plan(locator)
            bindings = await records.fetch_execution_bindings(locator)
            assert bindings is not None
            assert json.loads(plan_bytes)["schema"] == "tp.execution.plan.v4"
            assert locator.plan_digest == hashlib.sha256(plan_bytes).hexdigest()
            assert locator.plan_digest != json.loads(plan_bytes)["plan_fingerprint_sha256"]
            assert json.loads(bindings)["input_root"] == str(request_case.input_dir)
            queued = await redis.hget(prefix + ":dispatch:v1:job:" + job_id, "request")
            assert DispatchLocator.from_json(queued) == locator
            assert set(json.loads(queued)) == {
                "schema",
                "job_id",
                "attempt_id",
                "dispatch_id",
                "plan_digest",
                "tenant_id",
                "api_version",
            }
            assert "runtime_python" not in queued and "argv" not in queued

            with log_path.open("wb") as log:
                worker = subprocess.Popen(
                    [sys.executable, "-m", "transformation_portal.orchestrator.worker_process"],
                    cwd=repo_root,
                    env={**os.environ, **environment},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    application.JOBS.clear()
                    status = await client.get(f"{api_prefix}/jobs/{job_id}")
                    assert status.status_code == 200, status.text
                    if status.json()["data"]["state"] not in {"queued", "running"}:
                        break
                    assert worker.poll() is None, log_path.read_text()
                    await asyncio.sleep(0.1)
            assert status.json()["data"]["state"] == "succeeded", (status.text, log_path.read_text())
            native = json.loads(marker.read_text())
            assert native["pid"] != worker.pid
            assert native["pid"] == native["group"] == native["session"]
            assert native["plan_sha256"] == locator.plan_digest
            assert await records.fetch_plan(locator) == plan_bytes
            assert await records.fetch_execution_bindings(locator) == bindings
            manifest = await records.committed_manifest(job_id, "default")
            assert manifest and manifest["files"]
            inventory = {item["path"]: item for item in manifest["files"]}
            assert "input-0000/delivery.tif" in inventory
            assert inventory["execution-plan.json"]["sha256"] == locator.plan_digest
            descriptors = {item["relative_path"]: item for item in status.json()["data"]["artifacts"]["items"]}
            assert set(descriptors) == set(inventory)
            for relative, descriptor in descriptors.items():
                assert descriptor["path"] == relative
                assert descriptor["size_bytes"] == inventory[relative]["size_bytes"]
                assert descriptor["sha256"] == inventory[relative]["sha256"]
                assert descriptor["fingerprint_status"] == "ok"
                assert descriptor["url"] == descriptor["download_url"]
                assert descriptor["download_url"].endswith(f"/jobs/{job_id}/artifacts/{relative}")
            for relative in ("execution-plan.json", "execution-evidence.json", "input-0000/delivery.tif"):
                download = await client.get(descriptors[relative]["download_url"])
                assert download.status_code == 200, download.text
                assert hashlib.sha256(download.content).hexdigest() == inventory[relative]["sha256"]
            delivered_plan = await client.get(f"{api_prefix}/jobs/{job_id}/artifacts/execution-plan.json")
            assert delivered_plan.content == plan_bytes
            hidden = await client.get(
                f"{api_prefix}/jobs/{job_id}/artifacts/{inventory['execution-plan.json']['storage_path']}"
            )
            assert hidden.status_code == 404
            engine, _ = await _SharedEngine.get(database_url)
            async with engine.connect() as connection:
                row = (
                    await connection.execute(
                        text("SELECT state, lease_epoch, generation_id FROM dispatch_attempts WHERE job_id=:job"),
                        {"job": job_id},
                    )
                ).one()
                assert row.state == "succeeded" and row.lease_epoch > 0 and row.generation_id == manifest["generation_id"]
                kinds = set(
                    await connection.scalars(text("SELECT kind FROM operational_records WHERE job_id=:job"), {"job": job_id})
                )
                assert {"admitted", "claimed", "generation_committed"} <= kinds
    finally:
        if worker is not None:
            worker.terminate()
            try:
                await asyncio.to_thread(worker.wait, 5)
            except subprocess.TimeoutExpired:
                worker.kill()
                await asyncio.to_thread(worker.wait, 5)
        if job_id is not None:
            await records.cancel_dispatch(job_id, "default")
        await broker.reset()
        await broker.close()
        await get_queue_broker().close()
        await redis.aclose()
        await _SharedEngine.dispose(database_url)
        reset_singletons()
        reset_queue()
        reset_artifacts()
