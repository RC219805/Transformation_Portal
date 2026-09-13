"""Signed-actor HTTP authority through a separate real Redis/Postgres worker.

Requires a dedicated migrated TP_TENANT_TEST_DATABASE_URL and
TP_DISPATCH_TEST_REDIS_URL. Uses a unique broker namespace and real archive
fixture; does not truncate the database or substitute the job executor.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from contextlib import suppress
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_signed_tenants_isolate_real_external_worker_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    database_url = os.getenv("TP_TENANT_TEST_DATABASE_URL")
    redis_url = os.getenv("TP_DISPATCH_TEST_REDIS_URL")
    if not database_url or not redis_url:
        pytest.skip("dedicated migrated tenant Postgres and Redis are required")
    database_name = database_url.rsplit("/", 1)[-1]
    if database_name != "tp_signed_tenant" and not database_name.endswith("_test"):
        pytest.fail("tenant integration evidence requires a dedicated test database")

    import app as application
    from transformation_portal.orchestrator.artifact_store import reset_singleton as reset_artifacts
    from transformation_portal.orchestrator.queue import get_queue_broker
    from transformation_portal.orchestrator.queue import reset_singleton as reset_queue
    from transformation_portal.orchestrator.queue.locator import RedisLocatorQueueBroker
    from transformation_portal.orchestrator.storage import reset_singletons
    from transformation_portal.orchestrator.storage.operational import PostgresOperationalRecordStore
    from transformation_portal.orchestrator.storage.postgres import _SharedEngine

    repo_root = Path(__file__).resolve().parents[2]
    workspaces, cas = tmp_path / "workspaces", tmp_path / "cas"
    fixture = workspaces / "tenant_a" / "archive"
    shutil.copytree(repo_root / "tests" / "fixtures" / "archive_small", fixture)
    output = workspaces / "tenant_a" / "outputs"
    prefix = "signed-tenant-test-" + uuid.uuid4().hex
    api_key = "signed-tenant-service-test-api-key"
    identity_secret = "signed-tenant-service-test-dedicated-secret-32bytes"
    memberships = json.dumps({"a@example.com": "tenant_a", "b@example.com": "tenant_b"})
    environment = {
        "TP_ORCHESTRATOR_STATE_BACKEND": "postgres",
        "TP_ORCHESTRATOR_QUEUE_BACKEND": "redis",
        "TP_ORCHESTRATOR_EXECUTION_ROOT": str(tmp_path / "execution-storage"),
        "TP_DATABASE_URL": database_url,
        "TP_REDIS_URL": redis_url,
        "TP_REDIS_KEY_PREFIX": prefix,
        "TP_PILOT_CONTROL_PLANE_ENABLED": "1",
        "TP_PILOT_ALLOWED_PIPELINES": "archive-gate-a",
        "TP_PILOT_MAX_ACTIVE_JOBS_PER_TENANT": "2",
        "TP_PILOT_TENANT_WORKSPACE_ROOT": str(workspaces),
        "TP_PILOT_TENANT_CAS_ROOT": str(cas),
        "TP_PILOT_ACTOR_TENANTS_JSON": memberships,
        "TP_FRONTDOOR_IDENTITY_SECRET": identity_secret,
        "TP_API_KEY": api_key,
        "TP_ARTIFACT_STORE": "local",
        "TP_ARTIFACT_LOCAL_ROOT": str(tmp_path / "artifact-store"),
        "TP_ALLOWED_INPUT_ROOTS": str(tmp_path),
        "TP_ALLOWED_OUTPUT_ROOTS": str(tmp_path),
        "TP_WORKER_LEASE_SECONDS": "10",
        "TP_WORKER_HEARTBEAT_SECONDS": "1",
        "PYTHONPATH": f"{repo_root / 'src'}:{repo_root}",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    overrides = {
        "PILOT_CONTROL_PLANE_ENABLED": True,
        "PILOT_MAX_ACTIVE_JOBS_PER_TENANT": 2,
        "PILOT_ALLOWED_PIPELINES": {"archive-gate-a"},
        "PILOT_ALLOWED_TENANTS": {"tenant_a", "tenant_b"},
        "PILOT_TENANT_WORKSPACE_ROOT": workspaces,
        "PILOT_TENANT_CAS_ROOT": cas,
        "PILOT_ACTOR_TENANTS_JSON": memberships,
        "FRONTDOOR_IDENTITY_SECRET": identity_secret,
        "_PILOT_TENANT_MANAGER": None,
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

    def headers(method: str, target: str, email: str = "a@example.com") -> dict[str, str]:
        payload = {
            "v": 1,
            "iat": int(time.time()),
            "method": method,
            "target": target,
            "actor": {"username": "actor", "accessEmail": email, "role": "admin"},
        }
        encoded = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode().rstrip("=")
        signature = hmac.new(
            identity_secret.encode(), b"tp.frontdoor.actor.v1\n" + encoded.encode(), hashlib.sha256
        ).hexdigest()
        return {"x-api-key": api_key, "x-tp-actor-assertion": encoded + "." + signature}

    request_body = {
        "pipeline": "archive-gate-a",
        "args": {
            "input_dir": str(fixture / "archive_root"),
            "output_dir": str(output),
            "archive_command": "fixity-scan",
            "archive_index": str(fixture / "archive_index_normalized.csv.gz"),
            "validate_schemas": False,
        },
    }
    worker = None
    admitted_ids: list[str] = []
    broker = RedisLocatorQueueBroker(redis_url=redis_url, key_prefix=prefix + ":dispatch:v1:")
    records = PostgresOperationalRecordStore(database_url=database_url)
    log_path = tmp_path / "signed-worker.log"
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application.app), base_url="http://testserver"
        ) as client:
            # An authenticated tenant can cancel its own queued admission.
            canceled = await client.post("/v1/jobs", headers=headers("POST", "/v1/jobs"), json=request_body)
            assert canceled.status_code == 200, canceled.text
            canceled_id = canceled.json()["data"]["id"]
            admitted_ids.append(canceled_id)
            cancel_path = f"/v1/jobs/{canceled_id}/cancel"
            response = await client.post(cancel_path, headers=headers("POST", cancel_path))
            assert response.status_code == 200, response.text
            status_path = f"/v1/jobs/{canceled_id}"
            status = await client.get(status_path, headers=headers("GET", status_path))
            assert status.json()["data"]["state"] == "canceled"

            admitted = await client.post("/v1/jobs", headers=headers("POST", "/v1/jobs"), json=request_body)
            assert admitted.status_code == 200, admitted.text
            job_id = admitted.json()["data"]["id"]
            admitted_ids.append(job_id)
            assert not output.exists(), "admission and queued cancellation cannot create execution outputs"
            with log_path.open("wb") as log:
                worker = subprocess.Popen(
                    [sys.executable, "-m", "transformation_portal.orchestrator.worker_process"],
                    cwd=repo_root,
                    env={**os.environ, **environment},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                status_path = f"/v1/jobs/{job_id}"
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    status = await client.get(status_path, headers=headers("GET", status_path))
                    assert status.status_code == 200, status.text
                    if status.json()["data"]["state"] not in {"queued", "running"}:
                        break
                    await asyncio.sleep(0.1)
            assert status.json()["data"]["state"] == "succeeded", (status.text, log_path.read_text())
            manifest = await records.committed_manifest(job_id, "tenant_a")
            assert manifest and manifest["files"]
            item = manifest["files"][0]
            artifact_path = f"/v1/jobs/{job_id}/artifacts/{item['path']}"
            download = await client.get(artifact_path, headers=headers("GET", artifact_path))
            assert download.status_code == 200, download.text
            assert hashlib.sha256(download.content).hexdigest() == item["sha256"]

            for method, target in (
                ("GET", status_path),
                ("GET", f"/v1/jobs/{job_id}/events"),
                ("GET", artifact_path),
                ("POST", f"/v1/jobs/{job_id}/cancel"),
                ("DELETE", f"/v1/jobs/{job_id}/artifacts"),
            ):
                denied = await client.request(method, target, headers=headers(method, target, "b@example.com"))
                assert denied.status_code == 404, (method, target, denied.text)
            denied = await client.get(status_path, headers={"x-api-key": api_key, "x-tp-actor-email": "a@example.com"})
            assert denied.status_code == 401
            mismatch = await client.get(status_path, headers={**headers("GET", status_path), "x-tp-tenant-id": "tenant_b"})
            assert mismatch.status_code == 403
            assert await records.committed_manifest(job_id, "tenant_b") is None
            assert await records.committed_manifest(job_id, "tenant_a") == manifest
    finally:
        for job_id in admitted_ids:
            with suppress(Exception):
                await records.cancel_dispatch(job_id, "tenant_a")
                await broker.cancel(job_id)
        if worker is not None:
            worker.terminate()
            try:
                await asyncio.to_thread(worker.wait, 5)
            except subprocess.TimeoutExpired:
                worker.kill()
                await asyncio.to_thread(worker.wait, 5)
        await broker.reset()
        await broker.close()
        await get_queue_broker().close()
        await _SharedEngine.dispose(database_url)
        reset_singletons()
        reset_queue()
        reset_artifacts()
