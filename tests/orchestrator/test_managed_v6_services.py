"""Managed V6 HTTP -> Postgres -> Redis -> external worker -> fenced delivery.

Opt in with TP_PHOTOGRAPHY_TEST_DATABASE_URL (a dedicated migrated *_test DB)
and TP_DISPATCH_TEST_REDIS_URL. No database is truncated; each run owns a unique
queue namespace. Only inference/runtime materialization are controlled fixtures.
The separate worker, fixed consumer, photo graph, verifier, publisher and HTTP
artifact retrieval execute normally. This is integration, not native-quality evidence.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
import tifffile
from PIL import Image
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import DBAPIError

from tests.lux_depth_v5 import test_pipeline as controlled_pipeline
from tests.orchestrator.test_dispatch_authority_services import _record
from tests.orchestrator.test_managed_photography_services import _CONTROLLED_NATIVE
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v6.managed import ManagedLuxDepthV6Request
from transformation_portal.lux_depth_v6.publication import _publish_admitted_result
from transformation_portal.orchestrator.artifact_store.local import LocalArtifactStore
from transformation_portal.orchestrator.dispatch import DispatchLocator
from transformation_portal.orchestrator.execution_dispatch import execute_dispatch_plan
from transformation_portal.orchestrator.photography_v6_adapter import ManagedV6PhotographyPublisher, prepare_v6_dispatch
from transformation_portal.orchestrator.queue.locator import RedisLocatorQueueBroker
from transformation_portal.orchestrator.storage.operational import DispatchAuthorityLost, PostgresOperationalRecordStore
from transformation_portal.orchestrator.storage.postgres import _SharedEngine

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]
request_case = controlled_pipeline.request_case


@pytest.fixture
def service_urls():
    """Use an explicitly selected test DB; never truncate shared service state."""
    database_url = os.getenv("TP_PHOTOGRAPHY_TEST_DATABASE_URL") or os.getenv("TP_DISPATCH_TEST_DATABASE_URL")
    redis_url = os.getenv("TP_DISPATCH_TEST_REDIS_URL")
    if not database_url or not redis_url:
        pytest.skip("dedicated migrated photography Postgres and Redis are required")
    if not (make_url(database_url).database or "").endswith("_test"):
        pytest.fail("managed V6 integration requires a dedicated *_test database")
    return database_url, redis_url


@pytest.mark.parametrize("api_prefix", ["/v1", "/v2"])
async def test_v6_external_worker_keeps_admission_and_publishes_photo_depth_and_retained_evidence(
    service_urls, request_case, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, api_prefix: str
) -> None:
    database_url, redis_url = service_urls

    import app as application
    from transformation_portal.orchestrator.artifact_store import reset_singleton as reset_artifacts
    from transformation_portal.orchestrator.queue import get_queue_broker
    from transformation_portal.orchestrator.queue import reset_singleton as reset_queue
    from transformation_portal.orchestrator.storage import reset_singletons

    repo_root = Path(__file__).resolve().parents[2]
    prefix = "photography-v6-test-" + uuid.uuid4().hex
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
        "TP_LUX_V5_MANAGED_ENABLED": "0",
        "TP_LUX_V6_MANAGED_ENABLED": "1",
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
                    "pipeline": "lux-depth-v6",
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
            plan = json.loads(plan_bytes)
            assert plan["schema"] == "tp.execution.plan.v5"
            assert plan["inference"]["schema"] == "tp.execution.plan.v4"
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
            assert native["plan_sha256"] == hashlib.sha256(canonicalize_json(plan["inference"])).hexdigest()
            assert await records.fetch_plan(locator) == plan_bytes
            assert await records.fetch_execution_bindings(locator) == bindings
            manifest = await records.committed_manifest(job_id, "default")
            assert manifest and manifest["files"]
            inventory = {item["path"]: item for item in manifest["files"]}
            assert "v6/input-0000/delivery.tif" in inventory
            assert inventory["execution-plan.json"]["sha256"] == locator.plan_digest
            assert {
                "source-v5/execution-plan.json",
                "source-v5/execution-evidence.json",
                "source-v5/input-0000/native-depth.npy",
                "v6/plan.json",
                "v6/evidence.json",
                "v6/input-0000/depth-relative.tif",
                "v6/input-0000/depth-preview.png",
                "v6/input-0000/depth-preview-valid.png",
            } <= set(inventory)
            assert status.json()["data"]["artifacts"]["schema"] == "tp.lux.delivery.v4"
            assert status.json()["data"]["run_summary"]["pipeline"] == "lux_depth_v6"
            descriptors = {item["relative_path"]: item for item in status.json()["data"]["artifacts"]["items"]}
            assert set(descriptors) == set(inventory)
            for relative, descriptor in descriptors.items():
                assert descriptor["path"] == relative
                assert descriptor["size_bytes"] == inventory[relative]["size_bytes"]
                assert descriptor["sha256"] == inventory[relative]["sha256"]
                assert descriptor["fingerprint_status"] == "ok"
                assert descriptor["url"] == descriptor["download_url"]
                assert descriptor["download_url"].endswith(f"/jobs/{job_id}/artifacts/{relative}")
            assert descriptors["v6/input-0000/delivery.tif"]["browser_previewable"] is False
            assert descriptors["v6/input-0000/delivery.tif"]["preview_url"] == descriptors["v6/input-0000/preview.png"]["url"]
            assert descriptors["v6/input-0000/delivery.tif"]["preview_mime_type"] == "image/png"
            photo = descriptors["v6/input-0000/delivery.tif"]
            depth = descriptors["v6/input-0000/depth-relative.tif"]
            assert depth["preview_url"] == descriptors["v6/input-0000/depth-preview.png"]["url"]
            assert photo["display_hint"]["priority"] > depth["display_hint"]["priority"]
            assert photo["display_hint"]["compare_group"] == "lux-depth-v6|input-0000|photograph"
            assert (
                descriptors["v6/input-0000/preview.png"]["display_hint"]["compare_group"]
                == photo["display_hint"]["compare_group"]
            )
            assert depth["display_hint"]["compare_group"] == "lux-depth-v6|input-0000|relative-depth"
            assert (
                descriptors["v6/input-0000/depth-preview.png"]["display_hint"]["compare_group"]
                == depth["display_hint"]["compare_group"]
            )
            assert (
                descriptors["v6/input-0000/depth-preview-valid.png"]["display_hint"]["compare_group"]
                == "lux-depth-v6|input-0000|depth-validity"
            )
            assert (
                descriptors["source-v5/input-0000/delivery.tif"]["display_hint"]["compare_group"]
                == "lux-depth-v6|input-0000|source-v5"
            )
            assert "preview_url" not in descriptors["source-v5/input-0000/delivery.tif"]
            for relative in (
                "execution-plan.json",
                "execution-evidence.json",
                "v6/input-0000/delivery.tif",
                "v6/input-0000/preview.png",
                "v6/input-0000/depth-relative.tif",
                "v6/input-0000/depth-preview.png",
                "source-v5/input-0000/native-depth.npy",
            ):
                download = await client.get(descriptors[relative]["download_url"])
                assert download.status_code == 200, download.text
                assert hashlib.sha256(download.content).hexdigest() == inventory[relative]["sha256"]
                if relative.endswith(".tif"):
                    with tifffile.TiffFile(io.BytesIO(download.content)) as delivery:
                        pixels = delivery.asarray()
                        if relative == "v6/input-0000/delivery.tif":
                            assert pixels.dtype == "uint16" and pixels.shape == (28, 42, 3)
                        else:
                            assert pixels.dtype == "float32" and pixels.shape == (28, 42)
                if relative.endswith("preview.png"):
                    assert download.headers["content-type"] == "image/png"
                    with Image.open(io.BytesIO(download.content)) as preview:
                        assert preview.format == "PNG"
                        assert 0 < min(preview.size) <= max(preview.size) <= 1600
                        if relative == "v6/input-0000/preview.png":
                            assert preview.mode == "RGB"
                            assert preview.info.get("icc_profile")
                        else:
                            assert preview.mode == "I;16"
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
                binding_row = (
                    await connection.execute(
                        text("SELECT execution_bindings, execution_bindings_digest FROM dispatch_attempts WHERE job_id=:job"),
                        {"job": job_id},
                    )
                ).one()
                assert binding_row.execution_bindings == bindings
                assert binding_row.execution_bindings_digest == hashlib.sha256(bindings).hexdigest()
            with pytest.raises(DBAPIError, match="immutable dispatch identity"):
                async with engine.begin() as connection:
                    await connection.execute(text("SELECT set_config('tp.operational_write', 'on', true)"))
                    await connection.execute(
                        text("UPDATE dispatch_attempts SET execution_bindings_digest=repeat('f',64) WHERE job_id=:job"),
                        {"job": job_id},
                    )
            assert await records.fetch_execution_bindings(locator) == bindings
            assert not any(Path(environment["TP_ORCHESTRATOR_EXECUTION_ROOT"]).iterdir())
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


@pytest.mark.parametrize("revocation", ["canceled", "expired", "epoch", "holder"])
async def test_v6_fenced_publication_rejects_cancel_and_stale_postgres_authority(
    service_urls, request_case, tmp_path, monkeypatch, revocation
):
    """Valid output bytes cannot override a canceled or stale execution lease."""
    database_url, redis_url = service_urls
    monkeypatch.setenv("TRANSFORMATION_PORTAL_DA3_PYTHON", sys.executable)
    monkeypatch.delenv("TRANSFORMATION_PORTAL_RAW_PYTHON", raising=False)
    monkeypatch.setenv("TP_ORCHESTRATOR_EXECUTION_ROOT", str(tmp_path / "private"))
    records = PostgresOperationalRecordStore(database_url=database_url)
    broker = RedisLocatorQueueBroker(redis_url=redis_url, key_prefix=f"v6-fence-test-{uuid.uuid4().hex}:dispatch:v1:")
    publisher = ManagedV6PhotographyPublisher(
        artifact_store=LocalArtifactStore(root_dir=tmp_path / "artifacts"), record_store=records
    )
    admitted = prepare_v6_dispatch(ManagedLuxDepthV6Request(request_case), publisher=publisher)
    job_id = "job_v6_fence_" + uuid.uuid4().hex
    record = _record(job_id)
    record.request["pipeline"] = record.effective_request["pipeline"] = "lux-depth-v6"
    locator = None
    try:
        locator = await records.admit(
            record,
            admitted.plan_bytes,
            tenant_id="tenant_a",
            output_root=str(tmp_path / "attempt"),
            requested_output_root=str(request_case.output_dir),
            global_limit=4,
            tenant_limit=2,
            execution_bindings=admitted.bindings_bytes,
        )
        await broker.enqueue(locator)
        await broker.enqueue(locator)
        lease = await broker.acquire_lease("v6-worker", lease_seconds=30)
        assert lease is not None and lease.request == locator
        assert await broker.acquire_lease("v6-duplicate", lease_seconds=30) is None
        fence = await records.claim_dispatch(lease.request, "v6-worker", lease_seconds=30)
        assert fence.lease_epoch > 0
        with pytest.raises(DispatchAuthorityLost):
            await records.claim_dispatch(locator, "v6-duplicate", lease_seconds=30)
        assert await records.fetch_plan(locator) == admitted.plan_bytes
        assert await records.fetch_execution_bindings(locator) == admitted.bindings_bytes
        assert (
            execute_dispatch_plan(
                admitted.plan_bytes,
                execution_bindings=admitted.bindings_bytes,
                output_root=Path(fence.output_root),
            )
            == 0
        )
        assert (Path(fence.output_root) / "v6/input-0000/delivery.tif").is_file()
        if revocation == "canceled":
            assert await records.cancel_dispatch(job_id, "tenant_a")
        elif revocation == "expired":
            engine, _ = await _SharedEngine.get(database_url)
            async with engine.begin() as connection:
                await connection.execute(
                    text(
                        "UPDATE dispatch_attempts SET lease_valid_until=extract(epoch FROM clock_timestamp())-1 WHERE job_id=:job"
                    ),
                    {"job": job_id},
                )
        elif revocation == "epoch":
            fence = replace(fence, lease_epoch=fence.lease_epoch + 1)
        else:
            fence = replace(fence, holder="v6-unclaimed-worker")
        with pytest.raises(DispatchAuthorityLost):
            await records.assert_dispatch_authority(fence)
        # Exercise actual independently verified V6 staging and the final database
        # transaction: staging is never sufficient to make a generation visible.
        with pytest.raises(DispatchAuthorityLost):
            await _publish_admitted_result(admitted.plan_bytes, publisher=publisher, fence=fence)
        assert await records.committed_manifest(job_id, "tenant_a") is None
        engine, _ = await _SharedEngine.get(database_url)
        async with engine.connect() as connection:
            assert (
                await connection.scalar(
                    text("SELECT count(*) FROM operational_records WHERE job_id=:job AND kind='generation_committed'"),
                    {"job": job_id},
                )
                == 0
            )
            assert (
                await connection.scalar(text("SELECT count(*) FROM committed_generations WHERE job_id=:job"), {"job": job_id})
                == 0
            )
    finally:
        if locator is not None:
            await records.cancel_dispatch(job_id, "tenant_a")
        await broker.reset()
        await broker.close()
        await _SharedEngine.dispose(database_url)
