"""Contract tests for the orchestrator backend factories.

Covers the three env-keyed singleton factories — storage (repository /
event store), artifact store, and queue broker — including default
selection, explicit backend selection, missing-configuration errors,
unsupported-backend rejection, singleton caching, and reset.

The alternate backends (postgres / s3 / redis) construct lazily, so the
factory paths are exercised without live infrastructure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import pytest

from transformation_portal.orchestrator import artifact_store as artifact_store_factory
from transformation_portal.orchestrator import queue as queue_factory
from transformation_portal.orchestrator import storage as storage_factory

pytestmark = pytest.mark.unit


_STORAGE_ENVS = ("TP_ORCHESTRATOR_STATE_BACKEND", "TP_DATABASE_URL")
_ARTIFACT_ENVS = (
    "TP_ARTIFACT_STORE",
    "TP_ARTIFACT_BUCKET",
    "TP_ARTIFACT_PREFIX",
    "TP_ARTIFACT_ENDPOINT_URL",
    "TP_ARTIFACT_REGION",
    "TP_ARTIFACT_LOCAL_ROOT",
)
_QUEUE_ENVS = ("TP_ORCHESTRATOR_QUEUE_BACKEND", "TP_REDIS_URL")


@pytest.fixture(autouse=True)
def _isolate_factories(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Clear factory env vars and singletons before and after each test."""
    for name in _STORAGE_ENVS + _ARTIFACT_ENVS + _QUEUE_ENVS:
        monkeypatch.delenv(name, raising=False)
    storage_factory.reset_singletons()
    artifact_store_factory.reset_singleton()
    queue_factory.reset_singleton()
    yield
    storage_factory.reset_singletons()
    artifact_store_factory.reset_singleton()
    queue_factory.reset_singleton()


class TestStorageFactory:
    """Tests for transformation_portal.orchestrator.storage factory."""

    def test_defaults_to_memory_repository(self) -> None:
        from transformation_portal.orchestrator.storage.memory import MemoryJobRepository

        repo = storage_factory.get_job_repository()
        assert isinstance(repo, MemoryJobRepository)

    def test_defaults_to_memory_event_store(self) -> None:
        from transformation_portal.orchestrator.storage.memory import MemoryJobEventStore

        events = storage_factory.get_job_event_store()
        assert isinstance(events, MemoryJobEventStore)

    def test_operational_audit_requires_postgres_backend(self) -> None:
        with pytest.raises(RuntimeError, match="operational audit requires"):
            storage_factory.get_operational_audit_store()

    def test_explicit_memory_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TP_ORCHESTRATOR_STATE_BACKEND", "MEMORY")  # case-insensitive
        from transformation_portal.orchestrator.storage.memory import MemoryJobRepository

        assert isinstance(storage_factory.get_job_repository(), MemoryJobRepository)

    def test_repository_is_cached_singleton(self) -> None:
        assert storage_factory.get_job_repository() is storage_factory.get_job_repository()

    def test_event_store_is_cached_singleton(self) -> None:
        assert storage_factory.get_job_event_store() is storage_factory.get_job_event_store()

    def test_reset_singletons_drops_cache(self) -> None:
        first = storage_factory.get_job_repository()
        storage_factory.reset_singletons()
        assert storage_factory.get_job_repository() is not first

    def test_postgres_without_database_url_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TP_ORCHESTRATOR_STATE_BACKEND", "postgres")

        with pytest.raises(RuntimeError, match="TP_DATABASE_URL"):
            storage_factory.get_job_repository()
        with pytest.raises(RuntimeError, match="TP_DATABASE_URL"):
            storage_factory.get_job_event_store()

    def test_postgres_with_database_url_constructs_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TP_ORCHESTRATOR_STATE_BACKEND", "postgres")
        monkeypatch.setenv("TP_DATABASE_URL", "postgresql+asyncpg://user:pw@host:5432/db")
        from transformation_portal.orchestrator.storage.postgres import (
            PostgresJobEventStore,
            PostgresJobRepository,
            PostgresOperationalAuditStore,
        )

        assert isinstance(storage_factory.get_job_repository(), PostgresJobRepository)
        assert isinstance(storage_factory.get_job_event_store(), PostgresJobEventStore)
        assert isinstance(storage_factory.get_operational_audit_store(), PostgresOperationalAuditStore)

    @pytest.mark.parametrize(
        ("factory_name", "class_name"),
        [
            ("get_job_repository", "PostgresJobRepository"),
            ("get_job_event_store", "PostgresJobEventStore"),
            ("get_operational_audit_store", "PostgresOperationalAuditStore"),
        ],
    )
    def test_postgres_backend_missing_sql_dependencies_reports_install_guidance(
        self,
        monkeypatch: pytest.MonkeyPatch,
        factory_name: str,
        class_name: str,
    ) -> None:
        import builtins

        monkeypatch.setenv("TP_ORCHESTRATOR_STATE_BACKEND", "postgres")
        monkeypatch.setenv("TP_DATABASE_URL", "postgresql+asyncpg://user:pw@host:5432/db")
        real_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: ANN001, ANN202
            if name == "transformation_portal.orchestrator.storage.postgres" and class_name in fromlist:
                raise ImportError("sqlalchemy unavailable")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

        with pytest.raises(RuntimeError) as exc_info:
            getattr(storage_factory, factory_name)()

        message = str(exc_info.value)
        assert "requires sqlalchemy[asyncio] + asyncpg" in message
        assert "make install-core" in message

    def test_postgres_record_projection_isolates_nested_json_fields(self) -> None:
        from transformation_portal.orchestrator.models import JobArtifactModel, JobModel
        from transformation_portal.orchestrator.storage.postgres import _record_from_model

        model = JobModel(
            id="job-pg-projection",
            created_at=1.0,
            state="running",
            progress=50,
            request={"args": {"quality": "premium"}},
            effective_request={"args": {"resolved_backend": "da3"}},
            logs_tail=["line-1"],
            artifacts={"items": [{"relative_path": "report.json"}]},
            run_summary={"counts": {"succeeded": 1}},
            error={"details": {"code": "original"}},
            version=1,
        )
        model.artifact_index = [
            JobArtifactModel(
                job_id=model.id,
                path="out/report.json",
                absolute_path="/abs/out/report.json",
            )
        ]

        record = _record_from_model(model)
        record.request["args"]["quality"] = "record-mutated"
        record.effective_request["args"]["resolved_backend"] = "record-mutated"
        record.artifacts["items"][0]["relative_path"] = "record-mutated.json"
        record.run_summary["counts"]["succeeded"] = 99
        assert record.error is not None
        record.error["details"]["code"] = "record-mutated"
        record.logs_tail.append("record-mutated")

        reprojected = _record_from_model(model)

        assert reprojected.request == {"args": {"quality": "premium"}}
        assert reprojected.effective_request == {"args": {"resolved_backend": "da3"}}
        assert reprojected.artifacts == {"items": [{"relative_path": "report.json"}]}
        assert reprojected.run_summary == {"counts": {"succeeded": 1}}
        assert reprojected.error == {"details": {"code": "original"}}
        assert reprojected.logs_tail == ["line-1"]
        assert reprojected.artifact_lookup == {"out/report.json": Path("/abs/out/report.json")}

        model.request["args"]["quality"] = "model-mutated"
        model.effective_request["args"]["resolved_backend"] = "model-mutated"
        model.artifacts["items"][0]["relative_path"] = "model-mutated.json"
        model.run_summary["counts"]["succeeded"] = 100
        assert model.error is not None
        model.error["details"]["code"] = "model-mutated"
        model.logs_tail.append("model-mutated")

        assert reprojected.request == {"args": {"quality": "premium"}}
        assert reprojected.effective_request == {"args": {"resolved_backend": "da3"}}
        assert reprojected.artifacts == {"items": [{"relative_path": "report.json"}]}
        assert reprojected.run_summary == {"counts": {"succeeded": 1}}
        assert reprojected.error == {"details": {"code": "original"}}
        assert reprojected.logs_tail == ["line-1"]

    def test_unsupported_backend_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TP_ORCHESTRATOR_STATE_BACKEND", "cassandra")

        with pytest.raises(RuntimeError, match="Unsupported"):
            storage_factory.get_job_repository()
        with pytest.raises(RuntimeError, match="Unsupported"):
            storage_factory.get_job_event_store()


class TestArtifactStoreFactory:
    """Tests for transformation_portal.orchestrator.artifact_store factory."""

    def test_defaults_to_local_backend(self) -> None:
        from transformation_portal.orchestrator.artifact_store.local import LocalArtifactStore

        store = artifact_store_factory.get_artifact_store()
        assert isinstance(store, LocalArtifactStore)

    def test_local_backend_honours_custom_root(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.setenv("TP_ARTIFACT_STORE", "local")
        monkeypatch.setenv("TP_ARTIFACT_LOCAL_ROOT", str(tmp_path / "artifacts"))
        from transformation_portal.orchestrator.artifact_store.local import LocalArtifactStore

        assert isinstance(artifact_store_factory.get_artifact_store(), LocalArtifactStore)

    def test_store_is_cached_singleton(self) -> None:
        assert artifact_store_factory.get_artifact_store() is artifact_store_factory.get_artifact_store()

    def test_reset_singleton_drops_cache(self) -> None:
        first = artifact_store_factory.get_artifact_store()
        artifact_store_factory.reset_singleton()
        assert artifact_store_factory.get_artifact_store() is not first

    def test_s3_without_bucket_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TP_ARTIFACT_STORE", "s3")

        with pytest.raises(artifact_store_factory.ArtifactStoreError, match="TP_ARTIFACT_BUCKET"):
            artifact_store_factory.get_artifact_store()

    def test_s3_with_bucket_constructs_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TP_ARTIFACT_STORE", "s3")
        monkeypatch.setenv("TP_ARTIFACT_BUCKET", "tp-artifacts")
        monkeypatch.setenv("TP_ARTIFACT_ENDPOINT_URL", "https://minio.example")
        monkeypatch.setenv("TP_ARTIFACT_REGION", "us-east-1")
        from transformation_portal.orchestrator.artifact_store.s3 import S3ArtifactStore

        store = artifact_store_factory.get_artifact_store()
        assert isinstance(store, S3ArtifactStore)
        assert store.backend == "s3"

    def test_unsupported_backend_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TP_ARTIFACT_STORE", "gcs")

        with pytest.raises(artifact_store_factory.ArtifactStoreError, match="Unsupported"):
            artifact_store_factory.get_artifact_store()


class TestQueueFactory:
    """Tests for transformation_portal.orchestrator.queue factory."""

    def test_defaults_to_memory_broker(self) -> None:
        from transformation_portal.orchestrator.queue.memory import MemoryQueueBroker

        broker = queue_factory.get_queue_broker()
        assert isinstance(broker, MemoryQueueBroker)

    def test_explicit_memory_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TP_ORCHESTRATOR_QUEUE_BACKEND", "memory")
        from transformation_portal.orchestrator.queue.memory import MemoryQueueBroker

        assert isinstance(queue_factory.get_queue_broker(), MemoryQueueBroker)

    def test_broker_is_cached_singleton(self) -> None:
        assert queue_factory.get_queue_broker() is queue_factory.get_queue_broker()

    def test_reset_singleton_drops_cache(self) -> None:
        first = queue_factory.get_queue_broker()
        queue_factory.reset_singleton()
        assert queue_factory.get_queue_broker() is not first

    def test_redis_without_url_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TP_ORCHESTRATOR_QUEUE_BACKEND", "redis")

        with pytest.raises(RuntimeError, match="TP_REDIS_URL"):
            queue_factory.get_queue_broker()

    def test_redis_with_url_constructs_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TP_ORCHESTRATOR_QUEUE_BACKEND", "redis")
        monkeypatch.setenv("TP_REDIS_URL", "redis://localhost:6379/0")
        from transformation_portal.orchestrator.queue.redis import RedisQueueBroker

        assert isinstance(queue_factory.get_queue_broker(), RedisQueueBroker)

    def test_unsupported_backend_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TP_ORCHESTRATOR_QUEUE_BACKEND", "rabbitmq")

        with pytest.raises(RuntimeError, match="Unsupported"):
            queue_factory.get_queue_broker()
