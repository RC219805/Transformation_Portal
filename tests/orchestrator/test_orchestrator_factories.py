"""Contract tests for the orchestrator backend factories.

Covers the three env-keyed singleton factories — storage (repository /
event store), artifact store, and queue broker — including default
selection, explicit backend selection, missing-configuration errors,
unsupported-backend rejection, singleton caching, and reset.

The alternate backends (postgres / s3 / redis) construct lazily, so the
factory paths are exercised without live infrastructure.
"""

from __future__ import annotations

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
        )

        assert isinstance(storage_factory.get_job_repository(), PostgresJobRepository)
        assert isinstance(storage_factory.get_job_event_store(), PostgresJobEventStore)

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
