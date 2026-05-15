"""Unit tests for the orchestrator in-process runtime handle registry.

RuntimeHandles / RuntimeRegistry hold non-persistent per-job state
(subprocess + tasks) that must never reach the durable JobRepository.
These are pure in-memory structures, exercised directly.
"""

from __future__ import annotations

from typing import Generator

import pytest

from transformation_portal.orchestrator.runtime_handles import (
    RuntimeHandles,
    RuntimeRegistry,
    get_runtime_registry,
    reset_runtime_registry,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_registry() -> Generator[None, None, None]:
    reset_runtime_registry()
    yield
    reset_runtime_registry()


class _FakeProc:
    """Stand-in for asyncio.subprocess.Process exposing only returncode."""

    def __init__(self, returncode: object = None) -> None:
        self.returncode = returncode


class TestRuntimeHandles:
    """Tests for the RuntimeHandles dataclass."""

    def test_defaults_are_empty(self) -> None:
        handles = RuntimeHandles()
        assert handles.proc is None
        assert handles.terminate_task is None
        assert handles.subscribers == {}

    def test_subscribers_dict_is_not_shared(self) -> None:
        first = RuntimeHandles()
        second = RuntimeHandles()
        first.subscribers["a"] = object()  # type: ignore[assignment]
        assert second.subscribers == {}


class TestRuntimeRegistry:
    """Tests for the RuntimeRegistry."""

    def test_ensure_creates_then_returns_same_handle(self) -> None:
        registry = RuntimeRegistry()

        created = registry.ensure("job-1")
        assert isinstance(created, RuntimeHandles)
        assert registry.ensure("job-1") is created

    def test_get_returns_none_for_unknown(self) -> None:
        assert RuntimeRegistry().get("missing") is None

    def test_get_returns_existing_handle(self) -> None:
        registry = RuntimeRegistry()
        handle = registry.ensure("job-1")
        assert registry.get("job-1") is handle

    def test_has_reflects_membership(self) -> None:
        registry = RuntimeRegistry()
        assert registry.has("job-1") is False
        registry.ensure("job-1")
        assert registry.has("job-1") is True

    def test_pop_removes_and_returns_handle(self) -> None:
        registry = RuntimeRegistry()
        handle = registry.ensure("job-1")

        assert registry.pop("job-1") is handle
        assert registry.has("job-1") is False
        assert registry.pop("job-1") is None

    def test_clear_drops_everything(self) -> None:
        registry = RuntimeRegistry()
        registry.ensure("job-1")
        registry.ensure("job-2")

        registry.clear()

        assert registry.has("job-1") is False
        assert registry.has("job-2") is False

    def test_live_job_ids_only_includes_running_processes(self) -> None:
        registry = RuntimeRegistry()
        # No process attached -> not live.
        registry.ensure("idle")
        # Running process (returncode is None) -> live.
        registry.ensure("running").proc = _FakeProc(returncode=None)
        # Finished process (returncode set) -> not live.
        registry.ensure("done").proc = _FakeProc(returncode=0)

        assert registry.live_job_ids() == ["running"]

    def test_live_job_ids_empty_when_no_jobs(self) -> None:
        assert RuntimeRegistry().live_job_ids() == []


class TestRuntimeRegistrySingleton:
    """Tests for the module-level registry singleton accessors."""

    def test_get_runtime_registry_returns_singleton(self) -> None:
        assert get_runtime_registry() is get_runtime_registry()

    def test_reset_runtime_registry_drops_singleton(self) -> None:
        first = get_runtime_registry()
        reset_runtime_registry()
        assert get_runtime_registry() is not first
