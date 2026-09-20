"""App-independent managed worker ports for the shared execution service."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from transformation_portal.orchestrator import execution_process
from transformation_portal.orchestrator.artifact_limits import configured_max_indexed_artifacts
from transformation_portal.orchestrator.artifact_store import get_artifact_store
from transformation_portal.orchestrator.dispatch import current_dispatch_fence
from transformation_portal.orchestrator.execution_summary import refresh_job_run_summary
from transformation_portal.orchestrator.job_execution import (
    ExecutionJob,
    ExecutionSettings,
    JobExecutionRuntime,
    JobExecutionService,
)
from transformation_portal.orchestrator.storage import (
    get_job_event_store,
    get_job_repository,
    get_operational_record_store,
)
from transformation_portal.orchestrator.storage.base import JobRecord
from transformation_portal.portal import job_artifacts

LOGGER = logging.getLogger(__name__)


class ExecutionRepositoryUnavailable(RuntimeError):
    """The worker cannot safely hydrate its durable projection."""


@dataclass
class WorkerJob(JobRecord):
    """Job projection plus process-local handles, never serialized as a whole."""

    proc: asyncio.subprocess.Process | None = None
    terminate_task: asyncio.Task[None] | None = None

    def add_log(self, line: str, limit: int = 2000) -> None:
        self.logs_tail.append(line)
        if len(self.logs_tail) > limit:
            del self.logs_tail[:-limit]


def job_from_record(record: JobRecord) -> WorkerJob:
    return WorkerJob(**vars(record.copy()))


async def load_record(job_id: str) -> JobRecord | None:
    try:
        return await get_job_repository().get(job_id)
    except Exception as exc:
        raise ExecutionRepositoryUnavailable("job repository read failed") from exc


async def load_job(job_id: str) -> WorkerJob | None:
    record = await load_record(job_id)
    return None if record is None else job_from_record(record)


async def persist_fields(job: ExecutionJob, context: str, **fields: Any) -> bool:
    # Operational transactions exclusively own state and publication fields.
    fields = {key: value for key, value in fields.items() if key in {"progress", "logs_tail", "last_event_at"}}
    if not fields:
        return True
    try:
        await get_job_repository().update(job.id, **fields)
    except Exception:
        LOGGER.debug("worker telemetry %s persist failed for %s", context, job.id, exc_info=True)
        return False
    return True


class ManagedExecutionEvents:
    """Persist bounded replay observations without stealing API outbox delivery."""

    def __init__(self) -> None:
        self._last_persisted: dict[str, float] = {}

    def cache_job(self, job: ExecutionJob) -> ExecutionJob:
        if job.finished_at is not None or job.done_published_at is not None:
            self._last_persisted.pop(job.id, None)
        return job

    async def publish_event(self, job_id: str, event: str, payload: dict[str, Any]) -> None:
        # Terminal/artifact events belong to the publication transaction. This
        # port receives only running/log/progress observations from the runner.
        timestamp = time.time()
        try:
            await get_job_event_store().append(job_id, event, payload, created_at=timestamp)
        except Exception:
            LOGGER.debug("worker event persist failed for %s", job_id, exc_info=True)
        last = self._last_persisted.get(job_id)
        if event == "state" or last is None or timestamp - last >= 5.0:
            try:
                await get_job_repository().update(job_id, last_event_at=timestamp)
            except Exception:
                LOGGER.debug("worker last_event_at persist failed for %s", job_id, exc_info=True)
            else:
                self._last_persisted[job_id] = timestamp

    def release_job(self, job_id: str) -> None:
        self._last_persisted.pop(job_id, None)


def _output_root(job: ExecutionJob) -> Path:
    fence = current_dispatch_fence()
    if fence is None or fence.locator.job_id != job.id:
        raise RuntimeError("managed worker artifacts require the matching dispatch fence")
    return Path(fence.output_root)


def _configured_fingerprint_bytes() -> int:
    try:
        return max(1024, int(os.getenv("TP_ARTIFACT_FINGERPRINT_MAX_BYTES", str(8 * 1024 * 1024))))
    except ValueError:
        return 8 * 1024 * 1024


def index_artifacts(job: ExecutionJob) -> list[dict[str, Any]]:
    result = job_artifacts._index_job_artifacts(
        job_id=job.id,
        output_dir=_output_root(job),
        max_indexed_artifacts=configured_max_indexed_artifacts(),
        fingerprint_max_bytes=_configured_fingerprint_bytes(),
        fingerprint_chunk_bytes=1024 * 1024,
        run_summary_max_bytes=1024 * 1024,
    )
    job.artifacts, job.artifact_lookup = result.artifacts, result.artifact_lookup
    return result.items


def refresh_summary(job: ExecutionJob) -> dict[str, Any]:
    return refresh_job_run_summary(job, _output_root(job))


async def _unused_legacy_port(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError("unfenced legacy execution is disabled in external workers")


async def _leave_outbox_for_api() -> None:
    """Keep committed event delivery pending for API subscriber fanout."""


def create_managed_execution_service() -> JobExecutionService[ExecutionJob]:
    """Build independent worker ports without loading the ASGI application."""
    from transformation_portal.orchestrator.execution_policy import load_execution_policy

    policy = load_execution_policy()
    events = ManagedExecutionEvents()
    try:
        grace = max(0.1, float(os.getenv("TP_CANCEL_GRACE_SECONDS", "5")))
    except ValueError:
        grace = 5.0
    return JobExecutionService(
        JobExecutionRuntime(
            load_job=load_job,
            cache_job=events.cache_job,
            repository=get_job_repository,
            repository_unavailable=ExecutionRepositoryUnavailable,
            persist_fields=persist_fields,
            publish_event=events.publish_event,
            now=time.time,
            child_environment=execution_process._sanitized_child_env,
            owned_process_group=execution_process._owned_process_group_id,
            terminate_process=partial(execution_process._terminate_process, grace_seconds=grace),
            redact_log=execution_process._redact_log_line,
            extract_progress=execution_process._extract_progress_percent,
            index_artifacts=index_artifacts,
            refresh_summary=refresh_summary,
            artifact_store=get_artifact_store,
            operational_records=get_operational_record_store,
            flush_outbox=_leave_outbox_for_api,
            load_record=load_record,
            job_from_record=job_from_record,
            mirror_artifacts=_unused_legacy_port,
            persist_state=_unused_legacy_port,
            cleanup_jobs=_unused_legacy_port,
            validate_paths=policy.validate_dispatch_paths,
            resolve_output_root=policy.resolve_output_root,
            release_job=events.release_job,
        ),
        ExecutionSettings(cancel_grace_seconds=grace),
    )
