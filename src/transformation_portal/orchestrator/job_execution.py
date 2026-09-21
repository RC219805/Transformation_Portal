"""Shared job execution coordinator for HTTP and standalone orchestrator workers.

The service owns child lifetime, cancellation, dispatch workspaces and fenced
publication. Its explicit ports isolate API runtime/SSE projection from the
coordinator: no HTTP application is imported to execute a job.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import suppress
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Coroutine, Dict, Generic, List, Protocol, TypeVar

from sqlalchemy.exc import SQLAlchemyError

from transformation_portal.orchestrator.artifact_store.base import ArtifactStore
from transformation_portal.orchestrator.artifact_store.generation import GenerationPublisher
from transformation_portal.orchestrator.artifact_store.generation_cleanup import cleanup_terminal_generation
from transformation_portal.orchestrator.dispatch import DispatchFence, DispatchLocator, current_dispatch_fence
from transformation_portal.orchestrator.queue.base import JobEnqueueRequest
from transformation_portal.orchestrator.storage.base import JobRecord, JobRepository, RepositoryError
from transformation_portal.orchestrator.storage.operational import DispatchAuthorityLost
from transformation_portal.orchestrator.worker import RetryableExecutorUnavailable

LOGGER = logging.getLogger(__name__)
TERMINAL_JOB_STATES = frozenset({"succeeded", "partial", "failed", "canceled", "worker_lost"})
_active_plan: ContextVar[bytes | None] = ContextVar("job_execution_plan", default=None)


class ExecutionJob(Protocol):
    """Mutable worker projection; live handles are never persisted."""

    id: str
    state: str
    created_at: float
    started_at: float | None
    finished_at: float | None
    done_published_at: float | None
    last_event_at: float | None
    cancel_requested: bool
    progress: int
    exit_code: int | None
    request: dict[str, Any]
    effective_request: dict[str, Any]
    logs_tail: list[str]
    artifacts: dict[str, Any]
    artifact_lookup: dict[str, Path]
    run_summary: dict[str, Any]
    error: dict[str, Any] | None
    proc: asyncio.subprocess.Process | None
    terminate_task: asyncio.Task[None] | None

    def add_log(self, line: str, limit: int = 2000) -> None: ...


_JobT = TypeVar("_JobT", bound=ExecutionJob)


@dataclass(frozen=True)
class ExecutionSettings:
    """Explicit bounded runner telemetry and shutdown settings."""

    log_tail_limit: int = 2000
    log_batch_size: int = 25
    log_flush_interval: float = 1.0
    progress_flush_interval: float = 1.0
    cancel_grace_seconds: float = 5.0


@dataclass(frozen=True)
class JobExecutionRuntime(Generic[_JobT]):
    """Concrete ports; HTTP and standalone runtimes share one execution loop."""

    load_job: Callable[[str], Awaitable[_JobT | None]]
    cache_job: Callable[[_JobT], _JobT]
    repository: Callable[[], JobRepository]
    repository_unavailable: type[Exception]
    persist_fields: Callable[..., Awaitable[bool]]
    publish_event: Callable[[str, str, dict[str, Any]], Awaitable[None]]
    now: Callable[[], float]
    child_environment: Callable[[], dict[str, str]]
    owned_process_group: Callable[[asyncio.subprocess.Process], int | None]
    terminate_process: Callable[[asyncio.subprocess.Process], Coroutine[Any, Any, None]]
    redact_log: Callable[[str], str]
    extract_progress: Callable[[str], int | None]
    index_artifacts: Callable[[_JobT], list[dict[str, Any]]]
    refresh_summary: Callable[[_JobT], dict[str, Any]]
    artifact_store: Callable[[], ArtifactStore]
    operational_records: Callable[[], Any]
    flush_outbox: Callable[[], Awaitable[None]]
    load_record: Callable[[str], Awaitable[JobRecord | None]]
    job_from_record: Callable[[JobRecord], _JobT]
    mirror_artifacts: Callable[[_JobT], Awaitable[object]]
    persist_state: Callable[[_JobT], Awaitable[None]]
    cleanup_jobs: Callable[..., Awaitable[Any]]
    validate_paths: Callable[..., None]
    resolve_output_root: Callable[[str], Path]
    allow_legacy_commands: bool = False
    release_job: Callable[[str], None] | None = None


def execution_error(
    code: str, message: str, details: dict[str, Any] | None = None, *, retriable: bool | None = None
) -> dict[str, Any]:
    result: dict[str, Any] = {"code": code, "message": message, "details": details or {}}
    if retriable is not None:
        result["retriable"] = retriable
    return result


class JobExecutionService(Generic[_JobT]):
    """The sole coordinator for accepted jobs, independent of the HTTP facade."""

    def __init__(self, runtime: JobExecutionRuntime[_JobT], settings: ExecutionSettings | None = None) -> None:
        self.runtime = runtime
        self.settings = settings or ExecutionSettings()

    async def execute(
        self,
        request: JobEnqueueRequest | DispatchLocator,
        cancellation_event: "asyncio.Event",
        *,
        run_job: Callable[[_JobT, list[str]], Awaitable[None]] | None = None,
    ) -> int:
        """Hydrate one claimed job, freeze its workspace and run with cancellation."""
        try:
            job = await self.runtime.load_job(request.job_id)
        except self.runtime.repository_unavailable:
            LOGGER.exception("worker could not load job_id=%s from repository", request.job_id)
            raise RetryableExecutorUnavailable(request.job_id)
        if job is None:
            LOGGER.warning("worker leased unknown job_id=%s; releasing without dispatch", request.job_id)
            return 0
        self.runtime.cache_job(job)
        if job.finished_at is not None or job.state in TERMINAL_JOB_STATES:
            LOGGER.info(
                "worker leased already-terminal job_id=%s state=%s; releasing without dispatch",
                request.job_id,
                job.state,
            )
            return 0

        async def _bridge_cancel() -> None:
            await cancellation_event.wait()
            # Signal the live subprocess immediately: it may be silent while
            # ``_run_job`` is awaiting stdout and never start another iteration.
            if not job.cancel_requested:
                job.cancel_requested = True
            if job.proc is not None and (
                job.proc.returncode is None or self.runtime.owned_process_group(job.proc) is not None
            ):
                if job.terminate_task is None or job.terminate_task.done():
                    job.terminate_task = asyncio.create_task(self.runtime.terminate_process(job.proc))

        bridge_task = asyncio.create_task(_bridge_cancel())
        execution_output_root = None
        plan_token = None
        try:
            if isinstance(request, DispatchLocator):
                from transformation_portal.orchestrator.execution_dispatch import (
                    _pin_output_directory,
                    command_from_dispatch_plan,
                    validate_dispatch_plan,
                )
                from transformation_portal.orchestrator.execution_workspace import create_execution_workspace

                fence = current_dispatch_fence()
                if fence is None or fence.locator != request:
                    raise DispatchAuthorityLost("worker has no matching claim")
                try:
                    records = self.runtime.operational_records()
                    raw = await records.fetch_plan(request)
                    bindings = await records.fetch_execution_bindings(request)
                except DispatchAuthorityLost:
                    raise
                except (OSError, RepositoryError, SQLAlchemyError, self.runtime.repository_unavailable) as exc:
                    LOGGER.exception("worker could not read execution authority for job_id=%s", request.job_id)
                    raise RetryableExecutorUnavailable(request.job_id) from exc
                output_root = Path(fence.output_root)
                self.runtime.validate_paths(request, raw, output_root, execution_bindings=bindings)
                resolved_output = self.runtime.resolve_output_root(fence.output_root)
                if resolved_output != output_root:
                    raise DispatchAuthorityLost("admitted output root changed before worker pickup")
                plan = validate_dispatch_plan(raw, execution_bindings=bindings)
                if plan.schema == "tp.execution.plan.v4":
                    from transformation_portal.lux_depth_v5.publication import validate_publication_plan

                    publisher = GenerationPublisher(
                        artifact_store=self.runtime.artifact_store(), record_store=self.runtime.operational_records()
                    )
                    validate_publication_plan(plan.to_payload(), publisher.limits)
                # An attempt root is created once, only after a committed claim.
                parent = _pin_output_directory(output_root.parent)
                try:
                    os.mkdir(output_root.name, mode=0o700, dir_fd=parent)
                finally:
                    os.close(parent)
                workspace = create_execution_workspace(output_root, require_configured=True)
                execution_output_root = output_root
                plan_path = workspace / "plan.json"
                plan_path.write_bytes(raw)
                bindings_path = None
                if bindings is not None:
                    bindings_path = workspace / "bindings.json"
                    bindings_path.write_bytes(bindings)
                command = command_from_dispatch_plan(
                    raw,
                    output_root=output_root,
                    plan_path=plan_path,
                    execution_workspace=workspace,
                    execution_bindings=bindings,
                    bindings_path=bindings_path,
                )
                plan_token = _active_plan.set(raw)
            else:
                if not self.runtime.allow_legacy_commands:
                    raise DispatchAuthorityLost("external workers require an immutable dispatch locator")
                command = list(request.argv)
            if cancellation_event.is_set():
                return 0
            await (run_job or self.run)(job, command)
        finally:
            if plan_token is not None:
                _active_plan.reset(plan_token)
            if execution_output_root is not None:
                from transformation_portal.orchestrator.execution_workspace import remove_execution_workspace

                try:
                    remove_execution_workspace(execution_output_root)
                except Exception:
                    LOGGER.exception("private execution cleanup deferred for %s", request.job_id)
            bridge_task.cancel()
            with suppress(asyncio.CancelledError):
                await bridge_task
            if self.runtime.release_job is not None:
                self.runtime.release_job(request.job_id)
        return int(job.exit_code) if job.exit_code is not None else 0

    async def _publish_fenced_result(self, job: _JobT, fence: DispatchFence) -> None:
        """Select a closed publication adapter; successful V5 output must verify."""
        from transformation_portal.core.execution_plan import decode_bounded_json_object

        raw = _active_plan.get()
        schema = None if raw is None else decode_bounded_json_object(raw).get("schema")

        publishing_success = job.state in {"succeeded", "partial"}

        def require_publication_authority() -> None:
            if job.cancel_requested and (publishing_success or job.state in {"succeeded", "partial"}):
                raise DispatchAuthorityLost("publication canceled before generation commit")

        publisher_type = GenerationPublisher
        if schema == "tp.execution.plan.v4":
            from transformation_portal.orchestrator.photography_adapter import ManagedPhotographyPublisher

            publisher_type = ManagedPhotographyPublisher
        publisher = publisher_type(
            artifact_store=self.runtime.artifact_store(),
            record_store=self.runtime.operational_records(),
            publication_guard=require_publication_authority,
        )
        if schema == "tp.execution.plan.v4" and raw is not None:
            if job.state != "succeeded" or job.exit_code != 0:
                await self.runtime.operational_records().finish_dispatch(
                    fence, state=job.state, exit_code=job.exit_code, error=job.error
                )
                return
            from transformation_portal.lux_depth_v5.publication import _publish_admitted_result

            await _publish_admitted_result(raw, publisher=publisher, fence=fence)
            return
        await asyncio.to_thread(self.runtime.index_artifacts, job)
        self.runtime.refresh_summary(job)
        await publisher.publish(
            fence,
            job.artifact_lookup,
            state=job.state,
            exit_code=job.exit_code,
            artifacts=job.artifacts,
            run_summary=job.run_summary,
            error=job.error,
        )

    async def run(self, job: _JobT, argv: list[str]) -> None:
        if not self.runtime.allow_legacy_commands:
            fence = current_dispatch_fence()
            if fence is None or fence.locator.job_id != job.id or _active_plan.get() is None:
                raise DispatchAuthorityLost("managed execution requires a prepared dispatch context")
        self.runtime.cache_job(job)

        dispatch_authority_lost = False
        retryable_startup_error: RetryableExecutorUnavailable | None = None
        pending_log_lines: List[str] = []
        last_log_flush_at = self.runtime.now()
        pending_progress_persist = False
        last_progress_flush_at = self.runtime.now()

        async def _flush_log_batch() -> None:
            nonlocal last_log_flush_at
            if not pending_log_lines:
                return
            batch = list(pending_log_lines)
            pending_log_lines.clear()
            last_log_flush_at = self.runtime.now()
            try:
                await self.runtime.repository().append_logs(job.id, batch, tail_limit=self.settings.log_tail_limit)
            except Exception:  # noqa: BLE001 - keep the runner moving if log persistence lags
                LOGGER.exception("job repository append_logs failed for %s", job.id)

        async def _flush_progress(*, force: bool = False) -> None:
            nonlocal last_progress_flush_at, pending_progress_persist
            if not pending_progress_persist:
                return
            now = self.runtime.now()
            if not force and self.settings.progress_flush_interval > 0:
                if now - last_progress_flush_at < self.settings.progress_flush_interval:
                    return
            if await self.runtime.persist_fields(job, "progress", progress=job.progress):
                pending_progress_persist = False
                last_progress_flush_at = now

        try:
            if job.done_published_at is not None or job.finished_at is not None:
                return
            if job.cancel_requested:
                job.state = "canceled"
                return
            job.state = "running"
            job.started_at = self.runtime.now()
            await self.runtime.persist_fields(job, "runner_start", state=job.state, started_at=job.started_at)
            await self.runtime.publish_event(job.id, "state", {"id": job.id, "state": job.state})
            # Startup persistence yields to cancellation and lease watchdogs. Do
            # not launch native work after either path has withdrawn authority.
            if job.done_published_at is not None or job.finished_at is not None:
                return
            if job.cancel_requested:
                job.state = "canceled"
                return
            # NO SHELL EXECUTION.
            spawn_kwargs: Dict[str, Any] = {
                "stdout": asyncio.subprocess.PIPE,
                "stderr": asyncio.subprocess.STDOUT,
                "env": self.runtime.child_environment(),
            }
            if os.name != "nt":
                # Put the runner in its own session so cancel can signal the
                # whole process tree, not just the direct child.
                spawn_kwargs["start_new_session"] = True
            fence = current_dispatch_fence()
            if fence is not None:
                if fence.locator.job_id != job.id:
                    raise DispatchAuthorityLost("worker has no matching dispatch claim")
                try:
                    await self.runtime.operational_records().assert_dispatch_authority(fence)
                except DispatchAuthorityLost:
                    raise
                except (OSError, RepositoryError, SQLAlchemyError, self.runtime.repository_unavailable) as exc:
                    raise RetryableExecutorUnavailable(job.id) from exc
                # The authority read yields to the broker cancellation bridge.
                if job.cancel_requested:
                    job.state = "canceled"
                    return
            proc = await asyncio.create_subprocess_exec(*argv, **spawn_kwargs)
            if spawn_kwargs.get("start_new_session"):
                # Spawn itself establishes ownership even if the group leader
                # exits before a later cancellation observes its descendants.
                setattr(proc, "_tp_owned_process_group_id", proc.pid)
            job.proc = proc

            if proc.stdout is None:
                raise RuntimeError("failed to capture subprocess stdout")

            while True:
                if (
                    job.cancel_requested
                    and (proc.returncode is None or self.runtime.owned_process_group(proc) is not None)
                    and (job.terminate_task is None or job.terminate_task.done())
                ):
                    job.terminate_task = asyncio.create_task(
                        self.runtime.terminate_process(proc),
                    )

                raw_line = await proc.stdout.readline()
                if not raw_line:
                    break

                line = raw_line.decode(
                    "utf-8",
                    errors="replace",
                ).rstrip("\n")
                line = self.runtime.redact_log(line)
                job.add_log(line)
                pending_log_lines.append(line)
                if (
                    len(pending_log_lines) >= self.settings.log_batch_size
                    or self.runtime.now() - last_log_flush_at >= self.settings.log_flush_interval
                ):
                    await _flush_log_batch()
                await self.runtime.publish_event(
                    job.id,
                    "log",
                    {"id": job.id, "line": line},
                )

                pct = self.runtime.extract_progress(line)
                if pct is not None and pct != job.progress:
                    job.progress = pct
                    pending_progress_persist = True
                    await _flush_progress()
                    await self.runtime.publish_event(
                        job.id,
                        "progress",
                        {"id": job.id, "progress": job.progress},
                    )

            await _flush_log_batch()
            await _flush_progress(force=True)
            rc = await proc.wait()
            if job.done_published_at is not None or job.finished_at is not None:
                return
            job.exit_code = int(rc)
            if job.cancel_requested:
                job.state = "canceled"
            else:
                job.state = "succeeded" if rc == 0 else "failed"
                if rc != 0:
                    # Phase 2.D — ``retriable=False`` distinguishes executor-level
                    # failures (the work itself is broken) from the broker-level
                    # ``worker_lost`` state (the worker died holding the lease).
                    job.error = execution_error(
                        "RUNNER_EXIT_NONZERO",
                        f"runner exited with code {rc}",
                        {"exit_code": int(rc)},
                        retriable=False,
                    )
            await self.runtime.persist_fields(
                job, "runner_terminal", state=job.state, exit_code=job.exit_code, error=job.error
            )

        except asyncio.CancelledError:
            # Lifespan shutdown cancels worker tasks after their grace period.
            # Reap the child in finally and publish a terminal cancellation,
            # without replacing an outcome already committed by another path.
            if job.done_published_at is None and job.finished_at is None:
                job.cancel_requested = True
                job.state = "canceled"
            raise
        except RetryableExecutorUnavailable as exc:
            retryable_startup_error = exc
            LOGGER.exception("worker could not confirm execution authority before native launch for job_id=%s", job.id)
        except DispatchAuthorityLost:
            dispatch_authority_lost = True
            LOGGER.warning("job %s lost its dispatch authority before native launch", job.id)
        except FileNotFoundError:
            if job.done_published_at is not None or job.finished_at is not None:
                return
            job.state = "failed"
            job.exit_code = 127
            runner_repr = " ".join(argv[:3]) if len(argv) >= 3 else argv[0]
            job.error = execution_error(
                "RUNNER_NOT_FOUND",
                f"Runner executable not found: '{argv[0]}'.",
                {"command": argv[0], "runner": runner_repr},
                retriable=False,
            )
            msg = f"runner_error: {job.error['message']}"
            job.add_log(msg)
            with suppress(Exception):
                await self.runtime.repository().append_logs(job.id, [msg], tail_limit=self.settings.log_tail_limit)
            await self.runtime.persist_fields(
                job,
                "runner_not_found",
                state=job.state,
                exit_code=job.exit_code,
                error=job.error,
                logs_tail=job.logs_tail,
            )
            await self.runtime.publish_event(
                job.id,
                "log",
                {"id": job.id, "line": msg},
            )
        except Exception as exc:
            LOGGER.exception(
                "Unhandled runner exception for job %s",
                job.id,
            )
            await _flush_log_batch()
            if job.done_published_at is not None or job.finished_at is not None:
                return
            job.state = "failed"
            job.exit_code = 1
            job.error = execution_error(
                "RUNNER_ERROR",
                "unexpected runner failure",
                {"exception_type": type(exc).__name__},
                retriable=False,
            )
            msg = "runner_error: unexpected runner failure"
            job.add_log(msg)
            with suppress(Exception):
                await self.runtime.repository().append_logs(job.id, [msg], tail_limit=self.settings.log_tail_limit)
            await self.runtime.persist_fields(
                job,
                "runner_error",
                state=job.state,
                exit_code=job.exit_code,
                error=job.error,
                logs_tail=job.logs_tail,
            )
            await self.runtime.publish_event(
                job.id,
                "log",
                {"id": job.id, "line": msg},
            )
        finally:
            # An exception while reading stdout (including an oversized line)
            # or task cancellation must not leave an unobserved child running
            # after terminal artifacts/events are published.
            stdout_drain_task = None
            if job.proc is not None and job.proc.stdout is not None:
                stdout = job.proc.stdout

                async def _discard_remaining_stdout() -> None:
                    # A full pipe can otherwise keep proc.wait() pending even
                    # after the child exits. Discard in bounded chunks.
                    while await stdout.read(64 * 1024):
                        pass

                stdout_drain_task = asyncio.create_task(_discard_remaining_stdout())
            if job.proc is not None and (
                job.proc.returncode is None or self.runtime.owned_process_group(job.proc) is not None
            ):
                if job.terminate_task is None or job.terminate_task.done():
                    job.terminate_task = asyncio.create_task(self.runtime.terminate_process(job.proc))
            with suppress(Exception):
                await _flush_log_batch()
            with suppress(Exception):
                await _flush_progress(force=True)
            if job.terminate_task is not None:
                try:
                    await job.terminate_task
                except Exception:
                    pass
            if stdout_drain_task is not None:
                try:
                    await asyncio.wait_for(stdout_drain_task, timeout=self.settings.cancel_grace_seconds)
                except asyncio.TimeoutError:
                    LOGGER.warning("job %s stdout pipe did not close during process cleanup", job.id)
                    # StreamReader exposes no public close method. Close this
                    # process-owned pipe transport after cancelling the reader;
                    # an escaped descendant must not keep worker shutdown open.
                    stdout_transport = getattr(stdout, "_transport", None)
                    if stdout_transport is not None:
                        stdout_transport.close()
                except Exception:
                    LOGGER.debug("job %s stdout cleanup failed", job.id, exc_info=True)
            if job.proc is not None:
                with suppress(AttributeError):
                    delattr(job.proc, "_tp_owned_process_group_id")
            if retryable_startup_error is not None:
                # No child started and no terminal result exists. Preserve the
                # claim for database-clock worker_lost recovery, even when an
                # API projection changed while startup I/O was unavailable.
                raise retryable_startup_error
            # Phase 2.D — terminal-state authority. If an *external* path
            # (the reclaim sweep, restart recovery, etc.) already drove
            # this Job to a terminal state AND published the terminal
            # event while ``_run_job`` was mid-flight, that earlier
            # terminal event is authoritative: do NOT republish ``done``,
            # do NOT mutate ``state``/``exit_code``/``error``, do NOT
            # touch ``finished_at``. The presence of ``done_published_at``
            # is the canonical signal that an external terminal event
            # already went out — ``_run_job`` itself sets that timestamp
            # AFTER publishing ``done`` (further down in this finally
            # block), so by definition only an external publisher could
            # have set it by now. Checking ``state in
            # TERMINAL_JOB_STATES`` here would be wrong because
            # ``_run_job`` writes ``succeeded``/``failed``/``canceled``
            # into ``job.state`` ABOVE this finally block on normal
            # completion, so guarding on state alone would skip the
            # done-event publication for every happy-path job.
            if job.done_published_at is not None or job.finished_at is not None:
                LOGGER.info(
                    "job %s reached terminal state=%s via an external publisher; skipping duplicate done event",
                    job.id,
                    job.state,
                )
                return

            if dispatch_authority_lost:
                return

            if job.state == "canceled" and job.proc is not None and job.exit_code is None:
                job.exit_code = job.proc.returncode

            fence = current_dispatch_fence()
            if fence is not None:
                try:
                    await self._publish_fenced_result(job, fence)
                except DispatchAuthorityLost:
                    LOGGER.warning("job %s publication rejected after authority loss", job.id)
                    if job.cancel_requested:
                        job.state = "canceled"
                        with suppress(DispatchAuthorityLost):
                            await self.runtime.operational_records().finish_dispatch(
                                fence, state="canceled", exit_code=job.exit_code, error=job.error
                            )
                except Exception:
                    LOGGER.exception("job %s generation publication failed", job.id)
                    with suppress(DispatchAuthorityLost):
                        await self.runtime.operational_records().finish_dispatch(
                            fence,
                            state="failed",
                            exit_code=job.exit_code,
                            error=execution_error("ARTIFACT_STORE_UNAVAILABLE", "Artifact generation could not be committed."),
                        )
                await self.runtime.flush_outbox()
                try:
                    await cleanup_terminal_generation(
                        record_store=self.runtime.operational_records(),
                        artifact_store=self.runtime.artifact_store(),
                        job_id=job.id,
                    )
                except Exception:
                    LOGGER.exception("owned attempt cleanup deferred for %s", job.id)
                committed = await self.runtime.load_record(job.id)
                if committed is not None:
                    self.runtime.cache_job(self.runtime.job_from_record(committed))
                return

            # Index artifacts and publish terminal events BEFORE setting finished_at.
            # This ensures late-connecting SSE clients can deterministically check
            # done_published_at to know if they need to wait for real events or can
            # safely synthesize a 'done' from job state. Indexing also computes
            # bounded SHA-256 fingerprints, so run it in a worker thread to keep
            # the event loop responsive while large jobs are wrapping up.
            indexed_artifacts = await asyncio.to_thread(self.runtime.index_artifacts, job)
            await self.runtime.mirror_artifacts(job)
            self.runtime.refresh_summary(job)
            for artifact in indexed_artifacts:
                await self.runtime.publish_event(
                    job.id,
                    "artifact",
                    {"id": job.id, **artifact},
                )

            await self.runtime.publish_event(
                job.id,
                "done",
                {
                    "id": job.id,
                    "state": job.state,
                    "exit_code": job.exit_code,
                    "error": job.error,
                    "artifacts": job.artifacts,
                    "run_summary": job.run_summary or None,
                },
            )
            # Mark timestamps AFTER all events are published, so SSE endpoint knows
            # it's safe to synthesize 'done' if done_published_at is set.
            job.done_published_at = self.runtime.now()
            job.finished_at = job.done_published_at
            try:
                await self.runtime.persist_state(job)
            except Exception:  # noqa: BLE001 - repository lag must not rewrite runner outcome
                LOGGER.exception("job repository final state persist failed for %s", job.id)
            self.runtime.cache_job(job)
            await self.runtime.cleanup_jobs(self.runtime.now(), force=False)
