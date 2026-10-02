"""Guarded local batch execution with explicit, caller-owned replay authority.

Ownership requires a protected local POSIX filesystem and cooperating writers.
Acquiring its OS lock proves the former engine owner released ownership; it does
not prove detached work stopped or make arbitrary external effects idempotent.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import stat
import tempfile
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

from transformation_portal.ingest.canonical_json import canonicalize_json

from .job import BatchJob, JobItem, JobStatus

SCHEMA = "tp.batch.checkpoint.v1"
_JOB_FIELDS = {"name", "output_dir", "items", "created_at", "last_updated", "metadata"}
_ITEM_FIELDS = {"id", "input_path", "output_path", "status", "error", "execution_time", "retries", "metadata"}
_ENVELOPE_FIELDS = {"schema", "job_id", "revision", "identity", "idempotency", "owner_id", "job", "attempts"}
_TERMINAL = {JobStatus.COMPLETED, JobStatus.SKIPPED}


class BatchOwnershipError(RuntimeError):
    """Checkpoint ownership is held elsewhere or cannot be established."""


class BatchRecoveryRequiredError(RuntimeError):
    """Execution would replay work without its original identity and contract."""


class BatchCheckpointError(RuntimeError):
    """Checkpoint data is invalid or could not be persisted durably."""


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


@dataclass(frozen=True)
class RecoveryIdentity:
    """Caller-asserted implementation version and immutable input/config digest."""

    processor: str
    inputs_sha256: str

    def __post_init__(self) -> None:
        if not _nonempty_string(self.processor) or not isinstance(self.inputs_sha256, str):
            raise ValueError("processor and inputs_sha256 must be non-empty strings")
        if re.fullmatch(r"[0-9a-f]{64}", self.inputs_sha256) is None:
            raise ValueError("inputs_sha256 must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class IdempotencyContract:
    """Caller attests repeats with one stable key are safe for all side effects.

    The engine cannot verify this claim. The callback and its downstream systems
    must honor it, including effects committed before an interrupted attempt.
    """

    namespace: str
    version: str

    def __post_init__(self) -> None:
        if not _nonempty_string(self.namespace) or not _nonempty_string(self.version):
            raise ValueError("idempotency namespace and version must be non-empty strings")


@dataclass(frozen=True)
class AttemptContext:
    """Attempt number and a stable operation key; the key excludes attempt number."""

    attempt: int
    idempotency_key: str


def job_payload(job: BatchJob) -> dict[str, Any]:
    """Return the existing plain checkpoint shape, excluding the lookup cache."""
    return {key: copy.deepcopy(getattr(job, key)) for key in _JOB_FIELDS if key != "items"} | {
        "items": [asdict(item) for item in job.items]
    }


def parse_job(data: Any) -> BatchJob:
    """Validate complete checkpoint data before trusting any item state."""
    if not isinstance(data, dict) or set(data) != _JOB_FIELDS:
        raise BatchCheckpointError("invalid batch checkpoint fields")
    if any(not _nonempty_string(data[key]) for key in ("name", "output_dir", "created_at", "last_updated")):
        raise BatchCheckpointError("invalid batch checkpoint identity")
    if not isinstance(data["metadata"], dict) or not isinstance(data["items"], list):
        raise BatchCheckpointError("invalid batch metadata or items")
    items = []
    seen = set()
    for item in data["items"]:
        if not isinstance(item, dict) or set(item) != _ITEM_FIELDS:
            raise BatchCheckpointError("invalid batch item fields")
        if any(not _nonempty_string(item[key]) for key in ("id", "input_path", "output_path")):
            raise BatchCheckpointError("invalid batch item identity")
        if item["id"] in seen:
            raise BatchCheckpointError("duplicate batch item ID")
        seen.add(item["id"])
        if not isinstance(item["metadata"], dict) or (item["error"] is not None and not isinstance(item["error"], str)):
            raise BatchCheckpointError("invalid batch item metadata or error")
        duration = item["execution_time"]
        if type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0:
            raise BatchCheckpointError("invalid batch item execution_time")
        if type(item["retries"]) is not int or item["retries"] < 0:
            raise BatchCheckpointError("invalid batch item retries")
        try:
            items.append(JobItem.from_dict(item))
        except (TypeError, ValueError) as exc:
            raise BatchCheckpointError("invalid batch item status") from exc
    return BatchJob(**{**data, "items": items})


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BatchCheckpointError("duplicate JSON field in checkpoint")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise BatchCheckpointError(f"non-finite JSON constant in checkpoint: {value}")


def read_checkpoint(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    except (OSError, ValueError) as exc:
        raise BatchCheckpointError(f"cannot read checkpoint: {path}") from exc


def _validate_json_value(value: Any) -> None:
    """Reject coercions that would change retained state after JSON recovery."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_value(item)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _validate_json_value(item)
        return
    raise BatchCheckpointError("checkpoint values must be JSON-native with string keys and finite numbers")


def atomic_write_checkpoint(path: Path, data: dict[str, Any]) -> None:
    """Publish one complete, durable snapshot; failures never authorize work."""
    temporary: Path | None = None
    try:
        _validate_json_value(data)
        encoded = canonicalize_json(data)
        descriptor, filename = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temporary = Path(filename)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except (OSError, TypeError, ValueError) as exc:
        raise BatchCheckpointError(f"cannot persist checkpoint: {path}") from exc
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)


@contextmanager
def checkpoint_ownership(path: Path) -> Iterator[Path]:
    """Hold a stable local POSIX lock; no timestamp or PID can steal ownership."""
    try:
        import fcntl
    except ImportError as exc:
        raise BatchOwnershipError("batch ownership requires local POSIX file locking") from exc
    descriptor = None
    try:
        path = path.parent.resolve(strict=True) / path.name
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise BatchOwnershipError("checkpoint must be a regular, non-symlink file")
        lock_path = path.with_name(path.name + ".lock")
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise BatchOwnershipError("checkpoint lock must be a regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise BatchOwnershipError(f"checkpoint is already owned: {path}") from exc
        yield path
    except OSError as exc:
        raise BatchOwnershipError(f"cannot establish checkpoint ownership: {path}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _valid_uuid(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{32}", value) is not None


def _parse_envelope(data: Any) -> tuple[dict[str, Any], BatchJob]:
    _validate_json_value(data)
    if isinstance(data, dict) and "schema" not in data:
        raise BatchRecoveryRequiredError("legacy checkpoints have no guarded execution authority")
    if not isinstance(data, dict) or set(data) != _ENVELOPE_FIELDS or data["schema"] != SCHEMA:
        raise BatchCheckpointError("unsupported or invalid guarded checkpoint schema")
    if not _valid_uuid(data["job_id"]) or not _valid_uuid(data["owner_id"]):
        raise BatchCheckpointError("invalid guarded checkpoint owner or job ID")
    if type(data["revision"]) is not int or data["revision"] < 1:
        raise BatchCheckpointError("invalid guarded checkpoint revision")
    try:
        RecoveryIdentity(**data["identity"])
        if data["idempotency"] is not None:
            IdempotencyContract(**data["idempotency"])
    except (TypeError, ValueError) as exc:
        raise BatchCheckpointError("invalid guarded execution identity or idempotency contract") from exc
    job = parse_job(data["job"])
    attempts = data["attempts"]
    if not isinstance(attempts, dict) or set(attempts) - {item.id for item in job.items}:
        raise BatchCheckpointError("invalid guarded attempt inventory")
    for item in job.items:
        attempt = attempts.get(item.id)
        if item.id not in attempts:
            if item.status not in {JobStatus.PENDING, *_TERMINAL} or item.retries != 0:
                raise BatchCheckpointError("item state requires a persisted attempt")
            continue
        if not isinstance(attempt, dict) or set(attempt) != {"attempt", "owner_id", "outcome"}:
            raise BatchCheckpointError("invalid guarded attempt fields")
        if type(attempt["attempt"]) is not int or attempt["attempt"] < 1 or not _valid_uuid(attempt["owner_id"]):
            raise BatchCheckpointError("invalid guarded attempt identity")
        outcomes = {
            "running": JobStatus.RUNNING,
            "abandoned": JobStatus.RUNNING,
            "completed": JobStatus.COMPLETED,
            "failed": JobStatus.FAILED,
        }
        if (
            not isinstance(attempt["outcome"], str)
            or attempt["outcome"] not in outcomes
            or outcomes[attempt["outcome"]] != item.status
        ):
            raise BatchCheckpointError("attempt outcome does not match item state")
        if item.retries != attempt["attempt"] - 1:
            raise BatchCheckpointError("item retries do not match persisted attempt")
        if attempt["outcome"] == "running" and attempt["owner_id"] != data["owner_id"]:
            raise BatchCheckpointError("running attempt does not belong to the recorded owner")
    return data, job


def _persist(path: Path, job: BatchJob, envelope: dict[str, Any] | None) -> None:
    payload = job_payload(job)
    updated = datetime.now().isoformat()
    payload["last_updated"] = updated
    parse_job(payload)
    if envelope is None:
        atomic_write_checkpoint(path, payload)
    else:
        snapshot = {**envelope, "revision": envelope["revision"] + 1, "job": payload}
        _parse_envelope(snapshot)
        atomic_write_checkpoint(path, snapshot)
        envelope.update(snapshot)
    job.last_updated = updated


def _execute(callback: Callable[[JobItem, AttemptContext], Any], item: JobItem, context: AttemptContext) -> JobItem:
    started = time.monotonic()
    immutable = (item.id, item.input_path, item.output_path, item.retries)
    try:
        callback(item, context)
        if immutable != (item.id, item.input_path, item.output_path, item.retries):
            raise ValueError("callback must not modify item identity or attempt authority")
        item.status, item.error = JobStatus.COMPLETED, None
    except Exception as exc:
        item.status, item.error = JobStatus.FAILED, str(exc)
    finally:
        item.execution_time = time.monotonic() - started
    return item


def _dispatch(
    callback: Callable[[JobItem, AttemptContext], Any], item: JobItem, context: AttemptContext, future: Future[JobItem]
) -> None:
    # Register our completion signal before submitting to the executor. An
    # interruption inside submit must not hide an already-started callback.
    if not future.set_running_or_notify_cancel():
        return
    try:
        result = _execute(callback, item, context)
    except BaseException as exc:
        future.set_exception(exc)
    else:
        future.set_result(result)


def _run(
    job: BatchJob,
    callback: Callable[[JobItem, AttemptContext], Any],
    path: Path,
    envelope: dict[str, Any] | None,
    *,
    max_workers: int,
    stop_on_errors: bool,
) -> BatchJob:
    """One bounded scheduler; ownership is held by its caller through all cleanup."""
    pending = iter(item for item in job.items if item.status not in _TERMINAL)
    active: dict[Future[JobItem], JobItem] = {}
    executor = ThreadPoolExecutor(max_workers=max_workers)
    failure: BaseException | None = None
    stop = False

    def record(future: Future[JobItem], item: JobItem) -> None:
        nonlocal failure, stop
        try:
            if not future.cancelled():
                result = future.result()
                if not isinstance(result.metadata, dict):
                    raise BatchCheckpointError("callback metadata must remain a JSON object")
                _validate_json_value(result.metadata)
                metadata = copy.deepcopy(result.metadata)
                item.status, item.error = result.status, result.error
                item.execution_time, item.metadata = result.execution_time, metadata
                if item.status == JobStatus.FAILED and stop_on_errors:
                    stop = True
        except BaseException as exc:
            failure = failure if failure is not None else exc
            stop = True
        if envelope is not None:
            envelope["attempts"][item.id]["outcome"] = (
                "abandoned" if item.status == JobStatus.RUNNING else item.status.value.lower()
            )

    try:
        while True:
            while not stop and len(active) < max_workers:
                item = next(pending, None)
                if item is None:
                    break
                attempt = 1 if envelope is None else envelope["attempts"].get(item.id, {}).get("attempt", 0) + 1
                item.status, item.error, item.retries = JobStatus.RUNNING, None, attempt - 1
                item.execution_time = 0.0
                if envelope is not None:
                    envelope["attempts"][item.id] = {
                        "attempt": attempt,
                        "owner_id": envelope["owner_id"],
                        "outcome": "running",
                    }
                    material = [envelope["job_id"], envelope["identity"], envelope["idempotency"], item.id]
                    key = hashlib.sha256(canonicalize_json(material)).hexdigest()
                else:
                    key = ""
                _persist(path, job, envelope)
                detached = copy.deepcopy(item)
                future = Future()
                active[future] = item
                executor.submit(_dispatch, callback, detached, AttemptContext(attempt, key), future)
            if not active:
                break
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                record(future, active.pop(future))
            _persist(path, job, envelope)
    except BaseException as exc:
        failure = failure if failure is not None else exc
    finally:
        # Never release checkpoint ownership while a callback can still mutate
        # downstream state. Interrupted thread joins alone cannot prove this;
        # wait for the separately tracked callback completion signals first.
        while True:
            try:
                for future in active:
                    future.cancel()
                unfinished = [future for future in active if not future.done()]
                if unfinished:
                    wait(unfinished, timeout=0.1)
                    continue
                executor.shutdown(wait=True, cancel_futures=True)
                break
            except BaseException as exc:
                failure = failure if failure is not None else exc
        for future, item in active.items():
            record(future, item)
        try:
            _persist(path, job, envelope)
        except BaseException as exc:
            failure = failure if failure is not None else exc
    if failure is not None:
        raise failure
    return job


def run_fresh(
    job: BatchJob, callback: Callable[[JobItem], Any], path: Path, *, max_workers: int, stop_on_errors: bool
) -> BatchJob:
    """Compatibility execution never resumes an existing or uncertain checkpoint."""
    parse_job(job_payload(job))
    if all(item.status in _TERMINAL for item in job.items):
        return job
    if job._restored_from_checkpoint:
        raise BatchRecoveryRequiredError("loaded plain checkpoints cannot authorize fresh execution")
    if any(item.status not in {JobStatus.PENDING, *_TERMINAL} or item.retries for item in job.items):
        raise BatchRecoveryRequiredError("legacy execution cannot replay RUNNING or FAILED work")
    with checkpoint_ownership(path) as owned_path:
        if owned_path.exists():
            raise BatchRecoveryRequiredError("existing checkpoints require guarded recovery; legacy replay is unsafe")
        _persist(owned_path, job, None)
        return _run(
            job, lambda item, context: callback(item), owned_path, None, max_workers=max_workers, stop_on_errors=stop_on_errors
        )


def run_guarded(
    job: BatchJob | None,
    callback: Callable[[JobItem, AttemptContext], Any],
    path: Path,
    *,
    identity: RecoveryIdentity,
    idempotency: IdempotencyContract | None,
    max_workers: int,
    stop_on_errors: bool,
) -> BatchJob:
    if not isinstance(identity, RecoveryIdentity) or (
        idempotency is not None and not isinstance(idempotency, IdempotencyContract)
    ):
        raise ValueError("guarded execution requires typed identity and idempotency contracts")
    if job is not None:
        parse_job(job_payload(job))
        if job._restored_from_checkpoint and any(item.status not in _TERMINAL for item in job.items):
            raise BatchRecoveryRequiredError("loaded plain checkpoints cannot authorize fresh guarded execution")
        if any(item.status not in {JobStatus.PENDING, *_TERMINAL} or item.retries for item in job.items):
            raise BatchRecoveryRequiredError("start_guarded accepts only fresh or completed work")
    with checkpoint_ownership(path) as owned_path:
        if job is None:
            envelope, job = _parse_envelope(read_checkpoint(owned_path))
            contract = None if idempotency is None else asdict(idempotency)
            if envelope["identity"] != asdict(identity) or envelope["idempotency"] != contract:
                raise BatchRecoveryRequiredError("recovery identity and idempotency must match the original checkpoint")
            if contract is None and any(item.status in {JobStatus.RUNNING, JobStatus.FAILED} for item in job.items):
                raise BatchRecoveryRequiredError(
                    "replaying attempted work requires its originally recorded idempotency contract"
                )
            for attempt in envelope["attempts"].values():
                if attempt["outcome"] == "running":
                    attempt["outcome"] = "abandoned"
            envelope["owner_id"] = uuid.uuid4().hex
        else:
            if owned_path.exists():
                raise BatchRecoveryRequiredError("start_guarded refuses to overwrite an existing checkpoint")
            envelope = {
                "schema": SCHEMA,
                "job_id": uuid.uuid4().hex,
                "revision": 0,
                "identity": asdict(identity),
                "idempotency": None if idempotency is None else asdict(idempotency),
                "owner_id": uuid.uuid4().hex,
                "job": job_payload(job),
                "attempts": {},
            }
        _persist(owned_path, job, envelope)
        return _run(job, callback, owned_path, envelope, max_workers=max_workers, stop_on_errors=stop_on_errors)
