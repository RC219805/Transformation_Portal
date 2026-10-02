# Guarded local batch recovery

`transformation_portal.core.batch` is an internal/shared Python helper. Its
guarded API provides explicit ownership and conditional replay for a local
batch checkpoint. It has no production callers and is not integrated with
portal jobs, Lux execution, or distributed orchestrator recovery. The
unimplemented umbrella CLI adapters remain separate work.

## API and caller authority

The public types are exported from `transformation_portal.core.batch`:

```python
from transformation_portal.core.batch import (
    AttemptContext,
    BatchJob,
    BatchProcessor,
    IdempotencyContract,
    JobItem,
    RecoveryIdentity,
)
```

The entrypoints are:

```python
processor = BatchProcessor(max_workers=4, stop_on_errors=True)

# First execution: checkpoint_path must not already exist.
result = processor.start_guarded(
    job, callback, checkpoint_path, identity=identity, idempotency=contract,
)

# A later invocation: load and validate the existing guarded checkpoint.
result = processor.resume_guarded(
    checkpoint_path, callback, identity=identity, idempotency=contract,
)
```

These are API shapes, not a ready-to-run processing recipe. The caller supplies
the `BatchJob`, callback, checkpoint path, and authority described below. Both
methods return a `BatchJob`; inspect item states because a callback exception
is recorded as `FAILED` and is not itself a successful batch outcome.

`identity` is a `RecoveryIdentity(processor, inputs_sha256)`. The processor
string must identify the callback implementation/version; the digest must be
a lowercase SHA-256 covering the caller's immutable inputs and relevant
configuration. The helper validates the digest's shape and equality, not the
source bytes or callback implementation. The caller must verify these again
before resuming and must not reuse an identity after changing their meaning.

`contract` is an optional `IdempotencyContract(namespace, version)`. Supplying
one asserts that repeating an item with its original operation key is safe
for **all** callback effects, including effects committed before a crash.
It must be recorded before the first attempt. Recovery requires exact equality
with the stored identity and contract; supplying a contract later cannot
authorize work originally started without one.

The guarded callback has the positional signature
`callback(item: JobItem, context: AttemptContext)`. `context.attempt` starts
at one and increments for each recorded attempt. `context.idempotency_key`
is stable across attempts of the same item in the same guarded job; the
attempt number is deliberately excluded from that key. The callback and its
downstream systems must actually honor the key. Merely receiving it does not
deduplicate file writes, messages, payments, subprocesses, or remote requests.

Callbacks receive detached item copies. They may update `item.metadata` with
strict JSON-native values: objects with string keys, arrays, strings, booleans,
null, and finite numeric scalars. Unsupported values are rejected without
coercion. The coordinator owns status, timing, retry counts,
and checkpoint publication; callbacks must not change item IDs, input/output
paths, or attempt authority. Callback return values are not persisted as
results. Item IDs must be unique within a job.

## Ownership, persistence, and replay

Use an existing protected directory on a **local POSIX filesystem** with
working `flock`, atomic replacement, and file/directory `fsync`. All writers
must cooperate through this API and the same checkpoint path. Network
filesystems, distributed leases, hostile checkpoint writers, and alternate
hard-link aliases are outside this ownership contract.

The engine holds a nonblocking exclusive advisory lock on
`<checkpoint_path>.lock` while validating/loading the checkpoint, recording
attempts, running callbacks, draining started workers, and publishing final
state. The lock file stays in place. Never delete it, replace it, or steal
ownership based on PID or timestamp age; a remaining lock file does not mean
the lock is currently held.

The `tp.batch.checkpoint.v1` envelope records the job ID, revision, execution
identity, original idempotency contract, owner ID, job state, and per-item
attempt ownership/outcomes. Each `RUNNING` attempt is durably recorded before
its callback is submitted. Publication uses a unique temporary file, file
`fsync`, atomic replacement, and directory `fsync`. Persistence errors stop
new submissions and propagate; started callbacks drain before ownership is
released. `stop_on_errors=True` stops additional scheduling after an observed
callback failure and also drains already-started work.

`checkpoint_interval` remains an accepted positive integer for constructor
compatibility; it does not defer these safety-critical state writes.

After ownership can be reacquired, prior running attempts are considered
abandoned by that engine owner. **This does not prove that detached processes
or remote work stopped.** Before resuming, the caller must establish that
those effects are quiescent or that downstream idempotency safely handles
overlapping attempts with the same key. Callbacks must not return while
unmanaged work can still mutate their effects without such a guarantee.

`COMPLETED` and `SKIPPED` items are not replayed. Previously attempted
`RUNNING` and `FAILED` items can be retried only with the originally recorded
matching idempotency contract. Unattempted `PENDING` items can continue without
one. There is no automatic replay loop: each explicit resume attempts eligible
items once. This is conditional recovery, not an exactly-once guarantee.
A claimed attempt may remain uncertain after an interruption during submission;
absence of a completion record does not establish whether its effects occurred.

## Legacy compatibility boundary

`BatchProcessor.process(job, callback, checkpoint_path)` retains the
one-argument positional callback `callback(item)` for first-run execution.
That callback also receives a detached item; only metadata and execution
outcomes merge back into the caller's job.
It now refuses execution against an existing checkpoint and refuses uncertain
`RUNNING` or `FAILED` items rather than silently leaving running work untouched
or retrying failures without replay authority. Fresh `PENDING` work can run
with already-terminal items retained; a fully terminal job is a no-op.

This is an intentional behavior change. Plain `BatchJob.save`/`load` data
remains inspectable, but a legacy checkpoint does not authorize guarded
recovery. Jobs returned by `BatchJob.load` retain restored provenance:
neither `process` nor `start_guarded` can execute their remaining items by
using a different checkpoint path or renaming the original file. After
explicitly reconciling prior effects, construct a new `BatchJob` containing
only work the caller has established is safe to start; new construction alone
is not evidence of that safety. Plain saves now acquire the same local ownership lock, propagate
persistence errors, and cannot overwrite a guarded checkpoint. Do not rename,
delete, or rewrite a checkpoint to force an old job through the fresh-run
entrypoint. Reconcile prior effects explicitly; migrate
callers to guarded execution **before** starting work that may need replay.

## Fail-closed outcomes

| Exception | Meaning and response |
| --- | --- |
| `BatchOwnershipError` | Another owner holds the lock, the path is unsafe, or local POSIX ownership cannot be established. Do not bypass the lock. |
| `BatchRecoveryRequiredError` | Existing/legacy or uncertain work lacks matching original replay authority, or a fresh start would overwrite a checkpoint. Reconcile effects and authority before proceeding. |
| `BatchCheckpointError` | Checkpoint data is invalid, unreadable, inconsistent, or cannot be persisted. Preserve it for investigation; do not infer completion from partial state. |
| `ValueError` | Caller configuration or typed identity/contract values are invalid. Correct the invocation before execution. |

Unknown schema versions, duplicate IDs or JSON fields, non-finite data,
inconsistent item/attempt states, mismatched identity/contract, and attempts
to overwrite an existing checkpoint fail closed. An interrupted callback may
have completed external effects even when its persisted state is uncertain.

## Validation and evidence limits

From the repository root:

```bash
PYTHONPATH=src ./.venv/bin/pytest tests/core/test_batch_guarded_recovery.py tests/core/test_batch_recovery_reporting.py tests/core/test_low_traffic_core_packages.py -q -rs
```

The guarded tests use local POSIX processes and synthetic callbacks to cover
live-owner exclusion, owner death, durable claims before execution, crash after
an idempotent effect, exact identity/contract matching, malformed checkpoints,
legacy rejection, persistence failure, and draining started work. They require
no models or external services and run in the existing core unit/regression
lane; POSIX-specific cases skip on unsupported hosts. These contracts do not
establish production adoption, real callback idempotency, or distributed
recovery acceptance.
